"""Reproducible generation suite. Score images separately with score-evaluation.py."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
import mlx.core as mx
from feature_reuse.klein import CHECKPOINT, CHECKPOINT_REVISION, SIZE, GUIDANCE, SCHEDULER, load_klein, encode_prompt, denoise, decode
from feature_reuse.profiling import BLOCK_TARGETS, OP_TARGETS, memory_snapshot, profile_summary, profiled
from feature_reuse.reuse import SpeCa, TeaCache
from feature_reuse.spectrum import LastBlockTaylor, Spectrum
from feature_reuse.vde import VDE
from evaluation.experiment import BaselineCache, baseline_fields, load_completed, sha256_file, trial_key

spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark-klein.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
DEFAULTS = ["full-50", "full-17", "fixed-3", "taylor-3", "speca-0.1-0.5", "speca-0.3-0.5", "speca-0.3-1.0", "speca-nocheck"]


def digest_sources():
    files = sorted((ROOT / "feature_reuse").glob("*.py")) + [Path(__file__), ROOT / "scripts/benchmark-klein.py",
                                                            ROOT / "evaluation/experiment.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def run_profiles(model, configs, prompt, seed, path, diagnostics):
    targets = OP_TARGETS if diagnostics["mode"] == "ops" else BLOCK_TARGETS
    linear_names = {
        id(layer): name.rsplit(".", 1)[-1]
        for name, layer in model.transformer.named_modules()
    } if diagnostics["mode"] == "ops" else None
    embeds = encode_prompt(model, prompt)
    for name in configs:
        if name in diagnostics["configs"]:
            continue
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        with profiled(targets, linear_names) as profiler:
            result = denoise(model, embeds, seed, *configs[name])
        diagnostics["configs"][name] = dict(
            profile_summary(profiler, result.seconds),
            sampling_steps=configs[name][0], full_steps=result.full_steps,
            full_transformer_passes=result.full_passes,
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        print(f"Profiled {name} ({diagnostics['mode']})", flush=True)
        del result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts/benchmark.txt")
    parser.add_argument("--configs", nargs="+", default=DEFAULTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt-ids", nargs="+", type=int,
                        help="Zero-based lines of --prompts to run (default: all); prompt_id indexes this selection")
    parser.add_argument("--experiment", default="custom",
                        help="Label recorded in the identity, e.g. curvature-quick, so different experiments cannot share a directory")
    parser.add_argument("--baseline-cache", type=Path, default=ROOT / "outputs/baseline-cache",
                        help="Shared store of full-step images reused only when every baseline field matches")
    parser.add_argument("--no-baseline-cache", action="store_true")
    parser.add_argument("--sharpeuler-profile", type=Path, help="Calibration JSON from calibrate-sharpeuler.py")
    parser.add_argument("--profile", choices=("blocks", "ops"),
                        help="After timed trials, profile the first prompt and seed once per configuration")
    args = parser.parse_args()
    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()]
    if args.prompt_ids is not None:
        if len(set(args.prompt_ids)) != len(args.prompt_ids) or not all(0 <= i < len(prompts) for i in args.prompt_ids):
            parser.error(f"--prompt-ids must be distinct indices below {len(prompts)}")
        prompts = [prompts[i] for i in args.prompt_ids]
    if not prompts or not args.seeds or args.repeats < 1:
        parser.error("Need prompts, seeds, and positive repeats")
    try:
        profile = benchmark.load_sharpeuler_profile(args.sharpeuler_profile)
        configs = {c: benchmark.parse_config(c, profile) for c in args.configs}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for steps, policy, *_ in configs.values():
        if steps < 1 or (hasattr(policy, "interval") and policy.interval < 1):
            parser.error("Steps and reuse intervals must be positive")
    hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    identity = dict(experiment=args.experiment, prompts=prompts, seeds=args.seeds, repeats=args.repeats, configs=args.configs,
                    size=SIZE, guidance=GUIDANCE, scheduler=SCHEDULER, source_sha256=digest_sources(),
                    checkpoint=CHECKPOINT, checkpoint_revision=CHECKPOINT_REVISION,
                    packages={p: version(p) for p in ["mlx", "mflux", "numpy"]},
                    macos=platform.mac_ver()[0], machine=platform.machine(), hardware=hardware)
    identity["mlx_buffer_env"] = {key: os.environ[key] for key in (
        "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_BFS_MAX_WIDTH"
    ) if key in os.environ}
    if args.profile:
        identity["profile"] = args.profile
    grids = {name: config[4] for name, config in configs.items() if len(config) == 5}
    if grids:
        identity.update(sharpeuler_profile=profile, sigma_grids=grids)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError("Experiment identity changed; choose a new output directory")
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(identity=identity, started_at=time.time(),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            hardware=hardware,
            energy_status="not measured: sudo powermetrics requires authentication",
            timing_scope="resident weights; prompt encoding + denoising + decode; PNG writing excluded"), indent=2))
    path = args.out / "measurements.jsonl"
    profile_path = args.out / "profile.json"
    diagnostics = dict(mode=args.profile, prompt_id=0, prompt=prompts[0], seed=args.seeds[0], configs={})
    if args.profile and profile_path.is_file():
        saved = json.loads(profile_path.read_text())
        if (any(saved.get(key) != diagnostics[key] for key in ("mode", "prompt_id", "prompt", "seed"))
                or not saved["configs"].keys() <= configs.keys()):
            raise ValueError("Saved profiling rows do not match the experiment")
        diagnostics = saved
    (args.out / "images").mkdir(exist_ok=True)
    done = load_completed(path, args.out, identity)
    cache = None if args.no_baseline_cache else BaselineCache(args.baseline_cache)

    def record(row):
        row["image_sha256"] = sha256_file(args.out / row["image"])
        with path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())

    # Same rotation as before: each prompt/seed trial starts one config later than the previous one.
    pending = []
    trial = 0
    for repeat in range(args.repeats):
        for pid, prompt in enumerate(prompts):
            for seed in args.seeds:
                names = list(configs)
                names = names[trial % len(names):] + names[:trial % len(names)]
                trial += 1
                pending += [(repeat, pid, prompt, seed, name) for name in names
                            if trial_key(repeat, pid, seed, name) not in done]
    print(f"{len(done)} completed trials validated, {len(pending)} pending", flush=True)

    def fields(repeat, prompt, seed, name):
        steps, policy, *_ = configs[name]
        if cache and policy is None and name not in grids:
            return baseline_fields(identity, name, steps, prompt, seed, repeat)

    generate = []
    for repeat, pid, prompt, seed, name in pending:
        hit = fields(repeat, prompt, seed, name) and cache.lookup(fields(repeat, prompt, seed, name))
        if not hit:
            generate.append((repeat, pid, prompt, seed, name))
            continue
        cache_key, cached, image = hit
        key = trial_key(repeat, pid, seed, name)
        row = dict(cached, key=key, repeat=repeat, prompt_id=pid, image=f"images/{key}.png",
                   baseline_cache=cache_key, cached_from=cached["cached_from"] or cached["key"])
        shutil.copyfile(image, args.out / row["image"])
        record(row)
        print(f"{key}: reused cached baseline {cache_key[:12]} (measured {row['total_s']:.2f}s in an earlier session)", flush=True)
    pending_profiles = args.profile and configs.keys() != diagnostics["configs"].keys()
    if not generate and not pending_profiles:
        return

    print("Loading model", flush=True)
    model = load_klein()
    # Exercise full, forecast, and verification paths before measured runs.
    warm = encode_prompt(model, prompts[0])
    decode(model, denoise(model, warm, args.seeds[0], 5, SpeCa(0.3, 0.5, 5)).latents)
    remaining_policies = {type(configs[name][1]) for *_, name in generate}
    if Spectrum in remaining_policies:
        denoise(model, warm, args.seeds[0], 7, Spectrum())
    if LastBlockTaylor in remaining_policies:
        denoise(model, warm, args.seeds[0], 4, LastBlockTaylor(interval=2))
    if TeaCache in remaining_policies:
        denoise(model, warm, args.seeds[0], 3, TeaCache(0.11, 3))
    if VDE in remaining_policies:
        denoise(model, warm, args.seeds[0], 4, VDE(warmup=2))
        if any(isinstance(configs[name][1], VDE) and not configs[name][1].decompose for *_, name in generate):
            denoise(model, warm, args.seeds[0], 4, VDE(warmup=2, decompose=False))
    del warm
    print("Warmup complete", flush=True)
    for repeat, pid, prompt, seed, name in generate:
        key = trial_key(repeat, pid, seed, name)
        gc.collect(); mx.clear_cache(); mx.synchronize(); mx.reset_peak_memory()
        active_before = mx.get_active_memory()
        started_unix = time.time()
        start = time.perf_counter()
        embeds = encode_prompt(model, prompt)
        encoded = time.perf_counter()
        active_encoded, peak_encoded = memory_snapshot()
        mx.reset_peak_memory()
        result = denoise(model, embeds, seed, *configs[name])
        denoised = time.perf_counter()
        active_denoised, peak_denoised = memory_snapshot()
        mx.reset_peak_memory()
        image = decode(model, result.latents)
        mx.synchronize()
        end = time.perf_counter()
        ended_unix = time.time()
        active_decoded, peak_decoded = memory_snapshot()
        peak_encoded = max(active_before, peak_encoded)
        peak_denoised = max(active_encoded, peak_denoised)
        peak_decoded = max(active_denoised, peak_decoded)
        row = dict(key=key, repeat=repeat, prompt_id=pid, prompt=prompt, seed=seed, config=name,
            total_s=end-start, encode_s=encoded-start, denoise_s=denoised-encoded,
            decode_s=end-denoised, full_steps=result.full_steps,
            full_transformer_passes=result.full_passes, checked_steps=result.checked_steps,
            verification_block_passes=result.verification_passes, guided_steps=result.guided_steps,
            rejected_steps=result.rejected_steps, sampling_steps=configs[name][0],
            accepted_reuse_steps=configs[name][0]-result.full_steps,
            peak_mlx_gb=max(peak_encoded, peak_denoised, peak_decoded)/1e9,
            active_before_mlx_gb=active_before/1e9,
            encode_peak_mlx_gb=peak_encoded/1e9, encode_active_mlx_gb=active_encoded/1e9,
            denoise_peak_mlx_gb=peak_denoised/1e9, denoise_active_mlx_gb=active_denoised/1e9,
            decode_peak_mlx_gb=peak_decoded/1e9, decode_active_mlx_gb=active_decoded/1e9,
            cache_mb=result.cache_bytes/1e6,
            started_unix=started_unix, ended_unix=ended_unix, energy_j=None,
            image=f"images/{key}.png", baseline_cache=None, cached_from=None)
        image.save(args.out / row["image"])
        record(row)
        if fields(repeat, prompt, seed, name):
            row["cached_from"] = f"{args.out.resolve()}:{key}"
            cache.store(fields(repeat, prompt, seed, name), row, args.out / row["image"])
        print(f"{key}: {row['total_s']:.2f}s, {row['full_steps']} full, {row['rejected_steps']} rejected", flush=True)
        del embeds, result, image
    if pending_profiles:
        run_profiles(model, configs, prompts[0], args.seeds[0], profile_path, diagnostics)


if __name__ == "__main__":
    main()
