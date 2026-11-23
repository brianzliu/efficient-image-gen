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
from feature_reuse.klein import SIZE, GUIDANCE, SCHEDULER, MODEL_PATH, load_klein, encode_prompt, denoise, decode
from evaluation.experiment import BaselineCache, baseline_fields, load_completed, sha256_file, trial_key

spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark-klein.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
DEFAULTS = ["full-50", "full-17", "fixed-3", "taylor-3", "speca-0.1-0.5", "speca-0.3-0.5", "speca-0.3-1.0", "speca-nocheck"]


def digest_sources():
    files = sorted((ROOT / "feature_reuse").glob("*.py")) + [Path(__file__), ROOT / "scripts/benchmark-klein.py",
                                                            ROOT / "evaluation/experiment.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


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
    args = parser.parse_args()
    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()]
    if args.prompt_ids is not None:
        if len(set(args.prompt_ids)) != len(args.prompt_ids) or not all(0 <= i < len(prompts) for i in args.prompt_ids):
            parser.error(f"--prompt-ids must be distinct indices below {len(prompts)}")
        prompts = [prompts[i] for i in args.prompt_ids]
    if not prompts or not args.seeds or args.repeats < 1:
        parser.error("Need prompts, seeds, and positive repeats")
    from feature_reuse.reuse import SpeCa
    configs = {c: (50, SpeCa(0.3, 0.5, 50, min_predicted=100)) if c == "speca-nocheck"
               else benchmark.parse_config(c) for c in args.configs}
    for steps, policy in configs.values():
        if steps < 1 or (hasattr(policy, "interval") and policy.interval < 1):
            parser.error("Steps and reuse intervals must be positive")
    hardware = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    identity = dict(experiment=args.experiment, prompts=prompts, seeds=args.seeds, repeats=args.repeats, configs=args.configs,
                    size=SIZE, guidance=GUIDANCE, scheduler=SCHEDULER, source_sha256=digest_sources(),
                    checkpoint="mflux-community/flux-2-klein-base-4b-mflux-q4",
                    checkpoint_revision="f69aa0b43285d0db24695ba68faabbdb1a0ad016",
                    packages={p: version(p) for p in ["mlx", "mflux", "numpy"]},
                    macos=platform.mac_ver()[0], machine=platform.machine(), hardware=hardware)
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
        steps, policy = configs[name]
        return baseline_fields(identity, name, steps, prompt, seed, repeat) if cache and policy is None else None

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
    if not generate:
        return

    print("Loading model", flush=True)
    model = load_klein()
    # Exercise full, forecast, and verification paths before measured runs.
    from feature_reuse.reuse import SpeCa
    warm = encode_prompt(model, prompts[0])
    decode(model, denoise(model, warm, args.seeds[0], 5, SpeCa(0.3, 0.5, 5)).latents)
    del warm
    print("Warmup complete", flush=True)
    for repeat, pid, prompt, seed, name in generate:
        key = trial_key(repeat, pid, seed, name)
        gc.collect(); mx.clear_cache(); mx.synchronize(); mx.reset_peak_memory()
        started_unix = time.time()
        start = time.perf_counter()
        embeds = encode_prompt(model, prompt)
        encoded = time.perf_counter()
        result = denoise(model, embeds, seed, *configs[name])
        denoised = time.perf_counter()
        image = decode(model, result.latents)
        mx.synchronize()
        end = time.perf_counter()
        ended_unix = time.time()
        row = dict(key=key, repeat=repeat, prompt_id=pid, prompt=prompt, seed=seed, config=name,
            total_s=end-start, encode_s=encoded-start, denoise_s=denoised-encoded,
            decode_s=end-denoised, full_steps=result.full_steps,
            full_transformer_passes=2*result.full_steps, checked_steps=result.checked_steps,
            verification_block_passes=2*result.checked_steps,
            rejected_steps=result.rejected_steps, sampling_steps=configs[name][0],
            accepted_reuse_steps=configs[name][0]-result.full_steps,
            peak_mlx_gb=mx.get_peak_memory()/1e9, cache_mb=result.cache_bytes/1e6,
            started_unix=started_unix, ended_unix=ended_unix, energy_j=None,
            image=f"images/{key}.png", baseline_cache=None, cached_from=None)
        image.save(args.out / row["image"])
        record(row)
        if fields(repeat, prompt, seed, name):
            row["cached_from"] = f"{args.out.resolve()}:{key}"
            cache.store(fields(repeat, prompt, seed, name), row, args.out / row["image"])
        print(f"{key}: {row['total_s']:.2f}s, {row['full_steps']} full, {row['rejected_steps']} rejected", flush=True)
        del embeds, result, image


if __name__ == "__main__":
    main()
