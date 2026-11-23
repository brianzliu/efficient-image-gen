"""Reproducible generation suite. Score images separately with score-evaluation.py."""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import platform
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
from feature_reuse.streaming import Streamer, split_mode
from feature_reuse.svdquant import load_transformer, parse_variant, split_variant

VARIANTS = ROOT / "models/klein-base-4b-bf16/variants"


def stream_dir(variant):
    # Per-layer weight files for streaming, one set per transformer variant.
    return MODEL_PATH / "stream" if variant == "q4" else VARIANTS / f"stream-{variant}"

spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/benchmark-klein.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
DEFAULTS = ["full-50", "full-17", "fixed-3", "taylor-3", "speca-0.1-0.5", "speca-0.3-0.5", "speca-0.3-1.0", "speca-nocheck"]


def digest_sources():
    files = sorted((ROOT / "feature_reuse").glob("*.py")) + [Path(__file__), ROOT / "scripts/benchmark-klein.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts/benchmark.txt")
    parser.add_argument("--configs", nargs="+", default=DEFAULTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--transformer", default="q4", help="variant for configs without +bf16, +q4 or +svdq-N")
    parser.add_argument("--purge", action="store_true", help="empty the OS file cache before each image (needs sudo purge)")
    args = parser.parse_args()
    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()]
    if not prompts or not args.seeds or args.repeats < 1:
        parser.error("Need prompts, seeds, and positive repeats")
    parse_variant(args.transformer)
    variants = {c: split_variant(c, args.transformer) for c in args.configs}
    modes = {c: split_mode(variants[c][0]) for c in args.configs}
    configs = {c: benchmark.parse_config(modes[c][0]) for c in args.configs}
    modes = {c: mode for c, (_, mode) in modes.items()}
    variants = {c: variant for c, (_, variant) in variants.items()}
    # The default Q4 transformer is the one load_klein reads; other variants come from build-svdquant.py.
    swaps = set(variants.values()) != {"q4"}
    for steps, policy, *_ in configs.values():
        if steps < 1 or (hasattr(policy, "interval") and policy.interval < 1):
            parser.error("Steps and reuse intervals must be positive")
    identity = dict(prompts=prompts, seeds=args.seeds, repeats=args.repeats, configs=args.configs,
                    size=SIZE, guidance=GUIDANCE, transformer=args.transformer, scheduler=SCHEDULER, source_sha256=digest_sources(),
                    checkpoint="mflux-community/flux-2-klein-base-4b-mflux-q4",
                    checkpoint_revision="f69aa0b43285d0db24695ba68faabbdb1a0ad016",
                    packages={p: version(p) for p in ["mlx", "mflux", "numpy"]},
                    macos=platform.mac_ver()[0], machine=platform.machine())
    if args.purge:
        identity["purge"] = True  # Only when set, so earlier output directories still resume.
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError("Experiment identity changed; choose a new output directory")
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(identity=identity, started_at=time.time(),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            energy_status="not measured: sudo powermetrics requires authentication",
            timing_scope="resident weights; prompt encoding + denoising + decode; PNG writing excluded"), indent=2))
    path = args.out / "measurements.jsonl"
    done = {r["key"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    (args.out / "images").mkdir(exist_ok=True)
    print("Loading model", flush=True)
    model = load_klein()
    # Exercise full, forecast, and verification paths before measured runs.
    from feature_reuse.reuse import SpeCa
    warm = encode_prompt(model, prompts[0])
    decode(model, denoise(model, warm, args.seeds[0], 5, SpeCa(0.3, 0.5, 5)).latents)
    del warm
    # Streaming modes reload weights from per-layer files written once next to the checkpoint.
    streamer = Streamer(model, MODEL_PATH / "stream") if set(modes.values()) != {"resident"} else None
    if streamer:
        streamer.prepare("layer")
        with streamer.stage("text_encoder", "layer"):
            warm = encode_prompt(model, prompts[0])
        with streamer.stage("transformer", "layer"):
            latents = denoise(model, warm, args.seeds[0], 2).latents
        with streamer.stage("vae", "layer"):
            decode(model, latents)
        del warm, latents

    def swap(variant):
        if streamer:
            streamer.release("transformer")
        load_transformer(model, variant, VARIANTS)
        if streamer:
            streamer.bind("transformer", stream_dir(variant))
            streamer.prepare("resident")

    current = "q4"
    if swaps:
        for variant in sorted(set(variants.values())):
            swap(variant)
            denoise(model, encode_prompt(model, prompts[0]), args.seeds[0], 2)
            current = variant
    print("Warmup complete", flush=True)
    trial = 0
    for repeat in range(args.repeats):
        for pid, prompt in enumerate(prompts):
            for seed in args.seeds:
                names = list(configs)
                names = names[trial % len(names):] + names[:trial % len(names)]
                trial += 1
                for name in names:
                    key = f"r{repeat}-p{pid}-s{seed}-{name}"
                    if key in done:
                        continue
                    mode = modes[name]
                    if variants[name] != current:
                        swap(variants[name])
                        current = variants[name]
                    stage = streamer.stage if streamer else (lambda part, mode: contextlib.nullcontext())
                    if streamer:
                        streamer.prepare(mode)
                    if args.purge:
                        # Streamed weights then come from the SSD, not memory, at least on the first pass.
                        subprocess.run(["sudo", "-n", "/usr/sbin/purge"], check=True)
                    gc.collect(); mx.clear_cache(); mx.synchronize(); mx.reset_peak_memory()
                    started_unix = time.time()
                    start = time.perf_counter()
                    with stage("text_encoder", mode):
                        embeds = encode_prompt(model, prompt)
                    encoded = time.perf_counter()
                    with stage("transformer", mode):
                        result = denoise(model, embeds, seed, *configs[name])
                    denoised = time.perf_counter()
                    with stage("vae", mode):
                        image = decode(model, result.latents)
                    mx.synchronize()
                    end = time.perf_counter()
                    ended_unix = time.time()
                    row = dict(key=key, repeat=repeat, prompt_id=pid, prompt=prompt, seed=seed, config=name,
                        total_s=end-start, encode_s=encoded-start, denoise_s=denoised-encoded,
                        decode_s=end-denoised, full_steps=result.full_steps,
                        full_transformer_passes=result.full_passes, checked_steps=result.checked_steps,
                        verification_block_passes=result.verification_passes, guided_steps=result.guided_steps,
                        rejected_steps=result.rejected_steps, sampling_steps=configs[name][0],
                        accepted_reuse_steps=configs[name][0]-result.full_steps,
                        peak_mlx_gb=mx.get_peak_memory()/1e9, cache_mb=result.cache_bytes/1e6,
                        started_unix=started_unix, ended_unix=ended_unix, energy_j=None,
                        image=f"images/{key}.png")
                    image.save(args.out / row["image"])
                    with path.open("a") as f:
                        f.write(json.dumps(row, allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
                    print(f"{key}: {row['total_s']:.2f}s, {row['full_steps']} full, {row['rejected_steps']} rejected", flush=True)
                    del embeds, result, image


if __name__ == "__main__":
    main()
