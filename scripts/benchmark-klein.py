import argparse
import csv
import gc
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx

from feature_reuse.klein import decode, denoise, encode_prompt, load_klein
from feature_reuse.reuse import FixedInterval, SpeCa

REUSE_STEPS = 50
FIELDS = ["prompt_id", "seed", "config", "full_steps", "rejected_steps", "denoise_s", "peak_mlx_gb", "cache_mb"]


def parse_config(name: str):
    if match := re.fullmatch(r"full-(\d+)", name):
        return int(match[1]), None
    if match := re.fullmatch(r"fixed-(\d+)", name):
        return REUSE_STEPS, FixedInterval(int(match[1]))
    if match := re.fullmatch(r"taylor-(\d+)", name):
        return REUSE_STEPS, FixedInterval(int(match[1]), forecast=True)
    if match := re.fullmatch(r"speca-([\d.]+|inf)-([\d.]+)", name):
        return REUSE_STEPS, SpeCa(float(match[1]), float(match[2]), REUSE_STEPS)
    raise ValueError(f"unknown config {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "benchmark.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    args = parser.parse_args()

    configs = {name: parse_config(name) for name in args.configs}
    prompts = args.prompts.read_text().splitlines()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    with (args.out / "run-info.jsonl").open("a") as file:
        info = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "chip": chip,
            "macos": platform.mac_ver()[0],
            "mlx": version("mlx"),
            "mflux": version("mflux"),
            "seeds": args.seeds,
            "configs": args.configs,
            "prompts": prompts,
        }
        file.write(json.dumps(info) + "\n")

    results = args.out / "results.csv"
    done = set()
    if results.exists():
        with results.open(newline="") as file:
            done = {(int(row["prompt_id"]), int(row["seed"]), row["config"]) for row in csv.DictReader(file)}

    model = load_klein()
    embeds = [encode_prompt(model, prompt) for prompt in prompts]
    model.text_encoder = None
    # Read the lazily loaded weights now so no benchmarked run pays for them.
    decode(model, denoise(model, embeds[0], args.seeds[0], 1).latents)

    names = list(configs)
    trials = [(prompt_id, seed) for prompt_id in range(len(prompts)) for seed in args.seeds]
    with results.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if results.stat().st_size == 0:
            writer.writeheader()
        for trial, (prompt_id, seed) in enumerate(trials):
            # Rotate the order so heat does not always slow the same config.
            for name in names[trial % len(names) :] + names[: trial % len(names)]:
                if (prompt_id, seed, name) in done:
                    continue
                steps, policy = configs[name]
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                result = denoise(model, embeds[prompt_id], seed, steps, policy)
                peak_gb = mx.get_peak_memory() / 1000**3
                decode(model, result.latents).save(args.out / "images" / f"p{prompt_id}-s{seed}-{name}.png")
                writer.writerow(
                    {
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "config": name,
                        "full_steps": result.full_steps,
                        "rejected_steps": result.rejected_steps,
                        "denoise_s": f"{result.seconds:.2f}",
                        "peak_mlx_gb": f"{peak_gb:.2f}",
                        "cache_mb": f"{result.cache_bytes / 1000**2:.1f}",
                    }
                )
                file.flush()
                print(f"p{prompt_id} s{seed} {name}: {result.seconds:.1f}s, {result.full_steps} full", flush=True)


if __name__ == "__main__":
    main()
