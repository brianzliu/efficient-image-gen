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

from feature_reuse.klein import CHECKPOINT, CHECKPOINT_REVISION, GUIDANCE, SIZE, decode, denoise, encode_prompt, load_klein
from feature_reuse.reuse import FixedInterval, SpeCa, TeaCache
from feature_reuse.guidance import parse_guidance
from feature_reuse.samplers import SAMPLERS
from feature_reuse.sharpeuler import sigma_grid
from feature_reuse.spectrum import Spectrum

REUSE_STEPS = 50
FIELDS = ["prompt_id", "seed", "config", "full_steps", "rejected_steps", "denoise_s", "peak_mlx_gb", "cache_mb"]


def _order_and_damping(name: str, order: str, damping: str | None) -> tuple[int, float]:
    order = int(order or 1)
    if damping is None:
        return order, 1.0
    if order != 2:
        raise ValueError(f"curvature damping needs second-order forecasting: {name!r}")
    return order, float(damping)


def load_sharpeuler_profile(path: Path | None):
    if path is None:
        return None
    profile = json.loads(path.read_text())
    expected = dict(method="sharpeuler-uniform-v1", checkpoint=CHECKPOINT,
                    checkpoint_revision=CHECKPOINT_REVISION, size=SIZE, guidance=GUIDANCE)
    if not isinstance(profile, dict) or any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("SharpEuler profile does not match the Klein Q4 evaluation settings")
    values = profile.get("sharpness")
    if not isinstance(values, list) or profile.get("reference_steps") != len(values) + 1:
        raise ValueError("SharpEuler profile length does not match its calibration step count")
    return profile


def parse_config(name: str, profile=None):
    """Return (steps, policy, sampler, guidance), plus the grid for SharpEuler.

    Optional "+" suffixes pick a sampler (default euler) and a guidance policy (default cfg),
    in either order, e.g. taylor-3+dpmpp2m+ag-0.995 or full-50+interval-0.9-0.2.
    """
    base, *suffixes = name.split("+")
    sampler = guidance = None
    for suffix in suffixes:
        if suffix in SAMPLERS and sampler is None:
            sampler = suffix
        elif suffix not in SAMPLERS and guidance is None:
            guidance = parse_guidance(suffix)
        else:
            raise ValueError(f"repeated sampler or guidance {suffix!r} in config {name!r}")
    if match := re.fullmatch(r"sharpeuler-(\d+)", base):
        if profile is None:
            raise ValueError("Pass --sharpeuler-profile for the sharpeuler configuration")
        if sampler not in (None, "euler") or guidance not in (None, parse_guidance("cfg")):
            raise ValueError("SharpEuler calibration uses Euler and plain CFG")
        steps = int(match[1])
        return steps, None, "euler", parse_guidance("cfg"), sigma_grid(steps, profile["sharpness"])
    return (*parse_policy(base), sampler or "euler", guidance or parse_guidance("cfg"))


def parse_policy(name: str):
    if match := re.fullmatch(r"full-(\d+)", name):
        return int(match[1]), None
    if match := re.fullmatch(r"fixed-(\d+)", name):
        return REUSE_STEPS, FixedInterval(int(match[1]))
    # An optional digit after "taylor" or "speca" sets the forecast order, e.g. taylor2-3.
    # Second order also accepts a curvature damping suffix, e.g. taylor2-3-c0.25.
    if match := re.fullmatch(r"taylor([1-9]?)-(\d+)(?:-c([\d.]+))?", name):
        order, damping = _order_and_damping(name, match[1], match[3])
        return REUSE_STEPS, FixedInterval(int(match[2]), forecast=True, order=order, curvature_weight=damping)
    if match := re.fullmatch(r"speca([1-9]?)-([\d.]+|inf)-([\d.]+)(?:-c([\d.]+))?", name):
        order, damping = _order_and_damping(name, match[1], match[4])
        return REUSE_STEPS, SpeCa(float(match[2]), float(match[3]), REUSE_STEPS, order=order, curvature_weight=damping)
    if name == "speca-nocheck":
        return REUSE_STEPS, SpeCa(0.3, 0.5, REUSE_STEPS, min_predicted=100)
    if name == "spectrum":
        return REUSE_STEPS, Spectrum()
    if match := re.fullmatch(r"spectrum-([\d.]+)", name):
        return REUSE_STEPS, Spectrum(growth=float(match[1]))
    if name == "teacache":
        return REUSE_STEPS, TeaCache(0.11, REUSE_STEPS)
    if match := re.fullmatch(r"teacache(-taylor)?-([\d.]+)", name):
        return REUSE_STEPS, TeaCache(float(match[2]), REUSE_STEPS, forecast=bool(match[1]))
    raise ValueError(f"unknown config {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "benchmark.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "benchmark")
    parser.add_argument("--sharpeuler-profile", type=Path)
    args = parser.parse_args()

    try:
        profile = load_sharpeuler_profile(args.sharpeuler_profile)
        configs = {name: parse_config(name, profile) for name in args.configs}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    prompts = args.prompts.read_text().splitlines()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    run_info = args.out / "run-info.jsonl"
    grids = {name: config[4] for name, config in configs.items() if len(config) == 5}
    if grids and run_info.exists():
        for line in run_info.read_text().splitlines():
            previous = json.loads(line)
            if grids.keys() & set(previous["configs"]) and previous.get("sharpeuler_profile") != profile:
                parser.error("SharpEuler calibration changed; choose a new output directory")
    with run_info.open("a") as file:
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
        if grids:
            info["sharpeuler_profile"] = profile
            info["sigma_grids"] = grids
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
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                result = denoise(model, embeds[prompt_id], seed, *configs[name])
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
