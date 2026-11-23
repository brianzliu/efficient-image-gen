"""Measure Klein Q4's velocity changes on a separate set of calibration prompts."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

import mlx.core as mx
import numpy as np

from feature_reuse.klein import CHECKPOINT, CHECKPOINT_REVISION, GUIDANCE, SIZE, denoise, encode_prompt, load_klein


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path, help="One calibration prompt per line")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--reference-steps", type=int, default=50)
    args = parser.parse_args()
    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()]
    steps = args.reference_steps
    if not prompts or steps < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least one prompt, distinct seeds, and two or more reference steps")

    sigmas = np.linspace(1.0, 0.0, steps + 1).tolist()
    measured = []
    model = load_klein()
    for prompt_id, prompt in enumerate(prompts):
        embeds = encode_prompt(model, prompt)
        for seed in args.seeds:
            previous = None
            differences = []

            def observe(velocity):
                nonlocal previous
                velocity = velocity.astype(mx.float32)
                if previous is not None:
                    speed_change = velocity - previous
                    differences.append(mx.sqrt(mx.sum(speed_change**2)).item() * steps)
                previous = velocity

            denoise(model, embeds, seed, steps, sigma_grid=sigmas, observe_velocity=observe)
            measured.append(differences)
            print(f"Calibrated prompt {prompt_id + 1}/{len(prompts)}, seed {seed}", flush=True)

    profile = dict(
        method="sharpeuler-uniform-v1",
        checkpoint=CHECKPOINT,
        checkpoint_revision=CHECKPOINT_REVISION,
        size=SIZE,
        guidance=GUIDANCE,
        reference_steps=steps,
        prompts=prompts,
        seeds=args.seeds,
        packages={name: version(name) for name in ("mflux", "mlx")},
        sharpness=np.mean(measured, axis=0).tolist(),
    )
    contents = json.dumps(profile, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(contents)


if __name__ == "__main__":
    main()
