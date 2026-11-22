import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import mlx.core as mx
import numpy as np
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

from feature_reuse.klein import denoise, encode_prompt, load_klein
from feature_reuse.reuse import BlockReuse, TeaCache

STEPS = 50
DEGREE = 4
TARGET_FULL_STEPS = 17


def relative_l1(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.abs(current - previous).mean() / np.abs(previous).mean())


def record(model, embeds, seed: int) -> tuple[list[float], list[float]]:
    """Run every step in full and return per-step input and output changes."""
    outputs, instances = [], []
    call, init = Flux2Transformer.__call__, BlockReuse.__init__

    def recording_call(self, *args, **kwargs):
        output = call(self, *args, **kwargs)
        outputs.append(output)
        return output

    def recording_init(self, policy):
        init(self, policy)
        instances.append(self)

    Flux2Transformer.__call__, BlockReuse.__init__ = recording_call, recording_init
    try:
        # A zero threshold makes every step full while still measuring the input change.
        denoise(model, embeds, seed, STEPS, TeaCache(0.0, STEPS, coefficients=(1.0, 0.0)))
    finally:
        Flux2Transformer.__call__, BlockReuse.__init__ = call, init
    # Guidance runs the conditional branch first at every step.
    cond = [np.array(output.astype(mx.float32)) for output in outputs[::2]]
    return instances[0].changes, [relative_l1(current, previous) for previous, current in zip(cond, cond[1:])]


def simulated_full_steps(changes: list[float], coefficients, threshold: float) -> int:
    policy = TeaCache(threshold, STEPS, coefficients=tuple(coefficients))
    accumulated, full = 0.0, 1
    for step, change in enumerate(changes, start=1):
        accumulated += policy.distance(change)
        if step == STEPS - 1 or accumulated >= threshold:
            full += 1
            accumulated = 0.0
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "calibration.txt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "teacache-calibration.json")
    args = parser.parse_args()
    prompts = [line for line in args.prompts.read_text().splitlines() if line.strip()]

    model = load_klein()
    runs = []
    for prompt_id, prompt in enumerate(prompts):
        embeds = encode_prompt(model, prompt)
        for seed in args.seeds:
            input_changes, output_changes = record(model, embeds, seed)
            runs.append(dict(prompt_id=prompt_id, seed=seed, input=input_changes, output=output_changes))
            print(f"p{prompt_id} s{seed}: recorded {len(input_changes)} steps", flush=True)

    x = np.concatenate([run["input"] for run in runs])
    y = np.concatenate([run["output"] for run in runs])
    coefficients = np.polyfit(x, y, DEGREE)
    fitted = np.polyval(coefficients, x)
    r_squared = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)

    thresholds = np.round(np.arange(0.05, 2.0001, 0.05), 2)
    full_steps = {
        float(t): float(np.mean([simulated_full_steps(run["input"], coefficients, t) for run in runs]))
        for t in thresholds
    }
    closest = min(full_steps, key=lambda t: abs(full_steps[t] - TARGET_FULL_STEPS))
    report = dict(
        coefficients=coefficients.tolist(),
        r_squared=float(r_squared),
        simulated_full_steps=full_steps,
        threshold_for_target=closest,
        target_full_steps=TARGET_FULL_STEPS,
        runs=runs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("coefficients", tuple(float(c) for c in coefficients))
    print(f"r_squared {r_squared:.3f}")
    print(f"threshold {closest} gives about {full_steps[closest]:.1f} full steps")


if __name__ == "__main__":
    main()
