import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np

from feature_reuse.klein import GUIDANCE, SCHEDULER, SIZE, decode, denoise, encode_prompt, load_klein
from feature_reuse.reuse import FixedInterval, SpeCa

STEPS = 4
SEED = 42


def main() -> int:
    prompt = (ROOT / "prompts" / "klein-test.txt").read_text().strip()
    model = load_klein()
    embeds = encode_prompt(model, prompt)
    expected = model.generate_image(
        seed=SEED,
        prompt=prompt,
        num_inference_steps=STEPS,
        height=SIZE,
        width=SIZE,
        guidance=GUIDANCE,
        scheduler=SCHEDULER,
    ).image

    def image(policy=None, sampler="euler") -> np.ndarray:
        return np.asarray(decode(model, denoise(model, embeds, SEED, STEPS, policy, sampler).latents))

    expected = np.asarray(expected)
    dpm = image(sampler="dpmpp2m")
    unipc = image(sampler="unipc")
    checks = {
        "loop == mflux": (image(), expected),
        "euler sampler == mflux": (image(sampler="euler"), expected),
        "every step full, fixed == mflux": (image(FixedInterval(1)), expected),
        "every step full, forecast == mflux": (image(FixedInterval(1, forecast=True)), expected),
        "SpeCa warmup only == mflux": (image(SpeCa(float("inf"), 1.0, STEPS, warmup_steps=STEPS)), expected),
        # Composition: the sampler sees the same velocities when every reuse step runs in full.
        "dpmpp2m, every step full, forecast == dpmpp2m": (image(FixedInterval(1, forecast=True), "dpmpp2m"), dpm),
        "dpmpp2m, SpeCa warmup only == dpmpp2m": (
            image(SpeCa(float("inf"), 1.0, STEPS, warmup_steps=STEPS), "dpmpp2m"),
            dpm,
        ),
        "unipc, every step full, forecast == unipc": (image(FixedInterval(1, forecast=True), "unipc"), unipc),
        "unipc, SpeCa warmup only == unipc": (
            image(SpeCa(float("inf"), 1.0, STEPS, warmup_steps=STEPS), "unipc"),
            unipc,
        ),
    }
    results = {name: np.array_equal(actual, reference) for name, (actual, reference) in checks.items()}
    # At 4 steps, Klein's step 2 is second order, so DPM-Solver++ must change the image.
    results["dpmpp2m != euler"] = not np.array_equal(dpm, expected)
    # Its corrector runs from step 1, so UniPC must differ from both.
    results["unipc != euler"] = not np.array_equal(unipc, expected)
    results["unipc != dpmpp2m"] = not np.array_equal(unipc, dpm)
    for name, passed in results.items():
        print("PASS" if passed else "FAIL", name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
