import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np

from feature_reuse.klein import GUIDANCE, SCHEDULER, SIZE, decode, denoise, encode_prompt, load_klein
from feature_reuse.guidance import CosineSwitchOff, Interval
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

    def run(policy=None, sampler="euler", guidance=None):
        return denoise(model, embeds, SEED, STEPS, policy, sampler, guidance)

    def image(policy=None, sampler="euler", guidance=None) -> np.ndarray:
        return np.asarray(decode(model, run(policy, sampler, guidance).latents))

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
        # Guidance policies that never drop the unconditional branch are plain CFG.
        "interval-1-0 == mflux": (image(guidance=Interval(1.0, 0.0)), expected),
        "ag-inf == mflux": (image(guidance=CosineSwitchOff(float("inf"))), expected),
        "dpmpp2m, ag-inf == dpmpp2m": (image(sampler="dpmpp2m", guidance=CosineSwitchOff(float("inf"))), dpm),
        "unipc, interval-1-0 == unipc": (image(sampler="unipc", guidance=Interval(1.0, 0.0)), unipc),
    }
    # Composition: with every step full, reuse must not change a guided run, including the
    # steps where the unconditional branch is missing or switches back on.
    switch_off, late_start = CosineSwitchOff(0.5), Interval(0.99, 0.0)
    for name, guidance in (("ag-0.5", switch_off), ("interval-0.99-0", late_start)):
        guided = image(guidance=guidance)
        checks[f"{name}, every step full, forecast == {name}"] = (
            image(FixedInterval(1, forecast=True), guidance=guidance),
            guided,
        )
        checks[f"{name} != cfg"] = (guided, None)
    results = {
        name: not np.array_equal(actual, expected) if reference is None else np.array_equal(actual, reference)
        for name, (actual, reference) in checks.items()
    }
    # Pass accounting, (full steps, full branch passes, guided steps), worked out by hand for 4 steps.
    # ag-0.5 switches off after step 0 (cosine is about 0.97).
    # interval-0.99-0 skips guidance only at sigma = 1; with taylor-2, step 1 would reuse but
    # uncond has no cache, so it becomes a full step.
    taylor = FixedInterval(2, forecast=True)
    for name, result, counts in (
        ("ag-0.5", run(guidance=switch_off), (4, 5, 1)),
        ("taylor-2+ag-0.5", run(taylor, guidance=switch_off), (2, 3, 1)),
        ("taylor-2+interval-0.99-0", run(taylor, guidance=late_start), (3, 5, 3)),
        ("taylor-2+cfg", run(taylor), (2, 4, 4)),
    ):
        actual = (result.full_steps, result.full_passes, result.guided_steps)
        results[f"{name} counts {actual} == {counts}"] = actual == counts
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
