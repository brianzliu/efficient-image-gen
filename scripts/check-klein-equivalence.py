import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np
from mflux.utils.apple_silicon import AppleSiliconUtil

from feature_reuse.klein import GUIDANCE, SCHEDULER, SIZE, decode, denoise, encode_prompt, load_klein
from feature_reuse.reuse import FixedInterval, SpeCa

STEPS = 4
SEED = 42


@contextmanager
def uncompiled_mflux():
    # MFLUX wraps its CFG predict in mx.compile except on M1/M2. Our loop runs eagerly, and kernel
    # fusion changes FP16 rounding, so exact equality is only meaningful against MFLUX's eager path.
    # This selects MFLUX's own M1/M2 branch at runtime; MFLUX itself is not modified.
    original = AppleSiliconUtil.__dict__["is_m1_or_m2"]
    AppleSiliconUtil.is_m1_or_m2 = staticmethod(lambda: True)
    try:
        yield
    finally:
        AppleSiliconUtil.is_m1_or_m2 = original


def differences(actual: np.ndarray, expected: np.ndarray) -> str:
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    error = np.abs(actual - expected)
    relative = np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-30)
    return (f"max_abs={error.max():.4g} mean_abs={error.mean():.4g} "
            f"rel_l2={relative:.4g} pixels_differing={(error > 0).mean():.2%}")


def main() -> int:
    prompt = (ROOT / "prompts" / "klein-test.txt").read_text().strip()
    model = load_klein()
    embeds = encode_prompt(model, prompt)

    def mflux() -> np.ndarray:
        return np.asarray(model.generate_image(
            seed=SEED,
            prompt=prompt,
            num_inference_steps=STEPS,
            height=SIZE,
            width=SIZE,
            guidance=GUIDANCE,
            scheduler=SCHEDULER,
        ).image)

    def image(policy=None) -> np.ndarray:
        return np.asarray(decode(model, denoise(model, embeds, SEED, STEPS, policy).latents))

    with uncompiled_mflux():
        eager = mflux()
    compiles = not AppleSiliconUtil.is_m1_or_m2()
    compiled = mflux()

    checks = {
        "loop": image(),
        "every step full, fixed": image(FixedInterval(1)),
        "every step full, forecast": image(FixedInterval(1, forecast=True)),
        "SpeCa warmup only": image(SpeCa(float("inf"), 1.0, STEPS, warmup_steps=STEPS)),
    }

    print("[exact] ours (uncompiled) vs MFLUX uncompiled path; requires bit-identical pixels")
    results = {name: np.array_equal(pixels, eager) for name, pixels in checks.items()}
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name} == mflux-uncompiled  ({differences(checks[name], eager)})")

    mode = "mx.compile" if compiles else "eager on M1/M2, so identical to the exact reference"
    print(f"\n[report] ours (uncompiled) vs MFLUX default path ({mode}); "
          "floating-point differences are expected and do not fail the check")
    print(f"  mflux-default vs mflux-uncompiled  ({differences(compiled, eager)})")
    for name, pixels in checks.items():
        print(f"  INFO {name} vs mflux-default  ({differences(pixels, compiled)})")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
