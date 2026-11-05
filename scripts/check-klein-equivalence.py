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

    def image(policy=None) -> np.ndarray:
        return np.asarray(decode(model, denoise(model, embeds, SEED, STEPS, policy).latents))

    checks = {
        "loop == mflux": image(),
        "every step full, fixed == mflux": image(FixedInterval(1)),
        "every step full, forecast == mflux": image(FixedInterval(1, order=4)),
        "SpeCa warmup only == mflux": image(SpeCa(float("inf"), 1.0, STEPS, warmup_steps=STEPS)),
    }
    results = {name: np.array_equal(pixels, np.asarray(expected)) for name, pixels in checks.items()}
    for name, passed in results.items():
        print("PASS" if passed else "FAIL", name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
