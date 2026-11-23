"""Second-order forecasting with curvature damping on the real Klein pipeline.

Loads the Q4 checkpoint and generates on the GPU, so it is opt-in:
    KLEIN_INTEGRATION=1 .venv/bin/python -m unittest tests.test_klein_integration -v

With FixedInterval(2) over six steps, steps 0, 2 and 4 run the blocks and 1, 3 and 5 are
forecast. Only step 5 has three anchors, so it is the only step a curvature term can reach.
"""
import os
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get("KLEIN_INTEGRATION") == "1"
MODEL_PRESENT = (ROOT / "models" / "klein-base-4b-q4" / "transformer").is_dir()

STEPS, SEED, INTERVAL = 6, 42, 2
CURVATURE_STEP = 5


@contextmanager
def traced_latents():
    import mlx.core as mx
    # denoise() integrates with feature_reuse.samplers, not MFLUX's scheduler; Euler is its default.
    from feature_reuse.samplers import Euler

    original = Euler.__dict__["step"]
    trace = []

    def step(self, *args, **kwargs):
        latents = original(self, *args, **kwargs)
        mx.eval(latents)
        trace.append(np.array(latents.astype(mx.float32)))
        return latents

    Euler.step = step
    try:
        yield trace
    finally:
        Euler.step = original


@unittest.skipUnless(ENABLED and MODEL_PRESENT, "set KLEIN_INTEGRATION=1 with the Klein checkpoint downloaded")
class CurvatureDampingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        from feature_reuse.klein import decode, denoise, encode_prompt, load_klein
        from feature_reuse.reuse import FixedInterval

        model = load_klein()
        embeds = encode_prompt(model, (ROOT / "prompts" / "klein-test.txt").read_text().strip())

        def run(policy):
            with traced_latents() as trace:
                result = denoise(model, embeds, SEED, STEPS, policy)
            return result, trace, np.asarray(decode(model, result.latents))

        cls.full = run(None)
        cls.first_order = run(FixedInterval(INTERVAL, forecast=True))
        cls.damped = {
            weight: run(FixedInterval(INTERVAL, forecast=True, order=2, curvature_weight=weight))
            for weight in (0.0, 0.5)
        }
        # Unsuffixed second order, as taylor2-N builds it, is the undamped weight 1.0.
        cls.damped[1.0] = run(FixedInterval(INTERVAL, forecast=True, order=2))

        reference = cls.full[2].astype(np.float64)
        for name, (_, _, image) in [("order 1", cls.first_order)] + [(f"order 2, c={w}", r) for w, r in cls.damped.items()]:
            error = image.astype(np.float64) - reference
            print(f"\n  {name} vs full {STEPS}-step image: max_abs={np.abs(error).max():.0f} "
                  f"mean_abs={np.abs(error).mean():.3f} rel_l2={np.linalg.norm(error) / np.linalg.norm(reference):.4f}")

    def test_schedule_runs_expected_full_steps(self) -> None:
        for result, trace, _ in [self.first_order, *self.damped.values()]:
            self.assertEqual(result.full_steps, STEPS // INTERVAL)
            self.assertEqual(len(trace), STEPS)
            self.assertTrue(all(np.isfinite(latents).all() for latents in trace))

    def test_zero_weight_reproduces_first_order_exactly(self) -> None:
        _, first_trace, first_image = self.first_order
        _, trace, image = self.damped[0.0]
        for step, (expected, actual) in enumerate(zip(first_trace, trace)):
            self.assertTrue(np.array_equal(actual, expected), f"latents differ at step {step}")
        self.assertTrue(np.array_equal(image, first_image))

    def test_curvature_only_acts_once_three_anchors_exist(self) -> None:
        _, first_trace, _ = self.first_order
        for weight in (0.5, 1.0):
            _, trace, _ = self.damped[weight]
            for step in range(CURVATURE_STEP):
                self.assertTrue(np.array_equal(trace[step], first_trace[step]), f"c={weight} differs at step {step}")

    def test_weight_scales_the_curvature_term(self) -> None:
        final = {weight: trace[CURVATURE_STEP] for weight, (_, trace, _) in self.damped.items()}
        self.assertFalse(np.array_equal(final[0.5], final[0.0]))
        self.assertFalse(np.array_equal(final[1.0], final[0.5]))
        # The output layers are nonlinear in the forecast, but for a correction this small halving the
        # weight should still move the step-5 latents less than the undamped correction does.
        self.assertLess(np.linalg.norm(final[0.5] - final[0.0]), np.linalg.norm(final[1.0] - final[0.0]))


if __name__ == "__main__":
    unittest.main()
