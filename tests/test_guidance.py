import importlib.util
import unittest
from pathlib import Path

import mlx.core as mx

from feature_reuse.guidance import AlwaysOn, CosineSwitchOff, Interval, cosine, parse_guidance
from feature_reuse.reuse import FixedInterval

ROOT = Path(__file__).resolve().parents[1]


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark-klein.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntervalTest(unittest.TestCase):
    def test_bounds_are_low_exclusive_high_inclusive(self) -> None:
        guide = Interval(0.8, 0.3).begin()
        self.assertEqual(
            [guide.wants_uncond(0, sigma) for sigma in (1.0, 0.8, 0.5, 0.3, 0.0)],
            [False, True, True, False, False],
        )

    def test_full_interval_covers_every_scheduled_sigma(self) -> None:
        guide = Interval(1.0, 0.0).begin()
        # Scheduled sigmas run from 1 down to just above 0; the final 0 is never evaluated.
        self.assertTrue(all(guide.wants_uncond(0, sigma) for sigma in (1.0, 0.5, 1e-3)))

    def test_rejects_invalid_bounds(self) -> None:
        for high, low in ((0.3, 0.8), (0.5, 0.5), (1.5, 0.2), (0.5, -0.1)):
            with self.assertRaises(ValueError):
                Interval(high, low)


class CosineSwitchOffTest(unittest.TestCase):
    def test_switches_off_for_good_after_reaching_threshold(self) -> None:
        guide = CosineSwitchOff(0.9).begin()
        aligned, orthogonal = mx.array([1.0, 0.0]), mx.array([0.0, 1.0])
        guide.observe(aligned, orthogonal)
        self.assertTrue(guide.wants_uncond(1, 0.9))
        guide.observe(aligned, aligned)
        self.assertFalse(guide.wants_uncond(2, 0.8))
        self.assertFalse(guide.wants_uncond(3, 0.7))

    def test_each_run_starts_on(self) -> None:
        policy = CosineSwitchOff(0.9)
        first = policy.begin()
        first.observe(mx.array([1.0]), mx.array([1.0]))
        self.assertFalse(first.wants_uncond(1, 0.5))
        self.assertTrue(policy.begin().wants_uncond(0, 1.0))

    def test_infinite_threshold_never_switches_off(self) -> None:
        guide = CosineSwitchOff(float("inf")).begin()
        guide.observe(mx.array([1.0]), mx.array([1.0]))
        self.assertTrue(guide.wants_uncond(1, 0.5))

    def test_cosine_uses_float32(self) -> None:
        a = mx.array([3.0, 4.0], dtype=mx.float16)
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)


class ParseTest(unittest.TestCase):
    def test_guidance_names(self) -> None:
        self.assertEqual(parse_guidance("cfg"), AlwaysOn())
        self.assertEqual(parse_guidance("interval-0.9-0.2"), Interval(0.9, 0.2))
        self.assertEqual(parse_guidance("ag-0.995"), CosineSwitchOff(0.995))
        for bad in ("ag", "ag-x", "interval-0.9", "interval-0.2-0.9", "cfg-1"):
            with self.assertRaises(ValueError, msg=bad):
                parse_guidance(bad)

    def test_config_suffixes_in_any_order(self) -> None:
        parse = load_benchmark().parse_config
        self.assertEqual(parse("full-50"), (50, None, "euler", AlwaysOn()))
        self.assertEqual(parse("full-17+unipc"), (17, None, "unipc", AlwaysOn()))
        expected = (50, FixedInterval(3, forecast=True), "dpmpp2m", CosineSwitchOff(0.995))
        self.assertEqual(parse("taylor-3+dpmpp2m+ag-0.995"), expected)
        self.assertEqual(parse("taylor-3+ag-0.995+dpmpp2m"), expected)
        self.assertEqual(parse("full-50+interval-0.9-0.2")[3], Interval(0.9, 0.2))
        for bad in ("full-50+unipc+euler", "full-50+cfg+ag-0.9", "full-50+heun", "full-50+"):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad)


if __name__ == "__main__":
    unittest.main()
