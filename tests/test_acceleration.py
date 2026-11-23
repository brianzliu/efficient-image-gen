import runpy
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from feature_reuse.sharpeuler import sigma_grid
from feature_reuse.spectrum import SpectralForecast, Spectrum
from feature_reuse.teacache import TeaCache, TeaCacheGate


class SpectrumTest(unittest.TestCase):
    def test_default_budget_matches_taylor_3(self):
        steps = Spectrum().refresh_steps(50)
        self.assertEqual(len(steps), 17)
        self.assertTrue(set(range(5)) <= steps)

    def test_incremental_fit_matches_batch_ridge(self):
        policy = Spectrum(degree=2)
        sigmas = [1.0, 0.9, 0.6, 0.3, 0.1, 0.0]
        forecast = SpectralForecast(policy, sigmas)
        observed = np.array([[1.0, 6.0], [2.0, 4.0], [3.0, 1.0]], dtype=np.float32)
        for step, row in enumerate(observed):
            forecast.update(step, mx.array(row[None, :]))
        basis = np.polynomial.chebyshev.chebvander(1 - 2 * np.array(sigmas[:-1]), policy.degree)
        anchors = basis[:3]
        expected = basis[4] @ np.linalg.solve(anchors.T @ anchors + policy.ridge * np.eye(3), anchors.T @ observed)
        actual = np.asarray(forecast.predict(4, mx.zeros((1, 2))))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)


class SharpEulerTest(unittest.TestCase):
    def test_profile_concentrates_steps_in_sharper_region(self):
        grid = sigma_grid(20, [100.0] * 8 + [0.01] * 8)
        self.assertEqual(len(grid), 21)
        self.assertEqual((grid[0], grid[-1]), (1.0, 0.0))
        self.assertTrue(np.all(np.diff(grid) < 0))
        self.assertGreater(sum(value >= 0.5 for value in grid), 12)

    def test_rejects_invalid_profile(self):
        for profile in ([], [-1.0], [float("nan")], [[1.0]]):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                sigma_grid(17, profile)


class TeaCacheTest(unittest.TestCase):
    def test_accumulates_small_changes_until_refresh(self):
        gate = TeaCacheGate(TeaCache(0.25))
        signal = mx.ones((1, 4))
        self.assertFalse(gate.reuse(signal, force=True))
        self.assertTrue(gate.reuse(signal * 1.01, force=False))
        self.assertFalse(gate.reuse(signal * 1.02, force=False))
        self.assertTrue(gate.reuse(signal * 1.03, force=False))
        self.assertFalse(gate.reuse(signal * 2, force=False))
        self.assertTrue(gate.reuse(signal * 2.01, force=False))
        self.assertFalse(gate.reuse(signal * 2.02, force=True))


class AccelerationConfigTest(unittest.TestCase):
    def test_new_names_and_calibration_requirement(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-klein.py"
        parse = runpy.run_path(str(script))["parse_config"]
        self.assertIsInstance(parse("spectrum+unipc")[1], Spectrum)
        self.assertEqual(parse("teacache-0.18")[1], TeaCache(0.18))
        with self.assertRaisesRegex(ValueError, "--sharpeuler-profile"):
            parse("sharpeuler-17")
        profile = {"sharpness": [1.0] * 8}
        with self.assertRaisesRegex(ValueError, "plain CFG"):
            parse("sharpeuler-17+unipc", profile)
        self.assertEqual(len(parse("sharpeuler-17+cfg", profile)[4]), 18)
