import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np

from feature_reuse.sharpeuler import sigma_grid
from feature_reuse.spectrum import LastBlockTaylor, SpectralForecast, Spectrum
from feature_reuse.reuse import TeaCache
from feature_reuse.vde import VDE, VelocityEstimator
from feature_reuse.klein import Flux2LatentCreator, ModelConfig, denoise


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

    def test_fixed_forecasters_refresh_on_the_same_seventeen_steps(self):
        expected = set(range(0, 50, 3))
        self.assertEqual(Spectrum(interval=3, fixed=True).refresh_steps(50), expected)
        self.assertEqual(LastBlockTaylor().refresh_steps(50), expected)


class SharpEulerTest(unittest.TestCase):
    def test_tail_matches_native_interval_without_changing_budget(self):
        grid = sigma_grid(17, [1.0] * 20, terminal_sigma=0.3)
        self.assertEqual(len(grid), 18)
        self.assertEqual((grid[0], grid[-1]), (1.0, 0.0))
        self.assertAlmostEqual(grid[-2], 0.3)
        self.assertTrue(np.all(np.diff(grid) < 0))

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


class VDETest(unittest.TestCase):
    def test_full_steps_and_endpoints(self):
        policy = VDE()
        anchors = [step for step in range(50) if policy.runs_model(step, 50)]
        self.assertEqual(anchors, [*range(7), *range(10, 47, 4), 49])

    def test_estimates_velocity_at_current_latent(self):
        estimator = VelocityEstimator()
        estimator.observe(1.0, mx.array([[[1., 0.]]]), mx.array([[[2., 3.]]]))
        estimator.observe(0.75, mx.array([[[2., 0.]]]), mx.array([[[6., 8.]]]))
        actual = estimator.predict(0.5, mx.array([[[3., 0.]]]))
        np.testing.assert_allclose(np.asarray(actual), [[[12., 15.]]])

    def test_parallel_and_zero_velocity_stay_finite(self):
        estimator = VelocityEstimator()
        latent = mx.array([[[2., 0.]]])
        estimator.observe(1.0, latent, latent)
        estimator.observe(0.75, latent, mx.zeros_like(latent))
        np.testing.assert_allclose(np.asarray(estimator.predict(0.5, latent)), [[[-2., 0.]]])

    def test_skips_both_cfg_passes_between_anchors(self):
        latent = mx.array([[[1., 2.]]])
        transformer = Mock(side_effect=lambda **kwargs: kwargs["hidden_states"] * 0.1)
        model = SimpleNamespace(transformer=transformer, model_config=ModelConfig.flux2_klein_base_4b())
        velocities = []
        for decompose in (True, False):
            transformer.reset_mock()
            velocities.clear()
            with patch.object(Flux2LatentCreator, "prepare_packed_latents", return_value=(latent, latent, None, None)):
                result = denoise(model, (latent,) * 4, 42, 6, VDE(interval=3, warmup=2, decompose=decompose),
                                 observe_velocity=velocities.append)
            self.assertEqual((result.full_steps, result.full_passes, result.guided_steps), (4, 8, 6))
            self.assertEqual(transformer.call_count, 8)
            self.assertEqual(len(velocities), 6)
            self.assertTrue(mx.all(mx.isfinite(result.latents)).item())


class AccelerationConfigTest(unittest.TestCase):
    def test_new_names_and_calibration_requirement(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-klein.py"
        parse = runpy.run_path(str(script))["parse_config"]
        self.assertIsInstance(parse("spectrum+unipc")[1], Spectrum)
        self.assertEqual(parse("teacache-0.18")[1], TeaCache(0.18, 50))
        self.assertEqual(parse("vde-4")[1], VDE())
        self.assertEqual(parse("velocity-linear-4")[1], VDE(decompose=False))
        self.assertEqual(parse("spectrum-fixed-3")[1], Spectrum(interval=3, fixed=True))
        self.assertEqual(parse("taylor-last-3")[1], LastBlockTaylor())
        with self.assertRaisesRegex(ValueError, "plain CFG"):
            parse("vde-4+ag-0.9")
        with self.assertRaisesRegex(ValueError, "--sharpeuler-profile"):
            parse("sharpeuler-17")
        profile = {"sharpness": [1.0] * 8}
        with self.assertRaisesRegex(ValueError, "plain CFG"):
            parse("sharpeuler-17+unipc", profile)
        self.assertEqual(len(parse("sharpeuler-17+cfg", profile)[4]), 18)
