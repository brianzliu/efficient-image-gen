import math
import unittest

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig

from feature_reuse.klein import GUIDANCE, SCHEDULER, SIZE
from feature_reuse.samplers import DPMSolverPP2M, Euler, UniPC2, make_sampler

DATA_STD = 0.5


def klein_scheduler(steps: int):
    return Config(
        model_config=ModelConfig.flux2_klein_base_4b(),
        num_inference_steps=steps,
        height=SIZE,
        width=SIZE,
        guidance=GUIDANCE,
        scheduler=SCHEDULER,
    ).scheduler


def gaussian_velocity(x: mx.array, sigma: float) -> mx.array:
    """Exact flow-matching velocity when the data are N(0, DATA_STD**2)."""
    variance = (1 - sigma) ** 2 * DATA_STD**2 + sigma**2
    data = (1 - sigma) * DATA_STD**2 * x / variance
    return (x - data) / sigma


def gaussian_ratio(start: float, end: float) -> float:
    """x(end) / x(start) along the exact ODE path for gaussian_velocity."""
    scale = lambda sigma: math.sqrt((1 - sigma) ** 2 * DATA_STD**2 + sigma**2)
    return scale(end) / scale(start)


def integrate(sampler_class, sigmas: list[float]) -> float:
    sampler = sampler_class(mx.array(sigmas))
    x = mx.array([1.0])
    for step in range(len(sigmas) - 1):
        x = sampler.step(gaussian_velocity(x, sigmas[step]), step, x)
    return x.item()


class EulerTest(unittest.TestCase):
    def test_matches_mflux_scheduler_bitwise(self) -> None:
        scheduler = klein_scheduler(17)
        sampler = Euler(scheduler.sigmas)
        latents = mx.random.normal((1, 1024, 128), key=mx.random.key(0))
        for step in range(17):
            velocity = mx.random.normal(latents.shape, key=mx.random.key(step + 1)).astype(mx.float16)
            expected = scheduler.step(noise=velocity, timestep=step, latents=latents, sigmas=scheduler.sigmas)
            actual = sampler.step(velocity, step, latents)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(mx.array_equal(actual, expected).item(), f"step {step}")
            latents = expected


class DPMSolverPP2MTest(unittest.TestCase):
    def test_steps_without_usable_history_are_euler(self) -> None:
        # Klein's schedule starts at sigma = 1, so steps 0 and 1 are first order, as is the step to 0.
        sigmas = klein_scheduler(17).sigmas
        euler, dpm = Euler(sigmas), DPMSolverPP2M(sigmas)
        latents = mx.random.normal((1, 16, 8), key=mx.random.key(0))
        for step in range(17):
            velocity = mx.random.normal(latents.shape, key=mx.random.key(step + 1))
            expected = euler.step(velocity, step, latents)
            actual = dpm.step(velocity, step, latents)
            if step in (0, 1, 16):
                self.assertTrue(mx.array_equal(actual, expected).item(), f"step {step}")
            else:
                self.assertFalse(mx.array_equal(actual, expected).item(), f"step {step}")
            latents = actual

    def test_interior_steps_are_second_order(self) -> None:
        # Uniform-in-sigma steps near sigma = 0 stay large in log-SNR, which limits any log-SNR
        # solver to first order there, so the order check uses an interval away from both ends.
        errors = {}
        for steps in (20, 40):
            sigmas = [0.95 - 0.9 * i / steps for i in range(steps + 1)]
            target = gaussian_ratio(sigmas[0], sigmas[-1])
            errors[steps] = {cls: abs(integrate(cls, sigmas) - target) for cls in (Euler, DPMSolverPP2M)}
        self.assertGreater(errors[20][Euler] / errors[40][Euler], 1.8)
        self.assertLess(errors[20][Euler] / errors[40][Euler], 2.2)
        self.assertGreater(errors[20][DPMSolverPP2M] / errors[40][DPMSolverPP2M], 3.6)

    def test_more_accurate_than_euler_on_klein_schedule(self) -> None:
        # At 17 steps the last nonzero sigma is about 0.3. The final jump to 0 is first order for
        # both samplers and dominates their end error, so check before it and at the end separately.
        sigmas = klein_scheduler(17).sigmas.tolist()
        for end in (len(sigmas) - 2, len(sigmas) - 1):
            target = gaussian_ratio(sigmas[0], sigmas[end])
            euler_error = abs(integrate(Euler, sigmas[: end + 1]) - target)
            dpm_error = abs(integrate(DPMSolverPP2M, sigmas[: end + 1]) - target)
            self.assertLess(dpm_error, euler_error / 3 if sigmas[end] > 0 else euler_error)

    def test_preserves_latent_dtype(self) -> None:
        for sampler_class in (DPMSolverPP2M, UniPC2):
            sampler = sampler_class(mx.array([0.9, 0.6, 0.3, 0.0]))
            latents = mx.ones((1, 4), dtype=mx.float16)
            for step in range(3):
                latents = sampler.step(mx.ones((1, 4), dtype=mx.float16), step, latents)
                self.assertEqual(latents.dtype, mx.float16, sampler_class.__name__)


class UniPC2Test(unittest.TestCase):
    # Matches diffusers' UniPCMultistepScheduler (bh2, order 2, flow prediction, lower_order_final)
    # to float32 precision; that comparison needs torch and runs outside this suite.

    def test_first_step_is_euler(self) -> None:
        sigmas = klein_scheduler(17).sigmas
        latents = mx.random.normal((1, 16, 8), key=mx.random.key(0))
        velocity = mx.random.normal(latents.shape, key=mx.random.key(1))
        expected = Euler(sigmas).step(velocity, 0, latents)
        self.assertTrue(mx.allclose(UniPC2(sigmas).step(velocity, 0, latents), expected, atol=1e-6).item())

    def test_sigma_one_start_is_the_limit_of_nearby_starts(self) -> None:
        # Klein starts at sigma = 1 exactly, where log-SNR is -inf and the code takes limits.
        sigmas = klein_scheduler(17).sigmas.tolist()
        self.assertEqual(sigmas[0], 1.0)
        exact_start = integrate(UniPC2, sigmas)
        nearby_start = integrate(UniPC2, [0.9999] + sigmas[1:])
        self.assertTrue(math.isfinite(exact_start))
        self.assertAlmostEqual(exact_start, nearby_start, delta=1e-4)

    def test_corrector_beats_dpm_solver_on_interior_steps(self) -> None:
        errors = {}
        for steps in (10, 20, 40):
            sigmas = [0.95 - 0.9 * i / steps for i in range(steps + 1)]
            target = gaussian_ratio(sigmas[0], sigmas[-1])
            errors[steps] = abs(integrate(UniPC2, sigmas) - target)
            self.assertLess(errors[steps], abs(integrate(DPMSolverPP2M, sigmas) - target) / 3, f"{steps} steps")
        self.assertGreater(errors[10] / errors[20], 2.2)
        self.assertGreater(errors[20] / errors[40], 2.2)

    def test_more_accurate_than_euler_on_klein_schedule(self) -> None:
        sigmas = klein_scheduler(17).sigmas.tolist()
        target = gaussian_ratio(sigmas[0], 0.0)
        self.assertLess(abs(integrate(UniPC2, sigmas) - target), abs(integrate(Euler, sigmas) - target))


class MakeSamplerTest(unittest.TestCase):
    def test_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            make_sampler("heun", mx.array([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
