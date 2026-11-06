"""Deterministic ODE samplers for flow matching, one velocity evaluation per step.

Klein predicts a velocity v = noise - data along x(sigma) = (1 - sigma) * data + sigma * noise,
and every sampler integrates dx/dsigma = v over the scheduler's sigmas. Each sampler uses only
the velocity at the scheduled timestep, so block reuse policies work unchanged.
"""
import math

import mlx.core as mx


class Euler:
    """Matches FlowMatchEulerDiscreteScheduler.step in MFLUX."""

    def __init__(self, sigmas: mx.array) -> None:
        self.sigmas = sigmas

    def step(self, velocity: mx.array, step: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[step + 1] - self.sigmas[step]).astype(latents.dtype)
        return latents + dt * velocity.astype(latents.dtype)


class DPMSolverPP2M:
    """Second-order multistep DPM-Solver++ with the midpoint update, as in diffusers.

    Works with the data prediction data = x - sigma * v and log-SNR lambda = log((1 - sigma) / sigma).
    A first-order step equals Euler exactly for flow matching, so steps without usable history
    use Euler. That covers the first step, the step after sigma = 1 (where lambda is -inf and the
    second-order correction vanishes), and the final step to sigma = 0 (where lambda is +inf).
    """

    def __init__(self, sigmas: mx.array) -> None:
        self.sigmas = sigmas
        self.values = sigmas.tolist()
        self.euler = Euler(sigmas)
        self.previous_data = None

    def step(self, velocity: mx.array, step: int, latents: mx.array) -> mx.array:
        sigma_prev = self.values[step - 1] if step > 0 else None
        sigma, sigma_next = self.values[step], self.values[step + 1]
        x = latents.astype(mx.float32)
        data = x - sigma * velocity.astype(mx.float32)
        previous_data, self.previous_data = self.previous_data, data
        if previous_data is None or not 0 < sigma_prev < 1 or sigma_next == 0:
            return self.euler.step(velocity, step, latents)
        lam_prev, lam, lam_next = (_log_snr(s) for s in (sigma_prev, sigma, sigma_next))
        h = lam_next - lam
        r = (lam - lam_prev) / h
        # -alpha_next * (exp(-h) - 1), the exponential-integrator weight on the data prediction.
        weight = (1 - sigma_next) * -math.expm1(-h)
        corrected = data + (data - previous_data) / (2 * r)
        return ((sigma_next / sigma) * x + weight * corrected).astype(latents.dtype)


class UniPC2:
    """Second-order UniPC with the B(h) = expm1(-h) ("bh2") variant, as in diffusers.

    Each step first applies the UniC corrector to the latents the previous step produced, using
    the velocity just evaluated at those latents, then takes a UniP predictor step from the
    corrected latents. The corrector reuses the existing evaluation, so there is no extra model
    pass. The second-order predictor equals DPM-Solver++ 2M; the corrector is the difference.
    The first and final steps use first-order prediction, and the final latents are uncorrected.
    """

    def __init__(self, sigmas: mx.array) -> None:
        self.sigmas = sigmas
        self.values = sigmas.tolist()
        # sigma = 1 gives lambda = -inf and sigma = 0 gives +inf; the formulas below take those limits.
        self.lambdas = [_log_snr(s) if 0 < s < 1 else (math.inf if s == 0 else -math.inf) for s in self.values]
        self.steps = len(self.values) - 1
        self.data = []
        self.last_sample = None
        self.last_order = 0

    def step(self, velocity: mx.array, step: int, latents: mx.array) -> mx.array:
        x = latents.astype(mx.float32)
        data = x - self.values[step] * velocity.astype(mx.float32)
        if self.last_sample is not None:
            x = self._correct(data, step)
        self.data = (self.data + [data])[-2:]
        order = min(2, self.steps - step, self.last_order + 1)
        self.last_sample, self.last_order = x, order
        return self._predict(x, step, order).astype(latents.dtype)

    def _first_order(self, x: mx.array, source: int, target: int):
        """Return the first-order update from source to target, its weight B(h), and h."""
        sigma, sigma_next = self.values[source], self.values[target]
        h = self.lambdas[target] - self.lambdas[source]
        weight = math.expm1(-h)
        return (sigma_next / sigma) * x - (1 - sigma_next) * weight * self.data[-1], weight, h

    def _predict(self, x: mx.array, step: int, order: int) -> mx.array:
        update, weight, h = self._first_order(x, step, step + 1)
        if order == 2:
            ratio = (self.lambdas[step - 1] - self.lambdas[step]) / h
            if math.isfinite(ratio):  # an infinite ratio (history at sigma = 1) zeroes the correction
                difference = (self.data[-2] - self.data[-1]) / ratio
                update = update - (1 - self.values[step + 1]) * weight * 0.5 * difference
        return update

    def _correct(self, data: mx.array, step: int) -> mx.array:
        # self.data still ends with the previous step's prediction, the base of the correction.
        update, weight, h = self._first_order(self.last_sample, step - 1, step)
        history = 0.0
        if self.last_order == 1:
            current = 0.5
        else:
            hh = -h
            phi = weight / hh - 1
            b0 = phi / weight
            b1 = 2 * (phi / hh - 0.5) / weight
            ratio = (self.lambdas[step - 2] - self.lambdas[step - 1]) / h
            if math.isfinite(ratio):
                # Solve [[1, 1], [ratio, 1]] @ [rho_history, rho_current] = [b0, b1].
                rho_history = (b0 - b1) / (1 - ratio)
                current = b0 - rho_history
                history = rho_history * (self.data[-2] - self.data[-1]) / ratio
            else:
                current = b0  # the limit as ratio -> -inf, where rho_history -> 0
        return update - (1 - self.values[step]) * weight * (history + current * (data - self.data[-1]))


def _log_snr(sigma: float) -> float:
    return math.log1p(-sigma) - math.log(sigma)


SAMPLERS = {"euler": Euler, "dpmpp2m": DPMSolverPP2M, "unipc": UniPC2}


def make_sampler(name: str, sigmas: mx.array):
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; choose from {sorted(SAMPLERS)}")
    return SAMPLERS[name](sigmas)
