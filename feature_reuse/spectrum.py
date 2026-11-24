from dataclasses import dataclass
from math import floor, isfinite

import mlx.core as mx
import numpy as np

from feature_reuse.linear_forecast import LinearForecast


@dataclass(frozen=True)
class Spectrum:
    interval: int = 2
    # Match the 17 full steps used by taylor-3 in the 50-step suite.
    growth: float = 0.33
    warmup: int = 5
    degree: int = 4
    ridge: float = 0.1
    fixed: bool = False

    def __post_init__(self) -> None:
        if self.interval < 1 or self.warmup < 1 or self.degree < 0:
            raise ValueError("Spectrum needs a positive interval and warmup and a nonnegative degree")
        if not isfinite(self.growth) or self.growth < 0 or not isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("Spectrum needs finite, nonnegative growth and positive regularization")

    def refresh_steps(self, steps: int) -> set[int]:
        if self.fixed:
            return set(range(0, steps, self.interval))
        refresh = set(range(min(self.warmup, steps)))
        for r in range(steps):
            step = self.warmup - 1 + floor((r + 1) * self.interval + self.growth * r * (r + 1) / 2)
            if step >= steps:
                break
            refresh.add(step)
        return refresh


class SpectralForecast:
    """Fit last-block image features with online ridge regression in a Chebyshev basis."""

    def __init__(self, policy: Spectrum, sigmas: list[float]) -> None:
        self.basis = np.polynomial.chebyshev.chebvander(1 - 2 * np.asarray(sigmas[:-1]), policy.degree)
        self.gram = policy.ridge * np.eye(policy.degree + 1)
        self.rhs = None

    def update(self, step: int, features: mx.array) -> None:
        basis = self.basis[step]
        self.gram += np.outer(basis, basis)
        # Accumulate Phi.T @ features instead of retaining every full step's activations.
        vector = mx.array(basis[:, None], dtype=mx.float32)
        term = vector * features.flatten().astype(mx.float32)
        self.rhs = term if self.rhs is None else self.rhs + term

    def predict(self, step: int, like: mx.array) -> mx.array:
        weights = np.linalg.solve(self.gram, self.basis[step])
        prediction = mx.array(weights, dtype=mx.float32) @ self.rhs
        return prediction.reshape(like.shape).astype(like.dtype)

    def arrays(self) -> list[mx.array]:
        return [] if self.rhs is None else [self.rhs]


@dataclass(frozen=True)
class LastBlockTaylor:
    interval: int = 3

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("Last-block Taylor needs a positive interval")

    def refresh_steps(self, steps: int) -> set[int]:
        return set(range(0, steps, self.interval))


class LastBlockLinearForecast:
    def __init__(self, sigmas: list[float]) -> None:
        self.sigmas = sigmas
        self.history = LinearForecast()

    def update(self, step: int, features: mx.array) -> None:
        self.history.update(self.sigmas[step], features)

    def predict(self, step: int, like: mx.array) -> mx.array:
        return self.history.predict(self.sigmas[step], like.dtype)

    def arrays(self) -> list[mx.array]:
        return self.history.arrays()
