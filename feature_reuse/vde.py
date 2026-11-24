"""Velocity Decomposition and Estimation for rectified-flow sampling."""

from collections import deque
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class VDE:
    interval: int = 4
    warmup: int = 7

    def __post_init__(self) -> None:
        if self.interval < 1 or self.warmup < 2:
            raise ValueError("VDE needs a positive anchor interval and at least two warmup steps")

    def runs_model(self, step: int, steps: int) -> bool:
        return step < self.warmup or step == steps - 1 or (step - self.warmup + 1) % self.interval == 0


class VelocityEstimator:
    def __init__(self) -> None:
        self.anchors = deque(maxlen=2)
        self.direction = None

    def observe(self, sigma: float, latent: mx.array, velocity: mx.array) -> None:
        x, v = latent.astype(mx.float32), velocity.astype(mx.float32)
        axes = (-2, -1)
        norm_sq = mx.sum(x * x, axis=axes, keepdims=True)
        parallel = mx.sum(v * x, axis=axes, keepdims=True) / mx.maximum(norm_sq, 1e-12)
        orthogonal = v - parallel * x
        norm_orthogonal = mx.sqrt(mx.sum(orthogonal**2, axis=axes, keepdims=True))
        magnitude = norm_orthogonal / mx.maximum(mx.sqrt(norm_sq), 1e-6)
        self.anchors.append((sigma, parallel, magnitude))
        self.direction = orthogonal / mx.maximum(norm_orthogonal, 1e-6)

    def predict(self, sigma: float, latent: mx.array) -> mx.array:
        (first, a1, b1), (last, a2, b2) = self.anchors
        distance = (sigma - last) / (last - first)
        parallel = a2 + distance * (a2 - a1)
        magnitude = b2 + distance * (b2 - b1)
        x = latent.astype(mx.float32)
        norm = mx.sqrt(mx.sum(x * x, axis=(-2, -1), keepdims=True))
        return (parallel * x + magnitude * norm * self.direction).astype(latent.dtype)

    def arrays(self) -> list[mx.array]:
        return [self.direction, *(value for _, a, b in self.anchors for value in (a, b))]
