"""Classifier-free guidance policies: which steps run the unconditional branch.

Policies are frozen specs that can be shared across runs; begin() returns the per-run state.
On a step without the unconditional branch, the velocity is the conditional prediction alone.
"""
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class AlwaysOn:
    """Plain CFG: both branches on every step."""

    def begin(self):
        return self

    def wants_uncond(self, step: int, sigma: float) -> bool:
        return True

    def observe(self, cond: mx.array, uncond: mx.array) -> None:
        pass


@dataclass(frozen=True)
class Interval:
    """CFG only while low < sigma <= high (Kynkäänniemi et al., 2024); interval-1-0 is plain CFG."""

    high: float
    low: float

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high <= 1:
            raise ValueError(f"need 0 <= low < high <= 1, got high={self.high}, low={self.low}")

    def begin(self):
        return self

    def wants_uncond(self, step: int, sigma: float) -> bool:
        return self.low < sigma <= self.high

    def observe(self, cond: mx.array, uncond: mx.array) -> None:
        pass


@dataclass(frozen=True)
class CosineSwitchOff:
    """Adaptive guidance (Castillo et al., 2023): drop the unconditional branch for good once
    cos(cond, uncond) reaches the threshold. The step that reaches it still uses full CFG."""

    threshold: float

    def begin(self):
        return _CosineSwitchOffRun(self.threshold)


class _CosineSwitchOffRun:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.on = True
        self.similarities = []

    def wants_uncond(self, step: int, sigma: float) -> bool:
        return self.on

    def observe(self, cond: mx.array, uncond: mx.array) -> None:
        similarity = cosine(cond, uncond)
        self.similarities.append(similarity)
        if similarity >= self.threshold:
            self.on = False


def cosine(a: mx.array, b: mx.array) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)))).item()


def parse_guidance(name: str):
    """cfg, interval-HIGH-LOW, or ag-THRESHOLD (ag-inf never switches off)."""
    if name == "cfg":
        return AlwaysOn()
    parts = name.split("-")
    try:
        if parts[0] == "interval" and len(parts) == 3:
            return Interval(float(parts[1]), float(parts[2]))
        if parts[0] == "ag" and len(parts) == 2:
            return CosineSwitchOff(float(parts[1]))
    except ValueError as error:
        raise ValueError(f"invalid guidance {name!r}: {error}") from None
    raise ValueError(f"unknown guidance {name!r}; use cfg, interval-HIGH-LOW, or ag-THRESHOLD")
