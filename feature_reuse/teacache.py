from dataclasses import dataclass
from math import isfinite

import mlx.core as mx


# The Klein Base 4B calibration from IonDen/mlx-teacache. See its variant config
# and docs/variants/flux2-klein-base-4b.md for the polynomial and its fitted domain.
COEFFICIENTS = (-1841.022165607874, 848.4417137572868, -131.3554469956159, 8.179509586828413, 0.0)
CALIBRATED_MIN = 0.0283
CALIBRATED_MAX = 0.2198


@dataclass(frozen=True)
class TeaCache:
    threshold: float = 0.17

    def __post_init__(self) -> None:
        if not isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("TeaCache threshold must be nonnegative and finite")


class TeaCacheGate:
    def __init__(self, policy: TeaCache) -> None:
        self.threshold = policy.threshold
        self.previous_input = None
        self.accumulated = 0.0

    def reuse(self, signal: mx.array, force: bool) -> bool:
        previous = self.previous_input
        # Compare consecutive inputs, including steps that reused features.
        self.previous_input = signal
        if force or previous is None or self.threshold == 0:
            self.accumulated = 0.0
            return False

        signal_f32, previous_f32 = signal.astype(mx.float32), previous.astype(mx.float32)
        change = (mx.mean(mx.abs(signal_f32 - previous_f32)) / (mx.mean(mx.abs(previous_f32)) + 1e-8)).item()
        if isfinite(change) and change <= CALIBRATED_MAX:
            estimate = 0.0
            for coefficient in COEFFICIENTS:
                estimate = estimate * max(change, CALIBRATED_MIN) + coefficient
            self.accumulated += max(0.0, estimate)
            if self.accumulated < self.threshold:
                return True
        self.accumulated = 0.0
        return False
