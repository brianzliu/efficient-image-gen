import math
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx

from feature_reuse.spectrum import LastBlockLinearForecast, LastBlockTaylor, SpectralForecast, Spectrum


def _check_curvature_weight(weight: float) -> None:
    if not math.isfinite(weight):
        raise ValueError(f"curvature_weight must be finite, got {weight}")


@dataclass(frozen=True)
class FixedInterval:
    interval: int
    forecast: bool = False
    order: int = 1
    # Scales the second-order term; 1.0 is undamped and 0.0 reduces order 2 to order 1.
    curvature_weight: float = 1.0

    def __post_init__(self) -> None:
        _check_curvature_weight(self.curvature_weight)

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step % self.interval == 0

    def checks(self, predicted: int) -> bool:
        return False


@dataclass(frozen=True)
class SpeCa:
    base_threshold: float
    decay: float
    total_steps: int
    warmup_steps: int = 3
    # The reference code uses 2, which leaves little room to adapt near 17 full steps.
    min_predicted: int = 1
    max_predicted: int = 8
    order: int = 1
    curvature_weight: float = 1.0
    forecast = True

    def __post_init__(self) -> None:
        _check_curvature_weight(self.curvature_weight)

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step < self.warmup_steps or predicted >= self.max_predicted

    def checks(self, predicted: int) -> bool:
        return predicted >= self.min_predicted

    def threshold(self, step: int) -> float:
        return self.base_threshold * self.decay ** (step / self.total_steps)


# Fitted for Klein by scripts/calibrate-teacache.py. Highest power first.
KLEIN_TEACACHE_COEFFICIENTS = (
    21309.223068516723,
    -4293.211814252035,
    287.8941826749216,
    -7.476169497672445,
    0.10480946102897497,
)


@dataclass(frozen=True)
class TeaCache:
    """Skip blocks until the rescaled input change since the last full step reaches the threshold."""

    threshold: float
    total_steps: int
    forecast: bool = False
    coefficients: tuple[float, ...] = KLEIN_TEACACHE_COEFFICIENTS
    order = 1

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step == 0 or step == self.total_steps - 1

    def checks(self, predicted: int) -> bool:
        return False

    def distance(self, change: float) -> float:
        total = 0.0
        for coefficient in self.coefficients:
            total = total * change + coefficient
        return total


class BlockReuse:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.branch = None
        self.step = 0
        self.skip = False
        self.checking = False
        self.predicted = 0
        self.full_steps = 0
        self.rejected_steps = 0
        self.errors = []
        # (branch, segment) -> divided differences and the full steps they were computed at.
        # Timing is kept per branch because guidance policies can leave the unconditional branch
        # out of a full step.
        self.factors = {}
        self.knots = {}
        self.stream_in = None
        self.text_tokens = 0
        self.checked_steps = 0
        self.observed = False
        self.previous_input = None
        self.accumulated = 0.0
        self.changes = []

    def begin_step(self, step: int, branches=("cond", "uncond")) -> None:
        self.step = step
        self.observed = False
        if (
            self.policy is None
            or self.policy.runs_blocks(step, self.predicted)
            # A branch with nothing cached cannot be predicted, e.g. guidance switching on mid-run.
            or any((branch, "last") not in self.factors for branch in branches)
        ):
            self._start_full_step()
        else:
            self.skip = True
            self.checking = self.policy.checks(self.predicted)
            self.errors = []

    def _start_full_step(self) -> None:
        self.skip = False
        self.checking = False
        self.predicted = 0
        self.full_steps += 1

    def confirm_step(self) -> bool:
        if not self.skip:
            return True
        if self.checking:
            self.checked_steps += 1
            mx.eval(*self.errors)
            if max(error.item() for error in self.errors) > self.policy.threshold(self.step):
                self.rejected_steps += 1
                self._start_full_step()
                return False
        self.predicted += 1
        return True

    def observe(self, modulated: mx.array) -> None:
        # TeaCache decides once per step, from the first branch's input, before any block runs.
        if self.observed:
            return
        self.observed = True
        if self.previous_input is not None:
            previous = self.previous_input.astype(mx.float32)
            change = (mx.mean(mx.abs(modulated.astype(mx.float32) - previous)) / mx.mean(mx.abs(previous))).item()
            self.changes.append(change)
            self.accumulated += self.policy.distance(change)
        self.previous_input = modulated
        if self.skip and self.accumulated >= self.policy.threshold:
            self._start_full_step()
        if not self.skip:
            self.accumulated = 0.0

    def store(self, segment: str, residual: mx.array) -> None:
        # Newton divided differences over the most recent full steps, newest first.
        # They stay exact for polynomials up to the policy order even when spacing is uneven.
        key = (self.branch, segment)
        factors, knots = [residual], [self.step]
        if self.policy.forecast and key in self.factors:
            old_factors, old_knots = self.factors[key], self.knots[key]
            for i in range(min(self.policy.order, len(old_factors))):
                factors.append((factors[i] - old_factors[i]) / (self.step - old_knots[i]))
                knots.append(old_knots[i])
        self.factors[key], self.knots[key] = factors, knots

    def predict(self, segment: str) -> mx.array:
        key = (self.branch, segment)
        prediction, weight = self.factors[key][0], 1
        for order, (factor, knot) in enumerate(zip(self.factors[key][1:], self.knots[key]), start=1):
            weight *= self.step - knot
            if order == 2:
                damping = getattr(self.policy, "curvature_weight", 1.0)
                if damping == 0:
                    # Skip rather than multiply so a non-finite curvature cannot leak in as 0 * inf.
                    continue
                prediction = prediction + (damping * weight) * factor
            else:
                prediction = prediction + weight * factor
        return prediction

    def record_error(self, predicted: mx.array, actual: mx.array) -> None:
        # Text tokens never reach the output, and their large values would dominate.
        difference = (predicted - actual)[:, self.text_tokens :].astype(mx.float32)
        actual = actual[:, self.text_tokens :].astype(mx.float32)
        self.errors.append(mx.sqrt(mx.sum(difference**2)) / (mx.sqrt(mx.sum(actual**2)) + 1e-8))

    def arrays(self) -> list[mx.array]:
        arrays = [factor for factors in self.factors.values() for factor in factors]
        if self.previous_input is not None:
            arrays.append(self.previous_input)
        return arrays

    def capture_input(self, text, image, block, **kwargs) -> None:
        self.text_tokens = text.shape[1]
        self.stream_in = mx.concatenate([text, image], axis=1)
        if isinstance(self.policy, TeaCache):
            (shift, scale, _), _ = kwargs["temb_mod_params_img"]
            self.observe((1 + scale) * block.norm1(image) + shift)

    def finish(self, block, hidden_states, **kwargs) -> mx.array:
        if self.skip:
            before_last = hidden_states + self.predict("body")
            predicted = self.predict("last")
            if self.checking:
                self.record_error(predicted, block(hidden_states=before_last, **kwargs) - before_last)
            return before_last + predicted
        output = block(hidden_states=hidden_states, **kwargs)
        self.store("body", hidden_states - self.stream_in)
        self.store("last", output - hidden_states)
        return output


class _LastBlockReuse(BlockReuse):
    """Forecast image features at the boundary between the blocks and the output head."""

    def __init__(self, policy: Spectrum | LastBlockTaylor, sigmas: list[float]) -> None:
        super().__init__(policy)
        self.refresh = policy.refresh_steps(len(sigmas) - 1)
        self.forecasts = {
            branch: SpectralForecast(policy, sigmas) if isinstance(policy, Spectrum) else LastBlockLinearForecast(sigmas)
            for branch in ("cond", "uncond")
        }

    def begin_step(self, step: int, branches=("cond", "uncond")) -> None:
        self.step = step
        if step in self.refresh or any(not self.forecasts[b].arrays() for b in branches):
            self._start_full_step()
        else:
            self.skip = True

    def capture_input(self, text, image, block, **kwargs) -> None:
        self.text_tokens = text.shape[1]
        self.stream_in = image

    def finish(self, block, hidden_states, **kwargs) -> mx.array:
        if self.skip:
            image = self.forecasts[self.branch].predict(self.step, self.stream_in)
            return mx.concatenate([hidden_states[:, :self.text_tokens], image], axis=1)
        output = block(hidden_states=hidden_states, **kwargs)
        image = output[:, self.text_tokens:]
        self.forecasts[self.branch].update(self.step, image)
        return output

    def arrays(self) -> list[mx.array]:
        return [array for forecast in self.forecasts.values() for array in forecast.arrays()]


class _DoubleBlock:
    def __init__(self, block, reuse: BlockReuse, is_first: bool) -> None:
        self.block = block
        self.reuse = reuse
        self.is_first = is_first

    def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
        if self.is_first:
            self.reuse.capture_input(encoder_hidden_states, hidden_states, self.block, **kwargs)
        if self.reuse.skip:
            return encoder_hidden_states, hidden_states
        return self.block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs)


class _SingleBlock:
    def __init__(self, block, reuse: BlockReuse, is_last: bool) -> None:
        self.block = block
        self.reuse = reuse
        self.is_last = is_last

    def __call__(self, *, hidden_states, **kwargs):
        if self.is_last:
            return self.reuse.finish(self.block, hidden_states, **kwargs)
        if self.reuse.skip:
            return hidden_states
        return self.block(hidden_states=hidden_states, **kwargs)


@contextmanager
def reuse_blocks(transformer, policy, sigmas=None):
    reuse = _LastBlockReuse(policy, sigmas) if isinstance(policy, (Spectrum, LastBlockTaylor)) else BlockReuse(policy)
    if policy is None:
        yield reuse
        return
    double_blocks = transformer.transformer_blocks
    single_blocks = transformer.single_transformer_blocks
    transformer.transformer_blocks = [_DoubleBlock(block, reuse, i == 0) for i, block in enumerate(double_blocks)]
    transformer.single_transformer_blocks = [
        _SingleBlock(block, reuse, i == len(single_blocks) - 1) for i, block in enumerate(single_blocks)
    ]
    try:
        yield reuse
    finally:
        transformer.transformer_blocks = double_blocks
        transformer.single_transformer_blocks = single_blocks
