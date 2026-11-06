from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class FixedInterval:
    interval: int
    forecast: bool = False

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
    forecast = True

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step < self.warmup_steps or predicted >= self.max_predicted

    def checks(self, predicted: int) -> bool:
        return predicted >= self.min_predicted

    def threshold(self, step: int) -> float:
        return self.base_threshold * self.decay ** (step / self.total_steps)


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
        # (branch, segment) -> (step the factors were computed at, factors). Timing is kept per
        # branch because guidance policies can leave the unconditional branch out of a full step.
        self.factors = {}
        self.stream_in = None
        self.text_tokens = 0
        self.checked_steps = 0

    def begin_step(self, step: int, branches=("cond", "uncond")) -> None:
        self.step = step
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

    def store(self, segment: str, residual: mx.array) -> None:
        key = (self.branch, segment)
        factors = [residual]
        if self.policy.forecast and key in self.factors:
            stored_step, previous = self.factors[key]
            factors.append((residual - previous[0]) / (self.step - stored_step))
        self.factors[key] = (self.step, factors)

    def predict(self, segment: str) -> mx.array:
        stored_step, factors = self.factors[(self.branch, segment)]
        if len(factors) == 1:
            return factors[0]
        return factors[0] + (self.step - stored_step) * factors[1]

    def record_error(self, predicted: mx.array, actual: mx.array) -> None:
        # Text tokens never reach the output, and their large values would dominate.
        difference = (predicted - actual)[:, self.text_tokens :].astype(mx.float32)
        actual = actual[:, self.text_tokens :].astype(mx.float32)
        self.errors.append(mx.sqrt(mx.sum(difference**2)) / (mx.sqrt(mx.sum(actual**2)) + 1e-8))

    def arrays(self) -> list[mx.array]:
        return [factor for _, factors in self.factors.values() for factor in factors]


class _DoubleBlock:
    def __init__(self, block, reuse: BlockReuse, is_first: bool) -> None:
        self.block = block
        self.reuse = reuse
        self.is_first = is_first

    def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
        if self.is_first:
            self.reuse.text_tokens = encoder_hidden_states.shape[1]
            self.reuse.stream_in = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        if self.reuse.skip:
            return encoder_hidden_states, hidden_states
        return self.block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **kwargs)


class _SingleBlock:
    def __init__(self, block, reuse: BlockReuse, is_last: bool) -> None:
        self.block = block
        self.reuse = reuse
        self.is_last = is_last

    def __call__(self, *, hidden_states, **kwargs):
        reuse = self.reuse
        if reuse.skip:
            if not self.is_last:
                return hidden_states
            before_last = hidden_states + reuse.predict("body")
            predicted = reuse.predict("last")
            if reuse.checking:
                reuse.record_error(predicted, self.block(hidden_states=before_last, **kwargs) - before_last)
            return before_last + predicted
        output = self.block(hidden_states=hidden_states, **kwargs)
        if self.is_last:
            reuse.store("body", hidden_states - reuse.stream_in)
            reuse.store("last", output - hidden_states)
        return output


@contextmanager
def reuse_blocks(transformer, policy):
    reuse = BlockReuse(policy)
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
