"""Synchronized MLX profiling for diagnostic runs."""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import mlx.core as mx
from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils
from mflux.models.flux2.model.flux2_transformer.attention import Flux2Attention
from mflux.models.flux2.model.flux2_transformer.feed_forward import Flux2FeedForward
from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import Flux2ParallelSelfAttention
from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
from mlx import nn


def linear_macs(args: tuple, kwargs: dict, output: mx.array) -> int:
    return output.size * args[1].shape[-1]


def attention_macs(args: tuple, kwargs: dict, output: mx.array) -> int:
    heads, tokens, head_dim = kwargs["query"].shape[1:]
    return 2 * heads * tokens**2 * head_dim


BLOCK_TARGETS = [
    (Flux2Transformer, "__call__", "transformer", None),
    (Flux2TransformerBlock, "__call__", "double_block", None),
    (Flux2SingleTransformerBlock, "__call__", "single_block", None),
]
OP_TARGETS = BLOCK_TARGETS + [
    (Flux2Attention, "__call__", "attention", None),
    (Flux2ParallelSelfAttention, "__call__", "attention", None),
    (Flux2FeedForward, "__call__", "feed_forward", None),
    (AttentionUtils, "compute_attention", "dot_product", attention_macs),
    (AttentionUtils, "apply_rope_bshd", "rope", None),
    (nn.QuantizedLinear, "__call__", "linear", linear_macs),
    (nn.Linear, "__call__", "linear", linear_macs),
]

def arrays(value) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, nn.Module):
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [array for item in value for array in arrays(item)]
    return []


@dataclass
class _Frame:
    name: str
    peak: int
    child_seconds: float = 0.0


def memory_snapshot() -> tuple[int, int]:
    """Return active and peak MLX bytes for the current measurement window."""
    active = mx.get_active_memory()
    return active, max(active, mx.get_peak_memory())


class Profiler:
    def __init__(self) -> None:
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.macs = defaultdict(int)
        self.transient = defaultdict(int)
        self.path = []

    def run(self, name: str, count_macs, fn, args: tuple, kwargs: dict):
        # Finish pending work first, so it counts toward the caller and not this operation.
        mx.eval(arrays(args), arrays(kwargs))
        active, peak = memory_snapshot()
        if self.path:
            parent = self.path[-1]
            parent.peak = max(parent.peak, peak)
        frame = _Frame(name, active)
        self.path.append(frame)
        path = "/".join(scope.name for scope in self.path)
        mx.reset_peak_memory()
        start = perf_counter()
        try:
            output = fn(*args, **kwargs)
            mx.eval(arrays(output))
        finally:
            elapsed = perf_counter() - start
            frame.peak = max(frame.peak, memory_snapshot()[1])
            self.path.pop()
            if self.path:
                parent.child_seconds += elapsed
                parent.peak = max(parent.peak, frame.peak)
        self.transient[path] = max(self.transient[path], frame.peak - active)
        self.seconds[path] += elapsed - frame.child_seconds
        self.calls[path] += 1
        if count_macs:
            self.macs[path] += count_macs(args, kwargs, output)
        return output


@contextmanager
def profiled(targets: list, linear_names: dict[int, str] | None = None):
    profiler = Profiler()
    originals = []
    try:
        for owner, attribute, name, count_macs in targets:
            original = owner.__dict__[attribute]
            is_static = isinstance(original, staticmethod)
            function = original.__func__ if is_static else original

            def wrapper(*args, _name=name, _count=count_macs, _function=function, **kwargs):
                label = _name
                if linear_names and _name == "linear" and id(args[0]) in linear_names:
                    label = f"linear:{linear_names[id(args[0])]}"
                return profiler.run(label, _count, _function, args, kwargs)

            setattr(owner, attribute, staticmethod(wrapper) if is_static else wrapper)
            originals.append((owner, attribute, original))
        yield profiler
    finally:
        for owner, attribute, original in originals:
            setattr(owner, attribute, original)


def profile_summary(profiler: Profiler, step_seconds: float) -> dict:
    return {
        "denoise_s": step_seconds,
        "paths": {
            path: {
                "seconds": seconds,
                "calls": profiler.calls[path],
                "macs": profiler.macs[path],
                "transient_mb": profiler.transient[path] / 1000**2,
            }
            for path, seconds in sorted(profiler.seconds.items(), key=lambda item: -item[1])
        },
    }
