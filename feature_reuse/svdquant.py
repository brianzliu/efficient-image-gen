import gc
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mflux.models.common.compute_precision.compute_precision import ComputePrecision
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mlx.utils import tree_map_with_path

GROUP_SIZE = 64
BITS = 4
# Smoothing strengths tried per layer; None means no smoothing.
ALPHAS = (None, 0.25, 0.5, 0.75)


class SVDQuantLinear(nn.Module):
    """A 4-bit linear layer with optional smoothing and a 16-bit low-rank branch (SVDQuant, weights only).

    y = Q(R) x' + up (down x') + bias, where x' = x / smooth and W * smooth = up down + R.
    """

    def __init__(self, input_dims: int, output_dims: int, rank: int, smooth: bool, bias: bool, dtype=mx.bfloat16):
        super().__init__()
        self.group_size, self.bits, self.rank = GROUP_SIZE, BITS, rank
        # Placeholders with the right shapes; load_weights or build fills them in.
        self.weight = mx.zeros((output_dims, input_dims * BITS // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // GROUP_SIZE), dtype=dtype)
        self.biases = mx.zeros((output_dims, input_dims // GROUP_SIZE), dtype=dtype)
        if rank:
            self.down = mx.zeros((rank, input_dims), dtype=dtype)
            self.up = mx.zeros((output_dims, rank), dtype=dtype)
        if smooth:
            self.inv_smooth = mx.ones((input_dims,), dtype=dtype)
        if bias:
            self.bias = mx.zeros((output_dims,), dtype=dtype)

    def __call__(self, x: mx.array) -> mx.array:
        if "inv_smooth" in self:
            x = x * self.inv_smooth
        y = mx.quantized_matmul(
            x, self.weight, self.scales, self.biases, transpose=True, group_size=self.group_size, bits=self.bits
        )
        if self.rank:
            y = y + (x @ self.down.T) @ self.up.T
        if "bias" in self:
            y = y + self.bias
        return y


def low_rank(weight: mx.array, rank: int, iterations: int = 4) -> tuple[mx.array, mx.array]:
    """Top-rank factors up, down of a float32 matrix by randomized SVD."""
    sketch = weight @ mx.random.normal((weight.shape[1], rank + 16), key=mx.random.key(0))
    for _ in range(iterations):
        q, _ = mx.linalg.qr(sketch, stream=mx.cpu)
        sketch = weight @ (weight.T @ q)
    q, _ = mx.linalg.qr(sketch, stream=mx.cpu)
    u, s, vt = np.linalg.svd(np.array(q.T @ weight), full_matrices=False)
    up = q @ mx.array(u[:, :rank] * s[:rank])
    return up, mx.array(vt[:rank])


def smoothing(weight: mx.array, input_max: mx.array, alpha: float) -> mx.array:
    """Per-input-channel scale that moves activation outliers into the weights (SmoothQuant / SVDQuant)."""
    weight_max = mx.max(mx.abs(weight), axis=0)
    scale = mx.power(mx.maximum(input_max, 1e-5), alpha) / mx.power(mx.maximum(weight_max, 1e-5), 1 - alpha)
    # Center on 1 so neither the activations nor the weights drift far from their own range.
    return scale / mx.exp(mx.mean(mx.log(scale)))


def build(linear: nn.Linear, rank: int, scale: mx.array | None) -> SVDQuantLinear:
    dtype = linear.weight.dtype
    output_dims, input_dims = linear.weight.shape
    layer = SVDQuantLinear(input_dims, output_dims, rank, scale is not None, "bias" in linear, dtype)
    weight = linear.weight.astype(mx.float32)
    if scale is not None:
        weight = weight * scale
        layer.inv_smooth = (1 / scale).astype(dtype)
    if rank:
        up, down = low_rank(weight, rank)
        weight = weight - up @ down
        layer.up, layer.down = up.astype(dtype), down.astype(dtype)
    layer.weight, layer.scales, layer.biases = mx.quantize(weight.astype(dtype), GROUP_SIZE, BITS)
    if "bias" in linear:
        layer.bias = linear.bias
    mx.eval(layer.parameters())
    return layer


def output_error(layer: nn.Module, linear: nn.Linear, inputs: mx.array) -> float:
    expected = linear(inputs)
    return (mx.linalg.norm(layer(inputs) - expected) / mx.linalg.norm(expected)).item()


def best_layer(linear: nn.Linear, rank: int, smooth: bool, input_max, inputs):
    """Build the layer with the smoothing strength that gives the lowest output error on calibration inputs.

    Returns (layer, alpha, error).
    """
    best = None
    for alpha in ALPHAS if smooth else (None,):
        if alpha is None:
            # Smoothed variants keep an all-ones scale here, so every layer has the same parameters.
            scale = mx.ones((linear.weight.shape[1],)) if smooth else None
        else:
            scale = smoothing(linear.weight.astype(mx.float32), input_max, alpha)
        layer = build(linear, rank, scale)
        error = output_error(layer, linear, inputs)
        if best is None or error < best[2]:
            best = (layer, alpha, error)
    return best


def _convert(transformer: nn.Module, convert) -> None:
    leaves = tree_map_with_path(convert, transformer.leaf_modules(), is_leaf=nn.Module.is_module)
    transformer.update_modules(leaves)


def quantize(transformer: nn.Module, bits: int = BITS) -> None:
    # Same as the mflux conversions: every quantizable layer, affine, group size 64.
    nn.quantize(transformer, GROUP_SIZE, bits, class_predicate=lambda _, m: hasattr(m, "to_quantized"))


def parse_variant(name: str) -> tuple[str, int, bool]:
    """Return (kind, rank, smooth) for bf16, q4, q8, smooth, lowrank-R or svdq-R."""
    if name in ("bf16", "q4", "q8"):
        return name, 0, False
    if name == "smooth":
        return "svdq", 0, True
    if match := re.fullmatch(r"(lowrank|svdq)-(\d+)", name):
        return "svdq", int(match[2]), match[1] == "svdq"
    raise ValueError(f"unknown transformer variant {name!r}")


def shell(transformer: nn.Module, rank: int, smooth: bool) -> None:
    """Swap every linear layer for an empty SVDQuantLinear of the same shape."""

    def convert(_, m):
        if not isinstance(m, nn.Linear):
            return m
        output_dims, input_dims = m.weight.shape
        return SVDQuantLinear(input_dims, output_dims, rank, smooth, "bias" in m)

    _convert(transformer, convert)


def load_transformer(model, variant: str, variants_dir: Path) -> None:
    """Replace model.transformer with a saved variant, using fp16 compute like load_klein."""
    kind, rank, smooth = parse_variant(variant)
    # Free the current transformer first; holding BF16 and another variant at once would need about 10 GB.
    model.transformer = None
    gc.collect()
    mx.clear_cache()
    transformer = Flux2Transformer(**model.model_config.transformer_overrides)
    if kind in ("q4", "q8"):
        quantize(transformer, int(kind[1]))
    elif kind == "svdq":
        shell(transformer, rank, smooth)
    transformer.load_weights(str(variants_dir / f"{variant}.safetensors"))
    transformer.apply_compute_precision(ComputePrecision(mx.float16))
    model.transformer = transformer


def split_variant(name: str, default: str) -> tuple[str, str]:
    """Split a transformer variant suffix ("+bf16", "+q4", "+svdq-32", ...) off a config name."""
    base, *suffixes = name.split("+")
    found = []
    for suffix in suffixes:
        try:
            parse_variant(suffix)
            found.append(suffix)
        except ValueError:
            pass
    if len(found) > 1:
        raise ValueError(f"repeated transformer variant in config {name!r}")
    return "+".join([base] + [s for s in suffixes if s not in found]), (found or [default])[0]
