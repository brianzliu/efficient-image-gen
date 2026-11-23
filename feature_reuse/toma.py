import re
from contextlib import contextmanager

import mlx.core as mx
from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock

from feature_reuse.streaming import StreamedLayer

# Image tokens form a GRID x GRID raster at 512 x 512. Destinations are chosen inside TILE x TILE tiles
# (64 tiles, the paper's default for FLUX), so selection and merge weights stay local.
GRID = 32
TILE = 4
# Softmax temperature on cosine similarity. Neighbouring tokens are 0.9-0.99 similar, so it must be very
# low or every token becomes a tile average: 0.1 blurred images badly, 0.002 gives near-hard assignment.
TEMPERATURE = 0.002


def _tiles(x: mx.array) -> mx.array:
    """(B, GRID*GRID, d) raster tokens -> (B, tiles, TILE*TILE, d)."""
    b, _, d = x.shape
    n = GRID // TILE
    x = x.reshape(b, n, TILE, n, TILE, d).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, n * n, TILE * TILE, d)


def _untiles(x: mx.array) -> mx.array:
    b, _, _, d = x.shape
    n = GRID // TILE
    x = x.reshape(b, n, n, TILE, TILE, d).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, GRID * GRID, d)


# Raster index of each tile slot, for gathering the destinations' rotary embeddings.
_SLOT_INDEX = _tiles(mx.arange(GRID * GRID).reshape(1, -1, 1))[0, :, :, 0]


def select_destinations(similarity: mx.array, count: int) -> mx.array:
    """Greedy facility location per tile: pick `count` tokens that best cover the others.

    similarity: (..., n, n) cosine similarities. Returns (..., count) slot indices in pick order.
    """
    *lead, n, _ = similarity.shape
    covered = mx.zeros((*lead, n), dtype=similarity.dtype)
    chosen = mx.zeros((*lead, n), dtype=mx.bool_)
    picks = []
    for _ in range(count):
        # Gain of adding token i: how much it raises each token's best similarity to the chosen set.
        gains = mx.sum(mx.maximum(similarity - covered[..., None, :], 0), axis=-1)
        pick = mx.argmax(mx.where(chosen, -mx.inf, gains), axis=-1)
        row = mx.take_along_axis(similarity, pick[..., None, None], axis=-2)[..., 0, :]
        covered = mx.maximum(covered, row)
        chosen = chosen | (mx.arange(n) == pick[..., None])
        picks.append(pick)
    return mx.stack(picks, axis=-1)


def merge_plan(image: mx.array, ratio: float, temperature: float = TEMPERATURE):
    """Destinations and merge/unmerge weights for one pass (ToMA, Lu et al., 2025).

    Returns (merge, unmerge, positions): merge (B, tiles, k, n) averages each destination's tokens,
    unmerge (B, tiles, n, k) rebuilds each token from the destinations, positions (B, tiles * k) are the
    destinations' raster indices.
    """
    count = max(1, round(ratio * TILE * TILE))
    x = _tiles(image.astype(mx.float32))
    x = x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    similarity = x @ x.swapaxes(-1, -2)
    picks = select_destinations(similarity, count)
    # Each source token spreads over the destinations by a softmax of its similarity to them.
    logits = mx.take_along_axis(similarity, picks[..., None], axis=-2) / temperature
    assignment = mx.softmax(logits, axis=-2)
    merge = assignment / mx.sum(assignment, axis=-1, keepdims=True)
    unmerge = assignment.swapaxes(-1, -2)
    positions = mx.take_along_axis(_SLOT_INDEX[None], picks.reshape(picks.shape[0], -1, count), axis=-1)
    return merge, unmerge, positions.reshape(picks.shape[0], -1)


class ToMA:
    """Merge image tokens in the single-stream blocks, which take about 80% of Klein's time.

    The plan is computed at the first merged block of every transformer pass and shared by the rest,
    as the paper does for FLUX. Text tokens are left alone.
    """

    def __init__(self, ratio: float, blocks: list) -> None:
        self.ratio = ratio
        self.first = id(blocks[0])
        self.merged = {id(b) for b in blocks}
        self.plan = None

    def __call__(self, block, hidden_states, temb_mod_params, image_rotary_emb, **kwargs):
        if id(block) not in self.merged:
            return _original_call(block, hidden_states, temb_mod_params, image_rotary_emb, **kwargs)
        shift, scale, gate = temb_mod_params
        normed = (1 + scale) * block.norm(hidden_states) + shift
        # Text tokens come first; the conditional and unconditional prompts may differ in length.
        t = hidden_states.shape[1] - GRID * GRID
        text, image = normed[:, :t], normed[:, t:]
        if id(block) == self.first:
            self.plan = merge_plan(image, self.ratio, TEMPERATURE)
            mx.eval(*self.plan)
        merge, unmerge, positions = self.plan
        dtype = image.dtype
        merged = (merge.astype(dtype) @ _tiles(image)).reshape(image.shape[0], -1, image.shape[-1])
        cos, sin = image_rotary_emb
        rope = tuple(mx.concatenate([r[:t], r[t:][positions[0]]], axis=0) for r in (cos, sin))
        output = block.attn(mx.concatenate([text, merged], axis=1), rope)
        count = merge.shape[-2]
        out_image = output[:, t:].reshape(output.shape[0], -1, count, output.shape[-1])
        out_image = _untiles(unmerge.astype(output.dtype) @ out_image)
        return hidden_states + gate * mx.concatenate([output[:, :t], out_image], axis=1)


_original_call = Flux2SingleTransformerBlock.__call__


@contextmanager
def token_merging(transformer, setting: tuple[float, int] | None):
    """Merge image tokens in the single-stream blocks while active.

    setting is (ratio, skip): the fraction of image tokens kept, and how many leading single blocks run unmerged.
    """
    if setting is None:
        yield
        return
    # Streaming may have wrapped the blocks in StreamedLayer; match the real modules.
    blocks = [b.weights.module if isinstance(b, StreamedLayer) else b for b in transformer.single_transformer_blocks]
    ratio, skip = setting
    toma = ToMA(ratio, blocks[skip:])

    def call(self, hidden_states, temb_mod_params, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
        return toma(self, hidden_states, temb_mod_params, image_rotary_emb)

    Flux2SingleTransformerBlock.__call__ = call
    try:
        yield toma
    finally:
        Flux2SingleTransformerBlock.__call__ = _original_call


def split_toma(name: str) -> tuple[str, tuple[float, int] | None]:
    """Split a "+toma-R" or "+toma-R-sN" suffix off a config name.

    R is the fraction of image tokens kept; N leading single blocks run unmerged (default 0).
    """
    base, *suffixes = name.split("+")
    found = [m for s in suffixes if (m := re.fullmatch(r"toma-(0?\.\d+|1(?:\.0)?)(?:-s(\d+))?", s))]
    if len(found) > 1:
        raise ValueError(f"repeated toma suffix in config {name!r}")
    rest = [s for s in suffixes if not re.fullmatch(r"toma-.*", s)]
    setting = (float(found[0][1]), int(found[0][2] or 0)) if found else None
    return "+".join([base] + rest), setting
