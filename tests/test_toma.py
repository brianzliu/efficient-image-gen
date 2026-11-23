import unittest

import mlx.core as mx
from mflux.models.flux2.model.flux2_transformer.pos_embed import Flux2PosEmbed
from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock

from feature_reuse import toma
from feature_reuse.toma import GRID, TILE, merge_plan, select_destinations, split_toma, token_merging

TEXT = 8
DIM = 64


class Transformer:
    def __init__(self) -> None:
        mx.random.seed(0)
        self.single_transformer_blocks = [Flux2SingleTransformerBlock(DIM, 2, 32) for _ in range(2)]


def inputs():
    mx.random.seed(1)
    # Smooth image features, like neighbouring latent tokens, plus random text tokens.
    rows, cols = mx.meshgrid(mx.arange(GRID), mx.arange(GRID), indexing="ij")
    phase = mx.random.normal((1, 1, DIM))
    image = mx.sin(rows.reshape(1, -1, 1) / 5 + phase) + mx.cos(cols.reshape(1, -1, 1) / 7 + phase)
    hidden = mx.concatenate([mx.random.normal((1, TEXT, DIM)), image], axis=1)
    modulation = tuple(0.1 * mx.random.normal((1, 1, DIM)) for _ in range(3))
    image_ids = mx.stack([mx.zeros(GRID * GRID), rows.reshape(-1), cols.reshape(-1), mx.zeros(GRID * GRID)], axis=-1)
    text_ids = mx.stack([mx.zeros(TEXT), mx.zeros(TEXT), mx.zeros(TEXT), mx.arange(TEXT)], axis=-1)
    rope = Flux2PosEmbed(axes_dim=(8, 8, 8, 8))(mx.concatenate([text_ids, image_ids]).astype(mx.int32))
    return hidden, modulation, rope


def run(transformer, ratio):
    hidden, modulation, rope = inputs()
    with token_merging(transformer, None if ratio is None else (ratio, 0)):
        for block in transformer.single_transformer_blocks:
            hidden = block(hidden, modulation, rope)
    return hidden


class SelectionTest(unittest.TestCase):
    def test_picks_one_token_per_cluster(self) -> None:
        # Two clusters of identical tokens: the second pick must come from the other cluster.
        a, b = mx.array([1.0, 0.0]), mx.array([0.0, 1.0])
        x = mx.stack([a, a, a, b, b, b])
        picks = select_destinations(x @ x.T, 2).tolist()
        self.assertEqual({p < 3 for p in picks}, {True, False})

    def test_plan_weights_and_positions(self) -> None:
        hidden, _, _ = inputs()
        merge, unmerge, positions = merge_plan(hidden[:, TEXT:], 0.5)
        tiles, count = (GRID // TILE) ** 2, TILE * TILE // 2
        self.assertEqual(merge.shape, (1, tiles, count, TILE * TILE))
        self.assertTrue(mx.allclose(merge.sum(-1), mx.ones((1, tiles, count)), atol=1e-5).item())
        self.assertTrue(mx.allclose(unmerge.sum(-1), mx.ones((1, tiles, TILE * TILE)), atol=1e-5).item())
        self.assertEqual(len(set(positions[0].tolist())), tiles * count)


class BlockTest(unittest.TestCase):
    def test_keeping_every_token_matches_the_block(self) -> None:
        transformer = Transformer()
        expected = run(transformer, None)
        original = toma.TEMPERATURE
        toma.TEMPERATURE = 1e-4
        try:
            merged = run(transformer, 1.0)
        finally:
            toma.TEMPERATURE = original
        # Near-identical neighbouring tokens still blend slightly, so compare relative error.
        self.assertLess((mx.linalg.norm(merged - expected) / mx.linalg.norm(expected)).item(), 1e-3)

    def test_merging_half_stays_close_and_restores_the_hook(self) -> None:
        transformer = Transformer()
        expected = run(transformer, None)
        merged = run(transformer, 0.5)
        self.assertEqual(merged.shape, expected.shape)
        error = (mx.linalg.norm(merged - expected) / mx.linalg.norm(expected)).item()
        self.assertLess(error, 0.2)
        self.assertTrue(mx.array_equal(run(transformer, None), expected).item())


class SplitTest(unittest.TestCase):
    def test_split_toma(self) -> None:
        self.assertEqual(split_toma("taylor-3+toma-0.5+layer"), ("taylor-3+layer", (0.5, 0)))
        self.assertEqual(split_toma("full-50+toma-0.75-s10"), ("full-50", (0.75, 10)))
        self.assertEqual(split_toma("full-50"), ("full-50", None))
        with self.assertRaises(ValueError):
            split_toma("full-50+toma-0.5+toma-0.75")


if __name__ == "__main__":
    unittest.main()
