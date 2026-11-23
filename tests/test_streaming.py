import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from feature_reuse.streaming import Streamer, split_mode


class Stack(nn.Module):
    def __init__(self, keys: tuple[str, ...], count: int) -> None:
        super().__init__()
        self.inp = nn.Linear(4, 4)
        for key in keys:
            setattr(self, key, [nn.Linear(4, 4) for _ in range(count)])
        self.order = keys

    def __call__(self, x: mx.array) -> mx.array:
        x = self.inp(x)
        for key in self.order:
            for layer in getattr(self, key):
                x = layer(x)
        return x


class Model:
    def __init__(self) -> None:
        mx.random.seed(0)
        self.text_encoder = Stack(("layers",), 3)
        self.transformer = Stack(("transformer_blocks", "single_transformer_blocks"), 2)
        self.vae = Stack(("decoder",), 1)
        self.vae.encoder = nn.Linear(4, 4)


def loaded_bytes(module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(module.parameters()))


class StreamerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Model()
        self.streamer = Streamer(self.model, Path(self.tmp.name))
        self.x = mx.ones((1, 4))
        self.expected = {name: getattr(self.model, name)(self.x) for name in ("text_encoder", "transformer", "vae")}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_modes_match_resident_output(self) -> None:
        for mode in ("resident", "stage", "layer"):
            self.streamer.prepare(mode)
            for name, expected in self.expected.items():
                with self.streamer.stage(name, mode):
                    output = getattr(self.model, name)(self.x)
                self.assertTrue(mx.array_equal(output, expected).item(), (mode, name))

    def test_stage_exit_frees_weights(self) -> None:
        self.streamer.prepare("layer")
        with self.streamer.stage("transformer", "layer"):
            self.model.transformer(self.x)
        self.assertEqual(loaded_bytes(self.model.transformer), 0)
        self.assertIsInstance(self.model.transformer.transformer_blocks[0], nn.Linear)

    def test_unused_children_are_left_alone(self) -> None:
        self.streamer.prepare("stage")
        self.assertGreater(loaded_bytes(self.model.vae.encoder), 0)

    def test_bind_follows_a_swapped_part(self) -> None:
        replacement = Stack(("transformer_blocks", "single_transformer_blocks"), 2)
        expected = replacement(self.x)
        self.streamer.release("transformer")
        self.model.transformer = replacement
        self.streamer.bind("transformer", Path(self.tmp.name) / "swapped")
        self.streamer.prepare("layer")
        with self.streamer.stage("transformer", "layer"):
            output = self.model.transformer(self.x)
        self.assertTrue(mx.array_equal(output, expected).item())
        self.assertEqual(loaded_bytes(replacement), 0)

    def test_split_mode(self) -> None:
        self.assertEqual(split_mode("taylor-3+layer+dpmpp2m"), ("taylor-3+dpmpp2m", "layer"))
        self.assertEqual(split_mode("full-50"), ("full-50", "resident"))
        with self.assertRaises(ValueError):
            split_mode("full-50+stage+layer")


if __name__ == "__main__":
    unittest.main()
