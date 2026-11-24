import unittest

import mlx.core as mx
import mlx.nn as nn

from feature_reuse.svdquant import (
    SVDQuantLinear,
    best_layer,
    build,
    low_rank,
    output_error,
    parse_variant,
    shell,
    smoothing,
    split_variant,
)


def layer_with_outliers() -> tuple[nn.Linear, mx.array]:
    mx.random.seed(0)
    linear = nn.Linear(128, 64)
    # One large weight direction, and a few input channels much larger than the rest.
    linear.weight = linear.weight + 5 * mx.random.normal((64, 1)) @ mx.random.normal((1, 128))
    inputs = mx.random.normal((256, 128)) * mx.where(mx.arange(128) < 4, 50.0, 1.0)
    return linear, inputs


class SVDQuantTest(unittest.TestCase):
    def test_plain_build_matches_mlx_quantization(self) -> None:
        linear, inputs = layer_with_outliers()
        q4 = nn.QuantizedLinear.from_linear(linear, 64, 4)
        self.assertTrue(mx.allclose(build(linear, 0, None)(inputs), q4(inputs)).item())

    def test_low_rank_and_smoothing_each_reduce_error(self) -> None:
        linear, inputs = layer_with_outliers()
        plain = output_error(build(linear, 0, None), linear, inputs)
        self.assertLess(output_error(build(linear, 8, None), linear, inputs), plain)
        scale = smoothing(linear.weight, mx.max(mx.abs(inputs), axis=0), 0.5)
        self.assertLess(output_error(build(linear, 0, scale), linear, inputs), plain)

    def test_best_layer_never_worse_than_no_smoothing(self) -> None:
        linear, inputs = layer_with_outliers()
        _, _, error = best_layer(linear, 8, True, mx.max(mx.abs(inputs), axis=0), inputs)
        self.assertLessEqual(error, output_error(build(linear, 8, None), linear, inputs))

    def test_smoothed_layers_always_have_a_scale(self) -> None:
        linear, inputs = layer_with_outliers()
        # Inputs without outliers, so no smoothing usually wins; the layer still needs its scale.
        layer, _, _ = best_layer(linear, 0, True, mx.ones((128,)), mx.random.normal((64, 128)))
        self.assertIn("inv_smooth", layer)

    def test_low_rank_matches_exact_svd(self) -> None:
        linear, _ = layer_with_outliers()
        up, down = low_rank(linear.weight, 4)
        u, s, vt = mx.linalg.svd(linear.weight, stream=mx.cpu)
        exact = (u[:, :4] * s[:4]) @ vt[:4]
        self.assertLess((mx.linalg.norm(up @ down - exact) / mx.linalg.norm(exact)).item(), 1e-3)

    def test_shell_loads_built_weights(self) -> None:
        linear, inputs = layer_with_outliers()
        built = build(linear, 8, smoothing(linear.weight, mx.max(mx.abs(inputs), axis=0), 0.5))
        model = nn.Sequential(nn.Linear(128, 64))
        shell(model, 8, True)
        self.assertIsInstance(model.layers[0], SVDQuantLinear)
        model.layers[0].update(built.parameters())
        self.assertTrue(mx.array_equal(model(inputs), built(inputs)).item())

    def test_parse_and_split_variant(self) -> None:
        self.assertEqual(parse_variant("svdq-32"), ("svdq", 32, True))
        self.assertEqual(parse_variant("lowrank-16"), ("svdq", 16, False))
        self.assertEqual(parse_variant("smooth"), ("svdq", 0, True))
        with self.assertRaises(ValueError):
            parse_variant("q3")
        self.assertEqual(split_variant("taylor-3+svdq-32+dpmpp2m", "bf16"), ("taylor-3+dpmpp2m", "svdq-32"))
        self.assertEqual(split_variant("full-50", "bf16"), ("full-50", "bf16"))


if __name__ == "__main__":
    unittest.main()
