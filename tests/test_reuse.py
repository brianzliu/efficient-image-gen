import importlib.util
import math
import unittest
from pathlib import Path

import mlx.core as mx

from feature_reuse.reuse import FixedInterval, SpeCa, reuse_blocks


class DoubleBlock:
    def __call__(self, *, hidden_states, encoder_hidden_states):
        return encoder_hidden_states * 6.0, hidden_states * 6.0


class SingleBlock:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, *, hidden_states):
        return hidden_states + self.offset


class TextOnlyBlock(SingleBlock):
    def __call__(self, *, hidden_states):
        return mx.concatenate([hidden_states[:, :2] + self.offset, hidden_states[:, 2:]], axis=1)


class Transformer:
    def __init__(self, last_block=None) -> None:
        self.transformer_blocks = [DoubleBlock()]
        self.single_transformer_blocks = [SingleBlock(1.0), last_block or SingleBlock(10.0)]

    def __call__(self, image: float) -> mx.array:
        text, image = self.transformer_blocks[0](hidden_states=tokens(image), encoder_hidden_states=tokens(5.0))
        stream = mx.concatenate([text, image], axis=1)
        for block in self.single_transformer_blocks:
            stream = block(hidden_states=stream)
        return stream[:, 2:]


def tokens(value: float) -> mx.array:
    return mx.full((1, 2), value)


def run_step(transformer, reuse, step: int, image: float = 1.0, branches=("cond",)) -> mx.array:
    reuse.begin_step(step)
    for branch in branches:
        reuse.branch = branch
        output = transformer(image)
    return output


class FixedIntervalTest(unittest.TestCase):
    def test_reused_step_adds_stored_residual(self) -> None:
        transformer = Transformer()
        with reuse_blocks(transformer, FixedInterval(2)) as reuse:
            self.assertTrue(mx.array_equal(run_step(transformer, reuse, 0), tokens(17.0)).item())
            self.assertTrue(mx.array_equal(run_step(transformer, reuse, 1, image=4.0), tokens(20.0)).item())
        self.assertEqual(reuse.full_steps, 1)

    def test_branches_keep_separate_residuals(self) -> None:
        transformer = Transformer()
        with reuse_blocks(transformer, FixedInterval(2)) as reuse:
            reuse.begin_step(0)
            reuse.branch = "cond"
            transformer(1.0)
            reuse.branch = "uncond"
            transformer(2.0)
            reuse.begin_step(1)
            self.assertTrue(mx.array_equal(transformer(0.0), tokens(21.0)).item())

    def test_forecast_extrapolates_linearly(self) -> None:
        transformer = Transformer()
        with reuse_blocks(transformer, FixedInterval(2, forecast=True)) as reuse:
            run_step(transformer, reuse, 0)
            run_step(transformer, reuse, 2, image=2.0)
            self.assertTrue(mx.array_equal(run_step(transformer, reuse, 3, image=0.0), tokens(23.5)).item())

    def test_second_order_forecast_is_exact_for_quadratic(self) -> None:
        # Inputs x = t^2 make the body residual 5x + 1 quadratic in the step t.
        outputs = {}
        for order in (1, 2):
            transformer = Transformer()
            with reuse_blocks(transformer, FixedInterval(3, forecast=True, order=order)) as reuse:
                for step in (0, 3, 6):
                    run_step(transformer, reuse, step, image=float(step**2))
                outputs[order] = run_step(transformer, reuse, 8, image=64.0)
            self.assertEqual(len(reuse.arrays()), 2 * (order + 1))
        # Fully computing step 8 would give 6 * 64 + 11 = 395.
        self.assertTrue(mx.array_equal(outputs[2], tokens(395.0)).item())
        self.assertTrue(mx.array_equal(outputs[1], tokens(345.0)).item())

    def test_second_order_forecast_handles_uneven_spacing(self) -> None:
        class Steps:
            forecast, order = True, 2

            def runs_blocks(self, step: int, predicted: int) -> bool:
                return step in (0, 2, 6)

            def checks(self, predicted: int) -> bool:
                return False

        transformer = Transformer()
        with reuse_blocks(transformer, Steps()) as reuse:
            for step in (0, 2, 6):
                run_step(transformer, reuse, step, image=float(step**2))
            self.assertTrue(mx.array_equal(run_step(transformer, reuse, 7, image=49.0), tokens(305.0)).item())

    def test_blocks_are_restored(self) -> None:
        transformer = Transformer()
        single_blocks = transformer.single_transformer_blocks
        with reuse_blocks(transformer, FixedInterval(1)):
            self.assertIsNot(transformer.single_transformer_blocks, single_blocks)
        self.assertIs(transformer.single_transformer_blocks, single_blocks)


def quadratic_forecast(policy, steps=(0, 3, 6), target=8) -> mx.array:
    # Inputs x = t^2 make the body residual 5x + 1 quadratic in the step t.
    transformer = Transformer()
    with reuse_blocks(transformer, policy) as reuse:
        for step in steps:
            run_step(transformer, reuse, step, image=float(step**2))
        return run_step(transformer, reuse, target, image=float(target**2))


class CurvatureWeightTest(unittest.TestCase):
    def test_default_weight_is_undamped(self) -> None:
        self.assertEqual(FixedInterval(3, forecast=True, order=2).curvature_weight, 1.0)
        self.assertEqual(SpeCa(0.1, 0.5, 50, order=2).curvature_weight, 1.0)
        explicit = quadratic_forecast(FixedInterval(3, forecast=True, order=2, curvature_weight=1.0))
        self.assertTrue(mx.array_equal(explicit, quadratic_forecast(FixedInterval(3, forecast=True, order=2))).item())
        self.assertTrue(mx.array_equal(explicit, tokens(395.0)).item())

    def test_zero_weight_matches_first_order(self) -> None:
        damped = quadratic_forecast(FixedInterval(3, forecast=True, order=2, curvature_weight=0.0))
        first = quadratic_forecast(FixedInterval(3, forecast=True, order=1))
        self.assertTrue(mx.array_equal(damped, first).item())
        self.assertTrue(mx.array_equal(damped, tokens(345.0)).item())

    def test_intermediate_weights_interpolate_the_correction(self) -> None:
        # First order predicts 345 and the curvature correction adds 50.
        for weight in (0.1, 0.25, 0.5, 0.75):
            output = quadratic_forecast(FixedInterval(3, forecast=True, order=2, curvature_weight=weight))
            self.assertTrue(mx.allclose(output, tokens(345.0 + 50.0 * weight)).item(), weight)

    def test_damping_respects_uneven_spacing(self) -> None:
        class Steps:
            forecast, order = True, 2

            def __init__(self, weight: float) -> None:
                self.curvature_weight = weight

            def runs_blocks(self, step: int, predicted: int) -> bool:
                return step in (0, 2, 6)

            def checks(self, predicted: int) -> bool:
                return False

        # Exact quadratic gives 305 at step 7; linear extrapolation of the body residual from steps 2 and 6 gives 280.
        for weight, expected in ((0.0, 280.0), (0.5, 292.5), (1.0, 305.0)):
            self.assertTrue(mx.allclose(quadratic_forecast(Steps(weight), (0, 2, 6), 7), tokens(expected)).item())

    def test_zero_weight_matches_first_order_under_adaptive_rejection(self) -> None:
        # Uneven residual growth forces rejections, so the full-step spacing becomes irregular.
        def trace(order: int, weight: float = 1.0):
            last_block = SingleBlock(10.0)
            transformer = Transformer(last_block)
            policy = SpeCa(0.01, 1.0, 12, warmup_steps=1, max_predicted=4, order=order, curvature_weight=weight)
            outputs = []
            with reuse_blocks(transformer, policy) as reuse:
                for step in range(12):
                    last_block.offset = 10.0 + 0.3 * step**2
                    output = run_step(transformer, reuse, step, image=float(step), branches=("cond", "uncond"))
                    if not reuse.confirm_step():
                        output = run_step(transformer, reuse, step, image=float(step), branches=("cond", "uncond"))
                    outputs.append(output)
            return outputs, (reuse.full_steps, reuse.rejected_steps, reuse.checked_steps)

        first, first_counts = trace(1)
        damped, damped_counts = trace(2, 0.0)
        self.assertGreater(first_counts[1], 0)
        self.assertEqual(first_counts, damped_counts)
        for a, b in zip(first, damped):
            self.assertTrue(mx.array_equal(a, b).item())

    def test_zero_weight_ignores_non_finite_curvature(self) -> None:
        transformer = Transformer()
        policy = FixedInterval(3, forecast=True, order=2, curvature_weight=0.0)
        with reuse_blocks(transformer, policy) as reuse:
            for step in (0, 3, 6):
                run_step(transformer, reuse, step, image=float(step**2))
            for key, factors in reuse.factors.items():
                reuse.factors[key] = factors[:2] + [mx.full(factors[2].shape, math.inf)]
            output = run_step(transformer, reuse, 8, image=64.0)
        self.assertTrue(mx.all(mx.isfinite(output)).item())
        self.assertTrue(mx.array_equal(output, tokens(345.0)).item())

    def test_float16_damped_forecasts_stay_finite_and_bounded(self) -> None:
        # Activation-scale residuals in fp16 across a long, mostly reused schedule.
        class Block:
            def __call__(self, *, hidden_states):
                return hidden_states + mx.sin(hidden_states).astype(hidden_states.dtype) * 900

        class Fp16Transformer(Transformer):
            def __init__(self) -> None:
                self.transformer_blocks = [DoubleBlock()]
                self.single_transformer_blocks = [Block(), Block()]

        bounds = []
        for weight in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            transformer = Fp16Transformer()
            with reuse_blocks(transformer, FixedInterval(3, forecast=True, order=2, curvature_weight=weight)) as reuse:
                for step in range(50):
                    reuse.begin_step(step)
                    reuse.branch = "cond"
                    image = mx.full((1, 2), 20.0 + 0.5 * step).astype(mx.float16)
                    text, image = transformer.transformer_blocks[0](
                        hidden_states=image, encoder_hidden_states=mx.full((1, 2), 5.0).astype(mx.float16)
                    )
                    stream = mx.concatenate([text, image], axis=1)
                    for block in transformer.single_transformer_blocks:
                        stream = block(hidden_states=stream)
                    self.assertTrue(mx.all(mx.isfinite(stream)).item(), (weight, step))
                    bounds.append(mx.max(mx.abs(stream)).item())
            self.assertTrue(all(mx.all(mx.isfinite(a)).item() for a in reuse.arrays()), weight)
        self.assertLess(max(bounds), 60000.0)

    def test_rejects_non_finite_weight(self) -> None:
        for weight in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                FixedInterval(3, forecast=True, order=2, curvature_weight=weight)
            with self.assertRaises(ValueError):
                SpeCa(0.1, 0.5, 50, order=2, curvature_weight=weight)


class ParseConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark-klein.py"
        spec = importlib.util.spec_from_file_location("benchmark_klein", path)
        cls.benchmark = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.benchmark)
        except ImportError as error:  # mflux is only needed to run generation.
            raise unittest.SkipTest(f"benchmark script imports unavailable: {error}")

    def test_existing_names_are_undamped(self) -> None:
        for name in ("taylor-3", "taylor2-3", "speca2-0.1-0.5"):
            self.assertEqual(self.benchmark.parse_config(name)[1].curvature_weight, 1.0)

    def test_damping_suffix(self) -> None:
        steps, policy = self.benchmark.parse_config("taylor2-3-c0.25")
        self.assertEqual((steps, policy.interval, policy.order, policy.curvature_weight), (50, 3, 2, 0.25))
        self.assertEqual(self.benchmark.parse_config("speca2-0.1-0.5-c0.5")[1].curvature_weight, 0.5)

    def test_damping_requires_second_order(self) -> None:
        for name in ("taylor-3-c0.5", "taylor1-3-c0.5", "taylor3-3-c0.5"):
            with self.assertRaises(ValueError):
                self.benchmark.parse_config(name)


class SpeCaTest(unittest.TestCase):
    policy = SpeCa(0.1, 1.0, 10, warmup_steps=1, min_predicted=1, max_predicted=3)

    def test_threshold_decays(self) -> None:
        self.assertAlmostEqual(SpeCa(0.8, 0.25, 50).threshold(25), 0.4)

    def test_accurate_predictions_run_until_the_limit(self) -> None:
        transformer = Transformer()
        kinds = []
        with reuse_blocks(transformer, self.policy) as reuse:
            for step in range(5):
                run_step(transformer, reuse, step)
                self.assertTrue(reuse.confirm_step())
                kinds.append(reuse.skip)
        self.assertEqual(kinds, [False, True, True, True, False])

    def test_failed_check_reruns_with_every_block(self) -> None:
        last_block = SingleBlock(10.0)
        transformer = Transformer(last_block)
        with reuse_blocks(transformer, self.policy) as reuse:
            run_step(transformer, reuse, 0)
            run_step(transformer, reuse, 1)
            reuse.confirm_step()
            last_block.offset = 20.0
            run_step(transformer, reuse, 2)
            self.assertFalse(reuse.confirm_step())
            output = transformer(1.0)
        self.assertEqual((reuse.rejected_steps, reuse.full_steps, reuse.predicted), (1, 2, 0))
        self.assertTrue(mx.array_equal(output, tokens(27.0)).item())

    def test_worst_branch_decides(self) -> None:
        last_block = SingleBlock(10.0)
        transformer = Transformer(last_block)
        with reuse_blocks(transformer, self.policy) as reuse:
            run_step(transformer, reuse, 0, branches=("cond", "uncond"))
            run_step(transformer, reuse, 1, branches=("cond", "uncond"))
            reuse.confirm_step()
            reuse.begin_step(2)
            reuse.branch = "cond"
            transformer(1.0)
            last_block.offset = 20.0
            reuse.branch = "uncond"
            transformer(1.0)
            self.assertFalse(reuse.confirm_step())

    def test_error_ignores_text_tokens(self) -> None:
        last_block = TextOnlyBlock(1.0)
        transformer = Transformer(last_block)
        with reuse_blocks(transformer, self.policy) as reuse:
            run_step(transformer, reuse, 0)
            run_step(transformer, reuse, 1)
            reuse.confirm_step()
            last_block.offset = 100.0
            run_step(transformer, reuse, 2)
            self.assertTrue(reuse.confirm_step())


if __name__ == "__main__":
    unittest.main()
