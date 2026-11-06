import unittest

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
    reuse.begin_step(step, branches)
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

    def test_forecast_timing_is_per_branch(self) -> None:
        # The image body residual is 5x + 1 and the last block adds 10. Uncond is computed at
        # steps 0 and 2, giving slope (11 - 6) / 2, then left out of the full step at 4.
        transformer = Transformer()
        both = ("cond", "uncond")
        with reuse_blocks(transformer, FixedInterval(2, forecast=True)) as reuse:
            run_step(transformer, reuse, 0, image=1.0, branches=both)
            run_step(transformer, reuse, 2, image=2.0, branches=both)
            run_step(transformer, reuse, 4, image=3.0, branches=("cond",))
            reuse.begin_step(5, both)
            reuse.branch = "uncond"
            # Forecast from step 2: 11 + 3 * 2.5, plus the last block and the zero input.
            self.assertTrue(mx.array_equal(transformer(0.0), tokens(28.5)).item())
            reuse.branch = "cond"
            # Cond was computed at steps 2 and 4: 16 + 1 * 2.5 + 10.
            self.assertTrue(mx.array_equal(transformer(0.0), tokens(28.5)).item())

    def test_branch_without_cache_forces_full_step(self) -> None:
        transformer = Transformer()
        with reuse_blocks(transformer, FixedInterval(2)) as reuse:
            run_step(transformer, reuse, 0, branches=("cond",))
            # Step 1 would reuse, but uncond has never been computed.
            output = run_step(transformer, reuse, 1, branches=("cond", "uncond"))
            self.assertFalse(reuse.skip)
            self.assertTrue(mx.array_equal(output, tokens(17.0)).item())
            run_step(transformer, reuse, 2, branches=("cond", "uncond"))
            self.assertFalse(reuse.skip)
            run_step(transformer, reuse, 3, branches=("cond", "uncond"))
            self.assertTrue(reuse.skip)
        self.assertEqual(reuse.full_steps, 3)

    def test_blocks_are_restored(self) -> None:
        transformer = Transformer()
        single_blocks = transformer.single_transformer_blocks
        with reuse_blocks(transformer, FixedInterval(1)):
            self.assertIsNot(transformer.single_transformer_blocks, single_blocks)
        self.assertIs(transformer.single_transformer_blocks, single_blocks)


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
