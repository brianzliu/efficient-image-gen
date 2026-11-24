import unittest
from unittest.mock import patch

from feature_reuse.profiling import Profiler, profiled


class ProfilerTest(unittest.TestCase):
    def test_nested_reset_preserves_parent_peak_and_exclusive_time(self):
        profiler = Profiler()
        snapshots = [(100, 100), (115, 140), (115, 190), (130, 160)]
        with (
            patch("feature_reuse.profiling.memory_snapshot", side_effect=snapshots),
            patch("feature_reuse.profiling.perf_counter", side_effect=[0, 2, 5, 10]),
            patch("feature_reuse.profiling.mx.reset_peak_memory"),
            patch("feature_reuse.profiling.mx.eval"),
        ):
            output = profiler.run("parent", None,
                                  lambda: profiler.run("child", None, lambda: 42, (), {}), (), {})
        self.assertEqual(output, 42)
        self.assertEqual(dict(profiler.calls), {"parent": 1, "parent/child": 1})
        self.assertEqual(dict(profiler.seconds), {"parent": 7, "parent/child": 3})
        self.assertEqual(dict(profiler.transient), {"parent": 90, "parent/child": 75})

    def test_restores_hook_and_stack_after_exception(self):
        class Target:
            @staticmethod
            def fail():
                raise RuntimeError("broken target")

        original = Target.__dict__["fail"]
        with patch("feature_reuse.profiling.mx.eval"), self.assertRaisesRegex(RuntimeError, "broken target"):
            with profiled([(Target, "fail", "target", None)]) as profiler:
                Target.fail()
        self.assertIs(Target.__dict__["fail"], original)
        self.assertEqual(profiler.path, [])
