import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_reuse.racfg import RACFGCache, fit_gain, gain, guided_risk, risk_schedule


def calibration(steps: int, ratio: float, alignment: float = 0.0, gain_params=(0.0, 1.0, 1.0)) -> dict:
    return dict(
        magnitude_ratio={"cond": [1.0] + [ratio] * (steps - 1), "uncond": [1.0] + [ratio] * (steps - 1)},
        alignment=[[alignment] * steps for _ in range(steps)],
        gain=list(gain_params),
    )


class GuidedRiskTest(unittest.TestCase):
    def test_aligned_errors_combine_like_cfg(self) -> None:
        # Fully aligned equal errors: (1 - s) e + s e = e.
        self.assertAlmostEqual(guided_risk(0.1, 0.1, 1.0, 3.5), 0.1)
        # Opposite errors add up: |(1 - s) - s| e.
        self.assertAlmostEqual(guided_risk(0.1, 0.1, -1.0, 3.5), 0.6)
        # Unaligned errors add in quadrature.
        self.assertAlmostEqual(guided_risk(0.1, 0.1, 0.0, 3.5), 0.1 * np.hypot(2.5, 3.5))


class GainTest(unittest.TestCase):
    def test_fit_recovers_the_curve(self) -> None:
        params = (2.0, 1.5, 0.3)
        probes = list(range(3, 50, 3))
        lam, alpha, beta = fit_gain(probes, [gain(k, 50, params) for k in probes], 50)
        self.assertAlmostEqual(lam, 2.0, places=1)
        self.assertAlmostEqual(alpha, 1.5, places=1)
        self.assertAlmostEqual(beta, 0.3, places=1)

    def test_fit_keeps_gains_nonnegative(self) -> None:
        lam, _, beta = fit_gain([5, 25, 45], [-1.0, -2.0, -3.0], 50)
        self.assertGreaterEqual(lam, 0.0)
        self.assertGreaterEqual(beta, 0.0)


class ScheduleTest(unittest.TestCase):
    def test_no_drift_never_refreshes(self) -> None:
        self.assertEqual(risk_schedule(calibration(10, 1.0), 0.01, 3.5), [0])

    def test_large_drift_refreshes_every_step(self) -> None:
        self.assertEqual(risk_schedule(calibration(10, 2.0), 0.01, 3.5), list(range(10)))

    def test_higher_threshold_refreshes_less(self) -> None:
        counts = [len(risk_schedule(calibration(50, 1.02), t, 3.5)) for t in (0.05, 0.2, 1.0)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(counts[0], counts[-1])

    def test_alignment_lowers_risk(self) -> None:
        aligned = risk_schedule(calibration(50, 1.02, alignment=0.9), 0.2, 3.5)
        unaligned = risk_schedule(calibration(50, 1.02, alignment=0.0), 0.2, 3.5)
        self.assertLess(len(aligned), len(unaligned))


class PolicyTest(unittest.TestCase):
    def test_policy_follows_the_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(calibration(50, 1.02)))
            policy = RACFGCache(0.2, 3.5, calibration_path=path)
            expected = risk_schedule(calibration(50, 1.02), 0.2, 3.5)
            self.assertEqual([s for s in range(50) if policy.runs_blocks(s, 0)], expected)
            self.assertFalse(policy.checks(5))


if __name__ == "__main__":
    unittest.main()
