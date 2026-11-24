import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Written by scripts/calibrate-racfg.py for Klein, 50 Euler steps and guidance 3.5.
CALIBRATION = Path(__file__).resolve().parent / "racfg_klein.json"


def guided_risk(r_cond: float, r_uncond: float, alignment: float, scale: float) -> float:
    """Combine the two branches' reuse errors the way CFG combines their outputs (RA-CFGCache, eq. 7).

    CFG outputs (1 - s) uncond + s cond, so errors that point the same way partly cancel when s > 1.
    """
    squared = ((1 - scale) * r_uncond) ** 2 + (scale * r_cond) ** 2
    squared += 2 * scale * (1 - scale) * alignment * r_uncond * r_cond
    return max(squared, 0.0) ** 0.5


def gain(step: int, steps: int, params) -> float:
    """How much an error at this step grows by the final latent: lambda ((T - k) / T)^alpha + beta."""
    lam, alpha, beta = params
    return lam * ((steps - step) / steps) ** alpha + beta


def fit_gain(probes, ratios, steps: int):
    """Least-squares fit of the gain curve with lambda, beta >= 0, searching alpha on a grid."""
    probes, ratios = np.asarray(probes, float), np.asarray(ratios, float)
    best = None
    for alpha in np.linspace(0.1, 6.0, 119):
        basis = ((steps - probes) / steps) ** alpha
        design = np.stack([basis, np.ones_like(basis)], axis=1)
        lam, beta = np.maximum(np.linalg.lstsq(design, ratios, rcond=None)[0], 0.0)
        error = float(np.sum((lam * basis + beta - ratios) ** 2))
        if best is None or error < best[0]:
            best = (error, (float(lam), float(alpha), float(beta)))
    return best[1]


def risk_schedule(calibration: dict, threshold: float, scale: float) -> list[int]:
    """Steps that run the full model; on the others both CFG branches reuse cached results.

    calibration holds per-step magnitude ratios for each branch, the branch alignment matrix and the
    fitted gain. Every input is fixed after calibration, so the schedule is the same for every prompt.
    """
    ratios, alignment = calibration["magnitude_ratio"], calibration["alignment"]
    steps = len(ratios["cond"])
    refresh, anchor, accumulated = [0], 0, 0.0
    for step in range(1, steps):
        # MagCache-style proxy: how far the residual's magnitude has drifted since the anchor.
        drift = {
            branch: abs(1 - float(np.prod(ratios[branch][anchor + 1 : step + 1]))) for branch in ("cond", "uncond")
        }
        risk = guided_risk(drift["cond"], drift["uncond"], alignment[anchor][step], scale)
        increment = max(0.0, gain(step, steps, calibration["gain"]) * risk)
        if accumulated + increment > threshold:
            refresh.append(step)
            anchor, accumulated = step, 0.0
        else:
            accumulated += increment
    return refresh


@dataclass(frozen=True)
class RACFGCache:
    """RA-CFGCache (Liu et al., 2026) with a MagCache-style proxy, as the paper uses for FLUX.

    Both CFG branches refresh together. forecast=True extrapolates between refreshes like taylor-N
    instead of reusing the last result.
    """

    threshold: float
    scale: float
    forecast: bool = False
    order: int = 1
    calibration_path: Path = CALIBRATION
    refresh: frozenset = field(init=False, compare=False)

    def __post_init__(self) -> None:
        calibration = json.loads(Path(self.calibration_path).read_text())
        object.__setattr__(self, "refresh", frozenset(risk_schedule(calibration, self.threshold, self.scale)))

    def runs_blocks(self, step: int, predicted: int) -> bool:
        return step in self.refresh

    def checks(self, predicted: int) -> bool:
        return False
