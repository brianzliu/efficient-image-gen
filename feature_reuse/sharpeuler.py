"""Offline sharpness calibration and quantile sampling from Sharpen Your Flow."""

import numpy as np


def sigma_grid(budget: int, sharpness: list[float]) -> list[float]:
    """Generate a decreasing sigma grid from a profile on uniform calibration midpoints."""
    try:
        profile = np.asarray(sharpness, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("SharpEuler calibration must contain numeric values") from error
    if budget < 2 or profile.ndim != 1 or profile.size < 1 or not np.all(np.isfinite(profile) & (profile >= 0)):
        raise ValueError("SharpEuler needs at least two steps and a finite nonnegative calibration profile")
    # Gamma = 1/2 and Gaussian smoothing with a bandwidth of one calibration step.
    weights = np.sqrt(profile + 1e-8)
    offsets = np.arange(-3, 4)
    kernel = np.exp(-0.5 * offsets**2)
    weights = np.convolve(np.pad(weights, 3, mode="reflect"), kernel / kernel.sum(), mode="valid")
    cdf = np.r_[0.0, np.cumsum(weights) / weights.sum()]
    midpoints = np.arange(1, profile.size + 1) / (profile.size + 1)
    quantiles = np.arange(budget) / budget
    grid = np.r_[1 - np.interp(quantiles, cdf, np.r_[0.0, midpoints]), 0.0].astype(np.float32)
    if not np.all(np.diff(grid) < 0):
        raise ValueError("SharpEuler schedule contains duplicate timesteps at float32 precision")
    return grid.tolist()
