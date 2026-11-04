"""Offline image metrics and explicit energy-window integration."""
import numpy as np
from skimage.metrics import structural_similarity


def similarity(reference, candidate):
    a, b = np.asarray(reference.convert("RGB")), np.asarray(candidate.convert("RGB"))
    if a.shape != b.shape:
        raise ValueError("Reference and candidate sizes differ")
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return {
        "psnr_db": None if mse == 0 else float(10 * np.log10(255**2 / mse)),
        "identical_pixels": mse == 0,
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
    }


def integrate_energy(samples, start, end, max_gap=2.0):
    """Integrate (Unix seconds, watts), requiring coverage and bounded gaps.

    Channel scope belongs in the supplied telemetry metadata. Never extrapolate.
    """
    values = np.asarray(samples, dtype=float)
    if end <= start or values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("Invalid energy window or samples")
    t, power = values.T
    if not np.isfinite(values).all() or np.any(np.diff(t) <= 0) or np.any(power < 0):
        raise ValueError("Samples must be finite, increasing, and nonnegative")
    if t[0] > start or t[-1] < end:
        raise ValueError("Power samples do not cover the generation window")
    relevant = (t[:-1] < end) & (t[1:] > start)
    if np.any(np.diff(t)[relevant] > max_gap):
        raise ValueError("Power sampling gap exceeds limit")
    times = np.concatenate(([start], t[(t > start) & (t < end)], [end]))
    return float(np.trapezoid(np.interp(times, t, power), times))
