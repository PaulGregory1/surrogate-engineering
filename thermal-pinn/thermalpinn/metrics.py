"""What counts as a good answer, and what nobody is allowed to be scored on.

Two of the repo's honesty rules live in this file rather than in prose.

*Nothing is scored on what the measurement destroyed.*  `naive.cutoff_frequency` says a
component above about 7 Hz never reached the thermocouple at 2 mm, so every error measure
on q(t) is computed after the same low-pass filter has been applied to the truth **and** to
the estimate.  A method that reproduced the out-of-band part of the flux would be reading
noise, not the disc, so rewarding it would be exactly backwards.  What was lost is reported
separately, as `out_of_band_rms`, because it is a property of the experiment and belongs in
the results rather than in a footnote.

*The headline number is the one the problem was posed for.*  The question is the peak
temperature of a surface that cannot be instrumented, so `surface_peak_error` is a signed
percentage of the true temperature *rise* -- not of the absolute temperature, which would
divide by 900 K of offset and flatter everyone by a factor of three.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass(y: np.ndarray, dt: float, f_c: float, *, order: int = 4) -> np.ndarray:
    """Zero-phase low pass at `f_c`, applied identically to truth and estimate.

    Butterworth run forwards and backwards (`filtfilt`), so there is no phase lag to be
    mistaken for a timing error, and the padding extends the edges with their own value:
    the flux starts at several MW/m^2, and the odd extension `filtfilt` defaults to would
    invent a discontinuity there that the signal does not have.

    The filter is a convention of the metric, not a physical model.  What makes it fair is
    that both sides of every comparison go through it; `tests/test_classical.py` checks
    that swapping it for a Gaussian of matched bandwidth moves the scores by well under
    their own spread.
    """
    y = np.asarray(y, dtype=float)
    nyquist = 0.5 / dt
    if f_c >= nyquist:
        return y.copy()
    b, a = butter(order, f_c / nyquist)
    return filtfilt(b, a, y, padtype="constant")


def band_error(q_est: np.ndarray, q_true: np.ndarray, dt: float, f_c: float) -> dict:
    """Error on the flux history, inside the recoverable band and outside it.

    `rmse_pct` is relative to the RMS of the band-limited truth, so it is a percentage of
    the signal any method could in principle have recovered, not of the peak.
    """
    q_est, q_true = np.asarray(q_est, dtype=float), np.asarray(q_true, dtype=float)
    est_band, true_band = lowpass(q_est, dt, f_c), lowpass(q_true, dt, f_c)
    residual = est_band - true_band
    scale = float(np.sqrt(np.mean(true_band**2)))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rmse_pct": float(100.0 * np.sqrt(np.mean(residual**2)) / scale),
        "max_abs": float(np.max(np.abs(residual))),
        "band_rms": scale,
        # The part of the truth the depth of the sensor removed: unrecoverable by anyone,
        # reported so that the band-limited scores above are read next to what they omit.
        "out_of_band_rms": float(np.sqrt(np.mean((q_true - true_band) ** 2))),
        "f_c": float(f_c),
    }


def surface_peak_error(T_est: np.ndarray, T_true: np.ndarray, T0: float) -> float:
    """Signed error on the peak surface temperature, per cent of the true rise."""
    T_est, T_true = np.asarray(T_est, dtype=float), np.asarray(T_true, dtype=float)
    return float(100.0 * (T_est.max() - T_true.max()) / (T_true.max() - T0))


def k_curve_error(k_est: np.ndarray, k_true: np.ndarray) -> float:
    """Relative L2 error between two conductivity curves, per cent.

    Sampled on whatever temperature grid the caller chose, which must be the range the
    event actually visited: outside it the data say nothing about k, and scoring there
    would measure the extrapolation habits of each parametrisation instead of the fit.
    """
    k_est, k_true = np.asarray(k_est, dtype=float), np.asarray(k_true, dtype=float)
    return float(100.0 * np.linalg.norm(k_est - k_true) / np.linalg.norm(k_true))
