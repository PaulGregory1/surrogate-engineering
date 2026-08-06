"""What counts as a good answer, and what nobody is allowed to be scored on.

Two of the repo's honesty rules live in this file rather than in prose.

*Nothing is scored on what the measurement destroyed.*  `naive.cutoff_frequency` says a
component above about 7 Hz never reached the thermocouple at 2 mm, so every error measure
on q(t) is computed after the same low-pass filter has been applied to the truth **and** to
the estimate.  A method that reproduced the out-of-band part of the flux would be reading
noise, not the disc, so rewarding it would be exactly backwards.  What was lost is reported
separately, as `out_of_band_rms`, because it is a property of the experiment and belongs in
the results rather than in a footnote.

The filter is a convention of the metric, not a physical model, and what makes it fair is
that both sides of every comparison go through it.  `tests/test_classical.py` holds it to two
things: adding a component far outside the band must not move any score, and a filter with a
quite different roll-off must not change the verdict.  Not the same as leaving the numbers
alone -- a Gaussian matched only at the -3 dB point moves them by about a factor of two,
because matching one point does not make it the same band.  What survives is the comparison
between methods, which is all that is ever claimed.

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

    Butterworth run forwards and backwards, so there is no phase lag to be mistaken for a
    timing error.  Two details of the padding are load-bearing rather than cosmetic, and
    both were found by measuring instead of by taste:

    *Odd extension.*  Reflecting through the end point keeps the padded signal continuous
    in value **and** slope, so the filter has nothing to ring on.  Extending the edges with
    their own value instead -- which looks like the safe choice for a flux that starts at
    several MW/m^2 -- injects a slope discontinuity, and the filter turns that into an
    in-band artefact 10^4 times larger at the edges than anything it does in the interior.

    *A pad matched to the cutoff.*  The filter's transient lasts about 1/f_c, so the pad is
    four of those rather than the fifteen samples `filtfilt` defaults to.

    Together these leave the peak at t = 0 untouched to four figures and reject a component
    at five times the cutoff by better than 10^-3 everywhere, edges included.  Band-limiting
    a finite record can never be exact at its ends; this is how close it gets.
    """
    y = np.asarray(y, dtype=float)
    nyquist = 0.5 / dt
    if f_c >= nyquist:
        return y.copy()
    b, a = butter(order, f_c / nyquist)
    padlen = min(y.size - 1, max(3 * max(len(a), len(b)), int(round(4.0 / (f_c * dt)))))
    return filtfilt(b, a, y, padtype="odd", padlen=padlen)


def band_error(q_est: np.ndarray, q_true: np.ndarray, dt: float, f_c: float, *, edge_margin: int | None = None) -> dict:
    """Error on the flux history, inside the recoverable band and outside it.

    Scored on the interior, excluding one filter transient (1 / f_c, about 0.15 s here) at
    each end, and the full-record number is returned beside it.  The reason is measured, not
    assumed: with the margin included, between 40 and 60 % of the reported error comes from
    those 15 % of samples, for every method.

    Two different things make the ends bad, and only one of them should be scored.  Band
    limiting a *finite* record cannot suppress out-of-band content near its boundary -- the
    padding has to guess how the signal continues, and a component at five times the cutoff
    leaks in at about a tenth of its amplitude there, against 10^-3 in the interior.  That is
    a property of the metric.  The other is real: no measurement after the record ends
    constrains the flux at the end of it, and Beck declines to answer for its final window
    at all.  Since the two are entangled at the edges, the primary number is taken where
    neither applies, and `edge_share_pct` says how much was set aside.

    `rmse_pct` is relative to the RMS of the band-limited truth over the same interior, so it
    is a percentage of the signal a method could in principle have recovered.
    """
    q_est, q_true = np.asarray(q_est, dtype=float), np.asarray(q_true, dtype=float)
    est_band, true_band = lowpass(q_est, dt, f_c), lowpass(q_true, dt, f_c)
    residual = est_band - true_band

    margin = int(round(1.0 / (f_c * dt))) if edge_margin is None else int(edge_margin)
    inside = slice(margin, residual.size - margin) if 2 * margin < residual.size - 2 else slice(None)
    core, core_true = residual[inside], true_band[inside]

    rmse = float(np.sqrt(np.mean(core**2)))
    full = float(np.sqrt(np.mean(residual**2)))
    return {
        "rmse": rmse,
        "rmse_pct": float(100.0 * rmse / np.sqrt(np.mean(core_true**2))),
        # Same error against the peak of the band-limited truth.  Worth carrying both: the
        # RMS denominator shrinks when a scenario spends most of its record cooling, so it
        # is the fairer within-scenario score and the harder one to read across scenarios.
        "rmse_pct_of_peak": float(100.0 * rmse / np.max(np.abs(core_true))),
        "max_abs": float(np.max(np.abs(core))),
        "rmse_full_record": full,
        "edge_margin": margin,
        "edge_share_pct": float(100.0 * (1.0 - (rmse**2 * core.size) / (full**2 * residual.size))) if full > 0 else 0.0,
        "band_rms": float(np.sqrt(np.mean(core_true**2))),
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
