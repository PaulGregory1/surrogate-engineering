"""Figures.  Each one has to be readable without the surrounding text.

The rule the repo holds itself to is that a figure states a measurement, so nothing here
computes physics: every function takes results that were produced and stored elsewhere and
only decides how to show them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .material import Material, to_celsius
from .naive import attenuation, cutoff_frequency
from .scenario import Scenario

TRUTH = "0.25"
DATA = "#1f4e9c"
DANGER = "#c0392b"
BAND = "#2e7d32"
#: 1, 2, 4, 6 mm -- shallow to deep.
DEPTH_COLOURS = ("#f4a259", "#1f4e9c", "#7b4ea3", "#4d4d4d")


def illposedness(
    traces: dict,
    measured: dict,
    scenario: Scenario,
    material: Material,
    path: Path,
    *,
    depths: tuple[float, ...] = (1.0e-3, 2.0e-3, 4.0e-3, 6.0e-3),
    eta: float = 0.1,
) -> None:
    """The three facts that justify everything after them, left to right.

    A tame measurement; the same measurement marched to the surface with nothing holding
    it back; and the attenuation that decided in advance which part of the flux history
    survives to be recovered at all.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0), constrained_layout=True)

    # ---------------------------------------------------------------------------- panel 1
    ax = axes[0]
    ax.plot(traces["t_truth"], to_celsius(traces["T_surface"]), color=TRUTH, lw=1.6, label="surface (unknown)")
    ax.plot(traces["t_sensor"], to_celsius(traces["sensor_noisy"]), color=DATA, lw=0.8, label="thermocouple at 2 mm")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [degC]")
    ax.set_title("what is measured", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.text(
        0.32,
        0.78,
        f"{scenario.f_sample:.0f} Hz, sigma = {scenario.sigma_noise} K\n"
        f"peak {to_celsius(traces['T_surface'].max()):.0f} degC at the surface,\n"
        f"{to_celsius(traces['T_sensor'].max()):.0f} degC at the sensor,\n"
        f"{to_celsius(traces['T_surface'].max()) - to_celsius(traces['T_sensor'].max()):.0f} K apart",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    # The noise is invisible at full scale, and its invisibility is the point: nothing in
    # the measurement warns that inverting it is dangerous.
    inset = ax.inset_axes((0.45, 0.13, 0.5, 0.3))
    zoom = (traces["t_sensor"] >= 3.0) & (traces["t_sensor"] <= 3.3)
    inset.plot(traces["t_sensor"][zoom], to_celsius(traces["sensor_clean"][zoom]), color=TRUTH, lw=1.2)
    inset.plot(traces["t_sensor"][zoom], to_celsius(traces["sensor_noisy"][zoom]), ".", color=DATA, ms=3)
    inset.set_title("0.3 s of it", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(alpha=0.25)

    # ---------------------------------------------------------------------------- panel 2
    ax = axes[1]
    # Worst first: the divergent cloud has to sit behind the two traces that have to stay
    # readable through it, and the truth has to sit behind the march that reproduces it.
    fast, slow = max(traces["marches"]), min(traces["marches"])
    excursions = {f_s: np.abs(traces["marches"][f_s].T_surface - scenario.T0) for f_s in traces["marches"]}
    ax.plot(traces["marches"][fast].t, np.maximum(excursions[fast], 1e-2), color=DANGER, lw=0.6, alpha=0.4)
    rise = np.abs(traces["T_surface"] - scenario.T0)
    ax.plot(traces["t_truth"], np.maximum(rise, 1e-2), color=TRUTH, lw=2.4)
    ax.plot(traces["marches"][slow].t, np.maximum(excursions[slow], 1e-2), color=DATA, lw=0.9)
    # Proxy handles: the divergent trace is drawn faint so the others survive it, which
    # would make its legend entry unreadable at the same alpha.
    ax.legend(
        handles=[
            Line2D([], [], color=colour, lw=1.6, label=f"marched from {f_s:.0f} Hz data  (max {excursions[f_s].max():.2e} K)")
            for f_s, colour in ((fast, DANGER), (slow, DATA))
        ]
        + [Line2D([], [], color=TRUTH, lw=2.4, label="the answer being looked for")],
        fontsize=8,
        loc="upper right",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e7)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("| reconstructed surface rise | [K]")
    ax.set_title("what a space march returns", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.text(
        0.03,
        0.10,
        "same code, same noise, same disc.\nonly the sampling rate differs --\nand the faster one is the better sensor",
        transform=ax.transAxes,
        fontsize=8,
    )

    # ---------------------------------------------------------------------------- panel 3
    ax = axes[2]
    f = np.logspace(-1.0, 2.0, 400)
    by_depth: dict[float, list[dict]] = {}
    for row in measured["rows"]:
        by_depth.setdefault(row["depth"], []).append(row)

    f_c_sensor = cutoff_frequency(scenario.x_s, material.alpha0, eta)
    ax.axvspan(f[0], f_c_sensor, color=BAND, alpha=0.10, lw=0)
    for d, colour in zip(depths, DEPTH_COLOURS):
        inside = d < scenario.L
        label = f"{1e3 * d:.0f} mm" if inside else f"{1e3 * d:.0f} mm (past the mid-plane)"
        ax.plot(f, attenuation(f, d, material.alpha0), color=colour, lw=1.4, ls="-" if inside else "--", label=label)
        rows = by_depth.get(d)
        if rows:
            ax.plot([r["f"] for r in rows], [r["measured"] for r in rows], "o", color=colour, ms=3.2, mfc="white")
    ax.axhline(eta, color="0.4", ls=":", lw=1.0)
    ax.axvline(scenario.abs_freq, color=DANGER, ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(1e-6, 1.6)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude at the sensor / at the surface")
    ax.set_title("what the depth has already removed", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, title="lines: closed form   circles: measured", title_fontsize=8, loc="lower left")

    abs_reaching = float(attenuation(scenario.abs_freq, scenario.x_s, material.alpha0))
    ax.annotate(
        f"ABS ripple, {scenario.abs_freq:.0f} Hz:\n{1e2 * abs_reaching:.0f} % of it reaches 2 mm",
        xy=(scenario.abs_freq, abs_reaching),
        xytext=(9.5, 0.45),
        fontsize=8,
        color=DANGER,
        arrowprops=dict(arrowstyle="->", color=DANGER, lw=0.8),
    )
    ax.text(95.0, 1.25 * eta, f"gain 1/eta = {1.0 / eta:.0f}", fontsize=7, color="0.4", ha="right")
    ax.text(0.115, 1.2, f"recoverable at 2 mm: f < {f_c_sensor:.1f} Hz", fontsize=8, color=BAND, ha="left")

    fig.savefig(path, dpi=130)
    plt.close(fig)


#: One colour per method, kept identical across every act so the three act figures can be
#: read side by side.  The PINN is the only one drawn with a spread, because it is the only
#: one that has one.
METHOD_COLOURS = {
    "beck": "#f4a259",
    "tikhonov": "#1f4e9c",
    "alifanov": "#7b4ea3",
    "pinn": "#c0392b",
}


def act1(results: dict, scenario: Scenario, path: Path, *, f_c: float) -> None:
    """The draw, drawn: four methods on one flux history and one surface temperature.

    Left is what was asked for -- the flux, over the braking phase, against the part of the
    truth that survived the depth of the sensor.  The pale line behind it is the raw truth
    including the 8 Hz ripple, and the gap between the two is the information the
    measurement destroyed before any method ran: nobody is scored on it, and nobody should
    appear to be recovering it.

    Right is what the question was actually about -- a surface nobody can instrument, and
    the peak every method has to land on.  The PINN carries a band across its five seeds and
    the three classical methods do not.  That is not a courtesy: they are deterministic given
    the data, and the asymmetry is part of what the comparison is measuring.
    """
    t_q = np.asarray(results["t_q"])
    t_s = np.asarray(results["t_surface"])
    seeds_q = np.asarray(results["pinn_q_seeds"])
    seeds_T = np.asarray(results["pinn_surface_seeds"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)

    # ---------------------------------------------------------------------------- the flux
    ax = axes[0]
    window = t_q <= min(scenario.t_brake + 1.0, scenario.t_end)
    ax.plot(t_q[window], 1e-6 * np.asarray(results["q_true"])[window], color="0.78", lw=1.0,
            label="truth, as it was")
    ax.plot(t_q[window], 1e-6 * np.asarray(results["q_band"])[window], color=TRUTH, lw=2.0,
            label=f"truth below {f_c:.1f} Hz -- what is scored")
    ax.fill_between(t_q[window], 1e-6 * seeds_q.min(axis=0)[window], 1e-6 * seeds_q.max(axis=0)[window],
                    color=METHOD_COLOURS["pinn"], alpha=0.18, lw=0)
    for name, q in results["q_est"].items():
        ax.plot(t_q[window], 1e-6 * np.asarray(q)[window], color=METHOD_COLOURS[name], lw=1.1,
                alpha=0.9, label=name)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("q [MW/m^2]")
    ax.set_title("the surface heat flux, recovered", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.06,
            "shaded: the five PINN seeds\n"
            f"out of band, unrecoverable by anyone: {1e-6 * results['out_of_band_rms']:.2f} MW/m^2 rms",
            transform=ax.transAxes, fontsize=8)

    # ------------------------------------------------------------------------- the surface
    ax = axes[1]
    T_true = np.asarray(results["T_surface_true"])
    ax.plot(t_s, to_celsius(T_true), color=TRUTH, lw=2.0, label="truth (never measured)")
    ax.fill_between(t_s, to_celsius(seeds_T.min(axis=0)), to_celsius(seeds_T.max(axis=0)),
                    color=METHOD_COLOURS["pinn"], alpha=0.18, lw=0)
    for name, T in results["T_surface_est"].items():
        ax.plot(t_s, to_celsius(np.asarray(T)), color=METHOD_COLOURS[name], lw=1.1, alpha=0.9, label=name)
    ax.plot(np.asarray(results["t_sensor"]), to_celsius(np.asarray(results["T_sensor_true"])),
            color=DATA, lw=0.9, ls="--", label=f"the thermocouple, {1e3 * scenario.x_s:.0f} mm down")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [degC]")
    ax.set_title("the temperature nobody can measure", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # The peak is the headline number, so it gets an inset rather than a caption.
    i_peak = int(np.argmax(T_true))
    span = max(2, int(round(0.4 / (t_s[1] - t_s[0]))))
    lo, hi = max(0, i_peak - span), min(t_s.size, i_peak + span)
    inset = ax.inset_axes((0.44, 0.13, 0.52, 0.36))
    inset.fill_between(t_s[lo:hi], to_celsius(seeds_T.min(axis=0)[lo:hi]), to_celsius(seeds_T.max(axis=0)[lo:hi]),
                       color=METHOD_COLOURS["pinn"], alpha=0.18, lw=0)
    inset.plot(t_s[lo:hi], to_celsius(T_true[lo:hi]), color=TRUTH, lw=2.0)
    for name, T in results["T_surface_est"].items():
        inset.plot(t_s[lo:hi], to_celsius(np.asarray(T)[lo:hi]), color=METHOD_COLOURS[name], lw=1.1)
    inset.set_title(f"the peak: {to_celsius(T_true.max()):.0f} degC", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.25)

    fig.savefig(path, dpi=130)
    plt.close(fig)
