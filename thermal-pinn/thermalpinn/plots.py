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
