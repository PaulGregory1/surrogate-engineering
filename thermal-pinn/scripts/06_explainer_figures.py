"""Establish the two limits every later method lives inside, and draw them.

    python scripts/06_explainer_figures.py            # ~2 min
    python scripts/06_explainer_figures.py --quick     # ~15 s, on the coarse grid

This is milestone M2, and it runs before any inverse method is written on purpose: the
two facts below decide what a good result even looks like, and a repo that measures them
afterwards has no way to tell regularisation from luck.

  (a) marching in space diverges.  On noise-free data the march recovers the surface peak
      to better than 0.1 %, so it is the problem and not the code; with the thermocouple's
      real 0.5 K it passes 10^5 K once the data are sampled at 1 kHz.
  (b) the depth of the sensor has already destroyed part of the flux history.  The
      measured attenuation has to agree with alpha ln(1/eta)^2 / (pi d^2) to within 20 %,
      which is what licenses every later metric being computed inside that band.

Writes results/illposedness.json and figs/illposed.png.  Numbering follows the repo
layout rather than the running order -- this is where figs/illposed.png lives, and the
"what the PINN optimises" schematic joins it at M4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermalpinn.material import LAW_A, to_celsius
from thermalpinn.naive import ETA, march_divergence, march_traces, measured_attenuation
from thermalpinn.plots import illposedness
from thermalpinn.scenario import SHOWCASE
from thermalpinn.solver import INVERSE_GRID, TRUTH_GRID, forward

#: The march has to reach this to count as divergent, and it is asked to do so at a
#: sampling rate a real DAQ would offer rather than at one chosen to make it explode.
DIVERGENCE_K = 1.0e5
DIVERGENCE_RATE = 1000.0
DIVERGENCE_N_MARCH = 20
#: Control: below this rate the *clean* march must stay this accurate on the surface peak,
#: or the divergence above means nothing.
CONTROL_RATE = 2000.0
CONTROL_TOL_PCT = 0.1
#: How far the measured cutoff may sit from the closed form.
CUTOFF_TOL = 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/illposedness.json"))
    ap.add_argument("--fig", type=Path, default=Path("figs/illposed.png"))
    ap.add_argument("--quick", action="store_true", help="coarse grid, fewer sampling rates")
    args = ap.parse_args()

    scenario, material = SHOWCASE, LAW_A
    grid = INVERSE_GRID if args.quick else TRUTH_GRID
    rates = (100.0, 200.0, 500.0, 1000.0, 2000.0) if args.quick else None

    t0 = time.perf_counter()
    # One truth run feeds both the divergence table and the traces the figure draws.
    truth = forward(scenario, material, scenario.q_true, grid=grid)
    kw = {} if rates is None else {"rates": rates}
    divergence = march_divergence(scenario, material, grid=grid, truth=truth, **kw)
    attenuation = measured_attenuation(material=material, grid=grid, eta=ETA)
    traces = march_traces(scenario, material, grid=grid, truth=truth, n_march=DIVERGENCE_N_MARCH)

    print("(a) space marching, on the real braking event")
    print(f"    truth: surface peak {to_celsius(divergence['T_surface_max']):.1f} degC, "
          f"sensor peak {to_celsius(divergence['T_sensor_max']):.1f} degC")
    print(f"\n{'f_s [Hz]':>9} {'steps':>6} | {'clean err [K]':>14} {'clean Tmax %':>13} |"
          f" {'noisy err [K]':>14} {'noisy Tmax %':>16}")
    for r in divergence["rows"]:
        print(
            f"{r['f_sample']:>9.0f} {r['n_march']:>6d} | {r['clean']['max_abs_error_K']:>14.3e} "
            f"{r['clean']['T_surface_max_error_pct']:>+13.3f} | {r['noisy']['max_abs_error_K']:>14.3e} "
            f"{r['noisy']['T_surface_max_error_pct']:>+16.2f}"
        )

    diverged = [
        r for r in divergence["rows"]
        if r["f_sample"] == DIVERGENCE_RATE and r["n_march"] == DIVERGENCE_N_MARCH
    ]
    control = [
        r for r in divergence["rows"]
        if r["f_sample"] <= CONTROL_RATE and abs(r["clean"]["T_surface_max_error_pct"]) >= CONTROL_TOL_PCT
    ]
    pass_a = bool(diverged) and diverged[0]["noisy"]["excursion_K"] > DIVERGENCE_K and not control

    print("\n(b) depth attenuation, measured against exp(-d sqrt(pi f / alpha))")
    print(f"{'depth':>7} {'f_c measured':>13} {'f_c analytic':>13} {'ratio':>7} {'penetration':>12}"
          f" {'slab behind':>12}  half-space valid")
    cutoff_rows = []
    for p in attenuation["per_depth"]:
        # Declared before the numbers are looked at: the closed form assumes the harmonic
        # dies before it reaches the mid-plane, so it is only binding where one penetration
        # depth at the cutoff fits behind the probe.  The rest is reported, not scored.
        valid = p["penetration_at_f_c_m"] < p["slab_behind_probe_m"]
        p["half_space_valid"] = bool(valid)
        cutoff_rows.append(p)
        print(
            f"{1e3 * p['depth']:>6.1f}mm {p['f_c_measured']:>13.3f} {p['f_c_analytic']:>13.3f} "
            f"{p['ratio']:>7.3f} {1e3 * p['penetration_at_f_c_m']:>11.2f}mm "
            f"{1e3 * p['slab_behind_probe_m']:>11.1f}mm  {'yes' if valid else 'no'}"
        )
    scored = [p for p in cutoff_rows if p["half_space_valid"]]
    pass_b = bool(scored) and all(abs(p["ratio"] - 1.0) <= CUTOFF_TOL for p in scored)

    verdict = {
        "space_march_diverges": {
            "pass": pass_a,
            "threshold_K": DIVERGENCE_K,
            "at_rate_Hz": DIVERGENCE_RATE,
            "n_march": DIVERGENCE_N_MARCH,
            "excursion_K": diverged[0]["noisy"]["excursion_K"] if diverged else None,
            "clean_control_tol_pct": CONTROL_TOL_PCT,
            "clean_control_failures": len(control),
        },
        "cutoff_frequency": {
            "pass": pass_b,
            "tol": CUTOFF_TOL,
            "eta": ETA,
            "worst_ratio": max((abs(p["ratio"] - 1.0) for p in scored), default=float("nan")),
            "n_scored": len(scored),
        },
    }
    verdict["all_pass"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))

    args.fig.parent.mkdir(parents=True, exist_ok=True)
    illposedness(traces, attenuation, scenario, material, args.fig, eta=ETA)
    print(f"\nwrote {args.fig}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "scenario": {
                    "x_s": scenario.x_s,
                    "L": scenario.L,
                    "f_sample": scenario.f_sample,
                    "sigma_noise": scenario.sigma_noise,
                    "abs_freq": scenario.abs_freq,
                },
                "alpha0": material.alpha0,
                "space_march": divergence,
                "attenuation": attenuation,
                "quick": args.quick,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")

    for key, v in verdict.items():
        if key != "all_pass":
            print(f"{'PASS' if v['pass'] else 'FAIL'}  {key}")
    print(f"{'PASS' if verdict['all_pass'] else 'FAIL'}  overall   ({time.perf_counter() - t0:.0f} s)")
    sys.exit(0 if verdict["all_pass"] else 1)


if __name__ == "__main__":
    main()
