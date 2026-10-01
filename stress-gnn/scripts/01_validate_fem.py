"""Mesh-convergence study: quantify the discretisation error the surrogate has to undo.

    python scripts/01_validate_fem.py

Solves a few designs at a sequence of element sizes and reports compliance and peak
von Mises against the finest mesh.  Compliance (an energy) converges quickly and from
below; the peak stress converges slowly, because it lives at a curved boundary where
constant-strain triangles are at their worst.  That difference is the entire premise
of the repo, so it is measured rather than asserted.

Correctness of the solver itself (patch test, energy balance, linearity) is checked in
`tests/test_fem.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from stressgnn.fem import solve
from stressgnn.geometry import sample_designs
from stressgnn.mesh import build_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/fem_convergence.json"))
    ap.add_argument("--n-designs", type=int, default=4)
    ap.add_argument("--sizes", type=float, nargs="*", default=[0.06, 0.045, 0.03, 0.02, 0.014, 0.010, 0.008])
    args = ap.parse_args()

    designs = sample_designs(np.random.default_rng(99), args.n_designs)
    rows = []
    for hi, h in enumerate(args.sizes):
        peaks, nodal_peaks, comps, nodes, times = [], [], [], [], []
        for design in designs:
            t0 = time.perf_counter()
            mesh = build_mesh(design, h)
            sol = solve(mesh, design)
            times.append(time.perf_counter() - t0)
            peaks.append(sol.max_vm)  # element-wise maximum: the reported QoI
            nodal_peaks.append(float(sol.vm_node.max()))  # after averaging to nodes
            comps.append(sol.compliance)
            nodes.append(mesh.n_nodes)
        rows.append(
            {
                "h": h,
                "mean_nodes": float(np.mean(nodes)),
                "mean_seconds": float(np.mean(times)),
                "peak_vm": [float(p) for p in peaks],
                "peak_vm_nodal": [float(p) for p in nodal_peaks],
                "compliance": [float(c) for c in comps],
            }
        )
        print(f"  h={h:.3f}  {np.mean(nodes):7.0f} nodes  {np.mean(times):5.2f} s", flush=True)

    ref = rows[-1]
    hdr = (
        f"{'h':>7} {'nodes':>8} {'time(s)':>8} {'peak err %':>11} "
        f"{'compliance err %':>18} {'nodal vs elem %':>17}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        pe = 100 * np.mean([(a - b) / b for a, b in zip(r["peak_vm"], ref["peak_vm"])])
        ce = 100 * np.mean([(a - b) / b for a, b in zip(r["compliance"], ref["compliance"])])
        # How much of the peak is lost purely by averaging element stresses to nodes,
        # independently of mesh size -- the fixed part of the "interpolation floor".
        ne = 100 * np.mean([(a - b) / b for a, b in zip(r["peak_vm_nodal"], r["peak_vm"])])
        r["peak_err_pct"] = float(pe)
        r["compliance_err_pct"] = float(ce)
        r["nodal_vs_element_pct"] = float(ne)
        print(
            f"{r['h']:>7.3f} {r['mean_nodes']:>8.0f} {r['mean_seconds']:>8.2f} {pe:>+11.2f} "
            f"{ce:>+18.3f} {ne:>+17.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reference_h": ref["h"], "rows": rows}, indent=2))
    print(f"\nreference: h = {ref['h']} ({ref['mean_nodes']:.0f} nodes)\nwrote {args.out}")


if __name__ == "__main__":
    main()
