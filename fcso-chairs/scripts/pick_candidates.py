"""Contact sheet of candidate chairs for splits/opt.txt: a seeded permutation of the test split, rendered
as surface clouds with the fixed camera. The 10 chairs are then picked by eye for variety (thin legs,
armrests, central pedestal) and written to splits/opt.txt with a comment per line.

    .venv/bin/python scripts/pick_candidates.py --seed 0 --n 48
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from fcso import mesh_render as R
from fcso import prior as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "runs", "m2", "candidates"))
    a = ap.parse_args()

    ids = P.read_split("test")
    perm = np.random.default_rng(a.seed).permutation(len(ids))
    cand = [ids[i] for i in perm[: a.n]]
    missing = [c for c in cand if not os.path.exists(os.path.join(P.SURF_DIR, c + ".npz"))]
    if missing:
        sys.exit(f"{len(missing)} candidate surfaces missing; run scripts/fetch_data.py shapes --split test first")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "candidates.txt"), "w") as f:
        f.write("\n".join(f"{k:02d} {c}" for k, c in enumerate(cand)) + "\n")
    for s in range(0, len(cand), a.per_sheet):
        chunk = cand[s: s + a.per_sheet]
        R.render_grid([P.load_surface(c)[::4] for c in chunk], os.path.join(a.out, f"sheet_{s // a.per_sheet}.png"),
                      labels=[f"{s + k:02d} {c[:6]}" for k, c in enumerate(chunk)], ncols=6, panel=(260, 260))
    print("candidates in", a.out)


if __name__ == "__main__":
    main()
