"""Tabulate evaluated runs (runs/<name>/<id>/{result.json, eval.json}) side by side.

    ./.venv/bin/python scripts/summarize_runs.py gd gd_reg fcso fcso_lamx3 ... [--ids 2b5953c9 ...]
         [--labels 50 70 final] [--out runs/m6/summary]

Per run and chair: stop reason, steps and time to 70 % (or budget), and at each requested snapshot the 64^3
estimator ratio V/V0, the 128^3 mesh ratio, the gap between them, components, stray share and Chamfer to
the initial shape. Writes <out>.json and <out>.md (markdown table) and prints the table.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--labels", nargs="*", default=["50", "70", "final"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "m6", "summary"))
    a = ap.parse_args()
    rows = []
    for name in a.runs:
        d = os.path.join(ROOT, "runs", name)
        if not os.path.isdir(d):
            print(f"skip {name}: no runs/{name}", file=sys.stderr)
            continue
        for sid in sorted(os.listdir(d)):
            rdir = os.path.join(d, sid)
            if not os.path.isfile(os.path.join(rdir, "eval.json")):
                continue
            if a.ids and not any(sid.startswith(p) for p in a.ids):
                continue
            res = json.load(open(os.path.join(rdir, "result.json")))
            ev = {r["label"]: r for r in json.load(open(os.path.join(rdir, "eval.json")))["rows"]}
            row = dict(run=name, id=sid, stop=res["stop_reason"], steps=res["steps"], time_s=res["elapsed_s"],
                       ratio_final=res["ratio_final"], failures=res["failures"], snaps={})
            for lab in a.labels:
                e = ev.get(lab)
                if e is None or not e.get("reached"):
                    row["snaps"][lab] = None
                    continue
                row["snaps"][lab] = {k: e[k] for k in ("step", "time_s", "ratio", "mesh_ratio", "components",
                                                       "stray_share", "chamfer_to_initial")}
                row["snaps"][lab]["gap"] = e["mesh_ratio"] - e["ratio"]
            rows.append(row)
    head = "| run | chair | stop | steps | time s | " + " | ".join(
        f"{l}: V/V0 (64³) / mesh (128³) / comps / stray / CD" for l in a.labels) + " |"
    lines = [head, "|" + " --- |" * (5 + len(a.labels))]
    for r in rows:
        cells = []
        for lab in a.labels:
            s = r["snaps"][lab]
            cells.append("—" if s is None else f"{s['ratio']:.3f} / {s['mesh_ratio']:.3f} / {s['components']} / "
                                               f"{100 * s['stray_share']:.1f} % / {s['chamfer_to_initial']:.3f}")
        lines.append(f"| {r['run']} | {r['id'][:8]} | {r['stop']} | {r['steps']} | {r['time_s']:.0f} | "
                     + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print(md)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".md", "w").write(md + "\n")


if __name__ == "__main__":
    main()
