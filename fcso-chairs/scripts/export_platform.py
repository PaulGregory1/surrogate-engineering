"""Export the FCSO review data the platform's /fcso page reads.

    python3 scripts/export_platform.py

Reads the outputs of scripts/make_report.py (report/results.json, report/viewer.html's META and
report/meshes/<id>.js) and writes, under engine-copilot/frontend/public/fcso/:

    summary.json        aggregates per method, paired FCSO vs GD quality, per-chair snapshot metrics
    meshes/<id>.json    web meshes of FCSO (x0) and GD only, at the snapshots the page shows

Standard library only, nothing is recomputed: the numbers are the report's.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report")
OUT = os.path.join(ROOT, "..", "engine-copilot", "frontend", "public", "fcso")

METHODS = ["fcso_x0", "gd"]           # the required comparison; the ablations stay in the table
SNAPSHOTS = ["init", "30", "50", "70"]  # "70" falls back to "final" for runs stopped by the budget


def read_meta():
    with open(os.path.join(REPORT, "viewer.html")) as f:
        line = next(l for l in f if l.startswith("const META = "))
    return json.loads(line[len("const META = "):].rstrip().rstrip(";"))


def read_meshes(sid):
    with open(os.path.join(REPORT, "meshes", f"{sid}.js")) as f:
        s = f.read()
    return json.loads(s[s.index("= {") + 2:].rstrip().rstrip(";"))


def main():
    meta = read_meta()
    with open(os.path.join(REPORT, "results.json")) as f:
        res = json.load(f)
    os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)

    chairs = []
    for c in meta["chairs"]:
        sid, entry = c["id"], dict(id=c["id"], name=c["name"], runs={})
        src = read_meshes(sid)
        out = dict(init=src["init"], runs={})
        for m in METHODS:
            run = c["runs"][m]
            cells, meshes = {}, {}
            for lab in SNAPSHOTS:
                key = lab
                if lab == "70" and run["cells"].get("70", {}).get("state") != "ok":
                    key = "final"
                cell = run["cells"].get(key)
                if not cell or cell.get("state") != "ok":
                    continue
                cells[lab] = dict(cell, final=key == "final")
                if lab != "init":
                    meshes[lab] = src["runs"][m][key]
            entry["runs"][m] = dict(stop=run["stop"], steps=run["steps"], time=run["time"], cells=cells)
            out["runs"][m] = meshes
        chairs.append(entry)
        with open(os.path.join(OUT, "meshes", f"{sid}.json"), "w") as f:
            json.dump(out, f, separators=(",", ":"))

    names = {m["key"]: m["name"] for m in meta["methods"]}
    summary = dict(
        methods=[dict(key=k, name=names[k], **res["aggregates"][k]) for k in names],
        overhead=res["overhead_fcso_x0_vs_gd"],
        paired=res["paired_quality_fcso_x0_vs_gd"],
        chairs=chairs,
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    print(f"wrote {len(chairs)} chairs to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
