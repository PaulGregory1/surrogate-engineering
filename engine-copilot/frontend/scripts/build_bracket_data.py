"""Extract the Bracket Lab page data from a bracket-lab checkout.

    python3 scripts/build_bracket_data.py /path/to/bracket-lab

Writes `public/bracket/bracket.json` plus two cropped images. Every number comes from
the study's own files — run logs (`runs/*/history.jsonl`), `runs/finer_mesh.json`, the
results table in `docs/results.md`, the parameter table in `README.md`, and the limit /
budget constants in its source. Nothing is typed in by hand. Needs only Pillow.
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageChops

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../../../bracket-lab").resolve()
OUT = Path(__file__).resolve().parent.parent / "public" / "bracket"
RUNS = [("agent-01", "Agent 1", "agent"), ("agent-02", "Agent 2", "agent"), ("optuna-01", "Optuna", "optuna")]


def const(path: str, name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([\d.]+)", (SRC / path).read_text(), re.M)
    return float(m.group(1))


def md_table(text: str, header_start: str) -> list[dict]:
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    head = cells(lines[i])
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append(dict(zip(head, cells(l))))
    return rows


def num(s: str):
    m = re.search(r"-?[\d.]+", s.replace("**", ""))
    return float(m.group()) if m and s.strip() not in ("-", "") else None


def history(run: str) -> list[dict]:
    return [json.loads(l) for l in (SRC / "runs" / run / "history.jsonl").read_text().splitlines() if l.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sigma_lim = const("fea/evaluate.py", "SIGMA_MAX")
    delta_lim = const("fea/evaluate.py", "DELTA_MAX")
    fea_budget = int(const("mcp_server/lab.py", "FEA_BUDGET"))

    results = md_table((SRC / "docs/results.md").read_text(), "| Design")
    finer = json.loads((SRC / "runs/finer_mesh.json").read_text())

    base_row = next(r for r in results if r["Design"] == "Baseline")
    baseline = {
        "mass_g": num(base_row["Mass (g)"]),
        "sigma_eval": num(base_row["sigma_eval (MPa)"]),
        "delta_max": num(base_row["delta_max (mm)"]),
        "feasible": base_row["Feasible"] == "yes",
    }

    runs = []
    for key, label, kind in RUNS:
        row = next(r for r in results if r["Design"].startswith(key))
        best_id = re.search(r"\((e\d+)\)", row["Design"]).group(1)
        rows = history(key)
        fea = [r for r in rows if r["fidelity"] == "fea" and r.get("budget_used", True)]
        best = next(r for r in fea if r["eval_id"] == best_id)
        trace, run_best = [], None
        for n, r in enumerate(fea, 1):
            o = r["outputs"]
            if r["feasible"] and (run_best is None or o["mass_g"] < run_best):
                run_best = o["mass_g"]
            trace.append({"n": n, "mass_g": o["mass_g"], "sigma_eval": o["sigma_eval"],
                          "delta_max": o["delta_max"], "feasible": r["feasible"], "best_g": run_best})
        assert len(fea) == int(num(row["FEA used"])), f"{key}: FEA count differs from results.md"
        runs.append({
            "key": key, "label": label, "kind": kind,
            "mass_g": best["outputs"]["mass_g"],
            "sigma_eval": best["outputs"]["sigma_eval"],
            "delta_max": best["outputs"]["delta_max"],
            "feasible": best["feasible"],
            "fea_used": len(fea),
            "sigma_eval_fine": finer[key]["sigma_eval"],
            "params": best["params"],
            "trace": trace,
        })

    # Parameter meanings and bounds, from the README's parameter table.
    ptab = md_table((SRC / "README.md").read_text(), "| Parameter")
    params = [{
        "name": p["Parameter"].strip("`"),
        "meaning": re.sub(r"\s*\((mm|mm, 0 = none)\)", "", p["Meaning"]),
        "unit": "mm" if "(mm" in p["Meaning"] else "",
        "min": num(p["Min"]), "max": num(p["Max"]), "baseline": num(p["Baseline"]),
    } for p in ptab]

    data = {
        "limits": {"sigma_eval": sigma_lim, "delta_max": delta_lim},
        "fea_budget": fea_budget,
        "baseline": baseline,
        "runs": runs,
        "params": params,
    }
    (OUT / "bracket.json").write_text(json.dumps(data, indent=1))

    # Images: trim the white margin off the baseline render; keep the iso half of the
    # best design's stress map and drop its one-line caption strip (it carries run ids).
    im = Image.open(SRC / "docs/img/baseline_iso.png").convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    pad = 40
    im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
             min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))).save(OUT / "baseline.png", optimize=True)

    sm = Image.open(SRC / "docs/img/stress_map_best.png").convert("RGB")
    sm.crop((0, 26, sm.width // 2 - 6, sm.height)).save(OUT / "best_stress.png", optimize=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
