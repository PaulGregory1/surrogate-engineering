"""Milestone 7 report: metrics tables, Figure-3-style grid, per-chair trajectory PNGs and the interactive viewer.

    ./.venv/bin/python scripts/make_report.py                  everything (tables, PNGs, viewer)
    ./.venv/bin/python scripts/make_report.py --no-png         tables + viewer only
    ./.venv/bin/python scripts/make_report.py --decode         decode meshes missing a meshes.npz (uses the GPU)

Inputs, per run and chair: runs/<run>/<id>/{result.json, eval.json, meshes.npz}. eval.json holds the
evaluation metrics of every snapshot (marching cubes at 128^3 on the logits, level 0, every component kept,
`fcso/metrics.evaluate_run`) and meshes.npz the same meshes, so nothing is decoded here unless --decode.
Only chairs whose run is finished (result.json done) are reported; a chair still running shows as pending.

Outputs:
    report/results.md, report/results.json   tables (per chair, per method aggregates, FCSO vs GD overhead)
    REPORT.md                                  the block between the AUTO markers is replaced by results.md
    report/grid_70.png                         initial + each method at 70 % (or its final state), all chairs
    report/trajectories/<id>.png               initial, 10 ... 70 %, final; FCSO and GD rows (+ others)
    report/viewer.html, report/meshes/<id>.js  interactive viewer (three.js from a CDN, works on file://)
"""
import argparse
import base64
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORT = os.path.join(ROOT, "report")
LABELS = ["init", "10", "20", "30", "40", "50", "60", "70", "final"]
# run name -> display name; order = row order. The first two are the required comparison, the others
# appear behind the viewer's toggle and as extra columns of the grid when they exist.
METHODS = [
    ("fcso_x0", "FCSO (x0 guidance, λ 1e4)"),
    ("gd", "GD (λ_reg 0)"),
    ("gd_reg", "GD (λ_reg 1e-3)"),
    ("fcso", "FCSO (noisy guidance, λ 1e3)"),
    ("fcso_noguide", "FCSO without guidance"),
    ("fmg", "FMG (x0, λ 1e4)"),
]
SHORT = {"fcso_x0": "FCSO", "gd": "GD", "gd_reg": "GD λ_reg", "fcso": "FCSO noisy", "fcso_noguide": "FCSO no guid.",
         "fmg": "FMG"}
PNG_SHORT = {"fcso_x0": "FCSO", "gd": "GD", "gd_reg": "GD lam_reg", "fcso": "FCSO noisy", "fcso_noguide": "FCSO no guid.",
             "fmg": "FMG"}  # VTK's default font has no Greek letters
PRIMARY = ("fcso_x0", "gd")
FACE_TARGET = 20000
AUTO_BEGIN, AUTO_END = "<!-- AUTO:BEGIN (scripts/make_report.py) -->", "<!-- AUTO:END -->"


def chair_ids():
    from fcso.prior import read_ids
    return read_ids(os.path.join(ROOT, "splits", "opt.txt"))


def chair_names():
    out = {}
    with open(os.path.join(ROOT, "splits", "opt.txt")) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                sid, _, note = line.partition("#")
                note = note.strip()
                out[sid.strip()] = note.split(" ", 1)[1] if " " in note else note
    return out


# ---------------------------------------------------------------------------------------------- loading
def load_run(name, sid):
    d = os.path.join(ROOT, "runs", name, sid)
    rp, ep = os.path.join(d, "result.json"), os.path.join(d, "eval.json")
    if not os.path.exists(rp):
        return None
    res = json.load(open(rp))
    if not res["done"]:
        return dict(status="running", result=res)
    if not os.path.exists(ep):
        return dict(status="unevaluated", result=res)
    ev = {r["label"]: r for r in json.load(open(ep))["rows"]}
    cyc = json.load(open(os.path.join(d, "cycles.json")))
    return dict(status="done", result=res, eval=ev, dir=d,
                s_per_cycle=statistics.median([c["time_s"] for c in cyc]) if cyc else None)


def load_all(ids):
    runs = {}
    for name, _ in METHODS:
        if not os.path.isdir(os.path.join(ROOT, "runs", name)):
            continue
        per = {sid: load_run(name, sid) for sid in ids}
        if any(v is not None for v in per.values()):
            runs[name] = per
    return runs


def load_meshes(run, sid, decode_prior=None):
    """{label: trimesh} of one run/chair from meshes.npz (or decoded if a prior is given and the file is missing)."""
    import trimesh
    p = os.path.join(run["dir"], "meshes.npz")
    if not os.path.exists(p):
        if decode_prior is None:
            return None
        from fcso.metrics import evaluate_run
        evaluate_run(decode_prior, run["dir"], log=lambda *a: None)
    z = np.load(p)
    out = {}
    for lab in LABELS:
        if f"{lab}_v" in z:
            out[lab] = trimesh.Trimesh(z[f"{lab}_v"], z[f"{lab}_f"], process=False)
    return out


# ---------------------------------------------------------------------------------------------- tables
def fmt(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.{nd}f} %" if pct else f"{x:.{nd}f}"


def cell(run, label):
    """Metric row of a snapshot label, or None if not reached / not available."""
    if run is None or run["status"] != "done":
        return None
    r = run["eval"].get(label)
    return r if r and r.get("reached") else None


def aggregate(runs, ids):
    agg = {}
    for name, per in runs.items():
        done = [sid for sid in ids if per.get(sid) and per[sid]["status"] == "done"]
        a = dict(n_done=len(done), n_total=len(ids), s_per_cycle=_med([per[s]["s_per_cycle"] for s in done]))
        for th in ("50", "70"):
            rows = [cell(per[s], th) for s in done]
            ok = [r for r in rows if r]
            a[th] = dict(
                success=len(ok), rate=len(ok) / len(done) if done else None,
                median_time=_med([r["time_s"] for r in ok]), median_steps=_med([r["step"] for r in ok]),
                mean_ratio=_mean([r["ratio"] for r in ok]), mean_mesh_ratio=_mean([r["mesh_ratio"] for r in ok]),
                mean_stray=_mean([r["stray_components"] for r in ok]),
                mean_stray_share=_mean([r["stray_share"] for r in ok]),
                mean_chamfer=_mean([r["chamfer_to_initial"] for r in ok]),
                whole=sum(r["stray_components"] == 0 for r in ok))
        fin = [per[s]["eval"].get("final") for s in done if per[s]["result"]["stop_reason"] == "budget"]
        fin = [r for r in fin if r]
        a["budget"] = dict(n=len(fin), mean_ratio=_mean([r["ratio"] for r in fin]),
                           mean_mesh_ratio=_mean([r["mesh_ratio"] for r in fin]),
                           whole=sum(r["stray_components"] == 0 for r in fin))
        agg[name] = a
    return agg


def overhead(runs, ids, a="fcso_x0", b="gd"):
    """Time of method a over method b at equal reduction, on the chairs where both reached the threshold."""
    out = {}
    if a not in runs or b not in runs:
        return out
    for th in ("10", "20", "30", "40", "50", "60", "70"):
        pairs = [(cell(runs[a].get(s), th), cell(runs[b].get(s), th)) for s in ids]
        pairs = [(x, y) for x, y in pairs if x and y]
        if not pairs:
            out[th] = dict(n=0)
            continue
        ta, tb = [x["time_s"] for x, _ in pairs], [y["time_s"] for _, y in pairs]
        out[th] = dict(n=len(pairs), median_a=_med(ta), median_b=_med(tb),
                       median_ratio=_med([x / max(y, 1e-9) for x, y in zip(ta, tb)]),
                       min_ratio=min(x / max(y, 1e-9) for x, y in zip(ta, tb)),
                       max_ratio=max(x / max(y, 1e-9) for x, y in zip(ta, tb)),
                       median_step_ratio=_med([x["step"] / max(y["step"], 1) for x, y in pairs]))
    return out


def paired_quality(runs, ids, a="fcso_x0", b="gd"):
    """Shape quality of methods a and b on the chairs where both reached the threshold (same chairs, same
    reduction): mean / median stray components and stray volume share, whole chairs, mean Chamfer."""
    out = {}
    if a not in runs or b not in runs:
        return out
    for th in ("50", "70"):
        pairs = [(cell(runs[a].get(s), th), cell(runs[b].get(s), th)) for s in ids]
        pairs = [(x, y) for x, y in pairs if x and y]
        o = dict(n=len(pairs))
        for key, rows in ((a, [x for x, _ in pairs]), (b, [y for _, y in pairs])):
            o[key] = dict(mean_stray=_mean([r["stray_components"] for r in rows]),
                          median_stray=_med([r["stray_components"] for r in rows]),
                          mean_share=_mean([r["stray_share"] for r in rows]),
                          median_share=_med([r["stray_share"] for r in rows]),
                          whole=sum(r["stray_components"] == 0 for r in rows),
                          mean_chamfer=_mean([r["chamfer_to_initial"] for r in rows]),
                          mean_mesh_ratio=_mean([r["mesh_ratio"] for r in rows]))
        o["fewer_stray"] = sum(x["stray_components"] < y["stray_components"] for x, y in pairs)
        o["smaller_share"] = sum(x["stray_share"] < y["stray_share"] for x, y in pairs)
        out[th] = o
    return out


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def results_markdown(runs, ids, names, agg, ovh):
    L = []
    disp = dict(METHODS)
    L.append(f"*Generated by `scripts/make_report.py` on {time.strftime('%Y-%m-%d %H:%M')}.*\n")
    pend = [(n, s[:8]) for n, per in runs.items() for s in ids
            if per.get(s) is None or per[s]["status"] != "done"]
    if pend:
        miss = {}
        for n, s in pend:
            miss.setdefault(n, []).append(s)
        L.append("Chairs not finished yet (excluded from the tables): " +
                 "; ".join(f"{n}: {len(v)}" for n, v in miss.items()) + ".\n")
    L.append("#### Aggregates per method\n")
    L.append("Success = threshold reached within the 1000-gradient-step budget. Time and steps are medians over the "
             "chairs that reached the threshold; the other columns are means over those chairs. V/V0 = 64³ "
             "estimator (the optimized quantity), mesh = 128³ marching-cubes volume ratio, stray = connected "
             "components other than the largest, whole = chairs with no stray component, CD = Chamfer distance "
             "to the initial shape (10k points; ≈ 0.025 for an unchanged shape).\n")
    L.append("| method | thr. | success | median time (s) | median steps | V/V0 | mesh | stray comps | stray vol. | whole | CD |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, a in agg.items():
        for th in ("50", "70"):
            t = a[th]
            L.append(f"| {disp[name]} | {th} % | {t['success']}/{a['n_done']} | {fmt(t['median_time'], 0)} | "
                     f"{fmt(t['median_steps'], 0)} | {fmt(t['mean_ratio'])} | {fmt(t['mean_mesh_ratio'])} | "
                     f"{fmt(t['mean_stray'], 1)} | {fmt(t['mean_stray_share'], 1, pct=True)} | "
                     f"{t['whole']}/{t['success']} | {fmt(t['mean_chamfer'], 3)} |")
    L.append("")
    L.append("Budget stops (70 % not reached in 1000 steps): final state, means over those chairs.\n")
    L.append("| method | budget stops | final V/V0 | final mesh | whole |")
    L.append("| --- | --- | --- | --- | --- |")
    for name, a in agg.items():
        b = a["budget"]
        L.append(f"| {disp[name]} | {b['n']}/{a['n_done']} | {fmt(b['mean_ratio'])} | {fmt(b['mean_mesh_ratio'])} | "
                 f"{b['whole']}/{b['n']} |")
    L.append("")
    L.append("Median wall time per 15-step cycle (s): " +
             ", ".join(f"{SHORT[n]} {fmt(a['s_per_cycle'], 1)}" for n, a in agg.items()) + ".\n")
    if ovh:
        L.append("#### FCSO vs GD time at equal reduction\n")
        L.append("Paired over the chairs where both FCSO (x0) and GD reached the threshold; ratio = FCSO time / GD time "
                 "per chair.\n")
        L.append("| threshold | chairs | median FCSO (s) | median GD (s) | median ratio | range | median step ratio |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for th, o in ovh.items():
            if o["n"]:
                L.append(f"| {th} % | {o['n']} | {fmt(o['median_a'], 0)} | {fmt(o['median_b'], 0)} | "
                         f"{fmt(o['median_ratio'], 1)}× | {fmt(o['min_ratio'], 1)}-{fmt(o['max_ratio'], 1)}× | "
                         f"{fmt(o['median_step_ratio'], 1)}× |")
            else:
                L.append(f"| {th} % | 0 | — | — | — | — | — |")
        L.append("")
    pq = paired_quality(runs, ids)
    if pq:
        L.append("#### FCSO vs GD shape quality on the same chairs\n")
        L.append("Only the chairs where both FCSO (x0) and GD reached the threshold, so both methods are judged on the "
                 "same shapes at the same reduction. Stray = components other than the largest.\n")
        L.append("| threshold | chairs | method | stray comps mean / median | stray vol. mean / median | whole | "
                 "mesh ratio | CD | FCSO fewer strays / smaller share |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for th, o in pq.items():
            for k in ("fcso_x0", "gd"):
                q = o[k]
                last = f"{o['fewer_stray']}/{o['n']} / {o['smaller_share']}/{o['n']}" if k == "fcso_x0" else ""
                L.append(f"| {th} % | {o['n']} | {SHORT[k]} | {fmt(q['mean_stray'], 1)} / {fmt(q['median_stray'], 0)} | "
                         f"{fmt(q['mean_share'], 1, pct=True)} / {fmt(q['median_share'], 1, pct=True)} | "
                         f"{q['whole']}/{o['n']} | {fmt(q['mean_mesh_ratio'])} | {fmt(q['mean_chamfer'], 3)} | {last} |")
        L.append("")
    L.append("#### Per chair\n")
    L.append("Each threshold cell: steps / seconds; V/V0 (64³) / mesh ratio (128³); components (stray volume share); "
             "Chamfer. \"no\" = not reached. Final = state at the budget for chairs that never reached 70 %.\n")
    for sid in ids:
        L.append(f"**{sid[:8]}** ({names.get(sid, '')})\n")
        L.append("| method | stop | 50 % | 70 % | final |")
        L.append("| --- | --- | --- | --- | --- |")
        for name, per in runs.items():
            run = per.get(sid)
            if run is None or run["status"] != "done":
                L.append(f"| {SHORT[name]} | {'pending' if run is None else run['status']} | | | |")
                continue
            res = run["result"]
            cells = []
            for lab in ("50", "70", "final"):
                r = cell(run, lab)
                if lab == "final" and res["stop_reason"] == "target":
                    cells.append("")
                elif r is None:
                    cells.append("no")
                else:
                    cells.append(f"{r['step']} / {r['time_s']:.0f} s; {r['ratio']:.2f} / {r['mesh_ratio']:.2f}; "
                                 f"{r['components']} ({100 * r['stray_share']:.1f} %); {r['chamfer_to_initial']:.3f}")
            L.append(f"| {SHORT[name]} | {res['stop_reason']} ({res['steps']} st, {res['elapsed_s']:.0f} s) | "
                     + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- PNGs
def png_label(head, r):
    return (f"{head}\nV/V0 {r['ratio']:.2f} (mesh {r['mesh_ratio']:.2f})\n"
            f"step {r['step']}, {r['time_s']:.0f} s\nstray {r['stray_components']} ({100 * r['stray_share']:.1f}%)")


def make_grid70(runs, ids, names, meshes):
    from fcso import mesh_render as R
    cols = [n for n, _ in METHODS if n in runs]
    objs, labels = [], []
    for sid in ids:
        base = next((runs[n][sid] for n in cols if runs[n].get(sid) and runs[n][sid]["status"] == "done"), None)
        m_init = next((meshes[(n, sid)]["init"] for n in cols if meshes.get((n, sid))), None)
        objs.append(m_init)
        labels.append(f"{sid[:8]} initial\n{names.get(sid, '')[:34]}" if base else f"{sid[:8]} (no run)")
        for n in cols:
            run = runs[n].get(sid)
            ms = meshes.get((n, sid))
            if run is None or run["status"] != "done" or ms is None:
                objs.append(None)
                labels.append(f"{PNG_SHORT[n]}: pending")
                continue
            r70 = cell(run, "70")
            if r70:
                objs.append(ms.get("70"))
                labels.append(png_label(f"{PNG_SHORT[n]} 70%", r70))
            else:
                rf = cell(run, "final")
                objs.append(ms.get("final"))
                labels.append(png_label(f"{PNG_SHORT[n]}: 70% not reached\nfinal", rf) if rf else f"{PNG_SHORT[n]}: n/a")
    path = os.path.join(REPORT, "grid_70.png")
    R.render_grid(objs, path, labels=labels, ncols=len(cols) + 1, panel=(300, 330), color_strays=True, empty_text="")
    return path


def make_trajectory_png(sid, runs, meshes, names):
    from fcso import mesh_render as R
    rows = [n for n, _ in METHODS if n in runs and runs[n].get(sid) and runs[n][sid]["status"] == "done"
            and meshes.get((n, sid))]
    if not rows:
        return None
    objs, labels = [], []
    for n in rows:
        run, ms = runs[n][sid], meshes[(n, sid)]
        for lab in LABELS:
            r = cell(run, lab)
            head = {"init": f"{PNG_SHORT[n]}  initial", "final": f"{PNG_SHORT[n]}  final"}.get(lab, f"{PNG_SHORT[n]}  {lab}%")
            if lab == "final" and run["result"]["stop_reason"] == "target":
                objs.append(None)
                labels.append(f"{PNG_SHORT[n]}: 70% reached")
            elif r is None:
                objs.append(None)
                labels.append(f"{head}\nnot reached")
            else:
                objs.append(ms.get(lab))
                labels.append(png_label(head, r))
    os.makedirs(os.path.join(REPORT, "trajectories"), exist_ok=True)
    path = os.path.join(REPORT, "trajectories", f"{sid}.png")
    R.render_grid(objs, path, labels=labels, ncols=len(LABELS), panel=(250, 280), color_strays=True, empty_text="")
    return path


# ---------------------------------------------------------------------------------------------- viewer
def decimate(m, target):
    """Quadric decimation (VTK, via PyVista) of a trimesh to about `target` faces; returns (verts, faces)."""
    import pyvista as pv
    v, f = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)
    if len(f) <= target or len(f) < 64:
        return v, f
    pd = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    d = pd.decimate(1.0 - target / len(f))
    if d.n_cells == 0:
        return v, f
    return np.asarray(d.points, np.float32), d.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def web_mesh(m, target=FACE_TARGET):
    """Largest component + stray components (merged), decimated together to about `target` faces, as one
    vertex/index buffer where the first `n_main` triangles are the largest component."""
    import trimesh
    if m is None:
        return None
    comps = m.split(only_watertight=False)
    comps = sorted(comps, key=lambda c: abs(float(c.volume)), reverse=True)
    main = comps[0]
    stray = trimesh.util.concatenate(comps[1:]) if len(comps) > 1 else None
    nf = len(m.faces)
    ratio = min(1.0, target / max(nf, 1))
    vm, fm = decimate(main, max(int(len(main.faces) * ratio), 200))
    parts_v, parts_f = [vm], [fm]
    n_main = len(fm)
    if stray is not None:
        vs, fs = decimate(stray, max(int(len(stray.faces) * ratio), 100))
        parts_v.append(vs)
        parts_f.append(fs + len(vm))
    v = np.concatenate(parts_v)
    f = np.concatenate(parts_f)
    q = np.clip(np.round((v + 1.0) / 2.0 * 65535.0), 0, 65535).astype("<u2")
    idx = f.astype("<u2") if len(v) < 65536 else f.astype("<u4")
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")  # noqa: E731
    return dict(p=b64(q), i=b64(idx), i32=int(idx.dtype.itemsize == 4), n=int(n_main), t=int(len(f)))


def make_viewer(runs, ids, names, meshes):
    os.makedirs(os.path.join(REPORT, "meshes"), exist_ok=True)
    meta = dict(chairs=[], methods=[dict(key=n, name=d, primary=n in PRIMARY) for n, d in METHODS if n in runs],
                labels=LABELS)
    for sid in ids:
        entry = dict(id=sid, name=names.get(sid, ""), runs={})
        data = dict(init=None, runs={})
        for n in runs:
            run = runs[n].get(sid)
            if run is None or run["status"] != "done" or not meshes.get((n, sid)):
                entry["runs"][n] = dict(status="pending" if run is None else
                                        ("meshes missing" if run["status"] == "done" else run["status"]))
                continue
            res = run["result"]
            cells = {}
            data["runs"][n] = {}
            for lab in LABELS:
                r = cell(run, lab)
                if lab == "final" and res["stop_reason"] == "target":
                    cells[lab] = dict(state="reached")
                    continue
                if r is None:
                    cells[lab] = dict(state="none")
                    continue
                cells[lab] = dict(state="ok", ratio=r["ratio"], mesh=r["mesh_ratio"], step=r["step"],
                                  time=r["time_s"], comps=r["components"], stray=r["stray_share"])
                if lab == "init":
                    if data["init"] is None:
                        data["init"] = web_mesh(meshes[(n, sid)].get("init"))
                else:
                    data["runs"][n][lab] = web_mesh(meshes[(n, sid)].get(lab))
            entry["runs"][n] = dict(status="done", stop=res["stop_reason"], steps=res["steps"],
                                    time=res["elapsed_s"], final_ratio=res["ratio_final"], cells=cells)
        meta["chairs"].append(entry)
        with open(os.path.join(REPORT, "meshes", f"{sid}.js"), "w") as f:
            f.write("window.FCSO_MESHES = window.FCSO_MESHES || {};\n")
            f.write(f"window.FCSO_MESHES[{json.dumps(sid)}] = {json.dumps(data, separators=(',', ':'))};\n")
    tpl = open(os.path.join(ROOT, "scripts", "viewer_template.html")).read()
    html = tpl.replace("/*__META__*/null", json.dumps(meta, separators=(",", ":")))
    html = html.replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(REPORT, "viewer.html"), "w") as f:
        f.write(html)
    return os.path.join(REPORT, "viewer.html")


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--decode", action="store_true", help="decode meshes of runs missing meshes.npz (GPU)")
    a = ap.parse_args()
    os.makedirs(REPORT, exist_ok=True)
    t0 = time.time()
    ids, names = chair_ids(), chair_names()
    runs = load_all(ids)
    agg, ovh = aggregate(runs, ids), overhead(runs, ids)
    md = results_markdown(runs, ids, names, agg, ovh)
    with open(os.path.join(REPORT, "results.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(REPORT, "results.json"), "w") as f:
        json.dump(dict(aggregates=agg, overhead_fcso_x0_vs_gd=ovh, paired_quality_fcso_x0_vs_gd=paired_quality(runs, ids),
                       chairs={n: {s: (None if r is None else dict(status=r["status"], result=r["result"],
                                                                   eval=r.get("eval")))
                                   for s, r in per.items()} for n, per in runs.items()}), f, indent=1)
    rp = os.path.join(ROOT, "REPORT.md")
    if os.path.exists(rp):
        txt = open(rp).read()
        if AUTO_BEGIN in txt and AUTO_END in txt:
            pre, rest = txt.split(AUTO_BEGIN, 1)
            _, post = rest.split(AUTO_END, 1)
            with open(rp, "w") as f:
                f.write(pre + AUTO_BEGIN + "\n" + md + "\n" + AUTO_END + post)
    print(f"tables: {time.time() - t0:.1f} s")

    if a.no_png and a.no_viewer:
        return
    prior = None
    if a.decode:
        from fcso.prior import Prior
        prior = Prior(load_dm=False)
    meshes = {}
    for n, per in runs.items():
        for sid, run in per.items():
            if run and run["status"] == "done":
                ms = load_meshes(run, sid, prior)
                if ms is not None:
                    meshes[(n, sid)] = ms
    print(f"meshes loaded: {len(meshes)} chair-runs, {time.time() - t0:.1f} s")
    if not a.no_png:
        print(make_grid70(runs, ids, names, meshes))
        for sid in ids:
            p = make_trajectory_png(sid, runs, meshes, names)
            if p:
                print(p)
        print(f"PNGs: {time.time() - t0:.1f} s")
    if not a.no_viewer:
        print(make_viewer(runs, ids, names, meshes), f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
