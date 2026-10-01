"""Deterministic Markdown report generation from an optimisation result.

§3.3 of PLATFORM_PLAN.md describes this route as calling Claude internally, but §5.3
requires the `render_report` *tool* to use "un template déterministe pour éviter les
boucles". Those two pull in opposite directions: the agent calling a tool that re-enters
Claude is exactly the recursion §5.3 forbids, and it would also make a report cost a CLI
subprocess spawn. The template wins for both callers — the endpoint and the tool share
this module, so a report is the same document whoever asked for it, and it is free.

Everything here is defensive about missing keys: the agent can hand us a partial context
(it summarises tool output before passing it on), and a report with "n/a" in one row is
worth more than a 500.
"""

from __future__ import annotations

from typing import Any

from . import config


def _num(value: Any, digits: int = 4, suffix: str = "") -> str:
    """Format a number for the report, or 'n/a' when the caller left it out."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _pct_drop(initial: Any, final: Any) -> str:
    """Signed relative change, so a reduction reads as a negative number.

    The sentence around it is "takes the Cd from 1.90 to 1.44, i.e. X" — an engineer
    reads a drop there as -24 %, not +24 %.
    """
    try:
        i, f = float(initial), float(final)
    except (TypeError, ValueError):
        return "n/a"
    if abs(i) < 1e-12:
        return "n/a"
    return f"{100.0 * (f - i) / abs(i):+.2f} %"


def _first(*values: Any) -> Any:
    """First value that was actually supplied — `None` is "absent", 0.0 is a number."""
    return next((v for v in values if v is not None), None)


def _coeff_row(coeffs: Any) -> str:
    names = ("a2", "b2", "a3", "b3", "a4", "b4")
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 6:
        return "n/a"
    return ", ".join(f"{n} = {float(c):+.4f}" for n, c in zip(names, coeffs))


def _history_table(best: Any, n_points: int = 10) -> str:
    """Downsample a convergence history to a small Markdown table."""
    if not isinstance(best, (list, tuple)) or not best:
        return "_No convergence history was supplied._"

    step = max(1, len(best) // n_points)
    rows = ["| Iteration | Best |", "| --- | --- |"]
    for i in range(0, len(best), step):
        rows.append(f"| {i} | {_num(best[i])} |")
    if (len(best) - 1) % step:
        rows.append(f"| {len(best) - 1} | {_num(best[-1])} |")
    return "\n".join(rows)


def _flow_report(ctx: dict) -> str:
    model = ctx.get("model", "unet-phys")
    quality = config.FLOW_MODEL_QUALITY.get(model, {})

    # Two callers, one renderer, and they do not name things the same way: the lab posts
    # its own optimisation state (`cd_surrogate_final`, `history.cd_best`), while the
    # agent is told by `render_report`'s tool description to forward the `flow_optimize`
    # payload as-is (`cd_final`, `convergence_cd_best`). Accept both spellings here
    # rather than at each call site — anything else silently prints `n/a` for the two
    # numbers the report exists to carry.
    history = ctx.get("history") or {}
    cd_series = history.get("cd_best") or ctx.get("convergence_cd_best") or []

    cd_initial = ctx.get("cd_initial", cd_series[0] if cd_series else None)
    cd_final = _first(
        ctx.get("cd_surrogate_final"), ctx.get("cd_final"), cd_series[-1] if cd_series else None
    )
    reynolds = ctx.get("reynolds")

    verify = ctx.get("verify") or {}
    if verify:
        verification = (
            f"The reference LBM solver gives $C_d = {_num(verify.get('cd_lbm'))}$ on the "
            f"optimised shape, against {_num(verify.get('cd_surrogate'))} claimed by the "
            f"surrogate — a gap of {_num(verify.get('gap_pct'), 2, ' %')}, obtained in "
            f"{_num(verify.get('wall_time_s'), 1, ' s')} of direct computation."
        )
    else:
        verification = (
            "_No LBM verification was run on this shape._ The values above are surrogate "
            "outputs: they commit to nothing beyond the model's own accuracy."
        )

    re_warning = ""
    try:
        if reynolds is not None and not (10.0 <= float(reynolds) <= 150.0):
            re_warning = (
                f"\n> **Outside the training range.** The model was trained on "
                f"Re ∈ [10, 150]; this result at Re = {_num(reynolds, 0)} is an "
                f"extrapolation, and the expected error does not apply to it.\n"
            )
    except (TypeError, ValueError):
        pass

    return f"""# Optimisation report — drag at constant area

## Executive summary

Descent on the shape coefficients takes the drag coefficient from
**{_num(cd_initial)}** to **{_num(cd_final)}**, i.e. **{_pct_drop(cd_initial, cd_final)}**,
at a fixed Reynolds number (Re = {_num(reynolds, 0)}) and constant area.
{re_warning}
## Conditions

| Parameter | Value |
| --- | --- |
| Reynolds number | {_num(reynolds, 0)} |
| Surrogate model | `{model}` |
| Expected $C_d$ error (test MAPE) | {_num(quality.get('cd_mape_pct'), 2, ' %')} |
| Worst case observed on test | {_num(quality.get('cd_worst_pct'), 2, ' %')} |
| Target area | {_num(ctx.get('target_area'), 5)} |
| Iterations | {len(cd_series) if cd_series else 'n/a'} |

## Retained shape

Fourier coefficients: {_coeff_row(ctx.get('final_coeffs'))}

## Convergence

{_history_table(cd_series)}

## Verification

{verification}

## Caveats

- The surrogate is a learned model: the error quoted above is a mean over the test
  split, not a guarantee on this particular shape.
- The optimisation runs *through* the surrogate. An optimum of the model is not an
  optimum of the real flow — which is what the LBM verification is for.
- Area is constrained, but no other manufacturing constraint is imposed.

---
_Report generated by Engine-Copilot from the optimisation result, with no language model
involved: the numbers come straight from the solver and the surrogate._
"""


def _stress_report(ctx: dict) -> str:
    model = ctx.get("model", "gnn-mp10-glob")
    quality = config.STRESS_MODEL_QUALITY.get(model, {})
    peak_initial = ctx.get("peak_vm_initial")
    peak_final = ctx.get("peak_vm_final")
    holes = ctx.get("final_holes") or []

    if isinstance(holes, list) and holes:
        hole_rows = "\n".join(
            f"| {i + 1} | {_num(h[0], 4)} | {_num(h[1], 4)} | {_num(h[2], 4)} |"
            for i, h in enumerate(holes)
            if isinstance(h, (list, tuple)) and len(h) == 3
        )
        hole_table = "| Hole | x | y | r |\n| --- | --- | --- | --- |\n" + hole_rows
    else:
        hole_table = "_No final layout was supplied._"

    return f"""# Optimisation report — von Mises stress

## Executive summary

Repositioning the holes takes the peak von Mises stress from **{_num(peak_initial)}** to
**{_num(peak_final)}**, i.e. **{_pct_drop(peak_initial, peak_final)}**, at constant removed
area and under the same load case.

## Conditions

| Parameter | Value |
| --- | --- |
| Load angle | {_num(ctx.get('load_angle_deg'), 1, '°')} |
| Surrogate model | `{model}` |
| Expected peak error (test) | {_num(quality.get('peak_err_pct'), 2, ' %')} |
| Relative $L_2$ error on the field | {_num(quality.get('field_rel_l2'), 3)} |
| Search iterations | {_num(ctx.get('n_iters'), 0)} |

## Retained layout

{hole_table}

## Caveats

- The search is a randomised local exploration over the surrogate, not a gradient
  descent: the mesh changes with every hole move, so there is no gradient of the peak
  with respect to hole positions.
- The von Mises peak is the hardest quantity for this model to predict (it is an
  extremum, not a mean). Check the retained design against a fine FE solve before any
  decision rests on it.
- Removed area is conserved; no manufacturing constraint is imposed.

---
_Report generated by Engine-Copilot from the optimisation result, with no language model
involved: the numbers come straight from the solver and the surrogate._
"""


def render_report(project: str, context: dict | None) -> str:
    """Render a Markdown report for a flow or stress optimisation result."""
    ctx = context or {}
    if project == "flow":
        return _flow_report(ctx)
    if project == "stress":
        return _stress_report(ctx)
    raise ValueError(f"unknown project '{project}' — expected 'flow' or 'stress'")
