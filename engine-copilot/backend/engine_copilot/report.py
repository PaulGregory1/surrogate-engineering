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

    The sentence around it is "ramène le Cd de 1.90 à 1.44, soit X" — an engineer reads
    a drop there as -24 %, not +24 %.
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
        return "_Historique de convergence non fourni._"

    step = max(1, len(best) // n_points)
    rows = ["| Itération | Meilleur |", "| --- | --- |"]
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
            f"Le solveur LBM de référence donne $C_d = {_num(verify.get('cd_lbm'))}$ sur la forme "
            f"optimisée, contre {_num(verify.get('cd_surrogate'))} annoncés par le surrogate — "
            f"soit un écart de {_num(verify.get('gap_pct'), 2, ' %')}, obtenu en "
            f"{_num(verify.get('wall_time_s'), 1, ' s')} de calcul direct."
        )
    else:
        verification = (
            "_Aucune vérification LBM n'a été lancée sur cette forme._ Les valeurs ci-dessus "
            "sont des sorties de surrogate : elles n'engagent que la précision du modèle."
        )

    re_warning = ""
    try:
        if reynolds is not None and not (10.0 <= float(reynolds) <= 150.0):
            re_warning = (
                f"\n> **Hors domaine d'entraînement.** Le modèle a été entraîné sur "
                f"Re ∈ [10, 150] ; ce résultat à Re = {_num(reynolds, 0)} est une "
                f"extrapolation et l'erreur attendue ne s'y applique pas.\n"
            )
    except (TypeError, ValueError):
        pass

    return f"""# Rapport d'optimisation — traînée à aire constante

## Résumé exécutif

La descente sur les coefficients de forme ramène le coefficient de traînée de
**{_num(cd_initial)}** à **{_num(cd_final)}**, soit **{_pct_drop(cd_initial, cd_final)}**,
à nombre de Reynolds fixé (Re = {_num(reynolds, 0)}) et à aire constante.
{re_warning}
## Conditions

| Paramètre | Valeur |
| --- | --- |
| Nombre de Reynolds | {_num(reynolds, 0)} |
| Modèle surrogate | `{model}` |
| Erreur $C_d$ attendue (MAPE test) | {_num(quality.get('cd_mape_pct'), 2, ' %')} |
| Pire cas observé au test | {_num(quality.get('cd_worst_pct'), 2, ' %')} |
| Aire cible | {_num(ctx.get('target_area'), 5)} |
| Itérations | {len(cd_series) if cd_series else 'n/a'} |

## Forme retenue

Coefficients de Fourier : {_coeff_row(ctx.get('final_coeffs'))}

## Convergence

{_history_table(cd_series)}

## Vérification

{verification}

## Réserves

- Le surrogate est un modèle appris : l'écart annoncé ci-dessus est une moyenne sur le
  jeu de test, pas une garantie sur cette forme précise.
- L'optimisation se fait *à travers* le surrogate. Un optimum du modèle n'est pas un
  optimum de l'écoulement réel — d'où l'intérêt de la vérification LBM.
- L'aire est contrainte, mais aucune autre contrainte de fabrication n'est imposée.

---
_Rapport généré par Engine-Copilot à partir du résultat d'optimisation, sans intervention
d'un modèle de langage : les chiffres proviennent directement du solveur et du surrogate._
"""


def render_report(project: str, context: dict | None) -> str:
    """Render a Markdown report for a flow optimisation result."""
    ctx = context or {}
    if project == "flow":
        return _flow_report(ctx)
    raise ValueError(f"unknown project '{project}' — expected 'flow'")
