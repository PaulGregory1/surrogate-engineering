"""Tools exposed to the Claude Agent SDK — thin async wrappers around the services.

Each tool is a small, safe, read-only wrapper around our own FastAPI service layer
(no shell, no arbitrary file access) so bypassPermissions is a reasonable
permission_mode for this MCP server (see agent.py).

Two rules the whole file obeys:

1. **Nothing heavy runs on the event loop.** Every service call here is CPU-bound torch
   or a compiled LBM solve, so it goes through `anyio.to_thread.run_sync`. The MCP server
   is in-process with FastAPI: blocking here would stall the SSE streams feeding the labs.
2. **Nothing large goes back to the model.** The streaming endpoints exist to feed the UI
   pixel data; a field, a mesh or 60 frames of vorticity would blow the context for no
   benefit. Tools consume the stream and return only the scalars an engineer would quote.
"""

from __future__ import annotations

import json

import anyio
from claude_agent_sdk import tool

from . import config
from .flow_service import flow_service
from .lbm_frames import simulate_streaming
from .report import render_report
from .stress_service import stress_service


def _ok(payload: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


def _err(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _downsample(series: list[float], n: int = 20) -> list[float]:
    """Keep ~n points of a convergence curve, always including the last one."""
    if len(series) <= n:
        return [round(v, 5) for v in series]
    step = len(series) / n
    idx = sorted({int(i * step) for i in range(n)} | {len(series) - 1})
    return [round(series[i], 5) for i in idx]


def _check_coeffs(coeffs) -> str | None:
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 6:
        return "coeffs must have exactly 6 floats: a2,b2,a3,b3,a4,b4"
    return None



@tool(
    "flow_predict",
    "Predict drag coefficient (Cd) and flow fields for a bluff-body shape defined by "
    "6 Fourier coefficients (a2,b2,a3,b3,a4,b4), at a given Reynolds number. Use this "
    "whenever the user asks for a prediction, a Cd estimate, or 'what happens if I change the shape'. "
    "Always report the expected error percentage alongside the Cd — never state Cd without it.",
    {
        "coeffs": list,
        "reynolds": float,
    },
)
async def flow_predict_tool(args: dict) -> dict:
    coeffs = args["coeffs"]
    reynolds = args.get("reynolds", 80.0)
    if problem := _check_coeffs(coeffs):
        return _err(problem)
    result = await anyio.to_thread.run_sync(
        lambda: flow_service.predict(coeffs, reynolds=reynolds, model_key="unet-phys")
    )
    return _ok(
        {
            "cd_surrogate": round(result["cd_surrogate"], 4),
            "cd_error_pct_expected": result["cd_error_pct_expected"],
            "reynolds": reynolds,
            "coeffs": coeffs,
            "in_training_range": 10 <= reynolds <= 150,
        }
    )


@tool(
    "flow_optimize",
    "Run the Adam shape optimisation through the frozen flow surrogate and return the "
    "optimised shape with its drag coefficient. Use this when the user asks to optimise, "
    "improve or minimise drag on a shape. Pass the shape currently on screen as `coeffs` so "
    "the descent starts from the user's design. This takes a few seconds. The returned Cd is "
    "a surrogate value: recommend flow_verify afterwards to confirm it against the real solver.",
    {
        "coeffs": list,
        "reynolds": float,
        "iters": int,
    },
)
async def flow_optimize_tool(args: dict) -> dict:
    coeffs = args.get("coeffs")
    reynolds = args.get("reynolds", 80.0)
    iters = int(args.get("iters", 250))
    if coeffs is not None and (problem := _check_coeffs(coeffs)):
        return _err(problem)

    def run() -> dict:
        # n_starts=1 for the same reason the UI uses it: seeded with the user's shape the
        # descent is effectively unimodal at fixed area, and extra random starts would only
        # multiply the wall time of a tool call the user is waiting on.
        final = {}
        for event in flow_service.optimise_streaming(
            reynolds=reynolds, n_starts=1, iters=iters, model_key="unet-phys", seed=0, init_coeffs=coeffs
        ):
            if event.get("done"):
                final = event
        return final

    final = await anyio.to_thread.run_sync(run)
    if not final:
        return _err("optimisation produced no result")

    history = final.get("history", {}).get("cd_best", [])
    quality = config.FLOW_MODEL_QUALITY["unet-phys"]
    return _ok(
        {
            "cd_initial": round(history[0], 4) if history else None,
            "cd_final": round(final["cd_surrogate_final"], 4),
            "reduction_pct": (
                round(100.0 * (history[0] - final["cd_surrogate_final"]) / abs(history[0]), 2)
                if history and abs(history[0]) > 1e-12
                else None
            ),
            "final_coeffs": [round(c, 5) for c in final["final_coeffs"]],
            "cd_error_pct_expected": quality["cd_mape_pct"],
            "reynolds": reynolds,
            "iters": iters,
            "convergence_cd_best": _downsample(history),
            "target_area": final.get("target_area"),
        }
    )


@tool(
    "flow_verify",
    "Run the real lattice-Boltzmann solver on a shape and compare its drag coefficient "
    "against the surrogate's prediction. Use this whenever the user asks to verify, check or "
    "confirm a Cd — and always offer it after flow_optimize. Slow on purpose (tens of seconds): "
    "it is the ground-truth solver, not the surrogate. Report both Cd values and the gap.",
    {
        "coeffs": list,
        "reynolds": float,
        "steps": int,
    },
)
async def flow_verify_tool(args: dict) -> dict:
    coeffs = args["coeffs"]
    reynolds = args.get("reynolds", 80.0)
    steps = int(args.get("steps", 9000))
    if problem := _check_coeffs(coeffs):
        return _err(problem)

    result = await anyio.to_thread.run_sync(
        lambda: flow_service.verify(coeffs, reynolds=reynolds, steps=steps)
    )
    return _ok(
        {
            "cd_lbm": round(result["cd_lbm"], 4),
            "cd_surrogate": round(result["cd_surrogate"], 4),
            "gap_pct": round(result["gap_pct"], 2),
            "wall_time_s": round(result["wall_time_s"], 1),
            "reynolds": reynolds,
            "lbm_steps": steps,
        }
    )


@tool(
    "flow_simulate",
    "Run an unsteady lattice-Boltzmann simulation of the flow around a shape and return a "
    "summary of the run (the animation frames themselves go to the Flow Lab, not to you). "
    "Use this when the user asks about wake behaviour, vortex shedding or time-dependent flow. "
    "Slow: tens of seconds.",
    {
        "coeffs": list,
        "reynolds": float,
        "n_frames": int,
    },
)
async def flow_simulate_tool(args: dict) -> dict:
    coeffs = args["coeffs"]
    reynolds = args.get("reynolds", 80.0)
    n_frames = int(args.get("n_frames", 40))
    if problem := _check_coeffs(coeffs):
        return _err(problem)

    def run() -> tuple[int, dict]:
        frames = 0
        final: dict = {}
        for event in simulate_streaming(
            flow_service.domain,
            flow_service.device,
            coeffs,
            flow_service.default_target_area,
            reynolds,
            n_frames=n_frames,
            steps=6000,
        ):
            if event.get("done"):
                final = event
            else:
                frames += 1
        return frames, final

    frames, final = await anyio.to_thread.run_sync(run)
    return _ok(
        {
            "frames_simulated": frames,
            "cd_lbm_time_averaged": round(final["cd_final"], 4) if "cd_final" in final else None,
            "reynolds": reynolds,
            "note": "Instantaneous velocity and vorticity frames were streamed to the Flow Lab viewer.",
        }
    )


@tool(
    "stress_predict",
    "Predict the von Mises stress field and its peak for a perforated plate, given 2-4 holes "
    "as [x, y, r] triples and a load angle in degrees. Use this whenever the user asks about "
    "stress, von Mises, or where a plate will fail. Always report the expected error percentage "
    "alongside the peak — the peak is the hardest quantity for this model to get right.",
    {
        "holes": list,
        "load_angle_deg": float,
    },
)
async def stress_predict_tool(args: dict) -> dict:
    holes = args["holes"]
    load_angle_deg = args.get("load_angle_deg", -90.0)
    try:
        result = await anyio.to_thread.run_sync(
            lambda: stress_service.predict(holes, load_angle_deg, "gnn-mp10-glob")
        )
    except ValueError as e:
        return _err(str(e))
    return _ok(
        {
            "peak_vm_surrogate": round(result["peak_vm_surrogate"], 4),
            "peak_error_pct_expected": result["peak_error_pct_expected"],
            "peak_location": [round(v, 4) for v in result["peak_location"]],
            "peak_vm_coarse_fem": round(result["peak_vm_coarse_fem"], 4),
            "load_angle_deg": load_angle_deg,
            "holes": holes,
        }
    )


@tool(
    "stress_optimize",
    "Search for a hole layout that lowers the peak von Mises stress, keeping the removed area "
    "constant. Use this when the user asks to optimise, improve or de-stress a perforated plate. "
    "Takes a few seconds. This is a randomised local search over the surrogate, not a gradient "
    "descent — say so if the user asks how it works.",
    {
        "holes": list,
        "load_angle_deg": float,
        "n_iters": int,
    },
)
async def stress_optimize_tool(args: dict) -> dict:
    holes = args["holes"]
    load_angle_deg = args.get("load_angle_deg", -90.0)
    n_iters = int(args.get("n_iters", 30))

    def run() -> dict:
        final: dict = {}
        for event in stress_service.optimise_streaming(holes, load_angle_deg, "gnn-mp10-glob", n_iters, 0):
            if event.get("done"):
                final = event
        return final

    try:
        final = await anyio.to_thread.run_sync(run)
    except ValueError as e:
        return _err(str(e))
    if not final:
        return _err("optimisation produced no result")

    initial, best = final["peak_vm_initial"], final["peak_vm_final"]
    return _ok(
        {
            "peak_vm_initial": round(initial, 4),
            "peak_vm_final": round(best, 4),
            "reduction_pct": round(100.0 * (initial - best) / abs(initial), 2) if abs(initial) > 1e-12 else None,
            "final_holes": [[round(v, 5) for v in h] for h in final["final_holes"]],
            "peak_error_pct_expected": config.STRESS_MODEL_QUALITY["gnn-mp10-glob"]["peak_err_pct"],
            "load_angle_deg": load_angle_deg,
            "n_iters": n_iters,
        }
    )


@tool(
    "get_model_card",
    "Return the known accuracy and speed numbers for the flow and stress surrogate models "
    "(from RESULTS.md). Use this when the user asks how accurate or how fast a model is.",
    {"model": str},
)
async def get_model_card_tool(args: dict) -> dict:
    key = args.get("model", "unet-phys")
    quality = config.FLOW_MODEL_QUALITY.get(key) or config.STRESS_MODEL_QUALITY.get(key)
    if quality is None:
        known = list(config.FLOW_MODEL_QUALITY) + list(config.STRESS_MODEL_QUALITY)
        return _err(f"Unknown model '{key}'. Known models: {known}")
    return _ok({"model": key, **quality})


@tool(
    "render_report",
    "Turn an optimisation result into a Markdown engineering report. `project` is 'flow' or "
    "'stress'; `context` is the result payload from flow_optimize or stress_optimize (add a "
    "`verify` key with the flow_verify result if you have one). Use this when the user asks for "
    "a report, a summary document or something to send to a colleague. Returns the Markdown "
    "itself — hand it back to the user as-is rather than rewriting it.",
    {
        "project": str,
        "context": dict,
    },
)
async def render_report_tool(args: dict) -> dict:
    try:
        markdown = render_report(args.get("project", ""), args.get("context") or {})
    except ValueError as e:
        return _err(str(e))
    return {"content": [{"type": "text", "text": markdown}]}


ALL_TOOLS = [
    flow_predict_tool,
    flow_optimize_tool,
    flow_verify_tool,
    flow_simulate_tool,
    stress_predict_tool,
    stress_optimize_tool,
    get_model_card_tool,
    render_report_tool,
]
