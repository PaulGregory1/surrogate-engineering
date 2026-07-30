"""Tools exposed to the Claude Agent SDK — thin async wrappers around flow_service.

Each tool is a small, safe, read-only wrapper around our own FastAPI service layer
(no shell, no arbitrary file access) so bypassPermissions is a reasonable
permission_mode for this MCP server (see agent.py).
"""

from __future__ import annotations

import json

from claude_agent_sdk import tool

from . import config
from .flow_service import flow_service


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
    if len(coeffs) != 6:
        return {"content": [{"type": "text", "text": "coeffs must have exactly 6 floats: a2,b2,a3,b3,a4,b4"}], "is_error": True}
    result = flow_service.predict(coeffs, reynolds=reynolds, model_key="unet-phys")
    summary = {
        "cd_surrogate": round(result["cd_surrogate"], 4),
        "cd_error_pct_expected": result["cd_error_pct_expected"],
        "reynolds": reynolds,
        "coeffs": coeffs,
        "in_training_range": 10 <= reynolds <= 150,
    }
    return {"content": [{"type": "text", "text": json.dumps(summary)}]}


@tool(
    "get_model_card",
    "Return the known accuracy and speed numbers for the flow surrogate models "
    "(from RESULTS.md). Use this when the user asks how accurate or how fast a model is.",
    {"model": str},
)
async def get_model_card_tool(args: dict) -> dict:
    key = args.get("model", "unet-phys")
    quality = config.FLOW_MODEL_QUALITY.get(key)
    if quality is None:
        return {
            "content": [{"type": "text", "text": f"Unknown model '{key}'. Known models: {list(config.FLOW_MODEL_QUALITY)}"}],
            "is_error": True,
        }
    return {"content": [{"type": "text", "text": json.dumps({"model": key, **quality})}]}


ALL_TOOLS = [flow_predict_tool, get_model_card_tool]
