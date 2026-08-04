import json

import anyio

from engine_copilot.tools import (
    ALL_TOOLS,
    flow_optimize_tool,
    flow_predict_tool,
    get_model_card_tool,
    render_report_tool,
    stress_predict_tool,
)

# @tool wraps each function into an SdkMcpTool dataclass; .handler is the
# original async function the Claude Agent SDK calls at runtime.

# flow_verify_tool and flow_simulate_tool are deliberately untested here: both run the
# real LBM solver for tens of seconds, which does not belong in a unit suite. Their
# service-layer counterparts are covered in test_flow_service.py.


def _payload(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def test_all_eight_tools_are_wired():
    assert {t.name for t in ALL_TOOLS} == {
        "flow_predict",
        "flow_optimize",
        "flow_verify",
        "flow_simulate",
        "stress_predict",
        "stress_optimize",
        "get_model_card",
        "render_report",
    }


def test_flow_predict_tool_returns_json_summary():
    result = anyio.run(flow_predict_tool.handler, {"coeffs": [0.05, -0.02, 0.03, 0.01, -0.04, 0.0], "reynolds": 80.0})
    text = result["content"][0]["text"]
    assert "cd_surrogate" in text
    assert "cd_error_pct_expected" in text


def test_flow_predict_tool_rejects_wrong_coeff_count():
    result = anyio.run(flow_predict_tool.handler, {"coeffs": [0.0, 0.0], "reynolds": 80.0})
    assert result.get("is_error") is True


def test_flow_optimize_tool_reduces_cd_and_stays_compact():
    result = anyio.run(
        flow_optimize_tool.handler,
        {"coeffs": [0.05, -0.02, 0.03, 0.01, -0.04, 0.0], "reynolds": 80.0, "iters": 12},
    )
    payload = _payload(result)
    assert payload["cd_final"] <= payload["cd_initial"]
    assert len(payload["final_coeffs"]) == 6
    # The convergence curve is downsampled, never handed over raw.
    assert len(payload["convergence_cd_best"]) <= 20
    assert payload["cd_error_pct_expected"] == 1.03
    # No field data must reach the model.
    assert "fields" not in payload and "boundary_xy" not in payload


def test_stress_predict_tool_returns_peak_without_the_mesh():
    result = anyio.run(
        stress_predict_tool.handler,
        {"holes": [[0.25, 0.25, 0.05], [0.75, 0.25, 0.05]], "load_angle_deg": -90.0},
    )
    payload = _payload(result)
    assert payload["peak_vm_surrogate"] > 0
    assert payload["peak_error_pct_expected"] == 6.56
    assert "mesh" not in payload and "field" not in payload


def test_stress_predict_tool_reports_invalid_layout_as_error():
    # Two identical holes overlap, which stress_service rejects.
    result = anyio.run(
        stress_predict_tool.handler,
        {"holes": [[0.5, 0.5, 0.05], [0.5, 0.5, 0.05]], "load_angle_deg": -90.0},
    )
    assert result.get("is_error") is True


def test_get_model_card_tool_known_model():
    result = anyio.run(get_model_card_tool.handler, {"model": "unet-phys"})
    assert "cd_mape_pct" in result["content"][0]["text"]


def test_get_model_card_tool_covers_stress_models_too():
    result = anyio.run(get_model_card_tool.handler, {"model": "gnn-mp10-glob"})
    assert "peak_err_pct" in result["content"][0]["text"]


def test_get_model_card_tool_unknown_model_is_error():
    result = anyio.run(get_model_card_tool.handler, {"model": "does-not-exist"})
    assert result.get("is_error") is True


def test_render_report_tool_returns_markdown():
    result = anyio.run(
        render_report_tool.handler,
        {"project": "flow", "context": {"history": {"cd_best": [2.0, 1.5]}, "reynolds": 80.0}},
    )
    assert result["content"][0]["text"].startswith("# Rapport d'optimisation")


def test_render_report_tool_rejects_unknown_project():
    result = anyio.run(render_report_tool.handler, {"project": "thermal", "context": {}})
    assert result.get("is_error") is True
