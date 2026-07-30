import anyio

from engine_copilot.tools import flow_predict_tool, get_model_card_tool

# @tool wraps each function into an SdkMcpTool dataclass; .handler is the
# original async function the Claude Agent SDK calls at runtime.


def test_flow_predict_tool_returns_json_summary():
    result = anyio.run(flow_predict_tool.handler, {"coeffs": [0.05, -0.02, 0.03, 0.01, -0.04, 0.0], "reynolds": 80.0})
    text = result["content"][0]["text"]
    assert "cd_surrogate" in text
    assert "cd_error_pct_expected" in text


def test_get_model_card_tool_known_model():
    result = anyio.run(get_model_card_tool.handler, {"model": "unet-phys"})
    assert "cd_mape_pct" in result["content"][0]["text"]


def test_get_model_card_tool_unknown_model_is_error():
    result = anyio.run(get_model_card_tool.handler, {"model": "does-not-exist"})
    assert result.get("is_error") is True
