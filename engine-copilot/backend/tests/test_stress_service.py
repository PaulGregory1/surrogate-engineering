from engine_copilot.stress_service import stress_service


def test_predict_returns_positive_peak_and_non_nan_field():
    result = stress_service.predict([[0.5, 0.5, 0.12], [1.3, 0.4, 0.10]], load_angle_deg=-90.0, model_key="gnn-mp10-glob")
    assert result["peak_vm_surrogate"] > 0
    grid = result["field"]["vm"]
    values = [v for row in grid for v in row if v is not None]
    assert values, "grid should have at least some non-masked cells"
    assert all(v == v for v in values)  # no NaN survived the mask
    assert len(result["mesh"]["nodes"]) > 0
    assert result["latency_ms"] > 0
