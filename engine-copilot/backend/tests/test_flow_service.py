from engine_copilot.flow_service import flow_service


def test_predict_returns_physical_cd_and_field_shape():
    result = flow_service.predict([0.05, -0.02, 0.03, 0.01, -0.04, 0.0], reynolds=80.0, model_key="unet-phys")
    assert 0.5 <= result["cd_surrogate"] <= 5.0
    u = result["fields"]["u"]
    assert len(u) == flow_service.domain.ny // 2
    assert len(u[0]) == flow_service.domain.nx // 2
    assert result["latency_ms"] > 0
