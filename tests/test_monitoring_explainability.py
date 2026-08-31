from src.monitoring.explainability import explain_model


def test_explain_model_for_persistence_baseline_is_not_supported():
    result = explain_model({"model_type": "persistence_baseline"}, [[1.0], [2.0]])

    assert result["supported"] is False
    assert "Persistence baseline" in result["reason"]
