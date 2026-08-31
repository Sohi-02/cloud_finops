from src.monitoring.anomaly import detect_anomalies


def test_detect_anomalies_flags_high_cost_spike():
    values = [100.0, 101.0, 102.0, 105.0, 99.0, 600.0]

    result = detect_anomalies(values)

    assert result["flagged"] is True
    assert result["severity"] in {"moderate", "high", "critical"}
