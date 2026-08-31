import pandas as pd

from api.main import app
from fastapi.testclient import TestClient


def test_raw_hourly_input_prediction():
    with TestClient(app) as client:
        payload = {
            "previous_hour": {
                "time_bucket": "2026-08-29T04:00:00Z",
                "cpu_mean": 20.0,
                "memory_mean_gb": 4.0,
                "disk_activity_kbps": 10.0,
                "network_activity_kbps": 5.0,
                "active_vms": 30.0,
                "resource_cost_index": 22.0,
                "estimated_cost_index": 24.0,
            },
            "current_hour": {
                "time_bucket": "2026-08-29T05:00:00Z",
                "cpu_mean": 22.0,
                "memory_mean_gb": 4.2,
                "disk_activity_kbps": 12.0,
                "network_activity_kbps": 6.0,
                "active_vms": 31.0,
                "resource_cost_index": 23.0,
                "estimated_cost_index": 25.0,
            },
        }

        response = client.post("/predict/raw-hourly", json=payload)
        body = response.json()

        assert response.status_code == 200
        assert body["forecast_horizon"] == "1_hour"
        assert body["feature_count"] == 22
        assert body["model_alias"] == "champion"
        assert isinstance(body["predicted_next_hour_cost"], float)
