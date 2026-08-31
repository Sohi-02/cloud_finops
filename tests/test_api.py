# ============================================================
# FASTAPI AUTOMATED TESTS
# ============================================================

import pytest

from fastapi.testclient import TestClient

from api.main import app


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:

        yield test_client


class FakeMonitoringStore:
    """
    Deterministic monitoring records for API tests.

    This fake store does not connect to or modify MongoDB.
    """

    def get_recent_feature_values(
        self,
        feature_name,
        limit=1000,
        model_version=None
    ):

        assert (
            feature_name
            == "estimated_cost_index"
        )

        return [
            20.0,
            24.0,
            28.0
        ]

    def get_completed_predictions(
        self,
        limit=1000
    ):

        return [
            {
                "status": "completed",
                "prediction": {
                    "predicted_next_hour_cost": (
                        24.397998
                    )
                },
                "actual": {
                    "actual_next_hour_cost": (
                        30.0
                    )
                }
            }
        ]


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root_endpoint(client):

    response = client.get("/")

    body = response.json()

    assert response.status_code == 200

    assert (
        body["status"]
        == "running"
    )

    assert (
        body["api_version"]
        == "1.4.0"
    )

    assert (
        body["documentation"]
        == "/docs"
    )

    assert (
        body["endpoints"][
            "baseline_prediction"
        ]
        == "/predict"
    )

    assert (
        body["endpoints"][
            "telemetry_prediction"
        ]
        == "/v2/predict"
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health_endpoint(client):

    response = client.get("/health")

    body = response.json()

    assert response.status_code == 200

    assert (
        body["status"]
        == "healthy"
    )

    assert (
        body["model_loaded"]
        is True
    )

    assert (
        body["drift_profile_loaded"]
        is True
    )

    assert (
        body["telemetry_model_loaded"]
        is False
    )

    assert (
        body["model_alias"]
        == "champion"
    )

    assert (
        body["registered_model"]
        == (
            "finops-cloud-cost-"
            "forecasting-clean-v1"
        )
    )


# ============================================================
# MODEL INFORMATION ENDPOINT
# ============================================================

def test_model_info_endpoint(client):

    response = client.get(
        "/model-info"
    )

    body = response.json()

    assert response.status_code == 200

    assert (
        body["alias"]
        == "champion"
    )

    assert (
        body["model_version"]
        == "1"
    )

    assert (
        body["forecast_horizon"]
        == "1_hour"
    )

    assert (
        body["drift_method"]
        == "PSI"
    )

    assert (
        body["drift_feature"]
        == "estimated_cost_index"
    )

    assert (
        "estimated_cost_index"
        in body["input_schema"]
    )

    assert (
        body["telemetry_endpoint"]
        == "/v2/predict"
    )

    assert (
        body["telemetry_feature_count"]
        == 22
    )

    assert (
        body["telemetry_model_loaded"]
        is False
    )

    assert (
        body["telemetry_model_version"]
        is None
    )


# ============================================================
# VALID PREDICTION
# ============================================================

def test_valid_prediction(client):

    current_cost = 24.397998

    response = client.post(
        "/predict",
        json={
            "estimated_cost_index": (
                current_cost
            )
        }
    )

    body = response.json()

    assert response.status_code == 200

    assert body[
        "current_hour_cost"
    ] == pytest.approx(
        current_cost,
        abs=1e-6
    )

    assert body[
        "predicted_next_hour_cost"
    ] == pytest.approx(
        current_cost,
        abs=1e-6
    )

    assert (
        body["forecast_horizon"]
        == "1_hour"
    )

    assert (
        body["model_alias"]
        == "champion"
    )

    assert (
        body["model_version"]
        == "1"
    )

    assert (
        body["data_quality_passed"]
        is True
    )

    assert isinstance(
        body["prediction_id"],
        str
    )

    assert len(
        body["prediction_id"]
    ) > 0

    assert isinstance(
        body["prediction_logged"],
        bool
    )


# ============================================================
# ZERO-COST PREDICTION
# ============================================================

def test_zero_cost_prediction(client):

    response = client.post(
        "/predict",
        json={
            "estimated_cost_index": 0.0
        }
    )

    body = response.json()

    assert response.status_code == 200

    assert body[
        "predicted_next_hour_cost"
    ] == pytest.approx(
        0.0,
        abs=1e-6
    )

    assert (
        body["data_quality_passed"]
        is True
    )


# ============================================================
# INVALID PREDICTION REQUESTS
# ============================================================

@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "estimated_cost_index": -10.0
        },
        {},
        {
            "estimated_cost_index": (
                "not-a-number"
            )
        },
        {
            "estimated_cost_index": None
        }
    ]
)
def test_invalid_prediction_requests(
    client,
    invalid_payload
):

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422


# ============================================================
# RETRAINING STATUS ENDPOINT
# ============================================================

def test_retraining_status_waits_for_data(
    client,
    monkeypatch
):

    fake_store = FakeMonitoringStore()

    monkeypatch.setattr(
        app.state,
        "prediction_store",
        fake_store
    )

    response = client.get(
        "/retraining-status"
    )

    body = response.json()

    assert response.status_code == 200

    assert (
        body["trigger_retraining"]
        is False
    )

    assert (
        body["decision"]
        == "wait_for_data"
    )

    assert (
        body["drift_status"]
        == "insufficient_data"
    )

    assert (
        body["performance_status"]
        == "insufficient_data"
    )

    assert (
        body["trigger_reasons"]
        == []
    )

    assert (
        body["drift_report"][
            "production_sample_count"
        ]
        == 3
    )

    assert (
        body["drift_report"]["psi"]
        is None
    )

    assert (
        body["drift_report"][
            "retraining_candidate"
        ]
        is False
    )

    assert (
        body["performance_report"][
            "sample_count"
        ]
        == 1
    )

    assert (
        body["performance_report"][
            "retraining_candidate"
        ]
        is False
    )

    assert isinstance(
        body["evaluated_at_utc"],
        str
    )

    assert len(
        body["evaluated_at_utc"]
    ) > 0


# ============================================================
# CSV BATCH PREDICTION
# ============================================================


def test_csv_batch_prediction(client):

    csv_payload = (
        "estimated_cost_index\n"
        "10\n"
        "20.5\n"
        "0\n"
    )

    response = client.post(
        "/predict/csv",
        files={
            "file": (
                "batch.csv",
                csv_payload,
                "text/csv"
            )
        }
    )

    body = response.json()

    assert response.status_code == 200
    assert body["count"] == 3
    assert body["results"][0]["current_hour_cost"] == pytest.approx(10.0, abs=1e-6)
    assert body["results"][0]["predicted_next_hour_cost"] == pytest.approx(10.0, abs=1e-6)
    assert body["results"][1]["predicted_next_hour_cost"] == pytest.approx(20.5, abs=1e-6)
    assert body["results"][2]["predicted_next_hour_cost"] == pytest.approx(0.0, abs=1e-6)