# ============================================================
# TELEMETRY FORECASTING API TESTS
# ============================================================

import numpy as np
import pytest

from fastapi.testclient import TestClient

from api.main import app


class DummyTelemetryModel:

    def predict(
        self,
        features
    ):

        return (
            features[
                "estimated_cost_index"
            ].to_numpy(
                dtype="float64"
            )
            + 1.0
        )


@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:

        yield test_client


@pytest.fixture
def telemetry_payload():

    return {
        "previous_hour": {
            "time_bucket": (
                "2026-08-29T04:00:00Z"
            ),
            "cpu_mean": 20.0,
            "memory_mean_gb": 4.0,
            "disk_activity_kbps": 10.0,
            "network_activity_kbps": 5.0,
            "active_vms": 30.0,
            "resource_cost_index": 22.0,
            "estimated_cost_index": 24.0
        },
        "current_hour": {
            "time_bucket": (
                "2026-08-29T05:00:00Z"
            ),
            "cpu_mean": 22.0,
            "memory_mean_gb": 4.2,
            "disk_activity_kbps": 12.0,
            "network_activity_kbps": 6.0,
            "active_vms": 31.0,
            "resource_cost_index": 23.0,
            "estimated_cost_index": 25.0
        }
    }


def test_v2_prediction_returns_503_without_model(
    client,
    telemetry_payload,
    monkeypatch
):

    monkeypatch.setattr(
        app.state,
        "telemetry_model",
        None
    )

    monkeypatch.setattr(
        app.state,
        "telemetry_manifest",
        None
    )

    response = client.post(
        "/v2/predict",
        json=telemetry_payload
    )

    assert response.status_code == 503

    assert (
        "not deployed"
        in response.json()["detail"]
    )


def test_valid_v2_prediction(
    client,
    telemetry_payload,
    monkeypatch
):

    monkeypatch.setattr(
        app.state,
        "telemetry_model",
        DummyTelemetryModel()
    )

    monkeypatch.setattr(
        app.state,
        "telemetry_manifest",
        {
            "model_version": "2",
            "alias": "challenger"
        }
    )

    response = client.post(
        "/v2/predict",
        json=telemetry_payload
    )

    body = response.json()

    assert response.status_code == 200

    assert body[
        "predicted_next_hour_cost"
    ] == pytest.approx(26.0)

    assert body[
        "forecast_horizon"
    ] == "1_hour"

    assert body[
        "feature_count"
    ] == 22

    assert body[
        "model_version"
    ] == "2"

    assert body[
        "model_alias"
    ] == "challenger"


def test_v2_rejects_nonconsecutive_hours(
    client,
    telemetry_payload,
    monkeypatch
):

    monkeypatch.setattr(
        app.state,
        "telemetry_model",
        DummyTelemetryModel()
    )

    monkeypatch.setattr(
        app.state,
        "telemetry_manifest",
        {
            "model_version": "2",
            "alias": "challenger"
        }
    )

    telemetry_payload[
        "current_hour"
    ][
        "time_bucket"
    ] = "2026-08-29T07:00:00Z"

    response = client.post(
        "/v2/predict",
        json=telemetry_payload
    )

    assert response.status_code == 422

    assert (
        "exactly one hour apart"
        in response.json()["detail"]
    )


def test_v2_rejects_negative_telemetry(
    client,
    telemetry_payload
):

    telemetry_payload[
        "current_hour"
    ][
        "cpu_mean"
    ] = -1.0

    response = client.post(
        "/v2/predict",
        json=telemetry_payload
    )

    assert response.status_code == 422