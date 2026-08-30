# ============================================================
# FINOPS TELEMETRY FORECASTING API V2
# ============================================================

from datetime import datetime
from typing import Literal

import numpy as np

from fastapi import (
    APIRouter,
    HTTPException,
    Request
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)

from src.retraining.inference import (
    build_latest_inference_features
)


telemetry_router = APIRouter(
    prefix="/v2",
    tags=["Telemetry Forecasting"]
)


class HourlyTelemetry(BaseModel):

    time_bucket: datetime

    cpu_mean: float = Field(
        ...,
        ge=0
    )

    memory_mean_gb: float = Field(
        ...,
        ge=0
    )

    disk_activity_kbps: float = Field(
        ...,
        ge=0
    )

    network_activity_kbps: float = Field(
        ...,
        ge=0
    )

    active_vms: float = Field(
        ...,
        ge=0
    )

    resource_cost_index: float = Field(
        ...,
        ge=0
    )

    estimated_cost_index: float = Field(
        ...,
        ge=0
    )


class TelemetryPredictionRequest(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    previous_hour: HourlyTelemetry

    current_hour: HourlyTelemetry


class TelemetryPredictionResponse(BaseModel):

    predicted_next_hour_cost: float

    forecast_horizon: Literal["1_hour"]

    feature_count: Literal[22]

    model_version: str

    model_alias: str


@telemetry_router.post(
    "/predict",
    response_model=(
        TelemetryPredictionResponse
    )
)
def predict_from_telemetry(
    prediction_request: (
        TelemetryPredictionRequest
    ),
    request: Request
):

    telemetry_model = getattr(
        request.app.state,
        "telemetry_model",
        None
    )

    telemetry_manifest = getattr(
        request.app.state,
        "telemetry_manifest",
        None
    )

    if (
        telemetry_model is None
        or telemetry_manifest is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "The 22-feature telemetry model "
                "is not deployed."
            )
        )

    try:

        feature_frame = (
            build_latest_inference_features(
                previous_hour=(
                    prediction_request
                    .previous_hour
                    .model_dump()
                ),
                current_hour=(
                    prediction_request
                    .current_hour
                    .model_dump()
                )
            )
        )

        predictions = np.asarray(
            telemetry_model.predict(
                feature_frame
            ),
            dtype="float64"
        ).reshape(-1)

        if (
            predictions.size != 1
            or not np.isfinite(
                predictions[0]
            )
            or predictions[0] < 0
        ):

            raise ValueError(
                "Telemetry model returned "
                "an invalid prediction."
            )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    return TelemetryPredictionResponse(
        predicted_next_hour_cost=float(
            predictions[0]
        ),
        forecast_horizon="1_hour",
        feature_count=22,
        model_version=str(
            telemetry_manifest[
                "model_version"
            ]
        ),
        model_alias=(
            telemetry_manifest[
                "alias"
            ]
        )
    )