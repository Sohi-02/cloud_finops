# ============================================================
# FINOPS FASTAPI MODEL SERVING APPLICATION
# ============================================================

import json
import logging
import os

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from api.telemetry import (
    telemetry_router
)
from src.monitoring.data_quality import (
    validate_input_features
)
from src.monitoring.drift import (
    calculate_profile_drift_report,
    load_reference_profile
)
from src.monitoring.performance import (
    calculate_performance_metrics
)
from src.retraining.contract import (
    evaluate_model_contract
)
from src.retraining.inference import (
    build_latest_inference_features
)
from src.retraining.pipeline import (
    FINOPS_FEATURE_COLUMNS
)
from src.retraining.trigger import (
    evaluate_retraining_trigger
)


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logger = logging.getLogger(
    "finops_api"
)


# ------------------------------------------------------------
# Optional MongoDB monitoring storage
# ------------------------------------------------------------

try:

    from src.monitoring.storage import (
        MongoPredictionStore
    )

except Exception as import_error:

    MongoPredictionStore = None

    logger.warning(
        "Monitoring storage import failed: %s",
        import_error
    )


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# ------------------------------------------------------------
# Current one-feature production champion
# ------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion"
)

MODEL_PATH = Path(
    os.getenv(
        "FINOPS_MODEL_PATH",
        str(DEFAULT_MODEL_PATH)
    )
).resolve()

MODEL_FILE_PATH = (
    MODEL_PATH
    / "MLmodel"
)

MANIFEST_PATH = (
    MODEL_PATH
    / "deployment_manifest.json"
)

REFERENCE_PROFILE_PATH = (
    MODEL_PATH
    / "reference_profile.json"
)


# ============================================================
# REQUEST AND RESPONSE SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "estimated_cost_index": 24.397998
            }
        }
    )

    estimated_cost_index: float = Field(
        ...,
        ge=0,
        description=(
            "Current-hour estimated cloud cost index."
        )
    )


class PredictionResponse(BaseModel):

    prediction_id: str

    prediction_logged: bool

    current_hour_cost: float

    predicted_next_hour_cost: float

    forecast_horizon: Literal["1_hour"]

    registered_model: str

    model_alias: str

    model_version: str

    prediction_timestamp_utc: str

    data_quality_passed: bool


class ActualCostRequest(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": (
                    "550e8400-e29b-41d4-a716-446655440000"
                ),
                "actual_next_hour_cost": 30.0
            }
        }
    )

    prediction_id: UUID

    actual_next_hour_cost: float = Field(
        ...,
        ge=0,
        description=(
            "Observed next-hour cloud cost."
        )
    )


class ActualCostResponse(BaseModel):

    prediction_id: str

    predicted_next_hour_cost: float

    actual_next_hour_cost: float

    absolute_error: float

    status: Literal["completed"]


class BatchPredictionItem(BaseModel):

    current_hour_cost: float

    predicted_next_hour_cost: float

    model_alias: str

    model_version: str

    data_quality_passed: bool


class BatchPredictionResponse(BaseModel):

    count: int

    results: list[BatchPredictionItem]


class RawHourlyInput(BaseModel):

    time_bucket: datetime

    cpu_mean: float = Field(..., ge=0)

    memory_mean_gb: float = Field(..., ge=0)

    disk_activity_kbps: float = Field(..., ge=0)

    network_activity_kbps: float = Field(..., ge=0)

    active_vms: float = Field(..., ge=0)

    resource_cost_index: float = Field(..., ge=0)

    estimated_cost_index: float = Field(..., ge=0)


class RawHourlyPredictionRequest(BaseModel):

    previous_hour: RawHourlyInput

    current_hour: RawHourlyInput


class RawHourlyPredictionResponse(BaseModel):

    predicted_next_hour_cost: float

    forecast_horizon: Literal["1_hour"]

    feature_count: Literal[22]

    model_alias: str

    model_version: str


class HealthResponse(BaseModel):

    status: Literal["healthy"]

    model_loaded: bool

    monitoring_connected: bool

    drift_profile_loaded: bool

    telemetry_model_loaded: bool

    registered_model: str

    model_alias: str

    timestamp_utc: str


class PerformanceResponse(BaseModel):

    sample_count: int

    mae: Optional[float]

    rmse: Optional[float]

    bias: Optional[float]

    mape_percent: Optional[float]

    baseline_mae: float

    mae_limit: float

    degradation_percent: Optional[float]

    performance_status: Literal[
        "insufficient_data",
        "stable",
        "degraded"
    ]

    retraining_candidate: bool


class DriftResponse(BaseModel):

    feature: str

    production_sample_count: int

    reference_sample_count: int

    minimum_required_samples: int

    moderate_threshold: float

    significant_threshold: float

    psi: Optional[float]

    status: Literal[
        "insufficient_data",
        "stable",
        "moderate",
        "significant"
    ]

    drift_detected: bool

    retraining_candidate: bool

    message: str

    production_bin_counts: Optional[
        list[int]
    ] = None

    production_bin_proportions: Optional[
        list[float]
    ] = None

    expected_bin_proportions: Optional[
        list[float]
    ] = None

    psi_by_bin: Optional[
        list[float]
    ] = None


class RetrainingStatusResponse(BaseModel):

    trigger_retraining: bool

    decision: Literal[
        "wait_for_data",
        "no_retraining",
        "monitor_closely",
        "trigger_retraining"
    ]

    drift_status: Literal[
        "unavailable",
        "insufficient_data",
        "stable",
        "moderate",
        "significant"
    ]

    performance_status: Literal[
        "unavailable",
        "insufficient_data",
        "stable",
        "degraded"
    ]

    trigger_reasons: list[str]

    observations: list[str]

    message: str

    evaluated_at_utc: str

    drift_report: DriftResponse

    performance_report: PerformanceResponse


# ============================================================
# APPLICATION STARTUP AND SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # 1. Verify required champion files
    # --------------------------------------------------------

    if not MODEL_FILE_PATH.exists():

        raise FileNotFoundError(
            "MLflow champion model file not found at: "
            f"{MODEL_FILE_PATH}"
        )

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            "Deployment manifest not found at: "
            f"{MANIFEST_PATH}"
        )

    if not REFERENCE_PROFILE_PATH.exists():

        raise FileNotFoundError(
            "Drift reference profile not found at: "
            f"{REFERENCE_PROFILE_PATH}"
        )

    # --------------------------------------------------------
    # 2. Load champion deployment manifest
    # --------------------------------------------------------

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8"
    ) as manifest_file:

        manifest = json.load(
            manifest_file
        )

    # --------------------------------------------------------
    # 3. Load champion drift reference profile
    # --------------------------------------------------------

    reference_profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    if (
        reference_profile["registered_model"]
        != manifest["registered_model"]
    ):

        raise ValueError(
            "Reference profile registered-model name "
            "does not match the deployed model."
        )

    if (
        str(
            reference_profile[
                "model_version"
            ]
        )
        != str(
            manifest[
                "model_version"
            ]
        )
    ):

        raise ValueError(
            "Reference profile model version does not "
            "match the deployed model version."
        )

    # --------------------------------------------------------
    # 4. Load one-feature champion
    # --------------------------------------------------------

    champion_model = (
        mlflow.pyfunc.load_model(
            str(MODEL_PATH)
        )
    )

    app.state.model = (
        champion_model
    )

    app.state.manifest = (
        manifest
    )

    app.state.reference_profile = (
        reference_profile
    )

    # --------------------------------------------------------
    # 5. Initialize optional 22-feature telemetry model
    # --------------------------------------------------------

    app.state.telemetry_model = None

    app.state.telemetry_manifest = None

    app.state.telemetry_contract = None

    telemetry_model_path_value = (
        os.getenv(
            "FINOPS_TELEMETRY_MODEL_PATH"
        )
    )

    if telemetry_model_path_value:

        telemetry_model_path = Path(
            telemetry_model_path_value
        ).resolve()

        telemetry_mlmodel_path = (
            telemetry_model_path
            / "MLmodel"
        )

        telemetry_manifest_path = (
            telemetry_model_path
            / "deployment_manifest.json"
        )

        if not telemetry_mlmodel_path.exists():

            raise FileNotFoundError(
                "Telemetry MLflow model was not "
                "found at: "
                f"{telemetry_mlmodel_path}"
            )

        if not telemetry_manifest_path.exists():

            raise FileNotFoundError(
                "Telemetry deployment manifest was "
                "not found at: "
                f"{telemetry_manifest_path}"
            )

        with telemetry_manifest_path.open(
            "r",
            encoding="utf-8"
        ) as telemetry_manifest_file:

            telemetry_manifest = json.load(
                telemetry_manifest_file
            )

        telemetry_contract = (
            evaluate_model_contract(
                deployment_manifest=(
                    telemetry_manifest
                ),
                candidate_feature_columns=(
                    FINOPS_FEATURE_COLUMNS
                ),
                candidate_output=(
                    "predicted_next_hour_cost"
                )
            )
        )

        if not telemetry_contract[
            "contract_compatible"
        ]:

            raise ValueError(
                "Telemetry model deployment manifest "
                "does not match the required "
                "22-feature inference contract."
            )

        telemetry_model = (
            mlflow.pyfunc.load_model(
                str(telemetry_model_path)
            )
        )

        app.state.telemetry_model = (
            telemetry_model
        )

        app.state.telemetry_manifest = (
            telemetry_manifest
        )

        app.state.telemetry_contract = (
            telemetry_contract
        )

        print(
            "Telemetry challenger model loaded."
        )

        print(
            "Telemetry model path:",
            telemetry_model_path
        )

        print(
            "Telemetry model version:",
            telemetry_manifest[
                "model_version"
            ]
        )

    else:

        logger.warning(
            "FINOPS_TELEMETRY_MODEL_PATH is not "
            "configured. The /v2/predict endpoint "
            "will return HTTP 503."
        )

    # --------------------------------------------------------
    # 6. Connect monitoring storage
    # --------------------------------------------------------

    app.state.prediction_store = None

    mongodb_uri = os.getenv(
        "MONGODB_URI"
    )

    mongodb_database = os.getenv(
        "MONGODB_DATABASE",
        "finops_monitoring"
    )

    if (
        mongodb_uri
        and MongoPredictionStore
        is not None
    ):

        try:

            app.state.prediction_store = (
                MongoPredictionStore(
                    mongodb_uri=mongodb_uri,
                    database_name=(
                        mongodb_database
                    )
                )
            )

            print(
                "MongoDB monitoring connected."
            )

        except Exception:

            logger.exception(
                "MongoDB connection failed. "
                "Predictions will continue without "
                "monitoring storage."
            )

    elif (
        mongodb_uri
        and MongoPredictionStore
        is None
    ):

        logger.warning(
            "MONGODB_URI is configured, but "
            "monitoring storage could not be "
            "imported. Prediction storage "
            "is disabled."
        )

    else:

        logger.warning(
            "MONGODB_URI is not configured. "
            "Prediction storage is disabled."
        )

    # --------------------------------------------------------
    # 7. Startup information
    # --------------------------------------------------------

    print(
        "FinOps champion model loaded."
    )

    print(
        "Champion model path:",
        MODEL_PATH
    )

    print(
        "Champion model version:",
        manifest[
            "model_version"
        ]
    )

    print(
        "Drift reference profile loaded."
    )

    print(
        "Reference training samples:",
        reference_profile[
            "features"
        ][
            "estimated_cost_index"
        ][
            "training_sample_count"
        ]
    )

    yield

    # --------------------------------------------------------
    # 8. Application shutdown
    # --------------------------------------------------------

    prediction_store = getattr(
        app.state,
        "prediction_store",
        None
    )

    if prediction_store is not None:

        prediction_store.close()

    app.state.model = None

    app.state.manifest = None

    app.state.reference_profile = None

    app.state.prediction_store = None

    app.state.telemetry_model = None

    app.state.telemetry_manifest = None

    app.state.telemetry_contract = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=(
        "FinOps Cloud Cost Forecasting API"
    ),
    description=(
        "Forecasts next-hour cloud cost and "
        "monitors production data quality, "
        "model performance, input drift, and "
        "retraining requirements. An optional "
        "22-feature telemetry endpoint is "
        "available under /v2."
    ),
    version="1.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


app.include_router(
    telemetry_router
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": (
            "FinOps Cloud Cost Forecasting API"
        ),
        "status": "running",
        "api_version": app.version,
        "documentation": "/docs",
        "endpoints": {
            "baseline_prediction": (
                "/predict"
            ),
            "telemetry_prediction": (
                "/v2/predict"
            ),
            "health": "/health",
            "performance": "/performance",
            "drift": "/drift",
            "retraining_status": (
                "/retraining-status"
            )
        }
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health(request: Request):

    manifest = (
        request.app.state.manifest
    )

    model = (
        request.app.state.model
    )

    reference_profile = getattr(
        request.app.state,
        "reference_profile",
        None
    )

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

    telemetry_model = getattr(
        request.app.state,
        "telemetry_model",
        None
    )

    return HealthResponse(
        status="healthy",
        model_loaded=(
            model is not None
        ),
        monitoring_connected=(
            prediction_store is not None
        ),
        drift_profile_loaded=(
            reference_profile is not None
        ),
        telemetry_model_loaded=(
            telemetry_model is not None
        ),
        registered_model=manifest[
            "registered_model"
        ],
        model_alias=manifest[
            "alias"
        ],
        timestamp_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )


# ============================================================
# MODEL INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info(request: Request):

    manifest = (
        request.app.state.manifest
    )

    reference_profile = (
        request.app.state.reference_profile
    )

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

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

    return {
        "registered_model": manifest[
            "registered_model"
        ],
        "alias": manifest[
            "alias"
        ],
        "model_version": manifest[
            "model_version"
        ],
        "source_run_id": manifest[
            "source_run_id"
        ],
        "model_role": manifest[
            "model_role"
        ],
        "forecast_horizon": manifest[
            "forecast_horizon"
        ],
        "input_schema": manifest[
            "input_schema"
        ],
        "output": manifest[
            "output"
        ],
        "production_test_mae": manifest[
            "production_test_mae"
        ],
        "monitoring_backend": "MongoDB",
        "monitoring_connected": (
            prediction_store is not None
        ),
        "drift_method": "PSI",
        "drift_feature": (
            "estimated_cost_index"
        ),
        "drift_minimum_samples": (
            reference_profile[
                "minimum_production_samples"
            ]
        ),
        "telemetry_endpoint": (
            "/v2/predict"
        ),
        "telemetry_model_loaded": (
            telemetry_model is not None
        ),
        "telemetry_model_version": (
            str(
                telemetry_manifest[
                    "model_version"
                ]
            )
            if telemetry_manifest
            is not None
            else None
        ),
        "telemetry_feature_count": (
            22
        )
    }


# ============================================================
# PERFORMANCE MONITORING ENDPOINT
# ============================================================

@app.get(
    "/performance",
    response_model=PerformanceResponse
)
def performance(request: Request):

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

    if prediction_store is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Monitoring storage is unavailable."
            )
        )

    manifest = (
        request.app.state.manifest
    )

    try:

        completed_predictions = (
            prediction_store
            .get_completed_predictions(
                limit=1000
            )
        )

        metrics = (
            calculate_performance_metrics(
                prediction_records=(
                    completed_predictions
                ),
                baseline_mae=float(
                    manifest[
                        "production_test_mae"
                    ]
                ),
                degradation_threshold_percent=20.0,
                minimum_samples=30
            )
        )

        return PerformanceResponse(
            **metrics
        )

    except Exception as error:

        logger.exception(
            "Performance calculation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Performance metrics could not "
                "be calculated."
            )
        ) from error


# ============================================================
# DRIFT MONITORING ENDPOINT
# ============================================================

@app.get(
    "/drift",
    response_model=DriftResponse
)
def drift(request: Request):

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

    if prediction_store is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Monitoring storage is unavailable."
            )
        )

    reference_profile = getattr(
        request.app.state,
        "reference_profile",
        None
    )

    if reference_profile is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Drift reference profile is "
                "unavailable."
            )
        )

    feature_name = (
        "estimated_cost_index"
    )

    try:

        model_version = str(
            reference_profile[
                "model_version"
            ]
        )

        production_values = (
            prediction_store
            .get_recent_feature_values(
                feature_name=feature_name,
                limit=1000,
                model_version=model_version
            )
        )

        drift_report = (
            calculate_profile_drift_report(
                reference_profile=(
                    reference_profile
                ),
                production_values=(
                    production_values
                ),
                feature_name=feature_name
            )
        )

        return DriftResponse(
            **drift_report
        )

    except Exception as error:

        logger.exception(
            "Drift calculation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Drift metrics could not "
                "be calculated."
            )
        ) from error


# ============================================================
# RETRAINING STATUS ENDPOINT
# ============================================================

@app.get(
    "/retraining-status",
    response_model=(
        RetrainingStatusResponse
    )
)
def retraining_status(
    request: Request
):
    """
    Combine drift and performance evidence into a
    read-only retraining recommendation.
    """

    try:

        drift_result = drift(
            request=request
        )

        performance_result = (
            performance(
                request=request
            )
        )

        drift_report = (
            drift_result.model_dump()
        )

        performance_report = (
            performance_result.model_dump()
        )

        decision = (
            evaluate_retraining_trigger(
                drift_report=drift_report,
                performance_report=(
                    performance_report
                )
            )
        )

        return RetrainingStatusResponse(
            **decision,
            evaluated_at_utc=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            drift_report=drift_result,
            performance_report=(
                performance_result
            )
        )

    except HTTPException:

        raise

    except Exception as error:

        logger.exception(
            "Retraining-status evaluation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Retraining status could not "
                "be evaluated."
            )
        ) from error


# ============================================================
# ACTUAL OUTCOME ENDPOINT
# ============================================================

@app.post(
    "/actual",
    response_model=ActualCostResponse
)
def record_actual(
    actual_request: ActualCostRequest,
    request: Request
):

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

    if prediction_store is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Monitoring storage is unavailable."
            )
        )

    try:

        result = (
            prediction_store
            .record_actual_cost(
                prediction_id=str(
                    actual_request.prediction_id
                ),
                actual_next_hour_cost=(
                    actual_request
                    .actual_next_hour_cost
                )
            )
        )

    except KeyError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=409,
            detail=str(error)
        ) from error

    except Exception as error:

        logger.exception(
            "Failed to record actual cost."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Actual cost could not "
                "be recorded."
            )
        ) from error

    return ActualCostResponse(
        prediction_id=result[
            "prediction_id"
        ],
        predicted_next_hour_cost=result[
            "predicted_next_hour_cost"
        ],
        actual_next_hour_cost=result[
            "actual_next_hour_cost"
        ],
        absolute_error=result[
            "absolute_error"
        ],
        status="completed"
    )

# ONE-FEATURE CHAMPION PREDICTION ENDPOINT


@app.post(
    "/predict/csv",
    response_model=BatchPredictionResponse
)
async def predict_csv(
    file: UploadFile,
    request: Request
):
    """Predict a batch of next-hour costs from a CSV file."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A CSV file is required."
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV uploads are supported."
        )

    try:
        content = await file.read()
        csv_text = content.decode("utf-8")
        batch_frame = pd.read_csv(
            pd.io.common.StringIO(csv_text)
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to parse CSV file: {error}"
        ) from error

    required_column = "estimated_cost_index"
    if required_column not in batch_frame.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV must contain the column "
                f"'{required_column}'."
            )
        )

    model = request.app.state.model
    manifest = request.app.state.manifest

    results: list[BatchPredictionItem] = []
    for raw_value in batch_frame[required_column].tolist():
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    "All values in the CSV must be numeric. "
                    f"Invalid value found: {raw_value!r}"
                )
            ) from error

        if numeric_value < 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "All values in the CSV must be non-negative. "
                    f"Invalid value found: {numeric_value}"
                )
            )

        input_features = {"estimated_cost_index": numeric_value}
        required_features = list(manifest["input_schema"].keys())

        data_quality_report = validate_input_features(
            input_features=input_features,
            required_features=required_features,
            feature_ranges={
                "estimated_cost_index": {
                    "minimum": 0.0,
                    "maximum": None
                }
            }
        )

        if not data_quality_report["passed"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Production input failed data-quality validation."
                    ),
                    "errors": data_quality_report["errors"],
                }
            )

        model_input = pd.DataFrame({
            "estimated_cost_index": pd.Series([numeric_value], dtype="float64")
        })

        model_output = np.asarray(
            model.predict(model_input)
        ).reshape(-1)

        predicted_cost = float(model_output[0])
        if not np.isfinite(predicted_cost) or predicted_cost < 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model returned an invalid prediction for a input value."
                )
            )

        results.append(
            BatchPredictionItem(
                current_hour_cost=numeric_value,
                predicted_next_hour_cost=predicted_cost,
                model_alias=manifest["alias"],
                model_version=str(manifest["model_version"]),
                data_quality_passed=data_quality_report["passed"],
            )
        )

    return BatchPredictionResponse(
        count=len(results),
        results=results,
    )


@app.post(
    "/predict/raw-hourly",
    response_model=RawHourlyPredictionResponse
)
def predict_raw_hourly(
    prediction_request: RawHourlyPredictionRequest,
    request: Request
):
    """Predict using two consecutive hourly telemetry rows."""
    model = request.app.state.model
    manifest = request.app.state.manifest

    try:
        feature_frame = build_latest_inference_features(
            previous_hour=prediction_request.previous_hour.model_dump(),
            current_hour=prediction_request.current_hour.model_dump(),
        )

        model_output = np.asarray(
            model.predict(feature_frame)
        ).reshape(-1)

        if model_output.size != 1:
            raise ValueError("Model returned an unexpected number of predictions.")

        predicted_cost = float(model_output[0])
        if not np.isfinite(predicted_cost) or predicted_cost < 0:
            raise ValueError("Model returned an invalid prediction.")

    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    return RawHourlyPredictionResponse(
        predicted_next_hour_cost=predicted_cost,
        forecast_horizon="1_hour",
        feature_count=22,
        model_alias=manifest["alias"],
        model_version=str(manifest["model_version"]),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    prediction_request: PredictionRequest,
    request: Request
):

    try:

        model = (
            request.app.state.model
        )

        manifest = (
            request.app.state.manifest
        )

        input_features = {
            "estimated_cost_index": float(
                prediction_request
                .estimated_cost_index
            )
        }

        required_features = list(
            manifest[
                "input_schema"
            ].keys()
        )

        data_quality_report = (
            validate_input_features(
                input_features=input_features,
                required_features=(
                    required_features
                ),
                feature_ranges={
                    "estimated_cost_index": {
                        "minimum": 0.0,
                        "maximum": None
                    }
                }
            )
        )

        if not data_quality_report[
            "passed"
        ]:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Production input failed "
                        "data-quality validation."
                    ),
                    "errors": (
                        data_quality_report[
                            "errors"
                        ]
                    )
                }
            )

        model_input = pd.DataFrame({
            "estimated_cost_index": pd.Series(
                [
                    prediction_request
                    .estimated_cost_index
                ],
                dtype="float64"
            )
        })

        model_output = np.asarray(
            model.predict(
                model_input
            )
        ).reshape(-1)

        if model_output.size != 1:

            raise ValueError(
                "Model returned an unexpected "
                "number of predictions."
            )

        predicted_cost = float(
            model_output[0]
        )

        if not np.isfinite(
            predicted_cost
        ):

            raise ValueError(
                "Model returned a non-finite "
                "prediction."
            )

        if predicted_cost < 0:

            raise ValueError(
                "Model returned a negative "
                "prediction."
            )

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {error}"
            )
        ) from error

    # --------------------------------------------------------
    # Monitoring identity
    # --------------------------------------------------------

    prediction_id = str(
        uuid4()
    )

    prediction_timestamp = (
        datetime.now(
            timezone.utc
        )
    )

   
    # Store prediction

    prediction_logged = False

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None
    )

    if prediction_store is not None:

        try:

            prediction_store.save_prediction(
                prediction_id=prediction_id,
                input_features=(
                    input_features
                ),
                predicted_next_hour_cost=(
                    predicted_cost
                ),
                registered_model=manifest[
                    "registered_model"
                ],
                model_alias=manifest[
                    "alias"
                ],
                model_version=manifest[
                    "model_version"
                ],
                prediction_timestamp_utc=(
                    prediction_timestamp
                ),
                data_quality_report=(
                    data_quality_report
                )
            )

            prediction_logged = True

        except Exception:

            logger.exception(
                "Prediction succeeded, but "
                "MongoDB logging failed."
            )

   
    # Response

    return PredictionResponse(
        prediction_id=prediction_id,
        prediction_logged=(
            prediction_logged
        ),
        current_hour_cost=(
            prediction_request
            .estimated_cost_index
        ),
        predicted_next_hour_cost=(
            predicted_cost
        ),
        forecast_horizon="1_hour",
        registered_model=manifest[
            "registered_model"
        ],
        model_alias=manifest[
            "alias"
        ],
        model_version=str(
            manifest[
                "model_version"
            ]
        ),
        prediction_timestamp_utc=(
            prediction_timestamp
            .isoformat()
        ),
        data_quality_passed=(
            data_quality_report[
                "passed"
            ]
        )
    )