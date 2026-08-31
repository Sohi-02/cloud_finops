# FINOPS STANDALONE RETRAINING JOB

import hashlib

from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from src.retraining.candidates import (
    train_and_select_challenger
)
from src.retraining.pipeline import (
    BASE_HOURLY_COLUMNS,
    build_forecasting_features,
    chronological_forecasting_split
)


def _utc_timestamp() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def _validate_minimum_rows(
    minimum_hourly_rows: int
) -> int:

    if (
        isinstance(minimum_hourly_rows, bool)
        or not isinstance(
            minimum_hourly_rows,
            Integral
        )
    ):

        raise TypeError(
            "minimum_hourly_rows must be an integer."
        )

    numeric_minimum = int(
        minimum_hourly_rows
    )

    if numeric_minimum < 3:

        raise ValueError(
            "minimum_hourly_rows must be at least 3."
        )

    return numeric_minimum


def _create_dataset_fingerprint(
    modeling_data: pd.DataFrame
) -> str:
    """
    Create an identifier for the exact dataset used by a job.
    """

    hashed_rows = (
        pd.util.hash_pandas_object(
            modeling_data,
            index=False
        )
        .to_numpy()
        .tobytes()
    )

    return hashlib.sha256(
        hashed_rows
    ).hexdigest()


def run_retraining_job(
    retraining_status: dict[str, Any],
    hourly_data_path,
    champion_model: Any = None,
    candidate_models: Optional[
        dict[str, Any]
    ] = None,
    dry_run: bool = True,
    minimum_hourly_rows: int = 202,
    minimum_improvement_percent: float = 2.0,
    minimum_evaluation_samples: int = 30
) -> dict[str, Any]:
    """
    Coordinate one retraining attempt.

    This function does not update MLflow aliases. A later registry
    stage will receive an approved candidate from this job.
    """

    job_id = str(
        uuid4()
    )

    started_at_utc = (
        _utc_timestamp()
    )

    if not isinstance(
        retraining_status,
        dict
    ):

        raise TypeError(
            "retraining_status must be a dictionary."
        )

    if not isinstance(
        dry_run,
        bool
    ):

        raise TypeError(
            "dry_run must be boolean."
        )

    numeric_minimum_rows = (
        _validate_minimum_rows(
            minimum_hourly_rows
        )
    )

    trigger_retraining = (
        retraining_status.get(
            "trigger_retraining"
        )
        is True
    )

    monitoring_decision = (
        retraining_status.get(
            "decision",
            "unavailable"
        )
    )

    # 1. Stop safely when monitoring does not trigger

    if not trigger_retraining:

        return {
            "candidate_model": None,
            "report": {
                "job_id": job_id,
                "job_status": "skipped",
                "monitoring_decision": (
                    monitoring_decision
                ),
                "trigger_retraining": False,
                "dry_run": dry_run,
                "reason": (
                    "Monitoring did not recommend "
                    "model retraining."
                ),
                "started_at_utc": (
                    started_at_utc
                ),
                "finished_at_utc": (
                    _utc_timestamp()
                )
            }
        }

    # 2. Validate the full hourly data source

    data_path = Path(
        hourly_data_path
    ).expanduser().resolve()

    if not data_path.exists():

        raise FileNotFoundError(
            "Retraining hourly dataset was not found: "
            f"{data_path}"
        )

    if not data_path.is_file():

        raise ValueError(
            "Retraining data path must point to a file."
        )

    hourly_data = pd.read_csv(
        data_path,
        parse_dates=[
            "time_bucket"
        ]
    )

    missing_base_columns = (
        set(BASE_HOURLY_COLUMNS)
        - set(hourly_data.columns)
    )

    if missing_base_columns:

        raise ValueError(
            "Retraining data is missing columns: "
            f"{sorted(missing_base_columns)}"
        )

    raw_hourly_rows = int(
        len(hourly_data)
    )

    # 3. Require enough full-feature hourly records

    if (
        raw_hourly_rows
        < numeric_minimum_rows
    ):

        return {
            "candidate_model": None,
            "report": {
                "job_id": job_id,
                "job_status": "blocked",
                "monitoring_decision": (
                    monitoring_decision
                ),
                "trigger_retraining": True,
                "dry_run": dry_run,
                "reason": (
                    "The full-feature hourly dataset "
                    "does not contain enough records."
                ),
                "raw_hourly_rows": (
                    raw_hourly_rows
                ),
                "minimum_hourly_rows": (
                    numeric_minimum_rows
                ),
                "started_at_utc": (
                    started_at_utc
                ),
                "finished_at_utc": (
                    _utc_timestamp()
                )
            }
        }

    if champion_model is None:

        raise ValueError(
            "champion_model is required when "
            "retraining is triggered."
        )

    # 4. Recreate the exact notebook feature pipeline

    modeling_data = (
        build_forecasting_features(
            hourly_data
        )
    )

    split_data = (
        chronological_forecasting_split(
            modeling_data
        )
    )

    dataset_fingerprint = (
        _create_dataset_fingerprint(
            modeling_data
        )
    )

    # 5. Train and evaluate challengers

    challenger_result = (
        train_and_select_challenger(
            split_data=split_data,
            champion_model=(
                champion_model
            ),
            candidate_models=(
                candidate_models
            ),
            minimum_improvement_percent=(
                minimum_improvement_percent
            ),
            minimum_evaluation_samples=(
                minimum_evaluation_samples
            )
        )
    )

    promotion_evaluation = (
        challenger_result[
            "promotion_evaluation"
        ]
    )

    promote_candidate = (
        promotion_evaluation[
            "promote_candidate"
        ]
    )

    if promote_candidate:

        job_status = "candidate_ready"

        reason = (
            "The challenger passed the promotion gate. "
            "Registry action remains pending."
        )

        candidate_model = (
            challenger_result[
                "selected_candidate_model"
            ]
        )

    else:

        job_status = "candidate_rejected"

        reason = (
            "The challenger did not pass the "
            "promotion gate."
        )

        candidate_model = None

    # 6. Return model object and serializable report

    return {
        "candidate_model": 
            candidate_model,
            "candidate_input_example": (
            split_data[
                "X_test"
        ]
        .head(5)
        .astype("float64")
        .copy()
        if candidate_model is not None
        else None
        ),
        "report": {
            "job_id": job_id,
            "job_status": job_status,
            "monitoring_decision": (
                monitoring_decision
            ),
            "trigger_retraining": True,
            "dry_run": dry_run,
            "registry_action": (
                "not_performed"
            ),
            "reason": reason,
            "source_data_path": str(
                data_path
            ),
            "raw_hourly_rows": (
                raw_hourly_rows
            ),
            "modeling_rows": int(
                len(modeling_data)
            ),
            "dataset_fingerprint": (
                dataset_fingerprint
            ),
            "selected_candidate_name": (
                challenger_result[
                    "selected_candidate_name"
                ]
            ),
            "validation_metrics": (
                challenger_result[
                    "validation_metrics"
                ]
            ),
            "candidate_failures": (
                challenger_result[
                    "candidate_failures"
                ]
            ),
            "candidate_test_metrics": (
                challenger_result[
                    "candidate_test_metrics"
                ]
            ),
            "candidate_input_example": None,
            "champion_test_metrics": (
                challenger_result[
                    "champion_test_metrics"
                ]
            ),
            "promotion_evaluation": (
                promotion_evaluation
            ),
            "training_report": (
                challenger_result[
                    "training_report"
                ]
            ),
            "split_report": (
                split_data["report"]
            ),
            "started_at_utc": (
                started_at_utc
            ),
            "finished_at_utc": (
                _utc_timestamp()
            )
        }
    }