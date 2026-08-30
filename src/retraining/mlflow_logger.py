# ============================================================
# RETRAINED CHALLENGER MLFLOW LOGGER
# ============================================================

from typing import Any

import numpy as np
import pandas as pd


def detect_model_flavor(
    candidate_name: str
) -> str:

    normalized_name = (
        candidate_name
        .strip()
        .lower()
    )

    if "xgboost" in normalized_name:

        return "xgboost"

    if "lightgbm" in normalized_name:

        return "lightgbm"

    if "catboost" in normalized_name:

        return "catboost"

    return "sklearn"


def _log_model_with_correct_flavor(
    mlflow,
    model,
    model_flavor: str,
    signature,
    input_example
):

    if model_flavor == "xgboost":

        return mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

    if model_flavor == "lightgbm":

        return mlflow.lightgbm.log_model(
            lgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

    if model_flavor == "catboost":

        return mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

    return mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example
    )


def log_and_register_challenger(
    candidate_model: Any,
    candidate_name: str,
    model_input_example: pd.DataFrame,
    job_report: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str
) -> dict[str, Any]:
    """
    Log and register a challenger.

    This function assigns only the challenger alias.
    It never changes the champion alias.
    """

    if candidate_model is None:

        raise ValueError(
            "candidate_model is required."
        )

    if not isinstance(
        model_input_example,
        pd.DataFrame
    ):

        raise TypeError(
            "model_input_example must be "
            "a pandas DataFrame."
        )

    if model_input_example.empty:

        raise ValueError(
            "model_input_example cannot be empty."
        )

    if not isinstance(
        job_report,
        dict
    ):

        raise TypeError(
            "job_report must be a dictionary."
        )

    import mlflow

    from mlflow.models import (
        infer_signature
    )
    from mlflow.tracking import (
        MlflowClient
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_registry_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    model_flavor = detect_model_flavor(
        candidate_name
    )

    input_example = (
        model_input_example
        .astype("float64")
        .copy()
    )

    example_predictions = np.asarray(
        candidate_model.predict(
            input_example
        )
    ).reshape(-1)

    signature = infer_signature(
        input_example,
        example_predictions
    )

    candidate_metrics = (
        job_report.get(
            "candidate_test_metrics",
            {}
        )
    )

    promotion_evaluation = (
        job_report.get(
            "promotion_evaluation",
            {}
        )
    )

    with mlflow.start_run(
        run_name=(
            "automatic_retraining_"
            f"{candidate_name}"
        )
    ) as run:

        mlflow.log_params({
            "model_type": candidate_name,
            "training_type": (
                "monitoring_triggered_retraining"
            ),
            "feature_count": int(
                input_example.shape[1]
            ),
            "dataset_fingerprint": (
                job_report.get(
                    "dataset_fingerprint",
                    "unavailable"
                )
            ),
            "training_samples": (
                job_report.get(
                    "training_report",
                    {}
                ).get(
                    "retraining_samples",
                    0
                )
            )
        })

        numeric_metrics = {
            "candidate_test_mae": (
                candidate_metrics.get("mae")
            ),
            "candidate_test_rmse": (
                candidate_metrics.get("rmse")
            ),
            "candidate_test_bias": (
                candidate_metrics.get("bias")
            ),
            "champion_test_mae": (
                job_report.get(
                    "champion_test_metrics",
                    {}
                ).get("mae")
            ),
            "improvement_percent": (
                promotion_evaluation.get(
                    "improvement_percent"
                )
            )
        }

        mlflow.log_metrics({
            key: float(value)
            for key, value
            in numeric_metrics.items()
            if value is not None
        })

        mlflow.set_tags({
            "project": (
                "FinOps Cloud Cost Forecasting"
            ),
            "lifecycle_role": (
                "automatic_challenger"
            ),
            "deployment_status": (
                "pending_contract_validation"
            ),
            "promotion_decision": (
                "pending"
            ),
            "job_id": job_report.get(
                "job_id",
                "unavailable"
            )
        })

        _log_model_with_correct_flavor(
            mlflow=mlflow,
            model=candidate_model,
            model_flavor=model_flavor,
            signature=signature,
            input_example=input_example
        )

        mlflow.log_dict(
            job_report,
            "metadata/retraining_job_report.json"
        )

        run_id = run.info.run_id

    registered_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=registered_model_name
    )

    registry_client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri
    )

    candidate_version = str(
        registered_version.version
    )

    registry_client.set_registered_model_alias(
        name=registered_model_name,
        alias="challenger",
        version=candidate_version
    )

    registry_client.set_model_version_tag(
        name=registered_model_name,
        version=candidate_version,
        key="lifecycle_role",
        value="automatic_challenger"
    )

    registry_client.set_model_version_tag(
        name=registered_model_name,
        version=candidate_version,
        key="job_id",
        value=job_report.get(
            "job_id",
            "unavailable"
        )
    )

    return {
        "run_id": run_id,
        "registered_model": (
            registered_model_name
        ),
        "candidate_version": (
            candidate_version
        ),
        "alias": "challenger",
        "model_flavor": (
            model_flavor
        ),
        "champion_changed": False
    }