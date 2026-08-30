# ============================================================
# FINOPS MODEL PERFORMANCE MONITORING
# ============================================================

import numpy as np


def calculate_performance_metrics(
    prediction_records,
    baseline_mae,
    degradation_threshold_percent=20.0,
    minimum_samples=30
):
    """
    Calculates aggregate performance metrics from completed
    MongoDB prediction records.

    Positive bias means the model underpredicts.
    Negative bias means the model overpredicts.
    """

    if baseline_mae <= 0:

        raise ValueError(
            "baseline_mae must be greater than zero."
        )

    completed_records = [
        record
        for record in prediction_records
        if (
            record.get("status") == "completed"
            and record.get("actual", {}).get(
                "actual_next_hour_cost"
            ) is not None
        )
    ]

    sample_count = len(
        completed_records
    )

    if sample_count == 0:

        return {
            "sample_count": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "mape_percent": None,
            "baseline_mae": float(
                baseline_mae
            ),
            "mae_limit": float(
                baseline_mae
                * (
                    1
                    + degradation_threshold_percent
                    / 100
                )
            ),
            "degradation_percent": None,
            "performance_status": (
                "insufficient_data"
            ),
            "retraining_candidate": False
        }

    predicted_values = np.asarray(
        [
            record["prediction"][
                "predicted_next_hour_cost"
            ]
            for record in completed_records
        ],
        dtype="float64"
    )

    actual_values = np.asarray(
        [
            record["actual"][
                "actual_next_hour_cost"
            ]
            for record in completed_records
        ],
        dtype="float64"
    )

    # Positive error means actual cost was higher than
    # the prediction, so the model underpredicted.
    errors = (
        actual_values - predicted_values
    )

    absolute_errors = np.abs(
        errors
    )

    mae = float(
        np.mean(absolute_errors)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(errors)
            )
        )
    )

    bias = float(
        np.mean(errors)
    )

    nonzero_actual_mask = (
        actual_values != 0
    )

    if nonzero_actual_mask.any():

        mape_percent = float(
            np.mean(
                absolute_errors[
                    nonzero_actual_mask
                ]
                / np.abs(
                    actual_values[
                        nonzero_actual_mask
                    ]
                )
            )
            * 100
        )

    else:

        mape_percent = None

    mae_limit = float(
        baseline_mae
        * (
            1
            + degradation_threshold_percent
            / 100
        )
    )

    degradation_percent = float(
        (
            mae - baseline_mae
        )
        / baseline_mae
        * 100
    )

    if sample_count < minimum_samples:

        performance_status = (
            "insufficient_data"
        )

        retraining_candidate = False

    elif mae > mae_limit:

        performance_status = "degraded"

        retraining_candidate = True

    else:

        performance_status = "stable"

        retraining_candidate = False

    return {
        "sample_count": sample_count,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape_percent": mape_percent,
        "baseline_mae": float(
            baseline_mae
        ),
        "mae_limit": mae_limit,
        "degradation_percent": (
            degradation_percent
        ),
        "performance_status": (
            performance_status
        ),
        "retraining_candidate": (
            retraining_candidate
        )
    }