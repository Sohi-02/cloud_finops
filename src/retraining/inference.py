# ============================================================
# FINOPS 22-FEATURE INFERENCE BUILDER
# ============================================================

import math
from typing import Any

import numpy as np
import pandas as pd

from src.retraining.pipeline import (
    FINOPS_FEATURE_COLUMNS
)


TELEMETRY_NUMERIC_FIELDS = [
    "cpu_mean",
    "memory_mean_gb",
    "disk_activity_kbps",
    "network_activity_kbps",
    "active_vms",
    "resource_cost_index",
    "estimated_cost_index"
]


def _validate_telemetry_number(
    field_name: str,
    value: Any
) -> float:

    if isinstance(value, bool):

        raise ValueError(
            f"{field_name} cannot be boolean."
        )

    try:

        numeric_value = float(
            value
        )

    except (TypeError, ValueError) as error:

        raise ValueError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(
        numeric_value
    ):

        raise ValueError(
            f"{field_name} must be finite."
        )

    if numeric_value < 0:

        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return numeric_value


def build_latest_inference_features(
    previous_hour: dict[str, Any],
    current_hour: dict[str, Any]
) -> pd.DataFrame:
    """
    Build the exact 22 features required by notebook 04.

    Two consecutive hourly records are required because change
    features and cost_lag_1 depend on the previous hour.
    """

    if not isinstance(
        previous_hour,
        dict
    ):

        raise TypeError(
            "previous_hour must be a dictionary."
        )

    if not isinstance(
        current_hour,
        dict
    ):

        raise TypeError(
            "current_hour must be a dictionary."
        )

    previous_timestamp = pd.to_datetime(
        previous_hour.get("time_bucket"),
        utc=True,
        errors="coerce"
    )

    current_timestamp = pd.to_datetime(
        current_hour.get("time_bucket"),
        utc=True,
        errors="coerce"
    )

    if (
        pd.isna(previous_timestamp)
        or pd.isna(current_timestamp)
    ):

        raise ValueError(
            "Telemetry timestamps are invalid."
        )

    if (
        current_timestamp
        - previous_timestamp
        != pd.Timedelta(hours=1)
    ):

        raise ValueError(
            "Previous and current telemetry must "
            "be exactly one hour apart."
        )

    previous_values = {}

    current_values = {}

    for field_name in (
        TELEMETRY_NUMERIC_FIELDS
    ):

        previous_values[field_name] = (
            _validate_telemetry_number(
                field_name,
                previous_hour.get(
                    field_name
                )
            )
        )

        current_values[field_name] = (
            _validate_telemetry_number(
                field_name,
                current_hour.get(
                    field_name
                )
            )
        )

    hour = int(
        current_timestamp.hour
    )

    day_of_week = int(
        current_timestamp.dayofweek
    )

    feature_row = {
        "cpu_mean": (
            current_values["cpu_mean"]
        ),
        "memory_mean_gb": (
            current_values[
                "memory_mean_gb"
            ]
        ),
        "disk_activity_kbps": (
            current_values[
                "disk_activity_kbps"
            ]
        ),
        "network_activity_kbps": (
            current_values[
                "network_activity_kbps"
            ]
        ),
        "active_vms": (
            current_values["active_vms"]
        ),
        "resource_cost_index": (
            current_values[
                "resource_cost_index"
            ]
        ),
        "estimated_cost_index": (
            current_values[
                "estimated_cost_index"
            ]
        ),
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": int(
            current_timestamp.day
        ),
        "is_weekend": int(
            day_of_week >= 5
        ),
        "hour_sin": float(
            np.sin(
                2 * np.pi * hour / 24
            )
        ),
        "hour_cos": float(
            np.cos(
                2 * np.pi * hour / 24
            )
        ),
        "dow_sin": float(
            np.sin(
                2
                * np.pi
                * day_of_week
                / 7
            )
        ),
        "dow_cos": float(
            np.cos(
                2
                * np.pi
                * day_of_week
                / 7
            )
        ),
        "cpu_change": (
            current_values["cpu_mean"]
            - previous_values["cpu_mean"]
        ),
        "memory_change": (
            current_values[
                "memory_mean_gb"
            ]
            - previous_values[
                "memory_mean_gb"
            ]
        ),
        "disk_change": (
            current_values[
                "disk_activity_kbps"
            ]
            - previous_values[
                "disk_activity_kbps"
            ]
        ),
        "network_change": (
            current_values[
                "network_activity_kbps"
            ]
            - previous_values[
                "network_activity_kbps"
            ]
        ),
        "active_vms_change": (
            current_values["active_vms"]
            - previous_values["active_vms"]
        ),
        "resource_cost_change": (
            current_values[
                "resource_cost_index"
            ]
            - previous_values[
                "resource_cost_index"
            ]
        ),
        "cost_lag_1": (
            previous_values[
                "estimated_cost_index"
            ]
        )
    }

    feature_frame = pd.DataFrame(
        [feature_row],
        columns=FINOPS_FEATURE_COLUMNS
    ).astype(
        "float64"
    )

    if feature_frame.shape != (1, 22):

        raise RuntimeError(
            "Inference feature shape is invalid."
        )

    return feature_frame