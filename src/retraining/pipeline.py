# ============================================================
# FINOPS FORECASTING FEATURE PIPELINE
# ============================================================

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


BASE_HOURLY_COLUMNS = [
    "time_bucket",
    "cpu_mean",
    "memory_mean_gb",
    "disk_activity_kbps",
    "network_activity_kbps",
    "active_vms",
    "resource_cost_index",
    "estimated_cost_index"
]


NUMERIC_HOURLY_COLUMNS = [
    column
    for column in BASE_HOURLY_COLUMNS
    if column != "time_bucket"
]


FINOPS_FEATURE_COLUMNS = [
    "cpu_mean",
    "memory_mean_gb",
    "disk_activity_kbps",
    "network_activity_kbps",
    "active_vms",
    "resource_cost_index",
    "estimated_cost_index",
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "cpu_change",
    "memory_change",
    "disk_change",
    "network_change",
    "active_vms_change",
    "resource_cost_change",
    "cost_lag_1"
]


FINOPS_TARGET_COLUMN = (
    "target_next_hour_cost"
)


def validate_hourly_finops_data(
    hourly_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate and chronologically order hourly FinOps telemetry.

    The returned DataFrame contains exactly the eight columns
    used by notebook 04.
    """

    if not isinstance(
        hourly_data,
        pd.DataFrame
    ):

        raise TypeError(
            "hourly_data must be a pandas DataFrame."
        )

    if len(hourly_data) < 3:

        raise ValueError(
            "At least three hourly records are required."
        )

    missing_columns = (
        set(BASE_HOURLY_COLUMNS)
        - set(hourly_data.columns)
    )

    if missing_columns:

        raise ValueError(
            "Hourly data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    validated_data = (
        hourly_data[
            BASE_HOURLY_COLUMNS
        ]
        .copy()
    )

    validated_data["time_bucket"] = (
        pd.to_datetime(
            validated_data[
                "time_bucket"
            ],
            utc=True,
            errors="coerce"
        )
    )

    if validated_data[
        "time_bucket"
    ].isna().any():

        raise ValueError(
            "time_bucket contains invalid timestamps."
        )

    for column_name in NUMERIC_HOURLY_COLUMNS:

        validated_data[column_name] = (
            pd.to_numeric(
                validated_data[column_name],
                errors="coerce"
            )
        )

    if validated_data[
        NUMERIC_HOURLY_COLUMNS
    ].isna().any().any():

        raise ValueError(
            "Hourly numeric columns contain missing "
            "or non-numeric values."
        )

    numeric_values = validated_data[
        NUMERIC_HOURLY_COLUMNS
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "Hourly numeric columns contain "
            "non-finite values."
        )

    if (
        numeric_values < 0
    ).any():

        raise ValueError(
            "Hourly numeric columns cannot contain "
            "negative values."
        )

    validated_data = (
        validated_data
        .sort_values(
            by="time_bucket",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    if validated_data[
        "time_bucket"
    ].duplicated().any():

        raise ValueError(
            "Duplicate hourly timestamps were found."
        )

    time_differences = (
        validated_data[
            "time_bucket"
        ]
        .diff()
        .dropna()
    )

    expected_interval = pd.Timedelta(
        hours=1
    )

    if (
        time_differences
        != expected_interval
    ).any():

        raise ValueError(
            "Telemetry must contain continuous "
            "one-hour intervals."
        )

    return validated_data


def build_forecasting_features(
    hourly_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Reproduce notebook 04, Cell 4.

    The first record is removed because its lag/change values are
    unavailable. The last record is removed because its future
    target is unavailable.
    """

    features = validate_hourly_finops_data(
        hourly_data
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    features["hour"] = (
        features["time_bucket"].dt.hour
    )

    features["day_of_week"] = (
        features[
            "time_bucket"
        ].dt.dayofweek
    )

    features["day_of_month"] = (
        features[
            "time_bucket"
        ].dt.day
    )

    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical calendar encoding
    # --------------------------------------------------------

    features["hour_sin"] = np.sin(
        2
        * np.pi
        * features["hour"]
        / 24
    )

    features["hour_cos"] = np.cos(
        2
        * np.pi
        * features["hour"]
        / 24
    )

    features["dow_sin"] = np.sin(
        2
        * np.pi
        * features["day_of_week"]
        / 7
    )

    features["dow_cos"] = np.cos(
        2
        * np.pi
        * features["day_of_week"]
        / 7
    )

    # --------------------------------------------------------
    # Hour-to-hour change features
    # --------------------------------------------------------

    features["cpu_change"] = (
        features["cpu_mean"].diff()
    )

    features["memory_change"] = (
        features[
            "memory_mean_gb"
        ].diff()
    )

    features["disk_change"] = (
        features[
            "disk_activity_kbps"
        ].diff()
    )

    features["network_change"] = (
        features[
            "network_activity_kbps"
        ].diff()
    )

    features["active_vms_change"] = (
        features["active_vms"].diff()
    )

    features["resource_cost_change"] = (
        features[
            "resource_cost_index"
        ].diff()
    )

    # --------------------------------------------------------
    # Historical cost input
    # --------------------------------------------------------

    features["cost_lag_1"] = (
        features[
            "estimated_cost_index"
        ].shift(1)
    )

    # --------------------------------------------------------
    # One-hour-ahead forecasting target
    # --------------------------------------------------------

    features[
        FINOPS_TARGET_COLUMN
    ] = (
        features[
            "estimated_cost_index"
        ].shift(-1)
    )

    modeling_data = (
        features
        .dropna()
        .reset_index(
            drop=True
        )
    )

    expected_modeling_rows = (
        len(features) - 2
    )

    if (
        len(modeling_data)
        != expected_modeling_rows
    ):

        raise ValueError(
            "Unexpected row loss during feature "
            "engineering."
        )

    expected_columns = (
        BASE_HOURLY_COLUMNS
        + [
            "hour",
            "day_of_week",
            "day_of_month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "cpu_change",
            "memory_change",
            "disk_change",
            "network_change",
            "active_vms_change",
            "resource_cost_change",
            "cost_lag_1",
            FINOPS_TARGET_COLUMN
        ]
    )

    modeling_data = modeling_data[
        expected_columns
    ]

    if modeling_data.isna().any().any():

        raise ValueError(
            "Engineered data contains missing values."
        )

    return modeling_data


def chronological_forecasting_split(
    modeling_data: pd.DataFrame,
    training_fraction: Real = 0.70,
    validation_fraction: Real = 0.15
) -> dict[str, Any]:
    """
    Reproduce notebook 04's chronological 70/15/15 split.
    """

    if not isinstance(
        modeling_data,
        pd.DataFrame
    ):

        raise TypeError(
            "modeling_data must be a pandas DataFrame."
        )

    required_columns = (
        set(FINOPS_FEATURE_COLUMNS)
        | {
            "time_bucket",
            FINOPS_TARGET_COLUMN
        }
    )

    missing_columns = (
        required_columns
        - set(modeling_data.columns)
    )

    if missing_columns:

        raise ValueError(
            "Modeling data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    try:

        numeric_training_fraction = float(
            training_fraction
        )

        numeric_validation_fraction = float(
            validation_fraction
        )

    except (TypeError, ValueError) as error:

        raise TypeError(
            "Split fractions must be numeric."
        ) from error

    if not (
        0 < numeric_training_fraction < 1
    ):

        raise ValueError(
            "training_fraction must be between "
            "0 and 1."
        )

    if not (
        0 < numeric_validation_fraction < 1
    ):

        raise ValueError(
            "validation_fraction must be between "
            "0 and 1."
        )

    if (
        numeric_training_fraction
        + numeric_validation_fraction
        >= 1
    ):

        raise ValueError(
            "Training and validation fractions must "
            "leave records for testing."
        )

    ordered_data = (
        modeling_data
        .sort_values(
            by="time_bucket",
            ascending=True
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    total_samples = int(
        len(ordered_data)
    )

    training_end = int(
        total_samples
        * numeric_training_fraction
    )

    validation_count = int(
        total_samples
        * numeric_validation_fraction
    )

    validation_end = (
        training_end
        + validation_count
    )

    if (
        training_end < 1
        or validation_count < 1
        or validation_end >= total_samples
    ):

        raise ValueError(
            "Dataset is too small for the requested "
            "chronological split."
        )

    training_data = (
        ordered_data.iloc[
            :training_end
        ].copy()
    )

    validation_data = (
        ordered_data.iloc[
            training_end:validation_end
        ].copy()
    )

    test_data = (
        ordered_data.iloc[
            validation_end:
        ].copy()
    )

    X_train = training_data[
        FINOPS_FEATURE_COLUMNS
    ].astype(
        "float64"
    )

    y_train = training_data[
        FINOPS_TARGET_COLUMN
    ].astype(
        "float64"
    )

    X_validation = validation_data[
        FINOPS_FEATURE_COLUMNS
    ].astype(
        "float64"
    )

    y_validation = validation_data[
        FINOPS_TARGET_COLUMN
    ].astype(
        "float64"
    )

    X_test = test_data[
        FINOPS_FEATURE_COLUMNS
    ].astype(
        "float64"
    )

    y_test = test_data[
        FINOPS_TARGET_COLUMN
    ].astype(
        "float64"
    )

    time_train = training_data[
        "time_bucket"
    ].copy()

    time_validation = validation_data[
        "time_bucket"
    ].copy()

    time_test = test_data[
        "time_bucket"
    ].copy()

    if not (
        time_train.max()
        < time_validation.min()
        < time_test.min()
    ):

        raise ValueError(
            "Chronological split order is invalid."
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "time_train": time_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "time_validation": time_validation,
        "X_test": X_test,
        "y_test": y_test,
        "time_test": time_test,
        "report": {
            "total_samples": total_samples,
            "feature_count": len(
                FINOPS_FEATURE_COLUMNS
            ),
            "training_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_validation)
            ),
            "test_samples": int(
                len(X_test)
            ),
            "training_fraction": (
                numeric_training_fraction
            ),
            "validation_fraction": (
                numeric_validation_fraction
            ),
            "test_fraction": float(
                1
                - numeric_training_fraction
                - numeric_validation_fraction
            )
        }
    }