# ============================================================
# FINOPS FORECASTING FEATURE PIPELINE TESTS
# ============================================================

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from src.retraining.pipeline import (
    FINOPS_FEATURE_COLUMNS,
    FINOPS_TARGET_COLUMN,
    build_forecasting_features,
    chronological_forecasting_split,
    validate_hourly_finops_data
)


def build_hourly_test_data(
    sample_count=12
):

    sequence = np.arange(
        sample_count,
        dtype="float64"
    )

    return pd.DataFrame({
        "time_bucket": pd.date_range(
            start="2026-01-01",
            periods=sample_count,
            freq="h",
            tz="UTC"
        ),
        "cpu_mean": 10.0 + sequence,
        "memory_mean_gb": (
            2.0 + sequence * 0.10
        ),
        "disk_activity_kbps": (
            5.0 + sequence * 2.0
        ),
        "network_activity_kbps": (
            3.0 + sequence * 0.50
        ),
        "active_vms": (
            20.0 + sequence
        ),
        "resource_cost_index": (
            4.0 + sequence * 1.50
        ),
        "estimated_cost_index": (
            5.0 + sequence * 2.0
        )
    })


def test_feature_pipeline_matches_notebook_formulas():

    hourly_data = build_hourly_test_data(
        sample_count=12
    )

    modeling_data = (
        build_forecasting_features(
            hourly_data
        )
    )

    assert len(modeling_data) == 10

    assert len(
        FINOPS_FEATURE_COLUMNS
    ) == 22

    first_modeling_row = (
        modeling_data.iloc[0]
    )

    assert first_modeling_row[
        "time_bucket"
    ] == hourly_data.iloc[1][
        "time_bucket"
    ]

    assert first_modeling_row[
        "cost_lag_1"
    ] == approx(
        hourly_data.iloc[0][
            "estimated_cost_index"
        ]
    )

    assert first_modeling_row[
        FINOPS_TARGET_COLUMN
    ] == approx(
        hourly_data.iloc[2][
            "estimated_cost_index"
        ]
    )

    assert first_modeling_row[
        "cpu_change"
    ] == approx(1.0)

    assert first_modeling_row[
        "active_vms_change"
    ] == approx(1.0)


def test_chronological_split_matches_notebook_ratio():

    hourly_data = build_hourly_test_data(
        sample_count=102
    )

    modeling_data = (
        build_forecasting_features(
            hourly_data
        )
    )

    result = (
        chronological_forecasting_split(
            modeling_data
        )
    )

    report = result["report"]

    assert report[
        "total_samples"
    ] == 100

    assert report[
        "training_samples"
    ] == 70

    assert report[
        "validation_samples"
    ] == 15

    assert report[
        "test_samples"
    ] == 15

    assert result[
        "X_train"
    ].shape == (70, 22)

    assert result[
        "X_validation"
    ].shape == (15, 22)

    assert result[
        "X_test"
    ].shape == (15, 22)


def test_unordered_data_is_sorted():

    hourly_data = (
        build_hourly_test_data(
            sample_count=12
        )
        .sample(
            frac=1.0,
            random_state=42
        )
        .reset_index(
            drop=True
        )
    )

    validated_data = (
        validate_hourly_finops_data(
            hourly_data
        )
    )

    assert validated_data[
        "time_bucket"
    ].is_monotonic_increasing


def test_duplicate_timestamp_is_rejected():

    hourly_data = build_hourly_test_data(
        sample_count=12
    )

    hourly_data.loc[
        5,
        "time_bucket"
    ] = hourly_data.loc[
        4,
        "time_bucket"
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate hourly timestamps"
    ):

        validate_hourly_finops_data(
            hourly_data
        )


def test_missing_numeric_value_is_rejected():

    hourly_data = build_hourly_test_data(
        sample_count=12
    )

    hourly_data.loc[
        4,
        "cpu_mean"
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing or non-numeric"
    ):

        validate_hourly_finops_data(
            hourly_data
        )