# ============================================================
# FINOPS CHALLENGER SELECTION TESTS
# ============================================================

import numpy as np
import pandas as pd
import pytest

from sklearn.base import (
    BaseEstimator,
    RegressorMixin
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.retraining.candidates import (
    calculate_candidate_metrics,
    train_and_select_challenger
)
from src.retraining.pipeline import (
    build_forecasting_features,
    chronological_forecasting_split
)


class IdentityChampion:
    """
    Test version of the deployed persistence champion.
    """

    def predict(
        self,
        features
    ):

        return features[
            "estimated_cost_index"
        ].to_numpy(
            dtype="float64"
        )


class FailingCandidate(
    BaseEstimator,
    RegressorMixin
):

    def fit(
        self,
        X,
        y
    ):

        raise RuntimeError(
            "Intentional training failure."
        )

    def predict(
        self,
        X
    ):

        return np.zeros(
            len(X),
            dtype="float64"
        )


def build_candidate_test_data(
    sample_count=202
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


def test_candidate_metrics_are_correct():

    metrics = calculate_candidate_metrics(
        actual_values=[
            1.0,
            3.0
        ],
        predicted_values=[
            2.0,
            2.0
        ]
    )

    assert metrics[
        "mae"
    ] == pytest.approx(1.0)

    assert metrics[
        "rmse"
    ] == pytest.approx(1.0)

    assert metrics[
        "bias"
    ] == pytest.approx(0.0)


def test_linear_challenger_beats_identity_champion():

    hourly_data = build_candidate_test_data()

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

    result = train_and_select_challenger(
        split_data=split_data,
        champion_model=IdentityChampion(),
        candidate_models={
            "linear_challenger": (
                LinearRegression()
            ),
            "mean_baseline": (
                DummyRegressor(
                    strategy="mean"
                )
            )
        },
        minimum_improvement_percent=2.0,
        minimum_evaluation_samples=30
    )

    assert (
        result[
            "selected_candidate_name"
        ]
        == "linear_challenger"
    )

    assert result[
        "candidate_test_metrics"
    ][
        "mae"
    ] == pytest.approx(
        0.0,
        abs=1e-8
    )

    assert result[
        "promotion_evaluation"
    ][
        "decision"
    ] == "promote_candidate"

    assert result[
        "promotion_evaluation"
    ][
        "promote_candidate"
    ] is True


def test_failed_candidate_does_not_stop_job():

    hourly_data = build_candidate_test_data()

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

    result = train_and_select_challenger(
        split_data=split_data,
        champion_model=IdentityChampion(),
        candidate_models={
            "failing_candidate": (
                FailingCandidate()
            ),
            "working_candidate": (
                LinearRegression()
            )
        },
        minimum_evaluation_samples=30
    )

    assert (
        "failing_candidate"
        in result["candidate_failures"]
    )

    assert (
        result[
            "selected_candidate_name"
        ]
        == "working_candidate"
    )