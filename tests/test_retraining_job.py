# ============================================================
# STANDALONE RETRAINING JOB TESTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.retraining.job import (
    run_retraining_job
)


class IdentityChampion:

    def predict(
        self,
        features
    ):

        return features[
            "estimated_cost_index"
        ].to_numpy(
            dtype="float64"
        )


def build_job_test_data(
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


def triggered_status():

    return {
        "trigger_retraining": True,
        "decision": "trigger_retraining"
    }


def test_wait_for_data_skips_job_without_reading_file():

    result = run_retraining_job(
        retraining_status={
            "trigger_retraining": False,
            "decision": "wait_for_data"
        },
        hourly_data_path=(
            "file-does-not-exist.csv"
        ),
        champion_model=None
    )

    assert result[
        "candidate_model"
    ] is None

    assert result[
        "report"
    ][
        "job_status"
    ] == "skipped"


def test_small_hourly_dataset_blocks_training(
    tmp_path
):

    data_path = (
        tmp_path
        / "small_hourly_data.csv"
    )

    build_job_test_data(
        sample_count=50
    ).to_csv(
        data_path,
        index=False
    )

    result = run_retraining_job(
        retraining_status=(
            triggered_status()
        ),
        hourly_data_path=data_path,
        champion_model=(
            IdentityChampion()
        ),
        candidate_models={
            "linear": (
                LinearRegression()
            )
        },
        minimum_hourly_rows=202
    )

    assert result[
        "candidate_model"
    ] is None

    assert result[
        "report"
    ][
        "job_status"
    ] == "blocked"

    assert result[
        "report"
    ][
        "raw_hourly_rows"
    ] == 50


def test_passing_candidate_becomes_ready(
    tmp_path
):

    data_path = (
        tmp_path
        / "complete_hourly_data.csv"
    )

    build_job_test_data().to_csv(
        data_path,
        index=False
    )

    result = run_retraining_job(
        retraining_status=(
            triggered_status()
        ),
        hourly_data_path=data_path,
        champion_model=(
            IdentityChampion()
        ),
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
        dry_run=True,
        minimum_hourly_rows=202,
        minimum_evaluation_samples=30
    )

    assert result[
        "candidate_model"
    ] is not None

    assert result[
        "report"
    ][
        "job_status"
    ] == "candidate_ready"

    assert result[
        "report"
    ][
        "registry_action"
    ] == "not_performed"

    assert result[
        "report"
    ][
        "promotion_evaluation"
    ][
        "promote_candidate"
    ] is True


def test_worse_candidate_is_rejected(
    tmp_path
):

    data_path = (
        tmp_path
        / "complete_hourly_data.csv"
    )

    build_job_test_data().to_csv(
        data_path,
        index=False
    )

    result = run_retraining_job(
        retraining_status=(
            triggered_status()
        ),
        hourly_data_path=data_path,
        champion_model=(
            IdentityChampion()
        ),
        candidate_models={
            "mean_baseline": (
                DummyRegressor(
                    strategy="mean"
                )
            )
        },
        dry_run=True,
        minimum_hourly_rows=202,
        minimum_evaluation_samples=30
    )

    assert result[
        "candidate_model"
    ] is None

    assert result[
        "report"
    ][
        "job_status"
    ] == "candidate_rejected"

    assert result[
        "report"
    ][
        "promotion_evaluation"
    ][
        "promote_candidate"
    ] is False