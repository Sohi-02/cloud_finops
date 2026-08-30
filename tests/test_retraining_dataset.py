# ============================================================
# PRODUCTION RETRAINING DATASET TESTS
# ============================================================

import pytest

from src.retraining.dataset import (
    build_retraining_dataset
)


def make_completed_record(
    prediction_id,
    timestamp,
    feature_value,
    actual_cost,
    model_version="1",
    data_quality_passed=True,
    status="completed"
):

    return {
        "prediction_id": prediction_id,
        "prediction_timestamp_utc": timestamp,
        "input_features": {
            "estimated_cost_index": (
                feature_value
            )
        },
        "data_quality": {
            "passed": data_quality_passed,
            "errors": [],
            "warnings": []
        },
        "prediction": {
            "predicted_next_hour_cost": (
                feature_value
            )
        },
        "actual": {
            "actual_next_hour_cost": (
                actual_cost
            )
        },
        "model": {
            "registered_model": (
                "finops-cloud-cost-forecasting-clean-v1"
            ),
            "alias": "champion",
            "version": model_version
        },
        "status": status
    }


def test_valid_records_are_sorted_chronologically():

    records = [
        make_completed_record(
            prediction_id="prediction-3",
            timestamp="2026-08-29T03:00:00+00:00",
            feature_value=30.0,
            actual_cost=31.0
        ),
        make_completed_record(
            prediction_id="prediction-1",
            timestamp="2026-08-29T01:00:00+00:00",
            feature_value=10.0,
            actual_cost=11.0
        ),
        make_completed_record(
            prediction_id="prediction-2",
            timestamp="2026-08-29T02:00:00+00:00",
            feature_value=20.0,
            actual_cost=21.0
        )
    ]

    result = build_retraining_dataset(
        prediction_records=records,
        expected_model_version="1",
        minimum_records=3
    )

    dataset = result["dataset"]

    report = result["report"]

    assert dataset[
        "prediction_id"
    ].tolist() == [
        "prediction-1",
        "prediction-2",
        "prediction-3"
    ]

    assert report[
        "accepted_record_count"
    ] == 3

    assert report[
        "enough_records"
    ] is True

    assert len(
        report["dataset_fingerprint"]
    ) == 64


def test_wrong_model_version_is_rejected():

    records = [
        make_completed_record(
            prediction_id="version-1",
            timestamp="2026-08-29T01:00:00+00:00",
            feature_value=10.0,
            actual_cost=11.0,
            model_version="1"
        ),
        make_completed_record(
            prediction_id="version-2",
            timestamp="2026-08-29T02:00:00+00:00",
            feature_value=20.0,
            actual_cost=21.0,
            model_version="2"
        )
    ]

    result = build_retraining_dataset(
        prediction_records=records,
        expected_model_version="1",
        minimum_records=1
    )

    report = result["report"]

    assert report[
        "accepted_record_count"
    ] == 1

    assert report[
        "rejection_counts"
    ][
        "model_version_mismatch"
    ] == 1


def test_invalid_production_records_are_rejected():

    records = [
        make_completed_record(
            prediction_id="valid",
            timestamp="2026-08-29T01:00:00+00:00",
            feature_value=10.0,
            actual_cost=11.0
        ),
        make_completed_record(
            prediction_id="quality-failed",
            timestamp="2026-08-29T02:00:00+00:00",
            feature_value=20.0,
            actual_cost=21.0,
            data_quality_passed=False
        ),
        make_completed_record(
            prediction_id="invalid-feature",
            timestamp="2026-08-29T03:00:00+00:00",
            feature_value=float("nan"),
            actual_cost=31.0
        ),
        make_completed_record(
            prediction_id="invalid-target",
            timestamp="2026-08-29T04:00:00+00:00",
            feature_value=40.0,
            actual_cost=-1.0
        )
    ]

    result = build_retraining_dataset(
        prediction_records=records,
        expected_model_version="1",
        minimum_records=1
    )

    report = result["report"]

    assert report[
        "accepted_record_count"
    ] == 1

    assert report[
        "rejected_record_count"
    ] == 3

    assert report[
        "rejection_counts"
    ][
        "data_quality_failed"
    ] == 1

    assert report[
        "rejection_counts"
    ][
        "invalid_feature"
    ] == 1

    assert report[
        "rejection_counts"
    ][
        "invalid_target"
    ] == 1


def test_duplicate_prediction_keeps_newest_record():

    records = [
        make_completed_record(
            prediction_id="duplicate-id",
            timestamp="2026-08-29T01:00:00+00:00",
            feature_value=10.0,
            actual_cost=11.0
        ),
        make_completed_record(
            prediction_id="duplicate-id",
            timestamp="2026-08-29T02:00:00+00:00",
            feature_value=20.0,
            actual_cost=21.0
        )
    ]

    result = build_retraining_dataset(
        prediction_records=records,
        expected_model_version="1",
        minimum_records=1
    )

    dataset = result["dataset"]

    report = result["report"]

    assert len(dataset) == 1

    assert dataset.iloc[0][
        "estimated_cost_index"
    ] == pytest.approx(20.0)

    assert report[
        "duplicate_record_count"
    ] == 1


def test_small_dataset_is_marked_insufficient():

    records = [
        make_completed_record(
            prediction_id="prediction-1",
            timestamp="2026-08-29T01:00:00+00:00",
            feature_value=10.0,
            actual_cost=11.0
        ),
        make_completed_record(
            prediction_id="prediction-2",
            timestamp="2026-08-29T02:00:00+00:00",
            feature_value=20.0,
            actual_cost=21.0
        )
    ]

    result = build_retraining_dataset(
        prediction_records=records,
        expected_model_version="1",
        minimum_records=200
    )

    assert result["report"][
        "accepted_record_count"
    ] == 2

    assert result["report"][
        "enough_records"
    ] is False