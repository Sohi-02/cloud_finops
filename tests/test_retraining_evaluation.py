# ============================================================
# RETRAINED MODEL EVALUATION TESTS
# ============================================================

import pytest

from src.retraining.evaluation import (
    evaluate_candidate_for_promotion
)


def test_good_candidate_is_promoted():

    result = evaluate_candidate_for_promotion(
        champion_mae=10.0,
        candidate_mae=8.0,
        evaluation_sample_count=100,
        minimum_evaluation_samples=30,
        minimum_improvement_percent=2.0
    )

    assert (
        result["decision"]
        == "promote_candidate"
    )

    assert (
        result["promote_candidate"]
        is True
    )

    assert (
        result["improvement_percent"]
        == pytest.approx(20.0)
    )


def test_worse_candidate_is_rejected():

    result = evaluate_candidate_for_promotion(
        champion_mae=10.0,
        candidate_mae=11.0,
        evaluation_sample_count=100
    )

    assert (
        result["decision"]
        == "reject_candidate"
    )

    assert (
        result["promote_candidate"]
        is False
    )

    assert (
        result["candidate_is_better"]
        is False
    )


def test_small_improvement_is_rejected():

    result = evaluate_candidate_for_promotion(
        champion_mae=10.0,
        candidate_mae=9.9,
        evaluation_sample_count=100,
        minimum_improvement_percent=2.0
    )

    assert (
        result["decision"]
        == "reject_candidate"
    )

    assert (
        result["candidate_is_better"]
        is True
    )

    assert (
        result[
            "improvement_requirement_met"
        ]
        is False
    )


def test_insufficient_evaluation_data_blocks_promotion():

    result = evaluate_candidate_for_promotion(
        champion_mae=10.0,
        candidate_mae=7.0,
        evaluation_sample_count=10,
        minimum_evaluation_samples=30
    )

    assert (
        result["decision"]
        == "insufficient_data"
    )

    assert (
        result["promote_candidate"]
        is False
    )

    assert (
        result["enough_evaluation_data"]
        is False
    )


@pytest.mark.parametrize(
    "champion_mae,candidate_mae",
    [
        (0.0, 5.0),
        (-1.0, 5.0),
        (10.0, -1.0),
        (float("nan"), 5.0),
        (10.0, float("inf"))
    ]
)
def test_invalid_metrics_are_rejected(
    champion_mae,
    candidate_mae
):

    with pytest.raises(
        (TypeError, ValueError)
    ):

        evaluate_candidate_for_promotion(
            champion_mae=champion_mae,
            candidate_mae=candidate_mae,
            evaluation_sample_count=100
        )