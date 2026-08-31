# RETRAINED MODEL EVALUATION AND PROMOTION GATE

import math

from numbers import Integral, Real
from typing import Any


def _validate_metric(
    metric_name: str,
    metric_value: Real,
    greater_than_zero: bool = False
) -> float:
    """
    Validate a model evaluation metric.
    """

    if isinstance(metric_value, bool):

        raise TypeError(
            f"{metric_name} must be numeric."
        )

    try:

        numeric_value = float(
            metric_value
        )

    except (TypeError, ValueError) as error:

        raise TypeError(
            f"{metric_name} must be numeric."
        ) from error

    if not math.isfinite(
        numeric_value
    ):

        raise ValueError(
            f"{metric_name} must be finite."
        )

    if greater_than_zero:

        if numeric_value <= 0:

            raise ValueError(
                f"{metric_name} must be greater "
                "than zero."
            )

    elif numeric_value < 0:

        raise ValueError(
            f"{metric_name} cannot be negative."
        )

    return numeric_value


def evaluate_candidate_for_promotion(
    champion_mae: Real,
    candidate_mae: Real,
    evaluation_sample_count: int,
    minimum_evaluation_samples: int = 30,
    minimum_improvement_percent: Real = 2.0
) -> dict[str, Any]:
    """
    Decide whether a retrained candidate should replace the
    deployed champion.

    Promotion requirements:

    1. The evaluation dataset must contain enough samples.
    2. Candidate MAE must be lower than champion MAE.
    3. Improvement must meet the configured percentage.
    """

    numeric_champion_mae = (
        _validate_metric(
            metric_name="champion_mae",
            metric_value=champion_mae,
            greater_than_zero=True
        )
    )

    numeric_candidate_mae = (
        _validate_metric(
            metric_name="candidate_mae",
            metric_value=candidate_mae
        )
    )

    numeric_improvement_requirement = (
        _validate_metric(
            metric_name=(
                "minimum_improvement_percent"
            ),
            metric_value=(
                minimum_improvement_percent
            )
        )
    )

    if numeric_improvement_requirement >= 100:

        raise ValueError(
            "minimum_improvement_percent must "
            "be below 100."
        )

    if (
        isinstance(evaluation_sample_count, bool)
        or not isinstance(
            evaluation_sample_count,
            Integral
        )
    ):

        raise TypeError(
            "evaluation_sample_count must be "
            "an integer."
        )

    if (
        isinstance(minimum_evaluation_samples, bool)
        or not isinstance(
            minimum_evaluation_samples,
            Integral
        )
    ):

        raise TypeError(
            "minimum_evaluation_samples must "
            "be an integer."
        )

    numeric_sample_count = int(
        evaluation_sample_count
    )

    numeric_minimum_samples = int(
        minimum_evaluation_samples
    )

    if numeric_sample_count < 0:

        raise ValueError(
            "evaluation_sample_count cannot "
            "be negative."
        )

    if numeric_minimum_samples < 1:

        raise ValueError(
            "minimum_evaluation_samples must "
            "be at least 1."
        )

    improvement_percent = float(
        (
            numeric_champion_mae
            - numeric_candidate_mae
        )
        / numeric_champion_mae
        * 100
    )

    required_candidate_mae = float(
        numeric_champion_mae
        * (
            1
            - numeric_improvement_requirement
            / 100
        )
    )

    candidate_is_better = (
        numeric_candidate_mae
        < numeric_champion_mae
    )

    improvement_requirement_met = (
        improvement_percent
        >= numeric_improvement_requirement
    )

    enough_evaluation_data = (
        numeric_sample_count
        >= numeric_minimum_samples
    )

    if not enough_evaluation_data:

        decision = "insufficient_data"

        promote_candidate = False

        message = (
            "The candidate cannot be promoted because "
            "the evaluation dataset is too small."
        )

    elif (
        candidate_is_better
        and improvement_requirement_met
    ):

        decision = "promote_candidate"

        promote_candidate = True

        message = (
            "The candidate passed the evaluation gate "
            "and can replace the current champion."
        )

    else:

        decision = "reject_candidate"

        promote_candidate = False

        if not candidate_is_better:

            message = (
                "The candidate was rejected because it "
                "did not outperform the champion."
            )

        else:

            message = (
                "The candidate improved performance, "
                "but not by the required percentage."
            )

    return {
        "decision": decision,
        "promote_candidate": (
            promote_candidate
        ),
        "candidate_is_better": (
            candidate_is_better
        ),
        "improvement_requirement_met": (
            improvement_requirement_met
        ),
        "enough_evaluation_data": (
            enough_evaluation_data
        ),
        "evaluation_sample_count": (
            numeric_sample_count
        ),
        "minimum_evaluation_samples": (
            numeric_minimum_samples
        ),
        "champion_mae": (
            numeric_champion_mae
        ),
        "candidate_mae": (
            numeric_candidate_mae
        ),
        "required_candidate_mae": (
            required_candidate_mae
        ),
        "improvement_percent": (
            improvement_percent
        ),
        "minimum_improvement_percent": (
            numeric_improvement_requirement
        ),
        "message": message
    }