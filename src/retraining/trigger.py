# ============================================================
# AUTOMATIC RETRAINING DECISION LOGIC
# ============================================================

from typing import Any, Optional


VALID_DRIFT_STATUSES = {
    "unavailable",
    "insufficient_data",
    "stable",
    "moderate",
    "significant"
}

VALID_PERFORMANCE_STATUSES = {
    "unavailable",
    "insufficient_data",
    "stable",
    "degraded"
}


def _get_report_status(
    report: Optional[dict[str, Any]],
    status_field: str,
    valid_statuses: set[str]
) -> str:
    """
    Safely obtain and validate a monitoring status.

    A missing report is treated as unavailable instead of
    causing the complete decision process to fail.
    """

    if report is None:
        return "unavailable"

    if not isinstance(report, dict):
        raise TypeError(
            "Monitoring report must be a dictionary."
        )

    status = report.get(
        status_field,
        "unavailable"
    )

    if status not in valid_statuses:
        raise ValueError(
            f"Unexpected {status_field}: {status}"
        )

    return status


def evaluate_retraining_trigger(
    drift_report: Optional[
        dict[str, Any]
    ] = None,
    performance_report: Optional[
        dict[str, Any]
    ] = None
) -> dict[str, Any]:
    """
    Decide whether the forecasting model should be retrained.

    This function only makes a decision. It does not start
    training or modify the MLflow champion model.
    """

    drift_status = _get_report_status(
        report=drift_report,
        status_field="status",
        valid_statuses=(
            VALID_DRIFT_STATUSES
        )
    )

    performance_status = _get_report_status(
        report=performance_report,
        status_field="performance_status",
        valid_statuses=(
            VALID_PERFORMANCE_STATUSES
        )
    )

    trigger_reasons = []

    observations = []

    # --------------------------------------------------------
    # Evaluate drift signal
    # --------------------------------------------------------

    if drift_status == "significant":

        trigger_reasons.append(
            "Significant production input drift detected."
        )

    elif drift_status == "moderate":

        observations.append(
            "Moderate input drift detected; continue "
            "close monitoring."
        )

    elif drift_status == "stable":

        observations.append(
            "Production input distribution is stable."
        )

    elif drift_status == "insufficient_data":

        observations.append(
            "Drift monitoring is waiting for more "
            "production samples."
        )

    else:

        observations.append(
            "Drift monitoring is unavailable."
        )

    # --------------------------------------------------------
    # Evaluate performance signal
    # --------------------------------------------------------

    if performance_status == "degraded":

        trigger_reasons.append(
            "Production model performance has degraded."
        )

    elif performance_status == "stable":

        observations.append(
            "Production model performance is stable."
        )

    elif performance_status == "insufficient_data":

        observations.append(
            "Performance monitoring is waiting for "
            "more completed predictions."
        )

    else:

        observations.append(
            "Performance monitoring is unavailable."
        )

    # --------------------------------------------------------
    # Produce final decision
    # --------------------------------------------------------

    trigger_retraining = (
        len(trigger_reasons) > 0
    )

    if trigger_retraining:

        decision = "trigger_retraining"

        message = (
            "One or more reliable monitoring signals "
            "recommend model retraining."
        )

    elif drift_status == "moderate":

        decision = "monitor_closely"

        message = (
            "Retraining is not required yet, but the "
            "input distribution should be monitored."
        )

    elif (
        drift_status
        in {
            "unavailable",
            "insufficient_data"
        }
        and performance_status
        in {
            "unavailable",
            "insufficient_data"
        }
    ):

        decision = "wait_for_data"

        message = (
            "There is not enough monitoring evidence "
            "to make a retraining decision."
        )

    else:

        decision = "no_retraining"

        message = (
            "Current monitoring signals do not "
            "recommend retraining."
        )

    return {
        "trigger_retraining": (
            trigger_retraining
        ),
        "decision": decision,
        "drift_status": drift_status,
        "performance_status": (
            performance_status
        ),
        "trigger_reasons": (
            trigger_reasons
        ),
        "observations": observations,
        "message": message
    }