# ============================================================
# RETRAINING TRIGGER TESTS
# ============================================================

from src.retraining.trigger import (
    evaluate_retraining_trigger
)


def test_missing_monitoring_reports_wait_for_data():

    result = evaluate_retraining_trigger()

    assert result["trigger_retraining"] is False
    assert result["decision"] == "wait_for_data"
    assert result["drift_status"] == "unavailable"
    assert (
        result["performance_status"]
        == "unavailable"
    )


def test_insufficient_samples_wait_for_data():

    drift_report = {
        "status": "insufficient_data"
    }

    performance_report = {
        "performance_status": (
            "insufficient_data"
        )
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is False
    assert result["decision"] == "wait_for_data"


def test_stable_signals_do_not_trigger_retraining():

    drift_report = {
        "status": "stable",
        "psi": 0.04
    }

    performance_report = {
        "performance_status": "stable",
        "mae": 8.0
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is False
    assert result["decision"] == "no_retraining"


def test_moderate_drift_requests_close_monitoring():

    drift_report = {
        "status": "moderate",
        "psi": 0.16
    }

    performance_report = {
        "performance_status": "stable",
        "mae": 8.0
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is False
    assert result["decision"] == "monitor_closely"


def test_significant_drift_triggers_retraining():

    drift_report = {
        "status": "significant",
        "psi": 0.40
    }

    performance_report = {
        "performance_status": "stable",
        "mae": 8.0
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is True
    assert result["decision"] == "trigger_retraining"

    assert (
        "Significant production input drift detected."
        in result["trigger_reasons"]
    )


def test_degraded_performance_triggers_retraining():

    drift_report = {
        "status": "stable",
        "psi": 0.03
    }

    performance_report = {
        "performance_status": "degraded",
        "mae": 11.0
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is True
    assert result["decision"] == "trigger_retraining"

    assert (
        "Production model performance has degraded."
        in result["trigger_reasons"]
    )


def test_multiple_failures_preserve_all_reasons():

    drift_report = {
        "status": "significant",
        "psi": 0.50
    }

    performance_report = {
        "performance_status": "degraded",
        "mae": 12.0
    }

    result = evaluate_retraining_trigger(
        drift_report=drift_report,
        performance_report=performance_report
    )

    assert result["trigger_retraining"] is True

    assert len(
        result["trigger_reasons"]
    ) == 2