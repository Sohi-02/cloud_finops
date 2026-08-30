# ============================================================
# LIVE RETRAINING RUNNER TESTS
# ============================================================

import json

import httpx

from src.retraining.runner import (
    fetch_retraining_status,
    run_live_retraining,
    write_retraining_report
)


def build_mock_client(
    response_body
):

    def handler(request):

        assert (
            request.url.path
            == "/retraining-status"
        )

        return httpx.Response(
            status_code=200,
            json=response_body
        )

    transport = httpx.MockTransport(
        handler
    )

    return httpx.Client(
        transport=transport
    )


def test_fetch_retraining_status():

    mock_client = build_mock_client({
        "trigger_retraining": False,
        "decision": "wait_for_data"
    })

    try:

        result = fetch_retraining_status(
            api_url="http://test-api",
            client=mock_client
        )

    finally:

        mock_client.close()

    assert (
        result["trigger_retraining"]
        is False
    )

    assert (
        result["decision"]
        == "wait_for_data"
    )


def test_write_retraining_report(
    tmp_path
):

    report = {
        "job_id": "test-job-123",
        "job_status": "skipped",
        "reason": "Test report."
    }

    report_path = (
        write_retraining_report(
            report=report,
            report_directory=tmp_path
        )
    )

    assert report_path.exists()

    saved_report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved_report["job_id"]
        == "test-job-123"
    )

    assert (
        saved_report["job_status"]
        == "skipped"
    )


def test_wait_for_data_creates_skipped_audit(
    tmp_path
):

    mock_client = build_mock_client({
        "trigger_retraining": False,
        "decision": "wait_for_data"
    })

    try:

        result = run_live_retraining(
            api_url="http://test-api",
            hourly_data_path=None,
            model_path=(
                "model-does-not-exist"
            ),
            report_directory=tmp_path,
            status_client=mock_client
        )

    finally:

        mock_client.close()

    assert result[
        "report"
    ][
        "job_status"
    ] == "skipped"

    assert result[
        "report"
    ][
        "trigger_retraining"
    ] is False

    assert (
        tmp_path
        / (
            result["report"]["job_id"]
            + ".json"
        )
    ).exists()