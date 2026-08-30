# ============================================================
# LIVE RETRAINING JOB RUNNER
# ============================================================

import argparse
import json

from pathlib import Path
from typing import Any, Optional

import httpx

from src.retraining.job import (
    run_retraining_job
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_API_URL = (
    "http://127.0.0.1:8000"
)


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion"
)


DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "retraining_jobs"
)


def fetch_retraining_status(
    api_url: str,
    client: Optional[
        httpx.Client
    ] = None
) -> dict[str, Any]:
    """
    Retrieve the current read-only monitoring decision.
    """

    if (
        not isinstance(api_url, str)
        or not api_url.strip()
    ):

        raise ValueError(
            "api_url must be a non-empty string."
        )

    endpoint_url = (
        api_url.rstrip("/")
        + "/retraining-status"
    )

    owns_client = (
        client is None
    )

    if client is None:

        client = httpx.Client(
            timeout=15.0
        )

    try:

        response = client.get(
            endpoint_url
        )

        response.raise_for_status()

        status_report = response.json()

    finally:

        if owns_client:

            client.close()

    if not isinstance(
        status_report,
        dict
    ):

        raise ValueError(
            "Retraining-status response must "
            "be a JSON object."
        )

    required_fields = {
        "trigger_retraining",
        "decision"
    }

    missing_fields = (
        required_fields
        - set(status_report)
    )

    if missing_fields:

        raise ValueError(
            "Retraining-status response is missing: "
            f"{sorted(missing_fields)}"
        )

    if not isinstance(
        status_report[
            "trigger_retraining"
        ],
        bool
    ):

        raise ValueError(
            "trigger_retraining must be boolean."
        )

    return status_report


def write_retraining_report(
    report: dict[str, Any],
    report_directory
) -> Path:
    """
    Save a retraining audit report using an atomic file move.
    """

    if not isinstance(
        report,
        dict
    ):

        raise TypeError(
            "report must be a dictionary."
        )

    job_id = report.get(
        "job_id"
    )

    if (
        not isinstance(job_id, str)
        or not job_id
    ):

        raise ValueError(
            "Retraining report requires a job_id."
        )

    output_directory = Path(
        report_directory
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_job_id = "".join(
        character
        for character in job_id
        if (
            character.isalnum()
            or character in {"-", "_"}
        )
    )

    if not safe_job_id:

        raise ValueError(
            "job_id does not contain safe "
            "filename characters."
        )

    final_report_path = (
        output_directory
        / f"{safe_job_id}.json"
    )

    temporary_report_path = (
        output_directory
        / f"{safe_job_id}.json.tmp"
    )

    with temporary_report_path.open(
        "w",
        encoding="utf-8"
    ) as report_file:

        json.dump(
            report,
            report_file,
            indent=2,
            sort_keys=True
        )

    temporary_report_path.replace(
        final_report_path
    )

    return final_report_path


def run_live_retraining(
    api_url: str = DEFAULT_API_URL,
    hourly_data_path=None,
    model_path=DEFAULT_MODEL_PATH,
    report_directory=DEFAULT_REPORT_DIRECTORY,
    minimum_hourly_rows: int = 202,
    minimum_improvement_percent: float = 2.0,
    status_client: Optional[
        httpx.Client
    ] = None
) -> dict[str, Any]:
    """
    Run one safe retraining check against live monitoring.

    Model loading and candidate training occur only when the live
    monitoring response explicitly triggers retraining.
    """

    retraining_status = (
        fetch_retraining_status(
            api_url=api_url,
            client=status_client
        )
    )

    trigger_retraining = (
        retraining_status[
            "trigger_retraining"
        ]
    )

    champion_model = None

    resolved_data_path = (
        hourly_data_path
    )

    # --------------------------------------------------------
    # Load training dependencies only after a real trigger
    # --------------------------------------------------------

    if trigger_retraining:

        if hourly_data_path is None:

            raise ValueError(
                "A full-feature hourly data path is "
                "required when retraining is triggered."
            )

        resolved_model_path = Path(
            model_path
        ).expanduser().resolve()

        if not resolved_model_path.exists():

            raise FileNotFoundError(
                "Champion model was not found: "
                f"{resolved_model_path}"
            )

        import mlflow

        champion_model = (
            mlflow.pyfunc.load_model(
                str(resolved_model_path)
            )
        )

    else:

        # This placeholder is never opened because job.py stops
        # before data access when no trigger exists.
        resolved_data_path = (
            "data-not-required.csv"
        )

    job_result = run_retraining_job(
        retraining_status=(
            retraining_status
        ),
        hourly_data_path=(
            resolved_data_path
        ),
        champion_model=(
            champion_model
        ),
        candidate_models=None,
        dry_run=True,
        minimum_hourly_rows=(
            minimum_hourly_rows
        ),
        minimum_improvement_percent=(
            minimum_improvement_percent
        )
    )

    report = job_result[
        "report"
    ]

    report_path = (
        write_retraining_report(
            report=report,
            report_directory=(
                report_directory
            )
        )
    )

    return {
        "report": report,
        "report_path": str(
            report_path
        )
    }


def build_argument_parser():
    """
    Build command-line arguments separately for testability.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run one FinOps retraining-status check "
            "and create an audit report."
        )
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=(
            "Base URL of the running FinOps API."
        )
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "Full-feature hourly CSV. It is required "
            "only when monitoring triggers retraining."
        )
    )

    parser.add_argument(
        "--model-path",
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help=(
            "Portable MLflow champion directory."
        )
    )

    parser.add_argument(
        "--report-directory",
        default=str(
            DEFAULT_REPORT_DIRECTORY
        ),
        help=(
            "Directory for retraining audit reports."
        )
    )

    parser.add_argument(
        "--minimum-hourly-rows",
        type=int,
        default=202
    )

    parser.add_argument(
        "--minimum-improvement-percent",
        type=float,
        default=2.0
    )

    return parser


def main():

    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = run_live_retraining(
        api_url=arguments.api_url,
        hourly_data_path=(
            arguments.data_path
        ),
        model_path=(
            arguments.model_path
        ),
        report_directory=(
            arguments.report_directory
        ),
        minimum_hourly_rows=(
            arguments.minimum_hourly_rows
        ),
        minimum_improvement_percent=(
            arguments
            .minimum_improvement_percent
        )
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":

    main()