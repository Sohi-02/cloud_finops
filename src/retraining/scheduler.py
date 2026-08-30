# ============================================================
# PERIODIC RETRAINING STATUS SCHEDULER
# ============================================================

import argparse
import json
import time

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.retraining.runner import (
    run_live_retraining
)


def run_retraining_scheduler(
    interval_seconds: int = 21600,
    run_once: bool = False,
    runner: Callable[..., dict[str, Any]] = (
        run_live_retraining
    ),
    runner_arguments: Optional[
        dict[str, Any]
    ] = None
):

    if interval_seconds < 60:

        raise ValueError(
            "interval_seconds must be at least 60."
        )

    if runner_arguments is None:

        runner_arguments = {}

    while True:

        started_at = datetime.now(
            timezone.utc
        ).isoformat()

        try:

            result = runner(
                **runner_arguments
            )

            scheduler_result = {
                "status": "completed",
                "started_at_utc": (
                    started_at
                ),
                "result": result
            }

        except Exception as error:

            scheduler_result = {
                "status": "failed",
                "started_at_utc": (
                    started_at
                ),
                "error_type": (
                    type(error).__name__
                ),
                "message": str(error)
            }

        print(
            json.dumps(
                scheduler_result,
                indent=2
            ),
            flush=True
        )

        if run_once:

            return scheduler_result

        time.sleep(
            interval_seconds
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Periodically check FinOps monitoring "
            "and run the safe retraining runner."
        )
    )

    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=21600
    )

    parser.add_argument(
        "--run-once",
        action="store_true"
    )

    parser.add_argument(
        "--api-url",
        default=(
            "http://127.0.0.1:8000"
        )
    )

    parser.add_argument(
        "--data-path",
        default=None
    )

    arguments = parser.parse_args()

    run_retraining_scheduler(
        interval_seconds=(
            arguments.interval_seconds
        ),
        run_once=arguments.run_once,
        runner_arguments={
            "api_url": arguments.api_url,
            "hourly_data_path": (
                arguments.data_path
            )
        }
    )


if __name__ == "__main__":

    main()