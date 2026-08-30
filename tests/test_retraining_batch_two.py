import json

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression
)

from src.retraining.contract import (
    evaluate_model_contract
)
from src.retraining.deployment import (
    export_versioned_model_bundle
)
from src.retraining.inference import (
    build_latest_inference_features
)
from src.retraining.mlflow_logger import (
    detect_model_flavor
)
from src.retraining.pipeline import (
    FINOPS_FEATURE_COLUMNS
)
from src.retraining.scheduler import (
    run_retraining_scheduler
)


def test_current_contract_blocks_22_feature_model():

    manifest = {
        "input_schema": {
            "estimated_cost_index": "float64"
        },
        "output": (
            "predicted_next_hour_cost"
        )
    }

    result = evaluate_model_contract(
        deployment_manifest=manifest,
        candidate_feature_columns=(
            FINOPS_FEATURE_COLUMNS
        )
    )

    assert (
        result["contract_compatible"]
        is False
    )

    assert (
        result["promotion_allowed"]
        is False
    )

    assert (
        result["candidate_feature_count"]
        == 22
    )


def test_model_flavors():

    assert (
        detect_model_flavor("xgboost")
        == "xgboost"
    )

    assert (
        detect_model_flavor("lightgbm")
        == "lightgbm"
    )

    assert (
        detect_model_flavor("catboost")
        == "catboost"
    )

    assert (
        detect_model_flavor(
            "linear_challenger"
        )
        == "sklearn"
    )


def test_latest_inference_features():

    previous_hour = {
        "time_bucket": (
            "2026-08-29T04:00:00Z"
        ),
        "cpu_mean": 20.0,
        "memory_mean_gb": 4.0,
        "disk_activity_kbps": 10.0,
        "network_activity_kbps": 5.0,
        "active_vms": 30.0,
        "resource_cost_index": 22.0,
        "estimated_cost_index": 24.0
    }

    current_hour = {
        "time_bucket": (
            "2026-08-29T05:00:00Z"
        ),
        "cpu_mean": 22.0,
        "memory_mean_gb": 4.2,
        "disk_activity_kbps": 12.0,
        "network_activity_kbps": 6.0,
        "active_vms": 31.0,
        "resource_cost_index": 23.0,
        "estimated_cost_index": 25.0
    }

    features = (
        build_latest_inference_features(
            previous_hour,
            current_hour
        )
    )

    assert features.shape == (1, 22)

    assert features.iloc[0][
        "cpu_change"
    ] == 2.0

    assert features.iloc[0][
        "cost_lag_1"
    ] == 24.0


def test_versioned_bundle_export(
    tmp_path
):

    source_model = (
        tmp_path
        / "source_model"
    )

    source_model.mkdir()

    (
        source_model
        / "MLmodel"
    ).write_text(
        "test-model",
        encoding="utf-8"
    )

    def fake_downloader(
        artifact_uri,
        dst_path
    ):

        return str(
            source_model
        )

    manifest = {
        "model_version": "2",
        "input_schema": {
            feature: "float64"
            for feature in (
                FINOPS_FEATURE_COLUMNS
            )
        },
        "output": (
            "predicted_next_hour_cost"
        )
    }

    bundle_path = (
        export_versioned_model_bundle(
            model_uri="models:/test/2",
            destination_root=(
                tmp_path
                / "deployments"
            ),
            deployment_manifest=manifest,
            artifact_downloader=(
                fake_downloader
            )
        )
    )

    assert (
        bundle_path
        / "MLmodel"
    ).exists()

    saved_manifest = json.loads(
        (
            bundle_path
            / "deployment_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved_manifest["model_version"]
        == "2"
    )


def test_scheduler_run_once():

    def fake_runner():

        return {
            "report": {
                "job_status": "skipped"
            }
        }

    result = run_retraining_scheduler(
        interval_seconds=60,
        run_once=True,
        runner=fake_runner
    )

    assert (
        result["status"]
        == "completed"
    )

    assert (
        result["result"]["report"][
            "job_status"
        ]
        == "skipped"
    )