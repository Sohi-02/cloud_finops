# ============================================================
# PRODUCTION RETRAINING DATASET BUILDER
# ============================================================

import hashlib
import math

from collections import Counter
from collections.abc import Iterable
from numbers import Integral
from typing import Any

import pandas as pd


RESERVED_COLUMN_NAMES = {
    "prediction_id",
    "prediction_timestamp_utc",
    "actual_next_hour_cost",
    "model_version"
}


def _validate_non_negative_number(
    value: Any
) -> float:
    """
    Convert a value to a finite, non-negative float.
    """

    if isinstance(value, bool):

        raise ValueError(
            "Boolean values are not valid numeric values."
        )

    try:

        numeric_value = float(
            value
        )

    except (TypeError, ValueError) as error:

        raise ValueError(
            "Value is not numeric."
        ) from error

    if not math.isfinite(
        numeric_value
    ):

        raise ValueError(
            "Value must be finite."
        )

    if numeric_value < 0:

        raise ValueError(
            "Value cannot be negative."
        )

    return numeric_value


def _validate_minimum_records(
    minimum_records: int
) -> int:
    """
    Validate the minimum retraining dataset size.
    """

    if (
        isinstance(minimum_records, bool)
        or not isinstance(
            minimum_records,
            Integral
        )
    ):

        raise TypeError(
            "minimum_records must be an integer."
        )

    numeric_minimum = int(
        minimum_records
    )

    if numeric_minimum < 1:

        raise ValueError(
            "minimum_records must be at least 1."
        )

    return numeric_minimum


def _create_dataset_fingerprint(
    dataset: pd.DataFrame,
    feature_name: str,
    model_version: str
) -> str:
    """
    Create a deterministic fingerprint for the clean dataset.

    The fingerprint later prevents the retraining scheduler from
    repeatedly training on exactly the same records.
    """

    fingerprint_rows = []

    for _, row in dataset.iterrows():

        fingerprint_rows.append(
            "|".join(
                [
                    str(
                        row["prediction_id"]
                    ),
                    row[
                        "prediction_timestamp_utc"
                    ].isoformat(),
                    format(
                        float(
                            row[feature_name]
                        ),
                        ".17g"
                    ),
                    format(
                        float(
                            row[
                                "actual_next_hour_cost"
                            ]
                        ),
                        ".17g"
                    ),
                    model_version
                ]
            )
        )

    fingerprint_payload = "\n".join(
        fingerprint_rows
    )

    return hashlib.sha256(
        fingerprint_payload.encode(
            "utf-8"
        )
    ).hexdigest()


def build_retraining_dataset(
    prediction_records: Iterable[
        dict[str, Any]
    ],
    expected_model_version: str,
    feature_name: str = "estimated_cost_index",
    minimum_records: int = 200
) -> dict[str, Any]:
    """
    Build a clean, chronological supervised dataset from
    completed production prediction records.

    Only records created by the expected deployed model version
    are accepted.
    """

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if (
        not isinstance(
            expected_model_version,
            str
        )
        or not expected_model_version.strip()
    ):

        raise ValueError(
            "expected_model_version must be a "
            "non-empty string."
        )

    normalized_model_version = (
        expected_model_version.strip()
    )

    if (
        not isinstance(feature_name, str)
        or not feature_name.strip()
    ):

        raise ValueError(
            "feature_name must be a non-empty string."
        )

    normalized_feature_name = (
        feature_name.strip()
    )

    if (
        "." in normalized_feature_name
        or normalized_feature_name.startswith("$")
        or normalized_feature_name
        in RESERVED_COLUMN_NAMES
    ):

        raise ValueError(
            "feature_name is invalid."
        )

    numeric_minimum_records = (
        _validate_minimum_records(
            minimum_records
        )
    )

    if (
        isinstance(
            prediction_records,
            (str, bytes)
        )
        or not isinstance(
            prediction_records,
            Iterable
        )
    ):

        raise TypeError(
            "prediction_records must be iterable."
        )

    records = list(
        prediction_records
    )

    rejection_counts = Counter()

    accepted_rows = []

    # --------------------------------------------------------
    # Validate individual MongoDB records
    # --------------------------------------------------------

    for record in records:

        if not isinstance(record, dict):

            rejection_counts[
                "invalid_record_type"
            ] += 1

            continue

        if record.get("status") != "completed":

            rejection_counts[
                "incomplete_status"
            ] += 1

            continue

        model_information = record.get(
            "model",
            {}
        )

        record_model_version = (
            model_information.get(
                "version"
            )
            if isinstance(
                model_information,
                dict
            )
            else None
        )

        if (
            record_model_version is None
            or str(record_model_version)
            != normalized_model_version
        ):

            rejection_counts[
                "model_version_mismatch"
            ] += 1

            continue

        data_quality = record.get(
            "data_quality",
            {}
        )

        if (
            not isinstance(
                data_quality,
                dict
            )
            or data_quality.get("passed")
            is not True
        ):

            rejection_counts[
                "data_quality_failed"
            ] += 1

            continue

        prediction_id = record.get(
            "prediction_id"
        )

        if (
            not isinstance(
                prediction_id,
                str
            )
            or not prediction_id.strip()
        ):

            rejection_counts[
                "invalid_prediction_id"
            ] += 1

            continue

        prediction_timestamp = (
            pd.to_datetime(
                record.get(
                    "prediction_timestamp_utc"
                ),
                utc=True,
                errors="coerce"
            )
        )

        if pd.isna(
            prediction_timestamp
        ):

            rejection_counts[
                "invalid_timestamp"
            ] += 1

            continue

        input_features = record.get(
            "input_features",
            {}
        )

        if not isinstance(
            input_features,
            dict
        ):

            rejection_counts[
                "invalid_feature"
            ] += 1

            continue

        try:

            feature_value = (
                _validate_non_negative_number(
                    input_features.get(
                        normalized_feature_name
                    )
                )
            )

        except ValueError:

            rejection_counts[
                "invalid_feature"
            ] += 1

            continue

        actual_information = record.get(
            "actual",
            {}
        )

        if not isinstance(
            actual_information,
            dict
        ):

            rejection_counts[
                "invalid_target"
            ] += 1

            continue

        try:

            actual_cost = (
                _validate_non_negative_number(
                    actual_information.get(
                        "actual_next_hour_cost"
                    )
                )
            )

        except ValueError:

            rejection_counts[
                "invalid_target"
            ] += 1

            continue

        accepted_rows.append({
            "prediction_id": (
                prediction_id.strip()
            ),
            "prediction_timestamp_utc": (
                prediction_timestamp
            ),
            normalized_feature_name: (
                feature_value
            ),
            "actual_next_hour_cost": (
                actual_cost
            ),
            "model_version": (
                normalized_model_version
            )
        })

    # --------------------------------------------------------
    # Create chronologically ordered DataFrame
    # --------------------------------------------------------

    dataset_columns = [
        "prediction_id",
        "prediction_timestamp_utc",
        normalized_feature_name,
        "actual_next_hour_cost",
        "model_version"
    ]

    dataset = pd.DataFrame(
        accepted_rows,
        columns=dataset_columns
    )

    duplicate_count = 0

    if not dataset.empty:

        dataset = dataset.sort_values(
            by=[
                "prediction_timestamp_utc",
                "prediction_id"
            ],
            ascending=True
        )

        duplicate_mask = (
            dataset.duplicated(
                subset=["prediction_id"],
                keep="last"
            )
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        if duplicate_count > 0:

            rejection_counts[
                "duplicate_prediction_id"
            ] += duplicate_count

            dataset = dataset.loc[
                ~duplicate_mask
            ]

        dataset = (
            dataset
            .sort_values(
                by=[
                    "prediction_timestamp_utc",
                    "prediction_id"
                ],
                ascending=True
            )
            .reset_index(
                drop=True
            )
        )

    accepted_record_count = int(
        len(dataset)
    )

    total_record_count = int(
        len(records)
    )

    enough_records = (
        accepted_record_count
        >= numeric_minimum_records
    )

    dataset_fingerprint = (
        _create_dataset_fingerprint(
            dataset=dataset,
            feature_name=(
                normalized_feature_name
            ),
            model_version=(
                normalized_model_version
            )
        )
    )

    # --------------------------------------------------------
    # Return dataset and validation report
    # --------------------------------------------------------

    return {
        "dataset": dataset,
        "report": {
            "total_record_count": (
                total_record_count
            ),
            "accepted_record_count": (
                accepted_record_count
            ),
            "rejected_record_count": (
                total_record_count
                - accepted_record_count
            ),
            "duplicate_record_count": (
                duplicate_count
            ),
            "minimum_required_records": (
                numeric_minimum_records
            ),
            "enough_records": (
                enough_records
            ),
            "expected_model_version": (
                normalized_model_version
            ),
            "feature_name": (
                normalized_feature_name
            ),
            "target_name": (
                "actual_next_hour_cost"
            ),
            "dataset_fingerprint": (
                dataset_fingerprint
            ),
            "rejection_counts": dict(
                rejection_counts
            )
        }
    }