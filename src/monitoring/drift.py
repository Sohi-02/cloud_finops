"""Production input-drift monitoring using PSI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MODERATE_THRESHOLD = 0.10
DEFAULT_SIGNIFICANT_THRESHOLD = 0.25
DEFAULT_MINIMUM_SAMPLES = 30
PSI_EPSILON = 1e-6


def _clean_numeric_values(
    values: Iterable[Any]
) -> np.ndarray:
    """
    Convert values to finite floats.

    Missing, invalid, infinite and boolean values are ignored.
    """

    cleaned_values: list[float] = []

    for value in values:

        if value is None or isinstance(value, bool):
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if np.isfinite(numeric_value):
            cleaned_values.append(numeric_value)

    return np.asarray(
        cleaned_values,
        dtype="float64"
    )


def load_reference_profile(
    profile_path: str | Path
) -> dict[str, Any]:
    """
    Load and perform basic validation of reference_profile.json.
    """

    resolved_path = Path(profile_path).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Reference profile not found: {resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8"
    ) as profile_file:

        profile = json.load(profile_file)

    required_sections = {
        "schema_version",
        "minimum_production_samples",
        "psi_thresholds",
        "features"
    }

    missing_sections = (
        required_sections - profile.keys()
    )

    if missing_sections:
        raise ValueError(
            "Reference profile is missing sections: "
            f"{sorted(missing_sections)}"
        )

    if not isinstance(profile["features"], dict):
        raise ValueError(
            "Reference profile 'features' must be an object."
        )

    return profile


def calculate_psi_from_profile(
    feature_profile: dict[str, Any],
    production_values: Iterable[Any]
) -> dict[str, Any]:
    """
    Calculate PSI using precomputed training bins.

    The training dataset itself is not required because its bin
    edges and expected proportions are stored in the profile.
    """

    production_array = _clean_numeric_values(
        production_values
    )

    if production_array.size == 0:
        raise ValueError(
            "No valid production values were provided."
        )

    internal_edges = np.asarray(
        feature_profile["internal_bin_edges"],
        dtype="float64"
    )

    expected_proportions = np.asarray(
        feature_profile["expected_bin_proportions"],
        dtype="float64"
    )

    if not np.all(np.isfinite(internal_edges)):
        raise ValueError(
            "Reference bin edges must contain finite numbers."
        )

    if (
        internal_edges.size > 1
        and np.any(np.diff(internal_edges) <= 0)
    ):
        raise ValueError(
            "Reference bin edges must be strictly increasing."
        )

    histogram_edges = np.concatenate([
        [-np.inf],
        internal_edges,
        [np.inf]
    ])

    expected_bin_count = len(histogram_edges) - 1

    if len(expected_proportions) != expected_bin_count:
        raise ValueError(
            "Expected proportions do not match the "
            "number of reference bins."
        )

    if not np.isclose(
        expected_proportions.sum(),
        1.0
    ):
        raise ValueError(
            "Expected reference proportions must sum to 1."
        )

    production_counts, _ = np.histogram(
        production_array,
        bins=histogram_edges
    )

    production_proportions = (
        production_counts
        / production_counts.sum()
    )

    expected_safe = np.clip(
        expected_proportions,
        PSI_EPSILON,
        None
    )

    production_safe = np.clip(
        production_proportions,
        PSI_EPSILON,
        None
    )

    psi_by_bin = (
        production_safe - expected_safe
    ) * np.log(
        production_safe / expected_safe
    )

    psi_value = float(
        np.sum(psi_by_bin)
    )

    return {
        "psi": psi_value,
        "production_sample_count": int(
            production_array.size
        ),
        "production_bin_counts": (
            production_counts.astype(int).tolist()
        ),
        "production_bin_proportions": (
            production_proportions.tolist()
        ),
        "expected_bin_proportions": (
            expected_proportions.tolist()
        ),
        "psi_by_bin": psi_by_bin.tolist()
    }


def calculate_profile_drift_report(
    reference_profile: dict[str, Any],
    production_values: Iterable[Any],
    feature_name: str = "estimated_cost_index"
) -> dict[str, Any]:
    """
    Create a production-ready drift report for one feature.
    """

    if feature_name not in reference_profile["features"]:
        raise KeyError(
            "Feature is not present in reference profile: "
            f"{feature_name}"
        )

    feature_profile = (
        reference_profile["features"][feature_name]
    )

    cleaned_production_values = _clean_numeric_values(
        production_values
    )

    minimum_samples = int(
        reference_profile.get(
            "minimum_production_samples",
            DEFAULT_MINIMUM_SAMPLES
        )
    )

    moderate_threshold = float(
        reference_profile.get(
            "psi_thresholds",
            {}
        ).get(
            "moderate",
            DEFAULT_MODERATE_THRESHOLD
        )
    )

    significant_threshold = float(
        reference_profile.get(
            "psi_thresholds",
            {}
        ).get(
            "significant",
            DEFAULT_SIGNIFICANT_THRESHOLD
        )
    )

    base_report = {
        "feature": feature_name,
        "production_sample_count": int(
            cleaned_production_values.size
        ),
        "reference_sample_count": int(
            feature_profile["training_sample_count"]
        ),
        "minimum_required_samples": minimum_samples,
        "moderate_threshold": moderate_threshold,
        "significant_threshold": significant_threshold
    }

    if cleaned_production_values.size < minimum_samples:

        return {
            **base_report,
            "psi": None,
            "status": "insufficient_data",
            "drift_detected": False,
            "retraining_candidate": False,
            "message": (
                f"At least {minimum_samples} production "
                "samples are required for drift analysis."
            )
        }

    psi_result = calculate_psi_from_profile(
        feature_profile=feature_profile,
        production_values=cleaned_production_values
    )

    psi_value = psi_result["psi"]

    if psi_value < moderate_threshold:
        status = "stable"

    elif psi_value < significant_threshold:
        status = "moderate"

    else:
        status = "significant"

    return {
        **base_report,
        **psi_result,
        "status": status,
        "drift_detected": (
            status in {"moderate", "significant"}
        ),
        "retraining_candidate": (
            status == "significant"
        ),
        "message": (
            "Production input distribution compared "
            "successfully with the training reference."
        )
    }