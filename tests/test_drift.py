# ============================================================
# DRIFT MONITORING TESTS
# ============================================================

from pathlib import Path

import numpy as np

from src.monitoring.drift import (
    calculate_profile_drift_report,
    load_reference_profile
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

REFERENCE_PROFILE_PATH = (
    PROJECT_ROOT
    / "models"
    / "champion"
    / "reference_profile.json"
)


def build_reference_like_values(
    reference_profile,
    sample_count=1000
):
    """
    Create synthetic values with approximately the same bin
    proportions as the training reference.

    This tests the stable-distribution case without writing
    fake requests into MongoDB.
    """

    feature_profile = (
        reference_profile[
            "features"
        ][
            "estimated_cost_index"
        ]
    )

    internal_edges = np.asarray(
        feature_profile[
            "internal_bin_edges"
        ],
        dtype="float64"
    )

    expected_proportions = np.asarray(
        feature_profile[
            "expected_bin_proportions"
        ],
        dtype="float64"
    )

    raw_counts = (
        expected_proportions
        * sample_count
    )

    bin_counts = np.floor(
        raw_counts
    ).astype(int)

    remaining_samples = (
        sample_count
        - int(bin_counts.sum())
    )

    fractional_order = np.argsort(
        raw_counts - bin_counts
    )[::-1]

    for bin_index in fractional_order[
        :remaining_samples
    ]:

        bin_counts[bin_index] += 1

    representatives = []

    # Representative for the first open-ended bin
    first_width = max(
        abs(internal_edges[0]) * 0.10,
        1.0
    )

    representatives.append(
        internal_edges[0] - first_width
    )

    # Representatives for the internal bins
    for left_edge, right_edge in zip(
        internal_edges[:-1],
        internal_edges[1:]
    ):

        representatives.append(
            (left_edge + right_edge) / 2.0
        )

    # Representative for the final open-ended bin
    last_width = max(
        abs(internal_edges[-1]) * 0.10,
        1.0
    )

    representatives.append(
        internal_edges[-1] + last_width
    )

    generated_values = []

    for representative, count in zip(
        representatives,
        bin_counts
    ):

        generated_values.extend(
            [float(representative)] * int(count)
        )

    return generated_values


def test_reference_profile_loads():

    profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    assert profile["model_version"] == "1"

    assert (
        profile[
            "features"
        ][
            "estimated_cost_index"
        ][
            "training_sample_count"
        ]
        == 501
    )


def test_insufficient_production_data():

    profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    report = calculate_profile_drift_report(
        reference_profile=profile,
        production_values=[
            20.0,
            24.0,
            28.0
        ]
    )

    assert report["psi"] is None
    assert report["status"] == "insufficient_data"
    assert report["drift_detected"] is False
    assert report["retraining_candidate"] is False


def test_reference_like_distribution_is_stable():

    profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    production_values = (
        build_reference_like_values(
            reference_profile=profile,
            sample_count=1000
        )
    )

    report = calculate_profile_drift_report(
        reference_profile=profile,
        production_values=production_values
    )

    assert report["psi"] is not None
    assert report["psi"] < 0.10
    assert report["status"] == "stable"
    assert report["drift_detected"] is False
    assert report["retraining_candidate"] is False


def test_concentrated_distribution_detects_drift():

    profile = load_reference_profile(
        REFERENCE_PROFILE_PATH
    )

    # All production values are concentrated around one point,
    # unlike the full training distribution.
    production_values = [
        24.397998
    ] * 100

    report = calculate_profile_drift_report(
        reference_profile=profile,
        production_values=production_values
    )

    assert report["psi"] is not None
    assert report["psi"] >= 0.25
    assert report["status"] == "significant"
    assert report["drift_detected"] is True
    assert report["retraining_candidate"] is True