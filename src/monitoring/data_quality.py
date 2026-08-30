# ============================================================
# FINOPS PRODUCTION DATA-QUALITY CHECKS
# ============================================================

import math

from numbers import Real


def validate_input_features(
    input_features,
    required_features,
    feature_ranges=None
):
    """
    Validates one production model-input record.

    Parameters
    ----------
    input_features:
        Dictionary containing the received feature values.

    required_features:
        Features required by the deployed model.

    feature_ranges:
        Optional minimum and maximum limits for features.

    Returns
    -------
    dict:
        Validation result containing errors and warnings.
    """

    if feature_ranges is None:

        feature_ranges = {}

    errors = []

    warnings = []

    if not isinstance(
        input_features,
        dict
    ):

        return {
            "passed": False,
            "checked_feature_count": 0,
            "errors": [
                "Input features must be a dictionary."
            ],
            "warnings": []
        }

    # --------------------------------------------------------
    # Missing required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_features
    ]

    if missing_features:

        errors.append(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Unexpected extra features
    # --------------------------------------------------------

    unexpected_features = [
        feature
        for feature in input_features
        if feature not in required_features
    ]

    if unexpected_features:

        warnings.append(
            "Unexpected features received: "
            + ", ".join(unexpected_features)
        )

    # --------------------------------------------------------
    # Validate each required feature
    # --------------------------------------------------------

    for feature in required_features:

        if feature not in input_features:

            continue

        value = input_features[
            feature
        ]

        # Boolean values are technically integers in Python,
        # but should not be accepted as cost values.
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):

            errors.append(
                f"{feature} must be numeric."
            )

            continue

        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):

            errors.append(
                f"{feature} must be finite."
            )

            continue

        range_rules = feature_ranges.get(
            feature,
            {}
        )

        minimum = range_rules.get(
            "minimum"
        )

        maximum = range_rules.get(
            "maximum"
        )

        if (
            minimum is not None
            and numeric_value < minimum
        ):

            errors.append(
                f"{feature} must be greater than "
                f"or equal to {minimum}."
            )

        if (
            maximum is not None
            and numeric_value > maximum
        ):

            errors.append(
                f"{feature} must be less than "
                f"or equal to {maximum}."
            )

    return {
        "passed": len(errors) == 0,
        "checked_feature_count": len(
            required_features
        ),
        "errors": errors,
        "warnings": warnings
    }