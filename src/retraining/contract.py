# MODEL SERVING CONTRACT COMPATIBILITY

from typing import Any


def evaluate_model_contract(
    deployment_manifest: dict[str, Any],
    candidate_feature_columns,
    candidate_output: str = (
        "predicted_next_hour_cost"
    )
) -> dict[str, Any]:

    if not isinstance(
        deployment_manifest,
        dict
    ):

        raise TypeError(
            "deployment_manifest must be a dictionary."
        )

    deployed_schema = (
        deployment_manifest.get(
            "input_schema"
        )
    )

    if not isinstance(
        deployed_schema,
        dict
    ):

        raise ValueError(
            "Manifest input_schema is missing."
        )

    deployed_features = list(
        deployed_schema.keys()
    )

    candidate_features = list(
        candidate_feature_columns
    )

    if (
        not candidate_features
        or not all(
            isinstance(feature, str)
            and feature
            for feature in candidate_features
        )
    ):

        raise ValueError(
            "Candidate features are invalid."
        )

    if len(
        set(candidate_features)
    ) != len(candidate_features):

        raise ValueError(
            "Candidate features contain duplicates."
        )

    deployed_output = (
        deployment_manifest.get(
            "output"
        )
    )

    missing_features = sorted(
        set(deployed_features)
        - set(candidate_features)
    )

    extra_features = sorted(
        set(candidate_features)
        - set(deployed_features)
    )

    feature_order_matches = (
        deployed_features
        == candidate_features
    )

    output_matches = (
        deployed_output
        == candidate_output
    )

    contract_compatible = (
        not missing_features
        and not extra_features
        and feature_order_matches
        and output_matches
    )

    if contract_compatible:

        message = (
            "Candidate matches the deployed "
            "serving contract."
        )

    else:

        message = (
            "Candidate cannot replace the deployed "
            "champion until a compatible serving "
            "endpoint is available."
        )

    return {
        "contract_compatible": (
            contract_compatible
        ),
        "deployed_features": (
            deployed_features
        ),
        "candidate_features": (
            candidate_features
        ),
        "deployed_feature_count": len(
            deployed_features
        ),
        "candidate_feature_count": len(
            candidate_features
        ),
        "missing_features": (
            missing_features
        ),
        "extra_features": (
            extra_features
        ),
        "feature_order_matches": (
            feature_order_matches
        ),
        "deployed_output": (
            deployed_output
        ),
        "candidate_output": (
            candidate_output
        ),
        "output_matches": (
            output_matches
        ),
        "promotion_allowed": (
            contract_compatible
        ),
        "message": message
    }