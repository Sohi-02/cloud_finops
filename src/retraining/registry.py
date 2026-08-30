# ============================================================
# SAFE MLFLOW MODEL REGISTRY DECISIONS
# ============================================================

from typing import Any


def _validate_non_empty_string(
    name: str,
    value: Any
) -> str:

    if (
        not isinstance(value, str)
        or not value.strip()
    ):

        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _set_version_tag(
    registry_client,
    registered_model_name: str,
    model_version: str,
    key: str,
    value: Any
):

    registry_client.set_model_version_tag(
        name=registered_model_name,
        version=model_version,
        key=key,
        value=str(value)
    )


def apply_candidate_registry_decision(
    registry_client,
    registered_model_name: str,
    candidate_version,
    promotion_evaluation: dict[str, Any],
    allow_champion_promotion: bool = False
) -> dict[str, Any]:
    """
    Apply an evaluated candidate decision to MLflow aliases.

    This function assumes the candidate model has already been
    logged and registered.

    Promotion requires both:

    1. The evaluation gate recommends promotion.
    2. allow_champion_promotion is explicitly True.
    """

    if registry_client is None:

        raise ValueError(
            "registry_client is required."
        )

    normalized_model_name = (
        _validate_non_empty_string(
            name="registered_model_name",
            value=registered_model_name
        )
    )

    normalized_candidate_version = (
        _validate_non_empty_string(
            name="candidate_version",
            value=str(candidate_version)
        )
    )

    if not isinstance(
        promotion_evaluation,
        dict
    ):

        raise TypeError(
            "promotion_evaluation must be "
            "a dictionary."
        )

    promote_candidate = (
        promotion_evaluation.get(
            "promote_candidate"
        )
    )

    if not isinstance(
        promote_candidate,
        bool
    ):

        raise ValueError(
            "promotion_evaluation must contain "
            "a boolean promote_candidate field."
        )

    if not isinstance(
        allow_champion_promotion,
        bool
    ):

        raise TypeError(
            "allow_champion_promotion must "
            "be boolean."
        )

    candidate_mae = (
        promotion_evaluation.get(
            "candidate_mae"
        )
    )

    champion_mae = (
        promotion_evaluation.get(
            "champion_mae"
        )
    )

    improvement_percent = (
        promotion_evaluation.get(
            "improvement_percent"
        )
    )

    # --------------------------------------------------------
    # 1. Assign challenger alias for traceability
    # --------------------------------------------------------

    registry_client.set_registered_model_alias(
        name=normalized_model_name,
        alias="challenger",
        version=(
            normalized_candidate_version
        )
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            normalized_candidate_version
        ),
        key="candidate_test_mae",
        value=candidate_mae
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            normalized_candidate_version
        ),
        key="champion_reference_mae",
        value=champion_mae
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            normalized_candidate_version
        ),
        key="improvement_percent",
        value=improvement_percent
    )

    # --------------------------------------------------------
    # 2. Rejected candidate: preserve champion
    # --------------------------------------------------------

    if not promote_candidate:

        _set_version_tag(
            registry_client=registry_client,
            registered_model_name=(
                normalized_model_name
            ),
            model_version=(
                normalized_candidate_version
            ),
            key="lifecycle_role",
            value="rejected_challenger"
        )

        _set_version_tag(
            registry_client=registry_client,
            registered_model_name=(
                normalized_model_name
            ),
            model_version=(
                normalized_candidate_version
            ),
            key="promotion_decision",
            value="rejected"
        )

        return {
            "registry_decision": (
                "candidate_rejected"
            ),
            "registered_model": (
                normalized_model_name
            ),
            "candidate_version": (
                normalized_candidate_version
            ),
            "champion_changed": False,
            "previous_champion_version": None,
            "champion_version": None,
            "challenger_version": (
                normalized_candidate_version
            )
        }

    # --------------------------------------------------------
    # 3. Approved but explicit promotion not enabled
    # --------------------------------------------------------

    if not allow_champion_promotion:

        _set_version_tag(
            registry_client=registry_client,
            registered_model_name=(
                normalized_model_name
            ),
            model_version=(
                normalized_candidate_version
            ),
            key="lifecycle_role",
            value="approved_challenger"
        )

        _set_version_tag(
            registry_client=registry_client,
            registered_model_name=(
                normalized_model_name
            ),
            model_version=(
                normalized_candidate_version
            ),
            key="promotion_decision",
            value="approved_pending_promotion"
        )

        return {
            "registry_decision": (
                "promotion_pending"
            ),
            "registered_model": (
                normalized_model_name
            ),
            "candidate_version": (
                normalized_candidate_version
            ),
            "champion_changed": False,
            "previous_champion_version": None,
            "champion_version": None,
            "challenger_version": (
                normalized_candidate_version
            )
        }

    # --------------------------------------------------------
    # 4. Resolve and preserve current champion
    # --------------------------------------------------------

    current_champion = (
        registry_client
        .get_model_version_by_alias(
            name=normalized_model_name,
            alias="champion"
        )
    )

    current_champion_version = str(
        current_champion.version
    )

    if (
        current_champion_version
        == normalized_candidate_version
    ):

        _set_version_tag(
            registry_client=registry_client,
            registered_model_name=(
                normalized_model_name
            ),
            model_version=(
                normalized_candidate_version
            ),
            key="lifecycle_role",
            value="production_champion"
        )

        return {
            "registry_decision": (
                "already_champion"
            ),
            "registered_model": (
                normalized_model_name
            ),
            "candidate_version": (
                normalized_candidate_version
            ),
            "champion_changed": False,
            "previous_champion_version": None,
            "champion_version": (
                normalized_candidate_version
            ),
            "challenger_version": (
                normalized_candidate_version
            )
        }

    registry_client.set_registered_model_alias(
        name=normalized_model_name,
        alias="previous_champion",
        version=current_champion_version
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            current_champion_version
        ),
        key="lifecycle_role",
        value="previous_champion"
    )

    # --------------------------------------------------------
    # 5. Promote the approved challenger
    # --------------------------------------------------------

    registry_client.set_registered_model_alias(
        name=normalized_model_name,
        alias="champion",
        version=(
            normalized_candidate_version
        )
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            normalized_candidate_version
        ),
        key="lifecycle_role",
        value="production_champion"
    )

    _set_version_tag(
        registry_client=registry_client,
        registered_model_name=(
            normalized_model_name
        ),
        model_version=(
            normalized_candidate_version
        ),
        key="promotion_decision",
        value="promoted"
    )

    return {
        "registry_decision": (
            "candidate_promoted"
        ),
        "registered_model": (
            normalized_model_name
        ),
        "candidate_version": (
            normalized_candidate_version
        ),
        "champion_changed": True,
        "previous_champion_version": (
            current_champion_version
        ),
        "champion_version": (
            normalized_candidate_version
        ),
        "challenger_version": (
            normalized_candidate_version
        )
    }