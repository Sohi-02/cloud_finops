# ============================================================
# SAFE MLFLOW REGISTRY DECISION TESTS
# ============================================================

from types import SimpleNamespace

from src.retraining.registry import (
    apply_candidate_registry_decision
)


class FakeRegistryClient:

    def __init__(self):

        self.aliases = {
            "champion": "1"
        }

        self.tags = {}

    def set_registered_model_alias(
        self,
        name,
        alias,
        version
    ):

        self.aliases[alias] = str(
            version
        )

    def get_model_version_by_alias(
        self,
        name,
        alias
    ):

        return SimpleNamespace(
            version=self.aliases[alias]
        )

    def set_model_version_tag(
        self,
        name,
        version,
        key,
        value
    ):

        version_key = str(
            version
        )

        if version_key not in self.tags:

            self.tags[
                version_key
            ] = {}

        self.tags[
            version_key
        ][key] = str(value)


def rejected_evaluation():

    return {
        "promote_candidate": False,
        "decision": "reject_candidate",
        "champion_mae": 7.8,
        "candidate_mae": 9.0,
        "improvement_percent": -15.38
    }


def approved_evaluation():

    return {
        "promote_candidate": True,
        "decision": "promote_candidate",
        "champion_mae": 10.0,
        "candidate_mae": 8.0,
        "improvement_percent": 20.0
    }


def test_rejected_candidate_preserves_champion():

    client = FakeRegistryClient()

    result = (
        apply_candidate_registry_decision(
            registry_client=client,
            registered_model_name=(
                "finops-model"
            ),
            candidate_version="2",
            promotion_evaluation=(
                rejected_evaluation()
            ),
            allow_champion_promotion=True
        )
    )

    assert (
        client.aliases["champion"]
        == "1"
    )

    assert (
        client.aliases["challenger"]
        == "2"
    )

    assert (
        result["champion_changed"]
        is False
    )

    assert (
        client.tags["2"][
            "promotion_decision"
        ]
        == "rejected"
    )


def test_approved_candidate_waits_without_permission():

    client = FakeRegistryClient()

    result = (
        apply_candidate_registry_decision(
            registry_client=client,
            registered_model_name=(
                "finops-model"
            ),
            candidate_version="2",
            promotion_evaluation=(
                approved_evaluation()
            ),
            allow_champion_promotion=False
        )
    )

    assert (
        client.aliases["champion"]
        == "1"
    )

    assert (
        client.aliases["challenger"]
        == "2"
    )

    assert (
        result["registry_decision"]
        == "promotion_pending"
    )

    assert (
        result["champion_changed"]
        is False
    )


def test_approved_candidate_preserves_previous_champion():

    client = FakeRegistryClient()

    result = (
        apply_candidate_registry_decision(
            registry_client=client,
            registered_model_name=(
                "finops-model"
            ),
            candidate_version="2",
            promotion_evaluation=(
                approved_evaluation()
            ),
            allow_champion_promotion=True
        )
    )

    assert (
        client.aliases[
            "previous_champion"
        ]
        == "1"
    )

    assert (
        client.aliases["champion"]
        == "2"
    )

    assert (
        client.aliases["challenger"]
        == "2"
    )

    assert (
        result["champion_changed"]
        is True
    )

    assert (
        result[
            "previous_champion_version"
        ]
        == "1"
    )


def test_candidate_already_champion_is_idempotent():

    client = FakeRegistryClient()

    client.aliases[
        "champion"
    ] = "2"

    result = (
        apply_candidate_registry_decision(
            registry_client=client,
            registered_model_name=(
                "finops-model"
            ),
            candidate_version="2",
            promotion_evaluation=(
                approved_evaluation()
            ),
            allow_champion_promotion=True
        )
    )

    assert (
        result["registry_decision"]
        == "already_champion"
    )

    assert (
        result["champion_changed"]
        is False
    )

    assert (
        "previous_champion"
        not in client.aliases
    )