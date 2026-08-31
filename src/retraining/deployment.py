# VERSIONED MODEL DEPLOYMENT BUNDLE EXPORT

import json
import shutil
import tempfile

from pathlib import Path
from typing import Any, Optional


def export_versioned_model_bundle(
    model_uri: str,
    destination_root,
    deployment_manifest: dict[str, Any],
    reference_profile: Optional[
        dict[str, Any]
    ] = None,
    artifact_downloader=None
) -> Path:
    """
    Export to a new versioned directory.

    Existing deployment directories are never overwritten.
    """

    if not isinstance(
        model_uri,
        str
    ) or not model_uri:

        raise ValueError(
            "model_uri is required."
        )

    if not isinstance(
        deployment_manifest,
        dict
    ):

        raise TypeError(
            "deployment_manifest must be "
            "a dictionary."
        )

    model_version = str(
        deployment_manifest.get(
            "model_version",
            ""
        )
    )

    if not model_version:

        raise ValueError(
            "Manifest model_version is required."
        )

    destination_root = Path(
        destination_root
    ).expanduser().resolve()

    destination_root.mkdir(
        parents=True,
        exist_ok=True
    )

    final_directory = (
        destination_root
        / f"version_{model_version}"
    )

    if final_directory.exists():

        raise FileExistsError(
            "Deployment version already exists: "
            f"{final_directory}"
        )

    if artifact_downloader is None:

        import mlflow

        artifact_downloader = (
            mlflow.artifacts
            .download_artifacts
        )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".finops_bundle_",
            dir=str(destination_root)
        )
    )

    try:

        download_directory = (
            staging_root
            / "download"
        )

        downloaded_path = Path(
            artifact_downloader(
                artifact_uri=model_uri,
                dst_path=str(
                    download_directory
                )
            )
        ).resolve()

        if not (
            downloaded_path
            / "MLmodel"
        ).exists():

            raise FileNotFoundError(
                "Downloaded artifact does not "
                "contain MLmodel."
            )

        bundle_directory = (
            staging_root
            / "bundle"
        )

        shutil.copytree(
            downloaded_path,
            bundle_directory
        )

        with (
            bundle_directory
            / "deployment_manifest.json"
        ).open(
            "w",
            encoding="utf-8"
        ) as manifest_file:

            json.dump(
                deployment_manifest,
                manifest_file,
                indent=2
            )

        if reference_profile is not None:

            with (
                bundle_directory
                / "reference_profile.json"
            ).open(
                "w",
                encoding="utf-8"
            ) as profile_file:

                json.dump(
                    reference_profile,
                    profile_file,
                    indent=2
                )

        bundle_directory.replace(
            final_directory
        )

    finally:

        if staging_root.exists():

            shutil.rmtree(
                staging_root,
                ignore_errors=True
            )

    return final_directory