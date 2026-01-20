"""
Modal volume management for datasets and models.

Provides utilities for uploading datasets to Modal volumes and
retrieving trained models.
"""
import os
from pathlib import Path
from typing import Optional

import modal

# Volume names
DATASETS_VOLUME_NAME = "ozera-datasets"
MODELS_VOLUME_NAME = "ozera-models"

# Create or get volumes
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


def get_dataset_path(user_id: int, dataset_id: str) -> str:
    """
    Get the path for a dataset in the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID

    Returns:
        Path string: /datasets/{user_id}/{dataset_id}/raw.txt
    """
    return f"/datasets/{user_id}/{dataset_id}/raw.txt"


def get_model_path(user_id: int, model_name: str) -> str:
    """
    Get the path for a model in the Modal volume.

    Args:
        user_id: User ID
        model_name: Model name

    Returns:
        Path string: /models/{user_id}/{model_name}/
    """
    return f"/models/{user_id}/{model_name}/"


async def upload_dataset_to_volume(
    local_path: Path,
    user_id: int,
    dataset_id: str,
) -> bool:
    """
    Upload a dataset file to the Modal volume.

    Args:
        local_path: Local path to the dataset file
        user_id: User ID
        dataset_id: Dataset ID

    Returns:
        True if successful
    """
    try:
        remote_dir = f"/{user_id}/{dataset_id}"

        with datasets_volume.batch_upload() as batch:
            batch.put_file(str(local_path), f"{remote_dir}/raw.txt")

        return True
    except Exception as e:
        print(f"Error uploading dataset to Modal volume: {e}")
        return False


async def check_dataset_exists(user_id: int, dataset_id: str) -> bool:
    """
    Check if a dataset exists in the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID

    Returns:
        True if dataset exists
    """
    try:
        remote_path = f"/{user_id}/{dataset_id}/raw.txt"
        # List files in the volume to check existence
        for entry in datasets_volume.listdir(f"/{user_id}/{dataset_id}"):
            if entry.path.endswith("raw.txt"):
                return True
        return False
    except Exception:
        return False


async def download_model_from_volume(
    user_id: int,
    model_name: str,
    local_dir: Path,
) -> bool:
    """
    Download a trained model from the Modal volume.

    Args:
        user_id: User ID
        model_name: Model name
        local_dir: Local directory to save the model

    Returns:
        True if successful
    """
    try:
        remote_dir = f"/{user_id}/{model_name}"
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download all files from the model directory
        for entry in models_volume.listdir(remote_dir):
            # FileEntry has type attribute: 'file' or 'directory'
            if entry.type == "file":
                local_file = local_dir / Path(entry.path).name
                # Read from volume and write locally
                with open(local_file, "wb") as f:
                    for chunk in models_volume.read_file(entry.path):
                        f.write(chunk)

        return True
    except Exception as e:
        print(f"Error downloading model from Modal volume: {e}")
        return False


def get_volume_mounts() -> dict:
    """
    Get the volume mount configuration for Modal functions.

    Returns:
        Dict mapping mount paths to volumes
    """
    return {
        "/datasets": datasets_volume,
        "/models": models_volume,
    }
