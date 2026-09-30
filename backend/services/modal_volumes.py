"""
Modal volume management for datasets and models.

Provides utilities for uploading datasets to Modal volumes and
retrieving trained models.
"""
import os
from pathlib import Path
from typing import Optional

import modal

from core.model_names import model_folder

# Volume names
DATASETS_VOLUME_NAME = "ozera-datasets"
MODELS_VOLUME_NAME = "ozera-models"

# Create or get volumes
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)


# Generic datasets folder name (datasets available to all users)
GENERIC_DATASETS_FOLDER = "generic"


def get_dataset_path(user_id: int, dataset_id: str) -> str:
    """
    Get the path for a user-uploaded dataset in the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID

    Returns:
        Path string: /datasets/{user_id}/{dataset_id}/raw.txt
    """
    return f"/datasets/{user_id}/{dataset_id}/raw.txt"


def get_generic_dataset_path(dataset_id: str) -> str:
    """
    Get the path for a generic dataset in the Modal volume.

    Generic datasets are stored in /datasets/generic/{dataset_id}/raw.txt
    and are available to all users.

    Args:
        dataset_id: Dataset ID (e.g., 'tinystories', 'openwebtext')

    Returns:
        Path string: /datasets/generic/{dataset_id}/raw.txt
    """
    return f"/datasets/{GENERIC_DATASETS_FOLDER}/{dataset_id}/raw.txt"


def is_generic_dataset(dataset_id: str) -> bool:
    """
    Check if a dataset ID refers to a generic dataset.

    Generic dataset IDs are prefixed with 'generic:'.

    Args:
        dataset_id: Dataset ID

    Returns:
        True if this is a generic dataset
    """
    return dataset_id.startswith("generic:")


def parse_dataset_id(dataset_id: str) -> tuple[bool, str]:
    """
    Parse a dataset ID to determine if it's generic and get the actual ID.

    Args:
        dataset_id: Dataset ID (e.g., 'generic:tinystories' or 'abc123')

    Returns:
        Tuple of (is_generic, actual_dataset_id)
    """
    if dataset_id.startswith("generic:"):
        return True, dataset_id[8:]  # Remove 'generic:' prefix
    return False, dataset_id


def get_model_path(user_id: int, model_name: str) -> str:
    """
    Get the path for a model in the Modal volume.

    Args:
        user_id: User ID
        model_name: Model name

    Returns:
        Path string: /models/{user_id}/{model_name}/
    """
    return f"/models/{model_folder(user_id, model_name)}/"


async def upload_dataset_to_volume(
    user_id: int,
    dataset_id: str,
    content: bytes,
    metadata_json: Optional[bytes] = None,
) -> bool:
    """
    Upload dataset content directly to the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID
        content: Raw dataset content as bytes
        metadata_json: Optional metadata JSON as bytes

    Returns:
        True if successful
    """
    import tempfile
    try:
        remote_dir = f"/{user_id}/{dataset_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.txt"
            raw_path.write_bytes(content)

            with datasets_volume.batch_upload() as batch:
                batch.put_file(str(raw_path), f"{remote_dir}/raw.txt")

                if metadata_json is not None:
                    meta_path = Path(tmpdir) / "metadata.json"
                    meta_path.write_bytes(metadata_json)
                    batch.put_file(str(meta_path), f"{remote_dir}/metadata.json")

        return True
    except Exception as e:
        print(f"Error uploading dataset to Modal volume: {e}")
        return False


def read_dataset_from_volume(user_id: int, dataset_id: str, max_bytes: Optional[int] = None) -> Optional[bytes]:
    """
    Read dataset content from the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID
        max_bytes: If set, only read up to this many bytes

    Returns:
        Dataset content as bytes, or None if not found
    """
    try:
        remote_path = f"/{user_id}/{dataset_id}/raw.txt"
        content = b""
        for chunk in datasets_volume.read_file(remote_path):
            content += chunk
            if max_bytes and len(content) >= max_bytes:
                return content[:max_bytes]
        return content
    except Exception:
        return None


def read_dataset_metadata_from_volume(user_id: int, dataset_id: str) -> Optional[dict]:
    """
    Read dataset metadata from the Modal volume.

    Args:
        user_id: User ID
        dataset_id: Dataset ID

    Returns:
        Metadata dict, or None if not found
    """
    import json
    try:
        remote_path = f"/{user_id}/{dataset_id}/metadata.json"
        content = b""
        for chunk in datasets_volume.read_file(remote_path):
            content += chunk
        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


def list_user_datasets(user_id: int) -> list[dict]:
    """
    List all datasets uploaded by a user.

    Returns:
        List of metadata dicts for each dataset, or empty list.
    """
    import json
    datasets = []
    try:
        for entry in datasets_volume.listdir(f"/{user_id}"):
            dataset_id = entry.path.strip("/").split("/")[-1]
            try:
                meta_content = b""
                for chunk in datasets_volume.read_file(f"/{user_id}/{dataset_id}/metadata.json"):
                    meta_content += chunk
                metadata = json.loads(meta_content.decode("utf-8"))
                datasets.append(metadata)
            except Exception:
                pass
    except Exception:
        pass
    return datasets


async def delete_user_datasets(user_id: int) -> bool:
    """
    Delete all datasets for a user (to make room for a new upload).
    """
    try:
        for entry in datasets_volume.listdir(f"/{user_id}"):
            dataset_id = entry.path.strip("/").split("/")[-1]
            try:
                datasets_volume.remove_file(f"/{user_id}/{dataset_id}", recursive=True)
            except Exception:
                pass
        return True
    except Exception:
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


async def check_generic_dataset_exists(dataset_id: str) -> bool:
    """
    Check if a generic dataset exists in the Modal volume.

    Args:
        dataset_id: Dataset ID (without 'generic:' prefix)

    Returns:
        True if dataset exists
    """
    try:
        for entry in datasets_volume.listdir(f"/{GENERIC_DATASETS_FOLDER}/{dataset_id}"):
            if entry.path.endswith("raw.txt"):
                return True
        return False
    except Exception:
        return False


def list_generic_datasets() -> list[dict]:
    """
    List all generic datasets available in the Modal volume.

    Returns:
        List of dicts with dataset info: {id, name, path}
    """
    datasets = []
    try:
        # List directories in /generic/
        for entry in datasets_volume.listdir(f"/{GENERIC_DATASETS_FOLDER}"):
            if entry.type == "directory":
                dataset_id = entry.path.strip("/").split("/")[-1]
                # Check if raw.txt exists
                try:
                    for file_entry in datasets_volume.listdir(entry.path):
                        if file_entry.path.endswith("raw.txt"):
                            datasets.append({
                                "id": f"generic:{dataset_id}",
                                "name": dataset_id.replace("-", " ").replace("_", " ").title(),
                                "path": f"/{GENERIC_DATASETS_FOLDER}/{dataset_id}/raw.txt",
                            })
                            break
                except Exception:
                    pass
    except Exception as e:
        print(f"Error listing generic datasets: {e}")
    return datasets


def get_generic_dataset_metadata(dataset_id: str) -> Optional[dict]:
    """
    Get metadata for a generic dataset.

    First checks for a metadata.json file in the dataset folder.
    If not found, returns basic metadata with the dataset name.

    Args:
        dataset_id: Dataset ID (without 'generic:' prefix)

    Returns:
        Dict with metadata or None if dataset not found
    """
    import json

    try:
        dataset_dir = f"/{GENERIC_DATASETS_FOLDER}/{dataset_id}"

        # Check if dataset exists
        has_raw_txt = False
        has_metadata = False

        for entry in datasets_volume.listdir(dataset_dir):
            if entry.path.endswith("raw.txt"):
                has_raw_txt = True
            if entry.path.endswith("metadata.json"):
                has_metadata = True

        if not has_raw_txt:
            return None

        # Try to read metadata.json if it exists
        if has_metadata:
            try:
                metadata_content = b""
                for chunk in datasets_volume.read_file(f"{dataset_dir}/metadata.json"):
                    metadata_content += chunk
                metadata = json.loads(metadata_content.decode("utf-8"))
                # Ensure required fields
                metadata["dataset_id"] = f"generic:{dataset_id}"
                metadata["name"] = metadata.get("name", dataset_id.replace("-", " ").replace("_", " ").title())
                return metadata
            except Exception:
                pass

        # Return basic metadata
        return {
            "dataset_id": f"generic:{dataset_id}",
            "name": dataset_id.replace("-", " ").replace("_", " ").title(),
            "num_tokens": 0,  # Unknown without reading the file
            "size_bytes": 0,
        }

    except Exception as e:
        print(f"Error getting generic dataset metadata: {e}")
        return None


def download_model_from_volume(
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
        remote_dir = f"/{model_folder(user_id, model_name)}"
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


def get_uploaded_model_path(user_id: int, model_name: str) -> str:
    """
    Get the path for an uploaded model in the Modal volume.

    Uploaded models use safetensors format and are stored separately.

    Args:
        user_id: User ID
        model_name: Model name

    Returns:
        Path string: /models/{user_id}/{model_name}/model.safetensors
    """
    return f"/{model_folder(user_id, model_name)}/model.safetensors"


async def upload_model_to_volume(
    local_path: Path,
    user_id: int,
    model_name: str,
) -> bool:
    """
    Upload a .safetensors model file to the Modal volume.

    Args:
        local_path: Local path to the model file
        user_id: User ID
        model_name: Model name

    Returns:
        True if successful
    """
    try:
        remote_dir = f"/{model_folder(user_id, model_name)}"

        with models_volume.batch_upload() as batch:
            batch.put_file(str(local_path), f"{remote_dir}/model.safetensors")

        return True
    except Exception as e:
        print(f"Error uploading model to Modal volume: {e}")
        return False


async def delete_model_from_volume(user_id: int, model_name: str) -> bool:
    """
    Delete a model directory from the Modal volume.

    Args:
        user_id: User ID
        model_name: Model name

    Returns:
        True if the directory is gone (including if it didn't exist)
    """
    try:
        remote_dir = f"/{model_folder(user_id, model_name)}"

        # Delete the entire directory recursively
        await models_volume.remove_file.aio(remote_dir, recursive=True)

        return True
    except FileNotFoundError:
        return True
    except Exception as e:
        print(f"Error deleting model from Modal volume: {e}")
        return False


def check_model_exists_in_volume(user_id: int, model_name: str) -> bool:
    """
    Check if a model exists in the Modal volume.

    Checks for both .pt (trained) and .safetensors (uploaded) formats.

    Args:
        user_id: User ID
        model_name: Model name

    Returns:
        True if model exists
    """
    try:
        remote_dir = f"/{model_folder(user_id, model_name)}"
        for entry in models_volume.listdir(remote_dir):
            if entry.path.endswith(".pt") or entry.path.endswith(".safetensors"):
                return True
        return False
    except Exception:
        return False
