"""
Dataset management API endpoints.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.tokenizer.bpe_tokenizer import get_tokenizer

from api.schemas.training import DatasetMetadata, DatasetDetail

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 10 * 1024  # 10KB


def get_dataset_dir(dataset_id: str) -> Path:
    """Get the directory for a dataset."""
    return DATA_DIR / dataset_id


def load_dataset_metadata(dataset_id: str) -> Optional[dict]:
    """Load dataset metadata from JSON file."""
    metadata_path = get_dataset_dir(dataset_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_metadata(dataset_id: str, metadata: dict) -> None:
    """Save dataset metadata to JSON file."""
    dataset_dir = get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


# list_all_datasets removed - datasets are session-only


@router.post("/upload", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a text dataset file.

    - Maximum file size: 50MB
    - Minimum file size: 10KB
    - Accepted formats: .txt (UTF-8 encoded)
    """
    # Validate file extension
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are accepted"
        )

    # Read file content
    content = await file.read()

    # Validate file size
    size_bytes = len(content)
    if size_bytes > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    if size_bytes < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too small. Minimum size is {MIN_FILE_SIZE // 1024}KB"
        )

    # Decode as UTF-8
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded text"
        )

    # Check for binary content
    if "\x00" in text:
        raise HTTPException(
            status_code=400,
            detail="File appears to be binary, not text"
        )

    # Count tokens using BPE tokenizer
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    # Generate dataset ID and save
    dataset_id = str(uuid.uuid4())[:8]
    dataset_dir = get_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save raw text file
    raw_path = dataset_dir / "raw.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Create metadata
    metadata = {
        "dataset_id": dataset_id,
        "name": file.filename,
        "size_bytes": size_bytes,
        "num_tokens": num_tokens,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_dataset_metadata(dataset_id, metadata)

    return DatasetMetadata(**metadata)


# User dataset listing removed - user-uploaded datasets are now session-only
# Generic datasets (available to all users) are listed from Modal volume


class GenericDatasetInfo(BaseModel):
    """Information about a generic dataset."""
    id: str  # e.g., "generic:tinystories"
    name: str  # e.g., "Tinystories"
    description: Optional[str] = None


class GenericDatasetListResponse(BaseModel):
    """Response for listing generic datasets."""
    datasets: list[GenericDatasetInfo]


@router.get("/generic/list", response_model=GenericDatasetListResponse)
async def list_generic_datasets_endpoint():
    """
    List all generic datasets available for training.

    Generic datasets are pre-uploaded datasets (e.g., TinyStories, OpenWebText)
    that are available to all users for model training.
    """
    from services.modal_volumes import list_generic_datasets

    datasets = list_generic_datasets()

    return GenericDatasetListResponse(
        datasets=[
            GenericDatasetInfo(
                id=d["id"],
                name=d["name"],
                description=None,  # Can be extended to include descriptions
            )
            for d in datasets
        ]
    )


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get dataset details including a preview of the content."""
    metadata = load_dataset_metadata(dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Read preview
    raw_path = get_dataset_dir(dataset_id) / "raw.txt"
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            preview = f.read(500)
    else:
        preview = ""

    return DatasetDetail(**metadata, preview=preview)


# delete_dataset endpoint removed - datasets are temporary and cleaned up automatically
