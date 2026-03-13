"""
Dataset management API endpoints.
"""

import os
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.tokenizer.bpe_tokenizer import get_tokenizer
from middleware.auth_middleware import get_current_user
from middleware.rate_limit import limiter
from models.database import User

from api.schemas.training import DatasetMetadata, DatasetDetail

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 10 * 1024  # 10KB


class CurrentDatasetResponse(BaseModel):
    """Response for getting the user's current dataset."""
    dataset: Optional[DatasetMetadata] = None


@router.get("/current", response_model=CurrentDatasetResponse)
async def get_current_dataset(
    current_user: User = Depends(get_current_user),
):
    """Get the user's currently uploaded dataset, if any."""
    from services.modal_volumes import list_user_datasets

    datasets = list_user_datasets(current_user.id)
    if not datasets:
        return CurrentDatasetResponse(dataset=None)
    return CurrentDatasetResponse(dataset=DatasetMetadata(**datasets[-1]))


@router.post("/upload", response_model=DatasetMetadata)
@limiter.limit("5/minute")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a text dataset file directly to Modal volume storage.
    Overwrites any existing user dataset.

    - Maximum file size: 50MB
    - Minimum file size: 10KB
    - Accepted formats: .txt (UTF-8 encoded)
    """
    from services.modal_volumes import upload_dataset_to_volume, delete_user_datasets

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

    # Generate dataset ID
    dataset_id = str(uuid.uuid4())[:8]

    # Create metadata
    metadata = {
        "dataset_id": dataset_id,
        "name": file.filename,
        "size_bytes": size_bytes,
        "num_tokens": num_tokens,
        "created_at": datetime.utcnow().isoformat(),
    }
    metadata_json = json.dumps(metadata, indent=2, default=str).encode("utf-8")

    # Delete any existing user datasets before uploading new one
    await delete_user_datasets(current_user.id)

    # Upload directly to Modal volume (no local storage)
    success = await upload_dataset_to_volume(
        user_id=current_user.id,
        dataset_id=dataset_id,
        content=content,
        metadata_json=metadata_json,
    )
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to upload dataset to storage"
        )

    return DatasetMetadata(**metadata)


# User dataset listing removed - user-uploaded datasets are now session-only
# Generic datasets (available to all users) are listed from Modal volume


class GenericDatasetInfo(BaseModel):
    """Information about a generic dataset."""
    id: str  # e.g., "generic:tinystories"
    name: str  # e.g., "Tinystories"
    description: Optional[str] = None


# Hardcoded generic datasets that are always available
COMMON_GENERIC_DATASETS = [
    GenericDatasetInfo(id="generic:tinystories", name="TinyStories", description="Short children's stories for small language models"),
    GenericDatasetInfo(id="generic:openwebtext", name="OpenWebText", description="Open-source recreation of the WebText dataset"),
    GenericDatasetInfo(id="generic:shakespeare", name="Shakespeare", description="Complete works of William Shakespeare"),
    GenericDatasetInfo(id="generic:wikitext", name="WikiText-103", description="Curated collection of Wikipedia articles"),
]


class GenericDatasetListResponse(BaseModel):
    """Response for listing generic datasets."""
    datasets: list[GenericDatasetInfo]


@router.get("/generic/list", response_model=GenericDatasetListResponse)
async def list_generic_datasets_endpoint():
    """
    List all generic datasets available for training.

    Returns hardcoded common datasets, supplemented by any additional
    datasets found in the Modal volume.
    """
    from services.modal_volumes import list_generic_datasets

    # Start with hardcoded datasets
    known_ids = {d.id for d in COMMON_GENERIC_DATASETS}
    result = list(COMMON_GENERIC_DATASETS)

    # Add any additional datasets from volume that aren't already in the list
    try:
        volume_datasets = list_generic_datasets()
        for d in volume_datasets:
            if d["id"] not in known_ids:
                result.append(GenericDatasetInfo(id=d["id"], name=d["name"], description=None))
    except Exception:
        pass

    return GenericDatasetListResponse(datasets=result)


@router.get("/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get dataset details including a preview of the content."""
    from services.modal_volumes import read_dataset_metadata_from_volume, read_dataset_from_volume

    # Validate dataset_id to prevent path traversal
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_id):
        raise HTTPException(status_code=400, detail="Invalid dataset ID")

    metadata = read_dataset_metadata_from_volume(current_user.id, dataset_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Read preview from Modal volume (first 500 bytes)
    preview_bytes = read_dataset_from_volume(current_user.id, dataset_id, max_bytes=500)
    preview = preview_bytes.decode("utf-8", errors="replace") if preview_bytes else ""

    return DatasetDetail(**metadata, preview=preview)
