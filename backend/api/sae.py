"""
SAE API endpoints.

Proxy endpoints to Modal SAE inference service with authentication and credit deduction.
"""
import os
import httpx
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, List

from db import get_db
from middleware.auth_middleware import get_current_user
from models.database import User
from services.credit_service import charge_sae, check_sufficient_balance, estimate_sae_cost

# Get SAE API URL from environment
SAE_API_URL = os.getenv("SAE_API_URL", "https://adriansimon477--ozera-sae-inference-serve.modal.run")

router = APIRouter(prefix="/sae", tags=["sae"])


# Request/Response models
class SAEAnalyzeRequest(BaseModel):
    model: str = Field(default="nano", description="Model to use ('nano' or 'mini')")
    layer: int = Field(..., ge=0, description="Layer index")
    activation_type: str = Field(..., description="'residual' or 'mlp_output'")
    text: str = Field(..., description="Text to analyze")
    top_k: int = Field(default=20, ge=1, le=100, description="Top K features to return")


class SAECompareSAEsRequest(BaseModel):
    model_a: str = Field(default="nano")
    layer_a: int = Field(..., ge=0)
    activation_type_a: str = Field(...)
    model_b: str = Field(default="nano")
    layer_b: int = Field(..., ge=0)
    activation_type_b: str = Field(...)
    text: str = Field(...)
    top_k: int = Field(default=100, ge=1, le=200)


class SAECompareLayersRequest(BaseModel):
    model_a: str = Field(default="nano")
    activation_type_a: str = Field(...)
    model_b: Optional[str] = None
    activation_type_b: Optional[str] = None
    text: str = Field(...)


class SAEAnalyzeBatchRequest(BaseModel):
    model: str = Field(default="nano")
    layer: int = Field(..., ge=0)
    activation_type: str = Field(...)
    texts: List[str] = Field(..., min_length=1, max_length=32)
    top_k_per_text: int = Field(default=10, ge=1, le=50)


@router.get("/list")
async def list_saes():
    """
    List available SAEs.

    This endpoint is public (no auth required).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{SAE_API_URL}/sae/list")
        response.raise_for_status()
        return response.json()


@router.post("/analyze")
async def analyze_text(
    request: SAEAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Analyze text through transformer + SAE.

    Requires authentication. Charges credits based on token count.
    """
    # Estimate cost before running
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id=request.model,
        operation_type="analyze",
    )

    # Check sufficient balance
    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):  # 1.5x safety margin
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request to Modal SAE service
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/analyze",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    # Check if there was an error in the response
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge credits based on actual token count
    num_tokens = result.get("num_tokens", 1)
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=num_tokens,
        model_name=request.model,
        operation_type="analyze",
        description=f"SAE analysis: {request.model} L{request.layer} {request.activation_type}",
    )

    return result


@router.get("/feature")
async def get_feature_info(
    model: str = Query("nano"),
    layer: int = Query(0, ge=0),
    activation_type: str = Query("residual"),
    feature_id: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get information about a specific SAE feature.

    Requires authentication. Charges a small fee for feature lookup.
    """
    # This is a cheap operation, charge minimal fee
    min_cost = 0.001  # $0.001

    if not check_sufficient_balance(db, user.id, min_cost):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request to Modal SAE service
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{SAE_API_URL}/sae/feature",
                params={
                    "model": model,
                    "layer": layer,
                    "activation_type": activation_type,
                    "feature_id": feature_id,
                },
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge minimal fee
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=1,
        model_name=model,
        operation_type="feature_info",
        description=f"SAE feature lookup: {model} L{layer} feature {feature_id}",
    )

    return result


@router.post("/analyze-batch")
async def analyze_batch(
    request: SAEAnalyzeBatchRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Analyze multiple texts in batch.

    Requires authentication. Charges credits based on total token count.
    """
    # Estimate cost
    total_text_length = sum(len(text) for text in request.texts)
    estimated_cost = estimate_sae_cost(
        text_length=total_text_length,
        model_id=request.model,
        operation_type="analyze",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/analyze-batch",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge based on actual tokens
    total_tokens = sum(r.get("num_tokens", 0) for r in result.get("results", []))
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=max(total_tokens, 1),
        model_name=request.model,
        operation_type="analyze",
        description=f"SAE batch analysis: {len(request.texts)} texts",
    )

    return result


@router.post("/compare")
async def compare_saes(
    request: SAECompareSAEsRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Compare features between two SAEs.

    Requires authentication. Charges 2x cost (processing two SAEs).
    """
    # Estimate cost (2x for comparing two SAEs)
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id=request.model_a,
        operation_type="compare",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/compare",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge credits
    num_tokens = result.get("num_tokens", 1)
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=num_tokens,
        model_name=request.model_a,
        operation_type="compare",
        description=f"SAE comparison: {request.model_a} vs {request.model_b}",
    )

    return result


@router.post("/compare-layers")
async def compare_layers(
    request: SAECompareLayersRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Compare layers across SAEs.

    Requires authentication. Charges 3x cost (most intensive operation).
    """
    # Estimate cost (3x for layer-by-layer comparison)
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id=request.model_a,
        operation_type="compare_layers",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/compare-layers",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge credits
    num_tokens = result.get("num_tokens", 1)
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=num_tokens,
        model_name=request.model_a,
        operation_type="compare_layers",
        description="SAE layer-by-layer comparison",
    )

    return result


@router.get("/health")
async def health():
    """
    Health check endpoint.

    Public endpoint (no auth).
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{SAE_API_URL}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"SAE service unhealthy: {str(e)}",
            )


# External SAE endpoints (proxy to Modal service)
@router.post("/external/load")
async def load_external_sae(
    request: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Load an external SAE from HuggingFace or Gemma Scope."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/external/load",
                json=request,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )


@router.post("/external/list-sources")
async def list_external_sae_sources(request: dict):
    """List available hookpoints in an external SAE repository."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/external/list-sources",
                json=request,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )


@router.get("/external/list-loaded")
async def list_loaded_external_saes():
    """List all loaded external SAEs."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{SAE_API_URL}/sae/external/list-loaded")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )


@router.delete("/external/delete")
async def delete_external_sae(
    sae_id: str = Query(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete a loaded external SAE."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.delete(
                f"{SAE_API_URL}/sae/external/delete",
                params={"sae_id": sae_id},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )


@router.get("/external/feature")
async def get_external_feature_info(
    sae_id: str = Query(...),
    feature_id: int = Query(..., ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get feature info for an external SAE."""
    # Charge minimal fee
    min_cost = 0.001
    if not check_sufficient_balance(db, user.id, min_cost):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{SAE_API_URL}/sae/external/feature",
                params={"sae_id": sae_id, "feature_id": feature_id},
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge minimal fee
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=1,
        model_name="external",
        operation_type="feature_info",
        description=f"External SAE feature lookup: {sae_id} feature {feature_id}",
    )

    return result


@router.post("/upload")
async def upload_sae(
    request: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a user SAE."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/upload",
                json=request,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SAE service error: {str(e)}",
            )
