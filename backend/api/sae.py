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
from models.database import TransactionType, User, UserExternalSAE
from services.credit_service import (
    MIN_SAE_CHARGE,
    charge_flat,
    charge_sae,
    check_sufficient_balance,
    estimate_sae_cost,
)

# Get SAE API URL and shared secret from environment (both required in production)
SAE_API_URL = os.getenv("SAE_API_URL", "")
SAE_API_SECRET = os.getenv("SAE_API_SECRET", "")
if os.getenv("APP_ENV", "development") == "production":
    if not SAE_API_URL:
        raise RuntimeError("SAE_API_URL must be set in production")
    if not SAE_API_SECRET:
        raise RuntimeError("SAE_API_SECRET must be set in production")


def _require_sae_request_balance(db: Session, user: User) -> None:
    """Check the user can pay MIN_SAE_CHARGE, the minimum for any call that starts the SAE GPU service."""
    if not check_sufficient_balance(db, user.id, MIN_SAE_CHARGE):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )


def _charge_sae_request(db: Session, user: User, description: str) -> None:
    """Charge MIN_SAE_CHARGE for a call to the SAE GPU service not priced by tokens."""
    charge_flat(db, user.id, MIN_SAE_CHARGE, TransactionType.SAE_CHARGE, description)


def _sae_client(timeout: float, follow_redirects: bool = False) -> httpx.AsyncClient:
    """Build an httpx client pre-configured with the shared-secret auth header."""
    headers = {"X-API-Secret": SAE_API_SECRET} if SAE_API_SECRET else {}
    return httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects, headers=headers)


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
    layer_a: int = Field(default=0, ge=0)
    activation_type_a: str = Field(default="residual")
    model_b: str = Field(default="nano")
    layer_b: int = Field(default=0, ge=0)
    activation_type_b: str = Field(default="residual")
    external_id_a: Optional[str] = None
    external_id_b: Optional[str] = None
    text: str = Field(...)
    top_k: int = Field(default=100, ge=1, le=200)


class SAECompareLayersRequest(BaseModel):
    model_a: str = Field(default="nano")
    activation_type_a: str = Field(...)
    model_b: Optional[str] = None
    activation_type_b: Optional[str] = None
    external_id_a: Optional[str] = None
    external_id_b: Optional[str] = None
    text: str = Field(...)


class SAEAnalyzeBatchRequest(BaseModel):
    model: str = Field(default="nano")
    layer: int = Field(..., ge=0)
    activation_type: str = Field(...)
    texts: List[str] = Field(..., min_length=1, max_length=32)
    top_k_per_text: int = Field(default=10, ge=1, le=50)


@router.get("/list")
async def list_saes(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    List available SAEs.

    Requires auth and charges MIN_SAE_CHARGE, since it starts the SAE GPU service.
    """
    _require_sae_request_balance(db, user)
    async with _sae_client(timeout=30.0) as client:
        response = await client.get(f"{SAE_API_URL}/sae/list")
        response.raise_for_status()
        result = response.json()
    _charge_sae_request(db, user, "SAE list")
    return result


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
    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/analyze",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE service timed out. The model may be loading for the first time — please try again.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="SAE service error",
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service connection error",
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
    min_cost = MIN_SAE_CHARGE  # Feature lookups are charged at least this

    if not check_sufficient_balance(db, user.id, min_cost):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request to Modal SAE service
    async with _sae_client(timeout=30.0) as client:
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
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service error",
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
    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/analyze-batch",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE service timed out. The model may be loading for the first time — please try again.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="SAE service error",
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service connection error",
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
    Compare features between two SAEs (built-in or external).

    Requires authentication. Charges 2x cost (processing two SAEs).
    Supports external SAEs via external_id_a/external_id_b fields.
    """
    # Estimate cost (2x for comparing two SAEs)
    model_for_cost = "external" if (request.external_id_a or request.external_id_b) else request.model_a
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id=model_for_cost,
        operation_type="compare",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request (longer timeout for external SAEs that need to load HF models)
    # follow_redirects=True handles Modal's 303 redirect for long-running requests (>150s)
    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/compare",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE service timed out. The model may be loading for the first time — please try again.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="SAE service error",
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service connection error",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    # Charge credits
    num_tokens = result.get("num_tokens", 1)
    name_a = request.external_id_a or request.model_a
    name_b = request.external_id_b or request.model_b
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=num_tokens,
        model_name=model_for_cost,
        operation_type="compare",
        description=f"SAE comparison: {name_a} vs {name_b}",
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
    # Use "mini" as cost basis for external SAEs (conservative estimate)
    cost_model = request.model_a if not request.external_id_a else "mini"
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id=cost_model,
        operation_type="compare_layers",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    # Forward request
    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/compare-layers",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE service timed out. The model may be loading for the first time — please try again.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="SAE service error",
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service connection error",
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
        model_name=cost_model,
        operation_type="compare_layers",
        description="SAE layer-by-layer comparison",
    )

    return result


@router.get("/health")
async def health(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Health check endpoint.

    Requires auth and charges MIN_SAE_CHARGE, since it starts the SAE GPU service.
    """
    _require_sae_request_balance(db, user)
    async with _sae_client(timeout=10.0) as client:
        try:
            response = await client.get(f"{SAE_API_URL}/health")
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SAE service unhealthy",
            )
    _charge_sae_request(db, user, "SAE service health check")
    return result


class ExternalSAELoadRequest(BaseModel):
    source: str = Field(..., description="Source type: 'huggingface' or 'gemma_scope'")
    repo_id: str = Field(..., min_length=1, max_length=200, description="HuggingFace repo ID or Gemma Scope ID")
    hookpoint: Optional[str] = Field(None, max_length=200, description="Hookpoint name")
    device: Optional[str] = Field(None, max_length=20, description="Device to load on")


class ExternalSAEListSourcesRequest(BaseModel):
    source: str = Field(..., description="Source type: 'huggingface' or 'gemma_scope'")
    repo_id: str = Field(..., min_length=1, max_length=200, description="HuggingFace repo ID or Gemma Scope ID")


class SAEUploadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="SAE name")
    encoder_weights: List[List[float]] = Field(..., description="Encoder weight matrix")
    decoder_weights: List[List[float]] = Field(..., description="Decoder weight matrix")
    encoder_bias: Optional[List[float]] = Field(None, description="Encoder bias")
    decoder_bias: Optional[List[float]] = Field(None, description="Decoder bias")


class ExternalSAEAnalyzeRequest(BaseModel):
    sae_id: str = Field(..., description="External SAE ID")
    text: str = Field(..., description="Text to analyze")
    top_k: int = Field(default=20, ge=1, le=100, description="Top K features to return")


@router.post("/external/analyze")
async def analyze_external_sae(
    request: ExternalSAEAnalyzeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Analyze text using an external SAE.

    Loads the base model from HuggingFace, captures activations,
    and runs through the external SAE.
    Requires authentication. Charges credits based on token count.
    """
    estimated_cost = estimate_sae_cost(
        text_length=len(request.text),
        model_id="external",
        operation_type="analyze",
    )

    if not check_sufficient_balance(db, user.id, estimated_cost * 1.5):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/external/analyze",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE service timed out. The model may be loading for the first time — please try again.",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail="SAE service error",
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service connection error",
            )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"],
        )

    num_tokens = result.get("num_tokens", 1)
    charge_sae(
        db=db,
        user_id=user.id,
        num_tokens=num_tokens,
        model_name="external",
        operation_type="analyze",
        description=f"External SAE analysis: {request.sae_id}",
    )

    return result


# External SAE endpoints — per-user tracking via Supabase, shared Modal volume storage
@router.post("/external/load")
async def load_external_sae(
    request: ExternalSAELoadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Load an external SAE from HuggingFace or Gemma Scope.

    Downloads the SAE to shared Modal volume storage (if not already there),
    then creates a per-user reference in the database. Charges MIN_SAE_CHARGE.
    """
    _require_sae_request_balance(db, user)

    # Forward to Modal to download/store the SAE weights
    async with _sae_client(timeout=600.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/external/load",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="SAE loading timed out — the SAE may be large. Please try again.",
            )
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service error",
            )

    if "error" in result:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

    sae_id = result.get("sae_id", "")

    # Check if user already has this SAE referenced
    existing = db.query(UserExternalSAE).filter_by(user_id=user.id, sae_id=sae_id).first()
    if not existing:
        user_sae = UserExternalSAE(
            user_id=user.id,
            sae_id=sae_id,
            source=result.get("source", request.source),
            source_id=result.get("source_id", request.repo_id),
            display_name=result.get("display_name", sae_id),
            base_model=result.get("base_model"),
            hookpoint=result.get("hookpoint"),
            activation_type=result.get("activation_type"),
            d_input=result.get("d_input"),
            d_hidden=result.get("d_hidden"),
        )
        db.add(user_sae)
        db.commit()

    _charge_sae_request(db, user, f"External SAE load ({sae_id})")
    return result


@router.post("/external/list-sources")
async def list_external_sae_sources(
    request: ExternalSAEListSourcesRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List available hookpoints in an external SAE repository. Charges MIN_SAE_CHARGE."""
    _require_sae_request_balance(db, user)
    async with _sae_client(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/external/list-sources",
                json=request.model_dump(),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service error",
            )
    _charge_sae_request(db, user, f"External SAE source listing ({request.repo_id})")
    return result


@router.get("/external/list-loaded")
async def list_loaded_external_saes(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List external SAEs for the current user."""
    user_saes = (
        db.query(UserExternalSAE)
        .filter_by(user_id=user.id)
        .order_by(UserExternalSAE.created_at.desc())
        .all()
    )

    external_saes = [
        {
            "id": s.sae_id,
            "path": f"/saes/external/{s.sae_id}",
            "source": s.source,
            "source_id": s.source_id,
            "display_name": s.display_name,
            "base_model": s.base_model,
            "hookpoint": s.hookpoint,
            "activation_type": s.activation_type,
            "d_input": s.d_input,
            "d_hidden": s.d_hidden,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in user_saes
    ]

    return {"external_saes": external_saes, "count": len(external_saes)}


@router.delete("/external/delete")
async def delete_external_sae(
    sae_id: str = Query(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Remove an external SAE from the current user's list.

    Only removes the user's reference — shared weights on the Modal volume
    are left in place for other users and cleaned up periodically.
    """
    user_sae = db.query(UserExternalSAE).filter_by(user_id=user.id, sae_id=sae_id).first()
    if not user_sae:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SAE not found in your list")

    db.delete(user_sae)
    db.commit()

    return {"status": "deleted", "sae_id": sae_id}


@router.get("/external/feature")
async def get_external_feature_info(
    sae_id: str = Query(...),
    feature_id: int = Query(..., ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get feature info for an external SAE."""
    # Charge minimal fee
    min_cost = MIN_SAE_CHARGE  # Feature lookups are charged at least this
    if not check_sufficient_balance(db, user.id, min_cost):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="INSUFFICIENT_CREDITS",
        )

    async with _sae_client(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{SAE_API_URL}/sae/external/feature",
                params={"sae_id": sae_id, "feature_id": feature_id},
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service error",
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
    request: SAEUploadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload a user SAE (safetensors).

    Limited to 1 uploaded SAE per user. If the user already has an upload,
    the old one is replaced (reference removed; weights cleaned up periodically).
    Charges MIN_SAE_CHARGE.
    """
    _require_sae_request_balance(db, user)

    # Check for existing upload and remove reference if present
    existing_upload = (
        db.query(UserExternalSAE)
        .filter_by(user_id=user.id, source="user_upload")
        .first()
    )
    if existing_upload:
        db.delete(existing_upload)
        db.flush()

    # Forward to Modal to store the weights
    async with _sae_client(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{SAE_API_URL}/sae/upload",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SAE service error",
            )

    if "error" in result:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

    # Create the user reference
    sae_id = result.get("sae_id", "")
    user_sae = UserExternalSAE(
        user_id=user.id,
        sae_id=sae_id,
        source="user_upload",
        source_id=None,
        display_name=result.get("display_name", request.name),
        base_model=result.get("base_model"),
        hookpoint=result.get("hookpoint"),
        activation_type=result.get("activation_type"),
        d_input=result.get("d_input"),
        d_hidden=result.get("d_hidden"),
    )
    db.add(user_sae)
    db.commit()

    _charge_sae_request(db, user, f"SAE upload ({sae_id})")
    return result


@router.get("/external/has-upload")
async def has_uploaded_sae(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Check if the current user already has an uploaded SAE."""
    existing = (
        db.query(UserExternalSAE)
        .filter_by(user_id=user.id, source="user_upload")
        .first()
    )
    return {
        "has_upload": existing is not None,
        "upload_name": existing.display_name if existing else None,
        "upload_sae_id": existing.sae_id if existing else None,
    }
