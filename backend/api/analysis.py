"""
Analysis API endpoints for attention pattern analysis.

Provides endpoints for head classification, pattern comparison, and pattern mining.
All endpoints require authentication and charge credits.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional

from api.patching import BASE_MODEL_CONFIGS
from core.open_source import OPEN_SOURCE_MODELS
from core.patching import CapturedActivations, get_patching_engine
from core.analysis import HeadClassifier, PatternMiner, HeadType
from middleware.auth_middleware import get_current_user
from models.database import User
from db import get_db
from services.credit_service import (
    InsufficientBalanceError,
    estimate_analysis_cost,
    charge_analysis,
    check_sufficient_balance,
)


router = APIRouter(prefix="/analysis", tags=["Analysis"])


# ============= Pricing =============

def _num_heads(captured: CapturedActivations) -> int:
    """
    Attention heads per layer of the model the activations came from.

    Read from the captured attention weights ([batch, heads, seq, seq]), else from the
    model's config.
    """
    for layer in range(captured.num_layers):
        weights = captured.activations.get(f"layer_{layer}_attn_weights")
        if weights is not None and weights.dim() == 4:
            return weights.shape[1]

    if captured.model_id in OPEN_SOURCE_MODELS:
        return OPEN_SOURCE_MODELS[captured.model_id].num_heads
    return BASE_MODEL_CONFIGS.get(captured.model_id, BASE_MODEL_CONFIGS["nano"])["num_heads"]


def _check_analysis_balance(
    db: Session, user_id: int, analysis_type: str, captured: CapturedActivations, num_tokens: int
) -> None:
    """Raise 402 unless the user can pay for the analysis."""
    estimated_cost = estimate_analysis_cost(
        analysis_type=analysis_type,
        num_layers=captured.num_layers,
        num_heads=_num_heads(captured),
        num_tokens=num_tokens,
        model_id=captured.model_id,
    )
    if not check_sufficient_balance(db, user_id, estimated_cost):
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")


def _charge_analysis(
    db: Session, user_id: int, analysis_type: str, captured: CapturedActivations, num_tokens: int
) -> None:
    """Charge for a finished analysis. A failed charge fails the request, so no analysis is free."""
    try:
        charge_analysis(
            db=db,
            user_id=user_id,
            analysis_type=analysis_type,
            num_layers=captured.num_layers,
            num_heads=_num_heads(captured),
            num_tokens=num_tokens,
            model_name=captured.model_id,
        )
    except InsufficientBalanceError:
        raise HTTPException(status_code=402, detail="INSUFFICIENT_CREDITS")


def _get_captured(activation_id: str, user: User) -> CapturedActivations:
    """The user's captured activations, or 404 (also for other users' IDs)."""
    captured = get_patching_engine().get_captured_activations(activation_id, user.id)
    if captured is None:
        raise HTTPException(
            status_code=404,
            detail=f"Captured activations not found: {activation_id}"
        )
    return captured


# ============= Request/Response Models =============

class ClassifyHeadsRequest(BaseModel):
    """Request to classify attention heads for a captured activation."""
    activation_id: str = Field(..., description="ID of captured activations to analyze")


class HeadClassificationResponse(BaseModel):
    """Classification result for a single head."""
    layer: int
    head: int
    primary_type: str
    confidence: float
    scores: dict[str, float]
    pattern_summary: str


class ClassifyHeadsResponse(BaseModel):
    """Response containing classification results for all heads."""
    model_id: str
    activation_id: str
    prompt: str
    tokens: list[str]
    num_layers: int
    num_heads: int
    classifications: list[HeadClassificationResponse]
    summary: dict[str, int]  # Count of each head type


class CompareAttentionRequest(BaseModel):
    """Request to compare attention patterns across two prompts."""
    activation_id_1: str = Field(..., description="ID of first captured activations")
    activation_id_2: str = Field(..., description="ID of second captured activations")


class LayerSimilarity(BaseModel):
    """Similarity metrics for a single layer."""
    layer: int
    mean_similarity: float
    min_similarity: float
    max_similarity: float


class HeadDifference(BaseModel):
    """Information about a significantly different head."""
    layer: int
    head: int
    similarity: float
    difference: float


class CompareAttentionResponse(BaseModel):
    """Response containing attention comparison results."""
    prompt1: str
    prompt2: str
    activation_id1: str
    activation_id2: str
    layer_similarities: list[LayerSimilarity]
    head_differences: list[HeadDifference]
    common_patterns: list[str]
    divergent_patterns: list[str]


class MinePatternRequest(BaseModel):
    """Request to run full pattern mining on captured activations."""
    activation_id: str = Field(..., description="ID of captured activations to analyze")


class HeadImportanceResponse(BaseModel):
    """Importance metrics for a single head."""
    layer: int
    head: int
    entropy_score: float
    max_attention_score: float
    variance_score: float
    overall_importance: float


class CrossLayerPatternResponse(BaseModel):
    """Cross-layer pattern information."""
    pattern_type: str
    description: str
    involved_heads: list[list[int]]  # List of [layer, head] pairs
    confidence: float
    evidence: dict


class CircuitCandidateResponse(BaseModel):
    """Potential circuit candidate."""
    type: str
    description: str
    components: dict
    confidence: float


class MinePatternResponse(BaseModel):
    """Response containing pattern mining results."""
    model_id: str
    activation_ids: list[str]
    head_importance: list[HeadImportanceResponse]
    cross_layer_patterns: list[CrossLayerPatternResponse]
    classification_summary: dict[str, int]
    top_heads: list[list]  # List of [layer, head, importance] tuples
    circuit_candidates: list[CircuitCandidateResponse]


class HeadImportanceRequest(BaseModel):
    """Request to compute head importance scores."""
    activation_id: str = Field(..., description="ID of captured activations to analyze")


class HeadImportanceListResponse(BaseModel):
    """Response containing head importance scores."""
    activation_id: str
    model_id: str
    importance: dict[str, float]  # Maps "layer_head" to importance score


# ============= API Endpoints =============

@router.post("/attention/classify-heads", response_model=ClassifyHeadsResponse)
def classify_heads(
    request: ClassifyHeadsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Classify all attention heads for a captured activation.

    Identifies head types including:
    - Induction heads (A...B...A -> B pattern)
    - Previous token heads (sub-diagonal attention)
    - Positional heads (fixed offset patterns)
    - Copying heads (attention to repeated tokens)

    Requires authentication and charges credits.

    Args:
        request: ClassifyHeadsRequest with activation_id

    Returns:
        ClassifyHeadsResponse with classification results for all heads
    """
    captured = _get_captured(request.activation_id, current_user)
    num_tokens = len(captured.tokens)

    # Check balance before running
    _check_analysis_balance(db, current_user.id, "classify", captured, num_tokens)

    # Run head classification
    classifier = HeadClassifier()
    result = classifier.classify_all_heads(
        activations=captured.activations,
        tokens=captured.tokens,
        decoded_tokens=captured.decoded_tokens,
        num_layers=captured.num_layers,
        model_id=captured.model_id,
        activation_id=captured.id,
        prompt=captured.prompt,
    )

    # Charge credits after successful classification
    _charge_analysis(db, current_user.id, "classify", captured, num_tokens)

    # Convert to response format
    classifications = [
        HeadClassificationResponse(
            layer=c.layer,
            head=c.head,
            primary_type=c.primary_type.value,
            confidence=c.confidence,
            scores=c.scores,
            pattern_summary=c.pattern_summary,
        )
        for c in result.classifications
    ]

    # Generate summary
    summary = {t.value: 0 for t in HeadType}
    for c in result.classifications:
        summary[c.primary_type.value] += 1

    return ClassifyHeadsResponse(
        model_id=result.model_id,
        activation_id=result.activation_id,
        prompt=result.prompt,
        tokens=result.tokens,
        num_layers=result.num_layers,
        num_heads=result.num_heads,
        classifications=classifications,
        summary=summary,
    )


@router.post("/attention/compare", response_model=CompareAttentionResponse)
def compare_attention(
    request: CompareAttentionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compare attention patterns between two captured activations.

    Useful for understanding how attention differs between prompts,
    identifying which heads are most sensitive to input changes.

    Requires authentication and charges credits.

    Args:
        request: CompareAttentionRequest with two activation IDs

    Returns:
        CompareAttentionResponse with similarity and difference analysis
    """
    captured1 = _get_captured(request.activation_id_1, current_user)
    captured2 = _get_captured(request.activation_id_2, current_user)

    if captured1.model_id != captured2.model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot compare activations from different models: {captured1.model_id} vs {captured2.model_id}"
        )

    # Priced on the combined token count
    num_tokens = len(captured1.tokens) + len(captured2.tokens)

    # Check balance before running
    _check_analysis_balance(db, current_user.id, "compare", captured1, num_tokens)

    # Run comparison
    miner = PatternMiner()
    comparison = miner.compare_attention_patterns(
        activations1=captured1.activations,
        activations2=captured2.activations,
        tokens1=captured1.decoded_tokens,
        tokens2=captured2.decoded_tokens,
        activation_id1=captured1.id,
        activation_id2=captured2.id,
        prompt1=captured1.prompt,
        prompt2=captured2.prompt,
        num_layers=captured1.num_layers,
    )

    # Charge credits after successful comparison
    _charge_analysis(db, current_user.id, "compare", captured1, num_tokens)

    return CompareAttentionResponse(
        prompt1=comparison.prompt1,
        prompt2=comparison.prompt2,
        activation_id1=comparison.activation_id1,
        activation_id2=comparison.activation_id2,
        layer_similarities=[
            LayerSimilarity(**ls) for ls in comparison.layer_similarities
        ],
        head_differences=[
            HeadDifference(**hd) for hd in comparison.head_differences
        ],
        common_patterns=comparison.common_patterns,
        divergent_patterns=comparison.divergent_patterns,
    )


@router.post("/attention/mine-patterns", response_model=MinePatternResponse)
def mine_patterns(
    request: MinePatternRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run comprehensive pattern mining on captured activations.

    Includes:
    - Head classification
    - Head importance scoring
    - Cross-layer pattern detection
    - Circuit candidate identification

    Requires authentication and charges credits.

    Args:
        request: MinePatternRequest with activation_id

    Returns:
        MinePatternResponse with complete pattern mining results
    """
    captured = _get_captured(request.activation_id, current_user)
    num_tokens = len(captured.tokens)

    # Check balance before running
    _check_analysis_balance(db, current_user.id, "mine", captured, num_tokens)

    # Run pattern mining
    miner = PatternMiner()
    result = miner.mine_patterns(
        activations=captured.activations,
        tokens=captured.tokens,
        decoded_tokens=captured.decoded_tokens,
        num_layers=captured.num_layers,
        model_id=captured.model_id,
        activation_id=captured.id,
        prompt=captured.prompt,
    )

    # Charge credits after successful mining
    _charge_analysis(db, current_user.id, "mine", captured, num_tokens)

    return MinePatternResponse(
        model_id=result.model_id,
        activation_ids=result.activation_ids,
        head_importance=[
            HeadImportanceResponse(
                layer=h.layer,
                head=h.head,
                entropy_score=h.entropy_score,
                max_attention_score=h.max_attention_score,
                variance_score=h.variance_score,
                overall_importance=h.overall_importance,
            )
            for h in result.head_importance
        ],
        cross_layer_patterns=[
            CrossLayerPatternResponse(
                pattern_type=p.pattern_type,
                description=p.description,
                involved_heads=[list(h) for h in p.involved_heads],
                confidence=p.confidence,
                evidence=p.evidence,
            )
            for p in result.cross_layer_patterns
        ],
        classification_summary=result.classification_summary,
        top_heads=[list(t) for t in result.top_heads],
        circuit_candidates=[
            CircuitCandidateResponse(**c) for c in result.circuit_candidates
        ],
    )


@router.post("/attention/head-importance", response_model=HeadImportanceListResponse)
def get_head_importance(
    request: HeadImportanceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Compute importance scores for each attention head.

    Importance is based on:
    - Attention entropy (lower = more focused)
    - Maximum attention weight (higher = more concentrated)
    - Variance (higher = more dynamic patterns)

    Requires authentication and charges credits.

    Args:
        request: HeadImportanceRequest with activation_id

    Returns:
        HeadImportanceListResponse with importance scores
    """
    captured = _get_captured(request.activation_id, current_user)
    num_tokens = len(captured.tokens)

    # Check balance before running
    _check_analysis_balance(db, current_user.id, "importance", captured, num_tokens)

    # Compute importance scores
    classifier = HeadClassifier()
    importance = classifier.get_head_importance(
        activations=captured.activations,
        num_layers=captured.num_layers,
    )

    # Charge credits after successful computation
    _charge_analysis(db, current_user.id, "importance", captured, num_tokens)

    return HeadImportanceListResponse(
        activation_id=captured.id,
        model_id=captured.model_id,
        importance=importance,
    )
