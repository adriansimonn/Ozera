"""
Pydantic schemas for credits API.
"""
from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field


class TransactionTypeEnum(str, Enum):
    """Transaction type enumeration for API responses."""
    CREDIT_PURCHASE = "credit_purchase"
    TRAINING_CHARGE = "training_charge"
    TRAINING_REFUND = "training_refund"
    ADMIN_ADJUSTMENT = "admin_adjustment"
    INFERENCE_CHARGE = "inference_charge"
    PATCHING_CHARGE = "patching_charge"


class CreditBalanceResponse(BaseModel):
    """Response schema for credit balance."""
    balance_usd: float = Field(..., description="Total credit balance")
    reserved_usd: float = Field(..., description="Credits reserved for running jobs")
    available_balance: float = Field(..., description="Available balance (total - reserved)")

    class Config:
        from_attributes = True


class TransactionResponse(BaseModel):
    """Response schema for a single transaction."""
    id: int
    amount_usd: float = Field(..., description="Transaction amount (positive = credit, negative = debit)")
    transaction_type: TransactionTypeEnum
    description: Optional[str] = None
    stripe_payment_intent_id: Optional[str] = None
    training_job_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionListResponse(BaseModel):
    """Response schema for transaction list."""
    transactions: List[TransactionResponse]
    total: int = Field(..., description="Total number of transactions")


class GPUPricing(BaseModel):
    """GPU pricing tier information."""
    gpu_type: str = Field(..., description="GPU type identifier")
    display_name: str = Field(..., description="Display name for the GPU")
    rate_per_hour: float = Field(..., description="Cost per hour in USD")
    description: str = Field(..., description="GPU description")


class InferencePricing(BaseModel):
    """Inference pricing information."""
    input_per_1k_tokens: float = Field(..., description="Base cost per 1000 input tokens in USD")
    output_per_1k_tokens: float = Field(..., description="Base cost per 1000 output tokens in USD")


class PricingResponse(BaseModel):
    """Response schema for GPU pricing tiers, inference pricing, and credit purchase limits."""
    gpu_pricing: List[GPUPricing]
    inference_pricing: InferencePricing = Field(..., description="Base token-based inference pricing (multiply by model_multipliers for actual cost)")
    model_multipliers: dict[str, float] = Field(..., description="Pricing multipliers per model based on size")
    min_purchase: float = Field(..., description="Minimum credit purchase amount in USD")
    max_purchase: float = Field(..., description="Maximum credit purchase amount in USD")
