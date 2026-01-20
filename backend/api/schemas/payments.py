"""
Pydantic schemas for payments API.
"""
from typing import Optional
from pydantic import BaseModel, Field


class CreatePaymentIntentRequest(BaseModel):
    """Request schema for creating a payment intent."""
    amount_usd: float = Field(
        ...,
        gt=0,
        description="Payment amount in USD (you receive exactly this amount in credits)",
    )


class CreatePaymentIntentResponse(BaseModel):
    """Response schema for payment intent creation."""
    client_secret: str = Field(..., description="Stripe client secret for frontend")
    payment_intent_id: str = Field(..., description="Stripe payment intent ID")
    amount_usd: float = Field(..., description="Payment amount in USD")
    credits_usd: float = Field(..., description="Credits to be received (equals payment amount)")


class PaymentConfirmationResponse(BaseModel):
    """Response schema for payment confirmation."""
    success: bool
    message: str
    new_balance: Optional[float] = Field(None, description="Updated credit balance")
    credits_added: Optional[float] = Field(None, description="Credits added to account")


class WebhookResponse(BaseModel):
    """Response schema for Stripe webhook."""
    received: bool = True
