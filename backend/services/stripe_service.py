"""
Stripe payment service for handling credit purchases.
"""
import os
from typing import Optional, Dict, Any

import stripe
from dotenv import load_dotenv

from services.credit_service import validate_purchase_amount, MIN_CREDIT_PURCHASE, MAX_CREDIT_PURCHASE

# Load environment variables
load_dotenv()

# Initialize Stripe with API key
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

# Webhook secret for verifying webhook signatures
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")


def is_stripe_configured() -> bool:
    """Check if Stripe is properly configured."""
    return bool(stripe.api_key and stripe.api_key.startswith("sk_"))


def create_payment_intent(
    amount_usd: float,
    user_id: int,
    user_email: str,
) -> Dict[str, Any]:
    """
    Create a Stripe PaymentIntent for a credit purchase.

    Args:
        amount_usd: Payment amount in USD
        user_id: User ID for metadata
        user_email: User email for receipt

    Returns:
        Dict with client_secret, payment_intent_id, and credits_usd
    """
    if not is_stripe_configured():
        raise ValueError("Stripe is not configured. Please set STRIPE_SECRET_KEY in environment.")

    # Validate amount is within allowed range
    if not validate_purchase_amount(amount_usd):
        raise ValueError(f"Amount must be between ${MIN_CREDIT_PURCHASE} and ${MAX_CREDIT_PURCHASE}")

    # Credits equal payment amount (1:1)
    credits_usd = amount_usd

    # Create PaymentIntent
    # Amount must be in cents for Stripe
    amount_cents = int(amount_usd * 100)

    intent = stripe.PaymentIntent.create(
        amount=amount_cents,
        currency="usd",
        metadata={
            "user_id": str(user_id),
            "credits_usd": str(credits_usd),
            "type": "credit_purchase",
        },
        receipt_email=user_email,
        description=f"Ozera Credits: ${amount_usd:.2f}",
        automatic_payment_methods={
            "enabled": True,
        },
    )

    return {
        "client_secret": intent.client_secret,
        "payment_intent_id": intent.id,
        "amount_usd": amount_usd,
        "credits_usd": credits_usd,
    }


def verify_webhook_signature(payload: bytes, signature: str) -> Optional[stripe.Event]:
    """
    Verify Stripe webhook signature and construct event.

    Args:
        payload: Raw request body
        signature: Stripe-Signature header

    Returns:
        Verified Stripe event or None if verification fails
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise ValueError("Stripe webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )
        return event
    except stripe.error.SignatureVerificationError:
        return None
    except ValueError:
        return None


def get_payment_intent(payment_intent_id: str) -> Optional[stripe.PaymentIntent]:
    """
    Retrieve a PaymentIntent by ID.

    Args:
        payment_intent_id: Stripe PaymentIntent ID

    Returns:
        PaymentIntent object or None if not found
    """
    if not is_stripe_configured():
        return None

    try:
        return stripe.PaymentIntent.retrieve(payment_intent_id)
    except stripe.error.StripeError:
        return None


def extract_payment_metadata(payment_intent: stripe.PaymentIntent) -> Dict[str, Any]:
    """
    Extract metadata from a completed PaymentIntent.

    Args:
        payment_intent: Stripe PaymentIntent object

    Returns:
        Dict with user_id and credits_usd
    """
    metadata = payment_intent.metadata
    return {
        "user_id": int(metadata.get("user_id", 0)),
        "credits_usd": float(metadata.get("credits_usd", 0)),
        "payment_intent_id": payment_intent.id,
        "amount_received": payment_intent.amount_received / 100,  # Convert from cents
    }
