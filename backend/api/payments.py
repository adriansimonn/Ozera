"""
Payment API endpoints for Stripe integration.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session

from db import get_db
from middleware.auth_middleware import get_current_user
from models.database import User
from services.stripe_service import (
    create_payment_intent,
    verify_webhook_signature,
    extract_payment_metadata,
    is_stripe_configured,
    get_payment_intent,
)
from services.credit_service import add_credits, get_credit_balance
from api.schemas.payments import (
    CreatePaymentIntentRequest,
    CreatePaymentIntentResponse,
    WebhookResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["payments"])


@router.post("/create-intent", response_model=CreatePaymentIntentResponse)
async def create_intent(
    request: CreatePaymentIntentRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Create a Stripe PaymentIntent for purchasing credits.

    The client_secret returned should be used with Stripe.js to complete the payment.
    """
    if not is_stripe_configured():
        raise HTTPException(
            status_code=503,
            detail="Payment processing is not configured. Please contact support.",
        )

    try:
        result = create_payment_intent(
            amount_usd=request.amount_usd,
            user_id=current_user.id,
            user_email=current_user.email,
        )

        return CreatePaymentIntentResponse(
            client_secret=result["client_secret"],
            payment_intent_id=result["payment_intent_id"],
            amount_usd=result["amount_usd"],
            credits_usd=result["credits_usd"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create payment intent: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create payment. Please try again.",
        )


@router.post("/webhook", response_model=WebhookResponse)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    db: Session = Depends(get_db),
):
    """
    Stripe webhook endpoint for payment events.

    Handles payment_intent.succeeded to add credits to user accounts.
    """
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    # Get raw body for signature verification
    payload = await request.body()

    # Verify webhook signature
    event = verify_webhook_signature(payload, stripe_signature)
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # Handle the event
    event_type = event.type

    if event_type == "payment_intent.succeeded":
        payment_intent = event.data.object

        try:
            # Extract metadata
            metadata = extract_payment_metadata(payment_intent)
            user_id = metadata["user_id"]
            credits_usd = metadata["credits_usd"]
            payment_intent_id = metadata["payment_intent_id"]

            if user_id and credits_usd:
                # Idempotency check: skip if already processed
                from models.database import Transaction
                existing = db.query(Transaction).filter(
                    Transaction.stripe_payment_intent_id == payment_intent_id
                ).first()

                if existing:
                    logger.info(
                        f"Webhook already processed for payment {payment_intent_id}, skipping"
                    )
                else:
                    # Add credits to user's account
                    add_credits(
                        db=db,
                        user_id=user_id,
                        amount_usd=credits_usd,
                        stripe_payment_intent_id=payment_intent_id,
                        description=f"Credit purchase: ${credits_usd:.2f}",
                    )
                    logger.info(
                        f"Added ${credits_usd:.2f} credits to user {user_id} "
                        f"(payment: {payment_intent_id})"
                    )
            else:
                logger.warning(
                    f"Payment succeeded but missing metadata: {payment_intent_id}"
                )

        except Exception as e:
            logger.error(f"Failed to process payment webhook: {e}")
            # Don't raise - we want to acknowledge receipt to Stripe
            # The transaction can be reconciled manually if needed

    elif event_type == "payment_intent.payment_failed":
        payment_intent = event.data.object
        logger.warning(
            f"Payment failed: {payment_intent.id} - "
            f"{payment_intent.last_payment_error.message if payment_intent.last_payment_error else 'Unknown error'}"
        )

    # Always acknowledge receipt to Stripe
    return WebhookResponse(received=True)


@router.post("/confirm/{payment_intent_id}")
async def confirm_payment(
    payment_intent_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Confirm a payment and add credits to user's account.

    This is an alternative to webhooks for local development.
    The frontend calls this after Stripe confirms the payment succeeded.
    """
    # Check if credits were already added (idempotency)
    from models.database import Transaction
    existing = db.query(Transaction).filter(
        Transaction.stripe_payment_intent_id == payment_intent_id
    ).first()

    if existing:
        # Already processed
        credit_balance = get_credit_balance(db, current_user.id)
        return {
            "success": True,
            "message": "Payment already processed",
            "credits_added": 0,
            "new_balance": credit_balance.balance_usd if credit_balance else 0,
        }

    # Retrieve payment intent from Stripe
    payment_intent = get_payment_intent(payment_intent_id)

    if not payment_intent:
        raise HTTPException(status_code=404, detail="Payment not found")

    # Verify payment succeeded
    if payment_intent.status != "succeeded":
        raise HTTPException(
            status_code=400,
            detail=f"Payment not completed. Status: {payment_intent.status}"
        )

    # Verify this payment belongs to the current user
    metadata = payment_intent.metadata
    if int(metadata.get("user_id", 0)) != current_user.id:
        raise HTTPException(status_code=403, detail="Payment does not belong to this user")

    # Add credits
    credits_usd = float(metadata.get("credits_usd", 0))
    amount_received = payment_intent.amount_received / 100

    add_credits(
        db=db,
        user_id=current_user.id,
        amount_usd=credits_usd,
        stripe_payment_intent_id=payment_intent_id,
        description=f"Credit purchase: ${credits_usd:.2f}",
    )

    credit_balance = get_credit_balance(db, current_user.id)

    logger.info(f"Added ${credits_usd:.2f} credits to user {current_user.id} via confirm endpoint")

    return {
        "success": True,
        "message": "Credits added successfully",
        "credits_added": credits_usd,
        "new_balance": credit_balance.balance_usd if credit_balance else credits_usd,
    }


@router.get("/config")
async def get_payment_config():
    """
    Get Stripe publishable key for frontend initialization.

    This endpoint is public to allow frontend to initialize Stripe.
    """
    import os

    publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

    if not publishable_key:
        raise HTTPException(
            status_code=503,
            detail="Payment configuration not available",
        )

    return {"publishable_key": publishable_key}
