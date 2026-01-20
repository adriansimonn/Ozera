"""
Credit service for managing user credit balances and transactions.
"""
from datetime import datetime
from typing import Optional, List, Tuple

from sqlalchemy.orm import Session

from models.database import CreditBalance, Transaction, TransactionType, User


# GPU pricing tiers (30% markup on Modal base costs)
GPU_PRICING = {
    "t4": {
        "display_name": "NVIDIA T4",
        "rate_per_hour": 0.40,
        "description": "Budget-friendly option for smaller models",
    },
    "a10g": {
        "display_name": "NVIDIA A10G",
        "rate_per_hour": 0.80,
        "description": "Best price/performance balance (recommended)",
    },
    "a100": {
        "display_name": "NVIDIA A100",
        "rate_per_hour": 3.25,
        "description": "Fastest training for large models",
    },
}

# Minimum and maximum credit purchase amounts
MIN_CREDIT_PURCHASE = 5.0
MAX_CREDIT_PURCHASE = 500.0


def get_credit_balance(db: Session, user_id: int) -> Optional[CreditBalance]:
    """Get user's credit balance."""
    return db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first()


def get_user_transactions(
    db: Session, user_id: int, limit: int = 50, offset: int = 0
) -> Tuple[List[Transaction], int]:
    """Get user's transaction history with pagination."""
    query = (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(Transaction.created_at.desc())
    )
    total = query.count()
    transactions = query.offset(offset).limit(limit).all()
    return transactions, total


def add_credits(
    db: Session,
    user_id: int,
    amount_usd: float,
    stripe_payment_intent_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Transaction:
    """
    Add credits to user's balance (e.g., from a Stripe payment).

    Args:
        db: Database session
        user_id: User ID
        amount_usd: Amount to add (positive)
        stripe_payment_intent_id: Stripe payment intent ID for reconciliation
        description: Optional description

    Returns:
        Created transaction record
    """
    # Get or create credit balance
    credit_balance = get_credit_balance(db, user_id)
    if not credit_balance:
        credit_balance = CreditBalance(user_id=user_id, balance_usd=0.0, reserved_usd=0.0)
        db.add(credit_balance)

    # Update balance
    credit_balance.balance_usd += amount_usd
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=amount_usd,
        transaction_type=TransactionType.CREDIT_PURCHASE,
        stripe_payment_intent_id=stripe_payment_intent_id,
        description=description or f"Credit purchase: ${amount_usd:.2f}",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


def reserve_credits(
    db: Session, user_id: int, amount_usd: float, job_id: str
) -> bool:
    """
    Reserve credits for a training job.

    Args:
        db: Database session
        user_id: User ID
        amount_usd: Amount to reserve
        job_id: Training job ID

    Returns:
        True if reservation successful, False if insufficient balance
    """
    credit_balance = get_credit_balance(db, user_id)
    if not credit_balance:
        return False

    if credit_balance.available_balance < amount_usd:
        return False

    credit_balance.reserved_usd += amount_usd
    credit_balance.updated_at = datetime.utcnow()
    db.commit()

    return True


def charge_credits(
    db: Session,
    user_id: int,
    amount_usd: float,
    reserved_amount: float,
    job_id: str,
    description: Optional[str] = None,
) -> Transaction:
    """
    Charge credits for a completed training job and release reservation.

    Args:
        db: Database session
        user_id: User ID
        amount_usd: Actual amount to charge
        reserved_amount: Amount that was reserved
        job_id: Training job ID
        description: Optional description

    Returns:
        Created transaction record
    """
    credit_balance = get_credit_balance(db, user_id)

    # Release reservation
    credit_balance.reserved_usd -= reserved_amount

    # Deduct actual cost from balance
    credit_balance.balance_usd -= amount_usd
    credit_balance.updated_at = datetime.utcnow()

    # Create charge transaction
    transaction = Transaction(
        user_id=user_id,
        amount_usd=-amount_usd,  # Negative for deduction
        transaction_type=TransactionType.TRAINING_CHARGE,
        training_job_id=job_id,
        description=description or f"Training job charge: ${amount_usd:.2f}",
    )
    db.add(transaction)

    # If there's a difference (refund), create refund transaction
    refund_amount = reserved_amount - amount_usd
    if refund_amount > 0.01:  # Only if meaningful difference
        refund_transaction = Transaction(
            user_id=user_id,
            amount_usd=refund_amount,  # Positive for refund
            transaction_type=TransactionType.TRAINING_REFUND,
            training_job_id=job_id,
            description=f"Training job refund (unused reservation): ${refund_amount:.2f}",
        )
        db.add(refund_transaction)

    db.commit()
    db.refresh(transaction)

    return transaction


def refund_credits(
    db: Session,
    user_id: int,
    amount_usd: float,
    reserved_amount: float,
    job_id: str,
    description: Optional[str] = None,
) -> Transaction:
    """
    Refund credits for a cancelled/failed training job.

    Args:
        db: Database session
        user_id: User ID
        amount_usd: Partial charge amount (if any work was done)
        reserved_amount: Amount that was reserved
        job_id: Training job ID
        description: Optional description

    Returns:
        Created transaction record
    """
    credit_balance = get_credit_balance(db, user_id)

    # Release reservation
    credit_balance.reserved_usd -= reserved_amount

    # Charge for partial work if any
    if amount_usd > 0:
        credit_balance.balance_usd -= amount_usd

    credit_balance.updated_at = datetime.utcnow()

    # Create refund transaction for unused portion
    refund_amount = reserved_amount - amount_usd
    transaction = Transaction(
        user_id=user_id,
        amount_usd=refund_amount,  # Positive for refund
        transaction_type=TransactionType.TRAINING_REFUND,
        training_job_id=job_id,
        description=description or f"Training job cancelled - refund: ${refund_amount:.2f}",
    )
    db.add(transaction)

    # If partial charge, also record that
    if amount_usd > 0:
        charge_transaction = Transaction(
            user_id=user_id,
            amount_usd=-amount_usd,
            transaction_type=TransactionType.TRAINING_CHARGE,
            training_job_id=job_id,
            description=f"Training job partial charge: ${amount_usd:.2f}",
        )
        db.add(charge_transaction)

    db.commit()
    db.refresh(transaction)

    return transaction


def validate_purchase_amount(amount_usd: float) -> bool:
    """
    Validate that a purchase amount is within allowed limits.

    Args:
        amount_usd: Amount in USD

    Returns:
        True if valid, False otherwise
    """
    return MIN_CREDIT_PURCHASE <= amount_usd <= MAX_CREDIT_PURCHASE


def check_sufficient_balance(db: Session, user_id: int, required_amount: float) -> bool:
    """Check if user has sufficient available balance."""
    credit_balance = get_credit_balance(db, user_id)
    if not credit_balance:
        return False
    return credit_balance.available_balance >= required_amount
