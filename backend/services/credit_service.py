"""
Credit service for managing user credit balances and transactions.
"""
import math
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Tuple

from sqlalchemy.orm import Session

from core.open_source import get_gpu_tier
from models.database import CreditBalance, Transaction, TransactionType, User
from services.modal_inference import SCALEDOWN_WINDOW_SECONDS as INFERENCE_SCALEDOWN_SECONDS


class InsufficientBalanceError(Exception):
    """Raised when a user's balance is insufficient for a charge."""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient balance: need ${required:.4f}, have ${available:.4f}")


def _to_decimal(value: float) -> Decimal:
    """Convert a float cost to Decimal for safe arithmetic with Numeric DB columns."""
    return Decimal(str(value))


# Modal list prices per second (https://modal.com/pricing, checked 2026-09-29).
# Nothing on Ozera may charge below Modal's cost; GPU containers are also billed for CPU and memory.
MODAL_GPU_PRICE_PER_SEC = {
    "t4": 0.000164,
    "l4": 0.000222,
    "a10g": 0.000306,
    "a100": 0.000583,  # A100 40GB
}
MODAL_CPU_CORE_PRICE_PER_SEC = 0.0000131
MODAL_MEMORY_GIB_PRICE_PER_SEC = 0.00000222

# Markup over Modal's cost
PRICE_MARKUP = 1.3

# CPU and memory a training container is billed for (the single-process training loop on a
# 50MB dataset stays under this; the markup leaves further headroom)
TRAINING_CPU_CORES = 2
TRAINING_MEMORY_GIB = 8


def container_cost_per_second(gpu_type: str, cpu_cores: float, memory_gib: float) -> float:
    """Modal's cost per second for a container with the given GPU, CPU, and memory."""
    return (
        MODAL_GPU_PRICE_PER_SEC[gpu_type]
        + cpu_cores * MODAL_CPU_CORE_PRICE_PER_SEC
        + memory_gib * MODAL_MEMORY_GIB_PRICE_PER_SEC
    )


def training_hourly_cost(gpu_type: str) -> float:
    """Modal's cost per hour for a training container on the given GPU."""
    return container_cost_per_second(gpu_type, TRAINING_CPU_CORES, TRAINING_MEMORY_GIB) * 3600


def lone_request_charge(
    gpu_type: str,
    cold_start_seconds: float,
    busy_seconds: float,
    idle_seconds: float,
    cpu_cores: float,
    memory_gib: float,
) -> float:
    """
    Charge that covers a request with a GPU container to itself.

    Modal bills the container's cold start, the request's work, and the warm window it
    idles for afterwards. Returns that cost plus markup, rounded up to the cent.
    """
    seconds = cold_start_seconds + busy_seconds + idle_seconds
    cost = seconds * container_cost_per_second(gpu_type, cpu_cores, memory_gib)
    return math.ceil(cost * PRICE_MARKUP * 100) / 100


# GPU pricing tiers for training (at least PRICE_MARKUP over Modal's cost, enforced below)
GPU_PRICING = {
    "t4": {
        "display_name": "NVIDIA T4",
        "rate_per_hour": 0.98,
        "description": "Budget-friendly option for smaller models",
    },
    "a10g": {
        "display_name": "NVIDIA A10G",
        "rate_per_hour": 1.64,
        "description": "Best price/performance balance (recommended)",
    },
    "a100": {
        "display_name": "NVIDIA A100",
        "rate_per_hour": 3.25,
        "description": "Fastest training for large models",
    },
}

for _gpu_type, _info in GPU_PRICING.items():
    if _info["rate_per_hour"] < training_hourly_cost(_gpu_type) * PRICE_MARKUP:
        raise RuntimeError(f"{_gpu_type} training rate ${_info['rate_per_hour']}/h is below Modal's cost plus markup")

# Minimum and maximum credit purchase amounts
MIN_CREDIT_PURCHASE = 5.0
MAX_CREDIT_PURCHASE = 500.0


def get_credit_balance(db: Session, user_id: int) -> Optional[CreditBalance]:
    """Get user's credit balance."""
    return db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first()


def get_credit_balance_for_update(db: Session, user_id: int) -> Optional[CreditBalance]:
    """Get user's credit balance with a row-level lock for safe read-modify-write."""
    return (
        db.query(CreditBalance)
        .filter(CreditBalance.user_id == user_id)
        .with_for_update()
        .first()
    )


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
    # Get or create credit balance with row lock for safe concurrent access
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        credit_balance = CreditBalance(user_id=user_id, balance_usd=0.0, reserved_usd=0.0)
        db.add(credit_balance)
        db.flush()

    # Update balance
    credit_balance.balance_usd += _to_decimal(amount_usd)
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=_to_decimal(amount_usd),
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
    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        return False

    if credit_balance.available_balance < amount_usd:
        return False

    credit_balance.reserved_usd += _to_decimal(amount_usd)
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
    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)

    # Release reservation
    credit_balance.reserved_usd -= _to_decimal(reserved_amount)

    # Deduct actual cost from balance
    credit_balance.balance_usd -= _to_decimal(amount_usd)
    credit_balance.updated_at = datetime.utcnow()

    # Create charge transaction
    transaction = Transaction(
        user_id=user_id,
        amount_usd=_to_decimal(-amount_usd),  # Negative for deduction
        transaction_type=TransactionType.TRAINING_CHARGE,
        training_job_id=job_id,
        description=description or f"Training job charge: ${amount_usd:.2f}",
    )
    db.add(transaction)

    # If there's a difference (refund), create refund transaction
    # (reserved amounts come back from the DB as Decimal, so do the arithmetic in Decimal)
    refund_amount = _to_decimal(reserved_amount) - _to_decimal(amount_usd)
    if refund_amount > Decimal("0.01"):  # Only if meaningful difference
        refund_transaction = Transaction(
            user_id=user_id,
            amount_usd=_to_decimal(refund_amount),  # Positive for refund
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
    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)

    # Release reservation
    credit_balance.reserved_usd -= _to_decimal(reserved_amount)

    # Charge for partial work if any
    if amount_usd > 0:
        credit_balance.balance_usd -= _to_decimal(amount_usd)

    credit_balance.updated_at = datetime.utcnow()

    # Create refund transaction for unused portion
    refund_amount = _to_decimal(reserved_amount) - _to_decimal(amount_usd)
    transaction = Transaction(
        user_id=user_id,
        amount_usd=_to_decimal(refund_amount),  # Positive for refund
        transaction_type=TransactionType.TRAINING_REFUND,
        training_job_id=job_id,
        description=description or f"Training job cancelled - refund: ${refund_amount:.2f}",
    )
    db.add(transaction)

    # If partial charge, also record that
    if amount_usd > 0:
        charge_transaction = Transaction(
            user_id=user_id,
            amount_usd=_to_decimal(-amount_usd),
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
    """Check if user has sufficient available balance.

    Uses FOR UPDATE row lock to prevent TOCTOU races when called
    within the same session/transaction that will later charge.
    """
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        return False
    return credit_balance.available_balance >= required_amount


# Base inference pricing (per 1000 tokens, 30% markup on Modal GPU costs)
# Set when small models ran on a T4; models now served by the L4 tier are scaled up
# by the GPU rate difference (see get_inference_multiplier)
BASE_INFERENCE_PRICING = {
    "input": 0.05,    # $0.05 per 1000 input tokens ($50 per 1M)
    "output": 0.10,   # $0.10 per 1000 output tokens ($100 per 1M)
}

# Model size multipliers for inference pricing
# Larger models use more GPU memory and compute, so they cost more
# Multipliers based on approximate compute requirements relative to smallest model
MODEL_SIZE_MULTIPLIERS = {
    # Ozera models (small, run on L4)
    "nano": 1.0,
    "mini": 1.5,
    # SmolLM family (instruct variants share the base model's compute)
    "smollm-135m": 1.0,      # 135M params, baseline (L4)
    "smollm-135m-it": 1.0,
    "smollm-360m": 2.0,      # 360M params, ~2.7x compute (L4)
    "smollm-360m-it": 2.0,
    "smollm3-3b": 12.0,      # 3B params (A10G, 2x T4 GPU cost)
    "smollm3-3b-it": 12.0,
    # Qwen family
    "qwen3-0.6b": 3.0,       # 600M params (L4)
    "qwen3-0.6b-it": 3.0,
    "qwen3-1.7b": 5.0,       # 1.7B params (L4)
    "qwen3-1.7b-it": 5.0,
    "qwen3-4b": 16.0,        # 4B params (A10G, 2x T4 GPU cost)
    "qwen3-4b-it": 16.0,
    # Gemma family
    "gemma-3-270m": 1.5,     # 270M params, 262K vocab (L4)
    "gemma-3-270m-it": 1.5,
    "gemma-3-1b": 4.0,       # 1B params, 262K vocab (L4)
    "gemma-3-1b-it": 4.0,
}

# Default multiplier for unknown models
DEFAULT_MODEL_MULTIPLIER = 1.0


# Per-request minimums for the inference workers (L4 and A10G tiers). A request that finds
# no warm container pays for the whole container lifetime Modal bills; per-token charges
# cover the work of requests that share a warm container.
INFERENCE_CPU_CORES = 2
INFERENCE_MEMORY_GIB = 8
INFERENCE_COLD_START_SECONDS = {"l4": 30, "a10g": 60}  # Boot, imports, model load (estimates)
INFERENCE_BUSY_SECONDS = {"l4": 10, "a10g": 20}  # Generation plus activation capture (estimates)


def min_gpu_request_charge(model_id: str, busy_seconds: dict[str, float] = INFERENCE_BUSY_SECONDS) -> float:
    """Minimum charge for a request to the inference worker that serves a model."""
    tier = get_gpu_tier(model_id)
    return lone_request_charge(
        gpu_type=tier,
        cold_start_seconds=INFERENCE_COLD_START_SECONDS[tier],
        busy_seconds=busy_seconds[tier],
        idle_seconds=INFERENCE_SCALEDOWN_SECONDS[tier],
        cpu_cores=INFERENCE_CPU_CORES,
        memory_gib=INFERENCE_MEMORY_GIB,
    )


def get_model_multiplier(model_id: str) -> float:
    """Get the pricing multiplier for a model based on its size."""
    return MODEL_SIZE_MULTIPLIERS.get(model_id, DEFAULT_MODEL_MULTIPLIER)


def get_inference_multiplier(model_id: str) -> float:
    """
    Get the pricing multiplier for GPU inference on a model.

    The size multiplier, scaled by the L4/T4 rate for models on the L4 tier. A10G
    models already carry their GPU cost in their size multiplier.
    """
    multiplier = get_model_multiplier(model_id)
    if get_gpu_tier(model_id) == "l4":
        multiplier *= MODAL_GPU_PRICE_PER_SEC["l4"] / MODAL_GPU_PRICE_PER_SEC["t4"]
    return multiplier


def calculate_inference_cost(prompt_tokens: int, generated_tokens: int, model_id: str = None) -> float:
    """
    Calculate the cost for an inference request.

    Args:
        prompt_tokens: Number of input tokens
        generated_tokens: Number of output tokens
        model_id: Model identifier for size-based pricing (optional)

    Returns:
        Cost in USD (at least the model's per-request minimum, see min_gpu_request_charge)
    """
    multiplier = get_inference_multiplier(model_id)
    input_cost = (prompt_tokens / 1000) * BASE_INFERENCE_PRICING["input"] * multiplier
    output_cost = (generated_tokens / 1000) * BASE_INFERENCE_PRICING["output"] * multiplier
    return max(input_cost + output_cost, min_gpu_request_charge(model_id))


def charge_inference(
    db: Session,
    user_id: int,
    prompt_tokens: int,
    generated_tokens: int,
    model_name: str,
    description: Optional[str] = None,
) -> Transaction:
    """
    Charge credits for an inference request.

    Args:
        db: Database session
        user_id: User ID
        prompt_tokens: Number of input tokens
        generated_tokens: Number of output tokens
        model_name: Name of the model used (also used for size-based pricing)
        description: Optional description

    Returns:
        Created transaction record
    """
    cost = calculate_inference_cost(prompt_tokens, generated_tokens, model_id=model_name)
    cost_decimal = _to_decimal(cost)

    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        raise InsufficientBalanceError(required=cost, available=0.0)

    # Check sufficient balance before deducting
    if credit_balance.balance_usd < cost_decimal:
        raise InsufficientBalanceError(required=cost, available=float(credit_balance.balance_usd))

    # Deduct cost from balance
    credit_balance.balance_usd -= cost_decimal
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=-cost_decimal,  # Negative for deduction
        transaction_type=TransactionType.INFERENCE_CHARGE,
        description=description or f"Inference ({model_name}): {prompt_tokens} input + {generated_tokens} output tokens",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


def charge_flat(
    db: Session,
    user_id: int,
    amount_usd: float,
    transaction_type: TransactionType,
    description: str,
) -> Transaction:
    """
    Charge a fixed amount, e.g. the per-request minimum for a GPU call not priced by tokens.

    Args:
        db: Database session
        user_id: User ID
        amount_usd: Amount to charge
        transaction_type: Transaction type to record
        description: Transaction description

    Returns:
        Created transaction record
    """
    cost_decimal = _to_decimal(amount_usd)

    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        raise InsufficientBalanceError(required=amount_usd, available=0.0)

    # Check sufficient balance before deducting
    if credit_balance.balance_usd < cost_decimal:
        raise InsufficientBalanceError(required=amount_usd, available=float(credit_balance.balance_usd))

    # Deduct cost from balance
    credit_balance.balance_usd -= cost_decimal
    credit_balance.updated_at = datetime.utcnow()

    transaction = Transaction(
        user_id=user_id,
        amount_usd=-cost_decimal,  # Negative for deduction
        transaction_type=transaction_type,
        description=description,
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


# Patching experiment pricing
# Patching is more compute-intensive than regular inference because it:
# 1. Captures activations from source prompt (forward pass with hooks)
# 2. Runs baseline generation (multiple forward passes)
# 3. Runs patched generation (multiple forward passes with activation injection)
# We charge a multiplier on top of regular inference pricing
PATCHING_MULTIPLIER = 3.0  # 3x cost of regular inference

# GPU work per patching experiment: capture, baseline, and patched generation (estimates)
PATCHING_BUSY_SECONDS = {"l4": 20, "a10g": 40}


def calculate_patching_cost(
    source_tokens: int,
    target_tokens: int,
    generated_tokens: int,
    model_id: str,
    num_patches: int = 1,
) -> float:
    """
    Calculate the cost for a patching experiment.

    Args:
        source_tokens: Number of tokens in source prompt
        target_tokens: Number of tokens in target prompt
        generated_tokens: Number of tokens generated (for both baseline and patched)
        model_id: Model identifier for size-based pricing
        num_patches: Number of patches applied (more patches = more compute)

    Returns:
        Cost in USD
    """
    model_multiplier = get_inference_multiplier(model_id)

    # Input cost: source prompt + target prompt (both processed)
    total_input_tokens = source_tokens + target_tokens
    input_cost = (total_input_tokens / 1000) * BASE_INFERENCE_PRICING["input"] * model_multiplier

    # Output cost: 2x generated tokens (baseline + patched generation)
    total_output_tokens = generated_tokens * 2
    output_cost = (total_output_tokens / 1000) * BASE_INFERENCE_PRICING["output"] * model_multiplier

    # Apply patching multiplier (for activation capture and injection overhead)
    base_cost = (input_cost + output_cost) * PATCHING_MULTIPLIER

    # Small additional cost per patch (more hooks = more overhead)
    patch_overhead = num_patches * 0.001 * model_multiplier  # $0.001 per patch

    return max(base_cost + patch_overhead, min_gpu_request_charge(model_id, PATCHING_BUSY_SECONDS))


def estimate_patching_cost(
    source_prompt_length: int,
    target_prompt_length: int,
    max_tokens: int,
    model_id: str,
    num_patches: int = 1,
) -> float:
    """
    Estimate the cost for a patching experiment before running.

    Uses character counts to estimate token counts (rough approximation).

    Args:
        source_prompt_length: Character length of source prompt
        target_prompt_length: Character length of target prompt
        max_tokens: Maximum tokens to generate
        model_id: Model identifier
        num_patches: Number of patches to apply

    Returns:
        Estimated cost in USD
    """
    # Rough estimate: 4 characters per token on average
    estimated_source_tokens = max(source_prompt_length // 4, 1)
    estimated_target_tokens = max(target_prompt_length // 4, 1)

    return calculate_patching_cost(
        source_tokens=estimated_source_tokens,
        target_tokens=estimated_target_tokens,
        generated_tokens=max_tokens,
        model_id=model_id,
        num_patches=num_patches,
    )


# Analysis pricing (attention pattern analysis)
# Analysis operations process cached activations and are computationally lighter than generation
# Base cost per analysis operation based on model complexity
BASE_ANALYSIS_COST = 0.002  # $0.002 base cost per operation

# Analysis type multipliers (some operations are more compute-intensive)
ANALYSIS_TYPE_MULTIPLIERS = {
    "classify": 0.8,     # Head classification - moderate compute
    "compare": 1.0,      # Comparing two activation sets - more compute
    "mine": 1.5,         # Pattern mining - most compute-intensive
    "importance": 0.3,   # Simple importance scoring - lightweight
}

# Minimum charge per analysis operation
MIN_ANALYSIS_CHARGE = 0.005  # $0.005 minimum


def calculate_analysis_cost(
    analysis_type: str,
    num_layers: int,
    num_heads: int,
    num_tokens: int,
    model_id: str = None,
) -> float:
    """
    Calculate the cost for an analysis operation.

    Args:
        analysis_type: Type of analysis ('classify', 'compare', 'mine', 'importance')
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer
        num_tokens: Number of tokens in the activation
        model_id: Model identifier for size-based pricing

    Returns:
        Cost in USD (minimum $0.01 per operation)
    """
    # Get analysis type multiplier
    type_multiplier = ANALYSIS_TYPE_MULTIPLIERS.get(analysis_type, 1.0)

    # Get model size multiplier (analysis runs on the backend CPU, so no GPU rate adjustment)
    model_multiplier = get_model_multiplier(model_id) if model_id else DEFAULT_MODEL_MULTIPLIER

    # Scale cost by model complexity (layers * heads) and token count
    complexity_factor = (num_layers * num_heads) / 36  # Normalize to 6x6 model
    token_factor = num_tokens / 100  # Normalize to 100 tokens

    cost = BASE_ANALYSIS_COST * type_multiplier * model_multiplier * max(complexity_factor, 0.5) * max(token_factor, 0.5)

    return max(cost, MIN_ANALYSIS_CHARGE)


def charge_analysis(
    db: Session,
    user_id: int,
    analysis_type: str,
    num_layers: int,
    num_heads: int,
    num_tokens: int,
    model_name: str,
    description: Optional[str] = None,
) -> Transaction:
    """
    Charge credits for an analysis operation.

    Args:
        db: Database session
        user_id: User ID
        analysis_type: Type of analysis ('classify', 'compare', 'mine', 'importance')
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer
        num_tokens: Number of tokens analyzed
        model_name: Name of the model
        description: Optional description

    Returns:
        Created transaction record
    """
    cost = calculate_analysis_cost(
        analysis_type=analysis_type,
        num_layers=num_layers,
        num_heads=num_heads,
        num_tokens=num_tokens,
        model_id=model_name,
    )
    cost_decimal = _to_decimal(cost)

    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        raise InsufficientBalanceError(required=cost, available=0.0)

    # Check sufficient balance before deducting
    if credit_balance.balance_usd < cost_decimal:
        raise InsufficientBalanceError(required=cost, available=float(credit_balance.balance_usd))

    # Deduct cost from balance
    credit_balance.balance_usd -= cost_decimal
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=-cost_decimal,  # Negative for deduction
        transaction_type=TransactionType.ANALYSIS_CHARGE,
        description=description or f"Analysis ({analysis_type}, {model_name}): {num_tokens} tokens, {num_layers}L x {num_heads}H",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


def estimate_analysis_cost(
    analysis_type: str,
    num_layers: int,
    num_heads: int,
    num_tokens: int,
    model_id: str = None,
) -> float:
    """
    Estimate the cost for an analysis operation before running.

    Args:
        analysis_type: Type of analysis
        num_layers: Number of layers
        num_heads: Number of heads per layer
        num_tokens: Number of tokens
        model_id: Model identifier

    Returns:
        Estimated cost in USD
    """
    return calculate_analysis_cost(
        analysis_type=analysis_type,
        num_layers=num_layers,
        num_heads=num_heads,
        num_tokens=num_tokens,
        model_id=model_id,
    )


def charge_patching(
    db: Session,
    user_id: int,
    source_tokens: int,
    target_tokens: int,
    generated_tokens: int,
    model_name: str,
    num_patches: int,
    description: Optional[str] = None,
) -> Transaction:
    """
    Charge credits for a patching experiment.

    Args:
        db: Database session
        user_id: User ID
        source_tokens: Number of tokens in source prompt
        target_tokens: Number of tokens in target prompt
        generated_tokens: Number of tokens generated
        model_name: Name of the model used
        num_patches: Number of patches applied
        description: Optional description

    Returns:
        Created transaction record
    """
    cost = calculate_patching_cost(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        generated_tokens=generated_tokens,
        model_id=model_name,
        num_patches=num_patches,
    )
    cost_decimal = _to_decimal(cost)

    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        raise InsufficientBalanceError(required=cost, available=0.0)

    # Check sufficient balance before deducting
    if credit_balance.balance_usd < cost_decimal:
        raise InsufficientBalanceError(required=cost, available=float(credit_balance.balance_usd))

    # Deduct cost from balance
    credit_balance.balance_usd -= cost_decimal
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=-cost_decimal,  # Negative for deduction
        transaction_type=TransactionType.PATCHING_CHARGE,
        description=description or f"Patching ({model_name}): {num_patches} patches, {generated_tokens} tokens generated",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


# SAE (Sparse Autoencoder) Analysis pricing
# SAE analysis processes text through transformer + SAE on GPU
# Pricing is based on tokens processed and model complexity
# Increased to match actual Modal GPU costs (previously undercharging by ~4x)
BASE_SAE_PRICING = {
    "per_token": 0.15,  # $0.15 per 1000 tokens ($150 per 1M tokens)
}

# Model multipliers for SAE (Ozera models are small and run on T4/L4 GPUs)
SAE_MODEL_MULTIPLIERS = {
    "nano": 1.0,   # 1M param model, baseline
    "mini": 1.5,   # 10M param model, 1.5x compute
}

# Minimum charge per SAE service request: a lone request pays for the L4 container's cold
# start, its work, and the warm window after it (the service reserves 16 GiB of memory)
SAE_SCALEDOWN_SECONDS = 60  # Must match scaledown_window in services/modal_sae_inference.py
MIN_SAE_CHARGE = lone_request_charge(
    gpu_type="l4",
    cold_start_seconds=45,  # Boot, imports, transformer + SAE load (estimate)
    busy_seconds=5,
    idle_seconds=SAE_SCALEDOWN_SECONDS,
    cpu_cores=2,
    memory_gib=16,
)


def calculate_sae_cost(
    num_tokens: int,
    model_id: str = "nano",
    operation_type: str = "analyze",
) -> float:
    """
    Calculate the cost for an SAE operation.

    Args:
        num_tokens: Number of tokens processed
        model_id: Model identifier ("nano" or "mini")
        operation_type: Type of operation ("analyze", "compare", "compare_layers")

    Returns:
        Cost in USD (at least MIN_SAE_CHARGE per operation)
    """
    # Get model multiplier
    model_multiplier = SAE_MODEL_MULTIPLIERS.get(model_id, 1.0)

    # Base cost per token
    token_cost = (num_tokens / 1000) * BASE_SAE_PRICING["per_token"] * model_multiplier

    # Operation type multipliers
    operation_multipliers = {
        "analyze": 1.0,           # Single SAE analysis
        "feature_info": 0.1,      # Just loading feature weights (very cheap)
        "compare": 2.0,           # Comparing two SAEs (2x work)
        "compare_layers": 3.0,    # Layer-by-layer comparison (most intensive)
    }

    operation_multiplier = operation_multipliers.get(operation_type, 1.0)
    total_cost = token_cost * operation_multiplier

    return max(total_cost, MIN_SAE_CHARGE)


def charge_sae(
    db: Session,
    user_id: int,
    num_tokens: int,
    model_name: str,
    operation_type: str = "analyze",
    description: Optional[str] = None,
) -> Transaction:
    """
    Charge credits for an SAE operation.

    Args:
        db: Database session
        user_id: User ID
        num_tokens: Number of tokens processed
        model_name: Name of the model used
        operation_type: Type of operation ("analyze", "compare", "compare_layers")
        description: Optional description

    Returns:
        Created transaction record
    """
    cost = calculate_sae_cost(
        num_tokens=num_tokens,
        model_id=model_name,
        operation_type=operation_type,
    )
    cost_decimal = _to_decimal(cost)

    # Lock the row to prevent concurrent read-modify-write races
    credit_balance = get_credit_balance_for_update(db, user_id)
    if not credit_balance:
        raise InsufficientBalanceError(required=cost, available=0.0)

    # Check sufficient balance before deducting
    if credit_balance.balance_usd < cost_decimal:
        raise InsufficientBalanceError(required=cost, available=float(credit_balance.balance_usd))

    # Deduct cost from balance
    credit_balance.balance_usd -= cost_decimal
    credit_balance.updated_at = datetime.utcnow()

    # Create transaction record
    transaction = Transaction(
        user_id=user_id,
        amount_usd=-cost_decimal,  # Negative for deduction
        transaction_type=TransactionType.SAE_CHARGE,
        description=description or f"SAE {operation_type} ({model_name}): {num_tokens} tokens",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


def estimate_sae_cost(
    text_length: int,
    model_id: str = "nano",
    operation_type: str = "analyze",
) -> float:
    """
    Estimate the cost for an SAE operation before running.

    Args:
        text_length: Character length of input text
        model_id: Model identifier
        operation_type: Type of operation

    Returns:
        Estimated cost in USD
    """
    # Rough estimate: 4 characters per token on average
    estimated_tokens = max(text_length // 4, 1)

    return calculate_sae_cost(
        num_tokens=estimated_tokens,
        model_id=model_id,
        operation_type=operation_type,
    )
