"""
SQLAlchemy database models for Ozera cloud training platform.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    false,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=True)
    full_name = Column(String(255), nullable=True)
    supabase_user_id = Column(String(36), unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    settings = Column(JSON, nullable=False, server_default='{}')

    # Relationships
    credit_balance = relationship(
        "CreditBalance", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    training_jobs = relationship(
        "TrainingJob", back_populates="user", cascade="all, delete-orphan"
    )
    transactions = relationship(
        "Transaction", back_populates="user", cascade="all, delete-orphan"
    )
    datasets = relationship(
        "Dataset", back_populates="user", cascade="all, delete-orphan"
    )
    uploaded_models = relationship(
        "UploadedModel", back_populates="user", cascade="all, delete-orphan"
    )
    external_saes = relationship(
        "UserExternalSAE", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class CreditBalance(Base):
    """User credit balance model."""

    __tablename__ = "credit_balances"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance_usd = Column(Numeric(precision=10, scale=4), default=0.0, nullable=False)
    reserved_usd = Column(
        Numeric(precision=10, scale=4), default=0.0, nullable=False
    )  # Reserved for running jobs
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="credit_balance")

    @property
    def available_balance(self) -> float:
        """Calculate available balance (total - reserved)."""
        return self.balance_usd - self.reserved_usd

    def __repr__(self):
        return f"<CreditBalance(user_id={self.user_id}, balance=${self.balance_usd:.2f}, reserved=${self.reserved_usd:.2f})>"


class TransactionType(PyEnum):
    """Transaction type enumeration."""

    CREDIT_PURCHASE = "credit_purchase"
    TRAINING_CHARGE = "training_charge"
    TRAINING_REFUND = "training_refund"
    ADMIN_ADJUSTMENT = "admin_adjustment"
    INFERENCE_CHARGE = "inference_charge"
    INFERENCE_REFUND = "inference_refund"
    PATCHING_CHARGE = "patching_charge"
    ANALYSIS_CHARGE = "analysis_charge"
    SAE_CHARGE = "sae_charge"


class Transaction(Base):
    """Transaction history model."""

    __tablename__ = "transactions"
    __table_args__ = (
        # A Stripe payment can only be credited once (the webhook and confirm endpoint race)
        Index(
            "ix_transactions_stripe_payment_intent_id",
            "stripe_payment_intent_id",
            unique=True,
            postgresql_where=text("stripe_payment_intent_id IS NOT NULL"),
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    amount_usd = Column(
        Numeric(precision=10, scale=4), nullable=False
    )  # Positive = credit added, Negative = credit deducted
    transaction_type = Column(Enum(TransactionType), nullable=False)
    description = Column(String(500), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)  # For refunds/reconciliation
    training_job_id = Column(
        String(50), ForeignKey("training_jobs.job_id"), nullable=True
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="transactions")
    training_job = relationship("TrainingJob", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.id}, user_id={self.user_id}, type={self.transaction_type.value}, amount=${self.amount_usd:.2f})>"


class JobStatus(PyEnum):
    """Training job status enumeration."""

    PENDING = "pending"
    QUEUED = "queued"  # Waiting in Modal queue
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    """Training job model."""

    __tablename__ = "training_jobs"

    job_id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Configuration
    dataset_id = Column(String(100), nullable=False)
    dataset_name = Column(String(255), nullable=True)
    model_config = Column(String(50), nullable=False)  # 'nano' or 'mini'
    model_name = Column(String(255), nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    seq_len = Column(Integer, nullable=False)

    # GPU configuration
    gpu_type = Column(String(50), default="a10g", nullable=False)  # 't4', 'a10g', 'a100'

    # Status
    status = Column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True
    )
    modal_call_id = Column(String(255), nullable=True, index=True)  # Modal's job ID

    # Cost tracking
    estimated_cost_usd = Column(Numeric(precision=10, scale=4), nullable=False)
    estimated_minutes = Column(Float, nullable=False)
    reserved_credits_usd = Column(
        Numeric(precision=10, scale=4), nullable=False
    )  # Amount reserved upfront (with buffer)
    reservation_released = Column(
        Boolean, default=False, server_default=false(), nullable=False
    )  # Set once the reservation is charged or refunded, so it's never released twice
    actual_cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)  # Final cost after completion
    actual_minutes = Column(Float, nullable=True)  # Actual duration

    # Progress tracking
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    train_ppl = Column(Float, nullable=True)
    val_ppl = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="training_jobs")
    transactions = relationship("Transaction", back_populates="training_job")

    def __repr__(self):
        return f"<TrainingJob(job_id={self.job_id}, user_id={self.user_id}, status={self.status.value}, gpu={self.gpu_type})>"


class Dataset(Base):
    """User dataset model (for user-scoped datasets)."""

    __tablename__ = "datasets"

    dataset_id = Column(String(100), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    num_tokens = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="datasets")

    def __repr__(self):
        return f"<Dataset(dataset_id={self.dataset_id}, user_id={self.user_id}, name={self.name})>"


class UploadedModel(Base):
    """User uploaded model (safetensors files)."""

    __tablename__ = "uploaded_models"

    # Keyed by (user_id, model_id): model_id is the model's name, and names are per user
    model_id = Column(String(100), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)

    # Model architecture info (extracted from safetensors metadata or config)
    num_parameters = Column(Integer, nullable=True)
    num_layers = Column(Integer, nullable=True)
    num_heads = Column(Integer, nullable=True)
    hidden_dim = Column(Integer, nullable=True)
    vocab_size = Column(Integer, nullable=True)
    max_seq_len = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="uploaded_models")

    def __repr__(self):
        return f"<UploadedModel(model_id={self.model_id}, user_id={self.user_id}, name={self.name})>"


class UserExternalSAE(Base):
    """Tracks which external SAEs a user has added to their workspace.

    SAE weights are stored in a shared Modal volume (/saes/external/{sae_id}/).
    This table maps users to the SAEs they've loaded, avoiding duplication of
    weights while giving each user their own SAE list.

    Users are limited to 1 uploaded (safetensors) SAE. HuggingFace/Gemma Scope
    SAEs are unlimited since they use shared storage.
    """

    __tablename__ = "user_external_saes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    sae_id = Column(String(500), nullable=False)  # Maps to folder name in Modal volume
    source = Column(String(50), nullable=False)  # 'huggingface', 'gemma_scope', 'user_upload'
    source_id = Column(String(500), nullable=True)  # e.g. repo_id for HF
    display_name = Column(String(500), nullable=False)
    base_model = Column(String(255), nullable=True)
    hookpoint = Column(String(255), nullable=True)
    activation_type = Column(String(50), nullable=True)
    d_input = Column(Integer, nullable=True)
    d_hidden = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="external_saes")

    def __repr__(self):
        return f"<UserExternalSAE(id={self.id}, user_id={self.user_id}, sae_id={self.sae_id})>"
