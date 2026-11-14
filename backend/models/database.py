"""
SQLAlchemy database models for Ozera cloud training platform.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
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
    probes = relationship(
        "Probe", back_populates="user", cascade="all, delete-orphan"
    )
    atlas_jobs = relationship(
        "AtlasJob", back_populates="user", cascade="all, delete-orphan"
    )
    autointerp_jobs = relationship(
        "AutointerpJob", back_populates="user", cascade="all, delete-orphan"
    )
    feature_hypotheses = relationship(
        "FeatureHypothesis", back_populates="user", cascade="all, delete-orphan"
    )
    generation_tests = relationship(
        "GenerationTest", back_populates="user", cascade="all, delete-orphan"
    )
    assistant_chats = relationship(
        "AssistantChat", back_populates="user", cascade="all, delete-orphan"
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
    PROBE_CHARGE = "probe_charge"
    GEOMETRY_CHARGE = "geometry_charge"
    GEOMETRY_REFUND = "geometry_refund"
    AUTOINTERP_CHARGE = "autointerp_charge"
    AUTOINTERP_REFUND = "autointerp_refund"
    # The assistant's Claude calls: each call's hold (negative) and what it didn't use (positive)
    ASSISTANT = "assistant"


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


class Probe(Base):
    """
    A linear probe a user saved from a probe training run.

    The probe reads the residual stream after decoder layer `layer` of model `model_id`,
    pooled over each text's read span. Its score is weights · activation + bias, with the
    weights on raw activations (any standardization is folded in).
    """

    __tablename__ = "probes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    model_id = Column(String(255), nullable=False)
    # A custom model's version when the probe was saved (it changes when the model is retrained
    # or re-uploaded); None for shared models
    model_version = Column(String(100), nullable=True)
    layer = Column(Integer, nullable=False)
    pooling = Column(String(20), nullable=False)  # 'last', 'mean', 'max'
    method = Column(String(20), nullable=False)  # 'logreg', 'diff_means'
    chat_template = Column(Boolean, default=False, server_default=false(), nullable=False)
    read_span = Column(String(20), default="text", server_default="text", nullable=False)
    hidden_dim = Column(Integer, nullable=False)
    weights = Column(LargeBinary, nullable=False)  # hidden_dim little-endian float32 values
    bias = Column(Float, nullable=False)
    # Training-set score mean/std and activation norm at the probe's layer
    normalization = Column(JSON, nullable=False, default=dict)
    # The probe's metrics, and the layer sweep it was chosen from
    metrics = Column(JSON, nullable=False, default=dict)
    # What it was trained on: dataset name/ID, label names, split sizes, seed
    dataset = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="probes")

    def __repr__(self):
        return f"<Probe(id={self.id}, user_id={self.user_id}, model={self.model_id}, layer={self.layer})>"


class AtlasJob(Base):
    """
    A Geometry Lab atlas job: a model and an SAE over corpus text on a GPU container of its own
    (services/modal_feature_stats.py), charged when submitted.

    Results are stored next to the SAE on the SAE volume, not here, and serve everyone who has
    the SAE. A job that fails is refunded all but the GPU time it used.
    """

    __tablename__ = "atlas_jobs"

    job_id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String(255), nullable=False)
    sae = Column(JSON, nullable=False)  # The SAE reference (core.probes.sae.sae_directory)
    sae_name = Column(String(500), nullable=False)
    # The SAE's directory on the volume, the model and the corpus: one active job per atlas
    atlas_key = Column(String(800), nullable=False, index=True)
    corpus = Column(String(50), nullable=False)
    tokens = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued")  # queued, running, completed, failed
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=False)
    refunded_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    estimated_seconds = Column(Float, nullable=False)
    modal_call_id = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="atlas_jobs")

    def __repr__(self):
        return f"<AtlasJob(job_id={self.job_id}, user_id={self.user_id}, status={self.status})>"


class FeatureExplanation(Base):
    """
    An SAE feature's autointerp explanation and its scores (core.autointerp), from one atlas.

    Shared like atlases: an explanation of a public SAE serves everyone with the SAE in their
    list, whoever paid for it (uploads' atlases are per user already). Reproducible from what's
    stored: the atlas (key and build), the models, and the prompt and example-set versions.

    status: running (a claim while a request explains it; at most one per feature, models and
    versions), scored, unscored (explained, but its sets can't be scored) or refused.
    """

    __tablename__ = "feature_explanations"
    __table_args__ = (
        Index("ix_feature_explanations_atlas_feature", "atlas_key", "feature"),
        Index(
            "uq_feature_explanations_running",
            "atlas_key", "feature", "explainer_model", "scorer_model", "prompt_version", "examples_version",
            unique=True,
            postgresql_where=text("status = 'running'"),
            sqlite_where=text("status = 'running'"),
        ),
    )

    id = Column(Integer, primary_key=True)
    atlas_key = Column(String(800), nullable=False)  # The atlas's directory on the SAE volume
    sae_key = Column(String(800), nullable=False, index=True)  # The SAE's directory
    model_id = Column(String(255), nullable=False)
    corpus = Column(String(50), nullable=False)
    atlas = Column(JSON, nullable=True)  # The atlas build: created_at, tokens, seed, version
    feature = Column(Integer, nullable=False)
    is_control = Column(Boolean, nullable=False, default=False)  # A random-direction control
    explanation = Column(Text, nullable=True)
    explainer_model = Column(String(100), nullable=False)
    scorer_model = Column(String(100), nullable=False)
    prompt_version = Column(Integer, nullable=False)
    examples_version = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    note = Column(String(500), nullable=True)
    detection = Column(Float, nullable=True)
    detection_hard = Column(Float, nullable=True)  # Against token-matched negatives alone
    fuzzing = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    token_detection = Column(Float, nullable=True)  # The token baseline's detection
    decile_recall = Column(JSON, nullable=True)
    details = Column(JSON, nullable=True)  # Intervals, refused prompts, example counts
    verdicts = Column(JSON, nullable=True)  # The scorer's verdict on every example
    usage = Column(JSON, nullable=True)  # Tokens by model
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    job_id = Column(String(50), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<FeatureExplanation(atlas_key={self.atlas_key}, feature={self.feature}, status={self.status})>"


class AutointerpJob(Base):
    """
    An autointerp job: a selection of an SAE's features, and the atlas's random-direction
    controls, explained (and scored) through the Message Batches API in stages
    (services/autointerp_jobs.py): queued, then running through preparing → explaining →
    scoring → importing, then completed or failed. Holds its upper bound once its requests are
    counted, and is settled on the usage reported.
    """

    __tablename__ = "autointerp_jobs"

    job_id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String(255), nullable=False)
    sae = Column(JSON, nullable=False)
    sae_name = Column(String(500), nullable=False)
    atlas_key = Column(String(800), nullable=False, index=True)
    corpus = Column(String(50), nullable=False)
    atlas = Column(JSON, nullable=True)  # The atlas build it reads: created_at, tokens, seed, version
    explainer_model = Column(String(100), nullable=False)
    scorer_model = Column(String(100), nullable=False)
    selection = Column(JSON, nullable=False)  # How its features were chosen, and how many are scored
    features = Column(Integer, nullable=False)  # Features to explain (controls apart)
    scored = Column(Integer, nullable=False)  # Of those, features to score
    controls = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="queued", index=True)
    stage = Column(String(20), nullable=True)
    modal_call_id = Column(String(100), nullable=True)  # The stage running on Modal, if any
    batch_ids = Column(JSON, nullable=True)  # {"explain": [...], "score": [...]}
    usage = Column(JSON, nullable=True)  # Tokens by stage and model
    details = Column(JSON, nullable=True)  # Stage reports, batch progress, results imported
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=False)  # Charged in all (holds)
    bound_usd = Column(Numeric(precision=10, scale=4), nullable=True)  # The upper bound, once counted
    estimated_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    settled_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    refunded_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    cpu_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="autointerp_jobs")

    def __repr__(self):
        return f"<AutointerpJob(job_id={self.job_id}, user_id={self.user_id}, status={self.status})>"


class FeatureHypothesis(Base):
    """
    A user's own explanation of a feature, scored like autointerp's: by the same scorer and
    prompts, on the same held-out set of the atlas (private to the user).

    status: scored, unscored (the scorer's answers couldn't be read) or refused.
    """

    __tablename__ = "feature_hypotheses"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    atlas_key = Column(String(800), nullable=False, index=True)
    sae_key = Column(String(800), nullable=False)
    model_id = Column(String(255), nullable=False)
    corpus = Column(String(50), nullable=False)
    atlas = Column(JSON, nullable=True)  # The atlas build it was scored on: created_at, tokens, seed, version
    feature = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    scorer_model = Column(String(100), nullable=False)
    prompt_version = Column(Integer, nullable=False)
    examples_version = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    note = Column(String(500), nullable=True)
    detection = Column(Float, nullable=True)
    detection_hard = Column(Float, nullable=True)
    fuzzing = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    token_detection = Column(Float, nullable=True)  # The token baseline's detection on the same examples
    decile_recall = Column(JSON, nullable=True)
    details = Column(JSON, nullable=True)  # Intervals, refused prompts, example counts
    verdicts = Column(JSON, nullable=True)
    usage = Column(JSON, nullable=True)
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="feature_hypotheses")

    def __repr__(self):
        return f"<FeatureHypothesis(user_id={self.user_id}, feature={self.feature})>"


class GenerationTest(Base):
    """
    A generation test of one explanation of a feature (core.autointerp.pipeline): texts Claude
    wrote from the explanation (ones it says the feature fires on, and near-misses), with the
    feature's activation on each as the model and SAE read them, and the shares that fire.

    A test of an autointerp explanation (explanation_id) is shared like the explanation; a test
    of a user's hypothesis (hypothesis_id) is private to that user.
    """

    __tablename__ = "generation_tests"
    __table_args__ = (Index("ix_generation_tests_atlas_feature", "atlas_key", "feature"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Who ran (and paid for) it
    atlas_key = Column(String(800), nullable=False)
    sae_key = Column(String(800), nullable=False)
    model_id = Column(String(255), nullable=False)
    corpus = Column(String(50), nullable=False)
    atlas = Column(JSON, nullable=True)  # The atlas build whose maximum it's measured against
    feature = Column(Integer, nullable=False)
    explanation_id = Column(Integer, ForeignKey("feature_explanations.id", ondelete="CASCADE"), nullable=True, index=True)
    hypothesis_id = Column(Integer, ForeignKey("feature_hypotheses.id", ondelete="CASCADE"), nullable=True, index=True)
    text = Column(Text, nullable=False)  # The explanation tested
    generator_model = Column(String(100), nullable=False)
    prompt_version = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)  # Share of the maximum a text must reach to count as firing
    maximum = Column(Float, nullable=False)  # The feature's maximum over the corpus
    firing_rate = Column(Float, nullable=True)  # Share of the firing texts that reach the threshold
    near_miss_rate = Column(Float, nullable=True)  # Share of the near-misses that stay below it
    texts = Column(JSON, nullable=False)  # [{"kind", "tokens", "levels", "read", "peak"}]
    usage = Column(JSON, nullable=True)  # Tokens by model
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="generation_tests")

    def __repr__(self):
        return f"<GenerationTest(feature={self.feature}, explanation_id={self.explanation_id}, hypothesis_id={self.hypothesis_id})>"


class CommunityName(Base):
    """
    An atlas community's name (core.autointerp.communities), written by Claude from its
    features' explanations and contexts. Shared like explanations. Community IDs belong to one
    atlas build, so a name serves only the build it was made from (atlas_created_at).
    """

    __tablename__ = "community_names"
    __table_args__ = (Index("ix_community_names_atlas_community", "atlas_key", "community"),)

    id = Column(Integer, primary_key=True)
    atlas_key = Column(String(800), nullable=False)
    sae_key = Column(String(800), nullable=False)
    model_id = Column(String(255), nullable=False)
    corpus = Column(String(50), nullable=False)
    atlas_created_at = Column(String(40), nullable=False)
    community = Column(Integer, nullable=False)
    name = Column(Text, nullable=False)
    namer_model = Column(String(100), nullable=False)
    prompt_version = Column(Integer, nullable=False)
    explained = Column(Integer, nullable=False)  # Its features with an explanation when it was named
    size = Column(Integer, nullable=False)
    usage = Column(JSON, nullable=True)
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    job_id = Column(String(50), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<CommunityName(atlas_key={self.atlas_key}, community={self.community}, name={self.name})>"


class AssistantChat(Base):
    """
    One of a user's chats with the assistant (services/assistant.py). Private to the user.

    status: idle, or running while a turn runs (one at a time; updated_at says since when, so a
    turn whose process died doesn't lock the chat). tokens: the size of the chat's last prompt
    and reply (what the next call reads at least). cost_usd: what its turns' calls were charged;
    run_cost_usd: what its runs were charged. mode: manual (every tool call waits for approval) or
    auto. budget_usd: a hard cap on cost_usd + run_cost_usd (null: none).
    """

    __tablename__ = "assistant_chats"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    model = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="idle")
    pending = Column(JSON, nullable=True)  # Runs waiting for approval (Phase 2)
    tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=False, default=0)
    run_cost_usd = Column(Numeric(precision=10, scale=4), nullable=False, default=0)
    mode = Column(String(10), nullable=False, default="manual")
    budget_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="assistant_chats")
    messages = relationship(
        "AssistantMessage", back_populates="chat", cascade="all, delete-orphan", order_by="AssistantMessage.seq",
    )

    def __repr__(self):
        return f"<AssistantChat(id={self.id}, user_id={self.user_id}, model={self.model})>"


class AssistantMessage(Base):
    """
    One message of a chat, exactly as it's sent to the Messages API (content), in order (seq).
    A turn is a user message and everything the assistant did before replying (turn numbers
    them). Stored content is never edited: thinking blocks are bound to the conversation, and
    any edit breaks the prompt cache from that point on. meta holds what the panel shows
    (the user's text, tool chips, how the turn ended) and may change.
    """

    __tablename__ = "assistant_messages"
    __table_args__ = (Index("ix_assistant_messages_chat_seq", "chat_id", "seq", unique=True),)

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("assistant_chats.id", ondelete="CASCADE"), nullable=False)
    seq = Column(Integer, nullable=False)
    turn = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(JSON, nullable=False)
    snapshot = Column(JSON, nullable=True)  # The page state the user's message came with
    usage = Column(JSON, nullable=True)  # An assistant message's call: tokens by attempt
    cost_usd = Column(Numeric(precision=10, scale=4), nullable=True)
    free = Column(Boolean, nullable=False, default=False)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chat = relationship("AssistantChat", back_populates="messages")

    def __repr__(self):
        return f"<AssistantMessage(chat_id={self.chat_id}, seq={self.seq}, role={self.role})>"


class AssistantFreeUsage(Base):
    """Free assistant messages a user has started on a day (UTC)."""

    __tablename__ = "assistant_free_usage"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    day = Column(Date, primary_key=True)
    count = Column(Integer, nullable=False, default=0)
