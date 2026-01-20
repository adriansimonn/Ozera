"""Initial schema with users, credits, transactions, jobs, and datasets

Revision ID: 001
Revises:
Create Date: 2026-01-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Create credit_balances table
    op.create_table(
        'credit_balances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('balance_usd', sa.Float(), nullable=False),
        sa.Column('reserved_usd', sa.Float(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_credit_balances_id'), 'credit_balances', ['id'], unique=False)

    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('dataset_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('num_tokens', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('dataset_id')
    )
    op.create_index(op.f('ix_datasets_created_at'), 'datasets', ['created_at'], unique=False)
    op.create_index(op.f('ix_datasets_user_id'), 'datasets', ['user_id'], unique=False)

    # Create training_jobs table
    op.create_table(
        'training_jobs',
        sa.Column('job_id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.String(length=100), nullable=False),
        sa.Column('dataset_name', sa.String(length=255), nullable=True),
        sa.Column('model_config', sa.String(length=50), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('epochs', sa.Integer(), nullable=False),
        sa.Column('batch_size', sa.Integer(), nullable=False),
        sa.Column('learning_rate', sa.Float(), nullable=False),
        sa.Column('seq_len', sa.Integer(), nullable=False),
        sa.Column('gpu_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='jobstatus'), nullable=False),
        sa.Column('modal_call_id', sa.String(length=255), nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('estimated_minutes', sa.Float(), nullable=False),
        sa.Column('reserved_credits_usd', sa.Float(), nullable=False),
        sa.Column('actual_cost_usd', sa.Float(), nullable=True),
        sa.Column('actual_minutes', sa.Float(), nullable=True),
        sa.Column('current_epoch', sa.Integer(), nullable=True),
        sa.Column('total_epochs', sa.Integer(), nullable=True),
        sa.Column('train_loss', sa.Float(), nullable=True),
        sa.Column('val_loss', sa.Float(), nullable=True),
        sa.Column('train_ppl', sa.Float(), nullable=True),
        sa.Column('val_ppl', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index(op.f('ix_training_jobs_created_at'), 'training_jobs', ['created_at'], unique=False)
    op.create_index(op.f('ix_training_jobs_modal_call_id'), 'training_jobs', ['modal_call_id'], unique=False)
    op.create_index(op.f('ix_training_jobs_status'), 'training_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_training_jobs_user_id'), 'training_jobs', ['user_id'], unique=False)

    # Create transactions table
    op.create_table(
        'transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('amount_usd', sa.Float(), nullable=False),
        sa.Column('transaction_type', sa.Enum('CREDIT_PURCHASE', 'TRAINING_CHARGE', 'TRAINING_REFUND', 'ADMIN_ADJUSTMENT', name='transactiontype'), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('stripe_payment_intent_id', sa.String(length=255), nullable=True),
        sa.Column('training_job_id', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['training_job_id'], ['training_jobs.job_id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transactions_created_at'), 'transactions', ['created_at'], unique=False)
    op.create_index(op.f('ix_transactions_id'), 'transactions', ['id'], unique=False)
    op.create_index(op.f('ix_transactions_stripe_payment_intent_id'), 'transactions', ['stripe_payment_intent_id'], unique=False)
    op.create_index(op.f('ix_transactions_user_id'), 'transactions', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_transactions_user_id'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_stripe_payment_intent_id'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_id'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_created_at'), table_name='transactions')
    op.drop_table('transactions')

    op.drop_index(op.f('ix_training_jobs_user_id'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_status'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_modal_call_id'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_created_at'), table_name='training_jobs')
    op.drop_table('training_jobs')

    op.drop_index(op.f('ix_datasets_user_id'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_created_at'), table_name='datasets')
    op.drop_table('datasets')

    op.drop_index(op.f('ix_credit_balances_id'), table_name='credit_balances')
    op.drop_table('credit_balances')

    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    # Drop enums
    sa.Enum(name='transactiontype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='jobstatus').drop(op.get_bind(), checkfirst=True)
