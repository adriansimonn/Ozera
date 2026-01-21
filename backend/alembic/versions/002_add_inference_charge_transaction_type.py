"""Add INFERENCE_CHARGE to transactiontype enum

Revision ID: 002
Revises: 001
Create Date: 2026-01-21 21:15:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new enum value to PostgreSQL enum type
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'INFERENCE_CHARGE'")


def downgrade() -> None:
    # Note: PostgreSQL does not support removing enum values directly.
    # To fully downgrade, you would need to:
    # 1. Create a new enum type without INFERENCE_CHARGE
    # 2. Update all columns using the old type
    # 3. Drop the old type
    # 4. Rename the new type
    # This is typically not done in practice, so we leave this as a no-op.
    pass
