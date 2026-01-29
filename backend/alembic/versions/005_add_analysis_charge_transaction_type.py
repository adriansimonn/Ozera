"""Add ANALYSIS_CHARGE to transactiontype enum

Revision ID: 005
Revises: 004
Create Date: 2026-01-28 22:45:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new enum value to PostgreSQL enum type
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'ANALYSIS_CHARGE'")


def downgrade() -> None:
    # Note: PostgreSQL does not support removing enum values directly.
    # To fully downgrade, you would need to:
    # 1. Create a new enum type without ANALYSIS_CHARGE
    # 2. Update all columns using the old type
    # 3. Drop the old type
    # 4. Rename the new type
    # This is typically not done in practice, so we leave this as a no-op.
    pass
