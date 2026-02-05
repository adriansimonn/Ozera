"""Add SAE_CHARGE to transactiontype enum

Revision ID: 006
Revises: 005
Create Date: 2026-02-04 00:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new enum value to PostgreSQL enum type
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'SAE_CHARGE'")


def downgrade() -> None:
    # Note: PostgreSQL does not support removing enum values directly.
    # To fully downgrade, you would need to:
    # 1. Create a new enum type without SAE_CHARGE
    # 2. Update all columns using the old type
    # 3. Drop the old type
    # 4. Rename the new type
    # This is typically not done in practice, so we leave this as a no-op.
    pass
