"""Add INFERENCE_REFUND to transactiontype enum

Streaming generation is charged for max_tokens up front; the unused part is refunded when
the stream ends (or all of it, if generation fails).

Revision ID: 014
Revises: 013
Create Date: 2026-09-30 00:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'INFERENCE_REFUND'")


def downgrade() -> None:
    # PostgreSQL can't drop an enum value; like 002-006, leave it in place
    pass
