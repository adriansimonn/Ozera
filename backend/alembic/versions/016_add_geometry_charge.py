"""Add GEOMETRY_CHARGE to transactiontype enum

Geometry Lab runs (concept geometry on the inference workers) are charged like probe runs.

Revision ID: 016
Revises: 015
Create Date: 2026-10-05 00:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '016'
down_revision = '015'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'GEOMETRY_CHARGE'")


def downgrade() -> None:
    # PostgreSQL can't drop an enum value; like 002-006, 014 and 015, leave it in place
    pass
