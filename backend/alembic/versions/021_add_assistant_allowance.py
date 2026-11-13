"""Add a chat's allowance to the assistant (ASSISTANT_PLAN.md, Phase 3): what runs may cost in all
without the user's click, and what's been taken from it. Off (0) by default.

Revision ID: 021
Revises: 020
Create Date: 2026-11-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '021'
down_revision = '020'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('assistant_chats', sa.Column('allowance_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False))
    op.add_column('assistant_chats', sa.Column('allowance_spent_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False))


def downgrade() -> None:
    op.drop_column('assistant_chats', 'allowance_spent_usd')
    op.drop_column('assistant_chats', 'allowance_usd')
