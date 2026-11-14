"""Replace a chat's allowance with its mode and hard budget: mode (manual: every tool call waits
for approval; auto: none do), budget_usd (a cap on everything the chat costs; null: none) and
run_cost_usd (what its runs were charged, alongside cost_usd for its calls).

Revision ID: 022
Revises: 021
Create Date: 2026-11-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '022'
down_revision = '021'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('assistant_chats', sa.Column('mode', sa.String(length=10), server_default='manual', nullable=False))
    op.add_column('assistant_chats', sa.Column('budget_usd', sa.Numeric(precision=10, scale=4), nullable=True))
    op.add_column('assistant_chats', sa.Column('run_cost_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False))
    op.drop_column('assistant_chats', 'allowance_spent_usd')
    op.drop_column('assistant_chats', 'allowance_usd')


def downgrade() -> None:
    op.add_column('assistant_chats', sa.Column('allowance_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False))
    op.add_column('assistant_chats', sa.Column('allowance_spent_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False))
    op.drop_column('assistant_chats', 'run_cost_usd')
    op.drop_column('assistant_chats', 'budget_usd')
    op.drop_column('assistant_chats', 'mode')
