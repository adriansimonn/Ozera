"""Add settings JSON column to users table

Revision ID: 008
Revises: 007
Create Date: 2026-03-03 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('settings', sa.JSON(), nullable=False, server_default='{}'))


def downgrade() -> None:
    op.drop_column('users', 'settings')
