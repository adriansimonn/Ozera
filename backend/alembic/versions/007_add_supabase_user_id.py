"""Add supabase_user_id to users table and make hashed_password nullable

Revision ID: 007
Revises: 006
Create Date: 2026-02-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('supabase_user_id', sa.String(36), nullable=True))
    op.create_unique_constraint('uq_users_supabase_user_id', 'users', ['supabase_user_id'])
    op.create_index('ix_users_supabase_user_id', 'users', ['supabase_user_id'])
    op.alter_column('users', 'hashed_password', existing_type=sa.String(255), nullable=True)


def downgrade() -> None:
    op.alter_column('users', 'hashed_password', existing_type=sa.String(255), nullable=False)
    op.drop_index('ix_users_supabase_user_id', table_name='users')
    op.drop_constraint('uq_users_supabase_user_id', 'users', type_='unique')
    op.drop_column('users', 'supabase_user_id')
