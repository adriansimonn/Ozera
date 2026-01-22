"""Add uploaded_models table for user model uploads

Revision ID: 003
Revises: 002
Create Date: 2026-01-22 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'uploaded_models',
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('num_parameters', sa.Integer(), nullable=True),
        sa.Column('num_layers', sa.Integer(), nullable=True),
        sa.Column('num_heads', sa.Integer(), nullable=True),
        sa.Column('hidden_dim', sa.Integer(), nullable=True),
        sa.Column('vocab_size', sa.Integer(), nullable=True),
        sa.Column('max_seq_len', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('model_id')
    )
    op.create_index(op.f('ix_uploaded_models_created_at'), 'uploaded_models', ['created_at'], unique=False)
    op.create_index(op.f('ix_uploaded_models_user_id'), 'uploaded_models', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_uploaded_models_user_id'), table_name='uploaded_models')
    op.drop_index(op.f('ix_uploaded_models_created_at'), table_name='uploaded_models')
    op.drop_table('uploaded_models')
