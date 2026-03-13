"""Add user_external_saes table for per-user external SAE tracking.

Instead of showing all external SAEs to all users, each user now has their
own list of external SAEs they've loaded. SAE weights remain in a shared
Modal volume to avoid duplication.

Revision ID: 010
Revises: 009
Create Date: 2026-03-12 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'user_external_saes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('sae_id', sa.String(500), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('source_id', sa.String(500), nullable=True),
        sa.Column('display_name', sa.String(500), nullable=False),
        sa.Column('base_model', sa.String(255), nullable=True),
        sa.Column('hookpoint', sa.String(255), nullable=True),
        sa.Column('activation_type', sa.String(50), nullable=True),
        sa.Column('d_input', sa.Integer(), nullable=True),
        sa.Column('d_hidden', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    # Each user can only have one reference to a given sae_id
    op.create_unique_constraint('uq_user_sae', 'user_external_saes', ['user_id', 'sae_id'])

    # Enable RLS
    op.execute('ALTER TABLE public.user_external_saes ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own external SAEs"
        ON public.user_external_saes
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own external SAEs" ON public.user_external_saes')
    op.drop_table('user_external_saes')
