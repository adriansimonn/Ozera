"""Add atlas_jobs table and GEOMETRY_REFUND transaction type.

Atlas jobs run a model and an SAE over corpus text on a GPU container of their own; they're
charged when submitted, and a failed job is refunded all but the GPU time it used.

Revision ID: 017
Revises: 016
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '017'
down_revision = '016'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'GEOMETRY_REFUND'")

    op.create_table(
        'atlas_jobs',
        sa.Column('job_id', sa.String(50), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('sae', sa.JSON(), nullable=False),
        sa.Column('sae_name', sa.String(500), nullable=False),
        sa.Column('atlas_key', sa.String(800), nullable=False, index=True),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), server_default='queued', nullable=False),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('refunded_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('estimated_seconds', sa.Float(), nullable=False),
        sa.Column('modal_call_id', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
    )

    # Row level security, like every other table (see 009, 010 and 015)
    op.execute('ALTER TABLE public.atlas_jobs ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own atlas jobs"
        ON public.atlas_jobs
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own atlas jobs" ON public.atlas_jobs')
    op.drop_table('atlas_jobs')
    # PostgreSQL can't drop an enum value; like 002-006 and 014-016, GEOMETRY_REFUND stays
