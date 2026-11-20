"""Add circuit_jobs table and the CIRCUIT_CHARGE and CIRCUIT_REFUND transaction types.

Circuit Tracing jobs build an attribution graph of one prompt on the circuits worker; they're
charged when submitted, and a failed job is refunded all but the GPU time it used. Running an
intervention on a graph is charged as CIRCUIT_CHARGE too.

Revision ID: 024
Revises: 023
Create Date: 2026-11-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '024'
down_revision = '023'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'CIRCUIT_CHARGE'")
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'CIRCUIT_REFUND'")

    op.create_table(
        'circuit_jobs',
        sa.Column('job_id', sa.String(50), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('transcoders', sa.String(100), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('target', sa.String(200), nullable=True),
        sa.Column('status', sa.String(20), server_default='queued', nullable=False),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('refunded_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('estimated_seconds', sa.Float(), nullable=False),
        sa.Column('modal_call_id', sa.String(100), nullable=True),
        sa.Column('summary', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
    )

    # Row level security, like every other table (see 009, 010, 015 and 017)
    op.execute('ALTER TABLE public.circuit_jobs ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own circuit jobs"
        ON public.circuit_jobs
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own circuit jobs" ON public.circuit_jobs')
    op.drop_table('circuit_jobs')
    # PostgreSQL can't drop an enum value; like 002-006 and 014-020, the two types stay
