"""Add autointerp: feature_explanations, autointerp_jobs and feature_hypotheses, and the
AUTOINTERP_CHARGE / AUTOINTERP_REFUND transaction types.

Explanations live here, small and read by many views; their example sets stay on the SAE
volume next to the atlas they came from. A partial unique index keeps one running claim per
feature, models and versions, so two requests for the same feature at once run it once.

Revision ID: 018
Revises: 017
Create Date: 2026-10-16 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '018'
down_revision = '017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'AUTOINTERP_CHARGE'")
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'AUTOINTERP_REFUND'")

    op.create_table(
        'feature_explanations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('atlas_key', sa.String(800), nullable=False),
        sa.Column('sae_key', sa.String(800), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('atlas', sa.JSON(), nullable=True),
        sa.Column('feature', sa.Integer(), nullable=False),
        sa.Column('is_control', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('explainer_model', sa.String(100), nullable=False),
        sa.Column('scorer_model', sa.String(100), nullable=False),
        sa.Column('prompt_version', sa.Integer(), nullable=False),
        sa.Column('examples_version', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('note', sa.String(500), nullable=True),
        sa.Column('detection', sa.Float(), nullable=True),
        sa.Column('detection_hard', sa.Float(), nullable=True),
        sa.Column('fuzzing', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('token_detection', sa.Float(), nullable=True),
        sa.Column('decile_recall', sa.JSON(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('verdicts', sa.JSON(), nullable=True),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('job_id', sa.String(50), nullable=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )
    op.create_index('ix_feature_explanations_atlas_feature', 'feature_explanations', ['atlas_key', 'feature'])
    op.create_index(
        'uq_feature_explanations_running', 'feature_explanations',
        ['atlas_key', 'feature', 'explainer_model', 'scorer_model', 'prompt_version', 'examples_version'],
        unique=True, postgresql_where=sa.text("status = 'running'"),
    )
    # Full-text search over explanations
    op.execute(
        "CREATE INDEX ix_feature_explanations_search ON feature_explanations "
        "USING gin (to_tsvector('english', coalesce(explanation, '')))"
    )

    op.create_table(
        'autointerp_jobs',
        sa.Column('job_id', sa.String(50), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('sae', sa.JSON(), nullable=False),
        sa.Column('sae_name', sa.String(500), nullable=False),
        sa.Column('atlas_key', sa.String(800), nullable=False, index=True),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('explainer_model', sa.String(100), nullable=False),
        sa.Column('scorer_model', sa.String(100), nullable=False),
        sa.Column('atlas', sa.JSON(), nullable=True),
        sa.Column('selection', sa.JSON(), nullable=False),
        sa.Column('features', sa.Integer(), nullable=False),
        sa.Column('scored', sa.Integer(), nullable=False),
        sa.Column('controls', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), server_default='queued', nullable=False, index=True),
        sa.Column('stage', sa.String(20), nullable=True),
        sa.Column('modal_call_id', sa.String(100), nullable=True),
        sa.Column('batch_ids', sa.JSON(), nullable=True),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('bound_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('estimated_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('settled_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('refunded_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('cpu_seconds', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'feature_hypotheses',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('atlas_key', sa.String(800), nullable=False, index=True),
        sa.Column('sae_key', sa.String(800), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('feature', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('scorer_model', sa.String(100), nullable=False),
        sa.Column('prompt_version', sa.Integer(), nullable=False),
        sa.Column('examples_version', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('detection', sa.Float(), nullable=True),
        sa.Column('detection_hard', sa.Float(), nullable=True),
        sa.Column('fuzzing', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('decile_recall', sa.JSON(), nullable=True),
        sa.Column('verdicts', sa.JSON(), nullable=True),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    # Row level security, like every other table (see 009, 010, 015 and 017). Explanations are
    # served only through the backend (which bypasses RLS), so they get no policy.
    for table in ('feature_explanations', 'autointerp_jobs', 'feature_hypotheses'):
        op.execute(f'ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own autointerp jobs"
        ON public.autointerp_jobs
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)
    op.execute("""
        CREATE POLICY "Users can read own feature hypotheses"
        ON public.feature_hypotheses
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own feature hypotheses" ON public.feature_hypotheses')
    op.execute('DROP POLICY IF EXISTS "Users can read own autointerp jobs" ON public.autointerp_jobs')
    op.drop_table('feature_hypotheses')
    op.drop_table('autointerp_jobs')
    op.execute('DROP INDEX IF EXISTS ix_feature_explanations_search')
    op.drop_table('feature_explanations')
    # PostgreSQL can't drop an enum value; like 002-006 and 014-017, the AUTOINTERP types stay
