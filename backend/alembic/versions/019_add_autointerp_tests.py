"""Add what testing explanations needs (AUTOINTERP_PLAN.md, Phase 3): generation_tests,
community_names, and feature_hypotheses' atlas build, note, token baseline and details.

A generation test of an autointerp explanation is shared like the explanation; one of a user's
hypothesis is private, like the hypothesis. Community names are shared, and kept with the atlas
build whose communities they name.

Revision ID: 019
Revises: 018
Create Date: 2026-11-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '019'
down_revision = '018'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('feature_hypotheses', sa.Column('atlas', sa.JSON(), nullable=True))
    op.add_column('feature_hypotheses', sa.Column('note', sa.String(500), nullable=True))
    op.add_column('feature_hypotheses', sa.Column('token_detection', sa.Float(), nullable=True))
    op.add_column('feature_hypotheses', sa.Column('details', sa.JSON(), nullable=True))

    op.create_table(
        'generation_tests',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('atlas_key', sa.String(800), nullable=False),
        sa.Column('sae_key', sa.String(800), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('atlas', sa.JSON(), nullable=True),
        sa.Column('feature', sa.Integer(), nullable=False),
        sa.Column('explanation_id', sa.Integer(), sa.ForeignKey('feature_explanations.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('hypothesis_id', sa.Integer(), sa.ForeignKey('feature_hypotheses.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('generator_model', sa.String(100), nullable=False),
        sa.Column('prompt_version', sa.Integer(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('maximum', sa.Float(), nullable=False),
        sa.Column('firing_rate', sa.Float(), nullable=True),
        sa.Column('near_miss_rate', sa.Float(), nullable=True),
        sa.Column('texts', sa.JSON(), nullable=False),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )
    op.create_index('ix_generation_tests_atlas_feature', 'generation_tests', ['atlas_key', 'feature'])

    op.create_table(
        'community_names',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('atlas_key', sa.String(800), nullable=False),
        sa.Column('sae_key', sa.String(800), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('corpus', sa.String(50), nullable=False),
        sa.Column('atlas_created_at', sa.String(40), nullable=False),
        sa.Column('community', sa.Integer(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('namer_model', sa.String(100), nullable=False),
        sa.Column('prompt_version', sa.Integer(), nullable=False),
        sa.Column('explained', sa.Integer(), nullable=False),
        sa.Column('size', sa.Integer(), nullable=False),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('job_id', sa.String(50), nullable=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )
    op.create_index('ix_community_names_atlas_community', 'community_names', ['atlas_key', 'community'])

    # Row level security, like every other table (see 018). Names, and tests of shared
    # explanations, are served only through the backend (which bypasses RLS).
    for table in ('generation_tests', 'community_names'):
        op.execute(f'ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own generation tests"
        ON public.generation_tests
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own generation tests" ON public.generation_tests')
    op.drop_table('community_names')
    op.drop_table('generation_tests')
    op.drop_column('feature_hypotheses', 'details')
    op.drop_column('feature_hypotheses', 'token_detection')
    op.drop_column('feature_hypotheses', 'note')
    op.drop_column('feature_hypotheses', 'atlas')
