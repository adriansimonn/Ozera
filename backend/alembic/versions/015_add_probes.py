"""Add probes table and PROBE_CHARGE transaction type.

Probes are linear probes users save from the Probe Lab's training runs. Their weights are
stored as float32 bytes (at most a few KB per probe).

Revision ID: 015
Revises: 014
Create Date: 2026-10-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'PROBE_CHARGE'")

    op.create_table(
        'probes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('model_version', sa.String(100), nullable=True),
        sa.Column('layer', sa.Integer(), nullable=False),
        sa.Column('pooling', sa.String(20), nullable=False),
        sa.Column('method', sa.String(20), nullable=False),
        sa.Column('chat_template', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('read_span', sa.String(20), server_default='text', nullable=False),
        sa.Column('hidden_dim', sa.Integer(), nullable=False),
        sa.Column('weights', sa.LargeBinary(), nullable=False),
        sa.Column('bias', sa.Float(), nullable=False),
        sa.Column('normalization', sa.JSON(), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('dataset', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    # Row level security, like every other table (see 009 and 010)
    op.execute('ALTER TABLE public.probes ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own probes"
        ON public.probes
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own probes" ON public.probes')
    op.drop_table('probes')
    # PostgreSQL can't drop an enum value; like 002-006 and 014, PROBE_CHARGE stays
