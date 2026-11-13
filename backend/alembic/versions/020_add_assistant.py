"""Add the assistant (ASSISTANT_PLAN.md, Phase 1): assistant_chats, assistant_messages,
assistant_free_usage, and the ASSISTANT transaction type.

Messages keep their content as json, not jsonb: it's sent back to the Messages API exactly as
stored, and jsonb reorders object keys (a replayed tool call's input would then render
differently, missing the prompt cache and failing thinking blocks' prefix check). A user's
free messages are counted per UTC day, one row per user and day, claimed with a conditional
upsert so parallel chats can't exceed the day's allowance.

Revision ID: 020
Revises: 019
Create Date: 2026-11-12 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '020'
down_revision = '019'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'ASSISTANT'")

    op.create_table(
        'assistant_chats',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), server_default='idle', nullable=False),
        sa.Column('pending', sa.JSON(), nullable=True),
        sa.Column('tokens', sa.Integer(), server_default='0', nullable=False),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )

    op.create_table(
        'assistant_messages',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('chat_id', sa.Integer(), sa.ForeignKey('assistant_chats.id', ondelete='CASCADE'), nullable=False),
        sa.Column('seq', sa.Integer(), nullable=False),
        sa.Column('turn', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.JSON(), nullable=False),
        sa.Column('snapshot', sa.JSON(), nullable=True),
        sa.Column('usage', sa.JSON(), nullable=True),
        sa.Column('cost_usd', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('free', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_assistant_messages_chat_seq', 'assistant_messages', ['chat_id', 'seq'], unique=True)

    op.create_table(
        'assistant_free_usage',
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('day', sa.Date(), primary_key=True),
        sa.Column('count', sa.Integer(), server_default='0', nullable=False),
    )

    # Row level security, like every other table (see 018 and 019). Chats are private: the
    # backend (which bypasses RLS) serves them; these policies only let users read their own.
    for table in ('assistant_chats', 'assistant_messages', 'assistant_free_usage'):
        op.execute(f'ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY')
    op.execute("""
        CREATE POLICY "Users can read own assistant chats"
        ON public.assistant_chats
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)
    op.execute("""
        CREATE POLICY "Users can read own assistant messages"
        ON public.assistant_messages
        FOR SELECT
        TO authenticated
        USING (chat_id IN (
            SELECT c.id FROM public.assistant_chats c
            JOIN public.users u ON u.id = c.user_id
            WHERE u.supabase_user_id = auth.uid()::text
        ))
    """)
    op.execute("""
        CREATE POLICY "Users can read own assistant free usage"
        ON public.assistant_free_usage
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)


def downgrade() -> None:
    op.execute('DROP POLICY IF EXISTS "Users can read own assistant free usage" ON public.assistant_free_usage')
    op.execute('DROP POLICY IF EXISTS "Users can read own assistant messages" ON public.assistant_messages')
    op.execute('DROP POLICY IF EXISTS "Users can read own assistant chats" ON public.assistant_chats')
    op.drop_table('assistant_free_usage')
    op.drop_table('assistant_messages')
    op.drop_table('assistant_chats')
    # PostgreSQL can't drop an enum value; like 002-006 and 014-019, ASSISTANT stays
