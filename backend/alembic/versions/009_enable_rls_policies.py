"""Enable Row Level Security and add policies for all public tables.

Fixes Supabase Security Advisor vulnerabilities: RLS was disabled on all
public tables, meaning anyone with the anon key could query/modify data
directly via PostgREST.

Policies:
- Users can only SELECT their own data (matched via supabase_user_id / auth.uid())
- All INSERT/UPDATE/DELETE happens through the backend (postgres role, bypasses RLS)
- alembic_version: no access via PostgREST (internal migration tracking)

Revision ID: 009
Revises: 008
Create Date: 2026-03-04 00:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable RLS on all public tables
    tables = [
        'users',
        'credit_balances',
        'transactions',
        'training_jobs',
        'datasets',
        'uploaded_models',
        'alembic_version',
    ]
    for table in tables:
        op.execute(f'ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY')

    # -- users: match on supabase_user_id = auth.uid()
    op.execute("""
        CREATE POLICY "Users can read own record"
        ON public.users
        FOR SELECT
        TO authenticated
        USING (supabase_user_id = auth.uid()::text)
    """)

    # -- credit_balances: join through users to match owner
    op.execute("""
        CREATE POLICY "Users can read own balance"
        ON public.credit_balances
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)

    # -- transactions: join through users to match owner
    op.execute("""
        CREATE POLICY "Users can read own transactions"
        ON public.transactions
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)

    # -- training_jobs: join through users to match owner
    op.execute("""
        CREATE POLICY "Users can read own jobs"
        ON public.training_jobs
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)

    # -- datasets: join through users to match owner
    op.execute("""
        CREATE POLICY "Users can read own datasets"
        ON public.datasets
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)

    # -- uploaded_models: join through users to match owner
    op.execute("""
        CREATE POLICY "Users can read own models"
        ON public.uploaded_models
        FOR SELECT
        TO authenticated
        USING (user_id IN (
            SELECT id FROM public.users WHERE supabase_user_id = auth.uid()::text
        ))
    """)

    # -- alembic_version: no access via PostgREST (RLS enabled with no policies = deny all)


def downgrade() -> None:
    # Drop all policies
    op.execute('DROP POLICY IF EXISTS "Users can read own record" ON public.users')
    op.execute('DROP POLICY IF EXISTS "Users can read own balance" ON public.credit_balances')
    op.execute('DROP POLICY IF EXISTS "Users can read own transactions" ON public.transactions')
    op.execute('DROP POLICY IF EXISTS "Users can read own jobs" ON public.training_jobs')
    op.execute('DROP POLICY IF EXISTS "Users can read own datasets" ON public.datasets')
    op.execute('DROP POLICY IF EXISTS "Users can read own models" ON public.uploaded_models')

    # Disable RLS on all tables
    tables = [
        'users',
        'credit_balances',
        'transactions',
        'training_jobs',
        'datasets',
        'uploaded_models',
        'alembic_version',
    ]
    for table in tables:
        op.execute(f'ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY')
