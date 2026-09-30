"""Make transactions.stripe_payment_intent_id unique, so a payment can't be credited twice.

The Stripe webhook and the frontend's confirm call both credit a payment, usually within a
second of each other. The index makes the second insert fail even if both pass the
application's existence check.

Revision ID: 011
Revises: 010
Create Date: 2026-09-29 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    duplicates = op.get_bind().execute(sa.text("""
        SELECT stripe_payment_intent_id, COUNT(*)
        FROM transactions
        WHERE stripe_payment_intent_id IS NOT NULL
        GROUP BY stripe_payment_intent_id
        HAVING COUNT(*) > 1
    """)).fetchall()
    if duplicates:
        listed = ", ".join(f"{payment_id} ({count} transactions)" for payment_id, count in duplicates)
        raise RuntimeError(
            f"These Stripe payments were credited more than once: {listed}. Reconcile them "
            "before rerunning this migration: keep one transaction per payment, set "
            "stripe_payment_intent_id to NULL on the others, and correct the balances "
            "(e.g. with an ADMIN_ADJUSTMENT transaction)."
        )

    op.drop_index('ix_transactions_stripe_payment_intent_id', table_name='transactions', if_exists=True)
    op.create_index(
        'ix_transactions_stripe_payment_intent_id',
        'transactions',
        ['stripe_payment_intent_id'],
        unique=True,
        postgresql_where=sa.text('stripe_payment_intent_id IS NOT NULL'),
    )


def downgrade() -> None:
    op.drop_index('ix_transactions_stripe_payment_intent_id', table_name='transactions')
    op.create_index(
        'ix_transactions_stripe_payment_intent_id',
        'transactions',
        ['stripe_payment_intent_id'],
        unique=False,
    )
