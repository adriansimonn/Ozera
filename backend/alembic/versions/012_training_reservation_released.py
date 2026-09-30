"""Track whether a training job's credit reservation was released, so it's released once.

A failed submission used to refund the reservation but leave the job PENDING/QUEUED, and
cancelling that job refunded it again, driving reserved_usd negative. This adds the flag,
sets it for jobs already charged or refunded, fails the jobs left behind by failed
submissions, and recomputes reserved_usd from the reservations still outstanding.

Revision ID: 012
Revises: 011
Create Date: 2026-09-29 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'training_jobs',
        sa.Column('reservation_released', sa.Boolean(), server_default=sa.false(), nullable=False),
    )

    # Every release (charge_credits or refund_credits) records a charge or refund for the job
    op.execute("""
        UPDATE training_jobs SET reservation_released = true
        WHERE job_id IN (
            SELECT training_job_id FROM transactions
            WHERE training_job_id IS NOT NULL
              AND transaction_type IN ('TRAINING_CHARGE', 'TRAINING_REFUND')
        )
    """)

    # Jobs whose submission failed: refunded, but never marked failed
    op.execute("""
        UPDATE training_jobs
        SET status = 'FAILED',
            error_message = COALESCE(error_message, 'Job submission failed'),
            completed_at = COALESCE(completed_at, now() AT TIME ZONE 'utc')
        WHERE reservation_released AND status IN ('PENDING', 'QUEUED')
    """)

    # reserved_usd is the sum of the reservations not yet released; double releases left it
    # below that (inflating available balance)
    op.execute("""
        UPDATE credit_balances cb
        SET reserved_usd = COALESCE((
            SELECT SUM(tj.reserved_credits_usd) FROM training_jobs tj
            WHERE tj.user_id = cb.user_id AND NOT tj.reservation_released
        ), 0)
    """)


def downgrade() -> None:
    op.drop_column('training_jobs', 'reservation_released')
