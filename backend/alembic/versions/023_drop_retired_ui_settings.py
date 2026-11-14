"""Clean retired UI preferences out of users.settings: drop ui.interface (the glass interface is
gone) and ui.default_view_mode (the split/single toggle is gone), and turn ui.background "glow"
into "dark".

The downgrade is a no-op: the old values aren't kept, and the code that read them is gone.

Revision ID: 023
Revises: 022
Create Date: 2026-11-13 00:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '023'
down_revision = '022'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE users
        SET settings = (
            CASE WHEN settings::jsonb #>> '{ui,background}' = 'glow'
                 THEN jsonb_set(settings::jsonb, '{ui,background}', '"dark"')
                 ELSE settings::jsonb
            END #- '{ui,interface}' #- '{ui,default_view_mode}'
        )::json
        WHERE jsonb_typeof(settings::jsonb -> 'ui') = 'object'
          AND (settings::jsonb #> '{ui,interface}' IS NOT NULL
               OR settings::jsonb #> '{ui,default_view_mode}' IS NOT NULL
               OR settings::jsonb #>> '{ui,background}' = 'glow')
    """)


def downgrade() -> None:
    pass
