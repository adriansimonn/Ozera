"""Key uploaded models by (user_id, model_id) instead of model_id alone.

model_id is the model's name, and model names are per user. With model_id alone as the
primary key, a user uploading a name another user already had got an error (after their
own previous model had already been deleted by the overwrite).

Downgrading fails if two users have uploaded models with the same name.

Revision ID: 013
Revises: 012
Create Date: 2026-09-29 00:00:02.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint('uploaded_models_pkey', 'uploaded_models', type_='primary')
    op.create_primary_key('uploaded_models_pkey', 'uploaded_models', ['user_id', 'model_id'])


def downgrade() -> None:
    op.drop_constraint('uploaded_models_pkey', 'uploaded_models', type_='primary')
    op.create_primary_key('uploaded_models_pkey', 'uploaded_models', ['model_id'])
