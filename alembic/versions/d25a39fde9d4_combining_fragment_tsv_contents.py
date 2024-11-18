"""combining fragment tsv contents'

Revision ID: d25a39fde9d4
Revises: 17b6b94de49c
Create Date: 2024-11-16 13:32:24.532432

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd25a39fde9d4'
down_revision: Union[str, None] = '17b6b94de49c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index('ix_document_fragments_summary_tsv', table_name='document_fragments', postgresql_using='gin')
    op.drop_column('document_fragments', 'summary_tsv')

    # Add the new tsv_content computed column
    op.add_column('document_fragments', sa.Column(
        'tsv_content',
        sa.TSVECTOR(),
        sa.Computed(
            "setweight(to_tsvector('english', coalesce(meta->>'name', '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(fragment_content, '')), 'C')",
            persisted=True
        ),
        nullable=True
    ))


def downgrade() -> None:
    op.drop_column("document_fragments", "tsv_content")
    op.add_column(
        "document_fragments",
        sa.Column(
            "summary_tsv", postgresql.TSVECTOR(), autoincrement=False, nullable=True
        ),
    )
    op.create_index(
        "ix_document_fragments_summary_tsv", "document_fragments", ["summary_tsv"], unique=False, postgresql_using="gin"
    )
