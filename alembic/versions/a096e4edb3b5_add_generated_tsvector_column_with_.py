"""Add generated TSVECTOR column with weights

Revision ID: a096e4edb3b5
Revises: df29e8ab4adf
Create Date: 2024-06-19 20:38:57.308100

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a096e4edb3b5'
down_revision: Union[str, None] = 'df29e8ab4adf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add generated TSVECTOR column for filename, summary, and file_content with weights
    op.execute(
        """
                ALTER TABLE documents 
                ADD COLUMN tsv_content tsvector 
                GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(filename, '')), 'A') || 
                    setweight(to_tsvector('english', coalesce(summary, '')), 'B') || 
                    setweight(to_tsvector('english', coalesce(file_content, '')), 'C')
                ) STORED
            """
        )

    # Create GIN index on the generated TSVECTOR column
    op.create_index('ix_documents_tsv_content', 'documents', ['tsv_content'], postgresql_using='gin')


def downgrade():
    # Drop the GIN index
    op.drop_index('ix_documents_tsv_content', table_name='documents')

    # Drop the generated TSVECTOR column
    op.execute(
        """
                ALTER TABLE documents 
                DROP COLUMN tsv_content
            """
        )
