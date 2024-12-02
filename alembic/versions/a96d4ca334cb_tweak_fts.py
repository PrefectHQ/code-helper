"""tweak-fts

Revision ID: a96d4ca334cb
Revises: efe803356999
Create Date: 2024-11-30 22:19:08.138582

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a96d4ca334cb'
down_revision: Union[str, None] = 'efe803356999'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First create the text search configuration
    op.execute("""
        DROP TEXT SEARCH CONFIGURATION IF EXISTS code_search;
        CREATE TEXT SEARCH CONFIGURATION code_search (COPY = english);
        ALTER TEXT SEARCH CONFIGURATION code_search 
            ALTER MAPPING FOR word, asciiword WITH simple, english_stem;
    """)
    
    # Then update the TSVector column
    op.execute("""
        ALTER TABLE document_fragments 
        ALTER COLUMN fragment_content_tsv SET DATA TYPE tsvector 
        USING setweight(to_tsvector('code_search', coalesce(meta->>'name', '')), 'A') || 
              setweight(to_tsvector('code_search', coalesce(meta->>'type', '')), 'A') || 
              setweight(to_tsvector('code_search', 
                  regexp_replace(coalesce(summary, ''), '[-_]', ' ', 'g')
              ), 'B') || 
              setweight(to_tsvector('code_search', 
                  regexp_replace(coalesce(fragment_content, ''), '[-_]', ' ', 'g')
              ), 'C')
    """)

    # ### end Alembic commands ###


def downgrade() -> None:
    # Add downgrade commands
    op.execute("DROP TEXT SEARCH CONFIGURATION IF EXISTS code_search;")
    # ### end Alembic commands ###
