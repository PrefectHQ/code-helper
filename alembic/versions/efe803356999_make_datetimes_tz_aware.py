"""make datetimes tz-aware

Revision ID: efe803356999
Revises: b592c2b7dd26
Create Date: 2024-11-28 13:31:22.503361

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector

# revision identifiers, used by Alembic.
revision: str = 'efe803356999'
down_revision: Union[str, None] = 'b592c2b7dd26'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column('filepath', sa.String(), nullable=False),
    sa.Column('path_array', postgresql.ARRAY(sa.String()), nullable=False),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('tsv_content', postgresql.TSVECTOR(), sa.Computed("setweight(to_tsvector('english', coalesce(filename, '')), 'A') || setweight(to_tsvector('english', coalesce(summary, '')), 'B')", persisted=True), nullable=True),
    sa.Column('vector', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=False),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('hierarchy_meta', sa.JSON(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('filepath')
    )
    op.create_index('idx_document_filepath', 'documents', ['filepath'], unique=False)
    op.create_index('idx_document_path_array', 'documents', ['path_array'], unique=False, postgresql_using='gin')
    op.create_index('ix_documents_tsv_content', 'documents', ['tsv_content'], unique=False, postgresql_using='gin')
    op.create_table('document_fragments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('fragment_content', sa.Text(), nullable=True),
    sa.Column('fragment_content_tsv', postgresql.TSVECTOR(), sa.Computed("setweight(to_tsvector('english', coalesce(meta->>'name', '')), 'A') || setweight(to_tsvector('english', coalesce(summary, '')), 'B') || setweight(to_tsvector('english', coalesce(fragment_content, '')), 'C')", persisted=True), nullable=True),
    sa.Column('summary', sa.Text(), nullable=True),
    sa.Column('vector', pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=True),
    sa.Column('meta', sa.JSON(), nullable=True),
    sa.Column('hierarchy_meta', sa.JSON(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_fragment_document', 'document_fragments', ['document_id'], unique=False)
    op.create_index('ix_document_fragments_fragment_content_tsv', 'document_fragments', ['fragment_content_tsv'], unique=False, postgresql_using='gin')
    op.create_index('ix_document_fragments_tsv_content', 'document_fragments', ['fragment_content_tsv'], unique=False, postgresql_using='gin')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_document_fragments_tsv_content', table_name='document_fragments', postgresql_using='gin')
    op.drop_index('ix_document_fragments_fragment_content_tsv', table_name='document_fragments', postgresql_using='gin')
    op.drop_index('idx_fragment_document', table_name='document_fragments')
    op.drop_table('document_fragments')
    op.drop_index('ix_documents_tsv_content', table_name='documents', postgresql_using='gin')
    op.drop_index('idx_document_path_array', table_name='documents', postgresql_using='gin')
    op.drop_index('idx_document_filepath', table_name='documents')
    op.drop_table('documents')
    # ### end Alembic commands ###