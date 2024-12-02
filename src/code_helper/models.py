import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, List, Optional, Dict

import aiofiles
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    and_,
    delete,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import NullPool

ECHO_SQL_QUERIES = os.getenv("CODE_HELPER_ECHO_SQL_QUERIES", "False").lower() == "true"

engine = None
SessionLocal = None
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, unique=True, nullable=False)
    path_array = Column(ARRAY(String), nullable=False)
    summary = Column(Text, nullable=True)
    tsv_content = Column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(filename, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B')",
            persisted=True,
        ),
        nullable=True,
    )
    vector = Column(Vector(768), nullable=False)
    meta = Column(JSON, nullable=True)
    hierarchy_meta = Column(JSON, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("idx_document_path_array", path_array, postgresql_using="gin"),
        Index("idx_document_filepath", filepath),
        Index("ix_documents_tsv_content", "tsv_content", postgresql_using="gin"),
    )


class DocumentFragment(Base):
    __tablename__ = "document_fragments"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    fragment_content = Column(Text, nullable=True)
    fragment_content_tsv = Column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(meta->>'name', '')), 'A') || "
            "setweight(to_tsvector('english', regexp_replace(coalesce(summary, ''), '[-_]', ' ', 'g')), 'B') || "
            "setweight(to_tsvector('english', regexp_replace(coalesce(fragment_content, ''), '[-_]', ' ', 'g')), 'C')",
            persisted=True
        ),
        nullable=True
    )
    summary = Column(Text, nullable=True)
    vector = Column(Vector(768), nullable=True)
    meta = Column(JSON, nullable=True)
    hierarchy_meta = Column(JSON, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    document = relationship("Document", back_populates="fragments")

    __table_args__ = (
        Index("idx_fragment_document", document_id),
        Index(
            "ix_document_fragments_fragment_content_tsv",
            "fragment_content_tsv",
            postgresql_using="gin",
        ),
        Index('ix_document_fragments_tsv_content', 'fragment_content_tsv', postgresql_using='gin'),
    )


Document.fragments = relationship(
    "DocumentFragment", order_by=DocumentFragment.id, back_populates="document"
)


@asynccontextmanager
async def get_session():
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def async_drop_db(db_engine=engine):
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def init_db(engine):
    async with engine.begin() as conn:
        # Execute each SQL command separately
        await conn.execute(text("DROP TEXT SEARCH CONFIGURATION IF EXISTS code_search CASCADE"))
        await conn.execute(text("CREATE TEXT SEARCH CONFIGURATION code_search (COPY = english)"))
        await conn.execute(
            text("ALTER TEXT SEARCH CONFIGURATION code_search ALTER MAPPING FOR word, asciiword WITH simple, english_stem")
        )
        
        # Then create tables
        await conn.run_sync(Base.metadata.create_all)


async def create_document_fragment(
    session: AsyncSession,
    document_id: int,
    fragment_content: str,
    summary: str | None = None,
    vector: list[float] | None = None,
    meta: dict | None = None,
    hierarchy_meta: dict | None = None,
    updated_at: datetime | None = None,
) -> DocumentFragment:
    """
    Create a document fragment with all necessary fields.
    """
    fragment = DocumentFragment(
        document_id=document_id,
        fragment_content=fragment_content,
        summary=summary,
        vector=vector,
        meta=meta,
        hierarchy_meta=hierarchy_meta,
        updated_at=updated_at or datetime.now(UTC),
    )
    session.add(fragment)
    await session.flush()
    return fragment


async def create_document(session: AsyncSession, **document_data):
    """
    Create a document.
    """
    document = Document(**document_data)
    session.add(document)
    await session.flush()
    return document


async def keyword_search_documents(
    session: AsyncSession,
    query_text: str,
    limit: int = 20,
) -> list[tuple[Document, float]]:
    """Search documents using keyword matching."""
    # Convert terms to OR'd tsquery
    query_terms = ' | '.join(
        term.replace("'", "")
        for term in query_text.replace("-", " ").split()
    )
    tsquery = func.to_tsquery('english', query_terms)
    
    query = (
        select(
            Document,
            func.ts_rank(Document.tsv_content, tsquery).label('rank')
        )
        .where(Document.tsv_content.op('@@')(tsquery))
        .order_by(text('rank DESC'))
        .limit(limit)
    )

    result = await session.execute(query)
    return [(row.Document, row.rank) for row in result]


async def keyword_search_document_fragments(
    session: AsyncSession,
    query_text: str,
    document_ids: Optional[List[int]] = None,
    limit: int = 20,
) -> list[tuple[DocumentFragment, float]]:
    """Search document fragments using keyword matching."""
    # Convert terms to OR'd tsquery
    query_terms = ' | '.join(
        term.replace("'", "")
        for term in query_text.replace("-", " ").split()
    )
    tsquery = func.to_tsquery('english', query_terms)
    
    query = (
        select(
            DocumentFragment,
            func.ts_rank_cd(
                DocumentFragment.fragment_content_tsv,
                tsquery
            ).label('rank')
        )
        .where(DocumentFragment.fragment_content_tsv.op('@@')(tsquery))
    )

    if document_ids:
        query = query.where(DocumentFragment.document_id.in_(document_ids))

    query = query.order_by(text('rank DESC')).limit(limit)
    
    result = await session.execute(query)
    return [(row.DocumentFragment, float(row.rank)) for row in result]


async def vector_search_documents(
    session: AsyncSession,
    query_vector: list[float],
    limit: int = 20,
) -> list[tuple[Document, float]]:
    # Add similarity score to the query results
    query = (
        select(
            Document,
            (1 - Document.vector.cosine_distance(query_vector)).label('similarity')
        )
        .order_by(text('similarity DESC'))
        .limit(limit)
    )
    
    result = await session.execute(query)
    # Return both the document and its similarity score
    return [(row.Document, row.similarity) for row in result]


async def vector_search_document_fragments(
    session: AsyncSession,
    query_vector: list[float],
    document_ids: Optional[List[int]] = None,
    limit: int = 20,
) -> list[tuple[DocumentFragment, float]]:
    """Search document fragments using vector similarity."""
    query = (
        select(
            DocumentFragment,
            (1 - DocumentFragment.vector.cosine_distance(query_vector)).label('score')
        )
        .join(Document, DocumentFragment.document_id == Document.id)
    )

    if document_ids:
        query = query.where(DocumentFragment.document_id.in_(document_ids))

    query = query.order_by(text('score DESC')).limit(limit)
    
    result = await session.execute(query)
    return [(row.DocumentFragment, row.score) for row in result]


def reciprocal_rank_fusion(
    vector_results: list[tuple[Any, float]],
    keyword_results: list[tuple[Any, float]],
    k: int = 20,
) -> list[tuple[Any, float]]:
    """
    Combine vector and keyword search results using reciprocal rank fusion.

    Args:
        vector_results: List of tuples containing a search result and its similarity score.
        keyword_results: List of tuples containing a search result and its keyword rank.
        k: The constant used in the reciprocal rank fusion formula.

    Returns:
        List of tuples containing a search result and its combined rank.
    """
    # Create a map of item to its ranks in each result list
    rank_dict = defaultdict(float)
    
    # Process vector search results
    for rank, (item, _) in enumerate(vector_results):
        rank_dict[item] += 1.0 / (rank + k)
        
    # Process keyword search results    
    for rank, (item, _) in enumerate(keyword_results):
        rank_dict[item] += 1.0 / (rank + k)
        
    # Sort results by their cumulative rank
    sorted_results = sorted(
        [(item, score) for item, score in rank_dict.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_results


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_vector: list[float],
    limit: int = 20,
) -> list[dict]:
    """Perform hybrid search across documents and fragments."""
    # Get document-level matches and fuse results
    vector_documents = await vector_search_documents(session, query_vector, limit=limit * 2)
    keyword_documents = await keyword_search_documents(session, query_text, limit=limit * 2)

    print("\nDocument scores:")
    for doc, similarity in vector_documents:
        print(f"Vector - {doc.filename}: {similarity}")
    for doc, rank in keyword_documents:
        print(f"Keyword - {doc.filename}: {rank}")

    # Fuse document results
    document_results = reciprocal_rank_fusion(
        vector_documents,
        keyword_documents,
        k=20
    )

    # Get document IDs to constrain fragment search
    doc_ids = [doc.id for doc, _ in document_results[:limit]]

    # Get fragment-level matches from selected documents
    vector_fragments = await vector_search_document_fragments(
        session, query_vector, doc_ids, limit=limit * 2
    )
    keyword_fragments = await keyword_search_document_fragments(
        session, query_text, doc_ids, limit=limit * 2
    )

    # Debug fragment scores
    print("\nFragment scores:")
    for fragment, score in vector_fragments:
        print(f"Vector - {fragment.meta['name']}: {score}")
    for fragment, rank in keyword_fragments:
        print(f"Keyword - {fragment.meta['name']}: {rank}")

    # Fuse fragment results
    fragment_results = reciprocal_rank_fusion(
        vector_fragments,
        keyword_fragments,
        k=20
    )

    # Update the results format
    results = []
    for fragment, score in fragment_results[:limit]:
        # Get the parent document to access filename and filepath
        document = await session.get(Document, fragment.document_id)
        
        results.append({
            "id": fragment.id,
            "content": fragment.fragment_content,
            "summary": fragment.summary,
            "metadata": fragment.meta,
            "score": score,
            "document_id": str(fragment.document_id),  # Convert to string
            "filename": document.filename,
            "filepath": document.filepath,
            "fragment_content": fragment.fragment_content,
            "imports": document.meta.get("imports", [])
        })

    return results


async def read_and_validate_file(
    filepath: str, session, replace_existing: bool, reindex_updated: bool
) -> tuple[str, datetime] | None:
    """Handle file reading and validation of processing conditions."""
    try:
        async with aiofiles.open(filepath, "r") as f:
            file_content = await f.read()
    except UnicodeDecodeError:
        print(f"Error reading file: {filepath}")
        return None

    if not file_content:
        print(f"File is empty: {filepath}")
        return None

    # Get existing document if any
    existing_doc = await session.scalar(
        select(Document.updated_at).where(Document.filepath == filepath)
    )

    if existing_doc is not None:
        if not replace_existing:
            print(f"File already exists: {filepath}")
            return None
        # Delete existing document and its fragments
        print(f"Deleting existing document: {filepath}")
        await session.execute(
            delete(DocumentFragment).where(
                DocumentFragment.document_id.in_(
                    select(Document.id).where(Document.filepath == filepath)
                )
            )
        )
        await session.execute(delete(Document).where(Document.filepath == filepath))
        await session.commit()

        if reindex_updated:
            updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
            if updated_at <= existing_doc:
                return None
            print(f"File is updated: {filepath}. Reindexing.")

    updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_content, updated_at


def init_db_connection(database_url=None):
    """Initialize database connection."""
    global engine, SessionLocal
    
    if database_url is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://code_helper:help-me-code@localhost:5432/code_helper"
        )
    
    engine = create_async_engine(
        database_url,
        echo=os.getenv("CODE_HELPER_ECHO_SQL_QUERIES", "false").lower() == "true",
        poolclass=NullPool
    )
    
    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession
    )
    
    return engine, SessionLocal
