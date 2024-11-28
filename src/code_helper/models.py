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

ECHO_SQL_QUERIES = os.getenv("CODE_HELPER_ECHO_SQL_QUERIES", "False").lower() == "true"

DATABASE_URL = (
    "postgresql+asyncpg://code_helper:help-me-code@localhost:5432/code_helper"
)
engine = create_async_engine(DATABASE_URL, echo=ECHO_SQL_QUERIES)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
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
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("idx_document_path_array", path_array, postgresql_using="gin"),
        Index("idx_document_filepath", filepath),
        Index("ix_documents_tsv_content", "tsv_content", postgresql_using="gin"),
    )

    @classmethod
    async def get_directory_contents(cls, session, directory_path: str):
        """Get all files and subdirectories in a directory."""
        path_parts = directory_path.split(os.sep)
        return await session.execute(
            select(cls).where(
                and_(
                    func.array_length(cls.path_array, 1) == len(path_parts) + 1,
                    cls.path_array[: len(path_parts)].contains(path_parts),
                )
            )
        )

    @classmethod
    async def get_siblings(cls, session, filepath: str):
        """Get all files in the same directory."""
        doc = await session.execute(select(cls).where(cls.filepath == filepath))
        if doc is None:
            return []

        parent_path = doc.hierarchy_meta["parent"]
        if not parent_path:
            return []

        return await session.execute(
            select(cls).where(cls.hierarchy_meta["parent"] == parent_path)
        )

    @classmethod
    async def find_by_path(cls, session, filepath: str):
        """Find a document by its exact filepath."""
        return await session.scalar(select(cls).where(cls.filepath == filepath))

    @classmethod
    async def find_outdated(cls, session, max_age_days: int = 30):
        """Find documents that haven't been updated recently."""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        return await session.scalars(select(cls).where(cls.updated_at < cutoff_date))

    @classmethod
    async def search_by_text(cls, session, query: str, limit: int = 10):
        """Search documents using full-text search."""
        ts_query = func.plainto_tsquery("english", query)
        return await session.scalars(
            select(cls)
            .where(cls.tsv_content.op("@@")(ts_query))
            .order_by(func.ts_rank(cls.tsv_content, ts_query).desc())
            .limit(limit)
        )

    @classmethod
    async def find_similar_by_vector(cls, session, vector, limit: int = 5):
        """Find similar documents using vector similarity."""
        return await session.scalars(
            select(cls).order_by(cls.vector.cosine_distance(vector)).limit(limit)
        )

    @classmethod
    async def get_all_in_directory(
        cls, session, directory_path: str, recursive: bool = False
    ):
        """Get all documents in a directory, optionally recursive."""
        if recursive:
            return await session.scalars(
                select(cls).where(
                    func.array_to_string(cls.path_array, "/").like(
                        f"{directory_path}/%"
                    )
                )
            )
        else:
            path_parts = directory_path.split("/")
            return await session.scalars(
                select(cls).where(
                    and_(
                        func.array_length(cls.path_array, 1) == len(path_parts) + 1,
                        cls.path_array[: len(path_parts)] == path_parts,
                    )
                )
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
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(fragment_content, '')), 'C')",
            persisted=True
        ),
        nullable=True
    )
    summary = Column(Text, nullable=True)
    vector = Column(Vector(768), nullable=True)
    meta = Column(JSON, nullable=True)
    hierarchy_meta = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))

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

    @classmethod
    async def get_related_fragments(cls, session, fragment_id: int):
        """Get related fragments (siblings, parent, children) for a given fragment."""
        fragment = await session.execute(select(cls).where(cls.id == fragment_id))
        if fragment is None:
            return []

        return await session.execute(
            select(cls).where(
                and_(
                    cls.document_id == fragment.document_id,
                    cls.fragment_meta["parent"] == fragment.fragment_meta["parent"],
                )
            )
        )

    @classmethod
    async def find_by_document(cls, session, document_id: int):
        """Find all fragments for a given document."""
        return await session.scalars(
            select(cls)
            .where(cls.document_id == document_id)
            .order_by(cls.fragment_meta["index"])
        )

    @classmethod
    async def search_fragments(
        cls,
        session,
        query_vector: list[float],
        limit: int = 10,
        document_ids: list[int] = None,
    ) -> list[dict]:
        """Search for similar code fragments."""
        # Use the SQLAlchemy query directly instead of raw SQL
        query = (
            select(
                cls.id,
                cls.document_id,
                cls.fragment_content,
                cls.summary,
                cls.meta,
                cls.hierarchy_meta,
                Document.filepath,
                (1 - cls.vector.cosine_distance(query_vector)).label('similarity')
            )
            .join(Document, Document.id == cls.document_id)
            .where(
                (1 - cls.vector.cosine_distance(query_vector)) > 0.7
            )
        )

        if document_ids:
            query = query.where(cls.document_id.in_(document_ids))

        query = query.order_by('similarity DESC').limit(limit)

        results = await session.execute(query)
        return results.all()

    @classmethod
    async def find_similar_fragments(cls, session, vector, limit: int = 5):
        """Find similar fragments using vector similarity."""
        return await session.scalars(
            select(cls).order_by(cls.vector.cosine_distance(vector)).limit(limit)
        )

    @classmethod
    async def get_by_type(cls, session, document_id: int, fragment_type: str):
        """Get all fragments of a specific type (e.g., 'class', 'function')."""
        return await session.scalars(
            select(cls).where(
                and_(
                    cls.document_id == document_id,
                    cls.fragment_meta["type"].astext == fragment_type,
                )
            )
        )

    @classmethod
    async def get_children_of_fragment(cls, session, fragment_id: int):
        """Get all child fragments of a given fragment (e.g., methods of a class)."""
        parent_fragment = await session.get(cls, fragment_id)
        if not parent_fragment:
            return []

        return await session.scalars(
            select(cls).where(
                and_(
                    cls.document_id == parent_fragment.document_id,
                    cls.fragment_meta["parent"].astext
                    == parent_fragment.fragment_meta["name"],
                )
            )
        )

    @classmethod
    async def get_siblings(cls, session, fragment_id: int):
        """Get siblings of a given fragment."""
        fragment = await session.execute(
            select(cls).where(cls.id == fragment_id)
        )
        if fragment is None:
            return []

        # Get all fragments from the same document with the same parent
        return await session.execute(
            select(cls)
            .where(
                and_(
                    cls.document_id == fragment.document_id,
                    cls.fragment_meta["parent"]
                    == fragment.fragment_meta["parent"],
                )
            )
        )



Document.fragments = relationship(
    "DocumentFragment", order_by=DocumentFragment.id, back_populates="document"
)


@asynccontextmanager
async def get_session():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            if session.sync_session._connection_for_bind:
                await session.sync_session._connection_for_bind.close()


async def async_drop_db(db_engine=engine):
    async with db_engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.drop_all)
        finally:
            await conn.close()


async def init_db(db_engine=engine):
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
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


async def keyword_search_documents(session: AsyncSession, query: str, limit: int = 10):
    """Search documents using full-text search with keyword ranking.

    Args:
        session: Database session
        query: Search query string
        limit: Maximum number of results to return
    """
    # Convert space-separated keywords into OR query
    query_or = func.to_tsquery('|'.join(query.split()))

    # Build query using SQLAlchemy expressions
    query = (
        select(Document)
        .where(Document.tsv_content.op('@@')(query_or))
        .order_by(func.ts_rank(Document.tsv_content, query_or).desc())
        .limit(limit)
    )

    results = await session.execute(query)
    return results.scalars().all()


async def keyword_search_document_fragments(
    session: AsyncSession,
    query_text: str,
    document_ids: List[int],
    filenames: Optional[List[str]] = None,
    limit: int = 20,
):
    """Search document fragments using keyword search."""
    # Convert space-separated keywords into OR query
    query_or = func.to_tsquery('english', '|'.join(query_text.split()))

    # Add check for empty document_ids
    if not document_ids:
        return []

    # Build base query
    query = (
        select(
            DocumentFragment.id,
            Document.id.label('document_id'),
            Document.filename,
            Document.filepath,
            Document.meta.label('document_meta'),
            DocumentFragment.fragment_content,
            DocumentFragment.summary,
            DocumentFragment.meta,
            func.ts_rank(DocumentFragment.fragment_content_tsv, query_or).label('rank')
        )
        .join(Document, DocumentFragment.document_id == Document.id)
        .where(
            and_(
                DocumentFragment.document_id.in_(document_ids),
                DocumentFragment.fragment_content_tsv.op('@@')(query_or)
            )
        )
    )

    # Add filename filter if provided
    if filenames:
        query = query.where(Document.filename.in_(filenames))

    # Add ordering and limit
    query = query.order_by(text('rank DESC')).limit(limit)

    # Execute query
    results = await session.execute(query)
    return results.all()


async def vector_search_documents(session: AsyncSession, query_vector, limit=10):
    """Search documents using vector similarity."""
    query = (
        select(
            Document,  # Select the full Document object first
            (1 - Document.vector.cosine_distance(query_vector)).label('similarity')
        )
        .order_by(text('similarity DESC'))  # Fix: Wrap in text()
        .limit(limit)
    )

    results = await session.execute(query)
    return [result[0] for result in results]


async def vector_search_document_fragments(
    session: AsyncSession,
    query_vector,
    document_ids=None,
    filenames=None,
    limit=20
):
    """Search for document fragments using vector similarity.

    Args:
        session: Database session
        query_vector: Vector to compare against
        document_ids: Optional list of document IDs to filter by
        filenames: Optional list of filenames to filter by
        limit: Maximum number of results to return
    """
    # Start with base query
    query = (
        select(
            DocumentFragment.id,
            Document.id.label('document_id'),
            Document.filename,
            Document.filepath,
            Document.meta.label('document_meta'),
            DocumentFragment.fragment_content,
            DocumentFragment.summary,
            DocumentFragment.meta,
            (DocumentFragment.vector.cosine_distance(query_vector)).label('score')
        )
        .join(Document, DocumentFragment.document_id == Document.id)
    )

    # Add filters if provided
    if document_ids:
        query = query.where(DocumentFragment.document_id.in_(document_ids))
    if filenames:
        query = query.where(Document.filename.in_(filenames))

    # Add ordering and limit
    query = query.order_by('score').limit(limit)

    # Execute query
    results = await session.execute(query)
    return results.all()


def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    k=60,
) -> list[tuple[Any, float]]:
    rank_dict = defaultdict(float)

    # Assign ranks to vector search results
    for rank, result in enumerate(vector_results, start=1):
        # Handle both Document objects and Row results
        result_id = result.id if hasattr(result, 'id') else result[0].id
        rank_dict[result_id] += 1 / (rank + k)

    # Assign ranks to keyword search results
    for rank, result in enumerate(keyword_results, start=1):
        result_id = result.id if hasattr(result, 'id') else result[0].id
        rank_dict[result_id] += 1 / (rank + k)

    # Sort results by their cumulative rank
    sorted_results = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
    return sorted_results


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_vector: list[float],
    filenames: Optional[List[str]] = None,
    limit: int = 20,
):
    """Perform hybrid search across documents and fragments.

    Combines keyword and vector search results using reciprocal rank fusion.
    First searches documents, then uses those document IDs to constrain fragment search.
    """
    # Get document-level matches and fuse results
    vector_documents = await vector_search_documents(session, query_vector, limit=limit * 2)
    keyword_documents = await keyword_search_documents(session, query_text, limit=limit * 2)

    document_results = reciprocal_rank_fusion(vector_documents, keyword_documents)
    relevant_doc_ids = [doc_id for doc_id, _ in document_results[:limit]]

    # Search fragments within relevant documents
    vector_fragments = await vector_search_document_fragments(
        session,
        query_vector,
        document_ids=relevant_doc_ids,
        filenames=filenames,
        limit=limit * 2
    )

    keyword_fragments = await keyword_search_document_fragments(
        session,
        query_text,
        document_ids=relevant_doc_ids,
        filenames=filenames,
        limit=limit * 2
    )

    # Fuse fragment results and format response
    fragment_results = []
    seen_fragments = set()

    for frag_id, score in reciprocal_rank_fusion(vector_fragments, keyword_fragments):
        if frag_id in seen_fragments:
            continue

        fragment = next(f for f in vector_fragments + keyword_fragments if f.id == frag_id)
        seen_fragments.add(frag_id)

        fragment_results.append({
            "document_id": str(fragment.document_id),
            "score": score,
            "filename": fragment.filename,
            "filepath": fragment.filepath,
            "fragment_content": fragment.fragment_content,
            "metadata": fragment.meta,
            "imports": fragment.document_meta.get("imports", []) if fragment.document_meta else []
        })

        if len(fragment_results) >= limit:
            break

    return fragment_results


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


