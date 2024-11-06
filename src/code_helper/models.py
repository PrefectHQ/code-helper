import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, List, Optional

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
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
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
    fragment_content_tsv = Column(TSVECTOR, nullable=True)
    summary = Column(Text, nullable=True)
    summary_tsv = Column(TSVECTOR, nullable=True)
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
        Index(
            "ix_document_fragments_summary_tsv", "summary_tsv", postgresql_using="gin"
        ),
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
    async def search_fragments(cls, session, query: str, limit: int = 10):
        """Search fragments using full-text search."""
        ts_query = func.plainto_tsquery("english", query)
        return await session.scalars(
            select(cls)
            .where(
                or_(
                    cls.fragment_content_tsv.op("@@")(ts_query),
                    cls.summary_tsv.op("@@")(ts_query),
                )
            )
            .order_by(
                func.ts_rank(
                    cls.fragment_content_tsv + cls.summary_tsv, ts_query
                ).desc()
            )
            .limit(limit)
        )

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
async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


async def async_drop_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def update_document_fragment_tsvector(session: AsyncSession, fragment_id: int):
    """
    Update TSVECTOR columns for a given document fragment
    """
    await session.execute(
        text(
            """
            UPDATE document_fragments
            SET fragment_content_tsv = to_tsvector('english', fragment_content)
            WHERE id = :id
            """
        ),
        {"id": fragment_id},
    )
    await session.execute(
        text(
            """
            UPDATE document_fragments
            SET summary_tsv = to_tsvector('english', summary)
            WHERE id = :id
            """
        ),
        {"id": fragment_id},
    )
    await session.commit()


async def create_document_fragment(session: AsyncSession, **fragment_data):
    """
    Create a document fragment and update TSVECTOR columns
    """
    fragment = DocumentFragment(**fragment_data)
    session.add(fragment)
    await session.flush()
    await update_document_fragment_tsvector(session, fragment.id)
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
    query_or = "|".join(query.split())
    results = await session.execute(
        text(
            """
            SELECT * FROM documents
            WHERE tsv_content @@ to_tsquery(:query)
            ORDER BY ts_rank(tsv_content, to_tsquery(:query)) DESC
            LIMIT :limit
            """
        ),
        {"query": query_or, "limit": limit},
    )
    return results.fetchall()


async def keyword_search_document_fragments(
    session: AsyncSession,
    query_text: str,
    document_ids: List[int],
    filenames: Optional[List[str]] = None,
    limit: int = 20,
):
    query_or = "|".join(query_text.split())
    opts = {"query": query_or, "limit": limit, "document_ids": document_ids}
    if filenames:
        opts["filenames"] = filenames

    query = (
        """
        SELECT document_fragments.id,
               documents.id as document_id,
               documents.filename,
               documents.filepath,
               documents.meta as document_meta,
               document_fragments.fragment_content,
               document_fragments.summary,
               document_fragments.meta
        FROM document_fragments
                 LEFT JOIN documents ON documents.id = document_fragments.document_id
        WHERE document_fragments.document_id = ANY (:document_ids)
          AND (
            fragment_content_tsv @@ to_tsquery(:query)
                OR document_fragments.summary_tsv @@ to_tsquery(:query)
            )
            """
        + ("AND filename = ANY(:filenames)" if filenames else "")
        + """
            ORDER BY ts_rank(fragment_content_tsv, to_tsquery(:query)) DESC
            LIMIT :limit
        """
    )

    results = await session.execute(text(query), opts)
    return results.fetchall()


async def vector_search_documents(session: AsyncSession, query_vector, limit=10):
    query_fragments = text(
        """
        SELECT *, documents.vector <-> :query_vector as score
        FROM documents
        ORDER BY score
        LIMIT :limit
        """
    )

    results = await session.execute(
        query_fragments,
        {
            "query_vector": str(query_vector),
            "limit": limit,
        },
    )
    return results.fetchall()


async def vector_search_document_fragments(
    session: AsyncSession, query_vector, document_ids=None, filenames=None, limit=20
):
    opts = {
        "query_vector": str(query_vector),
        "limit": limit,
        "document_ids": document_ids,
    }
    if filenames:
        opts["filenames"] = filenames

    query = (
        """
        SELECT document_fragments.id,
               documents.id as document_id,
               documents.filename,
               documents.filepath,
               documents.meta as document_meta,
               document_fragments.fragment_content,
               document_fragments.summary,
               document_fragments.meta,
               document_fragments.vector <-> :query_vector as score
        FROM document_fragments
                 LEFT JOIN documents ON documents.id = document_fragments.document_id
        WHERE document_fragments.document_id = ANY (:document_ids)
    """
        + ("AND filename = ANY(:filenames)" if filenames else "")
        + """
        ORDER BY score
        LIMIT :limit
    """
    )
    query_fragments = text(query)

    results = await session.execute(query_fragments, opts)
    return results.fetchall()


def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    k=60,
) -> list[tuple[Any, float]]:
    rank_dict = defaultdict(float)

    # Assign ranks to vector search results
    for rank, result in enumerate(vector_results, start=1):
        rank_dict[result.id] += 1 / (rank + k)

    # Assign ranks to keyword search results
    for rank, result in enumerate(keyword_results, start=1):
        rank_dict[result.id] += 1 / (rank + k)

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
    keyword_documents = await keyword_search_documents(session, query_text)

    # TODO: RRF on documents yielded poorer results than just keyword search.
    # vector_documents = await vector_search_documents(session, query_vector)
    # documents = reciprocal_rank_fusion(vector_documents, keyword_documents)

    document_ids = [d[0] for d in keyword_documents]

    vector_fragment_results = await vector_search_document_fragments(
        session, query_vector, document_ids, filenames=filenames, limit=limit
    )
    keyword_fragment_results = await keyword_search_document_fragments(
        session, query_text, document_ids, filenames=filenames, limit=limit
    )
    combined_results = reciprocal_rank_fusion(
        vector_fragment_results, keyword_fragment_results
    )

    fragments_by_id = {
        frag.id: frag for frag in vector_fragment_results + keyword_fragment_results
    }

    # Build the response based on the combined results sorted by their RRF score
    response = []
    seen_fragments = set()  # Track seen fragment IDs to avoid duplicates

    for frag_id, score in combined_results:
        if frag_id not in seen_fragments:
            seen_fragments.add(frag_id)
            fragment = fragments_by_id[frag_id]
            imports = fragment.document_meta.get("imports", [])

            response.append(
                {
                    "document_id": str(fragment.document_id),
                    "score": score,
                    "filename": fragment.filename,
                    "filepath": fragment.filepath,
                    "fragment_content": fragment.fragment_content,
                    "metadata": fragment.meta,
                    "imports": imports,
                }
            )

    return response[:limit]
