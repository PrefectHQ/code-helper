from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Any
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Index,
    Computed,
    text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

DATABASE_URL = "postgresql+asyncpg://username:password@localhost:5432/code_helper"
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, unique=True, nullable=False)
    file_content = Column(Text, nullable=False)
    tsv_content = Column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(filename, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(file_content, '')), 'C')"
        ),
        nullable=True,
    )
    vector = Column(Vector(768), nullable=False)
    summary = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_documents_tsv_content", "tsv_content", postgresql_using="gin"),
    )


class DocumentFragment(Base):
    __tablename__ = "document_fragments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    fragment_content = Column(Text, nullable=False)
    fragment_content_tsv = Column(TSVECTOR, nullable=False)
    summary = Column(Text, nullable=True)
    summary_tsv = Column(TSVECTOR, nullable=True)
    vector = Column(Vector(768), nullable=False)
    meta = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="fragments")

    __table_args__ = (
        Index(
            "ix_document_fragments_fragment_content_tsv",
            "fragment_content_tsv",
            postgresql_using="gin",
        ),
        Index(
            "ix_document_fragments_summary_tsv", "summary_tsv", postgresql_using="gin"
        ),
    )


Document.fragments = relationship(
    "DocumentFragment", order_by=DocumentFragment.id, back_populates="document"
)


@asynccontextmanager
async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


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
    vector_documents = await vector_search_documents(session, query_vector)
    documents = reciprocal_rank_fusion(vector_documents, keyword_documents)
    document_ids = [d[0] for d in documents]

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
