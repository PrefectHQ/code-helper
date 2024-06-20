from collections import defaultdict
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

DATABASE_URL = "postgresql://username:password@localhost:5432/code_helper"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, unique=True, nullable=False)
    file_content = Column(Text, nullable=False)
    tsv_content = Column(TSVECTOR, nullable=True)  # Combined TSVECTOR column
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


def get_session():
    return SessionLocal()


def keyword_search_documents(session: Session, query: str, limit: int = 10):
    query_or = "|".join(query.split())
    search_query = text(
        """
        SELECT * FROM documents
        WHERE tsv_content @@ to_tsquery(:query)
        ORDER BY ts_rank(tsv_content, to_tsquery(:query)) DESC
        LIMIT :limit
        """
    )
    results = session.execute(
        search_query, {"query": query_or, "limit": limit}
    ).fetchall()
    return results


def keyword_search_document_fragments(
    session: Session,
    query_text: str,
    document_ids,
    filenames: Optional[List[str]] = None,
    limit: int = 20,
):
    query_or = "|".join(query_text.split())
    opts = {"query": query_or, "limit": limit, "document_ids": document_ids}
    if filenames:
        opts["filenames"] = filenames

    query = (
        """
        SELECT document_fragments.id, documents.id as document_id, document_fragments.fragment_content, document_fragments.summary, documents.filename, documents.filepath, document_fragments.meta
        FROM document_fragments
        LEFT JOIN documents ON documents.id = document_fragments.document_id
        WHERE document_fragments.document_id = ANY(:document_ids)          
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

    results = session.execute(text(query), opts).fetchall()
    return results


def vector_search_documents(session, query_vector, limit=10):
    query_fragments = text(
        """
        SELECT *, documents.vector <-> :query_vector as score
        FROM documents
        ORDER BY score
        LIMIT :limit
        """
    )

    return session.execute(
        query_fragments,
        {
            "query_vector": str(query_vector),
            "limit": limit,
        },
    ).fetchall()


def vector_search_document_fragments(
    session, query_vector, document_ids=None, filenames=None, limit=20
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
        SELECT document_fragments.id, documents.id as document_id, document_fragments.fragment_content, document_fragments.summary, documents.filename, documents.filepath, document_fragments.meta, document_fragments.vector <-> :query_vector as score
        FROM document_fragments
        LEFT JOIN documents ON documents.id = document_fragments.document_id
        WHERE document_fragments.document_id = ANY(:document_ids)
    """
        + ("AND filename = ANY(:filenames)" if filenames else "")
        + """
        ORDER BY score
        LIMIT :limit
    """
    )
    query_fragments = text(query)

    return session.execute(
        query_fragments,
        opts,
    ).fetchall()


def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
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


def hybrid_search(
    session: Session,
    query_text: str,
    query_vector: list[float],
    filenames: Optional[List[str]] = None,
):
    documents = keyword_search_documents(session, query_text)
    document_ids = [d.id for d in documents]

    vector_fragment_results = vector_search_document_fragments(
        session, query_vector, document_ids, filenames=filenames
    )
    keyword_fragment_results = keyword_search_document_fragments(
        session, query_text, document_ids, filenames=filenames
    )
    combined_results = reciprocal_rank_fusion(
        vector_fragment_results, keyword_fragment_results
    )

    import ipdb

    ipdb.set_trace()

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
            response.append(
                {
                    "document_id": str(fragment.document_id),
                    "score": score,
                    "filename": fragment.filename,
                    "filepath": fragment.filepath,
                    "fragment_content": fragment.fragment_content,
                    "metadata": fragment.meta,
                }
            )

    return response
