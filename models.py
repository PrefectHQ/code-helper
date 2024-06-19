from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector

DATABASE_URL = "postgresql://username:password@localhost:5432/code_helper"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, unique=True, nullable=False)
    file_content = Column(Text, nullable=False)
    vector = Column(Vector(768), nullable=False)  # Use pgvector for embedding
    summary = Column(Text, nullable=True)  # Field for storing the summary
    meta = Column(JSON, nullable=True)  # Field for storing the metadata
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentFragment(Base):
    __tablename__ = 'document_fragments'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    fragment_content = Column(Text, nullable=False)
    vector = Column(Vector(768), nullable=False)  # Use pgvector for embedding
    summary = Column(Text, nullable=True)  # Field for storing the summary
    meta = Column(JSON, nullable=True)  # Field for storing the metadata
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="fragments")


Document.fragments = relationship("DocumentFragment", order_by=DocumentFragment.id, back_populates="document")


def get_session():
    return SessionLocal()
