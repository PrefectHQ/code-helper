import pytest
from datetime import datetime, UTC
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from code_helper.models import (
    Document,
    DocumentFragment,
    hybrid_search,
    get_session,
)


@pytest.mark.asyncio
async def test_document_creation(db_session):
    doc = Document(
        filename="test.py",
        filepath="/path/to/test.py",
        path_array=["path", "to"],
        summary="Test document",
        vector=[0.1] * 768,
        meta={"type": "python"},
        hierarchy_meta={"parent": None, "children": []},
        updated_at=datetime.now(UTC)
    )
    db_session.add(doc)
    await db_session.commit()

    result = await db_session.execute(select(Document).where(Document.filename == "test.py"))
    saved_doc = result.scalar_one()

    assert saved_doc.filename == "test.py"
    assert saved_doc.filepath == "/path/to/test.py"
    assert saved_doc.path_array == ["path", "to"]
    assert saved_doc.summary == "Test document"
    assert len(saved_doc.vector) == 768
    assert saved_doc.meta == {"type": "python"}
    assert saved_doc.hierarchy_meta == {"parent": None, "children": []}


@pytest.mark.asyncio
async def test_document_fragment_relationship(db_session):
    # Create a document with fragments
    doc = Document(
        filename="test.py",
        filepath="/path/to/test.py",
        path_array=["path", "to"],
        vector=[0.1] * 768,
    )
    db_session.add(doc)
    await db_session.flush()

    fragment = DocumentFragment(
        document_id=doc.id,
        fragment_content="def test_function():\n    pass",
        summary="A test function",
        vector=[0.2] * 768,
        meta={"type": "function", "name": "test_function"},
    )
    db_session.add(fragment)
    await db_session.commit()

    # Query the relationship
    result = await db_session.execute(
        select(Document)
        .options(selectinload(Document.fragments))
        .where(Document.id == doc.id)
    )
    saved_doc = result.scalar_one()
    
    assert len(saved_doc.fragments) == 1
    assert saved_doc.fragments[0].fragment_content == "def test_function():\n    pass"
    assert saved_doc.fragments[0].meta["name"] == "test_function"


@pytest.mark.asyncio
async def test_document_directory_contents(db_session):
    # Create documents in a directory structure
    docs = [
        Document(
            filename=f"test{i}.py",
            filepath=f"/path/to/dir/test{i}.py",
            path_array=["path", "to", "dir", f"test{i}.py"],
            vector=[0.1] * 768,
        )
        for i in range(3)
    ]
    for doc in docs:
        db_session.add(doc)
    await db_session.commit()

    # Test get_directory_contents
    result = await Document.get_directory_contents(db_session, "/path/to/dir")
    files = list(result)

    assert len(files) == 3
    for doc in files:
        assert doc.path_array == ["path", "to", "dir", doc.filename]


@pytest.mark.asyncio
async def test_document_search_by_text(db_session):
    # Create test documents with searchable content
    doc1 = Document(
        filename="test1.py",
        filepath="/path/to/test1.py",
        path_array=["path", "to"],
        summary="A test document about Python functions",
        vector=[0.1] * 768,
    )
    doc2 = Document(
        filename="test2.py",
        filepath="/path/to/test2.py",
        path_array=["path", "to"],
        summary="A document about database models",
        vector=[0.1] * 768,
    )
    db_session.add_all([doc1, doc2])
    await db_session.commit()

    # Test text search
    results = await Document.search_by_text(db_session, "Python functions")
    docs = list(results)
    
    assert len(docs) == 1
    assert docs[0].filename == "test1.py"


@pytest.mark.asyncio
async def test_hybrid_search(db_session):
    # Create test documents and fragments
    doc = Document(
        filename="test.py",
        filepath="/path/to/test.py",
        path_array=["path", "to"],
        summary="Test document",
        vector=[0.1] * 768,
    )
    db_session.add(doc)
    await db_session.flush()

    fragments = [
        DocumentFragment(
            document_id=doc.id,
            fragment_content=f"def test_function_{i}():\n    pass",
            summary=f"Test function {i}",
            vector=[0.2] * 768,
            meta={"type": "function", "name": f"test_function_{i}"},
        )
        for i in range(2)
    ]
    db_session.add_all(fragments)
    await db_session.commit()

    # Test hybrid search
    results = await hybrid_search(
        db_session,
        query_text="test function",
        query_vector=[0.2] * 768,
        filenames=None,
        limit=10
    )

    assert len(results) > 0
    assert all("test_function" in r["fragment_content"] for r in results)


@pytest.mark.asyncio
async def test_document_fragment_search(db_session):
    # Create a document with searchable fragments
    doc = Document(
        filename="test.py",
        filepath="/path/to/test.py",
        path_array=["path", "to"],
        vector=[0.1] * 768,
    )
    db_session.add(doc)
    await db_session.flush()

    fragment = DocumentFragment(
        document_id=doc.id,
        fragment_content="def search_function():\n    return 'found'",
        summary="A searchable function",
        vector=[0.2] * 768,
        meta={"type": "function", "name": "search_function"},
    )
    db_session.add(fragment)
    await db_session.commit()

    # Test fragment search
    results = await DocumentFragment.search_fragments(
        db_session,
        [0.2] * 768,
        limit=5,
        document_ids=[doc.id]
    )

    assert len(results) > 0
    assert results[0].fragment_content == "def search_function():\n    return 'found'"


@pytest.mark.asyncio
async def test_get_session_context_manager():
    async with get_session() as db_session:
        # Test that we can execute a simple query
        result = await db_session.execute(select(Document))
        assert result is not None


@pytest.mark.asyncio
async def test_document_find_by_path(db_session):
    # Create a test document
    doc = Document(
        filename="test.py",
        filepath="/path/to/test.py",
        path_array=["path", "to"],
        vector=[0.1] * 768,
    )
    db_session.add(doc)
    await db_session.commit()

    # Test find_by_path
    found_doc = await Document.find_by_path(db_session, "/path/to/test.py")
    assert found_doc is not None
    assert found_doc.filename == "test.py"
