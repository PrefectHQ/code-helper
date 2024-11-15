import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from numpy import array
import pytest
from sqlalchemy import select

from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import pool

import pytest_asyncio

from code_helper.index import (
    process_document_content,
    process_file,
    process_fragments,
    read_and_validate_file,
)

# Mock for marvin.fn decorator
def mock_marvin_fn(f):
    return f


# Common fixtures
@pytest.fixture
def mock_session():
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
def sample_python_file(tmp_path):
    content = """
def hello():
    print("Hello, World!")

class TestClass:
    def method(self):
        return "test"
"""
    file_path = tmp_path / "test.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    file_path.write_text(content)
    return str(file_path)


# Helper class for async mock
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Stage 1: File Reading and Validation Tests
@pytest.mark.asyncio
async def test_read_and_validate_file_new_file(mock_session, sample_python_file):
    # Create an AsyncMock for execute that returns None for scalar
    mock_result = AsyncMock()
    mock_result.scalar.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.commit = AsyncMock()

    # Ensure the file exists and is readable
    with open(sample_python_file, 'r') as f:
        expected_content = f.read()

    # Mock the file modification time
    mock_mtime = datetime.now().timestamp()
    with patch('os.path.getmtime', return_value=mock_mtime):
        # Await the execute call
        content, updated_at = await read_and_validate_file(
            sample_python_file, mock_session, replace_existing=False, reindex_updated=False
        )

        # Verify the mock was called correctly
        mock_session.execute.assert_called_once()

        assert content == expected_content
        assert isinstance(updated_at, datetime)
        assert updated_at == datetime.fromtimestamp(mock_mtime)


@pytest.mark.asyncio
async def test_read_and_validate_file_existing_no_replace(
    mock_session, sample_python_file
):
    # Mock existing file in DB
    mock_session.execute.return_value.scalar.return_value = datetime.now()

    result = await read_and_validate_file(
        sample_python_file, mock_session, replace_existing=False, reindex_updated=False
    )

    assert result is None


# Stage 2: Document Processing Tests
@pytest.fixture
def mock_ai_functions():
    with patch('code_helper.index.clean_text') as mock_clean, \
         patch('code_helper.index.batch_process_with_retries') as mock_batch, \
         patch('code_helper.index.generate_embeddings') as mock_embed:

        async def mock_clean_impl(text):
            return text.replace("\x00", "")

        mock_clean.side_effect = mock_clean_impl
        mock_batch.return_value = ["This is a mock summary"]
        mock_embed.return_value = [0.1] * 768  # Mock embedding vector

        yield {
            "clean_text": mock_clean,
            "batch_process": mock_batch,
            "generate_embeddings": mock_embed,
        }


@pytest.mark.asyncio
async def test_process_document_content_success(mock_ai_functions):
    test_content = """
def test_function():
    pass
"""
    (
        cleaned_content,
        file_vector,
        file_metadata,
        file_summary,
    ) = await process_document_content(test_content)

    # Verify clean_text was called
    mock_ai_functions["clean_text"].assert_called_once_with(test_content)

    # Verify batch_process_with_retries was called for summary
    mock_ai_functions["batch_process"].assert_called_once()

    # Verify generate_embeddings was called
    mock_ai_functions["generate_embeddings"].assert_called_once()

    # Check return values
    assert cleaned_content == test_content  # Since our mock clean just returns the text
    assert file_vector == [0.1] * 768  # Our mock embedding vector
    assert isinstance(file_metadata, dict)  # Metadata from AST parsing
    assert file_summary == "This is a mock summary"


@pytest.mark.asyncio
async def test_process_document_content_with_null_bytes(mock_ai_functions):
    test_content = "def test():\x00    pass"

    (
        cleaned_content,
        file_vector,
        file_metadata,
        file_summary,
    ) = await process_document_content(test_content)

    # Verify null bytes were removed
    assert "\x00" not in cleaned_content
    assert file_vector == [0.1] * 768
    assert isinstance(file_metadata, dict)
    assert file_summary == "This is a mock summary"


@pytest.mark.asyncio
async def test_process_document_content_metadata_extraction(mock_ai_functions):
    test_content = """
def test_function(arg1, arg2):
    '''Test docstring'''
    return arg1 + arg2

class TestClass:
    def method(self):
        pass
"""
    _, _, file_metadata, _ = await process_document_content(test_content)

    # Verify metadata extraction
    assert isinstance(file_metadata, dict)
    assert "functions" in file_metadata
    assert "classes" in file_metadata
    assert any(f["name"] == "test_function" for f in file_metadata.get("functions", []))
    assert any(c["name"] == "TestClass" for c in file_metadata.get("classes", []))


# Stage 3: Fragment Processing Tests
@pytest.fixture
def mock_fragment_functions():
    with patch("code_helper.index.generate_embeddings") as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]  # Mock embedding vector
        yield mock_embed


@pytest.mark.asyncio
async def test_process_fragments_basic():
    test_content = """
def function1():
    pass

def function2():
    return True

class TestClass:
    def method1(self):
        pass
"""
    fragments, fragment_vectors, fragment_metadata = await process_fragments(
        test_content
    )

    # Check that fragments were extracted
    assert len(fragments) == 4  # Should have 3 functions/methods + 1 class
    assert any("function1" in f for f in fragments)
    assert any("function2" in f for f in fragments)
    assert any("TestClass" in f for f in fragments)
    assert any("method1" in f for f in fragments)

    # Check vectors
    assert len(fragment_vectors) == len(fragments)
    assert all(isinstance(v, list) for v in fragment_vectors)

    # Check metadata
    assert len(fragment_metadata) == len(fragments)
    assert all(isinstance(m, dict) for m in fragment_metadata)


@pytest.mark.asyncio
async def test_process_fragments_nested_classes():
    test_content = """
class OuterClass:
    class InnerClass:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
"""
    fragments, _, fragment_metadata = await process_fragments(test_content)

    # Check fragment extraction
    assert len(fragments) == 4  # OuterClass, InnerClass, inner_method, outer_method

    # Verify hierarchy in metadata
    inner_method_meta = next(
        m for m in fragment_metadata if m.get("name") == "inner_method"
    )
    assert inner_method_meta["parent"] == "InnerClass"

    outer_method_meta = next(
        m for m in fragment_metadata if m.get("name") == "outer_method"
    )
    assert outer_method_meta["parent"] == "OuterClass"


@pytest.mark.asyncio
async def test_process_fragments_with_docstrings():
    test_content = '''
def documented_function():
    """
    This is a docstring.
    With multiple lines.
    """
    return True
'''
    fragments, _, fragment_metadata = await process_fragments(test_content)

    # Check that docstring is preserved in fragment
    assert len(fragments) == 1
    assert "This is a docstring." in fragments[0]

    # Check metadata includes docstring info
    func_meta = fragment_metadata[0]
    assert func_meta["name"] == "documented_function"
    assert "docstring" in func_meta


@pytest.mark.asyncio
async def test_process_fragments_empty_file():
    test_content = "# Just a comment\n\n"
    fragments, fragment_vectors, fragment_metadata = await process_fragments(
        test_content
    )

    # Should handle empty files gracefully
    assert len(fragments) == 0
    assert len(fragment_vectors) == 0
    assert len(fragment_metadata) == 0


# Stage 4: Database Operations Tests

@pytest_asyncio.fixture
async def test_engine():
    """Create a test database and handle setup/teardown"""
    # Test database configuration
    TEST_DB_URL = (
        "postgresql+asyncpg://code_helper:help-me-code@localhost:5432/code_helper_test"
    )

    # Set environment variable for test database
    os.environ["DATABASE_URL"] = TEST_DB_URL

    # Import models after setting DATABASE_URL
    from code_helper.models import async_drop_db, init_db

    ECHO_SQL_QUERIES = os.getenv("CODE_HELPER_ECHO_SQL_QUERIES", "False").lower() == "true"
    test_engine = create_async_engine(
        TEST_DB_URL,
        echo=ECHO_SQL_QUERIES,
        poolclass=pool.NullPool  # Prevent connection pool issues
    )

    # Initialize test database
    await init_db(test_engine)

    yield test_engine

    # Cleanup - ensure all connections are closed
    await test_engine.dispose()
    await async_drop_db()


@pytest_asyncio.fixture
async def db_session(test_engine):
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_process_file_database_operations(
    db_session, mock_ai_functions, sample_python_file
):
    # Use db_session directly, no need for another context manager
    result = await process_file(sample_python_file, db_session)
    assert result is True

    # Query the database to verify document creation
    from code_helper.models import Document, DocumentFragment

    query = select(Document).where(Document.filename == "test.py")
    result = await db_session.execute(query)
    document = result.scalar_one()

    assert document is not None
    assert document.filename == "test.py"
    assert document.filepath == sample_python_file
    assert len(document.vector) == 768
    assert document.summary is not None
    assert isinstance(document.meta, dict)
    assert "functions" in document.meta
    assert "classes" in document.meta

    # Check fragments
    query = select(DocumentFragment).where(DocumentFragment.document_id == document.id)
    result = await db_session.execute(query)
    fragments = result.scalars().all()

    # Should have fragments for hello() and TestClass
    assert len(fragments) == 3  # hello function, TestClass, "method" method

    # Find the method fragment
    method_fragments = [f for f in fragments if f.meta.get("name") == "method"]
    method_fragment = method_fragments[0]
    assert method_fragment.meta["parent_classes"] == ["TestClass"]
    assert len(method_fragment.vector) == 768


@pytest.mark.asyncio
async def test_process_file_hierarchy(db_session, mock_ai_functions, tmp_path):
    # Create a nested directory structure
    module_dir = tmp_path / "src" / "module"
    module_dir.mkdir(parents=True)

    # Create two files in the module
    file1 = module_dir / "file1.py"
    file2 = module_dir / "file2.py"

    file1.write_text("def func1(): pass")
    file2.write_text("def func2(): pass")

    # Process both files - no need for async with since db_session is already a session
    await process_file(str(file1), db_session)
    await process_file(str(file2), db_session)

    from code_helper.models import Document

    # Check that both files are in the database with correct path information
    query = select(Document).where(Document.path_array.contains(["src", "module"]))
    result = await db_session.execute(query)
    documents = result.scalars().all()

    assert len(documents) == 2

    # Verify path information
    for doc in documents:
        assert doc.path_array[:2] == ["src", "module"]
        assert doc.filename in ["file1.py", "file2.py"]
        assert isinstance(doc.hierarchy_meta, dict)
        assert "children" in doc.hierarchy_meta


@pytest.mark.asyncio
async def test_process_file_with_class_hierarchy(db_session, mock_ai_functions):
    test_content = """
class ParentClass:
    def parent_method(self):
        pass

    class NestedClass:
        def nested_method(self):
            pass

    def another_method(self):
        pass
"""
    test_file = "test_hierarchy.py"
    with open(test_file, "w") as f:
        f.write(test_content)

    try:
        # Process the file
        await process_file(test_file, db_session)

        from code_helper.models import Document, DocumentFragment

        # Get the document
        query = select(Document).where(Document.filename == "test_hierarchy.py")
        result = await db_session.execute(query)
        document = result.scalar_one()

        # Get fragments
        query = select(DocumentFragment).where(
            DocumentFragment.document_id == document.id
        )
        result = await db_session.execute(query)
        fragments = result.scalars().all()

        # Verify class hierarchy
        parent_class_methods = [
            f for f in fragments if f.meta.get("parent") == "ParentClass"
        ]
        nested_class_methods = [
            f for f in fragments if f.meta.get("parent") == "NestedClass"
        ]

        assert len(parent_class_methods) == 2  # parent_method and another_method
        assert len(nested_class_methods) == 1  # nested_method

        # Check that methods have correct sibling relationships
        for method in parent_class_methods:
            print(f"----> Method: {method.meta.get('name')} {method.meta}")
            assert len(method.meta.get("siblings", [])) == 1
            assert method.meta["siblings"][0] in ["parent_method", "another_method"]
            assert method.meta["siblings"][0] != method.meta["name"]

    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


@pytest.mark.asyncio
async def test_process_file_reindexing(
    db_session, mock_ai_functions, sample_python_file
):
    # First indexing
    await process_file(sample_python_file, db_session)

    # Modify the file
    with open(sample_python_file, "a") as f:
        f.write("\ndef new_function(): pass\n")

    # Reindex with replace_existing=True
    result = await process_file(sample_python_file, db_session, replace_existing=True)
    assert result is True

    from code_helper.models import Document, DocumentFragment

    # Check that the new function was indexed
    query = (
        select(DocumentFragment)
        .join(Document)
        .where(Document.filepath == sample_python_file)
    )
    result = await db_session.execute(query)
    fragments = result.scalars().all()

    assert any(f.meta.get("name") == "new_function" for f in fragments)
