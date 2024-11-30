import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select

from code_helper.index import (
    process_document_content,
    process_file,
    process_fragments,
    read_and_validate_file,
)
from code_helper.models import Document, DocumentFragment


# Mock for marvin.fn decorator
def mock_marvin_fn(f):
    return f


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


@pytest.fixture
async def sample_python_file(tmp_path):
    """Create a temporary Python file with test content."""
    test_file = tmp_path / "test_process_file_reindexing0" / "test.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write initial content
    test_file.write_text("""
def hello():
    pass

class TestClass:
    def method(self):
        pass
""".strip())
    
    return str(test_file)


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        print(f"AsyncMock called with {args} and {kwargs}")
        result = super(AsyncMock, self).__call__(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def __await__(self):
        future = asyncio.Future()
        future.set_result(self())
        return future.__await__()


@pytest.mark.asyncio
async def test_read_and_validate_file_new_file(mock_session, sample_python_file):
    mock_result = MagicMock()
    mock_result.scalar = MagicMock(return_value=None)

    mock_session.execute = AsyncMock()
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()

    with open(sample_python_file, "r") as f:
        expected_content = f.read()

    mock_mtime = datetime.now(UTC)
    mock_getmtime = MagicMock(return_value=mock_mtime.timestamp())
    
    with patch("os.path.getmtime", mock_getmtime):
        result = await read_and_validate_file(
            sample_python_file,
            mock_session,
            replace_existing=False,
            reindex_updated=False,
        )
        
        mock_getmtime.assert_called_once_with(sample_python_file)

        assert result is not None
        content, timestamp = result
        assert content == expected_content
        assert isinstance(timestamp, datetime)
        assert timestamp == mock_mtime


@pytest.mark.asyncio
async def test_read_and_validate_file_existing_no_replace(
    db_session, sample_python_file
):
    # First create a document in the DB
    doc = Document(
        filename=os.path.basename(sample_python_file),
        filepath=sample_python_file,
        path_array=sample_python_file.strip("/").split("/"),
        vector=[0.1] * 768,  # Mock vector for testing
        updated_at=datetime.now(UTC)
    )
    db_session.add(doc)
    await db_session.commit()

    # Try to read and validate the file with replace_existing=False
    result = await read_and_validate_file(
        sample_python_file, 
        db_session, 
        replace_existing=False, 
        reindex_updated=False
    )

    # Should return None since file exists and replace_existing is False
    assert result is None

    # Verify the document still exists in DB
    query = select(Document).where(Document.filepath == sample_python_file)
    existing_doc = await db_session.execute(query)
    assert existing_doc.scalar_one() is not None


@pytest.fixture
def mock_ai_functions():
    with patch("code_helper.index.clean_text") as mock_clean, patch(
        "code_helper.index.run_with_retries"
    ) as mock_run, patch("code_helper.index.generate_embeddings") as mock_embed:

        async def mock_clean_impl(text):
            return text.replace("\x00", "")

        mock_clean.side_effect = mock_clean_impl
        
        async def mock_run_impl(llm_task, inputs, **kwargs):
            return ["This is a mock summary"] * len(inputs)
        
        mock_run.side_effect = mock_run_impl
        mock_embed.return_value = [0.1] * 768  # Mock embedding vector

        yield {
            "clean_text": mock_clean,
            "run_with_retries": mock_run,
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

    # Verify run_with_retries was called for summary
    mock_ai_functions["run_with_retries"].assert_called_once()

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
    fragments, fragment_summaries, fragment_vectors, fragment_metadata, hierarchy_metadata = await process_fragments(
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

    # Check summaries
    assert len(fragment_summaries) == len(fragments)
    assert all(isinstance(s, str) for s in fragment_summaries)

    # Check hierarchy metadata
    assert len(hierarchy_metadata) == len(fragments)
    assert all(isinstance(h, dict) for h in hierarchy_metadata)


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
    fragments, _, _, fragment_metadata, hierarchy_metadata = await process_fragments(test_content)

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
    
    # Check hierarchy metadata
    assert hierarchy_metadata[0]["children"] == ["InnerClass", "outer_method"]
    assert hierarchy_metadata[1]["parent"] == "OuterClass"
    assert hierarchy_metadata[1]["children"] == ["inner_method"]
    assert hierarchy_metadata[2]["parent"] == "InnerClass"
    assert hierarchy_metadata[2]["children"] == []
    assert hierarchy_metadata[3]["parent"] == "OuterClass"
    assert hierarchy_metadata[3]["children"] == []


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
    fragments, _, _, fragment_metadata, _ = await process_fragments(test_content)

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
    fragments, fragment_summaries, fragment_vectors, fragment_metadata, hierarchy_metadata = await process_fragments(
        test_content
    )

    # Should handle empty files gracefully
    assert len(fragments) == 0
    assert len(fragment_summaries) == 0
    assert len(fragment_vectors) == 0
    assert len(fragment_metadata) == 0
    assert len(hierarchy_metadata) == 0


# Stage 4: Database Operations Tests


@pytest.mark.asyncio
async def test_process_file_hierarchy(db_session, mock_ai_functions, tmp_path):
    module_dir = tmp_path / "src" / "module"
    module_dir.mkdir(parents=True)

    file1 = module_dir / "file1.py"
    file2 = module_dir / "file2.py"

    file1.write_text("def func1(): pass")
    file2.write_text("def func2(): pass")

    await process_file(str(file1), db_session)
    await process_file(str(file2), db_session)
    await db_session.commit()

    query = select(Document).where(Document.path_array.contains(["src", "module"]))
    result = await db_session.execute(query)
    documents = result.scalars().all()

    assert len(documents) == 2
    assert {doc.filename for doc in documents} == {"file1.py", "file2.py"}
    assert all("src" in doc.path_array for doc in documents)
    assert all("module" in doc.path_array for doc in documents)

    for doc in documents:
        assert hasattr(doc.vector, 'shape')
        assert doc.vector.shape == (768,)
        assert doc.vector.dtype == 'float32'
        
        assert isinstance(doc.meta, dict)
        assert doc.meta["type"] == "function"
        assert isinstance(doc.hierarchy_meta, dict)

        fragment_query = select(DocumentFragment).where(
            DocumentFragment.document_id == doc.id
        )
        fragment_result = await db_session.execute(fragment_query)
        fragments = fragment_result.scalars().all()

        assert len(fragments) == 1
        fragment = fragments[0]
        assert hasattr(fragment.vector, 'shape')
        assert fragment.vector.shape == (768,)
        assert fragment.vector.dtype == 'float32'
        assert isinstance(fragment.meta, dict)
        assert fragment.meta["type"] == "function"


@pytest.mark.asyncio
async def test_process_file_with_class_hierarchy(db_session, mock_ai_functions, tmp_path):
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
    # Use tmp_path to create a temporary file
    test_file = tmp_path / "test_hierarchy.py"
    test_file.write_text(test_content)

    # Process the file
    await process_file(str(test_file), db_session)
    await db_session.commit()

    # Get the document
    query = select(Document).where(Document.filename == "test_hierarchy.py")
    result = await db_session.execute(query)
    documents = result.scalars().all()

    # Verify document
    assert len(documents) == 1
    doc = documents[0]
    assert doc.filename == "test_hierarchy.py"
    assert isinstance(doc.vector, np.ndarray)
    assert doc.vector.shape == (768,)
    assert doc.vector.dtype == 'float32'

    # Get fragments for the document
    fragment_query = select(DocumentFragment).where(
        DocumentFragment.document_id == doc.id
    )
    fragment_result = await db_session.execute(fragment_query)
    fragments = fragment_result.scalars().all()

    # Should have 5 fragments: ParentClass, NestedClass, parent_method, nested_method, another_method
    assert len(fragments) == 5

    # Verify fragment types
    fragment_types = {f.meta["type"] for f in fragments}
    assert "class" in fragment_types
    assert "method" in fragment_types

    # Verify class hierarchy
    class_fragments = [f for f in fragments if f.meta["type"] == "class"]
    method_fragments = [f for f in fragments if f.meta["type"] == "method"]

    assert len(class_fragments) == 2  # ParentClass and NestedClass
    assert len(method_fragments) == 3  # parent_method, nested_method, another_method
    
    # Verify method parent classes
    assert method_fragments[0].hierarchy_meta["parent"] == "ParentClass"
    assert method_fragments[1].hierarchy_meta["parent"] == "NestedClass"
    assert method_fragments[2].hierarchy_meta["parent"] == "ParentClass"


@pytest.mark.asyncio
async def test_process_file_reindexing(
    db_session, mock_ai_functions, sample_python_file
):
    # Debug: Check file exists and has content
    assert os.path.exists(sample_python_file)
    with open(sample_python_file, 'r') as f:
        content = f.read()
    assert content.strip() != ""

    # First indexing
    await process_file(sample_python_file, db_session)
    await db_session.commit()

    # Verify first indexing worked
    result = await db_session.execute(
        select(Document).where(Document.filepath == sample_python_file)
    )
    document = result.scalar_one()  # Remove await
    assert document is not None
    original_id = document.id

    # Modify the file
    with open(sample_python_file, "w") as f:
        f.write("\ndef new_function(): pass\n")

    # Reindex with replace_existing=True
    result = await process_file(sample_python_file, db_session, replace_existing=True)
    assert result is True

    # Verify reindexing worked
    result = await db_session.execute(
        select(Document).where(Document.filepath == sample_python_file)
    )
    new_document = result.scalar_one()  # Remove await
    assert new_document is not None
    assert new_document.id != original_id

    # Verify old fragments were deleted
    result = await db_session.execute(
        select(DocumentFragment).where(DocumentFragment.document_id == original_id)
    )
    old_fragments = result.scalars().all()  # Remove await
    assert len(old_fragments) == 0
