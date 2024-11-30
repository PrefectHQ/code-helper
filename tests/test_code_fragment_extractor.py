import pytest
from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_metadata_from_fragment,
    extract_metadata_from_node,
)
import ast

from code_helper.index import process_fragments


@pytest.fixture
def sample_code():
    return '''
import os
from typing import List, Optional

def simple_function():
    """A simple function."""
    return "Hello"

async def async_function(param1: str, param2: int = 0) -> List[str]:
    """An async function with type hints."""
    return [param1] * param2

@decorator
class SimpleClass:
    class_var: str = "value"

    def __init__(self, name: str):
        self.name = name

    @property
    def title(self) -> str:
        return self.name.title()

def standalone_function(x: int) -> int:
    return x * 2
'''


def test_extract_code_fragments(sample_code):
    fragments = extract_code_fragments_from_file_content(sample_code)

    assert len(fragments) == 6
    assert all(isinstance(f, tuple) and len(f) == 2 for f in fragments)

    for expected_fragment in [
        "def simple_function",
        "async def async_function",
        "class SimpleClass",
        "def __init__",
        "def title",
        "def standalone_function",
    ]:
        assert any(
            expected_fragment in f[1] for f in fragments
        ), f"Expected {expected_fragment} not found"

    for node, _ in fragments:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metadata = extract_metadata_from_node(node)
            if hasattr(node, "parent_class"):
                assert metadata["type"] == "method"
                assert "parent" in metadata
                assert "parent_classes" in metadata
            else:
                assert metadata["type"] == "function"


def test_extract_metadata_simple_function():
    code = '''
def test_function(param1: str, param2: int = 0) -> bool:
    """Docstring."""
    return True
'''
    metadata = extract_metadata_from_fragment(code)

    assert metadata["type"] == "function"
    assert metadata["name"] == "test_function"
    assert metadata["is_async"] is False
    assert len(metadata["parameters"]) == 2
    assert metadata["return_type"] == "bool"
    assert metadata["decorators"] == []


def test_extract_metadata_class():
    code = """
@decorator
class TestClass:
    attr1: str
    attr2 = "default"

    def method1(self) -> None:
        pass

    @property
    def prop1(self) -> str:
        return "value"
"""
    # First get the fragments to test node extraction
    fragments = extract_code_fragments_from_file_content(code)

    # Find the class node and method nodes
    class_node = next(node for node, _ in fragments if isinstance(node, ast.ClassDef))
    method_nodes = [node for node, _ in fragments if isinstance(node, ast.FunctionDef)]

    # Test class metadata
    class_meta = extract_metadata_from_node(class_node)
    assert class_meta["type"] == "class"
    assert class_meta["name"] == "TestClass"
    assert len(class_meta["decorators"]) == 1
    assert class_meta["decorators"][0] == "decorator"

    # Test method metadata
    for node in method_nodes:
        method_meta = extract_metadata_from_node(node)
        assert method_meta["type"] == "method"
        assert method_meta["parent"] == "TestClass"
        assert method_meta["parent_classes"] == ["TestClass"]


def test_extract_metadata_async_function():
    code = """
async def async_func(param: str) -> List[str]:
    return [param]
"""
    metadata = extract_metadata_from_fragment(code)

    assert metadata["is_async"] is True
    assert metadata["name"] == "async_func"
    assert metadata["parameters"] == ["param: str"]
    assert metadata["return_type"] == "List[str]"


def test_extract_metadata_invalid_code():
    code = """
    def invalid syntax{
    """
    metadata = extract_metadata_from_fragment(code)

    assert metadata["type"] == "module"
    assert metadata["error"] == "Could not parse code"


def test_extract_fragments_empty_code():
    fragments = extract_code_fragments_from_file_content("")
    assert fragments == []


def test_extract_fragments_only_imports():
    code = """
import os
from typing import List
"""
    fragments = extract_code_fragments_from_file_content(code)
    assert fragments == []


def test_extract_metadata_inner_functions():
    code = """
def outer_function():
    def inner_function1():
        pass

    async def inner_function2():
        pass

    return True
"""
    metadata = extract_metadata_from_fragment(code)

    assert metadata["type"] == "function"
    assert metadata["name"] == "outer_function"
    assert len(metadata["inner_functions"]) == 2
    assert metadata["inner_functions"][0]["name"] == "inner_function1"
    assert metadata["inner_functions"][1]["name"] == "inner_function2"
    assert metadata["inner_functions"][1]["is_async"] is True


def test_extract_metadata_nested_classes():
    code = """
class OuterClass:
    class InnerClass1:
        def method1(self):
            pass

    class InnerClass2:
        pass

    def outer_method(self):
        pass
"""
    fragments = extract_code_fragments_from_file_content(code)

    # Verify all nodes have correct metadata
    for node, _ in fragments:
        metadata = extract_metadata_from_node(node)

        if isinstance(node, ast.ClassDef):
            assert metadata["type"] == "class"
            if metadata["name"] == "OuterClass":
                assert not hasattr(node, "parent_class")
            else:
                assert hasattr(node, "parent_class")
                assert node.parent_class == "OuterClass"
        else:  # FunctionDef
            assert metadata["type"] == "method"
            if metadata["name"] == "outer_method":
                assert metadata["parent"] == "OuterClass"
            else:
                assert metadata["parent"] == "InnerClass1"

    # Also test the full fragment metadata for nested structure
    metadata = extract_metadata_from_fragment(code)
    assert metadata["type"] == "class"
    assert metadata["name"] == "OuterClass"
    assert len(metadata["nested_classes"]) == 2
    assert metadata["nested_classes"][0]["name"] == "InnerClass1"
    assert len(metadata["nested_classes"][0]["methods"]) == 1
    assert metadata["nested_classes"][1]["name"] == "InnerClass2"
    assert len(metadata["methods"]) == 1
    assert metadata["methods"][0]["name"] == "outer_method"


@pytest.mark.asyncio

async def test_process_fragments_preserves_class_context():
    test_content = """
class TestClass:
    def method1(self):
        pass

    async def method2(self):
        pass
"""
    fragments, fragment_summaries, fragment_vectors, fragment_metadata, hierarchy_metadata = await process_fragments(test_content)

    # Should have 3 fragments: class and two methods
    assert len(fragments) == 3
    assert len(fragment_vectors) == 3
    assert len(fragment_metadata) == 3

    # Verify class metadata
    class_meta = next(m for m in fragment_metadata if m["type"] == "class")
    assert class_meta["name"] == "TestClass"

    # Verify method metadata
    method_metas = [m for m in fragment_metadata if m["type"] == "method"]
    assert len(method_metas) == 2

    # Check regular method
    method1 = next(m for m in method_metas if m["name"] == "method1")
    assert method1["parent"] == "TestClass"
    assert method1["parent_classes"] == ["TestClass"]
    assert method1["is_async"] is False

    # Check async method
    method2 = next(m for m in method_metas if m["name"] == "method2")
    assert method2["parent"] == "TestClass"
    assert method2["parent_classes"] == ["TestClass"]
    assert method2["is_async"] is True
