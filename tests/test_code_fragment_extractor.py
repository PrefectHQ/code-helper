import pytest
from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_metadata_from_fragment,
)


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

    # Should extract 4 fragments: 2 functions and 1 class
    assert len(fragments) == 4

    # Check if each expected fragment is present
    function_names = ['simple_function', 'async_function', 'SimpleClass', 'standalone_function']
    for fragment in fragments:
        assert any(name in fragment for name in function_names)

    # Verify specific content in fragments
    assert any('async def async_function' in f for f in fragments)
    assert any('@decorator\nclass SimpleClass' in f for f in fragments)
    assert any('def standalone_function' in f for f in fragments)


def test_extract_metadata_simple_function():
    code = '''
def test_function(param1: str, param2: int = 0) -> bool:
    """Docstring."""
    return True
'''
    metadata = extract_metadata_from_fragment(code)

    assert metadata['type'] == 'module'
    assert len(metadata['functions']) == 1

    function = metadata['functions'][0]
    assert function['type'] == 'function'
    assert function['name'] == 'test_function'
    assert function['is_async'] is False
    assert len(function['parameters']) == 2
    assert function['return_type'] == 'bool'
    assert function['decorators'] == []


def test_extract_metadata_class():
    code = '''
@decorator
class TestClass:
    attr1: str
    attr2 = "default"

    def method1(self) -> None:
        pass

    @property
    def prop1(self) -> str:
        return "value"
'''
    metadata = extract_metadata_from_fragment(code)

    assert metadata['type'] == 'module'
    assert len(metadata['classes']) == 1

    class_meta = metadata['classes'][0]
    assert class_meta['type'] == 'class'
    assert class_meta['name'] == 'TestClass'
    assert len(class_meta['decorators']) == 1
    assert class_meta['decorators'][0] == 'decorator'

    # Check attributes
    attributes = class_meta['attributes']
    assert len(attributes) == 2
    assert any(attr['name'] == 'attr1' and attr['type'] == 'str' for attr in attributes)
    assert any(attr['name'] == 'attr2' and attr['has_default'] for attr in attributes)

    # Check methods
    methods = class_meta['methods']
    assert len(methods) == 2
    assert any(method['name'] == 'method1' for method in methods)
    assert any(method['name'] == 'prop1' for method in methods)


def test_extract_metadata_async_function():
    code = '''
async def async_func(param: str) -> List[str]:
    return [param]
'''
    metadata = extract_metadata_from_fragment(code)

    function = metadata['functions'][0]
    assert function['is_async'] is True
    assert function['name'] == 'async_func'
    assert function['parameters'] == ['param: str']
    assert function['return_type'] == 'List[str]'


def test_extract_metadata_invalid_code():
    code = '''
    def invalid syntax{
    '''
    metadata = extract_metadata_from_fragment(code)

    assert metadata['type'] == 'module'
    assert metadata['error'] == 'Could not parse code'


def test_extract_fragments_empty_code():
    fragments = extract_code_fragments_from_file_content('')
    assert fragments == []


def test_extract_fragments_only_imports():
    code = '''
import os
from typing import List
'''
    fragments = extract_code_fragments_from_file_content(code)
    assert fragments == []


def test_extract_metadata_nested_classes():
    code = '''
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
'''
    metadata = extract_metadata_from_fragment(code)

    assert len(metadata['classes']) == 2
    outer_class = metadata['classes'][0]
    assert outer_class['name'] == 'Outer'
    assert len(outer_class['methods']) == 1
    assert outer_class['methods'][0]['name'] == 'outer_method'

    inner_class = metadata['classes'][1]
    assert inner_class['name'] == 'Inner'
    assert len(inner_class['methods']) == 1
    assert inner_class['methods'][0]['name'] == 'inner_method'
