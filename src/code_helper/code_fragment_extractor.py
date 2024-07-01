import ast
from typing import Type


class FragmentExtractor(ast.NodeVisitor):
    def __init__(self):
        self.fragments = []
        super().__init__()


class CodeExtractor(FragmentExtractor):
    def visit_FunctionDef(self, node):
        if not isinstance(node.parent, ast.ClassDef):
            self.fragments.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if not isinstance(node.parent, ast.ClassDef):
            self.fragments.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.fragments.append(node)
        self.generic_visit(node)


class ImportExtractor(FragmentExtractor):
    def visit_Import(self, node):
        self.fragments.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.fragments.append(node)
        self.generic_visit(node)


def _extract_from_file_content(
    file_content: str, extractor_cls: Type[FragmentExtractor]
):
    tree = ast.parse(file_content)

    # Set parent references for nodes
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    extractor = extractor_cls()
    extractor.visit(tree)

    code_fragments = []

    for node in extractor.fragments:
        fragment_code = ast.unparse(node)
        code_fragments.append(fragment_code)

    return code_fragments


def extract_imports_from_file_content(file_content: str):
    return _extract_from_file_content(file_content, ImportExtractor)


def extract_code_fragments_from_file_content(file_content: str):
    return _extract_from_file_content(file_content, CodeExtractor)


def extract_code_fragments_from_file(filepath):
    with open(filepath, "r") as file:
        file_content = file.read()

    return {
        "code": extract_code_fragments_from_file_content(file_content),
        "imports": extract_imports_from_file_content(file_content),
    }
