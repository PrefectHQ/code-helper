import ast
from typing import Any, Type, Union


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


def extract_code_fragments_from_file_content(file_content: str) -> list[str]:
    """Extract standalone functions and classes from Python source code.

    Args:
        file_content (str): Python source code as a string

    Returns:
        list[str]: List of code fragments, where each fragment is either a complete
            function definition or class definition (including all methods)
    """
    return _extract_from_file_content(file_content, CodeExtractor)


def extract_code_fragments_from_file(filepath):
    with open(filepath, "r") as file:
        file_content = file.read()

    return {
        "code": extract_code_fragments_from_file_content(file_content),
        "imports": extract_imports_from_file_content(file_content),
    }


def extract_metadata_from_node(
    node: Union[ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef],
) -> dict[str, Any]:
    """Extract metadata from an AST node representing a function or class."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {
            "type": "function",
            "name": node.name,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "parameters": [
                f"{arg.arg}: {ast.unparse(arg.annotation) if arg.annotation else 'Any'}"
                for arg in node.args.args
            ],
            "return_type": ast.unparse(node.returns) if node.returns else None,
            "decorators": [ast.unparse(d) for d in node.decorator_list],
        }
    elif isinstance(node, ast.ClassDef):
        return {
            "type": "class",
            "name": node.name,
            "parent_classes": [ast.unparse(base) for base in node.bases],
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "methods": [],  # Will be populated when processing class body
            "attributes": [],  # Will be populated when processing class body
        }
    return {}


def extract_class_attributes(node: ast.ClassDef) -> list[dict[str, Any]]:
    """Extract class attributes from assignments and annotations."""
    attributes = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign):
            # Handle type annotations: x: int
            attributes.append(
                {
                    "name": ast.unparse(item.target),
                    "type": ast.unparse(item.annotation),
                    "has_default": item.value is not None,
                }
            )
        elif isinstance(item, ast.Assign):
            # Handle regular assignments: x = value
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attributes.append(
                        {"name": target.id, "type": None, "has_default": True}
                    )
    return attributes


def extract_metadata_from_fragment(code: str) -> dict[str, Any]:
    """Extract structural metadata from Python code using AST parsing.

    Args:
        code (str): Python source code fragment as a string

    Returns:
        dict[str, Any]: Metadata dictionary containing:
            - type: "module"
            - classes: List of class metadata including:
                - name, parent_classes, decorators, methods, attributes
            - functions: List of function metadata including:
                - name, parameters, return_type, decorators, is_async
            - imports: List of imported module names

            If parsing fails, returns: {"type": "module", "error": "Could not parse code"}
    """
    try:
        tree = ast.parse(code)

        # Add parent references to all nodes
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, "parent", parent)

        metadata = {"type": "module", "classes": [], "functions": [], "imports": []}

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    metadata["imports"].extend(name.name for name in node.names)
                else:
                    module = node.module or ""
                    metadata["imports"].extend(
                        f"{module}.{name.name}" for name in node.names
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node.parent, ast.ClassDef):
                    continue  # Skip methods as they'll be handled with their class
                metadata["functions"].append(extract_metadata_from_node(node))
            elif isinstance(node, ast.ClassDef):
                class_meta = extract_metadata_from_node(node)
                class_meta["attributes"] = extract_class_attributes(node)
                class_meta["methods"] = [
                    extract_metadata_from_node(n)
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                metadata["classes"].append(class_meta)

        return metadata
    except SyntaxError:
        return {"type": "module", "error": "Could not parse code"}
