import ast
from typing import Any, Type, Union


class FragmentExtractor(ast.NodeVisitor):
    def __init__(self):
        self.fragments = []  # Will store tuples of (node, string_repr)
        super().__init__()


class CodeExtractor(FragmentExtractor):
    def __init__(self):
        self.current_class = None
        super().__init__()

    def visit_FunctionDef(self, node):
        # Save both node and string representation
        if self.current_class:
            node.parent_class = self.current_class
        self.fragments.append((node, ast.unparse(node)))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if self.current_class:
            node.parent_class = self.current_class
        self.fragments.append((node, ast.unparse(node)))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        previous_class = self.current_class
        self.current_class = node.name
        self.fragments.append((node, ast.unparse(node)))
        self.generic_visit(node)
        self.current_class = previous_class


class ImportExtractor(FragmentExtractor):
    def visit_Import(self, node):
        self.fragments.append((node, ast.unparse(node)))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.fragments.append((node, ast.unparse(node)))
        self.generic_visit(node)


def _extract_from_file_content(
    file_content: str, extractor_cls: Type[FragmentExtractor]
) -> list[tuple[ast.AST, str]]:
    tree = ast.parse(file_content)

    # Set parent references for nodes
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    extractor = extractor_cls()
    extractor.visit(tree)

    return extractor.fragments


def extract_imports_from_file_content(file_content: str) -> list[tuple[ast.AST, str]]:
    return _extract_from_file_content(file_content, ImportExtractor)


def extract_code_fragments_from_file_content(file_content: str) -> list[tuple[ast.AST, str]]:
    """Extract standalone functions and classes from Python source code.

    Args:
        file_content (str): Python source code as a string

    Returns:
        list[tuple[ast.AST, str]]: List of tuples containing (AST node, code string)
            for each fragment
    """
    return _extract_from_file_content(file_content, CodeExtractor)


def extract_code_fragments_from_file(filepath: str) -> dict[str, list[tuple[ast.AST, str]]]:
    with open(filepath, "r") as file:
        file_content = file.read()

    return {
        "code": extract_code_fragments_from_file_content(file_content),
        "imports": extract_imports_from_file_content(file_content),
    }


def extract_metadata_from_node(node: ast.AST) -> dict:
    metadata = {
        "name": getattr(node, "name", None),
        "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
        "decorators": [ast.unparse(d) for d in getattr(node, "decorator_list", [])],
    }
    
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        metadata.update({
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "parameters": [ast.unparse(arg) for arg in node.args.args],
            "return_type": ast.unparse(node.returns) if node.returns else None,
            "docstring": ast.get_docstring(node) or "",
        })
        
        if hasattr(node, "parent_class"):
            metadata["type"] = "method"
            metadata["parent"] = node.parent_class
            metadata["parent_classes"] = [node.parent_class]
            
    elif isinstance(node, ast.ClassDef):
        metadata["docstring"] = ast.get_docstring(node) or ""
        
    return metadata


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
            - type: "module", "class", or "function"
            - imports: List of imported module names

            If "type" is "module":
                - classes: List of class metadata
                - functions: List of function metadata (only top-level functions)

            If "type" is "class":
                - name: Class name
                - parent_classes: List of base classes
                - decorators: List of decorator strings
                - methods: List of method metadata
                - attributes: List of class attributes

            If "type" is "function":
                - name: Function name
                - parameters: List of parameter strings with type hints
                - return_type: Return type annotation if present
                - decorators: List of decorators
                - is_async: Boolean indicating if function is async
                - inner_functions: List of inner function metadata

            If "type" is "method":
                - parent_classes: List of parent class names
                - decorators: List of decorators
                - parameters: List of parameter strings with type hints
                - return_type: Return type annotation if present
                - is_async: Boolean indicating if function is async
                - inner_functions: List of inner function metadata
    """
    try:
        tree = ast.parse(code)

        # Add parent references to all nodes
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                setattr(child, "parent", parent)

        metadata = {"type": "module", "imports": [], "classes": [], "functions": []}

        def process_class_node(node):
            """Helper function to process a class node and its contents"""
            class_meta = extract_metadata_from_node(node)
            class_meta["attributes"] = extract_class_attributes(node)

            # Extract methods and add parent class info
            methods = []
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n.parent_class = node.name  # Set parent class for method
                    method_meta = extract_metadata_from_node(n)
                    methods.append(method_meta)
            class_meta["methods"] = methods

            # Extract nested classes recursively
            class_meta["nested_classes"] = [
                process_class_node(n)
                for n in node.body
                if isinstance(n, ast.ClassDef)
            ]

            return class_meta

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
                # Skip if this is a method (parent is a class) as it will be handled with the class
                if isinstance(node.parent, ast.ClassDef):
                    continue

                # Handle inner functions
                if isinstance(node.parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent_func = next(
                        (f for f in metadata["functions"] if f["name"] == node.parent.name),
                        None
                    )
                    if parent_func:
                        if "inner_functions" not in parent_func:
                            parent_func["inner_functions"] = []
                        parent_func["inner_functions"].append(extract_metadata_from_node(node))
                    continue

                # Add top-level function
                func_meta = extract_metadata_from_node(node)
                func_meta["inner_functions"] = []
                metadata["functions"].append(func_meta)

            elif isinstance(node, ast.ClassDef):
                # Skip nested classes as they'll be handled by process_class_node
                if isinstance(node.parent, ast.ClassDef):
                    continue

                # Process top-level class
                metadata["classes"].append(process_class_node(node))

        # Determine the most specific type based on content
        if len(metadata["classes"]) == 1 and not metadata["functions"] and not metadata["imports"]:
            return metadata["classes"][0]
        elif len(metadata["functions"]) == 1 and not metadata["classes"] and not metadata["imports"]:
            return metadata["functions"][0]

        return metadata

    except SyntaxError:
        return {"type": "module", "error": "Could not parse code"}
