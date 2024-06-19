import ast
import sys


class CodeExtractor(ast.NodeVisitor):
    def __init__(self):
        self.fragments = []

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


def extract_code_fragments_from_file_content(file_content: str):
    # Parse the file content into an AST
    tree = ast.parse(file_content)

    # Set parent references for nodes
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # Extract code fragments
    extractor = CodeExtractor()
    extractor.visit(tree)

    code_fragments = []

    for node in extractor.fragments:
        # Extract the code fragment
        fragment_code = ast.unparse(node)
        code_fragments.append(fragment_code)

    return code_fragments


def extract_code_fragments_from_file(filepath):
    with open(filepath, 'r') as file:
        file_content = file.read()

    return extract_code_fragments_from_file_content(file_content)


# Example usage
if __name__ == "__main__":
    filepath = sys.argv[1]
    code_fragments = extract_code_fragments_from_file(filepath)

    for i, fragment_code in enumerate(code_fragments):
        print(f"Fragment {i+1}:\n{fragment_code}\n")
