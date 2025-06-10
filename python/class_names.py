
import ast
import sys

def extract_class_names(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

if __name__ == "__main__":
    path = sys.argv[1]
    for name in extract_class_names(path):
        print(name)