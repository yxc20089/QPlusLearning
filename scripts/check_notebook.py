"""Check notebook syntax and pinned helper hashes without a GPU or model download."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "labs/lab-01-kev-pacman"


def check_notebook(path, lock):
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    code_cells = 0
    files = None
    revision = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        assert cell["outputs"] == [] and cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]))
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FILES"
                for target in statement.targets
            ):
                files = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "COURSE_REVISION"
                for target in statement.targets
            ):
                revision = ast.literal_eval(statement.value)
    assert files, "Missing notebook source manifest"
    assert revision == lock["revision"] and files == lock["files"]
    assert len(revision) == 40
    for name, expected in files.items():
        assert hashlib.sha256((LAB / name).read_bytes()).hexdigest() == expected, f"Stale helper: {name}"
    print(f"{path.name}: checked {code_cells} code cells and {len(files)} pinned helper files.")


def main():
    lock = json.loads((LAB / "notebook-source.json").read_text())
    for path in sorted((LAB / "notebooks").glob("*.ipynb")):
        check_notebook(path, lock)


if __name__ == "__main__":
    main()
