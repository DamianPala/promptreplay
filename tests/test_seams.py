"""The starter seams: `core/` depends on nothing domain-specific and modules stay small."""

import ast
import subprocess
import sys
from pathlib import Path

import provibench

PACKAGE = Path(provibench.__file__).parent
FORBIDDEN_FOR_CORE = ("provibench.commands", "provibench.bench", "provibench.app")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_imports_nothing_from_commands_or_bench() -> None:
    for path in (PACKAGE / "core").glob("*.py"):
        offending = {name for name in _imports(path) if name.startswith(FORBIDDEN_FOR_CORE)}
        assert not offending, f"{path.name} imports {offending}"


def test_no_module_exceeds_400_lines() -> None:
    for path in PACKAGE.rglob("*.py"):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 400, f"{path.relative_to(PACKAGE)} has {lines} lines"


def test_core_has_no_module_level_metadata_lookup() -> None:
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [alias.name for alias in node.names]
                module = getattr(node, "module", None)
                assert "importlib.metadata" not in names and module != "importlib.metadata", (
                    f"{path.name} imports importlib.metadata at module level"
                )


def test_constructing_the_cli_and_walking_details_stays_lightweight() -> None:
    """Building the parser and walking D8 details loads neither rich nor package metadata.

    Invoking the `schema` command itself would load `importlib.metadata` on purpose, for
    `tool_version`; what this pins is that nothing on the construction path pays for it.
    """
    code = (
        "from provibench.app import build_cli; "
        "from provibench.core.introspection import listed_commands, build_detail; "
        "import sys; "
        "root = build_cli(); "
        "[build_detail(command, name) for name, command in listed_commands(root)]; "
        "assert 'rich.console' not in sys.modules, sys.modules.keys(); "
        "assert 'importlib.metadata' not in sys.modules, sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
