#!/usr/bin/env python3
"""Prove that every lazily re-exported name actually resolves.

A package with a module-level ``__getattr__`` (PEP 562) resolves names at
runtime. ``check_symbols.py`` therefore treats such a package's ``__all__`` as
its contract and stops there — which means an entry pointing at a module that
does not exist looks fine to every static check, and to an import smoke that
only imports the package. It fails for the first caller that touches the
attribute, which can be deep into a long run.

That is not hypothetical: the LLM middleware package advertised seven
middlewares whose modules were never ported, and ruff, check_symbols and both
import-smoke stages were all green.

    python tools/check_lazy_exports.py

Exits non-zero on the first unresolvable name, listing all of them.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

AH = Path(__file__).resolve().parents[1]
ROOTS = ("frontier_agent", "apodex", "plugins", "workflows", "benchmarks")


def lazy_names(init: Path) -> list[str]:
    """``__all__`` of a package whose ``__init__`` defines ``__getattr__``."""
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body
    ):
        return []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets
        ):
            return [
                e.value for e in getattr(n.value, "elts", [])
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return []


def main() -> int:
    sys.path.insert(0, str(AH))
    holes: list[str] = []
    packages = checked = 0
    for init in sorted(AH.rglob("__init__.py")):
        if "__pycache__" in init.parts or ".venv" in init.parts:
            continue
        rel = init.relative_to(AH).parts
        if not rel or rel[0] not in ROOTS:
            continue
        names = lazy_names(init)
        if not names:
            continue
        packages += 1
        pkg = ".".join(init.parent.relative_to(AH).parts)
        try:
            mod = importlib.import_module(pkg)
        except Exception as e:
            holes.append(f"{pkg}: package itself failed to import: {e!r}")
            continue
        for name in names:
            checked += 1
            try:
                getattr(mod, name)
            except Exception as e:
                holes.append(f"{pkg}.{name}: {type(e).__name__}: {e}")

    for h in holes:
        print(h)
    print(
        f"\n{'FAIL' if holes else 'OK'}: {len(holes)} unresolvable lazy export(s); "
        f"checked {checked} name(s) across {packages} lazy package(s)"
    )
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
