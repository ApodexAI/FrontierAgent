#!/usr/bin/env python3
"""Find `from X import name` where `name` does not exist in X.

"Does the module exist" is the easy half. The half that bites is a file
importing a *name* that its target module does not define — Python reports it
only when that import actually runs, which for a lazily-reached module can be
deep into a long run.

    python tools/check_symbols.py                 # whole tree
    python tools/check_symbols.py frontier_agent/core/runtime/loop

Run it after any change that moves symbols between modules; it also answers
"which of these files have to land together", since a mutually-required pair
shows up as two errors that only clear simultaneously.

Understands PEP 562 lazy re-exports: a module with a module-level
``__getattr__`` has its ``__all__`` treated as the contract.

Only sees `from X import name`. Attribute access (`mod.name`) is invisible, so
this narrows the risk rather than eliminating it — `tools/import_smoke.py`
covers what actually resolves at import time.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

AH = Path(__file__).resolve().parents[1]
ROOTS = ("frontier_agent", "plugins", "workflows", "benchmarks")


def module_file(dotted: str) -> Path | None:
    sub = Path(*dotted.split("."))
    for cand in (AH / f"{sub}.py", AH / sub / "__init__.py"):
        if cand.exists():
            return cand
    return None


_cache: dict[Path, set[str]] = {}


_LAZY = object()  # marker: module resolves unknown names at runtime


def top_level_names(path: Path) -> set[str]:
    """Names a module binds at module scope, including re-exports.

    A module-level ``__getattr__`` (PEP 562) resolves names at runtime, so its
    ``__all__`` is the contract and anything in it counts as present.
    """
    if path in _cache:
        return _cache[path]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return _cache.setdefault(path, set())
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__getattr__":
            for m in tree.body:
                if isinstance(m, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in m.targets
                ):
                    names |= {
                        e.value for e in getattr(m.value, "elts", [])
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    }
    for n in tree.body:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.Import | ast.ImportFrom):
            names |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.If | ast.Try):
            # conditional definitions (TYPE_CHECKING, optional deps)
            for sub in ast.walk(n):
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    names |= {t.id for t in sub.targets if isinstance(t, ast.Name)}
                elif isinstance(sub, ast.Import | ast.ImportFrom):
                    names |= {a.asname or a.name.split(".")[0] for a in sub.names}
    return _cache.setdefault(path, names)


def scan(paths: list[Path]) -> list[str]:
    bad = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(AH)}: SYNTAX ERROR {e}")
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom) or not n.module:
                continue
            if n.module.split(".")[0] not in ROOTS:
                continue
            target = module_file(n.module)
            if target is None:
                bad.append(f"{p.relative_to(AH)}:{n.lineno}  MODULE MISSING  {n.module}")
                continue
            have = top_level_names(target)
            for a in n.names:
                if a.name == "*":
                    continue
                # a submodule import (from pkg import mod) is fine if the file exists
                if a.name not in have and module_file(f"{n.module}.{a.name}") is None:
                    bad.append(
                        f"{p.relative_to(AH)}:{n.lineno}  {n.module}.{a.name}"
                        f"  not in {target.relative_to(AH)}"
                    )
    return bad


def main() -> int:
    if len(sys.argv) > 1:
        targets = []
        for arg in sys.argv[1:]:
            p = (AH / arg) if not Path(arg).is_absolute() else Path(arg)
            targets += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    else:
        targets = [
            p for r in ROOTS if (AH / r).exists()
            for p in sorted((AH / r).rglob("*.py"))
            if "__pycache__" not in p.parts
        ]
    bad = scan([p for p in targets if "__pycache__" not in p.parts])
    for b in bad:
        print(b)
    print(f"\n{'FAIL' if bad else 'OK'}: {len(bad)} missing-symbol import(s) "
          f"across {len(targets)} file(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
