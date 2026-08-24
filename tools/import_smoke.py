#!/usr/bin/env python3
"""Import every module in the tree, in two stages.

Stage 1 (framework): frontier_agent + plugins + workflows, in a process where
importing ``benchmarks`` is made to fail. That enforces the layering rule —
the framework must not depend on the eval layer.

Stage 2 (eval): benchmarks too.

    python tools/import_smoke.py
    python tools/import_smoke.py --stage 1

Why every module and not a hand-picked list: a hand-picked list missed
``frontier_agent.scheduling.scheduler`` importing a name that no longer exists,
because the only importer reaches it through a function-local import inside
``BenchmarkSession._bootstrap()``. Nothing at module scope touched it.

This still cannot see:
  - names resolved lazily inside function bodies whose module *does* import
  - modules loaded by reading their source and ``exec``-ing it (the document
    reader/writer bundles under plugins/tools/) — no import statement exists
  - anything only a real run exercises
so a 1-question benchmark run stays the actual gate. The judge layer once
imported pandas eagerly through an optional dependency, which killed every
worker in a real run while this smoke stayed green.
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

AH = Path(__file__).resolve().parents[1]
# Running ``python tools/import_smoke.py`` puts ``tools/`` rather than the
# repository root on sys.path. Make the documented invocation work without
# requiring callers to remember PYTHONPATH=.
if str(AH) not in sys.path:
    sys.path.insert(0, str(AH))
FRAMEWORK = ("frontier_agent", "apodex", "plugins", "workflows")
EVAL = ("benchmarks",)


class _Blocked:
    """Meta-path finder that makes named top-level packages unimportable."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    def find_module(
        self, fullname: str, path: Sequence[str] | None = None,
    ) -> None:  # legacy hook, still consulted
        return self.find_spec(fullname, path)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> None:
        if fullname.split(".")[0] in self.names:
            raise ImportError(
                f"{fullname} is not importable from the framework layer — "
                f"the framework and CLI layers must not depend on benchmarks/"
            )
        return None


def walk(roots: tuple[str, ...]) -> list[str]:
    mods: list[str] = []
    for root in roots:
        d = AH / root
        if not d.exists():
            continue
        mods.append(root)
        for m in pkgutil.walk_packages([str(d)], prefix=f"{root}."):
            # test modules and port tooling are not part of the shipped tree
            if ".tests." in m.name or m.name.rsplit(".", 1)[-1].startswith("test_"):
                continue
            mods.append(m.name)
    return sorted(set(mods))


def run_stage(name: str, roots: tuple[str, ...], block: tuple[str, ...]) -> int:
    if block:
        sys.meta_path.insert(0, _Blocked(block))
    mods = walk(roots)
    failures: list[tuple[str, str]] = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            failures.append((m, traceback.format_exc(limit=3)))
    print(f"\n[{name}] {len(mods) - len(failures)}/{len(mods)} modules imported")
    for m, tb in failures:
        print(f"\n--- FAIL {m}\n{tb.rstrip()}")
    return len(failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), help="run only this stage")
    args = ap.parse_args()

    # Stages must not share a process: stage 1 poisons the import system.
    if args.stage == 1:
        return min(run_stage("framework", FRAMEWORK, EVAL), 1)
    if args.stage == 2:
        return min(run_stage("eval", FRAMEWORK + EVAL, ()), 1)

    import subprocess
    rc = 0
    for stage in (1, 2):
        r = subprocess.run([sys.executable, __file__, "--stage", str(stage)], cwd=AH)
        rc |= r.returncode
    print(f"\nimport smoke: {'FAIL' if rc else 'OK'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
