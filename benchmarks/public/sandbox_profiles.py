"""Benchmark-side contracts for file-aware sandbox tasks."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MountSpec = dict[str, str]


@dataclass(frozen=True)
class SandboxProfile:
    inputs: Callable[[dict[str, Any], Path], list[MountSpec]] = lambda _m, _r: []
    prompt_addendum: str = ""
    collect_outputs: bool = False
    default_profile: str = ""
    prepare: Callable[[dict[str, Any], Path, Path], None] = lambda _m, _r, _w: None
    # Whether the benchmark is answerable *only* from what it provides. A
    # closed-book benchmark mounts its corpus and expects the answer to be
    # derived from it, so leaving web tools bound measures something else — and
    # makes the score incomparable with anyone reporting it closed-book. The
    # runner turns this into REACT_NO_WEB / SWARM_NO_WEB for the workers, and
    # ``--web`` / ``--no-web`` overrides it when you want the other number.
    closed_book: bool = False


def _officeqa_inputs(_meta: dict[str, Any], _root: Path) -> list[MountSpec]:
    mode = os.environ.get("OFFICEQA_DOC_MODE", "parsed").strip().lower()
    corpus = "treasury_bulletin_pdfs" if mode == "raw" else "treasury_bulletins_parsed/transformed"
    return [{"src": corpus, "dst": "/inputs", "mode": "ro"}]


def _reference_files(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split("|") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _gdpval_inputs(meta: dict[str, Any], _root: Path) -> list[MountSpec]:
    mounts: list[MountSpec] = []
    used: set[str] = set()
    for relative in _reference_files(meta.get("reference_files")):
        path = Path(relative)
        destination = f"/inputs/{path.name}"
        if destination in used:
            destination = f"/inputs/{path.parent.name or 'ref'}__{path.name}"
        suffix = 1
        base_destination = destination
        while destination in used:
            destination = f"{base_destination}.{suffix}"
            suffix += 1
        used.add(destination)
        mounts.append({"src": relative, "dst": destination, "mode": "ro"})
    return mounts


def _apex_inputs(meta: dict[str, Any], _root: Path) -> list[MountSpec]:
    mounts: list[MountSpec] = []
    world_id = str(meta.get("world_id") or "").strip()
    task_id = str(meta.get("bench_task_id") or meta.get("question_id") or "").strip()
    if world_id:
        mounts.append({"src": f"world_files_zipped/{world_id}.zip", "dst": "/inputs/apex/world.zip", "mode": "ro"})
    if task_id:
        mounts.append({"src": f"task_files/{task_id}", "dst": "/inputs/apex/task_files", "mode": "ro"})
    return mounts


def _prepare_apex(meta: dict[str, Any], root: Path, worktree: Path) -> None:
    world_id = str(meta.get("world_id") or "").strip()
    if world_id:
        from benchmarks.public.apex_world import populate

        populate(world_id, str(meta.get("bench_task_id") or meta.get("question_id") or ""), root, worktree)


_OFFICEQA_PROMPT = (
    "The Treasury document corpus is mounted read-only at /inputs. Search it selectively and cite "
    "the source files used. Return the final value inside <FINAL_ANSWER>...</FINAL_ANSWER>."
)
_GDPVAL_PROMPT = (
    "Reference files are mounted read-only under /inputs. Save only verified final deliverables "
    "under /outputs; use /workspace for scratch work. Files under /outputs are the graded answer."
)
_APEX_PROMPT = (
    "The task world is staged under /workspace: filesystem contains user documents and .apps_data "
    "contains readable app state. Use only provided files, and write requested file changes under "
    "/workspace/filesystem."
)

SANDBOX_PROFILES = {
    "officeqa": SandboxProfile(inputs=_officeqa_inputs, prompt_addendum=_OFFICEQA_PROMPT, default_profile="default", closed_book=True),
    "officeqa_full": SandboxProfile(inputs=_officeqa_inputs, prompt_addendum=_OFFICEQA_PROMPT, default_profile="default", closed_book=True),
    "gdpval": SandboxProfile(inputs=_gdpval_inputs, prompt_addendum=_GDPVAL_PROMPT, collect_outputs=True, default_profile="default"),
    "apex": SandboxProfile(inputs=_apex_inputs, prompt_addendum=_APEX_PROMPT, default_profile="default", prepare=_prepare_apex, closed_book=True),
    "onemillion_bench": SandboxProfile(),
}


def _infer_benchmark(meta: dict[str, Any]) -> str:
    explicit = str(meta.get("benchmark") or "").strip()
    if explicit:
        return explicit
    answer_type = str(meta.get("answer_type") or "").strip().lower()
    if answer_type == "onemillion":
        return "onemillion_bench"
    if answer_type in SANDBOX_PROFILES:
        return answer_type
    if meta.get("world_id"):
        return "apex"
    if meta.get("reference_files"):
        return "gdpval"
    if meta.get("source_files"):
        return "officeqa"
    return ""


def apply_sandbox_profile(meta: dict[str, Any], *, worktree: str | Path | None = None) -> dict[str, Any]:
    """Expand compact dataset metadata into workflow sandbox metadata."""
    benchmark = _infer_benchmark(meta)
    profile = SANDBOX_PROFILES.get(benchmark)
    if profile is None:
        return meta
    from benchmarks.public.core.registry import dataset_root_for

    dataset_root = dataset_root_for(benchmark)
    meta["benchmark"] = benchmark
    meta["_dataset_root"] = str(dataset_root)
    meta["_sandbox_mounts"] = profile.inputs(meta, dataset_root)
    meta["_collect_outputs"] = profile.collect_outputs
    if profile.prompt_addendum:
        meta["_sys_prompt_addendum"] = profile.prompt_addendum
    if profile.default_profile:
        meta.setdefault("profile", profile.default_profile)
    meta.setdefault("experiment", benchmark)
    meta.setdefault("bench_task_id", meta.get("question_id", ""))
    if worktree is not None:
        profile.prepare(meta, dataset_root, Path(worktree))
    return meta


def resolve_closed_book(benchmark: str, override: bool | None = None) -> bool:
    """Whether *benchmark* should run without web tools.

    ``override`` (from ``--no-web`` / ``--web``) wins; otherwise the benchmark's
    own declaration decides, defaulting to open-book for anything unregistered
    (most benchmarks are web research tasks).
    """
    if override is not None:
        return override
    profile = SANDBOX_PROFILES.get(benchmark)
    return bool(profile.closed_book) if profile else False
