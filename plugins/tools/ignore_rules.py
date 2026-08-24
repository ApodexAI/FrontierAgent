"""Shared ignore rules for local repository inspection tools."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".git/",
    ".venv/",
    "venv/",
    # Legacy in-workspace recovery store. The current store is filtered by its
    # resolved root in grep/glob; reserving the generic name ``spill`` would hide
    # legitimate user directories with that common name.
    ".spill/",
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".next/",
    "dist/",
    "build/",
    "coverage/",
    ".coverage",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.tsbuildinfo",
)


@dataclass(frozen=True)
class IgnoreRule:
    pattern: str
    negated: bool = False


class IgnoreMatcher:
    """Apply ordered ignore rules to repository-relative paths."""

    def __init__(self, root: Path, rules: list[IgnoreRule]) -> None:
        self.root = root
        self._resolved_root = root.resolve()
        self.rules = rules

    def should_ignore(self, path: Path) -> bool:
        rel = path.resolve().relative_to(self._resolved_root).as_posix()
        ignored = False
        for rule in self.rules:
            if _matches(rule.pattern, rel):
                ignored = not rule.negated
        return ignored


def discover_repo_root(start: Path | None = None) -> Path:
    """Find the nearest repository-ish root from the current working directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / ".gitignore").exists():
            return candidate
    return current


@lru_cache(maxsize=16)
def load_ignore_matcher(root: str) -> IgnoreMatcher:
    root_path = Path(root).resolve()
    rules: list[IgnoreRule] = [IgnoreRule(pattern=p) for p in DEFAULT_IGNORE_PATTERNS]
    gitignore_path = root_path / ".gitignore"
    if gitignore_path.is_file():
        for raw_line in gitignore_path.read_text(encoding="utf-8").splitlines():
            rule = _parse_gitignore_line(raw_line)
            if rule is not None:
                rules.append(rule)
    return IgnoreMatcher(root=root_path, rules=rules)


def should_ignore_path(path: Path, root: Path | None = None) -> bool:
    """Check whether a file should be ignored for local repo inspection."""
    resolved_root = (root or discover_repo_root(path.parent if path.is_absolute() else Path.cwd())).resolve()
    try:
        path.resolve().relative_to(resolved_root)
    except ValueError:
        return False
    return load_ignore_matcher(str(resolved_root)).should_ignore(path)


def _parse_gitignore_line(raw_line: str) -> IgnoreRule | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    negated = line.startswith("!")
    if negated:
        line = line[1:].strip()
    if not line:
        return None
    return IgnoreRule(pattern=line, negated=negated)


def _matches(pattern: str, rel_path: str) -> bool:
    normalized = pattern
    if normalized.startswith("./"):
        normalized = normalized[2:]
    anchored = pattern.startswith("/")
    directory_only = normalized.endswith("/")
    if directory_only:
        normalized = normalized.rstrip("/")

    path_parts = rel_path.split("/")
    basename = path_parts[-1]

    if "/" in normalized:
        candidate = normalized.lstrip("/")
        if anchored:
            if directory_only:
                return rel_path == candidate or rel_path.startswith(candidate + "/")
            return fnmatch(rel_path, candidate)
        if directory_only:
            return candidate in rel_path or rel_path.startswith(candidate + "/")
        return fnmatch(rel_path, candidate) or fnmatch(basename, candidate)

    if directory_only:
        return normalized in path_parts[:-1] or rel_path == normalized

    return any(fnmatch(part, normalized) for part in path_parts)
