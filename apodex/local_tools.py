"""Local-host bash tool for the coding terminal.

Why this exists: FrontierAgent's shared ``plugins.tools.bash`` runs in an
fail-closed **E2B or bubblewrap sandbox** — never in the user's actual
repository. That is correct for research / SWE-bench isolation, but it
means a local coding agent's ``bash`` would run on a *different
filesystem* than the file-edit tools (read_file / file_editor), which DO
operate on the local repo. The agent then can't build/test the files it
just edited.

This tool runs the command with ``cwd=os.getcwd()`` — the same working
directory the session ``chdir``-ed into and the file tools are authorized
for — so bash, reads, and edits all share one filesystem. The
``assess_bash_command`` denylist is reused as a hard, fail-closed backstop
(defense-in-depth alongside the TUI approval gate).
"""

from __future__ import annotations

import functools
import os
import re
import shlex
from collections.abc import Iterator

from frontier_agent.core.tool import tool
from plugins.tools.bash import BASH_STDERR_SEPARATOR, assess_bash_command

_BASH_TIMEOUT = int(os.environ.get("MHT_BASH_TIMEOUT", "300"))
_MAX_READ_BYTES = 200_000
_MAX_GREP_FILE_BYTES = 5_000_000  # skip files larger than this in grep_search

# Generated / dependency / VCS / cache dirs that are artifacts in essentially
# EVERY project — never the target of a code search. This is the only
# hardcoded set, and it is deliberately project-agnostic (no app-specific
# names like "results"/"data"). Anything project-specific comes from the
# repo's OWN .gitignore (see _gitignored_dir_names), so the prune behaviour
# generalizes across repos instead of being tuned to one.
_UNIVERSAL_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", ".apodex",            # VCS / agent runtime state
    ".venv", "venv", "node_modules",             # dependency trees
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".gradle",            # tool caches
    ".next", ".nuxt", ".svelte-kit",             # JS build caches
})


@functools.lru_cache(maxsize=64)
def _gitignored_dir_names(root: str) -> frozenset[str]:
    """Directory names the repo's own ``.gitignore`` declares ignored.

    This is how repo-specific generated dirs (e.g. ``results/``, ``tmp/``,
    ``dist/`` — whatever THIS repo treats as output) get pruned, without
    hardcoding any project's layout. Only simple directory entries are taken
    (glob/negation lines are skipped); matched by basename, a conservative
    over-approximation that keeps search off declared-ignored output.
    """
    names: set[str] = set()
    try:
        with open(os.path.join(root, ".gitignore"), encoding="utf-8", errors="replace") as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#") or s.startswith("!"):
                    continue
                s = s.strip("/")
                if not s or any(c in s for c in "*?[]"):
                    continue
                # Only BARE directory names (no internal '/'). git treats a
                # bare ``foo/`` as "ignore foo anywhere", so basename pruning is
                # correct. Path-anchored entries like ``a/b/`` are skipped to
                # avoid wrongly pruning an unrelated dir that shares the leaf
                # name (e.g. ``src/results`` when only ``benchmarks/public/results`` is
                # ignored) — better to under-prune than hide real code.
                if "/" not in s:
                    names.add(s)
    except Exception:
        pass
    return frozenset(names)


def _prune_dir_names(cwd: str) -> frozenset[str]:
    """Directory names to skip during search = universal artifacts ∪ whatever
    this repo's .gitignore declares. Repo-driven, not tuned to any project."""
    return _UNIVERSAL_IGNORE_DIRS | _gitignored_dir_names(os.path.realpath(cwd))


def _first_unquoted(s: str, chars: str) -> int:
    """Index of the first char from ``chars`` that is OUTSIDE single/double
    quotes, or -1. Lets us split a pipeline without tripping on quoted args."""
    quote: str | None = None
    for i, c in enumerate(s):
        if quote:
            if c == quote:
                quote = None
        elif c in ("'", '"'):
            quote = c
        elif c in chars:
            return i
    return -1


def _optimize_find(command: str) -> tuple[str, bool]:
    """Inject ``-prune`` for heavy dirs into a top-level ``find`` so it doesn't
    descend into .venv/results/etc. Returns ``(command, changed)``.

    Conservative — only rewrites when: the command starts with ``find``, it
    hasn't already pruned/excluded those dirs, and it has no ``-exec``/
    ``-delete`` (which we leave entirely alone). ``-prune`` actually skips the
    subtree (unlike ``| grep -v`` / ``-not -path``, which still walk it).
    """
    s = command.strip()
    if not re.match(r"find(\s|$)", s):
        return command, False
    low = s.lower()
    # Leave it alone if the model already prunes, or if it mutates (-exec/-delete).
    if "-prune" in low or "-exec" in low or "-delete" in low:
        return command, False
    # Split off a trailing pipeline (| / ; / > / &) so prune lands inside find.
    # Scan for the operator OUTSIDE quotes so a quoted ``-name "a|b"`` isn't
    # mis-split (which would corrupt the command / abort the optimization).
    idx = _first_unquoted(s, "|;>&")
    head, tail = (s[:idx].rstrip(), s[idx:]) if idx >= 0 else (s, "")
    try:
        toks = shlex.split(head)
    except ValueError:
        return command, False
    if not toks or toks[0] != "find":
        return command, False
    i = 1
    paths: list[str] = []
    while i < len(toks) and not toks[i].startswith("-") and toks[i] not in ("(", ")", "!"):
        paths.append(toks[i])
        i += 1
    paths = paths or ["."]
    expr = toks[i:]
    prune: list[str] = []
    for d in sorted(_prune_dir_names(os.getcwd())):
        prune += ["-path", f"*/{d}/*", "-prune", "-o"]
    rebuilt = ["find", *paths, *prune]
    rebuilt += (["(", *expr, ")", "-print"] if expr else ["-print"])
    new_head = " ".join(shlex.quote(t) for t in rebuilt)
    return new_head + (" " + tail if tail else ""), True


@tool
async def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a text file or IMAGE from the LOCAL working directory.

    IMAGES: Pass an image file path (png/jpg/jpeg/gif/bmp/tif/tiff/webp) and the
    tool SEES it — diagrams, screenshots, figures, charts, photos, scanned pages
    are transcribed straight into the reply via Vision VLM. Use this to inspect
    any image attached by the user or generated in the workspace.

    Text files: Reads text with 1-indexed line numbers.
    Relative paths resolve against the working directory.

    Args:
        path: file or image path.
        start_line: optional 1-indexed start line for text files (0 = beginning).
        end_line: optional 1-indexed end line for text files (0 = end of file).
    """
    raw = os.path.expanduser((path or "").strip())
    if not raw:
        return "Error: file path is required."
    cwd = os.path.realpath(os.getcwd())

    candidates = []
    first = raw if os.path.isabs(raw) else os.path.join(cwd, raw)
    candidates.append(os.path.realpath(first))
    if os.path.isabs(raw):  # graceful: '/README.md' → '<cwd>/README.md'
        candidates.append(os.path.realpath(os.path.join(cwd, raw.lstrip("/"))))

    target = next((c for c in candidates if os.path.isfile(c)), None)
    if target is None:
        return f"Error: file not found: {raw}"

    staging = os.path.realpath(os.environ.get("APODEX_INPUT_STAGING_DIR") or os.path.expanduser("~/.apodex-inputs"))
    is_in_cwd = target == cwd or target.startswith(cwd + os.sep)
    is_in_staging = target == staging or target.startswith(staging + os.sep)

    if not (is_in_cwd or is_in_staging):
        return f"Error: '{raw}' is outside the working directory; reads are limited to {cwd}."

    ext = os.path.splitext(target)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}:
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/tiff" if ext in (".tif", ".tiff") else f"image/{ext.lstrip('.')}")
        try:
            with open(target, "rb") as fb:
                data = fb.read()
            from plugins.tools._reader_core import _vision_read
            vt = _vision_read(data, mime)
        except Exception:
            vt = None
        name = os.path.basename(target)
        if vt:
            return f"<!-- image {name} | read via vision -->\n\n{vt}"
        return (f"file: {name}\ntype: {mime}\n"
                "note: image — vision unavailable (set READDOC_VISION_URL/MODEL/KEY); not read.")

    try:
        with open(target, "rb") as fb:
            if b"\x00" in fb.read(8192):
                return f"Error: '{raw}' appears to be a binary file; not reading as text."
        with open(target, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        return f"Error reading {raw}: {exc}"

    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return "(empty file)"
    s = max(0, start_line - 1) if start_line > 0 else 0
    e = min(end_line, total) if end_line > 0 else total
    shown = lines[s:e]
    if not shown:
        return f"(no lines in range {start_line}-{end_line}; file has {total} lines)"
    # cat -n style numbering so the model can target edits precisely by line.
    numbered = "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(shown, start=s + 1))
    header = f"[lines {s + 1}-{e} of {total}]\n" if (s or e < total) else ""
    out = header + numbered
    if len(out) > _MAX_READ_BYTES:
        # Don't silently drop the tail — tell the model how to page (Claude Code
        # found a short "use offset/limit" error beats a large truncated blob).
        out = out[:_MAX_READ_BYTES] + (
            f"\n\n… [truncated at {_MAX_READ_BYTES} chars — {total} lines total. "
            "Re-read a narrower range with start_line/end_line to see the rest.]"
        )
    return out


@tool
async def bash(command: str, description: str = "") -> str:
    """Run a shell command on the LOCAL machine in the current working directory.

    Use for builds, tests, git, running scripts, and inspecting the repo.
    Runs on the host (not a remote sandbox), so it sees exactly the files the
    edit tools modify. Prefer non-interactive flags.

    Args:
        command: the shell command to run.
        description: a clear, concise description of what this command does, in
            active voice (5-10 words for simple commands; add context for piped
            or obscure ones). This is shown to the user at the approval prompt so
            they can see the command's intent at a glance — always provide it.
            Examples: ``ls -la`` → "List files in the current directory";
            ``git reset --hard origin/main`` → "Discard local changes and match
            remote main"; ``curl -s url | jq '.data[]'`` → "Fetch JSON and
            extract the data array".
    """
    cmd = (command or "").strip()
    if not cmd:
        return "Error: empty command."

    # Hard backstop: refuse denylisted commands regardless of the approval
    # gate (mirrors plugins.tools.bash, fail-closed). Assess the ORIGINAL cmd.
    assessment = assess_bash_command(cmd)
    if assessment.level == "deny":
        return f"Error: refused — {assessment.reason}"

    # Auto-prune heavy dirs from a bare `find` so it doesn't crawl
    # .venv/results/etc (~50s → <1s). No-op for non-find commands.
    cmd, pruned = _optimize_find(cmd)
    prune_note = (
        "[note: auto-pruned .venv/results/build dirs from find for speed; "
        "remove the prune clauses if you meant to search them]\n"
        if pruned else ""
    )

    # Where this actually runs is decided once at startup (bubblewrap jail on
    # Linux, the container on macOS, or the host after an explicit opt-in) —
    # see :mod:`apodex.sandbox`.
    from apodex.sandbox import active_strategy, run_shell
    try:
        rc, stdout, stderr = await run_shell(
            cmd, os.getcwd(), _BASH_TIMEOUT, active_strategy(),
        )
    except TimeoutError:
        return f"Error: command timed out after {_BASH_TIMEOUT}s."
    except Exception as exc:
        return f"Error: {exc}"

    result = stdout
    if stderr.strip():
        result += BASH_STDERR_SEPARATOR + stderr
    if rc != 0:
        result += f"\n[exit code {rc}]"
    return prune_note + (result if result.strip() else "(no output)")


def _walk_pruned(base: str) -> Iterator[tuple[str, list[str]]]:
    """os.walk under ``base``, pruning (in place) the universal artifact dirs
    plus whatever the repo's .gitignore declares — repo-driven, not hardcoded.
    """
    prune = _prune_dir_names(base)
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in prune]
        yield root, files


@tool
async def glob_search(pattern: str, path: str = ".", max_results: int = 200) -> str:
    """Find files by glob, skipping artifact dirs the repo ignores (fast, local).

    The shared glob_search crawls the whole tree (very slow when a gitignored
    dir holds many files); this prunes universal artifact dirs + the repo's own
    .gitignore'd dirs first. Supports ``*.py`` / ``**/*.py``-style patterns,
    matched against the path relative to ``path``.

    Args:
        pattern: glob, e.g. ``*.py`` or ``workflows/**/*.py``.
        path: directory to search under (default cwd).
        max_results: cap on returned paths.
    """
    import fnmatch

    base = os.path.realpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
    pat = pattern.strip()
    # '**/*.py' → '*.py' (basename pattern). Use prefix removal, NOT lstrip,
    # which would strip every leading '*'/'/' and turn '**/*.py' into '.py'
    # (so root-level files like 'x.py' would never match).
    bare = pat[3:] if pat.startswith("**/") else pat
    matches: list[str] = []
    for root, files in _walk_pruned(base):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), base)
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f, bare) or fnmatch.fnmatch(rel, bare):
                matches.append(rel)
        if len(matches) >= max_results:
            break
    if not matches:
        return f"No files match {pattern!r} under {path}"
    uniq = sorted(set(matches))
    body = "\n".join(uniq[:max_results])
    if len(uniq) > max_results:
        body += (f"\n… [showing {max_results} of {len(uniq)}+ files — narrow the "
                 "pattern or raise max_results to see the rest]")
    return body


@tool
async def grep_search(
    pattern: str, path: str = ".", glob: str = "",
    context_lines: int = 0, max_results: int = 100,
) -> str:
    """Search file CONTENTS by regex, skipping artifact dirs (fast, local).

    The shared grep_search crawls the whole tree (very slow with a large
    gitignored dir); this prunes universal artifact dirs + the repo's own
    .gitignore'd dirs, and skips binary files.

    Args:
        pattern: Python regex.
        path: directory to search under (default cwd).
        glob: optional filename glob filter, e.g. ``*.py``.
        context_lines: lines of context to show before & after each match.
        max_results: cap on matching/context lines.
    """
    import fnmatch
    import re as _re

    base = os.path.realpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
    try:
        rx = _re.compile(pattern)
    except _re.error as exc:
        return f"Error: invalid regex {pattern!r}: {exc}"
    n_ctx = max(0, int(context_lines))
    capped = f"\n… [capped at {max_results} lines — narrow the pattern/path or raise max_results]"
    out: list[str] = []
    for root, files in _walk_pruned(base):
        for f in files:
            if glob and not fnmatch.fnmatch(f, glob):
                continue
            full = os.path.join(root, f)
            try:
                if os.path.getsize(full) > _MAX_GREP_FILE_BYTES:
                    continue  # skip very large files (data dumps / minified bundles)
                with open(full, "rb") as fb:
                    if b"\x00" in fb.read(4096):
                        continue  # binary
                with open(full, encoding="utf-8", errors="replace") as fh:
                    flines = fh.read().splitlines()
            except Exception:
                continue
            rel = os.path.relpath(full, base)
            for i, line in enumerate(flines, 1):
                if not rx.search(line):
                    continue
                if n_ctx:
                    lo, hi = max(1, i - n_ctx), min(len(flines), i + n_ctx)
                    for j in range(lo, hi + 1):
                        sep = ":" if j == i else "-"
                        out.append(f"{rel}:{j}{sep} {flines[j - 1].rstrip()[:200]}")
                    out.append("--")
                else:
                    out.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                if len(out) >= max_results:
                    return "\n".join(out) + capped
    return "\n".join(out) if out else f"No matches for {pattern!r} under {path}"


@tool
async def delete_file(path: str) -> str:
    """Delete a file in the working directory.

    Local + cwd-scoped (cannot delete outside the working directory). This is
    a first-class, **revertable** delete — the session journals the file before
    removal, so `/revert` can restore it — preferred over `bash rm`.

    Args:
        path: file path (relative to the working directory).
    """
    raw = (path or "").strip()
    if not raw:
        return "Error: file path is required."
    cwd = os.path.realpath(os.getcwd())
    target = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(cwd, raw))
    if not (target == cwd or target.startswith(cwd + os.sep)):
        return f"Error: '{raw}' is outside the working directory; refusing to delete."
    if not os.path.exists(target):
        return f"Error: file not found: {raw}"
    if os.path.isdir(target):
        return f"Error: '{raw}' is a directory; this tool only deletes files."
    try:
        os.remove(target)
    except Exception as exc:
        return f"Error deleting {raw}: {exc}"
    return f"Deleted {raw}"


__all__ = ["bash", "delete_file", "glob_search", "grep_search", "read_file"]
