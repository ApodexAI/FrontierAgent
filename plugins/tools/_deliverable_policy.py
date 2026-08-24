"""Per-task policy for writes to the collected ``/outputs`` directory.

Controls file publishing, task outputs, and intermediate artifact isolation.
Ensures sub-agents write deliverables only to authorized manifest paths and
intermediate scratch files to ``/outputs/scratch``.
"""

from __future__ import annotations

import contextvars
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass

from plugins.tools._bash_policy import (
    redirection_token,
    strip_command_prefixes,
    tokenize_shell_segment,
)


@dataclass(frozen=True)
class _Policy:
    """Scoped publishing rights for the current sub-agent task."""

    manifest: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()


_UNSET = object()
_policy_var: contextvars.ContextVar[object | _Policy] = contextvars.ContextVar(
    "mh_deliverable_policy", default=_UNSET
)

# ``/outputs/scratch`` is the agreed cross-round intermediate area: persisted
# and remounted with the rest of ``/outputs`` but NOT a deliverable (whatever
# collects deliverables never publishes it). Writes there are allowed for publisher
# and non-publisher alike, gated only by a size quota so a run cannot grow the
# persisted volume without bound. The quota is enforced at write time with an
# error the model can act on (delete files, then retry).
_SCRATCH_DIR = "/outputs/scratch"
_SCRATCH_QUOTA_ENV = "FRONTIER_AGENT_SCRATCH_QUOTA_BYTES"
_DEFAULT_SCRATCH_QUOTA_BYTES = 536_870_912  # 512 MiB


def _runtime_outputs_root() -> str:
    """Return the filesystem output root visible to model tools.

    Container and bwrap runs expose the canonical ``/outputs`` mount. Native
    mode cannot create that top-level mount on hosts such as macOS (the root
    filesystem is read-only), so its tools must use the run-local host path
    exported by ``prepare_native_runtime`` instead.
    """
    backend = os.environ.get("SANDBOX_BACKEND", "").strip().lower()
    if backend != "native" and os.environ.get("APODEX_IN_NATIVE") != "1":
        return "/outputs"
    configured = os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "").strip()
    if not configured or not os.path.isabs(configured):
        return "/outputs"
    return os.path.normpath(configured)


def _runtime_workspace_root() -> str:
    """Return the filesystem workspace root visible to model tools."""
    if _runtime_outputs_root() == "/outputs":
        return "/workspace"
    configured = os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "").strip()
    if not configured or not os.path.isabs(configured):
        return "/workspace"
    return os.path.normpath(configured)


def _canonical_output_path(path: str) -> str:
    """Map a native physical output path into the ``/outputs`` namespace."""
    normalised = os.path.normpath(str(path or "").strip())
    root = _runtime_outputs_root()
    if root == "/outputs":
        return normalised
    if normalised == root:
        return "/outputs"
    prefix = root + os.sep
    if normalised.startswith(prefix):
        relative = normalised[len(prefix):]
        return os.path.normpath(f"/outputs/{relative}")
    return normalised


def _runtime_output_path(path: str) -> str:
    """Render one canonical output path as a path tools can actually open."""
    canonical = _normalise_output_path(path)
    root = _runtime_outputs_root()
    if root == "/outputs":
        return canonical
    return os.path.join(root, canonical.removeprefix("/outputs/"))


def _canonicalise_output_references(text: str) -> str:
    """Canonicalise native-root mentions before applying shell policy regexes."""
    root = _runtime_outputs_root()
    if root == "/outputs" or root not in text:
        return text
    # Component-aware suffix boundary: do not rewrite a sibling such as
    # ``.../outputs-old`` into the protected namespace.
    pattern = re.compile(
        re.escape(root) + r"(?=$|[/\s'\";|&<>=:(),])"
    )
    return pattern.sub("/outputs", text)

_OUTPUT_PATH_RE = re.compile(
    r"(?<![^\s'\"=:;|&<>(,])/outputs(?:/[^\s'\";|&<>]+)?"
)
# Verbs whose whole purpose is to produce or remove files. Read-only tools stay
# out of this set; ``cp``/``rsync`` are additionally allowed when /outputs only
# appears as a source operand.
_MUTATING_VERBS = frozenset({
    "cp", "mv", "mkdir", "rmdir", "touch", "rm", "unlink", "install",
    "truncate", "chmod", "chown", "tee", "dd", "ln", "rsync", "split",
    "convert", "magick", "ffmpeg", "zip", "unzip", "tar", "gzip", "gunzip",
    "bzip2", "xz", "pandoc", "wkhtmltopdf", "jupyter", "curl", "wget",
})
# Converters whose OUTPUT is a positional operand rather than a flag value, so
# neither ``_REDIRECT_TO_OUTPUTS_RE`` nor ``_OUTPUT_FLAG_RE`` sees it. They are
# resolved by operand position (``_converter_write_kind``) rather than listed in
# ``_MUTATING_VERBS``, because the verb alone cannot answer the question: the
# same binary reads /outputs when the directory appears as its INPUT
# (``pdftotext /outputs/final.pdf /workspace/x.txt``) and writes it when the
# directory is its output operand.
#
# ``True`` marks a PREFIX emitter: the tool appends its own ``-1.png`` to the
# operand, so the path the guard is shown is never a path the tool creates.
# Matching it against the manifest approves a file that will not exist while
# the real, undeclared one lands beside it — those are refused outright, and
# the /workspace-then-copy shape is the way through.
_POSITIONAL_OUTPUT_VERBS: dict[str, bool] = {
    "pdftotext": False,
    "ssconvert": False,
    "xlsx2csv": False,
    "pdftoppm": True,
    "pdfimages": True,
    "pdftocairo": False,  # mode-dependent, see _PDFTOCAIRO_PREFIX_FLAGS
}
# ``pdftocairo`` is the one mode-dependent member: image output takes a prefix
# root, while ``-pdf``/``-ps``/``-eps``/``-svg`` write exactly the file named.
_PDFTOCAIRO_PREFIX_FLAGS = frozenset({"-png", "-jpeg", "-jpg", "-tiff"})
# Verbs that derive a SIBLING output next to their input when no output operand
# is given: ``pdftotext /outputs/final.pdf`` silently creates
# ``/outputs/final.txt``. Also a derived path, so also refused.
_SIBLING_OUTPUT_VERBS = frozenset({"pdftotext"})
# Flags that turn a converter into a stdout reporter with no output operand.
_REPORT_ONLY_FLAGS: dict[str, frozenset[str]] = {
    "pdfimages": frozenset({"-list"}),
}
# What counts as a positional OPERAND of a converter, as opposed to the value
# of an option. Anchored on path shape — absolute, explicitly relative, home,
# or the ``-`` that means stdin/stdout — rather than "any token without a
# leading dash".
#
# That anchoring is what makes per-verb option arity unnecessary. Treating
# every non-flag token as an operand meant ``-r 72`` contributed a bare ``72``
# that displaced the real output operand, and the only way to know ``-r``
# consumes a value is a table of every option of every one of these tools. A
# value that is not path-shaped is simply not a candidate, whatever option it
# belongs to.
#
# It errs closed: an operand this misses (a bare relative path such as
# ``out.txt``) makes the /outputs token look like the output, which refuses the
# command. An operand it invents would do the opposite — let a write through —
# so the pattern deliberately excludes the value shapes those tools use
# (numbers, enums like ``-eol unix``, ``key=value`` in ``-jpegopt``). A lone
# ``-`` is an operand; ``-png`` is a flag.
_OPERAND_PATH_RE = re.compile(r"^(?:~|\.{1,2}/|/)")
# Fallback for segments that do not survive ``shlex`` (unbalanced quotes).
_MUTATING_COMMAND_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cp|mv|mkdir|rmdir|touch|rm|install|truncate|chmod|tee)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Commands that only ever read their /outputs operand. ``sed`` and ``find`` are
# handled separately because a flag flips them into mutating tools.
_READ_ONLY_VERBS = frozenset({
    "cat", "ls", "head", "tail", "file", "stat", "wc", "diff", "cmp", "grep",
    "egrep", "fgrep", "rg", "ag", "du", "tree", "nl", "strings", "xxd", "od",
    "base64", "md5sum", "sha1sum", "sha256sum", "sha512sum", "shasum", "cksum",
    "basename", "dirname", "realpath", "readlink", "jq", "yq", "identify",
    "ffprobe", "pdfinfo", "echo", "printf", "less", "more", "test", "[",
    "awk", "column", "uniq", "sort", "tr", "cut", "fold", "true", "false",
})
_INTERPRETER_VERBS = frozenset({
    "python", "python2", "python3", "ipython", "node", "deno", "bun", "ruby",
    "perl", "rscript",
})
_INTERPRETER_VERSION_RE = re.compile(r"^python3(?:\.\d+)?$")
_REDIRECT_TO_OUTPUTS_RE = re.compile(r">>?\s*['\"]?/outputs(?:/|\b)")
# ``--out /outputs/x`` / ``-o=/outputs/x``: the originating bug's real shape —
# a plotting or conversion script handed an output path as a flag value.
_OUTPUT_FLAG_RE = re.compile(
    r"(?:^|\s)(?:-o|-O|--out|--output|--outfile|--output-file|--output-dir|"
    r"--outdir|--output-path|--save|--save-to|--save-as|--dest|--destination|"
    r"--write-to)[=\s]+['\"]?/outputs(?:/|\b)",
    re.IGNORECASE,
)
_WRITE_OPEN_RE = re.compile(
    r"(?<![\w.])open\s*\([^,\n]+,\s*(?:mode\s*=\s*)?['\"][^'\"]*[wax+]",
    re.IGNORECASE,
)
_SED_IN_PLACE_RE = re.compile(
    r"(?:^|[;&|]\s*)sed\b[^\n;&|]*\s-(?:[A-Za-z]*i[A-Za-z]*|i(?:\s|$))",
    re.IGNORECASE | re.MULTILINE,
)
_FIND_MUTATES_RE = re.compile(r"-(?:delete|exec|execdir|ok|okdir|fprint|fls|fprintf)\b")
# Any path under /outputs counts, not just the root: a shell sitting in a
# subdirectory (scratch included) can write ``../`` relative targets the
# guard never sees, so ``cd /outputs/scratch && cp x ../leak.png`` must be
# caught the same as ``cd /outputs``.
_CD_OUTPUTS_RE = re.compile(
    r"\b(?:cd|pushd)\s+"
    r"(?:(?:--|-[A-Za-z]+|[+-]\d+)\s+)*"
    r"['\"]?/outputs(?:/[^\s'\";|&<>]*)?['\"]?(?:\s|$)"
)
# ``(?<!<)`` keeps the second ``<`` of a ``<<<`` herestring from reading as an
# opener: its delimiter would never terminate, swallowing the rest of the
# command — including the segment that does the writing — into one segment.
_HEREDOC_OPENER_RE = re.compile(
    r"(?<!<)<<-?\s*(?:'(?P<single>[^']*)'|\"(?P<double>[^\"]*)\"|"
    r"(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
# A real heredoc opener, excluding the ``<<<`` herestring.
_HEREDOC_MARK_RE = re.compile(r"(?<!<)<<(?!<)")
_LEADING_WORD_RE = re.compile(r"\s*([\w./+-]+)")
_INLINE_CODE_FLAG_RE = re.compile(r"(?:^|\s)-(?:c|e)(?=[\s'\"]|$)|(?:^|\s)--(?:command|eval)\b")
# Library calls that write. ``.save(`` covers PIL/openpyxl; the false positive
# (reading /outputs then saving elsewhere) is intentional — candidates belong
# in /workspace either way.
# Link-creation calls get their own tuple: besides counting as writes, a link
# whose target is under scratch is refused outright (``_scratch_link_error``).
_LINK_API_MARKERS = (
    "os.symlink", "os.link", ".symlink_to(", ".hardlink_to(",
)
_WRITE_API_MARKERS = (
    ".savefig(", ".save(", ".to_csv(", ".to_excel(", ".to_json(",
    ".to_parquet(", ".write_text(", ".write_bytes(", ".mkdir(", ".touch(",
    ".unlink(", ".rename(", "shutil.copy", "shutil.move", "os.remove",
    "os.rename", "os.replace", "os.unlink", "os.rmdir", "os.makedirs",
    *_LINK_API_MARKERS,
)

_SCRATCH_LINK_ERROR = (
    "Creating links under /outputs/scratch is blocked: a link could resolve "
    "outside scratch and would not survive the round trip through storage. "
    "Copy the file instead (`cp <src> /outputs/scratch/<name>`)."
)

_SCRATCH_ROOT_WRITE_ERROR = (
    "/outputs/scratch must remain a directory. Write files to an explicit "
    "child path such as /outputs/scratch/<name>; only mkdir and cleanup "
    "operations may target the directory itself."
)

_SPILL_WRITE_ERROR = (
    "That is a read-only recovery store. Read it with read_file or "
    "grep_search when older detail is genuinely needed; never create, "
    "overwrite, move, delete, chmod, or redirect output into it."
)


def _is_spill_path(path: str) -> bool:
    """Whether a write target enters the reserved recovery store.

    Delegates to :func:`plugins.tools._sandbox.is_spill_path`, which keys off the
    store's real root rather than a fixed ``.spill`` directory name.
    """
    from plugins.tools._sandbox import is_spill_path

    return is_spill_path(path)


def spill_write_error(path: str) -> str | None:
    """Return the recovery-store refusal for a write target, if applicable."""
    return _SPILL_WRITE_ERROR if _is_spill_path(path) else None
# ``/outputs/scratch/`` with a trailing slash and no child name is an
# unambiguous write INTO the directory (``cp x /outputs/scratch/`` keeps the
# source basename; the shell errors rather than creating a file if scratch is
# not a directory). Only the bare ``/outputs/scratch`` risks silently replacing
# the reserved directory with a regular file, so the trailing-slash form is not
# a root write — it is a normal scratch child write, quota-gated like any other.
# The lookbehind mirrors ``_OUTPUT_PATH_RE`` so a mid-token match cannot fire.
_SCRATCH_DIR_AS_DIR_RE = re.compile(
    r"(?<![^\s'\"=:;|&<>(,])/outputs/scratch/(?=$|[\s'\";|&<>])"
)

_NON_PUBLISHER_WRITE_ERROR = (
    "This command looks like it writes to /outputs, which is blocked: "
    "this task is not a publisher. (Reading /outputs is allowed — only "
    "writes are restricted.) Put candidates and all intermediate files "
    "under /workspace; only an explicit publish task may write /outputs."
)
_NON_PUBLISHER_UNVERIFIABLE_ERROR = (
    "This command names /outputs in a shape that cannot be verified as "
    "read-only, so it was blocked: this task is not a publisher. Reading "
    "/outputs is allowed — use read_file / grep_search, or a plain reader "
    "such as `cat`, `ls`, `head`, `sed -n`, `cp /outputs/x /workspace/x`, or "
    "an inline `python3 -c` snippet that only reads. Anything you create, "
    "move, or delete belongs under /workspace."
)
_NON_PUBLISHER_CD_ERROR = (
    "Do not `cd /outputs`: relative writes there cannot be validated and this "
    "task is not a publisher. Read /outputs through absolute paths instead, "
    "and keep everything you produce under /workspace."
)
_PUBLISHER_CD_ERROR = (
    "Publish commands must use the absolute declared output file paths; "
    "do not `cd /outputs` because relative writes cannot be validated."
)
_PUBLISHER_DERIVED_PATH_ERROR = (
    "This command's /outputs operand is a PREFIX the tool appends its own "
    "suffix to (`pdftoppm`, `pdftocairo -png`, `pdfimages` write "
    "`<prefix>-1.png`), so the file it actually creates is never the path "
    "given here and cannot be checked against the manifest. Render under "
    "/workspace, then copy the file you want to its exact declared /outputs "
    "path: `pdftoppm -png in.pdf /workspace/page && cp /workspace/page-1.png "
    "/outputs/<declared-name>.png`."
)
_PUBLISHER_DIR_ONLY_ERROR = (
    "Publish commands must name an absolute declared output file, not "
    "the /outputs directory. Converters that cannot be given an exact output "
    "FILE — because they take a DIRECTORY (`soffice --outdir`, `unzip -d`) or "
    "a PREFIX they append their own suffix to (`pdftoppm`, `pdftocairo`) — "
    "must write under /workspace first, then copy the result to its exact "
    "declared /outputs path: `soffice --headless --convert-to pdf --outdir "
    "/workspace in.pptx && cp /workspace/in.pdf /outputs/<declared-name>.pdf`."
)


def _normalise_output_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw.startswith("/outputs/"):
        raise ValueError(
            f"declared output path must be an absolute file under /outputs: {raw!r}"
        )
    normalised = os.path.normpath(raw)
    if normalised == "/outputs" or not normalised.startswith("/outputs/"):
        raise ValueError(f"invalid declared output path: {raw!r}")
    if any(ch in normalised for ch in "*?[]{}"):
        raise ValueError(f"output path must not contain glob characters: {raw!r}")
    return normalised


def normalise_output_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Validate and de-duplicate a publishing task's output manifest."""
    result: list[str] = []
    for raw in paths:
        path = _normalise_output_path(str(raw))
        if path not in result:
            result.append(path)
    return tuple(result)


def set_deliverable_write_paths(
    paths: Iterable[str] | None,
    retired_paths: Iterable[str] | None = None,
) -> contextvars.Token:
    """Scope output writes.

    ``None`` preserves legacy behavior for workflows that do not opt in.
    An empty iterable means read-only ``/outputs``.  A non-empty iterable is
    the exact publishing manifest for the current task.  ``retired_paths`` are
    superseded manifest entries the publisher may delete or move out of
    ``/outputs`` (but not rewrite).
    """
    if paths is None:
        return _policy_var.set(_UNSET)
    manifest = normalise_output_paths(paths)
    retired = tuple(
        path
        for path in normalise_output_paths(retired_paths or ())
        if path not in manifest
    )
    return _policy_var.set(_Policy(manifest=manifest, retired=retired))


def reset_deliverable_write_paths(token: contextvars.Token) -> None:
    _policy_var.reset(token)


# ---------------------------------------------------------------------------
# Denial escalation
#
# A blocked /outputs write is reported to the sub-agent that attempted it, and
# a sub-agent cannot grant itself permission. Measured over the 267-task APEX
# agent_team run: 234/267 trials hit that denial, and the usual next move was to
# write the deliverable somewhere else (mostly /outputs/scratch) and report
# success -- so the coordinator, the one party that CAN fix it by dispatching a
# publish task, was the only one never told. The log below carries those
# denials out to whoever assembles the sub-agent's result.
#
# Deliberately not recorded: scratch quota and scratch-root errors. Those are
# self-correctable by the agent that hit them and cost no deliverable.
# ---------------------------------------------------------------------------

_denial_log_var: contextvars.ContextVar[list[tuple[str, str]] | None] = (
    contextvars.ContextVar("mh_output_write_denials", default=None)
)


def new_output_write_denial_log() -> tuple[contextvars.Token, list[tuple[str, str]]]:
    """Start recording blocked ``/outputs`` writes for the current task.

    Returns the reset token and the log itself. Hold on to the list: the
    recording context is usually gone by the time the result is assembled, but
    every nested context mutates this same object.
    """
    log: list[tuple[str, str]] = []
    return _denial_log_var.set(log), log


def reset_output_write_denial_log(token: contextvars.Token) -> None:
    _denial_log_var.reset(token)


def _record_output_write_denial(kind: str, target: str) -> None:
    log = _denial_log_var.get()
    if log is None:
        return
    entry = (kind, target)
    if entry not in log:
        log.append(entry)


def render_denial_escalation(denials: Iterable[tuple[str, str]]) -> str:
    """The note prepended to a report whose agent was blocked from /outputs."""
    entries = tuple(dict.fromkeys(denials))
    if not entries:
        return ""
    unverifiable = tuple(
        entry for entry in entries if entry[0] == "unverifiable"
    )
    blocked_writes = tuple(
        entry for entry in entries if entry[0] != "unverifiable"
    )

    def _paths(items: Iterable[tuple[str, str]]) -> str:
        return ", ".join(
            f"`{_runtime_output_path(target)}`"
            if target.startswith("/outputs")
            else f"`{target}`"
            for _kind, target in items
        )

    notes: list[str] = []
    if blocked_writes and all(
        kind == "not_publisher" for kind, _target in blocked_writes
    ):
        paths = _paths(blocked_writes)
        notes.append(
            f"[BLOCKED WRITE: this agent tried to write {paths} and is not the "
            f"publisher for this run, so the write was refused. Nothing it "
            f"produced reached {_runtime_outputs_root()} — treat any claim in "
            f"the report below that a file was published there as false, and "
            f"look for a {_runtime_workspace_root()} path instead. If its work "
            f"is the deliverable, assign a publish task with output_paths "
            f"covering it.]\n"
        )
    elif blocked_writes:
        # A publisher may well have written its declared manifest successfully
        # and been refused only on some path beside it, so this must not claim
        # that nothing landed.
        paths = _paths(blocked_writes)
        notes.append(
            f"[BLOCKED WRITE: this agent tried to write {paths}, which its "
            f"publishing manifest does not cover, so those writes were refused. "
            f"Those files are not in {_runtime_outputs_root()}; anything the "
            f"report below says about them is unpublished. Re-assign the "
            f"publisher with the manifest the deliverable actually needs, or "
            f"accept the files it was authorized to write.]\n"
        )
    if unverifiable:
        paths = _paths(unverifiable)
        notes.append(
            f"[BLOCKED OUTPUTS ACCESS: this agent ran a command referencing "
            f"{paths} in a form the policy could not verify as read-only, so "
            f"the command was refused. This does not establish that the agent "
            f"attempted a write or that a deliverable failed to publish. Use "
            f"read_file, grep_search, or a plainly read-only command to inspect "
            f"{_runtime_outputs_root()}.]\n"
        )
    return "".join(notes)


def _current_policy() -> _Policy | None:
    value = _policy_var.get()
    return None if value is _UNSET else value  # type: ignore[return-value]


def declared_output_paths() -> tuple[str, ...] | None:
    """Return ``None`` outside an opted-in workflow, otherwise the manifest."""
    policy = _current_policy()
    return None if policy is None else policy.manifest


def retired_output_paths() -> tuple[str, ...]:
    """Superseded paths the current publisher may remove from ``/outputs``."""
    policy = _current_policy()
    return () if policy is None else policy.retired


def _is_scratch_path(normalised: str) -> bool:
    """Whether an already-normalised path falls under ``/outputs/scratch``.

    ``os.path.normpath`` has resolved ``..`` before this is asked, so
    ``/outputs/scratch/../final.png`` arrives as ``/outputs/final.png`` and is
    correctly NOT scratch. Symlinks are not resolved — link creation under
    scratch is refused separately (``_scratch_link_error``).
    """
    return normalised == _SCRATCH_DIR or normalised.startswith(_SCRATCH_DIR + "/")


def scratch_quota_bytes() -> int:
    """Read the quota per call so ``.env`` values and monkeypatch stay live.

    Unparseable or non-positive values fall back to the default (matching
    ``deliverable_security._env_int``): a quota of ``0`` would make every
    scratch write fail forever with an error whose advice — delete files —
    cannot help.
    """
    try:
        value = int((os.environ.get(_SCRATCH_QUOTA_ENV) or "").strip())
    except ValueError:
        return _DEFAULT_SCRATCH_QUOTA_BYTES
    return value if value > 0 else _DEFAULT_SCRATCH_QUOTA_BYTES


def _scratch_usage_bytes() -> int:
    """Current logical byte size of scratch, including invalid root files."""
    scratch_dir = _runtime_output_path(_SCRATCH_DIR)
    try:
        root_stat = os.lstat(scratch_dir)
    except OSError:
        return 0
    if not stat.S_ISDIR(root_stat.st_mode):
        # A legacy or policy-bypassing write may have replaced the reserved
        # directory with a regular file. Count it rather than treating the
        # corrupt namespace as an empty scratch area.
        return root_stat.st_size
    total = 0
    for dirpath, _dirnames, filenames in os.walk(scratch_dir):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _format_size(size: int) -> str:
    if size >= 1 << 20:
        return f"{size / (1 << 20):.0f}MB"
    return f"{size} bytes"


def scratch_write_error() -> str | None:
    """Quota gate for a write that is about to be allowed into scratch."""
    quota = scratch_quota_bytes()
    used = _scratch_usage_bytes()
    if used < quota:
        return None
    return (
        f"scratch quota exceeded: limit {_format_size(quota)}, currently "
        f"{_format_size(used)} used — delete files you no longer need under "
        "/outputs/scratch before writing more"
    )


def output_write_error(path: str) -> str | None:
    """Return an actionable error when ``path`` is not publishable."""
    if error := spill_write_error(path):
        return error
    normalised = _canonical_output_path(path)
    if normalised == _SCRATCH_DIR:
        # Direct file tools cannot create a directory, so an exact-root write
        # would replace the reserved namespace with a hidden regular file.
        return _SCRATCH_ROOT_WRITE_ERROR
    if _is_scratch_path(normalised):
        # Scratch is shared persistent intermediate space, not a deliverable:
        # the manifest never applies to it, only the quota does. Checked
        # before the opt-in gate below because the quota holds in EVERY
        # workflow, not just agent_team.
        return scratch_write_error()
    manifest = declared_output_paths()
    if manifest is None:
        return None
    if (
        normalised == "/outputs" or normalised.startswith("/outputs/")
    ) and normalised not in manifest:
        if not manifest:
            _record_output_write_denial("not_publisher", normalised)
            return (
                "This task is not a publisher. Write candidates and all "
                "intermediate files under /workspace; only an explicit "
                "publish task may write /outputs."
            )
        if normalised in retired_output_paths():
            _record_output_write_denial("retired", normalised)
            return (
                f"{normalised!r} was superseded by a manifest replacement. "
                "Delete it or move it under /workspace instead of writing "
                f"it again; this publish task may write only: "
                f"{', '.join(manifest)}"
            )
        _record_output_write_denial("undeclared", normalised)
        return (
            f"Undeclared deliverable path {normalised!r}. This publish "
            f"task may write only: {', '.join(manifest)}"
        )
    return None


def _split_segments(command: str) -> tuple[str, ...]:
    """Split a command on unquoted separators, preserving raw text.

    Quote-aware so a ``;`` inside ``python3 -c "import os; ..."`` does not
    fragment the snippet — the write heuristics need to see inline code whole
    to tell a read from a write. Heredoc bodies stay attached to the segment
    that opened them, so "heredoc a script, then run it" leaves the *runner*
    segment opaque instead of marking every segment as visible code.
    """
    segments: list[str] = [""]
    pending: list[tuple[int, str, bool]] = []
    quote = ""
    escaped = False
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        index += 1
        if escaped:
            segments[-1] += char
            escaped = False
            continue
        if char == "\\" and quote != "'":
            segments[-1] += char
            escaped = True
            continue
        if quote:
            segments[-1] += char
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
            segments[-1] += char
            continue
        if char == "<" and command.startswith("<", index):
            # The pattern's lookbehind rejects a ``<<<`` herestring at every
            # one of its three positions, so this never opens a heredoc.
            opener = _HEREDOC_OPENER_RE.match(command, index - 1)
            if opener is not None:
                delimiter = (
                    opener.group("single")
                    or opener.group("double")
                    or opener.group("bare")
                )
                pending.append(
                    (len(segments) - 1, delimiter, opener.group(0)[2:3] == "-")
                )
                segments[-1] += opener.group(0)
                index = opener.end()
                continue
        if char == "\n" and pending:
            for owner, delimiter, strip_tabs in pending:
                body: list[str] = []
                while index < length:
                    end = command.find("\n", index)
                    if end == -1:
                        end = length
                    line = command[index:end]
                    index = end + 1
                    probe = line.lstrip("\t") if strip_tabs else line
                    if probe.strip() == delimiter:
                        break
                    body.append(line)
                if body:
                    segments[owner] += "\n" + "\n".join(body)
            pending.clear()
            segments.append("")
            continue
        if char in ";|&\n":
            segments.append("")
            continue
        segments[-1] += char
    return tuple(segment.strip() for segment in segments if segment.strip())


def _tokens(segment: str) -> list[str] | None:
    """Argv of ``segment``, starting at the command it really runs.

    Command prefixes are stripped here rather than at each caller, so every
    rule that reads a verb — the mutating/read-only/converter checks, the copy
    and removal analyses — sees ``cp`` in ``env cp …`` and ``timeout 60 cp …``.
    Without it the publisher guard skipped ``env cp /workspace/x.png
    /outputs/leak.png`` entirely: ``env`` is not a mutating verb, so the segment
    never reached the manifest check. That hole predates the converters and
    applied to ``cp`` / ``tee`` / ``convert`` alike.

    ``strip_command_prefixes`` is ``_bash_policy``'s own unwrapper, reused so
    the two policies cannot disagree about what a wrapped command runs. It
    subsumes the ``VAR=val`` stripping this function used to do alone.
    """
    try:
        tokens = tokenize_shell_segment(segment)
    except ValueError:
        return None
    return strip_command_prefixes(tokens)


def _head_verb(segment: str) -> str | None:
    """Command name of ``segment``, or ``None`` when it cannot be parsed."""
    tokens = _tokens(segment)
    if tokens is None:
        return None
    if not tokens:
        return ""
    return os.path.basename(tokens[0]).lower()


def _is_interpreter(verb: str) -> bool:
    return verb in _INTERPRETER_VERBS or bool(_INTERPRETER_VERSION_RE.match(verb))


def _referenced_output_paths(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _OUTPUT_PATH_RE.finditer(text):
        raw = match.group(0).rstrip(".,:)")
        path = os.path.normpath(raw)
        if path not in found:
            found.append(path)
    return tuple(found)


def _copy_only_reads_outputs(segment: str) -> bool:
    """Return whether a copy command uses ``/outputs`` only as input."""
    tokens = _tokens(segment)
    if not tokens:
        return False
    if os.path.basename(tokens[0]).lower() not in {"cp", "rsync"}:
        return False
    if any(
        token == "-t"
        or token.startswith("--target-directory")
        for token in tokens[1:]
    ):
        return False
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    if len(operands) < 2:
        return False
    return _OUTPUT_PATH_RE.search(operands[-1]) is None


def _converter_write_kind(segment: str) -> str | None:
    """How a positional-output converter touches ``/outputs``.

    ``"exact"`` — it writes the operand it was given, so the manifest check can
    decide. ``"prefix"`` — it writes a path DERIVED from the operand, which the
    manifest can never approve. ``None`` — not one of these converters, or
    /outputs appears only as an input it reads.

    Resolving by operand position is what separates the two directions of the
    same binary: ``pdftoppm -png /workspace/in.pdf /outputs/page`` writes, while
    ``pdftoppm -png /outputs/final.pdf /workspace/page`` renders a published
    deliverable into scratch and must stay allowed for publisher and
    non-publisher alike.

    These tools take at most two positionals, input then output, so the
    decision is simply whether ANOTHER operand follows the /outputs one. That
    holds however the command is spelled, which is what the earlier "``-`` means
    stdout" and "the output is ``operands[-1]``" shortcuts did not:
    ``pdftotext - /outputs/leak.txt`` puts stdin first and still writes, and
    ``pdftoppm in.pdf /outputs/page -r 72`` puts options last and still writes.
    """
    tokens = _tokens(segment)
    if not tokens:
        return None
    verb = os.path.basename(tokens[0]).lower()
    if verb not in _POSITIONAL_OUTPUT_VERBS:
        return None
    rest = tokens[1:]
    # ``pdfimages -list`` reports to stdout and takes no output operand.
    if any(flag in rest for flag in _REPORT_ONLY_FLAGS.get(verb, ())):
        return None
    prefix = _POSITIONAL_OUTPUT_VERBS[verb] or (
        verb == "pdftocairo"
        and any(token in _PDFTOCAIRO_PREFIX_FLAGS for token in rest)
    )
    operands = [
        token for token in rest
        if token == "-" or _OPERAND_PATH_RE.match(token)
    ]
    if not operands:
        return None
    # The LAST /outputs operand is the one whose role is in question; anything
    # naming /outputs before it is an input either way.
    last_output = max(
        (i for i, token in enumerate(operands) if _OUTPUT_PATH_RE.search(token)),
        default=None,
    )
    if last_output is None:
        return None
    if last_output < len(operands) - 1:
        # Another operand follows, so this one is the input. Includes
        # ``pdftotext /outputs/x.pdf -``, where the ``-`` IS the stdout output.
        return None
    if len(operands) < 2 and verb not in _SIBLING_OUTPUT_VERBS:
        # Sole operand and no derived sibling: the tool reports to stdout.
        return None
    # Sole operand for a sibling emitter (``pdftotext /outputs/final.pdf``
    # creates ``/outputs/final.txt``) is a derived path, like a prefix.
    if len(operands) < 2:
        return "prefix"
    return "prefix" if prefix else "exact"


def _visible_code(segment: str) -> bool:
    """Whether this segment's program text is readable by the heuristics.

    Inline ``-c`` / ``-e`` snippets and the heredoc body attached to this
    segment are visible, so a genuine read (``Image.open``, ``os.listdir``) can
    be recognised as such. A script *file* handed a ``/outputs`` argument is
    opaque — including when an earlier segment heredoc'd that script into
    place, which is the usual way a model produces a chart.
    """
    if _HEREDOC_MARK_RE.search(segment):
        return True
    return _INLINE_CODE_FLAG_RE.search(segment) is not None


def _definite_write_signal(segment: str) -> bool:
    """Whether ``segment`` demonstrably writes something."""
    if _REDIRECT_TO_OUTPUTS_RE.search(segment) or _OUTPUT_FLAG_RE.search(segment):
        return True
    if _WRITE_OPEN_RE.search(segment) or _SED_IN_PLACE_RE.search(segment):
        return True
    if any(marker in segment for marker in _WRITE_API_MARKERS):
        return True
    verb = _head_verb(segment)
    if verb is None:
        return bool(_MUTATING_COMMAND_RE.search(segment))
    if verb == "find":
        return bool(_FIND_MUTATES_RE.search(segment))
    if verb in {"cp", "rsync"}:
        return not _copy_only_reads_outputs(segment)
    if verb in _POSITIONAL_OUTPUT_VERBS:
        return _converter_write_kind(segment) is not None
    return verb in _MUTATING_VERBS


def _opaque_write_shape(segment: str) -> bool:
    """Whether ``segment`` runs an interpreter whose target cannot be read."""
    verb = _head_verb(segment)
    if verb is None or not _is_interpreter(verb):
        return False
    return not _visible_code(segment)


def _segment_reads_only(segment: str) -> bool:
    """Whether ``segment`` is a recognisably read-only use of ``/outputs``."""
    verb = _head_verb(segment)
    if verb is None:
        # Quoting broke somewhere later in the segment (a heredoc body with an
        # apostrophe, say). The command name is still readable, and every write
        # heuristic has already run over the raw text.
        match = _LEADING_WORD_RE.match(segment)
        verb = os.path.basename(match.group(1)).lower() if match else ""
    if not verb:
        return True
    if verb in _READ_ONLY_VERBS:
        return True
    if verb == "sed":
        return True  # in-place edits were already rejected as writes
    if verb == "find":
        return True  # mutating flags were already rejected as writes
    if verb in {"cp", "rsync"}:
        return _copy_only_reads_outputs(segment)
    if verb in _POSITIONAL_OUTPUT_VERBS:
        # Extracting FROM a published deliverable into /workspace is a read,
        # whatever the tool is capable of writing elsewhere.
        return _converter_write_kind(segment) is None
    if _is_interpreter(verb):
        return _visible_code(segment)
    return False


# Fallback for a command the tokenizer could not split; the token path above uses
# ``_is_spill_path``, which knows the store's real root. Only unambiguous store
# names are listed: the canonical mount (``/spill`` at a path boundary), the temp
# default's distinctive directory, and the legacy in-workspace name. A bare
# ``spill`` component is deliberately absent — it would refuse a legitimate write
# to the user's own ``spill/`` directory, and the run-directory form is covered by
# the token path in every command that can actually be parsed.
_SPILL_NAMES = r"(?:/spill|apodex-spill|\.spill)"
_SPILL_RAW_RE = re.compile(
    r"(?:^|[/\s'\"=])" + _SPILL_NAMES + r"(?=$|[/\s'\";|&<>])",
)
_SPILL_ASSIGNMENT_RE = re.compile(
    r"(?:^|\s)[A-Za-z_][A-Za-z0-9_]*=[^;|&\n]*" + _SPILL_NAMES,
)
_REDIRECT_TOKEN_RE = re.compile(
    r"^(?:\d+|&)?(?:>>|>\||>&|<>|>)(.*)$",
)


def _token_mentions_spill(token: str) -> bool:
    """Recognise a reserved component inside an argv or redirect token."""
    probe = str(token or "")
    if redirect_token := redirection_token(probe):
        redirect = _REDIRECT_TOKEN_RE.match(redirect_token)
        if redirect and redirect.group(1):
            probe = redirect.group(1)
    else:
        probe = probe.strip()
    return _is_spill_path(probe)


def _segment_mentions_spill(segment: str) -> bool:
    try:
        tokens = tokenize_shell_segment(segment)
    except ValueError:
        tokens = None
    if tokens is not None and any(
        _token_mentions_spill(token) for token in tokens
    ):
        return True
    return bool(_SPILL_RAW_RE.search(segment))


def _redirects_to_spill(segment: str) -> bool:
    try:
        tokens = tokenize_shell_segment(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    for index, token in enumerate(tokens):
        redirect = redirection_token(token)
        if redirect is None:
            continue
        match = _REDIRECT_TOKEN_RE.match(redirect)
        if not match:
            continue
        inline_target = match.group(1)
        if inline_target and _token_mentions_spill(inline_target):
            return True
        if (
            not inline_target
            and index + 1 < len(tokens)
            and _token_mentions_spill(tokens[index + 1])
        ):
            return True
    return False


def _copy_only_reads_spill(segment: str) -> bool:
    """Allow copying FROM recovery while refusing a spill destination."""
    tokens = _tokens(segment)
    if not tokens or _head_verb(segment) not in {"cp", "rsync"}:
        return False
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    if len(operands) < 2:
        return False
    return (
        any(_token_mentions_spill(token) for token in operands[:-1])
        and not _token_mentions_spill(operands[-1])
    )


_SPILL_CD_ERROR = (
    ".spill is a read-only recovery store, and changing directory into it "
    "would hide a later relative write from this check. Read it in place with "
    "absolute paths — read_file, grep_search, or cat/grep on the full "
    "/…/.spill/… path."
)

# Verbs ``_READ_ONLY_VERBS`` counts as reads because they only write when asked
# to — but which CAN be asked to, naming the target where a token scan cannot
# tell it apart from an input: inside their own program text
# (``awk '{print > "f"}'``), behind an output flag (``sort -o f``, ``yq -i``), or
# as a trailing operand (``uniq in out``, ``xxd in out``). The spill gate
# therefore withholds its read-only exemption from them and fails closed.
#
# This costs nothing real: recovery has direct ``read_file`` / ``grep_search`` /
# ``cat`` / ``grep`` forms, and the refusal names them. Scoped to this gate, so
# ``/outputs`` handling is unchanged.
#
# It matters most where no filesystem layer backs the promise up. The bwrap jail
# re-mounts the store read-only, and container mode leaves it owned by the
# harness at 0755 so a *different* tool uid cannot write it — but that uid is
# best-effort: with no tool account, or a non-root harness, ``CurrentSandbox``
# warns and runs model commands with the harness's own uid, and then this gate is
# the only thing standing between the model and its own recovery files.
_SPILL_OPAQUE_READ_VERBS = frozenset({"awk", "sort", "uniq", "xxd", "yq"})

_SPILL_OPAQUE_VERB_ERROR = (
    "{verb} can write to a path given in its own arguments or program text, so "
    "it is not accepted against the read-only .spill recovery store. Read it "
    "with read_file, grep_search, or plain cat/grep instead."
)


def _spill_write_signal(segment: str) -> bool:
    """``_definite_write_signal`` minus signals that only ever name ``/outputs``.

    A segment reaching this point neither redirects nor assigns into the store
    (both checked first), so a redirect or ``--output`` flag pointing at
    ``/outputs`` says nothing about the store: ``grep -n x
    /workspace/.spill/a/b.md > /outputs/scratch/hits.txt`` reads recovery and
    publishes elsewhere. Whether that ``/outputs`` write is permitted belongs to
    :func:`output_write_error`, which runs next; refusing it here also stated a
    false reason ("redirect output into it") and made bash-based recovery
    unusable. Every other write signal — in-place sed, ``find`` mutations, write
    APIs, mutating verbs — is evaluated exactly as before.
    """
    without_outputs_target = _OUTPUT_FLAG_RE.sub(
        " ", _REDIRECT_TO_OUTPUTS_RE.sub(" ", segment),
    )
    return _definite_write_signal(without_outputs_target)


def _spill_bash_write_error(command: str) -> str | None:
    """Fail closed on mutations of the agent-visible recovery namespace."""
    for segment in _split_segments(command):
        if not _segment_mentions_spill(segment):
            continue
        verb = _head_verb(segment)
        if (
            not verb
            or _SPILL_ASSIGNMENT_RE.search(segment)
            or _redirects_to_spill(segment)
        ):
            return _SPILL_WRITE_ERROR
        if verb in {"cd", "pushd"}:
            # Load-bearing: ``_split_segments`` breaks on ``&&``/``;``, so a
            # following ``echo x > b.md`` segment never mentions .spill and
            # would escape this gate entirely.
            return _SPILL_CD_ERROR
        if verb in {"cp", "rsync"}:
            if _copy_only_reads_spill(segment):
                continue
            return _SPILL_WRITE_ERROR
        if _spill_write_signal(segment) or _opaque_write_shape(segment):
            return _SPILL_WRITE_ERROR
        if verb in _SPILL_OPAQUE_READ_VERBS:
            return _SPILL_OPAQUE_VERB_ERROR.format(verb=verb)
        if _segment_reads_only(segment):
            continue
        # Unknown wrappers/scripts may mutate the path even when their argv
        # looks harmless. Recovery reads have direct read_file/grep/cat forms.
        return _SPILL_WRITE_ERROR
    return None


def _removed_output_paths(segment: str) -> tuple[str, ...]:
    """``/outputs`` paths this segment takes *out* of the directory.

    Retired manifest entries would otherwise be stranded: every write path to
    them stays blocked, so a format change would end the run with the old and
    the new deliverable both present.
    """
    tokens = _tokens(segment)
    if not tokens:
        return ()
    verb = os.path.basename(tokens[0]).lower()
    if verb not in {"rm", "unlink", "mv"}:
        return ()
    operands = [
        os.path.normpath(token)
        for token in tokens[1:]
        if not token.startswith("-")
    ]
    if not operands:
        return ()
    if verb == "mv":
        destination = operands[-1]
        # Moving *within* /outputs still creates a file there.
        if destination == "/outputs" or destination.startswith("/outputs/"):
            return ()
        sources = operands[:-1]
    else:
        sources = operands
    return tuple(
        path for path in sources if path.startswith("/outputs/")
    )


def _scratch_link_error(segment: str) -> str | None:
    """Refuse link creation touching scratch, in every role and workflow.

    Elsewhere in ``/outputs`` the watcher unlinks any symlink it sees, so a
    policy miss is caught downstream (the watcher does sweep links out of
    scratch, but only on its next scan). ``_is_scratch_path`` does not resolve
    links, so ``/outputs/scratch/up/x`` could silently land outside scratch —
    refusing both the ``ln`` verb and the visible link-creating API calls
    keeps opaque script bodies the only shape the sweep alone must cover.
    """
    if not any(
        _is_scratch_path(path) for path in _referenced_output_paths(segment)
    ):
        return None
    verb = _head_verb(segment)
    if verb in {"ln", "link"}:
        return _SCRATCH_LINK_ERROR
    if any(marker in segment for marker in _LINK_API_MARKERS):
        return _SCRATCH_LINK_ERROR
    return None


def _scratch_only_segment(segment: str) -> bool:
    """Whether every ``/outputs`` reference in ``segment`` falls under scratch.

    Such a segment is exempt from the deliverable policy on both sides of the
    guard: whatever it writes cannot become a deliverable. The exemption is
    deliberately narrow — a single non-scratch reference (including a bare
    ``/outputs``, or a ``..`` that normalises out of scratch) keeps the
    segment on the fail-closed path. ``cd``/``pushd`` stay excluded because a
    shell sitting inside scratch can write ``../`` relative targets the guard
    never sees, and ``ln`` is refused separately (``_scratch_link_error``).
    """
    referenced = _referenced_output_paths(segment)
    if not referenced or not all(_is_scratch_path(p) for p in referenced):
        return False
    verb = _head_verb(segment)
    if verb is None:
        # Unparseable quoting: the references may not be what they look like.
        return False
    return verb not in {"cd", "pushd", "ln", "link"}


def _scratch_bash_quota_error(command: str) -> str | None:
    """Quota gate for bash writes into scratch; applies in every workflow.

    Removal shapes are exempt — they are the model's only way to get back
    under the quota, so blocking them would make the error text a dead end.
    """
    for segment in _split_segments(command):
        scratch_refs = [
            path
            for path in _referenced_output_paths(segment)
            if _is_scratch_path(path)
        ]
        if not scratch_refs:
            continue
        verb = _head_verb(segment)
        if verb in {"cd", "pushd"}:
            continue  # cwd changes themselves add no bytes
        if verb in {"rm", "unlink", "rmdir"}:
            continue
        if verb == "mkdir" and all(path == _SCRATCH_DIR for path in scratch_refs):
            continue  # idempotently creating the reserved directory adds no file bytes
        # ``mv`` out of /outputs also frees space; within it, it adds a file.
        removed = _removed_output_paths(segment)
        if removed and all(path in removed for path in scratch_refs):
            continue
        if not (
            _definite_write_signal(segment) or _opaque_write_shape(segment)
        ) and _segment_reads_only(segment):
            continue
        # Unknown commands and shell wrappers are deliberately treated as
        # possible writers here. Scratch is fail-open below quota, but once
        # full an unverifiable shape must not bypass the only recovery guard.
        error = scratch_write_error()
        if error:
            return error
    return None


def _writes_after_cd(segment: str) -> bool:
    """Write detection with no ``/outputs`` anchoring, for cd-into-outputs.

    After ``cd /outputs[/…]`` every RELATIVE target lands in /outputs, so the
    usual carve-outs — a ``cp`` whose destination does not name /outputs, a
    redirect to a relative path — are exactly the shapes the cd bypass uses
    (``cd /outputs/scratch && cp /workspace/a.png ../leak.png``). Anything
    that produces a file anywhere counts here; reads (``ls``, ``cat``) stay
    allowed. Over-approximating (a redirect to /dev/null) only steers the
    publisher to absolute declared paths, which the error text asks for.
    """
    if _definite_write_signal(segment) or _opaque_write_shape(segment):
        return True
    verb = _head_verb(segment)
    if verb == "cd":
        return False
    if verb is None:
        return bool(_MUTATING_COMMAND_RE.search(segment))
    if verb in _MUTATING_VERBS or verb in _POSITIONAL_OUTPUT_VERBS:
        return True
    return ">" in segment


def _non_publisher_bash_error(command: str) -> str | None:
    """Fail-closed check: only recognisably read-only shapes may name /outputs."""
    segments = _split_segments(command)
    if _CD_OUTPUTS_RE.search(command):
        return _NON_PUBLISHER_CD_ERROR
    for segment in segments:
        if _OUTPUT_PATH_RE.search(segment) is None:
            continue
        if _scratch_only_segment(segment):
            continue
        if _definite_write_signal(segment):
            return _NON_PUBLISHER_WRITE_ERROR
        if _opaque_write_shape(segment):
            return _NON_PUBLISHER_UNVERIFIABLE_ERROR
        if not _segment_reads_only(segment):
            return _NON_PUBLISHER_UNVERIFIABLE_ERROR
    return None


def _publisher_bash_error(command: str, retired: tuple[str, ...]) -> str | None:
    """Manifest-check every write signal; the shape itself is not restricted."""
    segments = _split_segments(command)
    if _CD_OUTPUTS_RE.search(command) and any(
        _writes_after_cd(segment) for segment in segments
    ):
        return _PUBLISHER_CD_ERROR
    for segment in segments:
        if _OUTPUT_PATH_RE.search(segment) is None:
            continue
        if not (
            _definite_write_signal(segment)
            or _opaque_write_shape(segment)
        ):
            continue
        # A segment that touches only scratch cannot produce a deliverable, so
        # the manifest has nothing to say about it (quota is checked by the
        # caller). This precedes the prefix check on purpose: a prefix operand
        # under scratch derives a path that is still under scratch.
        if _scratch_only_segment(segment):
            continue
        # A derived target cannot be matched against the manifest: the operand
        # is a prefix the tool appends to, so approving it would authorise a
        # path that never appears while the real file lands undeclared beside
        # it. ``/outputs/final.png`` as a pdftoppm root yields
        # ``/outputs/final.png-1.png``.
        if _converter_write_kind(segment) == "prefix":
            return _PUBLISHER_DERIVED_PATH_ERROR
        removed = _removed_output_paths(segment) if retired else ()
        referenced = _referenced_output_paths(segment)
        for path in referenced:
            if path == "/outputs":
                continue
            # A mixed segment's scratch operands are exempt like a scratch-only
            # segment's; its non-scratch operands still face the manifest.
            if _is_scratch_path(path):
                continue
            # Clearing a superseded entry out of /outputs is the one write the
            # manifest cannot authorise but the run needs.
            if path in retired and path in removed:
                continue
            error = output_write_error(path)
            if error:
                return error
        # A write that names only the directory can hide a relative target
        # (for example ``python script.py --out /outputs``). A scratch path
        # does not count as a concrete target here: it would let a bare
        # ``/outputs`` ride through unexamined.
        if not any(
            path != "/outputs" and not _is_scratch_path(path)
            for path in referenced
        ):
            return _PUBLISHER_DIR_ONLY_ERROR
    return None


def bash_output_write_error(command: str) -> str | None:
    """Guard bash mutations of ``/outputs``.

    Bubblewrap additionally mounts ``/outputs`` read-only for non-publishers,
    but container mode cannot remount per asyncio task, so this is the whole
    enforcement surface there. Non-publishers are therefore fail-closed on
    command shape; publishers are bound to their exact manifest. ``cd
    /outputs`` writes are rejected because their targets cannot be audited.

    ``/outputs/scratch`` sits outside both sides of that partition: writes
    there are allowed for every role and every workflow (including ones with
    no policy opted in), subject only to the size quota — checked last, so a
    policy violation reports the policy error rather than the quota. Link
    creation touching scratch is refused first, also in every workflow: the
    string-based scratch exemption is only sound while no link can redirect
    a scratch path elsewhere.
    """
    if error := _spill_bash_write_error(command):
        return error
    command = _canonicalise_output_references(command)
    if _OUTPUT_PATH_RE.search(command) is None:
        return None
    segments = _split_segments(command)
    for segment in segments:
        error = _scratch_link_error(segment)
        if error:
            return error
        referenced = _referenced_output_paths(segment)
        if _SCRATCH_DIR not in referenced:
            continue
        if _SCRATCH_DIR_AS_DIR_RE.search(segment):
            # A trailing-slash reference writes into the directory, never over
            # it. Fall through to the quota/policy checks like a child write.
            continue
        verb = _head_verb(segment)
        if verb in {"cd", "pushd", "mkdir", "rm", "rmdir", "unlink"}:
            continue
        removed = _removed_output_paths(segment)
        if _SCRATCH_DIR in removed:
            continue
        if (
            _definite_write_signal(segment)
            or _opaque_write_shape(segment)
            or (
                referenced
                and all(path == _SCRATCH_DIR for path in referenced)
                and not _segment_reads_only(segment)
            )
        ):
            return _SCRATCH_ROOT_WRITE_ERROR
    policy = _current_policy()
    if policy is not None:
        error = (
            _non_publisher_bash_error(command)
            if not policy.manifest
            else _publisher_bash_error(command, policy.retired)
        )
        if error:
            # The command names the target; a command that only names /outputs
            # in an unverifiable shape still records, under the bare root, so
            # the coordinator learns the write was refused.
            targets = tuple(
                path
                for path in _referenced_output_paths(command)
                if not _is_scratch_path(path)
            ) or ("/outputs",)
            if policy.manifest:
                kind = "undeclared"
            else:
                # The fail-closed non-publisher policy also rejects commands
                # whose shape it cannot prove read-only. That is not evidence
                # of an attempted write, so do not turn it into one when the
                # refusal reaches the coordinator.
                kind = (
                    "not_publisher"
                    if error == _NON_PUBLISHER_WRITE_ERROR
                    else "unverifiable"
                )
            for target in targets:
                _record_output_write_denial(kind, target)
            return error
    return _scratch_bash_quota_error(command)


# ---------------------------------------------------------------------------
# Inherited publishing directives
#
# APEX-style task text names the deliverable itself ("Write your reply to the
# user as /outputs/answer.md"), and ``assign_task`` prepends the original
# question verbatim to EVERY dispatched prompt. A workspace-only sub-agent
# therefore receives the user's own instruction to write /outputs/answer.md
# followed by a contract block telling it not to write /outputs at all, with
# nothing connecting the two. Measured over the 267-task APEX agent_team run:
# 267/267 task texts carried such a directive, and 1352/1767 (76.5%) of all
# non-publisher write denials targeted exactly ``/outputs/answer.md`` -- the
# path the user asked for. The agent was obeying the user.
#
# These helpers find those directives so the contract block can name the
# conflict and say who owns it. Detection is advisory only: a miss yields the
# generic contract wording and the write is still blocked at execution time, so
# the pattern is tuned for precision and nothing depends on its recall.
# ---------------------------------------------------------------------------

# A concrete file under /outputs. A bare ``/outputs`` mention is deliberately
# not a match: in these prompts it is almost always "do not touch /outputs",
# "ls /outputs", or a reference to where some other agent will write.
_OUT_FILE_RE = re.compile(
    r"(?<![\w/])/outputs/(?!scratch(?:/|\b))"
    r"(?:[A-Za-z0-9._\-]+/)*[A-Za-z0-9._\-]*[A-Za-z0-9_\-](?:\.[A-Za-z0-9]{1,8})?"
)

# Imperative, infinitive and gerund forms only. Past participles are excluded
# on purpose: "the published file /outputs/answer.md" and "the answer written
# to /outputs/answer.md" describe a file that already exists, which a verifier
# is entitled to read.
_DIRECTIVE_WRITE_RE = re.compile(
    r"\b(?:write|writes|writing|create|creates|creating|save|saves|saving"
    r"|produce|produces|producing|generate|generates|generating"
    r"|export|exports|exporting|place|places|placing|put|puts|putting"
    r"|deliver|delivers|delivering|publish|publishes|publishing"
    r"|emit|emits|emitting|render|renders|rendering|assemble|assembles"
    r"|output|outputs|outputting|persist|persists|persisting|store|stores"
    r"|storing|dump|dumps|dumping|copy|copies|copying|cp|move|moves|moving"
    r"|mv|append|appends|appending|touch|tee|redirect|redirects|redirecting"
    r"|final[iy]?[sz]e|final[iy]?[sz]es|final[iy]?[sz]ing)\b",
    re.I,
)
_DIRECTIVE_READ_RE = re.compile(
    r"\b(?:read|reads|reading|inspect|inspects|inspecting|check|checks"
    r"|checking|list|lists|listing|ls|cat|verify|verifies|verifying|verified"
    r"|review|reviews|reviewing|audit|audits|auditing|examine|examines"
    r"|examining|confirm|confirms|confirming|compare|compares|comparing"
    r"|published|written|existing|already|exists|present|against)\b",
    re.I,
)
# "do NOT write to /outputs/answer.md" is an instruction to stay out, and one
# of the most common shapes in coordinator-authored prompts.
_DIRECTIVE_NEGATOR_RE = re.compile(
    r"\b(?:do not|don't|dont|does not|doesn't|never|must not|mustn't|cannot"
    r"|can't|may not|should not|shouldn't|will not|won't|no|not|without"
    r"|avoid|avoids|avoiding|refrain|forbidden|blocked|denied|prohibited"
    r"|instead of|rather than)\b",
    re.I,
)
# Unambiguous shell writes need no surrounding prose.
_DIRECTIVE_SHELL_RE = re.compile(
    r"(?:>>?\s*['\"]?|\b(?:cp|mv|tee|install)\s+[^\n|;&]*?\s)"
    r"(?<![\w/])/outputs/(?!scratch(?:/|\b))",
    re.I,
)

# How far back a governing verb may sit from the path it governs.
_DIRECTIVE_LOOKBACK = 80


def output_write_directives(text: str) -> tuple[str, ...]:
    """Paths under ``/outputs`` that ``text`` appears to direct a write to.

    Conservative by construction (see the note above): a miss costs nothing,
    while a false positive would put a misleading sentence in a prompt.
    """
    if not text or "/outputs/" not in text:
        return ()
    found: list[str] = []

    def _add(path: str) -> None:
        path = path.rstrip(".")
        if path not in found and not _is_scratch_path(_normalise_output_path(path)):
            found.append(path)

    for match in _DIRECTIVE_SHELL_RE.finditer(text):
        path_match = _OUT_FILE_RE.search(text, match.start())
        if path_match is not None:
            _add(path_match.group(0))

    for match in _OUT_FILE_RE.finditer(text):
        window = text[max(0, match.start() - _DIRECTIVE_LOOKBACK):match.start()]
        writes = list(_DIRECTIVE_WRITE_RE.finditer(window))
        if not writes:
            continue
        verb = writes[-1]
        # A negator between the verb and the path -- or right before the verb --
        # flips the clause into "stay out of /outputs".
        if _DIRECTIVE_NEGATOR_RE.search(window[max(0, verb.start() - 30):]):
            continue
        # The nearest governing verb decides: an audit prompt may well say
        # "produce a report" before naming the published file it must read.
        reads = [m.end() for m in _DIRECTIVE_READ_RE.finditer(window)]
        if reads and max(reads) > verb.end():
            continue
        _add(match.group(0))
    return tuple(found)


def render_publish_assignment(paths: Iterable[str]) -> str:
    """Prompt block appended to an explicitly publishing task."""
    manifest = normalise_output_paths(paths)
    workspace_root = _runtime_workspace_root()
    runtime_paths = tuple(_runtime_output_path(path) for path in manifest)
    items = "\n".join(f"- `{path}`" for path in runtime_paths)
    native_note = ""
    if runtime_paths != manifest:
        declarations = "\n".join(f"- `{path}`" for path in manifest)
        native_note = (
            "\nThe coordinator declared these virtual manifest names:\n"
            f"{declarations}\n"
            "This is a native run: `/outputs` is only the manifest namespace "
            "and is not a usable filesystem mount. Write directly to the "
            "physical paths listed below; do not probe, create, or fall back "
            "from literal `/outputs`.\n"
        )
    return (
        "\n\n# Deliverable publishing contract\n"
        "You are the sole publisher for this assignment. First collect the "
        f"candidate artifacts your teammates produced under {workspace_root} (their "
        "reports give the exact paths), integrate and validate them there, "
        f"then publish the exact declared manifest.{native_note}\n"
        "Write exactly these final files using their absolute filesystem paths:\n"
        f"{items}\n"
        "Do not create README, verification, confirmation, alternate-format, "
        "or versioned files unless one is explicitly listed above."
    )


def render_retirement_note(paths: Iterable[str]) -> str:
    """Prompt block telling a publisher to clear superseded deliverables."""
    retired = normalise_output_paths(paths)
    items = "\n".join(f"- `{_runtime_output_path(path)}`" for path in retired)
    output_root = _runtime_outputs_root()
    workspace_root = _runtime_workspace_root()
    return (
        "\n\n# Superseded deliverables to remove\n"
        "The output manifest was replaced, so these files are no longer part "
        f"of the deliverable and must not be left in {output_root}:\n"
        f"{items}\n"
        f"Delete each one (`rm -f <path>`) or move it under {workspace_root} before "
        "you finish. Removing exactly these paths is permitted; rewriting "
        "them is not."
    )


def render_workspace_assignment(
    inherited_paths: Iterable[str] = (),
    publisher_declared: bool = True,
) -> str:
    """Prompt block appended to every non-publishing task.

    ``inherited_paths`` are deliverable paths named by text this sub-agent also
    receives -- the original question prepended to its prompt, or the
    coordinator's own wording. Left unnamed, those instructions simply
    contradict the contract below and the agent has no way to tell which one
    governs; most of them followed the user. Naming them and saying who owns
    them is the whole point of this argument.

    ``publisher_declared`` says only that the coordinator named a publisher in
    this dispatch or an earlier one. It does not promise that the assignment
    successfully dispatched or that its manifest covers these paths; the
    dispatch result reports actual authorization after submission finishes.
    """
    output_root = _runtime_outputs_root()
    workspace_root = _runtime_workspace_root()
    scratch_root = os.path.join(output_root, "scratch")
    inherited = tuple(dict.fromkeys(inherited_paths))
    if inherited:
        named = ", ".join(f"`{_runtime_output_path(p)}`" for p in inherited)
        owner = (
            "the coordinator named a separate publisher candidate, but only a "
            "successfully dispatched publishing assignment whose manifest "
            "covers this path can write it"
            if publisher_declared
            else "no agent in this run holds the publishing role yet, and the "
            "coordinator assigns it -- not you, and not by writing the file"
        )
        override = (
            "\n\nThe task text above tells you to write "
            f"{named}. That instruction is addressed to the team, not to you: "
            f"{owner}. Attempting it here is blocked, and a blocked write that "
            "you route somewhere else instead silently loses the deliverable. "
            f"Produce your part under {workspace_root} and give the exact path "
            "in your report; the publisher collects it from there."
        )
    else:
        owner = (
            "only a successfully dispatched publishing assignment whose "
            "manifest covers that file can write it; you are not that assignment"
            if publisher_declared
            else "the coordinator assigns the publishing role and no agent "
            "holds it yet"
        )
        override = (
            f"\n\nIf any instruction you were given names a file under "
            f"{output_root} as the deliverable, that instruction is addressed "
            f"to the team, not to you: {owner}. Report your {workspace_root} "
            "path instead."
        )
    return (
        "\n\n# Workspace-only contract\n"
        "This is not a publishing assignment. Do not create, modify, move, or "
        f"delete anything in {output_root}, with one exception: the top-level "
        f"{scratch_root}/ directory is writable by every assignment. It "
        "persists across rounds, so put intermediate products worth reusing "
        "in a later round there; it is not a deliverable area and is never "
        "shown to the user. It has a 512MB quota — over-quota writes fail "
        "with an error until you delete files you no longer need there. Only "
        f"the literal {scratch_root}/ prefix qualifies (a deeper .../scratch/ "
        "directory does not). Put all candidate artifacts, calculations, "
        f"reports, and temporary files under {workspace_root} and return their exact "
        "paths in your report. Verification reports are report text, not files."
        + override
    )
