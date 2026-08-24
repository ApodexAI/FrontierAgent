"""read_file tool — parse file formats (office / PDF) into structured markdown for the LLM."""

from __future__ import annotations

import hashlib
import logging
import os
import shlex

from frontier_agent.core.tool import tool
from plugins.tools._deliverable_policy import spill_write_error
from plugins.tools._doc_reader import reader_src
from plugins.tools._sandbox import (
    aget_sandbox,
    arun_sandbox_cmd,
    resolve_mount_dirs,
    resolve_runtime_path,
    resolve_sandbox_mode,
)

logger = logging.getLogger(__name__)


def _rejected_save_to(save_to: str) -> str | None:
    """Reject a ``save_to`` that lands in one of the task's shared mounts.

    Only enforced in container mode, where those mounts are real.

    ``/outputs`` is world-writable there and is diffed after every main-agent
    tool result to build ``file_delta`` events and ``final.deliverables`` — so a
    reader saving its markdown there would silently publish a deliverable. This
    tool renders documents; declaring deliverables belongs to the write tools.
    ``/inputs`` is the read-only source mount, so writing into it is a mistake
    worth naming rather than an OS-level permission error.

    Everything else stays allowed, including ``/tmp``: ``_build_tool_env`` hands
    model commands ``TMPDIR=/tmp`` on purpose and several skills tell the model
    to put files there, so it is ordinary scratch space, not a deliverable path.

    Symlinks are resolved before the comparison. A lexical check is not enough:
    with ``/workspace/link -> /outputs``, ``/workspace/link/report.md`` reads as
    a workspace path but the shell redirection lands in ``/outputs``. Both the
    fully resolved target and its resolved parent are checked, since the leaf
    itself usually does not exist yet.

    This is a guard against the model publishing a deliverable by accident, not
    a security boundary: the write happens later, in the sandbox, so a symlink
    created in between would not be seen here. Anything that needs to be
    airtight has to be enforced where the write occurs.
    """
    if error := spill_write_error(save_to):
        return f"Error: {error}"
    if resolve_sandbox_mode() != "container":
        return None
    workspace_dir, outputs_dir, inputs_dir = resolve_mount_dirs()
    # Relative paths resolve against the sandbox cwd, which IS the workspace.
    absolute = (
        save_to if os.path.isabs(save_to) else os.path.join(workspace_dir, save_to)
    )
    # realpath() resolves the components that exist and leaves the rest alone,
    # so this works for a leaf that has not been created yet.
    target = os.path.realpath(absolute)
    parent = os.path.realpath(os.path.dirname(absolute) or ".")
    for reserved, why in (
        (outputs_dir, "is for deliverables written by the write tools"),
        (inputs_dir, "is a read-only mount"),
    ):
        resolved_reserved = os.path.realpath(reserved)
        if any(
            candidate == resolved_reserved
            or candidate.startswith(resolved_reserved + os.sep)
            for candidate in (target, parent)
        ):
            return (
                f"Error: save_to cannot write into {reserved} — it {why} "
                f"(got {save_to!r}). Save under {workspace_dir} instead."
            )
    return None


def _dump_readout(path: str, content: str) -> None:
    """When READDOC_DUMP_DIR is set, persist what read_file read out (host-side)
    so it can be inspected. Does NOT alter the in-sandbox read path."""
    d = os.environ.get("READDOC_DUMP_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        base = os.path.basename(path) or "doc"
        h = hashlib.md5(path.encode()).hexdigest()[:8]
        with open(os.path.join(d, f"{base}.{h}.readout.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- read_file readout | path: {path} -->\n\n{content}")
    except Exception as e:
        logger.warning("read_file dump failed for '%s': %s", path, e)

# The parsing logic is split by file type into _reader_{core,xlsx,docx,pptx,pdf}.py; here they
# are concatenated into one
# self-contained script, piped into the sandbox over stdin and run with `python3 -`.
_READER_SRC = reader_src()

#: The comma-separated form of ``path`` is documented for images only, so a
#: comma anywhere else is far likelier to be part of a filename.
_BATCH_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp")


def _looks_like_batch(path: str) -> bool:
    """True when ``path`` is a comma-separated list of image paths.

    Every member has to look like an image: one segment that does not is
    enough to make ``a, b`` a single filename containing a comma, which
    splitting would corrupt (the space after the comma is stripped).
    """
    if "," not in path:
        return False
    parts = [item.strip() for item in path.split(",")]
    return all(
        part and part.lower().endswith(_BATCH_SUFFIXES) for part in parts
    )
# Feed the reader to python3 via stdin in a SINGLE sandbox command (no temp
# file). Under bwrap each command gets a fresh /tmp tmpfs, so a reader written
# to /tmp in one command is gone by the next — same reason run_python_code
# pipes its code via stdin. Writing+running as two commands silently breaks
# read_file under the per-task BwrapSandbox.
# Critical: never base64 the bundle and echo it into argv — the bundle is already ~95KB
# (~145KB base64),
# while execve's per-argument limit MAX_ARG_STRLEN is 128KB, so echoing it onto the command line gives E2BIG
# ("Argument list too long"). stdin is a data stream with no such limit.
_TIMEOUT = 120


@tool
async def read_file(
    path: str,
    max_chars: int | None = None,
    offset: int = 0,
    cell_range: str | None = None,
    save_to: str | None = None,
    pdf_mode: str = "auto",
    pages: str | None = None,
) -> str:
    """Read a document. Structured parsing of office/PDF/csv files into markdown.

    Use this to READ any file under /inputs (or anywhere) — do NOT write parsing
    scripts to read them: it renders office/PDF into high-signal markdown and
    READS IMAGES for you.

    IMAGES: pass an image path (png/jpg/jpeg/gif/bmp/tif/tiff/webp) and the tool
    SEES it — charts, diagrams, screenshots, photos, scanned pages are transcribed
    (title, axes, legend, series values, all visible text) straight into the reply.
    Use it to check your own rendered output too: after producing charts/PDF pages/images
    or PPT file(when the design and layout matter), read it back to confirm labels,
    layouts, numbers and non-ASCII text render (e.g., a missing font shows up as boxes,
    or overlapping text). Several images at once: see `path` below.

    Office/PDF (xlsx/docx/pptx/pdf) are parsed into structured markdown; csv/tsv become a
    relational table + column meta. Legacy Office binaries (doc/ppt/xls) are converted via
    LibreOffice then parsed the same way (readout notes the conversion). Any other file is
    sniffed in-sandbox: text (txt/md/json/yaml/code/log/…) is returned with line numbers;
    audio/video report only the detected file type; unsupported binary is reported
    as such (never dumped).
    Parses inside the sandbox (xlsx→openpyxl+LibreOffice recalc, docx→pandoc,
    pptx→python-pptx, pdf→pdftotext) and returns markdown: spreadsheets→coordinate
    grid + formulas/styles meta, docs→headings+tables, slides→per-slide rich
    text, pdf→per-page text.

    Large documents are paged at SECTION boundaries (sheet/slide/page/heading —
    never mid-table-row). A partial read ends with "[read_file] PARTIAL READ …
    continue with offset=N" plus a map of remaining sections; pass that offset
    back to continue reading. Rendered output is cached, so continuation reads
    are cheap.

    Args:
        path: Absolute path to the file in the sandbox. For MANY images at once
            (multi-page renders, several charts), pass a GLOB (e.g.
            "/workspace/pg-*.png") or a comma-separated list of image paths — all
            of them are read in parallel and returned image-by-image in ONE call
            (up to 8 per call; don't read them one at a time). For a PDF, prefer
            pdf_mode="image" with a pages range to render+read many pages at once.
        max_chars: Page size budget per call. Omit for the default (8000 chars;
            20000 when reading a batch of images via glob / comma-separated
            paths, since one image transcript alone is 2-6K). Any value you pass
            explicitly is used as-is. The ToolMessage budget leaves headroom
            above this so the trailing "PARTIAL READ … continue with offset=N"
            hint always survives.
        offset: Continue a previous partial read from this char position (use
            the exact offset given in the previous PARTIAL READ note).
        cell_range: xlsx only — A1-style range like "Sheet1!A3:D15" to dump just
            that range (second read of pivot output or a large sheet region).
        save_to: If set, write the FULL markdown (no paging) to this sandbox
            path and return a one-line confirmation. Useful for large documents:
            save once, then inspect selectively via read_file/grep. Use a scratch
            path — this is a reader, not a way to publish a deliverable.
        pdf_mode: pdf only — "auto" (default: text + mark image-only pages),
            "text" (pdftotext -layout, faithful, blanks left blank), or "image"
            (render the requested pages and READ them via vision — use it for
            scans, figures and stamped pages that carry no extractable text).
        pages: pdf only — which pages to read, 1-based like "1-5,12,40-"
            (empty = all). Use it to scope a large PDF or to render just the
            image-only pages flagged by an earlier auto read.

    Returns:
        Markdown text (or a save confirmation), or an error/hint message.
    """
    if not path or not path.strip():
        return "Error: path is required."
    # A comma-separated image batch is part of this tool's public contract;
    # resolve each member independently so native aliases do not become one
    # malformed physical path. Only split when every segment really looks like
    # a batch member — ``report, final.pdf`` is one filename, and splitting it
    # would strip the space after the comma and read a path that is not there.
    if _looks_like_batch(path):
        path = ",".join(resolve_runtime_path(item.strip()) for item in path.split(","))
    else:
        path = resolve_runtime_path(path)
    if save_to:
        rejected = _rejected_save_to(save_to)
        if rejected:
            return rejected
        save_to = resolve_runtime_path(save_to)
    # Every file goes through the sandbox reader: structured documents → markdown; the rest
    # are sniffed and routed inside the bundle
    # (text → numbered content; image/audio/video → type only; unsupported binary → said so explicitly).
    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        return f"Error: {e}"

    try:
        # argv = path max_chars [cell_range|-] [offset] [pdf_mode] [pages|-]
        # max_chars=None is the "not passed explicitly" sentinel (a literal 8000 cannot be used
        # — that could not distinguish
        # "explicitly passed 8000" from "used the default", and an explicit 8000 would be wrongly raised to 20000).
        # Default windows: 8K for an ordinary read; 20K for a batch image read (one image
        # transcribes to 2-6K, so 8K holds only 1-2 and loses the "N turns collapse into 1" win).
        # Any explicitly passed value is honoured exactly.
        if max_chars is None:
            is_batch = any(c in path for c in "*?[") or "," in path
            eff_max = 20_000 if is_batch else 8_000
        else:
            eff_max = int(max_chars)
        if save_to:
            eff_max = 10**9        # save_to = write the whole thing to disk, no pagination
        eff_off = 0 if save_to else int(offset)
        # Positional arguments: trim from the right by their defaults
        argv = [
            shlex.quote(cell_range) if cell_range else "-",
            str(eff_off),
            # Whitelist + quote: pdf_mode is the only argv element that was
            # interpolated raw, so an LLM value like `auto; touch /x` reached
            # the sandbox shell. Coerce to a known literal, then quote.
            shlex.quote(pdf_mode if pdf_mode in ("auto", "text", "image") else "auto"),
            shlex.quote(pages) if pages else "-",
        ]
        defaults = ["-", "0", "auto", "-"]
        while argv and argv[-1] == defaults[len(argv) - 1]:
            argv.pop()
        tail = (" " + " ".join(argv)) if argv else ""
        # The reader source goes over stdin (input=_READER_SRC), leaving only `python3 - <argv>` on the command line (short)
        cmd = f"python3 - {shlex.quote(path)} {eff_max}{tail}"
        if save_to:
            # Store to a sandbox file instead of returning the whole text (large documents: store once, then fetch on demand with read_file/grep/cell_range)
            q = shlex.quote(save_to)
            d = shlex.quote(os.path.dirname(save_to) or ".")
            cmd = f"mkdir -p {d} && {cmd} > {q} && wc -c < {q}"
        # The parser is trusted harness infrastructure: allow_net opens networking so it can
        # reach READDOC_OCR/VISION, and
        # env_allow passes through those two services' URL/KEY. Sandbox env is deny-all by
        # default (see _sandbox_env),
        # so READDOC_* has to be declared here explicitly; the model's bash in the same sandbox does not inherit these credentials.
        result = await arun_sandbox_cmd(sandbox, cmd, timeout=_TIMEOUT, input=_READER_SRC,
                                        allow_net=True, env_allow=("READDOC_",))
        if result.exit_code != 0:
            return f"[read_file error] exit={result.exit_code}: {result.stderr or result.stdout}"
        if save_to:
            size = (result.stdout or "?").strip()
            return (f"[read_file] markdown saved to {save_to} ({size} bytes). "
                    "Inspect with read_file/grep; for xlsx use cell_range for targeted re-reads.")
        out = result.stdout or "(empty)"
        _dump_readout(path, out)
        return out
    except TimeoutError:
        return f"Error: read_file timed out after {_TIMEOUT}s"
    except Exception as e:
        logger.warning("read_file error for '%s': %s", path, e)
        return f"[read_file error] {type(e).__name__}: {e}"
