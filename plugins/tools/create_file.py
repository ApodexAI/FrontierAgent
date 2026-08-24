"""create_file tool — write and edit deliverable files inside the sandbox, mirroring read_file."""
from __future__ import annotations

import json
import logging
import os
import shlex
import sys
from typing import Any

from frontier_agent.core.tool import tool
from plugins.tools._create_file import writer_src
from plugins.tools._deliverable_policy import (
    declared_output_paths,
    output_write_error,
)
from plugins.tools._path_auth import _path_within
from plugins.tools._sandbox import (
    _DEFAULT_OUTPUTS_DIR,
    _DEFAULT_WORKSPACE_DIR,
    aget_sandbox,
    arun_sandbox_cmd,
    resolve_mount_dirs,
    resolve_runtime_path,
    resolve_sandbox_mode,
)

logger = logging.getLogger(__name__)

_WRITER_SRC = writer_src()
_DOC_EXTS = {"docx", "xlsx", "pptx"}
_TEXT_EXTS = {"txt", "md", "csv", "tsv", "json", "jsonl", "html", "htm"}
_ALL_EXTS = _DOC_EXTS | _TEXT_EXTS
_TIMEOUT = 120
#: Ceiling for the ASSEMBLED command, which reaches ``sh -c`` as one execve
#: argument. Linux caps a single argument at MAX_ARG_STRLEN (32 pages =
#: 131072 bytes); this leaves margin for any wrapper a backend prepends.
#:
#: Measured after ``shlex.quote``, not on the raw payload: quoting expands each
#: apostrophe to four bytes (``'`` -> ``'"'"'``), so a 30KB body of apostrophes
#: assembles into a 150KB command. Bounding the payload alone let exactly that
#: case through to a bare E2BIG.
_MAX_COMMAND_BYTES = 120 * 1024

_PATH_ARGUMENT_KEYS = frozenset({"path", "image_path", "out", "output"})


def _runtime_ops_paths(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite only fields whose schema denotes a filesystem path.

    Text content may legitimately mention ``/outputs`` and must stay literal,
    so this intentionally does not rewrite arbitrary strings recursively.
    """
    def rewrite(value: Any) -> Any:
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key in _PATH_ARGUMENT_KEYS and isinstance(item, str):
                rewritten[key] = resolve_runtime_path(item)
            else:
                rewritten[key] = rewrite(item)
        return rewritten

    return rewrite(ops)


def _inlined_ops_program(program_path: str) -> str | None:
    """Return an ``@program`` file's ops inlined with runtime paths resolved.

    ``None`` means "keep the ``@`` reference": the program is not readable from
    this process (container / e2b keep it in the sandbox namespace, where the
    aliases are the real mount points and need no rewrite), it is not a JSON
    array (let the writer report that, as before), or inlining it would push
    the assembled command past ``_MAX_COMMAND_BYTES`` — the large-program case
    the ``@`` form exists to serve.
    """
    if resolve_sandbox_mode() != "native":
        return None
    try:
        with open(program_path, encoding="utf-8") as fh:
            program = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(program, list):
        return None
    payload = json.dumps(_runtime_ops_paths(program), ensure_ascii=False)
    return payload if len(payload) <= _MAX_COMMAND_BYTES // 2 else None


def _write_roots() -> tuple[str, ...]:
    """The roots ``create_file`` may write under, resolved per call.

    The container mount points (``/workspace`` / ``/outputs`` — the convention
    this tool's docstring teaches the model) plus whatever
    :func:`resolve_mount_dirs` currently maps them to. Native mode overrides
    them to real host directories and then hands the model those exact paths
    (``apodex.session._deliverable_context``), so accepting only the literals
    refused every native-mode write.
    """
    workspace, outputs, _inputs = resolve_mount_dirs()
    return tuple(dict.fromkeys(
        (_DEFAULT_WORKSPACE_DIR, _DEFAULT_OUTPUTS_DIR, workspace, outputs),
    ))


def _outside_write_roots(path: str) -> bool:
    """True when *path* is not contained by any write root.

    Containment is component-aware and computed on the normalized path, not by
    string prefix: with a resolved native root like ``/task/outputs``, a prefix
    test also accepts the sibling ``/task/outputs-escape/x.md`` and lets
    ``/task/outputs/../x.md`` through. Since the roots include real host
    directories, that is a host write escape rather than a cosmetic gap.

    ``normpath`` (not ``resolve``) keeps the check honest about what the writer
    will do: the sandbox resolves symlinks on its own filesystem, which may not
    be the one this process sees.
    """
    normalized = os.path.normpath(path)
    return not any(
        _path_within(normalized, os.path.normpath(root)) for root in _write_roots()
    )


def _deliverable_ops_error(path: str, ops: list[dict[str, Any]]) -> str | None:
    """Guard the *secondary* write targets inside ops.

    ``export_pdf`` writes a second file: ``out``, or ``path`` with a ``.pdf``
    extension when ``out`` is omitted. Without this check a publisher declaring
    ``/outputs/report.docx`` would silently also produce
    ``/outputs/report.pdf`` — exactly the undeclared sidecar the manifest
    exists to prevent. No-op for workflows that have not opted in.
    """
    for i, it in enumerate(ops):
        if not isinstance(it, dict):
            continue
        for op, args in it.items():
            params = args if isinstance(args, dict) else {}
            targets = []
            explicit = str(params.get("out") or params.get("output") or "").strip()
            if explicit:
                targets.append(explicit)
            elif op == "export_pdf":
                targets.append(path.rsplit(".", 1)[0] + ".pdf")
            for target in targets:
                err = output_write_error(target)
                if err:
                    return f"Error: ops[{i}] ({op}) writes {target!r}. {err}"
    return None


def _ops_program_error(path: str, program_path: str) -> str | None:
    """Validate an ``@/workspace/program.json`` ops file against the manifest.

    The writer resolves the reference inside the sandbox, so its ops are
    invisible at this boundary. When a manifest is active and the program
    cannot be read here, fail closed rather than let an unaudited
    ``export_pdf`` target through.
    """
    if declared_output_paths() is None:
        return None
    try:
        with open(program_path, encoding="utf-8") as fh:
            program = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return (
            f"Error: ops program {program_path!r} cannot be validated against "
            "the deliverable manifest here. Pass the ops array inline instead."
        )
    if not isinstance(program, list):
        return f"Error: ops program {program_path!r} must contain a JSON array."
    return _deliverable_ops_error(path, program)


def _validate_ops(ops: list[dict[str, Any]]) -> str | None:
    """Every ops entry must be a single-key object {op: {params}}; returns an error string when it is not, else None."""
    for i, it in enumerate(ops):
        if not isinstance(it, dict) or len(it) != 1:
            return (f'Error: ops[{i}] must be a single-key object {{op: {{params}}}}, '
                    f'e.g. {{"set_cell": {{"sheet":"S","cell":"A1","value":1}}}}; got {it!r}')
        v = next(iter(it.values()))
        if v is not None and not isinstance(v, dict):
            return f"Error: ops[{i}] params must be an object, got {type(v).__name__}"
    return None


def _failure_detail(result: Any) -> str:
    """Both of the writer's streams, stdout first.

    The writer reports *which* op failed and why on stdout — that receipt, and
    the containment refusals from ``_abort``, exist to be read by the model so
    it can correct its own call. Preferring ``stderr`` whenever it is non-empty
    throws the receipt away the moment anything else speaks up on that stream:
    an ``openpyxl`` warning, a line from ``soffice``, a stray installer
    message. A 50-op batch would then fail as "writer exited 1: UserWarning:
    ..." with no way to tell which op it was.
    """
    parts = [
        stream.strip()
        for stream in (getattr(result, "stdout", ""), getattr(result, "stderr", ""))
        if (stream or "").strip()
    ]
    return "\n".join(parts) or "(no output)"


def desugar_text_shorthand(
    *,
    ops: list[dict[str, Any]] | str | None,
    content: str | None,
    rows: list[Any] | str | None,
    data: dict[str, Any] | list[Any] | str | None,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]] | str | None, str]:
    """Fold a top-level ``content``/``rows``/``data`` into one ``create`` op.

    Returns ``(ops, error)``; a non-empty error is the tool's whole reply.

    The model reaches for ``content=`` because it is the obvious name for a
    file's body, and because the tool guidance named it in a sentence about
    calling the tool when it is really a ``create`` op param. Measured over two
    runs: 8 calls lost to ``unexpected keyword argument 'content'``, from five
    different sub-agents, a turn each. Re-wording the guidance might help;
    desugaring cannot fail to. The op layer stays the single implementation, so
    the shorthand inherits its guards — notably the refusal to overwrite an
    existing file without being told to.
    """
    shorthand: dict[str, Any] = {
        key: value
        for key, value in (("content", content), ("rows", rows), ("data", data))
        if value is not None
    }
    if not shorthand:
        return ops, ""
    if ops is not None:
        # Silently preferring one would write a file the caller did not describe.
        return ops, (
            "Error: pass EITHER ops, OR the content/rows/data shorthand — got "
            f"both (shorthand: {', '.join(sorted(shorthand))}). The shorthand "
            'is exactly ops=[{"create": {...}}]; to do more than one operation, '
            "put everything in ops."
        )
    if len(shorthand) > 1:
        return ops, (
            "Error: pass exactly ONE of content, rows, or data — got "
            f"{', '.join(sorted(shorthand))}. These inputs are alternatives; "
            "combining them would silently discard all but one."
        )
    if overwrite:
        shorthand["overwrite"] = True
    return [{"create": shorthand}], ""


@tool
async def create_file(
    path: str,
    ops: list[dict[str, Any]] | str | None = None,
    content: str | None = None,
    rows: list[Any] | str | None = None,
    data: dict[str, Any] | list[Any] | str | None = None,
    overwrite: bool = False,
) -> str:
    """Create or edit a deliverable file in the sandbox — office (docx/xlsx/pptx)
    OR text (txt/md/csv/tsv/json/jsonl/html/htm).

    This is THE tool for producing EVERY deliverable file — .docx / .xlsx /
    .pptx (plus PDF via export_pdf), AND text files .txt / .md / .csv / .tsv /
    .json / .jsonl / .html / .htm. Author every deliverable through it. The ONLY case
    where you may fall back to Python libraries (python-docx / openpyxl /
    python-pptx / reportlab) or bash is when a create_file call has EXPLICITLY
    returned an unsupported-operation/extension error for what the task requires
    — try create_file first; do not decide on your own that a feature is
    unsupported. The fallback order is strict: first create_file; then Python
    libraries; only then, when Python still cannot cover the operation or the
    user requires accurate preservation of an existing template that Python
    would not preserve, use the runtime-advertised `docx` or `pptxgenjs`
    packages through bash. `NODE_PATH` is already configured, so load them by
    package name instead of a hard-coded install path. Never hand-build
    deliverables with bash (echo / cat / heredoc / redirection); a csv/md/txt
    deliverable goes through create_file, NOT a shell redirect. For a whole text
    file in one shot pass `content` (or `rows` / `data`) directly and leave `ops`
    out: create_file(path="/outputs/report.md", content="# Title\n..."). Write scratch to /workspace and ONLY the final
    deliverable(s) to /outputs, kept clean (no scratch or duplicate versions).

    Pass `ops`, a JSON array where EACH item is a single-key object {op_name:
    {params}}, applied IN ORDER to the same file (do MANY operations in ONE call):
      ops=[{"create":{"sheets":[...]}},
           {"set_cell":{"sheet":"S","cell":"B2","value":42,"type":"number"}},
           {"set_cell_format":{"sheet":"S","cell_range":"A1:C1","bold":true}}]
    Run set_cell 50 times = 50 items in ONE call (not 50 calls). One sandbox call
    runs them sequentially; if an op errors, execution STOPS there and the result
    lists what ran. Each item must have EXACTLY ONE key (the op name). For a large
    program, write the JSON array to a /workspace file and pass
    ops="@/workspace/program.json".

    PATHS — only two directories are writable and persistent in the sandbox:
      • /workspace  — your private scratch dir; intermediate files + the ops JSON.
      • /outputs    — final deliverables ONLY (this is what gets collected/graded).
        Exception: /outputs/scratch/ persists across rounds — put intermediate
        products worth reusing in a later round there. NOT a deliverable, never
        shown to the user; 512MB quota (over-quota writes error until you delete
        files there). Only the literal top-level /outputs/scratch/ counts.
      Any other location (e.g. /home/..., /tmp/...) is NOT mounted: writes/reads
      there FAIL or do not persist across calls. So `path` (and any ops "@file")
      must be under /workspace or /outputs — write the final deliverable to /outputs.

    Incremental load-modify-save (existing files are edited in place, untouched
    parts preserved). Anchors use STABLE references: xlsx by sheet name + A1
    cell/range; pptx by 1-based slide number + placeholder role; docx by anchor
    TEXT (find / after_text) — never fragile positional indices.

    CREATE is for a NEW file. Calling `create` on a path that ALREADY exists is
    REFUSED (so you never silently wipe prior content) — to add to an existing
    file use insert_*/set_cell/add_slide/replace_text; pass {"overwrite":true}
    only if you truly mean to rebuild from scratch.

    XLSX FORMULAS — write Excel FORMULAS, do NOT compute values in your head and
    hardcode the number: use {"set_cell":{...,"value":"=SUM(B2:B9)","type":"formula"}}
    (or value starting with "="), NOT the literal sum. Formula caches are EMPTY
    until recalculated; create_file auto-runs LibreOffice recalc on save whenever the
    batch wrote any formula, fills the cached values, and reports any formula
    errors (#DIV/0!, #REF!, ...) back in the result so you can fix them.

    TEXT IS LITERAL — formatting goes through structured params, NEVER Markdown.
    Any "text" field accepts RichText: a plain string, OR a list of runs for
    inline formatting / links:
      [{"text":"Total ","bold":true}, {"text":"site","link":"https://x.com"}]
    Run fields (all optional except text): bold, italic, underline, strike,
    color("RRGGBB"), size(pt), font, link(url). Do NOT write "**bold**",
    "[t](url)", "- item" or "# h" in text — they are written verbatim; use the
    params below instead.

    FONTS — the deliverable is downloaded and opened on an unknown platform, so
    name only fonts that exist almost everywhere; anything else is silently
    substituted (different glyphs, different widths, shifted line/page breaks):
      Latin      : Arial / Times New Roman / Courier New / Calibri / Cambria
      Chinese    : SimSun (宋体, serif) / SimHei (黑体) / Microsoft YaHei (微软雅黑, sans)
      Japanese   : MS Gothic / MS Mincho / Meiryo / Yu Gothic
      Korean     : Malgun Gothic / Batang
    Use a font outside this list ONLY WHEN the task explicitly asks for it —
    Linux-only families in particular (Noto Sans CJK *, Source Han *, DejaVu *,
    Liberation *) are absent on stock Windows/macOS, so a reader always gets a
    substitution. When the text contains Chinese/Japanese/Korean, name a CJK
    family from the list above (`font` applies to the CJK characters too); a
    Latin-only font leaves CJK to whatever the reader's app falls back to.
    Fewer fonts = fewer surprises: prefer one family per document.

    op + args by format (args is a JSON object):
      docx: create{blocks:[<block>], metadata?} — block.type:
              heading{text:RichText, level:1-9, align?} |
              paragraph{text:RichText, align?, style?, list?, page_break_before?, keep_with_next?,
                        line_spacing?, space_before?(pt), space_after?(pt)} |
              table{rows:[[Cell,...]], header?(true→bold + repeating w:tblHeader), column_widths_in?} |
              image{path,width_in?,height_in?} | page_break
              list = {type:bullet|number, level:0-8}  (use this, not "- "/"1.")
              Cell = RichText, or {content:RichText, bold?, align?, fill_color?}
            replace_text{find,replace,count?} |
            insert_paragraph{text:RichText,after_text?,style?,list?,page_break_before?,keep_with_next?} |
            insert_heading{text:RichText,level?,after_text?} |
            insert_table{rows,after_text?,header?,column_widths_in?,cant_split?} |
            format_text{find,bold?,italic?,underline?,strike?,size?,color?} |
            format_paragraph{find, line_spacing?, space_before?(pt), space_after?(pt), align?,
                 keep_with_next?, page_break_before?} — tune an existing paragraph by anchor text |
            add_hyperlink{find,url} — turn existing text into a real hyperlink |
            set_page_number{location:footer|header, align?, start?, of_total?, fmt?} |
            set_page_margins{...} | set_page_orientation{...} | set_header_footer{...} |
            add_image{image_path,after_text?,width?,height?}
      xlsx: create{sheets:[{name,headers?,rows?}]} |
            set_cell{sheet,cell,value, type?:auto|number|text|formula|date|bool, number_format?} |
            set_range{sheet,start_cell,rows, types?} |
            add_sheet{sheet,headers?,rows?} | delete_sheet{sheet} |
            set_cell_format{sheet,cell_range, bold?,italic?,font_color?,font_size?,font_name?,
                 fill_color?,align_h?,align_v?,wrap?,number_format?,border?} |
            add_table{sheet,cell_range,name?,headers?,style?} — make a real Excel Table (ListObject + filter) |
            add_chart{sheet,data_range,chart_type,anchor_cell?,title?,width?,height?} | clear_charts{sheet} |
            merge_cells | unmerge_cells | freeze_panes{sheet,cell} |
            set_column_width{sheet,columns,hidden?:[cols]} | set_row_height{sheet,rows,hidden?:[rownums]} |
            set_page_setup{sheet, orientation?:portrait|landscape, fit_to_width?, fit_to_height?, scale?,
                 paper_size?:a4|letter|legal|a3, margins?:{left,right,top,bottom}, center_h?, center_v?,
                 print_area?("A1:H40"), print_title_rows?("1:1"), print_title_cols?("A:A")} |
            rename_sheet{sheet,new} | hide_sheet{sheet} | show_sheet{sheet} |
            add_named_range | delete_named_range | add_data_validation | add_conditional_formatting |
            set_auto_filter | set_number_format | add_image
            number_format named enum: general/integer/number2/percent/percent2/currency_usd/
              currency_eur/accounting/date_iso/date_us/datetime/time/scientific/text (or raw fmt)
      pptx: create{slides:[{layout?,title:RichText,subtitle:RichText,body?,table?,notes:RichText}]} |
            body = {items:[{text:RichText, level:0-4, bullet:true|false}], autofit?} (or ["a","b"]) |
            add_slide{layout?,title?,subtitle?,body?,notes?,index?} |
            set_text{slide,placeholder:title|body|subtitle,text:RichText} |
            add_textbox{slide,text:RichText,x?,y?,w?,h?,autofit?,align_h?} |
            add_table{slide,rows,...} | add_image{slide,...} | set_notes{slide,text:RichText} |
            replace_text{find,replace,slide?} | add_shape{slide,shape,...} | add_chart{slide,...} |
            format_text{slide,find,...} | duplicate_slide{slide} | delete_slide{slide} |
            set_slide_size{preset:16:9|4:3|16:10  OR  width_in,height_in} — canvas size/aspect
            autofit = none|shrink_text|resize_shape (default shrink_text recommended)
      text (txt/md/csv/tsv/json/jsonl/html): content is written LITERALLY (no Markdown/
            HTML parsing — for a .md/.html file the markup IS its content).
            create{content?, rows?, data?, overwrite?} — new file; supply ONE of:
              content = the full file text (string); for .json a string/obj both work |
              rows    = list of rows → csv/tsv (list-of-lists, auto-quoted) |
                        jsonl (list of objects, one JSON per line) | txt/md (one per line) |
              data    = a JSON value → pretty-printed for .json (list → one-per-line for .jsonl)
            append{content? | rows?} — append (newline-separated) |
            replace_text{find, replace, count?} — literal find/replace
      any: export_pdf{out?} — set `path` to an existing docx/xlsx/pptx → converted to PDF
            (LibreOffice). Build the document first, then export_pdf.

    Args:
        path: Absolute path of the file to write — under /outputs for a final
            deliverable, or /workspace for an intermediate file. Not other dirs.
        ops: JSON array of operations; each item is a single-key object
            {op_name: {params}}, applied in order. May be a JSON array string, or
            "@/workspace/program.json" pointing to such an array. Omit it when
            you are using the content/rows/data shorthand below.
        content: Shorthand for a text file written in one shot — the literal
            file body. Equivalent to ops=[{"create": {"content": ...}}]. Text
            formats only (txt/md/csv/tsv/json/jsonl/html/htm).
        rows: Shorthand alternative to `content`: an array of csv/tsv rows, or
            of jsonl objects.
        data: Shorthand alternative to `content`: a JSON object/array to
            serialise into the file.
        overwrite: Allow the shorthand to rebuild a file that already exists.
            Without it an existing path is refused, so you never silently wipe
            prior content.

    Returns:
        A per-op result summary, or an error/hint message.
    """
    if not path or not path.strip():
        return "Error: path is required."
    ops, shorthand_error = desugar_text_shorthand(
        ops=ops, content=content, rows=rows, data=data, overwrite=overwrite,
    )
    if shorthand_error:
        return shorthand_error
    deliverable_error = output_write_error(path)
    if deliverable_error:
        return f"Error: {deliverable_error}"
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext not in _ALL_EXTS:
        return (f"[create_file] unsupported extension .{ext}; supports docx/xlsx/pptx "
                "and text formats txt/md/csv/tsv/json/jsonl/html/htm")

    # C1: the path must be under a write root (other directories are not mounted, so writes/reads spin on ENOENT).
    # Lexical only — the writer re-checks against the real filesystem it opens.
    roots = _write_roots()
    _bad_path = _outside_write_roots
    if _bad_path(path):
        return (f"Error: path must be under /workspace or /outputs (got {path!r}); "
                "other dirs like /home or /tmp are NOT mounted and will fail. "
                "Write final deliverables to /outputs, intermediates to /workspace.")

    if ops is None:
        return ('Error: `ops` is required — a JSON array of single-key ops, e.g. '
                '[{"create":{...}}, {"set_cell":{"sheet":"S","cell":"A1","value":1}}]')
    runtime_path = resolve_runtime_path(path)

    if isinstance(ops, str):
        s = ops.strip()
        if s.startswith("@"):               # @/workspace/program.json
            if _bad_path(s[1:]):
                return f"Error: ops file must be under /workspace (got {s[1:]!r})."
            runtime_program_path = resolve_runtime_path(s[1:])
            err = _ops_program_error(path, runtime_program_path)
            if err:
                return err
            # The program's own path is resolved above; the paths INSIDE it
            # need the same treatment, or an ``@program`` that names
            # /workspace/chart.png fails in native mode while the identical
            # inline array works. Inlining the (already validated) program is
            # how those fields reach ``_runtime_ops_paths`` at all.
            payload = _inlined_ops_program(runtime_program_path)
            if payload is None:
                payload = "@" + runtime_program_path
        else:
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                return f"Error: ops is a string but not valid JSON: {e}"
            if not isinstance(parsed, list):
                return 'Error: ops must be a JSON array, e.g. [{"set_cell":{...}}, ...]'
            err = _validate_ops(parsed) or _deliverable_ops_error(path, parsed)
            if err:
                return err
            payload = json.dumps(_runtime_ops_paths(parsed), ensure_ascii=False)
    elif isinstance(ops, list):
        err = _validate_ops(ops) or _deliverable_ops_error(path, ops)
        if err:
            return err
        payload = json.dumps(_runtime_ops_paths(ops), ensure_ascii=False)
    else:
        return f"Error: ops must be a list (or JSON array string), got {type(ops).__name__}"

    # The writer source goes over stdin (input=_WRITER_SRC), leaving only
    # `python3 - <argv>` on the command line — the same shape read_file uses for
    # its reader bundle. Echoing the bundle into argv instead overflows execve's
    # 128KB single-argument limit (MAX_ARG_STRLEN): the writer is ~131KB
    # base64-encoded, so *every* call failed with a bare "Argument list too
    # long".
    #
    # The roots ride along so the writer can re-check containment against the
    # filesystem it will actually open (see ``_escapes_write_roots``); the
    # lexical check above cannot follow symlinks in another namespace.
    writer_python = sys.executable if resolve_sandbox_mode() == "native" else "python3"
    cmd = (
        f"{shlex.quote(writer_python)} - {shlex.quote(runtime_path)} {shlex.quote(payload)} "
        f"{shlex.quote(json.dumps(roots))}"
    )
    # Bound the FINAL command, quoting included — see _MAX_COMMAND_BYTES. Done
    # before the sandbox is acquired: provisioning one only to discard the call
    # would cost a VM start on the remote backends.
    assembled = len(cmd.encode("utf-8"))
    if assembled > _MAX_COMMAND_BYTES:
        return (
            f"Error: this call assembles a {assembled:,}-byte command, over the "
            f"{_MAX_COMMAND_BYTES:,}-byte limit for one command line (shell "
            "quoting can expand the ops payload several times over). Write the "
            "JSON array to a workspace file and pass it by reference instead — "
            'create_file(path=..., ops="@/workspace/program.json") — which has '
            "no size limit."
        )

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as e:
        raise RuntimeError(f"create_file could not acquire a sandbox: {e}") from e

    try:
        result = await arun_sandbox_cmd(
            sandbox, cmd, timeout=_TIMEOUT, input=_WRITER_SRC,
        )
    except TimeoutError:
        return f"Error: create_file timed out after {_TIMEOUT}s"
    except Exception as e:
        logger.warning("create_file error for '%s': %s", path, e)
        raise RuntimeError(f"create_file failed for {path!r}: {e}") from e

    # Outside the try: a writer that ran and reported a failure is an ordinary
    # result, not an unexpected fault. Raising it inside would have it caught
    # by the handler above, re-wrapped with a second prefix and logged as if
    # the sandbox call itself had blown up.
    if result.exit_code != 0:
        raise RuntimeError(
            f"create_file writer exited {result.exit_code}: {_failure_detail(result)}"
        )
    return result.stdout or "(no output)"
