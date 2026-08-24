
"""office_writer core: the shared _ensure plus per-format dispatch of write(path, op, args).
Incremental load-modify-save; anchors are stable values (xlsx = A1 / sheet name, pptx = slide / placeholder, docx = anchor text).
Feature set informed by Mercor-Intelligence/archipelago (Apache-2.0); the addressing and parameter design are this project's own.
"""
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile


def _ensure(mod: str, pkg: str) -> None:
    try:
        __import__(mod)
        return
    except ImportError as original:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"missing dependency {mod!r}; automatic install of {pkg!r} "
                f"with {sys.executable!r} failed (exit {result.returncode}): "
                f"{detail or original}"
            ) from original
        try:
            __import__(mod)
        except ImportError as installed_error:
            raise RuntimeError(
                f"installed {pkg!r} with {sys.executable!r}, but {mod!r} is "
                f"still not importable: {installed_error}"
            ) from installed_error


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _norm_runs(text):
    """RichText normalisation (shared by docx/pptx): turn ``text`` into a list of run dicts [{text, bold?, ...}].
    - a plain string → a single run;
    - a list → per item: a string becomes {text}; a dict must contain 'text' and is kept
      as-is (optional bold/italic/underline/
      strike/color/size/font/link fields are left for each format writer to interpret);
    - None / empty → [].
    ``text`` is always literal — no markdown parsing (formatting only via run fields)."""
    if text is None or text == "":
        return []
    if isinstance(text, str):
        return [{"text": text}]
    if isinstance(text, dict):
        return [text] if text.get("text") is not None else []
    if isinstance(text, (list, tuple)):
        out = []
        for r in text:
            if isinstance(r, str):
                out.append({"text": r})
            elif isinstance(r, dict) and r.get("text") is not None:
                out.append(r)
        return out
    return [{"text": str(text)}]


_WRITERS = {"docx": "_docx_write", "xlsx": "_xlsx_write",
            "pptx": "_pptx_write"}

# Text deliverables (_writer_text._text_write): written literally, with no Markdown/HTML parsing.
_TEXT_EXTS = {"txt", "md", "csv", "tsv", "json", "jsonl", "html", "htm"}

# Detect Markdown-looking text and add a hint to the return value (a teaching signal), but never convert it.
_MD_PATTERNS = [
    (r"\*\*[^*\n]+\*\*", "**bold** → use a run's bold field"),
    (r"\[[^\]\n]+\]\((?:https?|mailto):", "[text](url) → use a run's link field / add_hyperlink"),
    (r"(?m)^\s*(?:[-*]|\d+\.)\s+\S", "leading - / 1. → use the list / body.items parameters"),
    (r"(?m)^\s*#{1,6}\s+\S", "# heading → use a heading block + level"),
    (r"~~[^~\n]+~~", "~~strike~~ → use a run's strike field"),
]


def _collect_strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_strings(v, out)


def _md_hint(args):
    """Scan every string in args and return one hint line when Markdown traces are found (content is not modified)."""
    strs = []
    _collect_strings(args, strs)
    blob = "\n".join(strs)
    hits = [msg for pat, msg in _MD_PATTERNS if re.search(pat, blob)]
    if not hits:
        return ""
    return ("\nhint: the text looks like it contains Markdown ("  + "; ".join(hits)
            + "). This tool writes literally and does not parse Markdown — use the matching k-v parameters instead.")


def _archive_intent(op, path, args):
    """Archive the args this create_file/edit_file actually used (whether inline or read from params_file),
    so what the model intended to write can be inspected afterwards. Written to
    /workspace/.office_writer_intent/NNN_op.json (persists in the worktree,
    stays out of /outputs, never overwritten). Best-effort: a failure does not affect the main path."""
    import os
    try:
        d = "/workspace/.office_writer_intent"
        os.makedirs(d, exist_ok=True)
        n = len([f for f in os.listdir(d) if f.endswith(".json")])
        rec = {"seq": n, "op": op, "path": path, "args": args}
        with open(os.path.join(d, f"{n:03d}_{op}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _export_pdf(src, out):
    """Convert an office file (docx/xlsx/pptx) to pdf with LibreOffice.
    src = source file, out = target pdf path. Concurrency-safe (a separate UserInstallation per process)."""
    import os
    import shutil
    if not shutil.which("soffice"):
        return "[office_writer error] soffice/LibreOffice not available for pdf export"
    if not os.path.exists(src):
        return f"[office_writer error] source not found: {src}"
    outdir = os.path.dirname(out) or "."
    os.makedirs(outdir, exist_ok=True)
    prof = f"/tmp/lo_pdf_{os.getpid()}"
    try:
        r = subprocess.run(
            ["soffice", "--headless", f"-env:UserInstallation=file://{prof}",
             "--convert-to", "pdf", "--outdir", outdir, src],
            capture_output=True, text=True, timeout=180)
    except Exception as e:
        return f"[office_writer error] soffice convert failed: {e}"
    produced = os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + ".pdf")
    if not os.path.exists(produced):
        return f"[office_writer error] pdf not produced: {(r.stdout or r.stderr)[:200]}"
    if os.path.abspath(produced) != os.path.abspath(out):
        shutil.move(produced, out)
    return f"exported pdf: {out} (from {os.path.basename(src)})"


def _read_at(raw):
    """A payload starting with @<file> is read as JSON from a sandbox file (so the LLM need not hand-assemble long JSON). None on failure."""
    if raw.startswith("@"):
        fp = raw[1:]
        try:
            with open(fp, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            sys.stdout.write(f"[office_writer error] cannot read file {fp!r}: {e}")
            return None
    return raw


# ---------- Structured results + receipt (successes aggregate to one line; only problems expand) ----------

def _res(summary, *, ok=True, warn=None, wrote_formula=False, counts=None):
    """An op's structured return. summary = one sentence; warn = a non-fatal problem (0 matches, anchor not found) and execution continues;
    ok=False = fatal (stops the batch); wrote_formula = whether this op wrote a formula (triggers a recalc);
    counts = {singular label: count} (a create's write receipt, e.g. {'paragraph':6,'table':2})."""
    return {"ok": ok, "summary": summary, "warn": warn,
            "wrote_formula": wrote_formula, "counts": counts or {}}


def _norm_result(op, r):
    """Normalise an op return (str or dict) into a result dict. A bare string starting with [error is treated as fatal."""
    base = {"ok": True, "summary": "", "warn": None, "wrote_formula": False, "counts": {}}
    if isinstance(r, dict):
        base.update(r)
        base["op"] = op
        return base
    s = str(r)
    base["ok"] = not (s.startswith("[error") or s.startswith("[office_writer error"))
    base["summary"] = s
    base["op"] = op
    return base


def _fmt_counts(counts):
    """{'paragraph':6,'table':2,'heading':1} → '6 paragraphs, 2 tables, 1 heading' (zeros omitted)."""
    parts = [f"{n} {label}" + ("" if n == 1 else "s") for label, n in counts.items() if n]
    return ", ".join(parts) if parts else "nothing"


def _breakdown(items):
    """An op that succeeded with no problems folds into a count line; create expands its write receipt."""
    tally, order, creates = {}, [], []
    for d in items:
        if d["op"] == "create" and d.get("counts"):
            creates.append("create(" + _fmt_counts(d["counts"]) + ")")
        else:
            if d["op"] not in tally:
                order.append(d["op"])
            tally[d["op"]] = tally.get(d["op"], 0) + 1
    return ", ".join(creates + [f"{tally[o]} {o}" for o in order])


def _format_receipt(path, results, total, stopped_at, recalc_line):
    """Fold every op result into a compact receipt: 2-4 lines for a clean batch, one expanded entry per problem."""
    if stopped_at is not None:
        f = results[-1]
        lines = [f"✗ {path} — STOPPED at op {stopped_at}/{total}",
                 f"  [{f['idx']}] {f['op']}: {f['summary']}"]
        if stopped_at < total:
            tail = f"; file saved with ops 1-{stopped_at - 1}" if stopped_at > 1 else ""
            lines.append(f"  ops {stopped_at + 1}-{total} not executed{tail}")
        if recalc_line:
            lines.append("  " + recalc_line)
        return "\n".join(lines)

    anomalies = [d for d in results if d.get("warn")]
    ok_clean = [d for d in results if not d.get("warn")]

    if total == 1:
        d = results[0]
        base = d["summary"] or d["op"]
        head = f"⚠ {base} — {d['warn']}" if d.get("warn") else f"✓ {base}"
        out = [head]
        if recalc_line:
            out.append("  " + recalc_line)
        return "\n".join(out)

    note = f", {len(anomalies)} wrote nothing" if anomalies else ""
    lines = [f"✓ {path} — {total} ops applied{note}"]
    bd = _breakdown(ok_clean)
    if bd:
        lines.append("  " + bd)
    for d in anomalies:
        lines.append(f"  ⚠ [{d['idx']}] {d['op']}: {d['warn']}")
    if recalc_line:
        lines.append("  " + recalc_line)
    return "\n".join(lines)


# ---------- xlsx in-place recalc (openpyxl only writes the formula string, so cached values are filled in here and formula errors reported) ----------

_X_MACRO = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" \
script:language="StarBasic">
    Sub RecalculateAndSave()
      ThisComponent.calculateAll()
      ThisComponent.store()
      ThisComponent.close(True)
    End Sub
</script:module>"""

_XL_ERRORS = {"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!",
              "#VALUE!", "#SPILL!", "#CALC!", "#GETTING_DATA"}


def _xlsx_recalc(path):
    """Recalculate formulas in place with LibreOffice (calculateAll + store) and scan for formula errors.
    Returns ``(list of error strings, whether the convert fallback was used)``; None when soffice is unavailable or fails."""
    import os
    import shutil
    used_fallback = False
    if not shutil.which("soffice"):
        return None
    # The sandbox uses preexec to drop the RLIMIT_AS (virtual memory) soft limit to 12GB,
    # and the LibreOffice macro path (calculateAll
    # has to load the full UNO scripting framework) needs more VSZ than that, so it hangs
    # or fails. preexec only lowered the soft limit and kept hard,
    # so raise soft back to hard here (the soffice child inherits it) and let the recalc run.
    try:
        import resource
        _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (_hard, _hard))
    except Exception:
        pass
    try:
        env = dict(os.environ)
        env["SAL_USE_VCLPLUGIN"] = "svp"
        # Inside the bwrap sandbox HOME=/root does not exist → LibreOffice cannot create a profile and soffice fails silently (EXIT=1).
        # Force a writable HOME: /tmp is bound writable inside the sandbox and exists on the host too.
        home = env.get("HOME", "")
        if not (home and os.path.isdir(home) and os.access(home, os.W_OK)):
            home = "/tmp"
        env["HOME"] = home
        macro_dir = os.path.join(home, ".config/libreoffice/4/user/basic/Standard")
        if not os.path.isdir(macro_dir):  # first run: initialise the profile
            subprocess.run(["soffice", "--headless", "--terminate_after_init"],
                           capture_output=True, timeout=30, env=env)
        os.makedirs(macro_dir, exist_ok=True)
        mf = os.path.join(macro_dir, "Module1.xba")
        try:
            have = "RecalculateAndSave" in open(mf, encoding="utf-8").read()
        except OSError:
            have = False
        if not have:
            with open(mf, "w", encoding="utf-8") as fh:
                fh.write(_X_MACRO)
        cmd = ["soffice", "--headless", "--norestore",
               "vnd.sun.star.script:Standard.Module1.RecalculateAndSave"
               "?language=Basic&location=application", os.path.abspath(path)]
        # Budget: 60s for the macro path + 45s for the fallback = 105s < create_file's outer
        # 120s, so when the fallback is genuinely needed
        # (macro path slow and ineffective) the model still gets a receipt, instead of the
        # outer timeout firing with the file already changed and nobody knowing.
        r = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        if r.returncode != 0:
            return None
        # The macro path can silently do nothing (returncode 0 but no cached values written,
        # e.g. an environment where the macro framework never loaded) — so verify, then
        # fall back to the convert path (soffice --convert-to xlsx re-saves; the same trick the judge's render side uses, already proven).
        if _xlsx_cache_empty(path):
            import shutil as _sh
            import tempfile as _tf
            td = _tf.mkdtemp(prefix="recalc_")
            try:
                rc = subprocess.run(["soffice", "--headless", "--norestore", "--convert-to",
                                     "xlsx", "--outdir", td, os.path.abspath(path)],
                                    capture_output=True, timeout=45, env=env)
                out = os.path.join(td, os.path.basename(path))
                if rc.returncode == 0 and os.path.exists(out):
                    _sh.move(out, path)
                    used_fallback = True
                else:
                    return None
            finally:
                _sh.rmtree(td, ignore_errors=True)   # the fallback temp dir is removed as soon as it is used
    except Exception:
        return None
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        errs = []
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    if isinstance(c.value, str) and c.value in _XL_ERRORS:
                        errs.append(f"{ws.title}!{c.coordinate} {c.value}")
        return errs, used_fallback
    except Exception:
        return [], used_fallback


def _xlsx_cache_empty(path):
    """Whether any cell has "a formula but an empty calculation cache" (the probe for a silently failed macro recalc).

    Runs a byte-level regex directly on the sheet XML inside the zip, bypassing the openpyxl object model:
    a formula cell is ``<c ...><f>…</f><v>cache</v></c>``, and all three empty-cache shapes
    must be caught: **no <v>** after f,
    an **empty <v></v>** (what openpyxl actually writes for a new formula), and a
    self-closing **<v/>**; a shared-formula
    dependent cell is a self-closing ``<f t="shared" si=…/>``. Any hit means invalid. The
    scan is exhaustive and never truncated (no
    false negatives), and even a 100MB-scale sheet takes hundreds of milliseconds — one to
    two orders of magnitude faster than comparing two
    workbook views cell by cell.

    Exception: an empty ``<v></v>`` under ``<c t="str">`` is a **computed empty-string result**
    (the legitimate cache of a string formula),
    not an uncalculated cell — excluded, or such sheets would run LibreOffice pointlessly on every batch."""
    import re as _r
    import zipfile as _z
    # A variable-length (?<!…) is not allowed, so t="str" is excluded with a negative lookahead inside the <c open tag:
    # match a whole cell that "starts with <c, has no t="str" attribute, contains a formula, and has an empty cache".
    pat = _r.compile(
        rb'<c(?![^>]*\bt="str")[^>]*>\s*(?:<f\b[^>]*/>|<f\b[^>]*>[^<]*</f>)'
        rb'\s*(?:<v\s*/>|<v>\s*</v>)?\s*</c>')
    try:
        with _z.ZipFile(path) as zf:
            for name in zf.namelist():
                if (
                    name.startswith("xl/worksheets/sheet")
                    and name.endswith(".xml")
                    and pat.search(zf.read(name))
                ):
                    return True
        return False
    except Exception:
        return False


def _escapes_write_roots(path, roots):
    """True when *path*'s REAL location falls outside every allowed root.

    Must run here, in the execution namespace, rather than in the caller: only
    this process shares a filesystem with the ``open()`` below, so only here
    does ``realpath`` follow the same symlinks that write will. A lexical check
    on the caller's side (``normpath``) cannot see them, so a link planted
    inside an allowed root — as a parent component or as the final name —
    would redirect the write anywhere the runtime can reach.

    ``realpath`` on a path that does not exist yet still resolves the parents
    that DO exist, which is exactly the symlinked-parent case.
    """
    if not roots:
        return False
    real = os.path.realpath(path)
    for root in roots:
        r = os.path.realpath(root)
        if real == r or real.startswith(r + os.sep):
            return False
    return True


def _spill_roots():
    """The recovery store's roots, resolved without importing the harness.

    This module is CONCATENATED into a standalone script and run inside the
    sandbox (see ``_create_file._PARTS``), where the ``plugins`` package does not
    exist — so it cannot call ``_sandbox.is_spill_path``, which is the authority
    for this rule. Deliberate duplication: the precedence below mirrors
    ``_sandbox.spill_root`` and ``tests/test_agent_team_workflow.py`` asserts the
    two agree, so a change to one fails on the other.

    Both names are covered because both are reachable: inside a sandbox the model
    can only name the canonical mount, while under ``native`` the writer runs on
    the host against the physical path.
    """
    roots = ["/spill"]
    explicit = os.environ.get("APODEX_SPILL_DIR", "").strip()
    if explicit:
        roots.append(explicit)
    else:
        run_dir = os.environ.get("APODEX_RUN_DIR", "").strip()
        roots.append(
            os.path.join(run_dir, "spill") if run_dir
            else os.path.join(
                tempfile.gettempdir(), f"apodex-spill-{os.getuid()}",
            )
        )
    return roots


def _is_spill_path(path):
    """Whether *path* enters the recovery store, lexically or once resolved."""
    raw = str(path or "").strip()
    if not raw:
        return False
    candidates = [os.path.normpath(raw)]
    with contextlib.suppress(OSError):
        candidates.append(os.path.realpath(os.path.expanduser(raw)))
    for candidate in candidates:
        for root in _spill_roots():
            root = os.path.normpath(root)
            if candidate == root or candidate.startswith(root + os.sep):
                return True
    return False


def _write_root_error(path, roots):
    if _is_spill_path(path):
        return (f"[office_writer error] refusing to write {path}: the recovery "
                "store is read-only")
    if not _escapes_write_roots(path, roots):
        return None
    return (f"[office_writer error] refusing to write {path}: it resolves to "
            f"{os.path.realpath(path)}, outside the allowed roots "
            f"{', '.join(roots)}")


def _run_op(path, op, args):
    """Run one operation, returning str or a structured dict (normalised by _norm_result)."""
    if op == "export_pdf":  # format-independent: convert an office file to pdf
        import os as _o
        out = args.get("out") or args.get("output") or (_o.path.splitext(path)[0] + ".pdf")
        return _export_pdf(path, out)
    ext = _ext(path)
    if ext in _TEXT_EXTS:
        return _text_write(path, op, args)
    if ext not in _WRITERS:
        return f"[office_writer] unsupported extension .{ext}"
    return globals()[_WRITERS[ext]](path, op, args)


def _abort(message: str) -> SystemExit:
    """Report a refusal on stdout and return the exception to raise.

    Every way out of ``main`` that did not write the file has to exit non-zero:
    ``create_file`` decides success from the exit code, so a bare ``return``
    here would present a containment refusal or a malformed payload to the
    model as a completed write.
    """
    sys.stdout.write(message)
    return SystemExit(1)


def main() -> None:
    # Call shape: <path> <ops_json|@file>   ops = [{op_name: {params}}, ...] (executed in order)
    if len(sys.argv) < 2:
        raise _abort("[office_writer error] usage: <path> <ops_json|@file>")
    path = sys.argv[1]
    raw = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
    # Optional 4th word: the roots this write must stay inside, as a JSON
    # array. Absent (a direct/standalone invocation) => no containment check.
    try:
        roots = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
    except json.JSONDecodeError:
        roots = []
    roots = [r for r in roots if isinstance(r, str)] if isinstance(roots, list) else []
    err = _write_root_error(path, roots)
    if err:
        raise _abort(err)
    raw = _read_at(raw)
    if raw is None:
        raise SystemExit(1)      # _read_at already wrote the reason
    try:
        data = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as e:
        raise _abort(f"[office_writer error] bad ops_json: {e}") from e
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise _abort("[office_writer error] ops must be a JSON array, e.g. "
                     '[{"set_cell":{...}}, {"set_cell_format":{...}}]')
    ops = []
    for it in data:
        # Each entry = a single-key object {op_name: {params}}; exactly one key is enforced, so several ops cannot be crammed into one object
        if not isinstance(it, dict) or len(it) != 1:
            raise _abort("[office_writer error] each ops item must be a single-key object "
                         '{op: {params}}, e.g. {"set_cell": {"sheet":"S","cell":"A1","value":1}}')
        op, a = next(iter(it.items()))
        if a is None:
            a = {}
        if not isinstance(a, dict):
            raise _abort(f"[office_writer error] params for op {op!r} must be an object, "
                         f"got {type(a).__name__}")
        ops.append((op, a))

    total = len(ops)
    results, hints, stopped_at = [], [], None
    for i, (op, args) in enumerate(ops):
        # ``export_pdf`` writes a SECOND file; it needs the same containment as
        # ``path`` or it becomes the way around the check above.
        secondary = str(args.get("out") or args.get("output") or "").strip()
        if secondary:
            err = _write_root_error(secondary, roots)
            if err:
                raise _abort(err)
        _archive_intent(op, path, args)  # leave a trace per op (never overwritten)
        try:
            raw_r = _run_op(path, op, args)
        except Exception as e:
            raw_r = f"[office_writer error] {type(e).__name__}: {e}"
        d = _norm_result(op, raw_r)
        d["idx"] = i + 1
        results.append(d)
        # The Markdown hint applies to office formats only (text uses run fields); in a text deliverable, Markdown/HTML *is* the literal content, so no hint.
        h = "" if _ext(path) in _TEXT_EXTS else _md_hint(args)
        if h:
            hints.append(h)
        if not d["ok"]:           # fatal: stop the batch, later ops do not run
            stopped_at = i + 1
            break

    # xlsx and this batch wrote a formula → recalc once in place after the batch, and feed formula errors back into the receipt
    recalc_line = None
    if stopped_at is None and _ext(path) == "xlsx":
        nf = sum(1 for d in results if d.get("wrote_formula"))
        if nf:
            _r = _xlsx_recalc(path)
            if _r is None:
                recalc_line = ("recalc: could not run (soffice missing or failed in this "
                               "sandbox) — formula caches may be empty")
            else:
                errs, used_fallback = _r
                if errs:
                    shown = "  ".join(errs[:10]) + (" …" if len(errs) > 10 else "")
                    recalc_line = (
                        f"recalc: {nf} formula op(s), {len(errs)} error(s): {shown}"
                    )
                else:
                    recalc_line = f"recalc: {nf} formula op(s), 0 errors"
                if used_fallback:
                    # The convert fallback is a whole-workbook LibreOffice re-save, so charts,
                    # validation and the like are preserved on a best-effort basis —
                    # a lossy rewrite must not happen silently
                    recalc_line += (
                        "  (workbook re-saved by LibreOffice to fill formula "
                        "caches; chart/validation fidelity best-effort)"
                    )

    seen, uniq = set(), []
    for h in hints:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    sys.stdout.write(_format_receipt(path, results, total, stopped_at, recalc_line) + "".join(uniq))
    if stopped_at is not None:
        raise SystemExit(1)
