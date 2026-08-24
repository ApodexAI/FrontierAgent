
"""Writer for text deliverables: txt/md/csv/tsv/json/jsonl/html.

Symmetric with the three office formats, so create_file is the only deliverable tool (text does not have to fall back to bash).
Content is written literally, with no Markdown/HTML parsing — for .md/.html the "formatting" is the literal content.
ops: create (new file) / append / replace_text (incremental edit).
"""
import csv as _csv
import io as _io
import json as _json
import os as _os


def _text_encode(ext, args):
    """Encode args into the text to write, by extension. Priority: content (literal string) > data (json) > rows.
    Returns a string; None when all three are absent."""
    if args.get("content") is not None:
        c = args["content"]
        return c if isinstance(c, str) else _json.dumps(c, ensure_ascii=False, indent=2)
    if args.get("data") is not None:
        data = args["data"]
        if ext == "jsonl":
            seq = data if isinstance(data, (list, tuple)) else [data]
            return "".join(_json.dumps(x, ensure_ascii=False) + "\n" for x in seq)
        return _json.dumps(data, ensure_ascii=False, indent=2)
    if args.get("rows") is not None:
        rows = args["rows"]
        if ext in ("csv", "tsv"):
            buf = _io.StringIO()
            w = _csv.writer(buf, delimiter="\t" if ext == "tsv" else ",", lineterminator="\n")
            for r in rows:
                w.writerow(r if isinstance(r, (list, tuple)) else [r])
            return buf.getvalue()
        if ext == "jsonl":
            return "".join(_json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if ext == "json":
            # .json must produce valid JSON: rows are serialised as one array (line-by-line concatenation yields a broken file)
            return _json.dumps(rows, ensure_ascii=False, indent=2)
        # txt/md/html: one element per line
        return "\n".join(r if isinstance(r, str) else _json.dumps(r, ensure_ascii=False)
                         for r in rows)
    return None


def _text_write(path, op, args):
    """One op for text formats: create / append / replace_text."""
    ext = _ext(path)

    if op == "create":
        if _os.path.exists(path) and not args.get("overwrite"):
            return (f"[error] {path} already exists — use append/replace_text for an incremental edit,"
                    ' or pass "overwrite":true to rebuild it entirely.')
        content = _text_encode(ext, args)
        if content is None:
            return ('[error] create needs one of "content" (literal string), "data" (json object/array)'
                    ' or "rows" (an array of csv/tsv rows, or of jsonl objects).')
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        nlines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return _res(f"created {ext}: {path}", counts={"line": nlines})

    if op == "append":
        content = _text_encode(ext, args)
        if content is None:
            return '[error] append needs one of "content", "data" or "rows".'
        sep = ""
        if _os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = f.read()
            if old and not old.endswith("\n"):
                sep = "\n"
        else:
            _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(sep + content)
        return _res(f"appended to {path}")

    if op in ("replace_text", "replace"):
        find = args.get("find")
        repl = args.get("replace", "")
        if not find:
            return '[error] replace_text needs "find".'
        if not _os.path.exists(path):
            return f"[error] {path} not found."
        with open(path, encoding="utf-8") as f:
            s = f.read()
        n = s.count(find)
        cnt = args.get("count")
        # cnt=0 must still be honoured literally (replace 0 occurrences); a truthiness check here would fall through to replacing everything
        s2 = s.replace(find, repl, int(cnt)) if cnt is not None else s.replace(find, repl)
        done = min(n, int(cnt)) if cnt is not None else n
        with open(path, "w", encoding="utf-8") as f:
            f.write(s2)
        if done == 0:
            if cnt is not None and int(cnt) == 0:
                return _res(
                    f"replace_text: 0 replacements requested for {find!r}; file unchanged",
                )
            return _res(f"replace_text: 0 matches for {find!r}", warn=f"no match for {find!r}")
        return _res(f"replace_text: {done} replacement(s) in {path}")

    return f"[error] unsupported op {op!r} for .{ext} (text formats support create/append/replace_text)."
