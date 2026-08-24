
"""read_file parsers · core fragment (concatenated first)."""
import hashlib
import json
import os as _os
import re as _re
import subprocess
import sys

_CACHE_DIR = "/workspace/.readdoc_cache"  # bind-mounted, persists across commands; missing/unwritable → silently re-render

# Diagnostic trace: only when env READDOC_TRACE is set, record each step's routing and
# backend calls and dump them to stderr wrapped in sentinels when main() ends —
# stdout is always clean markdown (what the LLM sees); the trace is for debugging only.
_TRACE: list = []


def _trace(ev: dict) -> None:
    if _os.environ.get("READDOC_TRACE"):
        _TRACE.append(ev)


def _dump_trace() -> None:
    if _TRACE:
        sys.stderr.write("\n<<<READDOC_TRACE\n" + json.dumps(_TRACE, ensure_ascii=False)
                         + "\nREADDOC_TRACE>>>\n")


def _ensure(mod: str, pkg: str) -> None:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# Vision reading (charts / figures / text inside an image): the reader calls the gateway
# or a self-hosted VLM (OpenAI-compatible chat/completions) itself and inlines the
# result into the readout — view_image is a low-level tool not exposed to the main LLM,
# so the reader has to read and return images on its own.
# Config (env, reachable inside the sandbox): READDOC_VISION_URL (include /v1) /
# READDOC_VISION_MODEL / READDOC_VISION_KEY.
_VISION_PROMPT = (
    "Reproduce ALL content of this image faithfully and completely; do not summarize or guess. "
    "First, one line: what it is (chart/diagram/table/form/photo/screenshot). Then: transcribe text "
    "verbatim (exact numbers/units/labels); for any table preserve rows/columns and which cell each "
    "value belongs to; for a chart/diagram give title, axes, legend, series and the values/relationships "
    "it conveys; for purely visual elements describe only what carries information. Keep reading order. "
    "Use [illegible] rather than guessing."
)
# Figures only (for mixed pages): body text and ordinary tables were already extracted
# by OCR, so this only adds charts/figures and avoids duplicating the text.
_VISION_FIGURE_PROMPT = (
    "This is a full page image that may contain charts/plots/diagrams/flowcharts/infographics, "
    "possibly alongside body text and plain tables. Extract ONLY the visual figures — for EACH figure: "
    "its title, axis labels and scales, legend, data series, and the concrete values or relationships it "
    "conveys (read approximate values off the axes when not labeled). Do NOT transcribe ordinary paragraph "
    "text, headings, or plain data tables — those are captured elsewhere. Process figures in reading order; "
    "use [illegible] rather than guessing. If the page has no real figure (only text/logos), reply exactly: NO_FIGURE."
)


def _vision_read(img_bytes: bytes, mime: str = "image/png", question: str | None = None,
                 timeout: int = 120):
    """One image → VLM text. READDOC_VISION_URL unset or a failure → None (the caller
    falls back).
    timeout: lower for batch reads (45s) so the whole batch converges inside the outer
    sandbox timeout."""
    base = _os.environ.get("READDOC_VISION_URL", "").rstrip("/")
    if not base:
        return None
    import base64 as _b64
    import json as _json
    import urllib.request as _ur
    key = _os.environ.get("READDOC_VISION_KEY", "EMPTY")
    model = _os.environ.get("READDOC_VISION_MODEL", "")
    import time as _t
    b = _b64.b64encode(img_bytes).decode("ascii")
    # Reasoning models (Qwen3 and friends) burn the budget on reasoning by default →
    # max_tokens exhausted, content empty, and very slow
    # (measured: 130s for 4096 tok, still finish_reason=length with content=None).
    # Turning thinking off answers directly — faster and more accurate.
    # Most OpenAI-compatible servers ignore unknown fields; if an endpoint rejects it,
    # set READDOC_VISION_THINK=1 to disable this switch.
    payload = {"model": model, "max_tokens": 4096, "temperature": 0.2, "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b}"}},
            {"type": "text", "text": question or _VISION_PROMPT}]}]}
    if _os.environ.get("READDOC_VISION_THINK", "0") != "1":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = _t.time()
    try:
        req = _ur.Request(base + "/chat/completions", data=_json.dumps(payload).encode("utf-8"),
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = _json.loads(_ur.urlopen(req, timeout=timeout).read())
        out = r["choices"][0]["message"]["content"]
        _trace({"stage": "vision", "model": model, "bytes": len(img_bytes),
                "ms": int((_t.time() - t0) * 1000), "ok": bool(out),
                "figure_only": question is not None})
        return out
    except Exception as exc:
        if "chat_template_kwargs" in payload:
            payload.pop("chat_template_kwargs", None)
            try:
                req = _ur.Request(base + "/chat/completions", data=_json.dumps(payload).encode("utf-8"),
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                r = _json.loads(_ur.urlopen(req, timeout=timeout).read())
                out = r["choices"][0]["message"]["content"]
                _trace({"stage": "vision", "model": model, "bytes": len(img_bytes),
                        "ms": int((_t.time() - t0) * 1000), "ok": bool(out),
                        "figure_only": question is not None, "retry": True})
                return out
            except Exception:
                pass
        _trace({"stage": "vision", "model": model, "bytes": len(img_bytes),
                "ms": int((_t.time() - t0) * 1000), "ok": False, "error": str(exc)})
        return None


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


# ------------- Unstructured files: sniff and route (text / image·audio·video / unsupported binary) -------------
# Decided by content, not by an extension whitelist, so unlisted formats are still
# handled. Rule: never pipe binary bytes into the text channel.

# Bytes allowed in the text verdict: common control chars (BEL/BS/TAB/LF/FF/CR/ESC)
# plus 0x20-0xFF (covers utf-8 / latin-1 high bytes)
_TEXT_BYTES = bytes([7, 8, 9, 10, 12, 13, 27]) + bytes(range(0x20, 0x100))

# These formats are returned verbatim with **no line numbers**: json/yaml have to stay
# parseable and md has to stay renderable, so line numbers are pure noise;
# they only help for code / logs / plain txt (file:line references, str_replace edits).
_NO_LINENO_EXTS = {"json", "yaml", "yml", "md", "markdown"}


def _is_text(head: bytes) -> bool:
    """Does the first block look like text? A text BOM (utf-8/16/32) settles it as text
    (utf-16's NUL does not count as binary);
    otherwise a NUL → binary; failing that, by "share of non-text control bytes < 30%"
    (the same test file(1) and git use)."""
    if not head:
        return True
    if (head[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")
            or head[:3] == b"\xef\xbb\xbf" or head[:2] in (b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in head:
        return False
    nontext = head.translate(None, _TEXT_BYTES)
    return len(nontext) / len(head) < 0.30


def _magic_mime(head: bytes):
    """Binary magic bytes → mime (only to name image/audio/video); None when unknown."""
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:2] == b"BM":
        return "image/bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head[:4] == b"RIFF":
        sub = head[8:12]
        if sub == b"WEBP":
            return "image/webp"
        if sub == b"WAVE":
            return "audio/wav"
        if sub == b"AVI ":
            return "video/x-msvideo"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head[:4] == b"fLaC":
        return "audio/flac"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[4:8] == b"ftyp":
        return "audio/mp4" if head[8:11] == b"M4A" else "video/mp4"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video/x-matroska"
    if head[:3] == b"FLV":
        return "video/x-flv"
    return None


def _decode_bytes(raw: bytes) -> str:
    """Bytes → text, CJK-friendly. Order: BOM → strict utf-8 → charset_normalizer
    detection
    + scoring of verified CJK/Cyrillic/Western candidates → utf-8 replace as the floor.
    The old chain's BOM-less utf-16 attempt (any even-length byte string can "succeed"
    into garbage) and its latin-1 floor
    (SJIS mapped byte-by-byte into Latin mojibake) were the root cause of garbled
    non-major-language text, and are gone."""
    if not raw:
        return ""
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig", errors="replace")
    if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return raw.decode("utf-32", errors="replace")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Not utf-8: score the candidates. charset_normalizer's verdict competes rather than
    # deciding — measured, it misreads short Shift-JIS as EUC-KR, and calls
    # KOI8-R / Thai shift_jis (chaos=0 but coherence=0, i.e. no linguistic evidence at
    # all). Score every strictly-decodable candidate by character range:
    # full-width kana = strong Japanese signal; an all-half-width-katakana document is
    # the hallmark of a single-byte encoding misread by cp932, so it scores negative;
    # the normalizer candidate gets a small bonus as a tie-breaker (when it is right, it
    # should win).
    def _score(t: str, bonus: float = 0.0) -> float:
        if not t:
            return -1.0
        s = 0.0
        upper = lower = 0
        non_ascii = 0
        ascii_alpha = 0
        scripts = {
            "kana": 0, "hangul": 0, "cjk": 0, "latin": 0,
            "greek": 0, "cyrillic": 0, "arabic": 0, "hebrew": 0, "thai": 0,
        }
        sample = t[:8000]                   # score the first 8K chars; keeps big files fast
        for ch in sample:
            o = ord(ch)
            if o < 0x80:
                if ch.isalpha():
                    ascii_alpha += 1
                continue
            non_ascii += 1
            if ch.isupper():
                upper += 1
            elif ch.islower():
                lower += 1
            if 0x3040 <= o <= 0x30FF:      # full-width hiragana/katakana: strong Japanese signal
                s += 1
                scripts["kana"] += 1
            elif 0xAC00 <= o <= 0xD7A3:
                s += 1
                scripts["hangul"] += 1
            elif 0x4E00 <= o <= 0x9FFF:
                s += 1
                scripts["cjk"] += 1
            elif 0x00C0 <= o <= 0x024F:
                s += 1
                scripts["latin"] += 1
            elif 0x0370 <= o <= 0x03FF:
                s += 1
                scripts["greek"] += 1
            elif 0x0400 <= o <= 0x052F:
                s += 1
                scripts["cyrillic"] += 1
            elif 0x0590 <= o <= 0x05FF:
                s += 1
                scripts["hebrew"] += 1
            elif 0x0600 <= o <= 0x06FF:
                s += 1
                scripts["arabic"] += 1
            elif 0x0E00 <= o <= 0x0E7F:
                s += 1
                scripts["thai"] += 1
            elif 0xFF61 <= o <= 0xFF9F:    # half-width katakana: mis-decode hallmark (real Japanese is mostly full-width)
                s -= 2
            elif o == 0xFFFD or 0xE000 <= o <= 0xF8FF or 0x80 <= o <= 0x9F:
                s -= 5                      # replacement char / private use / C1 control: traces of a mis-decode
        # Denominator = number of non-ASCII chars: a single-byte codec misreading a
        # double-byte stream produces 2x the characters and halves the ASCII-space share,
        # so using total length would inflate the mis-decode's density past the correct one.
        score = s / max(non_ascii, 1) + bonus
        # Inverted-case penalty: the hallmark of KOI8 ↔ cp125x cross-decoding (the two
        # families swap case ranges, producing "lowercase initial, rest uppercase").
        # Normal text has far more lowercase than uppercase; caseless scripts are unaffected.
        cased = upper + lower
        if cased >= 10 and upper / cased > 0.6:
            score -= 0.5
        # Mixed writing systems are usually a strong signal of a misread double-byte
        # encoding (e.g. GBK → CJK + Hangul).
        # Real Japanese does mix kanji and kana, so treat the two as one system only when
        # the kana share is high enough — one stray
        # kana must not launder mojibake. Give the dominant script a small purity bonus and
        # penalise the share held by the rest.
        if scripts["kana"] >= 2 and scripts["kana"] * 5 >= scripts["cjk"]:
            scripts["kana"] += scripts["cjk"]
            scripts["cjk"] = 0
        script_total = sum(scripts.values())
        if script_total:
            purity = max(scripts.values()) / script_total
            score += 0.5 * purity - 1.5 * (1.0 - purity)
            # When a double-byte stream (Big5 and similar) is misread by a single-byte
            # codec, stray ASCII letters get wedged between non-Latin characters;
            # a correct decode usually keeps whole CJK characters. Penalise lightly by the
            # interleaving ratio, so genuinely
            # ASCII/Latin-dominant Western candidates are not affected.
            if scripts["latin"] == 0 and ascii_alpha:
                score -= ascii_alpha / (script_total + ascii_alpha)
        return score

    scored: list = []
    try:
        _ensure("charset_normalizer", "charset-normalizer")
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None:
            t = str(best)
            # ``charset_normalizer`` often returns coherence=0 guesses for very short text.
            # Chinese and Western single-byte short strings are especially prone to
            # same-script ties
            # (GB18030 "测试" → Big5 "聆彸"; cp1252 → cp1250). Let it into the tie-break only
            # when coherence is genuinely non-zero, or when Korean/Japanese is detected via
            # their distinctive glyphs; otherwise defer to the verified ordering below.
            coherence = float(getattr(best, "coherence", 0.0) or 0.0)
            language = str(getattr(best, "language", "") or "").lower()
            if coherence > 0.1 or language in {"japanese", "korean"}:
                scored.append((_score(t, bonus=0.1), t))
    except Exception:
        pass
    # Verified range: multi-byte CJK + Cyrillic + Western. Inside one single-byte family
    # (koi8 vs cp1251) no language model can separate them, so tie-break by
    # market share. Unverified Hebrew/Arabic/Greek/Thai candidates are deliberately absent
    # rather than letting tuple order masquerade as language identification.
    for enc in ("cp932", "gb18030", "big5", "cp949", "euc_jp",
                "cp1251", "koi8_r", "cp1252", "iso8859_2"):
        try:
            t = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
        scored.append((_score(t), t))
    if scored:
        return max(scored, key=lambda x: x[0])[1]
    return raw.decode("utf-8", errors="replace")


def _text_to_md(path: str) -> str:
    """Text file → full content with line numbers (cat -n style, added by the parser);
    encoding via _decode_bytes
    (BOM / utf-8 / charset detection, CJK-friendly)."""
    with open(path, "rb") as f:
        raw = f.read()
    text = _decode_bytes(raw)
    if _ext(path) in _NO_LINENO_EXTS:
        # json/yaml/md: verbatim (parseable / renderable), no line numbers, no header note
        return text if text.strip() else "(empty file)"
    head = "<!-- text readout: leading 'N\\t' line numbers are parser-added, not file content. -->"
    lines = text.splitlines()
    if not lines:
        return head + "\n\n(empty file)"
    w = len(str(len(lines)))
    body = "\n".join(f"{i:>{w}}\t{ln}" for i, ln in enumerate(lines, 1))
    return head + "\n" + body


def _classify_read(path: str) -> str:
    """One path for every unstructured file: text → numbered full content; image → VLM
    read (type only when unconfigured);
    audio/video → type only; anything else binary → unsupported (raises).

    A read failure (cannot open, or an unknown binary that cannot be read as text) always
    raises, and main() turns that into one
    fail-closed path (stderr + non-zero exit); reporting a media type is valid output and
    returns normally.
    """
    name = _os.path.basename(path)
    with open(path, "rb") as f:  # OSError propagates up → main() catches it
        head = f.read(65536)
    if _is_text(head):
        return _text_to_md(path)
    mime = _magic_mime(head)
    if mime and mime.split("/")[0] == "image":
        # A standalone image goes straight to the VLM (scans, charts, screenshots, photos all get read); vision unconfigured or failing → fall back to reporting the type.
        try:
            with open(path, "rb") as f:
                data = f.read()
            vt = _vision_read(data, mime)
        except Exception:
            vt = None
        if vt:
            return f"<!-- image {name} | read via vision -->\n\n{vt}"
        return (f"file: {name}\ntype: {mime}\n"
                "note: image — vision unavailable (set READDOC_VISION_URL/MODEL/KEY); not read.")
    if mime:
        kind = mime.split("/")[0]  # audio / video
        return (f"file: {name}\ntype: {mime}\n"
                f"note: binary {kind} — type detected only, not read as text.")
    # No text and no known media type: no document content can be read — raise instead of
    # returning a notice, or save_to would
    # store the notice as the document and report "saved N bytes".
    raise ValueError(f"unsupported binary {name} (application/octet-stream) "
                     "— cannot be read as text.")


# ---------------- Pagination (hard-boundary cutting + continuation notes) ----------------
# The markdown each format renders already carries hard boundary lines, so pagination is
# post-processing and the renderers stay untouched.
# One boundary marker for all: an HTML comment <!-- page/slide/sheet/... --> (not a
# heading, so it can never collide with content heading levels;
# visible in raw text, easy to match). docx injects no marker and uses the document's
# own headings as boundaries.
_BOUNDARY_RE = {
    "xlsx": r"(?m)^<!-- (sheet:|range |charts)",
    "pptx": r"(?m)^<!-- (slide |needs VLM)",
    "pdf": r"(?m)^<!-- page ",
    "docx": r"(?m)^#{1,6} ",
    "image_batch": r"(?m)^===== image ",   # batch image read: window on whole images, never cut a transcription in half
}


def _split_blocks(md: str, fmt: str):
    """md → list of hard-boundary block start offsets (the first block starts at 0; legend / lead-in belong to it)."""
    pat = _BOUNDARY_RE.get(fmt)
    starts = [m.start() for m in _re.finditer(pat, md)] if pat else []
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    return starts


def _weak_cut(md: str, start: int, hard_end: int) -> int:
    """Degraded cut point when one block exceeds the budget: line boundaries (an xlsx grid row is one line, never cut inside a cell row)."""
    nl = md.rfind("\n", start + 1, hard_end)
    return nl + 1 if nl > start else hard_end


def _resume_ctx(md: str, fmt: str, block_start: int, resume_at: int) -> str:
    """Rebuild context when resuming inside a block: the block's first line (sheet / slide / page title); for xlsx also the nearest column-letter header row."""
    head_line = md[block_start:md.find("\n", block_start) + 1].rstrip()
    # When the boundary marker is an HTML comment, put (continued) inside it so the
    # comment stays valid
    if head_line.endswith("-->"):
        cont = head_line[:-3].rstrip() + " (continued) -->"
    else:
        cont = f"{head_line} (continued)"
    lines = [cont]
    if fmt == "xlsx":
        seg = md[block_start:resume_at]
        for ln in reversed(seg.splitlines()):
            if ln.startswith("|   |"):  # grid column-letter header
                lines.append(ln)
                lines.append("| --- |" + " --- |" * (ln.count("|") - 2))
                break
    return "\n".join(lines) + "\n"


def _paginate(md: str, fmt: str, offset: int, max_chars: int) -> str:
    """Return one page of md as [offset, …]: whole blocks until the budget is spent; always
    cut on a hard boundary
    (sole exception: a single block larger than the budget → line boundary, noted in the
    message).
    Continuation notes are added at both ends; when one page holds the whole document and
    offset=0, the output is byte-identical to the source (zero additions)."""
    total = len(md)
    if max_chars <= 0:  # unset = send everything (offset still applies)
        return md[offset:] if offset else md
    if offset >= total:
        return (f"[read_file] offset {offset} >= total {total} chars; nothing left. "
                f"The document was fully covered by earlier reads.")
    starts = _split_blocks(md, fmt)
    # Snap the start: an offset landing mid-block (the model passed an arbitrary value, or
    # the last cut was line-level) → keep the exact position and re-add context
    import bisect
    bi = bisect.bisect_right(starts, offset) - 1
    start = max(offset, 0)
    mid_block_start = start > starts[bi]

    budget_end = start + max_chars
    end = start
    j = bi
    forced_weak = False
    if mid_block_start:  # finish the remainder of the current block first
        be = starts[j + 1] if j + 1 < len(starts) else total
        if be <= budget_end:
            end = be
            j += 1
        else:
            end = _weak_cut(md, start, budget_end)
            forced_weak = True
    if not forced_weak:
        while j < len(starts):
            be = starts[j + 1] if j + 1 < len(starts) else total
            if be - start > max_chars:
                break
            end = be
            j += 1
        if end == start:  # the very first whole block already exceeds the budget → degrade to line boundaries
            end = _weak_cut(md, start, budget_end)
            forced_weak = True

    body = md[start:end]
    done = end >= total
    if done and offset == 0:
        return body  # the whole document fits one page: identical to the old behaviour, zero additions

    parts = []
    if offset > 0:
        note = f"[read_file] continued read: chars {start}-{end} of {total}."
        parts.append(note + "\n")
        if mid_block_start:
            parts.append(_resume_ctx(md, fmt, starts[bi], start))
        parts.append("\n")
    parts.append(body)
    if not done:
        cut_desc = ("inside a section (single section exceeds max_chars; cut at a line "
                    "boundary)" if forced_weak else "at a section boundary")
        # Map of what is left: the unread blocks' title lines + their sizes
        remain = []
        k = bisect.bisect_right(starts, end) - 1
        if starts[k] < end:
            k += 1
        for idx in range(k, min(k + 10, len(starts))):
            s = starts[idx]
            e = starts[idx + 1] if idx + 1 < len(starts) else total
            title = md[s:md.find("\n", s) + 1].strip() or "(untitled)"
            remain.append(f"  - {title} (~{e - max(s, end)} chars)")
        more = len(starts) - k - len(remain)
        if more > 0:
            remain.append(f"  - … and {more} more sections")
        parts.append(
            f"\n\n[read_file] PARTIAL READ: returned chars {start}-{end} of {total}, "
            f"cut {cut_desc}. NOT finished — call read_file again with offset={end} "
            f"to continue."
            + ("\nRemaining:\n" + "\n".join(remain) if remain else "")
        )
    return "".join(parts)


# ------------- Render cache (persists under /workspace; any read/write failure just re-renders) -------------

def _render_cached(path: str, render) -> str:
    """Cache the full render keyed by (path, size, mtime), so every continuation page comes
    from the same md and parsing / recalc happens once.
    Any cache read/write failure silently falls back to rendering directly."""
    cf = None
    try:
        st = _os.stat(path)
        key = hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        cf = _os.path.join(_CACHE_DIR, key + ".md")
        if _os.path.exists(cf):
            return open(cf, encoding="utf-8").read()
    except OSError:
        cf = None
    md = render()
    if cf:
        try:
            _os.makedirs(_CACHE_DIR, exist_ok=True)
            tmp = cf + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(md)
            _os.replace(tmp, cf)
        except OSError:
            pass
    return md


# Look parser functions up by name via globals() (the per-format fragments are
# concatenated after this one, so they exist at runtime).
# Legacy binary formats (doc/ppt/xls) are absent from this table: openpyxl / pandoc /
# python-pptx do not recognise OLE2, so
# main() first converts them to OOXML through _legacy_to_ooxml (the soffice bridge) and
# then enters the matching reader.
_DISPATCH_NAMES = {
    "xlsx": "_xlsx_to_md", "xlsm": "_xlsx_to_md",
    "csv": "_csv_to_md", "tsv": "_csv_to_md",
    "docx": "_docx_to_md",
    "pptx": "_pptx_to_md", "pdf": "_pdf_to_md",
}
_FMT_NORM = {"xlsm": "xlsx"}
_LEGACY_TO_OOXML = {"doc": "docx", "ppt": "pptx", "xls": "xlsx"}


def _legacy_to_ooxml(path, target_ext):
    """Legacy Office binary (doc/ppt/xls) → an OOXML temp copy (LibreOffice); None on
    failure.
    The copy is used only for this parse; the markdown result is still cached under the
    original path via _render_cached."""
    import shutil as _sh
    import tempfile as _tf
    if not _sh.which("soffice"):
        return None
    td = _tf.mkdtemp(prefix="legacy_read_")
    env = dict(_os.environ)
    home = env.get("HOME", "")
    if not (home and _os.path.isdir(home) and _os.access(home, _os.W_OK)):
        env["HOME"] = "/tmp"   # /root may not exist inside the bwrap sandbox, and soffice needs a writable HOME
    try:
        r = subprocess.run(["soffice", "--headless", "--norestore", "--convert-to",
                            target_ext, "--outdir", td, _os.path.abspath(path)],
                           capture_output=True, timeout=90, env=env)  # < the outer 120s, so this
        # function's own failure message reaches the model instead of being swallowed by a generic outer timeout
        out = _os.path.join(td, _os.path.splitext(_os.path.basename(path))[0]
                            + "." + target_ext)
        if r.returncode == 0 and _os.path.exists(out):
            return out
    except Exception:
        pass
    # Conversion failed: clean up the temp dir we created (on the success path the caller cleans up after parsing)
    _sh.rmtree(td, ignore_errors=True)
    return None


_IMG_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"}
_IMG_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "gif": "image/gif", "bmp": "image/bmp", "tif": "image/tiff",
             "tiff": "image/tiff", "webp": "image/webp"}


def _resolve_image_batch(path):
    """path is a glob (contains * ? [) or a comma-separated list → (existing image files,
    skipped paths).
    Skipped = a named path that does not exist or is not an image (a glob expansion only
    contains existing entries, so skipped comes mostly from comma lists)."""
    import glob as _g
    cands = []
    if any(c in path for c in "*?["):
        cands = sorted(_g.glob(path))
    elif "," in path:
        cands = [q.strip() for q in path.split(",") if q.strip()]
    imgs = [q for q in cands if _os.path.isfile(q) and _ext(q) in _IMG_EXTS]
    skipped = [q for q in cands if q not in imgs]
    return imgs, skipped


# A batch read runs at most two waves: 4 concurrent x 2 waves x 45s per image = 90s worst
# case, leaving 30s of the outer 120s for
# process start, file reads, caching and result assembly. READDOC_BATCH_MAX lowers it
# further,
# but even with lower concurrency it never exceeds two waves.
_BATCH_MAX = max(1, int(_os.environ.get("READDOC_BATCH_MAX", "8") or "8"))
_BATCH_TIMEOUT = 45
_BATCH_MAX_WAVES = 2


def _batch_concurrency():
    return max(1, int(_os.environ.get("READDOC_VISION_CONCURRENCY", "4") or "4"))


def _batch_limit():
    return max(1, min(_BATCH_MAX, _batch_concurrency() * _BATCH_MAX_WAVES))


def _batch_image_read(paths, skipped=(), limit=None):
    """Batch images: one vision call **per image** (keeps fidelity — never packs several
    images into a single message) + parallel + reassembled in order.
    Concurrency READDOC_VISION_CONCURRENCY (default 4); per-image vision timeout 45s (so
    the whole batch converges inside the outer
    sandbox timeout). One read_file over N rendered pages collapses N agent turns into 1."""
    from concurrent.futures import ThreadPoolExecutor
    conc = _batch_concurrency()
    limit = max(1, _batch_limit() if limit is None else int(limit))
    truncated = paths[limit:]
    paths = paths[:limit]

    def _one(p):
        try:
            with open(p, "rb") as f:
                data = f.read()
            return _vision_read(
                data, _IMG_MIME.get(_ext(p), "image/png"), timeout=_BATCH_TIMEOUT
            )
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(conc, len(paths))) as ex:
        vts = list(ex.map(_one, paths))
    head = f"(batch image read — {len(paths)} images, one vision call per image, in parallel)"
    if skipped:
        head += ("\n<!-- skipped (not found / not an image): "
                 + ", ".join(_os.path.basename(s) for s in skipped) + " -->")
    if truncated:
        head += (f"\n<!-- NOTE: {len(truncated)} more images matched but only the first "
                 f"{limit} were read this call. Read the rest with another read_file "
                 f"call, e.g. a comma-separated list starting at "
                 f"{_os.path.basename(truncated[0])}. -->")
    out = [head]
    for i, (p, vt) in enumerate(zip(paths, vts, strict=False), 1):
        nm = _os.path.basename(p)
        out.append(f"\n===== image {i}/{len(paths)}: {nm} =====\n"
                   + (vt if vt else "(vision unavailable/failed for this image)"))
    return "\n".join(out), all(vt is not None for vt in vts)


def _batch_cache_key(pattern, imgs, limit):
    """Batch-read cache key: the full match set + file state + vision endpoint/model + this call's image cap."""
    parts = [
        pattern,
        f"vision_url={_os.environ.get('READDOC_VISION_URL', '').rstrip('/')}",
        f"vision_model={_os.environ.get('READDOC_VISION_MODEL', '')}",
        f"limit={limit}",
    ]
    for p in imgs:
        try:
            st = _os.stat(p)
            parts.append(f"{p}|{st.st_size}|{st.st_mtime_ns}")
        except OSError:
            parts.append(p)
    return _os.path.join(_CACHE_DIR,
                         hashlib.md5("\n".join(parts).encode()).hexdigest() + ".md")


def main() -> None:
    # argv: path [max_chars] [cell_range|-] [offset] [pdf_mode] [pages|-]
    #   cell_range — xlsx point lookup;  pdf_mode/pages — pdf mode and range;  offset — generic pagination
    if len(sys.argv) < 2:
        sys.stdout.write("[read_file error] usage: <path> [max_chars] [cell_range|-] "
                         "[offset] [pdf_mode] [pages|-]")
        return
    path = sys.argv[1]
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # unset = send everything
    cell_range = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    pdf_mode = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else "auto"
    pages = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    # Multi-image batch: path is a glob (build/pg-*.png) or a comma-separated list →
    # vision per image in parallel, reassembled. Results: (1) enter the render cache
    # (offset continuation re-calls no VLM); (2) pass through _paginate (windowed on image
    # boundaries, governed by
    # max_chars/offset — everything is returned at once when it fits, cut only when it
    # does not, never silently truncated).
    _imgs, _skipped = _resolve_image_batch(path)
    if any(c in path for c in "*?[") and not _imgs:
        non_image_count = len(_skipped)
        sys.stdout.write(
            "[read_file error] image glob matched no images "
            f"({non_image_count} non-image match"
            f"{'es' if non_image_count != 1 else ''}): {path}"
        )
        _dump_trace()
        return
    # 2+ images → batch read; exactly 1 but with skipped paths also goes batch (the output must report skipped, not swallow it)
    if len(_imgs) >= 2 or (len(_imgs) == 1 and _skipped):
        _trace({"stage": "batch_image", "n": len(_imgs), "skipped": len(_skipped)})
        limit = _batch_limit()
        cf = _batch_cache_key(path, _imgs, limit)
        md = None
        try:
            if _os.path.exists(cf):
                md = open(cf, encoding="utf-8").read()
        except OSError:
            pass
        if md is None:
            md, cacheable = _batch_image_read(_imgs, _skipped, limit)
            if cacheable:
                try:
                    _os.makedirs(_CACHE_DIR, exist_ok=True)
                    _tmp = cf + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as f:
                        f.write(md)
                    _os.replace(_tmp, cf)
                except OSError:
                    pass
        # max_chars is honoured exactly as passed (0 = everything). The larger default
        # window for batch reads is chosen by the tool layer
        # (read_file.py) when the caller did not pass one explicitly; the reader never
        # overrides an explicit value.
        sys.stdout.write(_paginate(md, "image_batch", offset, max_chars))
        _dump_trace()
        return
    if len(_imgs) == 1:
        # A glob / comma list matching exactly one image with nothing skipped: fall back to a plain single-file read (old behaviour)
        path = _imgs[0]
    ext = _ext(path)
    trace_on = bool(_os.environ.get("READDOC_TRACE"))
    # Legacy Office binary: the soffice bridge converts to an OOXML copy and the original
    # reader runs on it; the readout header notes the conversion source.
    # Cache first: the full render is cached under the **original path** (the dispatch
    # section writes the same key) — a hit returns immediately, so
    # continuing through a large .doc no longer re-runs soffice (seconds each) per offset.
    legacy_note = ""
    read_path = path
    _legacy_tmp = None
    if ext in _LEGACY_TO_OOXML:
        tgt = _LEGACY_TO_OOXML[ext]
        try:
            st = _os.stat(path)
            _cf0 = _os.path.join(_CACHE_DIR, hashlib.md5(
                f"{path}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest() + ".md")
            if _os.path.exists(_cf0):
                _trace({"stage": "legacy_convert", "ext": ext, "cache": "hit"})
                sys.stdout.write(_paginate(open(_cf0, encoding="utf-8").read(),
                                           _FMT_NORM.get(tgt, tgt), offset, max_chars))
                _dump_trace()
                return
        except OSError:
            pass
        _trace({"stage": "legacy_convert", "ext": ext, "to": tgt})
        conv = _legacy_to_ooxml(path, tgt)
        if conv is None:
            sys.stdout.write(f"[read_file error] legacy .{ext} requires LibreOffice "
                             "conversion, which is unavailable/failed in this sandbox.")
            _dump_trace()
            return
        legacy_note = f"`converted from .{ext} via LibreOffice; fidelity best-effort`\n\n"
        read_path, ext = conv, tgt
        _legacy_tmp = _os.path.dirname(conv)   # converted-copy dir, cleaned after parsing (keeps /tmp from growing)
    if ext not in _DISPATCH_NAMES:
        # Unstructured document: sniff and route — text → numbered content (paginated); image/audio/video → type only; anything else binary → unsupported
        _trace({"stage": "dispatch", "ext": ext, "fmt": "text", "reader": "_classify_read"})
        try:
            md = _render_cached(path, lambda: _classify_read(path))
        except Exception as e:
            # Fail closed like the structured route below: any read failure
            # (bad open / unsupported binary) goes to stderr + exit non-zero,
            # else under read_file's save_to (`... > <file>`) the error text is
            # captured AS the saved document and reported as a success. Media
            # type notices are valid output — _classify_read returns those.
            sys.stderr.write(f"[read_file error] {type(e).__name__}: {e}")
            _dump_trace()
            sys.exit(1)
        sys.stdout.write(_paginate(md, "text", offset, max_chars))
        _dump_trace()
        return
    fmt = _FMT_NORM.get(ext, ext)
    fn = globals()[_DISPATCH_NAMES[ext]]
    _trace({"stage": "dispatch", "ext": ext, "fmt": fmt, "reader": _DISPATCH_NAMES[ext]})
    # Parameterised reads (output varies with the parameters) skip the full-text cache;
    # only a plain full read is cached.
    # Tracing also bypasses the cache: a cache hit skips the routing code, leaving the
    # trace empty.
    pdf_param = ext == "pdf" and (pdf_mode != "auto" or pages)
    try:
        # Parse read_path (the converted copy for legacy formats); the cache key stays the original path
        if cell_range and ext in ("xlsx", "xlsm"):
            md = fn(read_path, cell_range)
        elif pdf_param:
            md = fn(read_path, pdf_mode, pages)
        elif ext == "pdf":
            md = fn(read_path, "auto", None) if trace_on else \
                _render_cached(path, lambda: fn(read_path, "auto", None))
        else:
            if trace_on:
                md = fn(read_path)
            elif legacy_note:
                # Cache the exact string that pagination indexes.  The first
                # page and continuation reads share this cache key, so omitting
                # the conversion note here shifts every later offset by its
                # length and silently skips document content.
                md = _render_cached(path, lambda: legacy_note + fn(read_path))
                legacy_note = ""
            else:
                md = _render_cached(path, lambda: fn(read_path))
    except Exception as e:
        # Parse error must NOT go to stdout: under read_file's ``save_to`` the
        # command is ``... > <file>``, so an error written to stdout would be
        # captured AS the saved document and read_file would report "saved N
        # bytes" (a silent success on a failed parse). Emit to stderr and exit
        # non-zero so read_file's ``exit_code != 0`` branch surfaces the error.
        sys.stderr.write(f"[read_file error parsing .{ext}] {type(e).__name__}: {e}")
        _dump_trace()
        sys.exit(1)
    finally:
        if _legacy_tmp:   # the legacy converted copy is removed as soon as it is used, so the sandbox /tmp does not grow per read
            import shutil as _sh
            _sh.rmtree(_legacy_tmp, ignore_errors=True)
    sys.stdout.write(_paginate(legacy_note + md, fmt, offset, max_chars))
    _dump_trace()
