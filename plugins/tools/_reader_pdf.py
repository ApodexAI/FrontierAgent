
# pdf reading: per-page routing.
#   pdf_mode: auto  — per-page gate (default): rule (1) no text (2) maths/LaTeX
#                     (3) garbled (4) large image block → OCR the whole page;
#                     everything else uses the text layer (two-column reflow).
#                     Document-level bypasses: AcroForm / attachments / OCG / Tagged.
#             text  — force the text layer, pdftotext -layout (faithful, never OCR)
#             image — pdftoppm converts the given pages to PNG and returns the paths (handed to view_image)
#   pages:    "1-5,12,40-" (1-based; empty = all)
#
#   Detection is all poppler (pdffonts / pdftotext-bbox / pdfimages / pdfinfo /
#   pdfdetach), no new packages;
#   pypdf is only an enhancement (form values / OCG names / Tagged tree / vector counts) and degrades automatically when absent.
#   OCR port: a synchronous API (POST {base}/ocr, body = PDF bytes + an Authorization
#   header → JSON:
#     text / text_with_img_link / layout_json); health check GET {base}/health/ready;
#     base url comes from env READDOC_OCR_URL. Unconfigured, the page is only marked "routed to OCR (reason) + text-layer fallback".
import json as _pjson
import os as _pos
import re as _pre
import subprocess as _psub
import time as _ptime

# ---- Thresholds (relative to scale / intrinsic properties, not fitted to a dataset) ----
_PDF_EMPTY_CHARS = 10        # rule 1: fewer extractable characters than this on a page → treat as empty (scan / pure image)
_PDF_MATH_FONTS = _pre.compile(
    r"(CMMI|CMSY|CMEX|CMMIB|CMBSY|MSAM|MSBM|RSFS|EU[FSM]|StandardSym|"
    r"Math|rsfs|cmmi|cmsy|cmex)", _pre.I)  # rule 2: maths-only font families
_PDF_GARBLE_RATIO = 0.15     # rule 3: share of unmappable glyphs (replacement char / PUA) above this → the text layer is untrustworthy
_PDF_IMG_COVER = 1.0 / 6     # rule 4: total image share of the page area above this → large image block (scan / screenshot / figure)
_PDF_VEC_OPS = 400           # rule 4 (vector, best-effort): more path operators than this plus little text → a figure
_PDF_SPARSE_TEXT = 200       # rule 4: less text than this plus substantial visual content → chart / scanned page (the text layer plainly is not carrying the content)

# auto overview / image deep read are separate, and both run concurrently
_PDF_CONCURRENCY = int(_pos.environ.get("READDOC_PDF_CONCURRENCY", "2"))  # per-page concurrency (shared by deep read and overview)
_PDF_INLINE_IMG_COVER = 0.08  # text page: raster coverage above this → an inline "figure not read" conclusion
_PDF_DRAW_OPS_FLOOR = 100     # text page: draw ops above this → report the count neutrally (draw no conclusion; let the model judge from the body text)
_OCR_FIG_MIN_WPCT = 15        # minimum share of page width for an OCR <img width="N%"> to count as a "real figure" (below this it is probably a logo)


def _pdf_parse_pages(pages, total):
    if not pages:
        return list(range(1, total + 1))
    out = set()
    for part in str(pages).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            lo = int(a) if a.strip() else 1
            hi = int(b) if b.strip() else total
        else:
            lo = hi = int(part)
        for p in range(max(lo, 1), min(hi, total) + 1):
            out.add(p)
    return sorted(out)


def _pdf_page_count(path):
    try:
        r = _psub.run(["pdfinfo", path], capture_output=True, text=True, timeout=30)
        for ln in r.stdout.splitlines():
            if ln.startswith("Pages:"):
                return int(ln.split(":")[1])
    except Exception:
        pass
    _ensure("pypdf", "pypdf")
    from pypdf import PdfReader
    return len(PdfReader(path).pages)


def _pdf_page_text(path, page_no):
    """One page of text: pdftotext -layout (preserves layout), falling back to pypdf on failure."""
    try:
        r = _psub.run(["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no), path, "-"],
                      capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return r.stdout
    except Exception:
        pass
    try:
        _ensure("pypdf", "pypdf")
        from pypdf import PdfReader
        return PdfReader(path).pages[page_no - 1].extract_text() or ""
    except Exception:
        return ""


# ---------- Document-level metadata ----------
def _pinfo(path):
    d = {}
    try:
        r = _psub.run(["pdfinfo", path], capture_output=True, text=True, timeout=30)
        for ln in r.stdout.splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                d[k.strip()] = v.strip()
    except Exception:
        pass
    return d


def _page_size(pinfo):
    m = _pre.search(r"([\d.]+)\s*x\s*([\d.]+)\s*pts", pinfo.get("Page size", ""))
    return (float(m.group(1)), float(m.group(2))) if m else (612.0, 792.0)


# ---------- Per-page detection (poppler) ----------
def _fonts_on_page(path, page_no):
    """[(name, has_tounicode)]; the uni column of pdffonts."""
    out = []
    try:
        r = _psub.run(["pdffonts", "-f", str(page_no), "-l", str(page_no), path],
                      capture_output=True, text=True, timeout=30)
        for ln in r.stdout.splitlines()[2:]:
            # The tail is always emb sub uni objid objgen — three yes/no plus two numbers (type contains spaces, so column splitting will not work)
            m = _pre.search(r"\b(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", ln)
            name = ln.split()[0] if ln.split() else ""
            if name:
                out.append((name, bool(m) and m.group(3) == "yes"))
    except Exception:
        pass
    return out


def _is_math_page(fonts):
    return any(_PDF_MATH_FONTS.search(n) for n, _ in fonts)


def _garble_ratio(text):
    if not text:
        return 0.0
    bad = sum(1 for c in text if c == "�" or 0xE000 <= ord(c) <= 0xF8FF)
    return bad / max(len(text), 1)


def _image_cover(path, page_no, pagew, pageh):
    """**Total** share of the page area taken by every image on it (placed area, capped at 1.0; smask excluded).
    From pdfimages -list width/height (px) + x/y-ppi → pt."""
    page_area = max(pagew * pageh, 1.0)
    total = 0.0
    try:
        r = _psub.run(["pdfimages", "-list", "-f", str(page_no), "-l", str(page_no), path],
                      capture_output=True, text=True, timeout=30)
        for ln in r.stdout.splitlines()[2:]:
            c = ln.split()
            if len(c) < 15 or c[2] == "smask":  # an smask is the companion mask, so its area is not counted twice
                continue
            try:
                w, h = float(c[3]), float(c[4])
                xppi, yppi = float(c[12]), float(c[13])
                if xppi <= 0 or yppi <= 0:
                    continue
                total += (w / xppi * 72.0) * (h / yppi * 72.0) / page_area
            except (ValueError, ZeroDivisionError):
                continue
    except Exception:
        pass
    return min(total, 1.0)


def _vector_ops(path, page_no):
    """Count of path-construction operators (including one level of Form XObject; best-effort, pypdf; missing or failing → -1)."""
    try:
        _ensure("pypdf", "pypdf")
        from pypdf import PdfReader
        from pypdf.generic import ContentStream
        rd = PdfReader(path)
        pg = rd.pages[page_no - 1]

        def _count(cs):
            return sum(1 for _, op in cs.operations if op in (b"l", b"c", b"re", b"m", b"v", b"y"))

        n = _count(ContentStream(pg.get_contents(), rd))
        xo = (pg.get("/Resources") or {}).get("/XObject")  # Office vector charts are usually wrapped in a Form XObject
        if xo:
            for ref in xo.values():
                try:
                    o = ref.get_object()
                    if o.get("/Subtype") == "/Form":
                        n += _count(ContentStream(o.get_data(), rd))
                except Exception:
                    continue
        return n
    except Exception:
        return -1


def _word_boxes(path, page_no):
    """[(xmin,ymin,xmax,ymax,text)] via pdftotext -bbox。"""
    out = []
    try:
        r = _psub.run(["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), path, "-"],
                      capture_output=True, text=True, timeout=60)
        for m in _pre.finditer(
                r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
                r.stdout):
            x0, y0, x1, y1, t = m.groups()
            out.append((float(x0), float(y0), float(x1), float(y1),
                        t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")))
    except Exception:
        pass
    return out


def _detect_columns(boxes, pagew):
    """Two-column gutter detection: returns split_x or None. The test = both sides hold a sizeable share and very few words straddle the gutter."""
    if len(boxes) < 30:
        return None
    best = None
    for frac in (0.45, 0.5, 0.55):
        split = pagew * frac
        left = sum(1 for b in boxes if b[2] < split)
        right = sum(1 for b in boxes if b[0] > split)
        cross = sum(1 for b in boxes if b[0] <= split <= b[2])
        n = len(boxes)
        if left > 0.25 * n and right > 0.25 * n and cross < 0.05 * n:
            score = min(left, right) - cross
            if best is None or score > best[1]:
                best = (split, score)
    return best[0] if best else None


def _reorder_columns(boxes, split):
    """Reflow by column: the whole left column (row by row) → the whole right column. A word straddling the gutter goes to the nearer side."""
    def col_text(words):
        words = sorted(words, key=lambda b: (round(b[1] / 6), b[0]))  # by row (~6pt granularity), then by column
        lines, cur, cy = [], [], None
        for b in words:
            if cy is None or abs(b[1] - cy) <= 6:
                cur.append(b[4])
                cy = b[1] if cy is None else cy
            else:
                lines.append(" ".join(cur))
                cur = [b[4]]
                cy = b[1]
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)
    left = [b for b in boxes if (b[0] + b[2]) / 2 < split]
    right = [b for b in boxes if (b[0] + b[2]) / 2 >= split]
    return col_text(left) + "\n\n" + col_text(right)


# ---------- OCR port (synchronous POST /ocr; env READDOC_OCR_URL + READDOC_OCR_KEY; returns None when unconfigured) ----------
def _ocr_base():
    return _pos.environ.get("READDOC_OCR_URL", "").rstrip("/")


def _one_page_pdf(path, page_no, outdir):
    out = _pos.path.join(outdir, f"ocr_p{page_no}.pdf")
    if not _pos.path.exists(out):
        _psub.run(["pdfseparate", "-f", str(page_no), "-l", str(page_no), path, out],
                  capture_output=True, timeout=60)
    return out if _pos.path.exists(out) else None


def _ocr_page(path, page_no, outdir):
    """Whole page → the OCR port (synchronous: POST {base}/ocr, body = PDF bytes, returns JSON) → markdown.
    Raises on failure; returns None when the port is unconfigured (the caller falls back). Auth via env READDOC_OCR_KEY (Authorization header).
    Response JSON: text / text_with_img_link (carries <img> figure markers) / layout_json (block bboxes)."""
    base = _ocr_base()
    if not base:
        return None
    import urllib.error
    import urllib.request
    src = _one_page_pdf(path, page_no, outdir) or path
    data = open(src, "rb").read()
    headers = {"Content-Type": "application/pdf"}
    key = _pos.environ.get("READDOC_OCR_KEY", "")
    if key:
        headers["Authorization"] = key
    t0 = _ptime.time()
    # An inference pool returns intermittent 503 "pool not ready" (scale-down, cold start), so 503 is retried with backoff; every other error is raised.
    resp = None
    for attempt in range(6):
        req = urllib.request.Request(f"{base}/ocr", data=data, headers=headers, method="POST")
        try:
            resp = _pjson.loads(urllib.request.urlopen(req, timeout=300).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < 5:
                _ptime.sleep(3 + attempt * 3)
                continue
            raise
    md = resp.get("text_with_img_link") or resp.get("text") or ""
    _trace({"stage": "ocr", "page": page_no,
            "ms": int((_ptime.time() - t0) * 1000), "status": "done",
            "has_img": "<img" in md})
    # OCR marks charts and figures as <img> (it does not read the data inside them). When
    # this page has a "real figure" (not a small logo):
    # the reader renders the whole page → calls vision with a figures-only prompt (body
    # text and tables stay as OCR produced them; vision only adds the figures) →
    # marks the position at the <img> and appends the figure content, clearly labelled, at
    # the end of the page. A mixed page thus keeps OCR body text, still gets its figures
    # read, and duplicates nothing.
    has_real = "<img" in md and bool(_pos.environ.get("READDOC_VISION_URL")) \
        and _ocr_has_real_figure(md)
    if "<img" in md:
        _trace({"stage": "figure-detect", "page": page_no, "has_img": True,
                "real_figure": has_real, "vision_url": bool(_pos.environ.get("READDOC_VISION_URL"))})
    if has_real:
        imgs = _pdf_to_images(path, [page_no])
        if imgs:
            try:
                vt = _vision_read(open(imgs[0][1], "rb").read(), "image/png",
                                  question=_VISION_FIGURE_PROMPT)
            except Exception:
                vt = None
            if vt and "NO_FIGURE" not in vt:
                marked = _pre.sub(r'<img[^>]*>', "`[figure — read via vision ↓]`", md)
                return marked + "\n\n`[figures on this page, read via vision]`\n\n" + vt
    return md


def _ocr_has_real_figure(md):
    """OCR already marks each figure's position and relative page-width share with <img ... width="N%">.
    Use that to judge a "real figure": any <img> whose width% >= the threshold (or, absent a width%, conservatively assume a figure) → True;
    False only when every <img> is clearly small (probably a logo or icon). This uses the
    labels OCR gave us directly, with no dependency on layout_json."""
    # (The threshold, and whether the OCR service emits <img> for logos at all, depend on the deployed service.)
    for m in _pre.finditer(r'<img\b[^>]*>', md or ""):
        wm = _pre.search(r'width\s*=\s*["\']?\s*(\d+(?:\.\d+)?)\s*%', m.group(0))
        if wm is None or float(wm.group(1)) >= _OCR_FIG_MIN_WPCT:
            return True
    return False


# ---------- Document-level bypasses ----------
def _attachments(path):
    """Embedded attachment names (pdfdetach -list, poppler)."""
    try:
        r = _psub.run(["pdfdetach", "-list", path], capture_output=True, text=True, timeout=30)
        names = _pre.findall(r"(?m)^\s*\d+:\s*(.+)$", r.stdout)
        return [n.strip() for n in names]
    except Exception:
        return []


def _acroform_fields(path):
    """AcroForm field values (pypdf; empty when absent)."""
    try:
        _ensure("pypdf", "pypdf")
        from pypdf import PdfReader
        f = PdfReader(path).get_fields()
        if not f:
            return []
        out = []
        for name, fld in f.items():
            v = fld.get("/V")
            out.append((str(name), "" if v is None else str(v)))
        return out
    except Exception:
        return []


def _ocg_layers(path):
    """Optional-content layer names (pypdf catalog /OCProperties)."""
    try:
        _ensure("pypdf", "pypdf")
        from pypdf import PdfReader
        root = PdfReader(path).trailer["/Root"]
        ocp = root.get("/OCProperties")
        if not ocp:
            return []
        names = []
        for g in (ocp.get("/OCGs") or []):
            try:
                names.append(str(g.get_object().get("/Name")))
            except Exception:
                continue
        return names
    except Exception:
        return []


_TAG_ROLE = {"/H1": "# ", "/H2": "## ", "/H3": "### ", "/H4": "#### ",
             "/H5": "##### ", "/H6": "###### ", "/Title": "# ", "/H": "## "}


def _tagged_outline(path, limit=400):
    """Tagged structure tree → reading-order outline (pypdf; best-effort).
    Takes each structure element's role + its /ActualText | /Alt | /T text, recursing in /K order."""
    try:
        _ensure("pypdf", "pypdf")
        from pypdf import PdfReader
        from pypdf.generic import IndirectObject
        root = PdfReader(path).trailer["/Root"]
        st = root.get("/StructTreeRoot")
        if not st:
            return ""
        lines = []

        def txt(node):
            for key in ("/ActualText", "/Alt", "/T"):
                v = node.get(key)
                if v:
                    return str(v)
            return ""

        def walk(node, depth):
            if len(lines) >= limit or depth > 12:
                return
            try:
                if isinstance(node, IndirectObject):
                    node = node.get_object()
            except Exception:
                return
            if isinstance(node, list):
                for c in node:
                    walk(c, depth)
                return
            if not hasattr(node, "get"):
                return
            role = node.get("/S")
            t = txt(node)
            if role is not None and (str(role) in _TAG_ROLE or t.strip()):
                r = str(role)  # keep only headings and nodes carrying text; skip pure structural noise like Div / NonStruct
                lines.append(_TAG_ROLE.get(r, "  " * min(depth, 6) + f"- [{r.lstrip('/')}] ") + t)
            k = node.get("/K")
            if k is not None:
                walk(k, depth + 1)

        walk(st.get("/K"), 0)
        body = "\n".join(x for x in lines if x.strip())
        return body
    except Exception:
        return ""


# ---------- Post-processing: strip repeated headers/footers ----------
def _norm_line(s):
    return _pre.sub(r"\d+", "#", s.strip())


def _dedup_headers(page_texts):
    """Detect first/last lines repeated across pages = running header/footer; returns (header, footer, cleaned_pages)."""
    n = len(page_texts)
    if n < 3:
        return "", "", page_texts
    firsts, lasts = {}, {}
    for t in page_texts:
        ls = [x for x in t.splitlines() if x.strip()]
        if ls:
            firsts[_norm_line(ls[0])] = firsts.get(_norm_line(ls[0]), 0) + 1
            lasts[_norm_line(ls[-1])] = lasts.get(_norm_line(ls[-1]), 0) + 1
    hdr = max(firsts, key=firsts.get) if firsts else ""
    ftr = max(lasts, key=lasts.get) if lasts else ""
    hdr_hit = firsts.get(hdr, 0) >= max(3, int(0.5 * n))
    ftr_hit = lasts.get(ftr, 0) >= max(3, int(0.5 * n))
    header = footer = ""
    cleaned = []
    for t in page_texts:
        ls = t.splitlines()
        nonempty = [i for i, x in enumerate(ls) if x.strip()]
        if hdr_hit and nonempty and _norm_line(ls[nonempty[0]]) == hdr:
            header = ls[nonempty[0]].strip()
            ls[nonempty[0]] = ""
        if ftr_hit and nonempty and _norm_line(ls[nonempty[-1]]) == ftr:
            footer = ls[nonempty[-1]].strip()
            ls[nonempty[-1]] = ""
        cleaned.append("\n".join(ls))
    return header, footer, cleaned


# ---------- image mode (render PNG, hand off to view_image) ----------
def _pdf_to_images(path, page_nos, dpi=150):
    stem = _pos.path.splitext(_pos.path.basename(path))[0].replace(" ", "_")
    outdir = _pos.path.join("/workspace", ".readdoc_pdf_img", stem)
    try:
        _pos.makedirs(outdir, exist_ok=True)
    except OSError:
        outdir = _pos.path.join("/tmp", ".readdoc_pdf_img", stem)
        _pos.makedirs(outdir, exist_ok=True)
    res = []
    for p in page_nos:
        prefix = _pos.path.join(outdir, f"p{p}")
        png = prefix + ".png"
        if not _pos.path.exists(png):
            _psub.run(["pdftoppm", "-png", "-r", str(dpi), "-f", str(p), "-l", str(p),
                       "-singlefile", path, prefix], capture_output=True, timeout=120)
        if _pos.path.exists(png):
            res.append((p, png))
    return res


def _ocr_workdir(path):
    stem = _pos.path.splitext(_pos.path.basename(path))[0].replace(" ", "_")
    for base in ("/workspace/.readdoc_pdf_img", "/tmp/.readdoc_pdf_img"):
        try:
            d = _pos.path.join(base, stem)
            _pos.makedirs(d, exist_ok=True)
            return d
        except OSError:
            continue
    return "."


# ---------- Main entry point ----------
def _pdf_route_decide(path, p, pagew, pageh):
    """The cheap per-page verdict (poppler only; runs no OCR and no vision). Shared by the auto overview and the image deep read.
    Returns a dict: route='text'|'ocr', reason (None for a text page), rule, sig (the signals), text (the extracted text layer)."""
    text = _pdf_page_text(path, p)
    fonts = _fonts_on_page(path, p)
    tlen = len(text.strip())
    sig = {"text_len": tlen, "math_font": _is_math_page(fonts),
           "garble": round(_garble_ratio(text), 3)}
    # The gate, in order; reason goes straight into the readout
    reason = rule = None
    if tlen < _PDF_EMPTY_CHARS:
        reason, rule = "scanned image or pure graphic", "1-empty"
    elif sig["math_font"]:
        reason, rule = "math/formula fonts (LaTeX)", "2-math"
    elif sig["garble"] > _PDF_GARBLE_RATIO:
        reason, rule = "garbled text layer (broken font encoding)", "3-garble"
    else:
        cov = _image_cover(path, p, pagew, pageh)
        vops = _vector_ops(path, p)
        sig["img_cover"], sig["vec_ops"] = round(cov, 3), vops
        if cov > _PDF_IMG_COVER:
            reason, rule = f"dominated by a raster image (covers {cov:.0%} of page)", "4a-large-image"
        elif vops > _PDF_VEC_OPS and tlen < 400:
            reason, rule = f"vector graphic ({vops} draw ops, little text)", "4b-vector"
        elif tlen < _PDF_SPARSE_TEXT and (cov > 0.05 or vops > 100):
            reason = f"sparse text ({tlen} chars) with visual content (img {cov:.0%}, draw ops {vops})"
            rule = "4c-sparse-visual"
    d = {"route": "ocr" if reason else "text", "reason": reason,
         "rule": rule or "text-fast", "sig": sig, "text": text}
    _trace({"stage": "route", "page": p, "signals": sig, "rule": d["rule"],
            "route": d["route"], "reason": reason})
    return d


def _text_body(path, p, pagew, text):
    """Body text of a text page: reflowed when two-column, otherwise the raw text layer."""
    boxes = _word_boxes(path, p)
    split = _detect_columns(boxes, pagew)
    return _reorder_columns(boxes, split) if split else text


def _overview_page(path, p, pagew, pageh):
    """auto overview (cheap; runs no OCR and no vision). Returns (body, tag, is_text).
    text page: the text plus an inline raster conclusion (cover above the threshold) and a neutral draw-ops count (above the threshold); ocr pages are placeholders only."""
    d = _pdf_route_decide(path, p, pagew, pageh)
    if d["route"] == "ocr":
        return ("", f" | not read: {d['reason']} — read via read_file with "
                f"pages={p} and pdf_mode=image", False)
    body = _text_body(path, p, pagew, d["text"])
    sig = d["sig"]
    extras = []
    if sig.get("img_cover", 0) > _PDF_INLINE_IMG_COVER:
        extras.append(f"figure not read: embedded image (covers {sig['img_cover']:.0%} of page)"
                      f" — read via read_file with pages={p} and pdf_mode=image")
    if sig.get("vec_ops", 0) > _PDF_DRAW_OPS_FLOOR:
        extras.append(f"draw ops = {sig['vec_ops']}")
    return (body, (" | " + " | ".join(extras)) if extras else "", True)


def _deep_page(path, p, pagew, pageh, outdir):
    """image deep read: runs the real logic behind the route (ocr class → OCR + figure vision; a text page with figures → text + figure vision).
    Returns (body, tag, is_text)."""
    d = _pdf_route_decide(path, p, pagew, pageh)
    text = d["text"]
    if d["route"] == "ocr":
        try:
            md = _ocr_page(path, p, outdir)
        except Exception as e:
            return ((text or "(no extractable text)")
                    + f"\n\n`[OCR failed: {type(e).__name__}; text-layer fallback]`",
                    " | OCR failed", False)
        if md is not None:
            return (md, " | OCR", False)
        note = (f"`[routed to OCR — {d['reason']}; OCR endpoint not configured "
                f"(set READDOC_OCR_URL). Showing text-layer fallback below.]`")
        return (note + ("\n\n" + text if text.strip() else ""), " | →OCR", False)
    # Deep-reading a text page: when the user explicitly chose image, always render the
    # whole page and run figures-only vision —
    # no longer gated on the cover threshold (a vector figure has cover=0% and still needs
    # reading). With no figure present the prompt returns NO_FIGURE and only the text layer remains.
    # The body text stays the high-quality pdftotext output; vision only adds figures and never re-transcribes the text.
    body = _text_body(path, p, pagew, text)
    if _pos.environ.get("READDOC_VISION_URL"):
        imgs = _pdf_to_images(path, [p])
        if imgs:
            try:
                vt = _vision_read(open(imgs[0][1], "rb").read(), "image/png",
                                  question=_VISION_FIGURE_PROMPT)
            except Exception:
                vt = None
            if vt and "NO_FIGURE" not in vt:
                return (body + "\n\n`[figure on this page, read via vision]`\n\n" + vt,
                        " | vision", False)
    return (body, "", True)


def _map_pages(sel, fn):
    """Run fn(p) concurrently while preserving page order; concurrency READDOC_PDF_CONCURRENCY (default 2), applied only to pages that actually need work."""
    if _PDF_CONCURRENCY <= 1 or len(sel) <= 1:
        return [fn(p) for p in sel]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=_PDF_CONCURRENCY) as ex:
        return list(ex.map(fn, sel))


def _pdf_doc_head(path, pinfo):
    """Document-level bypass header (encryption / embedded attachments / AcroForm / OCG / Tagged structure). Returns a list of lines."""
    head = []
    if pinfo.get("Encrypted", "no").startswith("yes"):
        head.append("`encrypted: yes (extraction may be limited)`")
    atts = _attachments(path)
    if atts:
        head.append("▸ embedded files: " + ", ".join(atts))
    if pinfo.get("Form", "none") not in ("none", ""):
        fields = _acroform_fields(path)
        if fields:
            head.append("▸ form fields:\n" + "\n".join(f"  - {n}: {v}" for n, v in fields if n))
    ocg = _ocg_layers(path)
    if ocg:
        head.append("▸ optional layers (OCG, may be hidden): " + ", ".join(ocg) +
                    " — re-read with pdf_mode='image' to render a specific layer")
    if pinfo.get("Tagged", "no").startswith("yes"):
        outline = _tagged_outline(path)
        head.append("▸ tagged-PDF structure (reading order):\n" + outline if outline
                    else "`tagged-PDF: yes (structure tree present)`")
    return head


def _pdf_to_md(path, pdf_mode="auto", pages=None):
    total = _pdf_page_count(path)
    sel = _pdf_parse_pages(pages, total)
    span = "" if (not pages) else f" (pages {pages})"

    if pdf_mode == "text":  # force the text layer: faithful, never OCR
        out = [f"(PDF: {total} pages{span})"]
        for p in sel:
            out += [f"\n<!-- page {p} -->\n", _pdf_page_text(path, p)]
        return "\n".join(out)

    pinfo = _pinfo(path)
    pagew, pageh = _page_size(pinfo)
    outdir = _ocr_workdir(path)
    head = _pdf_doc_head(path, pinfo)

    # auto = overview (verdict only; ocr pages are placeholders) / image = deep read (runs the real routing logic). Concurrent per page.
    if pdf_mode == "image":
        results = _map_pages(sel, lambda p: _deep_page(path, p, pagew, pageh, outdir))
        mode_note = f"deep read (pdf_mode=image) — {len(sel)} page(s) executed"
    else:
        results = _map_pages(sel, lambda p: _overview_page(path, p, pagew, pageh))
        mode_note = ("overview (pdf_mode=auto) — figure/scan pages are flagged, not read; "
                     "re-read a flagged page with pages=N and pdf_mode=image")

    bodies = [r[0] for r in results]
    tags = [r[1] for r in results]
    is_text = [r[2] for r in results]

    # Header/footer dedup: only between text pages (placeholder / OCR / vision pages stay out, to avoid false positives)
    text_idx = [i for i, t in enumerate(is_text) if t]
    header, footer, sub_clean = _dedup_headers([bodies[i] for i in text_idx])
    for j, i in enumerate(text_idx):
        bodies[i] = sub_clean[j]

    out = [f"(PDF: {total} pages{span}) — {mode_note}"]
    if head:
        out.append("\n```meta\n" + "\n".join(head) + "\n```")
    if header:
        out.append(f"\n`running header (all pages)`: {header}")
    if footer:
        out.append(f"`running footer (all pages)`: {footer}")
    for p, body, tag in zip(sel, bodies, tags, strict=False):
        out += [f"\n<!-- page {p}{tag} -->\n", body]
    return "\n".join(out)
