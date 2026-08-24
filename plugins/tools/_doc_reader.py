#!/usr/bin/env python3
"""Assembler for the read_file parsers + a local standalone entry point."""
from pathlib import Path

_PARTS = (
    "_reader_core.py",  # shared pieces (_ensure/_ext) + the dispatching main() (definition only)
    "_reader_xlsx.py",  # _xlsx_to_md: coordinate grid / R1C1 formulas / recalc / chart / pivot / Table
    "_reader_docx.py",  # _docx_to_md: pandoc → md, falling back to python-docx
    "_reader_pptx.py",  # _pptx_to_md: rich extraction / grouping / flow / Needs VLM
    "_reader_pdf.py",   # _pdf_to_md: per-page text via pypdf
)

_TAIL = '\n\nif __name__ == "__main__":\n    main()\n'


def reader_src() -> str:
    """Assemble the single-file parser script shared by the sandbox and local runs (fragment order is the table above; the entry call is appended at the end)."""
    d = Path(__file__).parent
    return "\n\n\n".join((d / p).read_text(encoding="utf-8") for p in _PARTS) + _TAIL


if __name__ == "__main__":
    exec(compile(reader_src(), "<doc_reader_bundle>", "exec"),
         {"__name__": "__main__", "__file__": __file__})
