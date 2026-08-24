#!/usr/bin/env python3
"""office_writer local standalone entry point: concatenates _writer_{core,docx,xlsx,pptx}.py into one script and runs it.
(Mirrors read_file's _doc_reader.py; inside the sandbox the create_file tool uses the same bundle.)

Usage: python3 _create_file.py <path> <op> '<args_json>'
"""
from pathlib import Path

_PARTS = ("_writer_core.py", "_writer_docx.py", "_writer_xlsx.py", "_writer_pptx.py",
          "_writer_text.py")
_TAIL = '\n\nif __name__ == "__main__":\n    main()\n'


def writer_src() -> str:
    d = Path(__file__).parent
    avail = [p for p in _PARTS if (d / p).exists()]  # skip format fragments that do not exist yet
    return "\n\n\n".join((d / p).read_text(encoding="utf-8") for p in avail) + _TAIL


if __name__ == "__main__":
    exec(compile(writer_src(), "<create_file_bundle>", "exec"),
         {"__name__": "__main__", "__file__": __file__})
