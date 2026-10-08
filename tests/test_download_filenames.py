from __future__ import annotations

from email.message import Message

import pytest

from plugins.tools._download_runner import _header_filename, _resolve_basename


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        ('attachment; filename="report.pdf"', "report.pdf"),
        ('attachment; filename="report;final.pdf"', "report;final.pdf"),
        ('attachment; filename="O\'Reilly.pdf"', "O'Reilly.pdf"),
        ('attachment; filename = "report.pdf"', "report.pdf"),
        ('attachment; filename="report%20draft.pdf"', "report%20draft.pdf"),
        ("attachment; filename*=UTF-8''report%20final.pdf", "report final.pdf"),
        ("attachment; filename*=UTF-8'en'report%20final.pdf", "report final.pdf"),
        ("attachment; filename*=UTF-8''caf%C3%A9.pdf", "café.pdf"),
        ("attachment; filename*=ISO-8859-1''caf%E9.pdf", "café.pdf"),
        (
            'attachment; filename="fallback.txt"; filename*=UTF-8\'\'actual.pdf',
            "actual.pdf",
        ),
        (
            'attachment; filename*=UTF-8\'\'actual.pdf; filename="fallback.txt"',
            "actual.pdf",
        ),
        ("attachment", ""),
    ],
)
def test_response_filename(disposition: str, expected: str) -> None:
    headers = Message()
    headers["Content-Disposition"] = disposition

    assert _header_filename(headers) == expected


@pytest.mark.parametrize(
    ("path_arg", "disposition", "expected"),
    [
        ("", 'attachment; filename="report;final.pdf"', "report_final.pdf"),
        ("", 'attachment; filename="O\'Reilly.pdf"', "O_Reilly.pdf"),
        (
            "",
            'attachment; filename="fallback.txt"; filename*=UTF-8\'\'actual.pdf',
            "actual.pdf",
        ),
        ("", 'attachment; filename="../../outside.pdf"', "outside.pdf"),
        ("", "attachment; filename*=UTF-8''..%2Foutside.pdf", "outside.pdf"),
        ("", "attachment", "url name.pdf"),
        ("", "", "url name.pdf"),
        ("chosen.pdf", 'attachment; filename="report;final.pdf"', "chosen.pdf"),
    ],
)
def test_download_basename(path_arg: str, disposition: str, expected: str) -> None:
    headers = Message()
    if disposition:
        headers["Content-Disposition"] = disposition

    assert _resolve_basename(path_arg, "https://example.com/url%20name.pdf", headers) == expected
