from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from plugins.tools._create_file import writer_src
from plugins.tools._doc_reader import reader_src


def _run_bundle(source: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", *args],
        input=source,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )


def _write(path: Path, operations: list[dict[str, object]]) -> str:
    result = _run_bundle(writer_src(), str(path), json.dumps(operations))
    assert path.exists(), result.stdout + result.stderr
    assert "error" not in result.stdout.lower(), result.stdout
    return result.stdout


def _read(path: Path) -> str:
    result = _run_bundle(reader_src(), str(path), "20000")
    assert "traceback" not in result.stderr.lower(), result.stderr
    assert "error" not in result.stdout.lower(), result.stdout
    return result.stdout


def test_exec_bundles_write_and_read_office_formats(tmp_path: Path) -> None:
    docx = tmp_path / "sample.docx"
    xlsx = tmp_path / "sample.xlsx"
    pptx = tmp_path / "sample.pptx"

    _write(
        docx,
        [{"create": {"blocks": [{"type": "paragraph", "text": "DOCX_SENTINEL"}]}}],
    )
    _write(
        xlsx,
        [{"create": {"sheets": [{"name": "Data", "headers": ["Name"], "rows": [["XLSX_SENTINEL"]]}]}}],
    )
    _write(
        pptx,
        [{"create": {"slides": [{"layout": "title", "title": "PPTX_SENTINEL"}]}}],
    )

    assert "DOCX_SENTINEL" in _read(docx)
    assert "XLSX_SENTINEL" in _read(xlsx)
    assert "PPTX_SENTINEL" in _read(pptx)


def test_exec_bundle_reads_pdf_without_external_cli(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "PDF_SENTINEL"})
    with pdf.open("wb") as handle:
        writer.write(handle)

    output = _read(pdf)
    assert "PDF" in output or "page" in output.lower()


# ``MAX_ARG_STRLEN`` — the kernel's cap on ONE execve argument (32 pages).
# The whole command reaches ``sh -c`` as a single argument, so the bundle has
# to travel over stdin like ``_run_bundle`` above does.
_MAX_ARG_STRLEN = 32 * 4096


async def test_create_file_pipes_the_writer_over_stdin(monkeypatch) -> None:
    """The writer bundle must not ride in argv.

    It is ~131KB base64-encoded, so embedding it in the command line left 46
    bytes for path + ops and every real call died with a bare "Argument list
    too long". The bundle tests above run ``writer_src()`` directly, which is
    why nothing caught it — this one asserts on the tool's own wiring.
    """
    # ``plugins.tools.create_file`` re-exports the tool under the module's own
    # name, so reach the module itself to patch its sandbox calls.
    create_file_mod = importlib.import_module("plugins.tools.create_file")

    captured: dict[str, object] = {}

    async def fake_sandbox() -> object:
        return object()

    async def fake_run(sandbox: object, command: str, **kwargs: object) -> object:
        captured["command"] = command
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(create_file_mod, "aget_sandbox", fake_sandbox)
    monkeypatch.setattr(create_file_mod, "arun_sandbox_cmd", fake_run)

    result = await create_file_mod.create_file.ainvoke({
        "path": "/outputs/report.md",
        "ops": [{"create": {"content": "x" * 5000}}],
    })

    assert result == "ok"
    assert captured["input"] == writer_src()
    assert len(str(captured["command"])) < _MAX_ARG_STRLEN


def test_create_file_accepts_the_resolved_mount_dirs(monkeypatch, tmp_path: Path) -> None:
    """Native mode remaps the mounts to real host dirs — and hands the model
    those exact paths — so the write roots follow ``resolve_mount_dirs``
    instead of only the ``/workspace`` and ``/outputs`` literals."""
    from plugins.tools.create_file import _write_roots

    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(tmp_path / "outputs"))

    roots = _write_roots()

    assert str(tmp_path) in roots
    assert str(tmp_path / "outputs") in roots
    # The container convention the tool's docstring teaches stays valid.
    assert "/workspace" in roots
    assert "/outputs" in roots


async def test_create_file_maps_native_aliases_to_resolved_mounts(
    monkeypatch, tmp_path: Path,
) -> None:
    """The generic aliases remain valid in native mode even on read-only macOS /."""
    create_file_mod = importlib.import_module("plugins.tools.create_file")
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    workspace.mkdir()
    outputs.mkdir()
    captured: dict[str, object] = {}

    async def fake_sandbox() -> object:
        return object()

    async def fake_run(sandbox: object, command: str, **kwargs: object) -> object:
        captured["command"] = command
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setattr(create_file_mod, "aget_sandbox", fake_sandbox)
    monkeypatch.setattr(create_file_mod, "arun_sandbox_cmd", fake_run)

    result = await create_file_mod.create_file.ainvoke({
        "path": "/outputs/report.docx",
        "ops": [{"create": {"blocks": [
            {"type": "image", "path": "/workspace/chart.png"},
        ]}}],
    })

    assert result == "ok"
    command = str(captured["command"])
    assert command.startswith(str(sys.executable))
    assert str(outputs / "report.docx") in command
    assert str(workspace / "chart.png") in command


async def test_create_file_raises_when_writer_exits_nonzero(monkeypatch) -> None:
    create_file_mod = importlib.import_module("plugins.tools.create_file")

    async def fake_sandbox() -> object:
        return object()

    async def fake_run(sandbox: object, command: str, **kwargs: object) -> object:
        return SimpleNamespace(
            exit_code=1,
            stdout="✗ /outputs/report.docx — STOPPED at op 1/1",
            stderr="",
        )

    monkeypatch.setattr(create_file_mod, "aget_sandbox", fake_sandbox)
    monkeypatch.setattr(create_file_mod, "arun_sandbox_cmd", fake_run)

    with pytest.raises(RuntimeError, match="writer exited 1"):
        await create_file_mod.create_file.ainvoke({
            "path": "/outputs/report.docx",
            "ops": [{"create": {"blocks": [
                {"type": "paragraph", "text": "x"},
            ]}}],
        })


async def test_writer_stderr_noise_never_hides_the_failure_receipt(monkeypatch) -> None:
    """The receipt says WHICH op failed, and it comes back on stdout.

    Preferring ``stderr`` whenever it is non-empty handed the model an
    unrelated ``UserWarning`` instead — a 50-op batch failing with no way to
    tell which op it was, which is the whole point of failing loudly.
    """
    create_file_mod = importlib.import_module("plugins.tools.create_file")

    async def fake_sandbox() -> object:
        return object()

    async def fake_run(sandbox: object, command: str, **kwargs: object) -> object:
        return SimpleNamespace(
            exit_code=1,
            stdout="✗ /outputs/report.docx — STOPPED at op 7/50: unknown block type",
            stderr="UserWarning: Workbook contains no default style",
        )

    monkeypatch.setattr(create_file_mod, "aget_sandbox", fake_sandbox)
    monkeypatch.setattr(create_file_mod, "arun_sandbox_cmd", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        await create_file_mod.create_file.ainvoke({
            "path": "/outputs/report.docx",
            "ops": [{"create": {"blocks": [
                {"type": "paragraph", "text": "x"},
            ]}}],
        })

    message = str(excinfo.value)
    assert "STOPPED at op 7/50" in message      # the receipt survives...
    assert "UserWarning" in message             # ...alongside the noise


async def test_create_file_resolves_paths_inside_an_ops_program(
    monkeypatch, tmp_path: Path,
) -> None:
    """``@program.json`` must resolve like the identical inline array.

    The aliases inside the program are the ones the tool's own docstring
    teaches; leaving them literal made the staged form fail in native mode on
    exactly the calls the inline form handles.
    """
    create_file_mod = importlib.import_module("plugins.tools.create_file")
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    workspace.mkdir()
    outputs.mkdir()
    program = workspace / "program.json"
    program.write_text(json.dumps([
        {"create": {"blocks": [{"type": "image", "path": "/workspace/chart.png"}]}},
    ]), encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_sandbox() -> object:
        return object()

    async def fake_run(sandbox: object, command: str, **kwargs: object) -> object:
        captured["command"] = command
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setattr(create_file_mod, "aget_sandbox", fake_sandbox)
    monkeypatch.setattr(create_file_mod, "arun_sandbox_cmd", fake_run)

    result = await create_file_mod.create_file.ainvoke({
        "path": "/outputs/report.docx",
        "ops": "@/workspace/program.json",
    })

    assert result == "ok"
    command = str(captured["command"])
    assert str(workspace / "chart.png") in command
    # The alias must be gone from the ops themselves — matching on the whole
    # command would hit the tmp_path, which itself ends in ``/workspace``.
    assert '"path": "/workspace/chart.png"' not in command
    assert '{"create"' in command       # inlined, not passed as @<file>


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        ("not json at all", "bad ops_json"),
        (json.dumps({"a": 1, "b": 2}), "single-key object"),
        (json.dumps([{"create": "a string"}]), "must be an object"),
    ],
)
def test_writer_exits_nonzero_on_every_refusal(
    tmp_path: Path, ops: str, expected: str,
) -> None:
    """create_file reads success from the exit code, so no refusal may exit 0."""
    result = subprocess.run(
        [sys.executable, "-", str(tmp_path / "out.md"), ops],
        input=writer_src(), text=True, capture_output=True,
        check=False, timeout=60,
    )

    assert result.returncode != 0, result.stdout
    assert expected in result.stdout


def test_writer_exits_nonzero_when_an_operation_fails(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(path),
            json.dumps([{"unknown_operation": {}}]),
        ],
        input=writer_src(),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "STOPPED" in result.stdout


@pytest.mark.parametrize("suffix", ["-escape/x.md", "/../x.md", "/sub/../../x.md"])
def test_write_roots_are_not_matched_by_string_prefix(
    monkeypatch, tmp_path: Path, suffix: str,
) -> None:
    """A prefix test also accepts ``<root>-escape`` and lets ``..`` through, and
    the roots are real host directories — so that is a host write escape."""
    from plugins.tools.create_file import _outside_write_roots

    workspace = tmp_path / "ws"
    outputs = tmp_path / "collected"
    workspace.mkdir()
    outputs.mkdir()
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))

    assert _outside_write_roots(f"{outputs}{suffix}")
    assert _outside_write_roots(f"{workspace}{suffix}")
    # The roots themselves and paths under them stay writable.
    assert not _outside_write_roots(str(outputs / "report.md"))
    assert not _outside_write_roots(str(workspace / "sub" / "scratch.md"))
    assert not _outside_write_roots("/outputs/report.md")


async def test_oversized_inline_ops_are_refused_with_the_staged_form(monkeypatch) -> None:
    """The writer moved to stdin but ops are still argv, so a large enough batch
    would hit MAX_ARG_STRLEN. Refuse it with the fix instead of E2BIG."""
    create_file_mod = importlib.import_module("plugins.tools.create_file")

    async def unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("an oversized payload reached the sandbox")

    monkeypatch.setattr(create_file_mod, "aget_sandbox", unreachable)

    result = await create_file_mod.create_file.ainvoke({
        "path": "/outputs/report.md",
        "ops": [{"create": {"content": "x" * 200_000}}],
    })

    assert "over the" in result
    assert "@/workspace/program.json" in result


@pytest.mark.parametrize("shape", ["symlinked-parent", "symlinked-final"])
def test_writer_refuses_symlinks_that_leave_the_write_roots(
    tmp_path: Path, shape: str,
) -> None:
    """A lexical check in the caller cannot see these.

    ``normpath`` normalises text; it does not resolve a link. In native mode the
    writer shares the host filesystem, so a link planted inside an allowed root
    — as a parent component or as the final name — redirected the write to
    wherever it pointed. The containment re-check runs in the writer, where
    ``realpath`` follows exactly what ``open()`` will.
    """
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if shape == "symlinked-parent":
        (root / "link").symlink_to(outside)
        target = root / "link" / "escaped.md"
        landed = outside / "escaped.md"
    else:
        (root / "final.md").symlink_to(outside / "victim.md")
        target = root / "final.md"
        landed = outside / "victim.md"

    result = subprocess.run(
        [
            sys.executable, "-", str(target),
            json.dumps([{"create": {"content": "ESCAPED"}}]),
            json.dumps([str(root)]),
        ],
        input=writer_src(), text=True, capture_output=True,
        check=False, timeout=60,
    )

    # Non-zero, not just a message: create_file reads success from the exit
    # code, so a refusal that exited 0 would be reported as a completed write.
    assert result.returncode != 0
    assert "refusing to write" in result.stdout
    assert not landed.exists()


def test_writer_still_writes_inside_the_roots(tmp_path: Path) -> None:
    """The containment check must not cost a legitimate write."""
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "ok.md"

    _run_bundle(
        writer_src(), str(target),
        json.dumps([{"create": {"content": "fine"}}]),
        json.dumps([str(root)]),
    )

    assert target.read_text().strip() == "fine"


async def test_command_bound_is_measured_after_shell_quoting(monkeypatch) -> None:
    """``shlex.quote`` turns each apostrophe into four bytes, so a payload that
    passes a raw-size check can still assemble past MAX_ARG_STRLEN."""
    create_file_mod = importlib.import_module("plugins.tools.create_file")

    async def unreachable(*args: object, **kwargs: object) -> object:
        raise AssertionError("an oversized command reached the sandbox")

    monkeypatch.setattr(create_file_mod, "aget_sandbox", unreachable)

    # ~30KB raw — under any sane payload bound — but ~150KB once quoted.
    result = await create_file_mod.create_file.ainvoke({
        "path": "/outputs/report.md",
        "ops": [{"create": {"content": "'" * 30_000}}],
    })

    assert "assembles a" in result
    assert "@/workspace/program.json" in result
