"""Spill recovery and scope isolation (handoff item 3; Harness #494 / #521).

The rules pinned here:

* one scope key decides the store directory, the bwrap mount and read
  authorization (``current_store_scope``);
* a ref is advertised only when the backend that runs commands can open it,
  and the compaction callback is withheld otherwise;
* a truncated body stays recoverable through compaction, twice over;
* nothing crosses scopes, and cleanup touches only this process's stores.

Persistence is AgentCore's ``SpillStore``; there is no second store.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from frontier_agent.core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from frontier_agent.core.runtime.loop.compact import KeepLastNToolResultsCompactor
from plugins.tools import _overflow, _sandbox
from plugins.tools.meta import get_tool_meta


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("APODEX_SPILL_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    saved = set(_overflow._created_stores)
    _overflow._created_stores.clear()
    try:
        yield
    finally:
        _overflow._created_stores.clear()
        _overflow._created_stores.update(saved)
        _overflow._child_scopes.clear()
        _overflow._preview_refs.clear()


class _scoped:
    def __init__(self, task: str, session: str = "") -> None:
        meta = {"llm_session_id": session} if session else {}
        self.scope = ExecutionScope(task_id=task, metadata=meta)

    def __enter__(self) -> ExecutionScope:
        self._token = set_current_execution_scope(self.scope)
        return self.scope

    def __exit__(self, *exc: object) -> None:
        reset_current_execution_scope(self._token)


def _body(n: int = 40_000) -> str:
    return "".join(f"line {i}: evidence {i * 7}\n" for i in range(n // 20))


def _mounted(monkeypatch) -> None:
    monkeypatch.setattr(_overflow, "_visible_root", lambda: "/spill")


# ── one identity ─────────────────────────────────────────────────────────


def test_store_mount_and_read_auth_share_one_scope_key(monkeypatch) -> None:
    from plugins.tools._path_auth import _resolve_spill_dirs

    _mounted(monkeypatch)
    with _scoped("T", "sess") as scope:
        ref = _overflow.spill_compacted_body("bash", _body())
        assert ref
        key = _overflow.current_store_scope()
        assert key == _overflow.spill_scope_key("T", "sess") == "T:sess"
        component = _overflow.scope_component(key)
        store_dir = (_sandbox.spill_root().resolve() / component)
        assert ref.startswith(f"/spill/{component}/")
        assert _resolve_spill_dirs() == [store_dir]
        mounts = _sandbox._spill_mount_args()
        assert mounts == [
            "--dir", "/spill", "--ro-bind-try", str(store_dir), f"/spill/{component}",
        ]
    assert scope.metadata["llm_session_id"] == "sess"


def test_unscoped_store_is_per_process() -> None:
    key = _overflow.current_store_scope()
    assert key == f"{_overflow.UNSCOPED_PROCESS_STORE}:{os.getpid()}"


def test_spill_files_are_read_only_and_symlinks_refused(monkeypatch, tmp_path) -> None:
    _mounted(monkeypatch)
    with _scoped("T"):
        path, _ref = _overflow._write_spill("bash", _body(), require_visible=True)
    assert oct(path.stat().st_mode & 0o777) == "0o444"

    real = tmp_path / "elsewhere"
    real.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(real)
    monkeypatch.setenv("APODEX_SPILL_DIR", str(link))
    monkeypatch.setattr(_sandbox, "_private_spill_root", None)
    first, second = _sandbox.spill_root(), _sandbox.spill_root()
    assert first == second, "writer, mount and reader must agree on one fallback"
    assert not first.is_symlink() and first != link


# ── refs only where the backend can open them ────────────────────────────


@pytest.mark.parametrize(("backend", "bwrap", "expected"), [
    ("e2b", True, None),
    ("bwrap", False, None),
    ("bwrap", True, "/spill"),
    ("local", True, "/spill"),
    ("native", False, "physical"),
    ("container", False, "physical"),
])
def test_visible_root_follows_the_backend(monkeypatch, backend, bwrap, expected) -> None:
    monkeypatch.setenv("SANDBOX_BACKEND", backend)
    monkeypatch.setenv("E2B_API_KEY", "k")
    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: bwrap)
    monkeypatch.setattr(_sandbox, "get_existing_sandbox", lambda: None)
    monkeypatch.setattr(_sandbox, "_container_inner_bwrap_enabled", lambda: False)
    want = str(_sandbox.spill_root().resolve()) if expected == "physical" else expected
    assert _overflow._visible_root() == want


def test_auto_with_an_e2b_key_names_no_host_path(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_BACKEND", "auto")
    monkeypatch.setattr(_sandbox, "get_existing_sandbox", lambda: None)
    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: True)
    monkeypatch.setattr(_sandbox, "_resolve_use_e2b", lambda: (True, "k", "t", 60))
    assert _overflow._visible_root() is None


def test_a_live_sandbox_wins_over_configuration(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_BACKEND", "e2b")

    class FakeBwrap(_sandbox.BwrapSandbox):
        def __init__(self) -> None:  # no real jail needed
            pass

    monkeypatch.setattr(_sandbox, "get_existing_sandbox", lambda: FakeBwrap())
    assert _overflow._visible_root() == "/spill"

    e2b_like = type("Sandbox", (), {"__module__": "e2b_code_interpreter.main"})()
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setattr(_sandbox, "bwrap_available", lambda: True)
    monkeypatch.setattr(_sandbox, "get_existing_sandbox", lambda: e2b_like)
    assert _overflow._visible_root() is None


def test_unrecoverable_backend_withholds_callback_and_writes_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(_overflow, "_visible_root", lambda: None)
    assert _overflow.default_compaction_spill() is None
    cap = get_tool_meta("bash").max_result_chars
    with _scoped("T"):
        out = _overflow.maybe_overflow("bash", _body(cap * 4))
    assert "not readable from this backend" in out
    assert "saved read-only at" not in out
    store = tmp_path / "store"
    assert not store.exists() or not any(store.rglob("*.md"))


def test_recoverable_backend_gets_the_callback(monkeypatch) -> None:
    _mounted(monkeypatch)
    assert _overflow.default_compaction_spill() is _overflow.spill_compacted_body


# ── recoverable through truncation and repeated compaction ───────────────


def _history(content: str) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "name": "bash", "args": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": content},
        {"role": "user", "content": "continue"},
    ]


def test_full_text_survives_truncation_and_two_compactions(monkeypatch) -> None:
    _mounted(monkeypatch)
    full = _body(60_000)
    with _scoped("T", "s"):
        preview = _overflow.maybe_overflow("bash", full)
        assert len(preview) < len(full)
        compactor = KeepLastNToolResultsCompactor(
            keep_tool_result=0, spill=_overflow.default_compaction_spill(),
        )
        once = compactor.compact(_history(preview), keep_recent=1)
        twice = compactor.compact(once, keep_recent=1)
        ref = once[3]["spill_refs"][0]
        assert twice[3]["spill_refs"] == [ref], "re-compaction must keep the same ref"
        assert f"[Full text] {ref}" in once[3]["content"]
        # The card points at the ORIGINAL body, not at a copy of the preview.
        assert _overflow.get_overflow_content(ref) == full
        files = list(_sandbox.spill_root().rglob("*.md"))
        assert len(files) == 1, "the preview must not be stored a second time"


def test_a_preview_re_cut_by_the_aggregate_budget_still_maps_to_the_full_body(monkeypatch) -> None:
    _mounted(monkeypatch)
    full = _body(60_000)
    with _scoped("T"):
        previews = [_overflow.maybe_overflow("bash", full + str(i)) for i in range(40)]
        adjusted = _overflow.check_aggregate_budget(previews, ["bash"] * 40)
        recut = next(a for a, p in zip(adjusted, previews, strict=True) if a != p)
        ref = _overflow.spill_compacted_body("bash", recut)
        assert ref and _overflow.get_overflow_content(ref).startswith(full)


def test_an_echoed_fragment_of_a_preview_claims_no_ref(monkeypatch) -> None:
    _mounted(monkeypatch)
    with _scoped("T"):
        preview = _overflow.maybe_overflow("bash", _body(60_000))
        fragment = preview[: len(preview) // 2] + "x" * 2_000
        assert _overflow._full_body_ref(fragment) == ""


# ── cross-scope access ───────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["bwrap", "native"])
def test_a_siblings_ref_is_not_readable(monkeypatch, backend) -> None:
    if backend == "native":
        monkeypatch.setenv("SANDBOX_BACKEND", "native")
    else:
        _mounted(monkeypatch)
    parent, a, b = (ExecutionScope(task_id="T", metadata={"llm_session_id": s})
                    for s in ("main", "a", "b"))
    _overflow.register_child_scope(parent, a)
    _overflow.register_child_scope(parent, b)
    with _scoped("T", "a"):
        ref_a = _overflow.spill_compacted_body("bash", _body())
    with _scoped("T", "b"):
        assert _overflow.get_overflow_content(ref_a) is None
        assert _overflow.readable_store_dirs() == []
    with _scoped("T", "main"):
        assert _overflow.get_overflow_content(ref_a) is not None


def test_entering_a_nested_loop_registers_it_as_a_child() -> None:
    from frontier_agent.core.loop_types import LoopConfig
    from frontier_agent.core.runtime.loop.agent_loop import _enter_scope

    with _scoped("T", "main"):
        child, token = _enter_scope(
            LoopConfig(task_id="T", llm_session_id="sub"), "p", {"llm_session_id": "sub"},
        )
        reset_current_execution_scope(token)
    assert "T:sub" in _overflow.readable_scopes("T:main")
    assert _overflow.readable_scopes("T:sub") == ["T:sub"]
    assert child.metadata["llm_session_id"] == "sub"


# ── cleanup ──────────────────────────────────────────────────────────────


def test_tree_cleanup_removes_only_this_conversation(monkeypatch, tmp_path) -> None:
    _mounted(monkeypatch)
    parent = ExecutionScope(task_id="T", metadata={"llm_session_id": "main"})
    child = ExecutionScope(task_id="T", metadata={"llm_session_id": "sub"})
    _overflow.register_child_scope(parent, child)
    with _scoped("T", "main"):
        _overflow.spill_compacted_body("bash", _body())
    with _scoped("T", "sub"):
        _overflow.spill_compacted_body("bash", _body())
    with _scoped("U", "main"):
        keep = _overflow.spill_compacted_body("bash", _body())
    # A store another process created, sitting under the same root.
    foreign = _sandbox.spill_root() / _overflow.scope_component("T:other-process")
    foreign.mkdir(parents=True)
    (foreign / "x.md").write_text("theirs")

    assert _overflow.cleanup_overflow_tree("T:main") == 2
    with _scoped("U", "main"):
        assert _overflow.get_overflow_content(keep) is not None
    assert (foreign / "x.md").exists()


# ── real jail (needs a host where bwrap can create namespaces) ───────────


@pytest.mark.skipif(
    not _sandbox.bwrap_available(),
    reason="bwrap cannot create namespaces here; isolation is NOT verified by a skip",
)
def test_real_jail_shows_only_the_current_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    sandbox = _sandbox.BwrapSandbox(workspace=tmp_path / "ws")
    with _scoped("T", "a"):
        ref_a = _overflow.spill_compacted_body("bash", _body())
    with _scoped("T", "b"):
        ref_b = _overflow.spill_compacted_body("bash", _body() + "b")
        listing = sandbox.commands.run("ls /spill").stdout.split()
        assert listing == [_overflow.scope_component("T:b")]
        assert sandbox.commands.run(f"cat {ref_a}").exit_code != 0
        assert sandbox.commands.run(f"cat {ref_b}").exit_code == 0
        assert sandbox.commands.run(f"touch {ref_b}").exit_code != 0
    sandbox.kill()


def test_bwrap_is_really_unavailable_here_when_skipped() -> None:
    """Records WHY the real-jail test skips on this host, so a skip is never
    read as a pass."""
    if _sandbox.bwrap_available() or shutil.which("bwrap") is None:
        pytest.skip("not applicable")
    probe = subprocess.run(
        ["bwrap", "--ro-bind", "/", "/", "true"], capture_output=True, text=True,
    )
    assert probe.returncode != 0
