from __future__ import annotations

from plugins.tools import ToolRegistry, get_builtin_tools


EXPECTED_TOOLS = {
    "add_task",
    "assign_task",
    "bash",
    "collect_reports",
    "create_file",
    "create_subagent",
    "download_file",
    "file_editor_create",
    "file_editor_str_replace",
    "file_editor_view",
    "finish_planning",
    "glob_search",
    "grep_search",
    "read_file",
    "recover_result",
    "run_python_code",
    "stop_subagent",
    "submit_report",
    "update_task",
    "view_image",
    "web_fetch",
    "web_search",
    "write_file",
}


def test_builtin_registry_is_an_explicit_allowlist() -> None:
    tools = get_builtin_tools()
    assert set(tools) == EXPECTED_TOOLS
    assert "finalize_answer" not in tools
    assert "tool_search" not in tools
    assert "dag_query" not in tools


def test_registry_round_trip() -> None:
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())

    assert registry.names() == sorted(EXPECTED_TOOLS)
    assert len(registry) == len(EXPECTED_TOOLS)
    assert registry.get("web_search") is get_builtin_tools()["web_search"]
