"""Explicit built-in tool registry for the OSS workflows."""

from __future__ import annotations

import logging

from frontier_agent.core.tool import Tool
from plugins.tools.assign_task import assign_task
from plugins.tools.bash import bash
from plugins.tools.collect_reports import collect_reports
from plugins.tools.create_file import create_file
from plugins.tools.create_subagent import create_subagent
from plugins.tools.download_file import download_file
from plugins.tools.file_editor import (
    file_editor_create,
    file_editor_str_replace,
    file_editor_view,
)
from plugins.tools.glob_search import glob_search
from plugins.tools.grep_search import grep_search
from plugins.tools.read_file import read_file
from plugins.tools.recover_result import recover_result
from plugins.tools.run_python_code import run_python_code
from plugins.tools.stop_subagent import stop_subagent
from plugins.tools.submit_report import submit_report
from plugins.tools.task_board import add_task, finish_planning, update_task
from plugins.tools.view_image import view_image
from plugins.tools.web_fetch import web_fetch
from plugins.tools.web_search import web_search
from plugins.tools.write_file import write_file

logger = logging.getLogger(__name__)

_BUILTIN_TOOLS: list[Tool] = [
    web_search,
    web_fetch,
    download_file,
    bash,
    create_subagent,
    assign_task,
    add_task,
    update_task,
    finish_planning,
    collect_reports,
    stop_subagent,
    read_file,
    create_file,
    write_file,
    file_editor_view,
    file_editor_create,
    file_editor_str_replace,
    submit_report,
    view_image,
    grep_search,
    glob_search,
    run_python_code,
    recover_result,
]


def get_builtin_tools() -> dict[str, Tool]:
    """Return the allowlisted built-ins as a name-to-tool mapping."""
    return {tool.name: tool for tool in _BUILTIN_TOOLS}


class ToolRegistry:
    """Central registration and fail-closed role filtering."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: dict[str, Tool]) -> None:
        self._tools.update(tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all(self) -> dict[str, Tool]:
        return dict(self._tools)

    def get_for_role(self, role_id: str) -> list[Tool]:
        try:
            from frontier_agent.core.runtime.registries import services
            from frontier_agent.core.runtime.registries.agents import AgentRegistry

            allowed = set(services.get(AgentRegistry).get_tools_for(role_id))
            return [tool for name, tool in self._tools.items() if name in allowed]
        except Exception as exc:
            logger.warning("Tool lookup for role %s failed: %s", role_id, exc)
            return []

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


__all__ = [
    "ToolRegistry",
    "get_builtin_tools",
]
