"""FrontierAgentAgent — Harbor BaseAgent adapter.

Bridges Harbor's agent protocol to FrontierAgent's research pipeline,
enabling benchmark evaluation via the Harbor framework.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "harbor is required for benchmark runs but isn't installed. "
        "Run: uv sync --extra eval"
    ) from exc

logger = logging.getLogger(__name__)

# ── TOML metadata parser (regex, no dependency) ──────────────────────

_METADATA_SECTION_RE = re.compile(
    r"^\[metadata\]\s*$",
    re.MULTILINE,
)
_KV_RE = re.compile(
    r'^(\w+)\s*=\s*"([^"]*)"',
)


def parse_task_metadata(toml_text: str) -> dict[str, str]:
    """Extract key-value pairs from a [metadata] TOML section.

    Uses simple regex parsing — no external TOML library needed.
    Returns an empty dict if [metadata] is missing or text is malformed.
    """
    try:
        m = _METADATA_SECTION_RE.search(toml_text)
        if not m:
            return {}

        result: dict[str, str] = {}
        for line in toml_text[m.end():].splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Stop at next section header
            if line.startswith("["):
                break
            kv = _KV_RE.match(line)
            if kv:
                result[kv.group(1)] = kv.group(2)
        return result
    except Exception:
        return {}


# ── FrontierAgentAgent ─────────────────────────────────────────────────


class FrontierAgentAgent(BaseAgent):
    """Harbor-compatible agent that delegates to FrontierAgent pipelines."""

    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Populated on first ``setup()``; reused across trials. Harbor
        # never calls a teardown hook, so the session lives until process
        # exit. Tests should drive ``BenchmarkSession`` via ``async with``
        # directly so the registry snapshot is restored cleanly.
        self._session: Any = None

    @staticmethod
    def name() -> str:
        return "FrontierAgent"

    def version(self) -> str | None:
        return "0.2.0"

    async def setup(
        self,
        environment: BaseEnvironment,
        *,
        db_path: str | None = None,
    ) -> None:
        """Bootstrap the FrontierAgent runtime (reused across trials)."""
        if self._session is not None:
            return
        from benchmarks.public.core.kernel_adapter import BenchmarkSession

        session = BenchmarkSession()
        await session.__aenter__()
        self._session = session
        logger.info("FrontierAgent session bootstrapped")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute a single benchmark question through the pipeline."""
        # Capture logs_dir immediately — concurrent tasks share the same agent
        # instance and may mutate self.logs_dir between coroutine yields.
        logs_dir = self.logs_dir
        t0 = time.monotonic()
        agent_dir = logs_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Read task metadata ────────────────────────────────
        meta = self._read_task_metadata()
        answer_type = meta.get("answer_type", "exactMatch")
        pipeline_id = meta.get("pipeline_id", "stateful-react-agent")
        image_path = meta.get("image_path", "")
        file_path = meta.get("file_path", "") or image_path
        question_id = meta.get("question_id", "")

        # Pass trial dir to nodes that write per-task artifacts (e.g. SWE trajectory)
        meta = dict(meta)
        meta["_trial_dir"] = str(logs_dir)
        meta["file_path"] = file_path
        meta["image_path"] = image_path
        try:
            from benchmarks.public.sandbox_profiles import apply_sandbox_profile

            apply_sandbox_profile(
                meta,
                worktree=logs_dir / "sandbox" / "worktree",
            )
        except Exception as exc:
            logger.warning(
                "Benchmark sandbox profile skipped for question %s: %s",
                question_id or "(unknown)",
                exc,
            )

        # ── 2. Execute pipeline via session ──────────────────────
        if self._session is None:
            raise RuntimeError("setup() was not called before run()")
        state: dict[str, Any] | None = None
        pipeline_error: str | None = None
        try:
            state = await self._session.run(
                instruction, meta=meta, pipeline_id=pipeline_id,
            )
        except Exception as exc:
            pipeline_error = str(exc)
            logger.error("Pipeline error: %s", exc)
        task_id = state.get("task_id", "") if state else ""
        logger.info(
            "Task %s ran question %s",
            task_id, question_id or "(unknown)",
        )

        # ── 5. Extract answer ────────────────────────────────────
        predicted = _extract_predicted_answer(state, answer_type)
        if meta.get("_collect_outputs"):
            from benchmarks.public.file_render import render_dir

            rendered = render_dir(logs_dir / "sandbox" / "outputs")
            if rendered:
                predicted = rendered

        if not predicted:
            predicted = "<ANSWER_NOT_FOUND>"

        duration = time.monotonic() - t0

        # ── 6. Write final_answer.txt ────────────────────────────
        (agent_dir / "final_answer.txt").write_text(
            predicted, encoding="utf-8",
        )

        # ── 7. Write error.log if pipeline errored ───────────────
        errors = state.get("errors", []) if state else []
        if pipeline_error or errors:
            lines = []
            if pipeline_error:
                lines.append(f"Pipeline error: {pipeline_error}")
            for e in errors:
                lines.append(str(e))
            (agent_dir / "error.log").write_text(
                "\n".join(lines), encoding="utf-8",
            )

        # ── 8. Populate context.metadata ─────────────────────────
        if context.metadata is None:
            context.metadata = {}
        context.metadata.update({
            "question_id": question_id,
            "answer_type": answer_type,
            "predicted_answer": predicted,
            "duration_seconds": round(duration, 2),
            "pipeline_id": pipeline_id,
        })

        logger.info(
            "Completed question %s in %.1fs → %s",
            question_id or task_id,
            duration,
            predicted[:60],
        )

    # ── Private helpers ──────────────────────────────────────────

    def _read_task_metadata(self) -> dict[str, str]:
        """Read task metadata from the trial's config.json → task.toml.

        Harbor writes a config.json in logs_dir with a 'task' key
        pointing to the task source directory, which contains task.toml.
        Falls back to empty dict if any file is missing.
        """
        config_path = self.logs_dir / "config.json"
        if not config_path.exists():
            logger.debug("No config.json at %s", config_path)
            return {}

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read config.json: %s", exc)
            return {}

        task_source = config.get("task", {})
        # task_source may be a dict with a 'path' key or a string path
        if isinstance(task_source, dict):
            task_path = task_source.get("path", "")
        else:
            task_path = str(task_source)

        if not task_path:
            return {}

        toml_path = Path(task_path) / "task.toml"
        if not toml_path.exists():
            logger.debug("No task.toml at %s", toml_path)
            return {}

        try:
            toml_text = toml_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read task.toml: %s", exc)
            return {}

        return parse_task_metadata(toml_text)


def _extract_predicted_answer(
    state: dict[str, Any] | None,
    answer_type: str,
) -> str:
    """Extract the benchmark-scored answer from pipeline state.

    stateful-react-agent sets ``state["final_answer"]`` to the short answer string;
    use it verbatim.
    """
    if not state:
        return ""
    answer_sentinel = str(state.get("answer_sentinel") or "").strip()
    if answer_sentinel:
        return answer_sentinel
    if state.get("answer_status") == "not_found":
        return "<ANSWER_NOT_FOUND>"
    for key in ("final_answer", "report", "answer", "output"):
        value = state.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
