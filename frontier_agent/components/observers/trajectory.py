"""Write workflow-neutral trajectories in JSON and/or JSONL formats.

JSON snapshots are atomically replaced; JSONL events are flushed incrementally
so partial runs remain readable without retaining full payloads in memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, Literal

from frontier_agent.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    CompactionEvent,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)

_FORMATS: tuple[str, ...] = ("json", "jsonl")
_DEFAULT_FORMATS: tuple[str, ...] = _FORMATS
_STREAM_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
_DEFAULT_FORMAT_ENV_VARS = (
    "FRONTIER_AGENT_TRAJECTORY_FORMATS",
    "SWARM_TRAJECTORY_FORMATS",
)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


# ── Trajectory memory guardrails ──────
#
# One instance of this observer exists per agent (main + every sub-agent).
# Messages are spooled one-per-line to disk, and two additional bounds keep
# each public JSON snapshot cheap:
#
#   _BODY_MAX_CHARS — per tool-result body kept in the JSON envelope.
#     Uncapped, a single entry could hold a full ``TOOL_RESULT_MAX_CHARS``
#     (150_000) payload; × turns × agents that dominated worker RAM. When
#     JSONL is enabled (the default), that stream still records every body
#     untruncated. JSON-only configurations intentionally keep only the cap.
#   _COALESCE_N / _COALESCE_MS — snapshot batching, mirroring the
#     equivalent defaults in the external worker-trace observer.
#     Bounds the O(n²) whole-document copy to
#     one per batch. The copy is streamed without a second full string.
#
# 0 disables the respective bound.
_BODY_MAX_CHARS: int = _env_int("SWARM_TRAJECTORY_BODY_MAX_CHARS", 16384)
_COALESCE_N: int = _env_int("SWARM_TRAJECTORY_COALESCE_N", 20)
_COALESCE_MS: float = float(_env_int("SWARM_TRAJECTORY_COALESCE_MS", 0))


def _clip(text: str, limit: int) -> str:
    """Head-slice *text* to *limit* chars, marking what was dropped."""
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n"
        f"... [truncated {len(text) - limit} of {len(text)} chars]"
    )


def _resolve_formats(
    arg: Iterable[str] | None,
    env_vars: Iterable[str] = _DEFAULT_FORMAT_ENV_VARS,
) -> set[str]:
    """Resolve enabled formats from an argument, environment, or defaults."""
    if arg is not None:
        return {f for f in arg if f in _FORMATS}
    env = ""
    for name in env_vars:
        env = os.getenv(name, "")
        if env:
            break
    if env:
        return {f.strip() for f in env.split(",") if f.strip() in _FORMATS}
    return set(_DEFAULT_FORMATS)


class TrajectoryFileObserver(BaseObserver):
    """Saves an agent's main / sub-agent traces in one or more formats."""

    critical: bool = False

    def __init__(
        self,
        output_dir: Path,
        *,
        filename: str | None = None,
        formats: Iterable[str] | None = None,
        tools: list[Any] | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        format_env_vars: Iterable[str] = _DEFAULT_FORMAT_ENV_VARS,
        tool_schema_detail: Literal["full", "minimal"] = "full",
        include_start_tool_names: bool = True,
    ) -> None:
        """Args:
            output_dir: Directory where trajectory file(s) are written.
            filename: Optional stem (without extension). Falls back to the
                loop's ``task_id`` when omitted. Sub-agents pass their
                session name so each agent gets its own file.
            formats: Subset of ``("json", "jsonl")``. ``None`` → env var
                ``FRONTIER_AGENT_TRAJECTORY_FORMATS`` (or legacy
                ``SWARM_TRAJECTORY_FORMATS``) → both.
            tools: Optional list of native ``Tool`` instances / dict
                objects bound to this agent's LLM. Serialized to OpenAI
                ``{"type": "function", "function": {...}}`` schema and
                emitted as the JSON envelope's top-level ``tools`` field
                so a replay consumer can reconstruct the LLM call
                signature. Tool *names* (derived from this list) are
                also recorded in the JSONL ``start`` event for
                lightweight downstream consumers.
            model_name: The model id (e.g. ``"Qwen3-235B"``). Recorded
                on both formats so post-run renderers know what ran.
            system_prompt: System prompt seeded into the loop. Recorded
                on the JSONL ``start`` event so post-run renderers can
                reconstruct the full message history without re-reading
                the JSON envelope.
            user_message: Initial user message. Same purpose as
                ``system_prompt``.
        """
        self._dir = output_dir
        self._filename = filename
        self._formats = _resolve_formats(formats, format_env_vars)
        self._tools_schema = self._serialize_tools(
            tools or [], detail=tool_schema_detail,
        )
        self._tool_names = []
        if include_start_tool_names:
            self._tool_names = [
                (t.get("function", {}) or {}).get("name", "")
                for t in self._tools_schema
                if isinstance(t, dict)
            ]
            self._tool_names = [n for n in self._tool_names if n]
        self._model_name = model_name or ""
        self._system_prompt = system_prompt or ""
        self._user_message = user_message or ""
        self._task_id: str = ""
        self._role_id: str = ""
        self._max_turns: int = 0
        self._turns_used: int = 0
        self._tool_calls_count: int = 0
        self._stopped_by: str = ""

        # JSON envelope state. Messages are append-only on disk rather than
        # retained in a per-agent list for the whole run. The public
        # ``<stem>.json`` snapshot is streamed from this spool.
        self._message_spool_handle: IO[str] | None = None
        self._message_count: int = 0
        self._emitted_prior: bool = False
        self._tool_call_ids: dict[int, list[str]] = {}
        self._tool_results_seen: dict[int, int] = {}
        # Flush coalescing cursors (see ``_flush_json``). ``0.0`` rather than
        # ``time.monotonic()`` so the FIRST flush is always due: the envelope
        # then appears on disk immediately instead of only after the first
        # batch fills, which matters when the process is SIGKILLed (OOM) —
        # a stale-but-present file beats no file at all.
        self._pending_flushes: int = 0
        self._last_flush_at: float = 0.0

        # JSONL append state
        self._jsonl_handle: IO[str] | None = None

    # ── Path helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe(stem: str) -> str:
        return stem.replace("/", "__").replace(":", "_")

    @staticmethod
    def _serialize_tools(
        tools: list[Any], *, detail: Literal["full", "minimal"] = "full",
    ) -> list[dict]:
        """Convert native :class:`Tool` instances / dicts to OpenAI schema.

        Accepts a heterogeneous list (``Tool`` / already-serialized dict)
        and returns OpenAI
        ``{"type": "function", "function": {"name", "description",
        "parameters"}}`` entries. A wire-critical byte-exact schema pinned
        on ``Tool.metadata["openai_schema"]`` is preferred over recomputing
        via ``to_openai_schema()``. Best effort: tools that can't be
        introspected fall through to a name + description stub.
        """
        out: list[dict] = []
        for t in tools:
            if isinstance(t, dict):
                if "function" in t and "type" in t:
                    out.append(t)
                elif "name" in t:
                    out.append({"type": "function", "function": t})
                continue
            if detail == "full":
                pinned = (getattr(t, "metadata", None) or {}).get("openai_schema")
                if isinstance(pinned, dict):
                    out.append(pinned)
                    continue
                schema_fn: Callable[[], dict[str, Any]] | None = getattr(
                    t, "to_openai_schema", None,
                )
                if callable(schema_fn):
                    try:
                        out.append(schema_fn())
                        continue
                    except Exception:
                        pass
            out.append({
                "type": "function",
                "function": {
                    "name": getattr(t, "name", None) or t.__class__.__name__,
                    "description": getattr(t, "description", "") or "",
                    "parameters": (
                        getattr(t, "parameters", {}) or {}
                        if detail == "full"
                        else {}
                    ),
                },
            })
        return out

    def _path(self, ext: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        stem = self._filename or self._task_id or "trace"
        return self._dir / f"{self._safe(stem)}.{ext}"

    def _message_spool_path(self) -> Path:
        # Deliberately avoid a ``.jsonl`` suffix: trajectory consumers glob
        # that namespace and expect every match to use the public event schema.
        return self._path("messages.spool")

    # ── JSONL writer ────────────────────────────────────────────────────

    def _write_jsonl(self, record: dict) -> None:
        if "jsonl" not in self._formats:
            return
        if self._jsonl_handle is None:
            self._jsonl_handle = open(  # noqa: SIM115
                self._path("jsonl"), "a", encoding="utf-8",
            )
        record.setdefault("ts", time.time())
        for chunk in _STREAM_ENCODER.iterencode(record):
            self._jsonl_handle.write(chunk)
        self._jsonl_handle.write("\n")
        self._jsonl_handle.flush()

    def _close_jsonl(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    # ── JSON envelope writer ────────────────────────────────────────────

    def _append_message(self, message: dict) -> None:
        """Append one OpenAI message to the private on-disk spool."""
        if self._message_spool_handle is None:
            self._message_spool_handle = open(  # noqa: SIM115
                self._message_spool_path(), "w", encoding="utf-8",
            )
        for chunk in _STREAM_ENCODER.iterencode(message):
            self._message_spool_handle.write(chunk)
        self._message_spool_handle.write("\n")
        self._message_spool_handle.flush()
        self._message_count += 1

    def _close_message_spool(self, *, cleanup: bool = False) -> None:
        if self._message_spool_handle is not None:
            self._message_spool_handle.close()
            self._message_spool_handle = None
        if cleanup:
            with contextlib.suppress(FileNotFoundError):
                self._message_spool_path().unlink()

    def _write_envelope(self, path: Path) -> None:
        """Stream one compatible JSON envelope without materialising it."""
        fields: list[tuple[str, Any]] = [
            ("task_id", self._task_id),
            ("role_id", self._role_id),
            ("max_turns", self._max_turns),
            ("turns_used", self._turns_used),
            ("tool_calls_count", self._tool_calls_count),
            ("stopped_by", self._stopped_by),
            ("tools", self._tools_schema),
        ]
        if self._model_name:
            fields.append(("model_name", self._model_name))
        with path.open("w", encoding="utf-8") as fh:
            fh.write("{")
            for idx, (key, value) in enumerate(fields):
                if idx:
                    fh.write(",")
                for chunk in _STREAM_ENCODER.iterencode(key):
                    fh.write(chunk)
                fh.write(":")
                for chunk in _STREAM_ENCODER.iterencode(value):
                    fh.write(chunk)
            fh.write(',"messages":[')
            spool_path = self._message_spool_path()
            if spool_path.exists():
                with spool_path.open("r", encoding="utf-8") as spool:
                    first = True
                    for line in spool:
                        # A hard crash can leave one partial tail record.
                        # Complete spool records always end in a newline.
                        if not line.endswith("\n") or not line.strip():
                            continue
                        if not first:
                            fh.write(",")
                        first = False
                        fh.write(line.rstrip("\n"))
            fh.write("]}")

    def _flush_json(self, *, force: bool = False) -> None:
        """Rewrite the JSON envelope, coalescing bursts unless *force*.

        The envelope is a single document, so every flush still copies all
        message chunks from the append-only spool. Coalescing bounds that
        O(n²) I/O to one copy per ``_COALESCE_N`` events or
        ``_COALESCE_MS`` milliseconds. The copy is streamed, so neither the
        message history nor its serialised twin is materialised in memory.
        ``on_loop_end`` forces a final flush; the public JSONL event stream
        remains the live-tail surface.
        """
        if "json" not in self._formats:
            return
        if not force:
            self._pending_flushes += 1
            now = time.monotonic()
            due = (
                self._last_flush_at == 0.0
                or (
                    _COALESCE_N > 0
                    and self._pending_flushes >= _COALESCE_N
                )
                or (
                    _COALESCE_MS > 0
                    and (now - self._last_flush_at) * 1000.0 >= _COALESCE_MS
                )
            )
            if not due:
                return
        self._pending_flushes = 0
        self._last_flush_at = time.monotonic()
        path = self._path("json")
        tmp = path.with_suffix(path.suffix + ".tmp")
        self._write_envelope(tmp)
        os.replace(tmp, path)

    @staticmethod
    def _synth_id(turn: int, idx: int) -> str:
        return f"call_{turn}_{idx}"

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return str(value) if value is not None else ""

    @staticmethod
    def _format_args(args: Any) -> str:
        if isinstance(args, (dict, list)):
            return json.dumps(args, ensure_ascii=False)
        return TrajectoryFileObserver._stringify(args)

    def _message_to_dict(self, m: Any) -> dict | None:
        """Pass through native OpenAI-shaped dict messages.

        Loop history is now a list of plain OpenAI-wire dicts
        (``{"role", "content", "tool_calls"?, "tool_call_id"?,
        "reasoning_content"?}``), so emitting them onto the JSON envelope
        is a copy. Anything that isn't a role-bearing dict is dropped.
        """
        if isinstance(m, dict) and m.get("role"):
            return dict(m)
        return None

    # ── Lifecycle hooks ─────────────────────────────────────────────────

    # ── Recovery handle ─────────────────────────────────────────────────
    #
    # The JSONL stream is the only place a site-3 truncation's discarded content
    # survives (the post-processor at ``agent_loop.py:762`` persists nothing, and
    # runs AFTER ``notify_tool_result``, so the recorded body predates the cut).
    # ``recover_result`` needs to find this file, and the file is not
    # sandbox-visible, so the path is published rather than mounted.
    #
    # Published into ``ExecutionScope.metadata`` and NOT into a contextvar. This
    # hook is dispatched by ``notify_observers`` as ``asyncio.create_task`` for
    # non-critical observers, and this observer is non-critical — a contextvar
    # ``.set()`` inside that task mutates only the task's own copy of the context
    # and is invisible to the loop. The scope OBJECT, by contrast, is shared by
    # reference through the inherited context, so mutating its dict here is seen
    # by the loop and by every nested tool-call task. Same channel
    # ``agent_loop.py`` already uses for ``current_turn``.
    SCOPE_KEY = "trajectory_jsonl"

    def _publish_jsonl_path(self) -> None:
        """Advertise this agent's JSONL file, or withdraw the key when there is
        none to advertise.

        Absence of the key is the signal that no handle can be minted, so a
        jsonl-disabled configuration degrades to today's behaviour instead of
        minting handles that resolve to nothing.
        """
        from frontier_agent.core.execution_context import (
            get_current_execution_scope,
        )
        scope = get_current_execution_scope()
        if scope is None:
            return
        if "jsonl" not in self._formats:
            scope.metadata.pop(self.SCOPE_KEY, None)
            return
        # Deliberately not ``_path()``: that mkdirs (see ``_path``), and a
        # "where is my trajectory" answer must not create directories.
        stem = self._safe(self._filename or self._task_id or "trace")
        scope.metadata[self.SCOPE_KEY] = str(self._dir / f"{stem}.jsonl")

    def _withdraw_jsonl_path(self) -> None:
        from frontier_agent.core.execution_context import (
            get_current_execution_scope,
        )
        scope = get_current_execution_scope()
        if scope is not None:
            scope.metadata.pop(self.SCOPE_KEY, None)

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._task_id = config.task_id
        self._role_id = config.role_id
        self._max_turns = config.max_turns
        # After ``_task_id`` is set: it is the stem fallback when no explicit
        # ``filename`` was passed.
        self._publish_jsonl_path()

        start_record: dict = {
            "t": "start",
            "task_id": config.task_id,
            "role_id": config.role_id,
            "max_turns": config.max_turns,
        }
        if self._model_name:
            start_record["model_name"] = self._model_name
        if self._system_prompt:
            start_record["system_prompt"] = self._system_prompt
        if self._user_message:
            start_record["user_message"] = self._user_message
        if self._tool_names:
            start_record["tool_names"] = self._tool_names
        self._write_jsonl(start_record)
        self._flush_json()

    async def on_llm_response(
        self, ctx: TurnContext,
    ) -> Intervention | None:
        # JSONL: append one event
        record: dict = {
            "t": "llm",
            "turn": ctx.turn,
            "content": ctx.ai_text,
            "tool_calls": [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in (ctx.tool_calls or [])
            ],
        }
        if ctx.thinking:
            record["thinking"] = ctx.thinking
        if ctx.usage:
            record["usage"] = ctx.usage
        if ctx.thinking_blocks:
            # Native Anthropic thinking / OpenAI Responses reasoning: the
            # verbatim block list (thinking+signature / reasoning+
            # encrypted_content / text) so the (sub-)agent trajectory stays
            # replay-able with signatures / encrypted reasoning intact.
            record["thinking_blocks"] = ctx.thinking_blocks
        self._write_jsonl(record)

        # JSON envelope: emit prior context once, then this turn.
        if "json" in self._formats:
            if not self._emitted_prior:
                # ctx.messages includes the assistant we just got; skip it
                # and emit ourselves below with reasoning_content split out.
                for m in list(ctx.messages or [])[:-1]:
                    d = self._message_to_dict(m)
                    if d:
                        self._append_message(d)
                self._emitted_prior = True

            msg: dict = {
                "role": "assistant",
                "content": self._stringify(ctx.ai_text),
            }
            if ctx.thinking:
                msg["reasoning_content"] = self._stringify(ctx.thinking)
            if ctx.usage:
                # Per-turn usage (incl. reasoning_tokens) so the JSON envelope
                # carries the same usage as the .jsonl stream.
                msg["usage"] = ctx.usage
            if ctx.thinking_blocks:
                # Verbatim thinking / reasoning blocks (signatures /
                # encrypted_content) so the JSON envelope stays replay-able.
                msg["thinking_blocks"] = ctx.thinking_blocks
            if ctx.tool_calls:
                ids: list[str] = []
                tcs: list[dict] = []
                for idx, tc in enumerate(ctx.tool_calls):
                    cid = tc.get("id") or self._synth_id(ctx.turn, idx)
                    ids.append(cid)
                    tcs.append({
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": self._format_args(
                                tc.get("args", {}),
                            ),
                        },
                    })
                msg["tool_calls"] = tcs
                self._tool_call_ids[ctx.turn] = ids
                self._tool_results_seen[ctx.turn] = 0
                self._tool_calls_count += len(tcs)

            self._append_message(msg)
            self._turns_used = ctx.turn
            self._flush_json()
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        # ``tool_call_id`` is what makes a JSONL entry addressable: a turn can
        # hold several results from the same tool (parallel_tool_calls is on),
        # so ``(turn, name)`` does not identify one. Recorded straight off the
        # result rather than through the JSON branch's synthesis fallback below
        # — that fallback advances ``_tool_results_seen``, so sharing it would
        # double-count, and a synthesised id matches nothing outside the
        # snapshot anyway. Empty here means the runtime itself had no id.
        self._write_jsonl({
            "t": "result",
            "turn": ctx.turn,
            "name": result.name,
            "tool_call_id": getattr(result, "tool_call_id", "") or "",
            "result": result.result,
            "error": result.is_error,
            "ms": result.duration_ms,
        })

        if "json" in self._formats:
            cid = getattr(result, "tool_call_id", "") or ""
            if not cid:
                ids = self._tool_call_ids.get(ctx.turn, [])
                seen = self._tool_results_seen.get(ctx.turn, 0)
                cid = (
                    ids[seen] if seen < len(ids)
                    else self._synth_id(ctx.turn, seen)
                )
                self._tool_results_seen[ctx.turn] = seen + 1
            body = _clip(self._stringify(result.result), _BODY_MAX_CHARS)
            self._append_message({
                "role": "tool",
                "tool_call_id": cid,
                "content": f"[error] {body}" if result.is_error else body,
            })
            self._flush_json()
        return None

    async def on_compaction(self, event: CompactionEvent) -> None:
        """Record what a compaction discarded and what it kept in its place.

        The summary is written whole. It is the one artifact that explains a
        rewrite the rest of the trajectory cannot show — the replaced turns are
        gone from the history by the time anything reads it — and it is bounded
        by the summariser's own output length, not by tool output, so it does
        not need the ``_BODY_MAX_CHARS`` guard that tool results do.
        """
        self._write_jsonl({
            "t": "compaction",
            "turn": event.turn,
            "seq": event.seq,
            "selected": event.selected,
            "tokens_before": event.tokens_before,
            "tokens_after": event.tokens_after,
            "relief_met": event.relief_met,
            "spill_refs": event.spill_refs,
            "attempts": event.attempts,
            "summary": event.summary,
            "rollback_reason": event.rollback_reason,
        })

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        self._turns_used = max(self._turns_used, result.turns_used)
        self._tool_calls_count = result.tool_calls_count
        self._stopped_by = (
            getattr(result, "stopped_by", "") or self._stopped_by
        )

        self._write_jsonl({
            "t": "end",
            "turns": result.turns_used,
            "tool_calls": result.tool_calls_count,
            "stopped_by": result.stopped_by,
        })
        self._close_jsonl()
        # The file stays on disk and stays readable; withdrawing the key only
        # stops a handle being minted against a loop that has ended.
        self._withdraw_jsonl_path()
        # force=True: the terminal envelope must be complete regardless of
        # where the coalescing cursors happen to sit.
        snapshot_written = False
        try:
            self._flush_json(force=True)
            snapshot_written = True
        finally:
            # Always release the FD. Keep the forensic spool when the terminal
            # snapshot fails; remove it only after a successful atomic replace.
            self._close_message_spool(cleanup=snapshot_written)

    async def on_loop_cancelled(self) -> None:
        """Close live handles while preserving crash-forensic sidecars."""
        self._close_jsonl()
        self._withdraw_jsonl_path()
        self._close_message_spool()
