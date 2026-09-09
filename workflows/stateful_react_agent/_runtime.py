"""Runtime helpers for the native stateful ReAct agent."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import ClassVar

from frontier_agent.components.finalization import (
    COMMON_RECOVERY_NUDGE_PREFIXES,
    build_recovery_context,
    chat_with_fallback_budget,
    minimal_best_effort_answer,
)
from frontier_agent.core.llm import LLMClient
from frontier_agent.core.loop_types import AgentLoopResult, ToolResult
from frontier_agent.core.messages import (
    Message,
    assistant_msg_with_reasoning,
    text_of,
    user_msg,
)
from frontier_agent.core.runtime.loop.tool_exec import ToolResultPostProcessor
from frontier_agent.infra.config import get_config
from plugins.tools.bash import BASH_STDERR_SEPARATOR
from plugins.tools.document_node_toolchain import (
    render_document_node_toolchain_note,
)
from workflows.stateful_react_agent.prompts import get_summarize_prompt

logger = logging.getLogger(__name__)

SANDBOX_FS_NOTE = (
    "\n\nFILESYSTEM CONVENTION: Your current working directory /workspace is "
    "private to you. Read task input files from /inputs when provided. Write "
    "ALL intermediate, scratch, and temporary files to /workspace. Write ONLY "
    "your final deliverable file(s) to /outputs. The /outputs directory is "
    "collected and graded verbatim, so keep it clean: no scratch files, no "
    "intermediate or duplicate versions, only the final deliverable(s) exactly "
    "as the task requests."
)

# matplotlib clips silently: an annotation straddling the edge of its axes is
# drawn with the outside half missing and no warning, and clipped white text on
# a white background leaves no trace for the eyeball check. This workflow wires
# no SkillInjectionMiddleware, so it never sees
# plugins/skills/chart-visualization — the rule has to ride in the prompt.
# agent_team carries its own copy for the same reason.
CHART_NOTE = (
    "\n\nCHART RULE: matplotlib clips silently — a label, badge or legend that "
    "straddles the edge of the axes is drawn with its outside half missing and "
    "no warning at all. So whenever you produce a chart: keep annotations "
    "wholly inside the axes (`y=0.98, va='top'`) or wholly outside "
    "(`y=1.02, va='bottom'`), never straddling `y=1.0`; pass `clip_on=False` to "
    "anything drawn outside the axes (`ax.text` defaults to False, but "
    "`ax.add_patch` defaults to True, so a hand-rolled badge gets clipped); "
    "prefer `ax.legend()`, which measures itself and is never clipped; and save "
    "with `bbox_inches='tight'`. Prove it in code before `savefig`: call "
    "`fig.canvas.draw()`, then for every `ax.texts` entry, `ax.get_legend()` and "
    "every standalone `ax.patches` decoration (bars are owned by "
    "`ax.containers` and are supposed to clip — skip those) compare "
    "`get_window_extent()` against `get_clip_box()` and fail loudly if anything "
    "falls outside its clip box. Do not rely on looking at the image — "
    "clipped white text on a white background is invisible."
)

# Everything appended to the system prompt once the filesystem tools are live:
# the /workspace-vs-/outputs convention, plus the chart rule (charts are
# produced through those same tools).
SYSTEM_PROMPT_NOTES = SANDBOX_FS_NOTE + CHART_NOTE


def render_system_prompt_notes(
    *,
    sandbox_mode: str,
    tool_names: list[str] | tuple[str, ...],
) -> str:
    """Return static filesystem/chart guidance plus discovered toolchains."""
    filesystem_note = SANDBOX_FS_NOTE
    project = os.environ.get("FRONTIER_AGENT_PROJECT_DIR", "").strip()
    if sandbox_mode == "native":
        workspace = os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", os.getcwd())
        inputs = os.environ.get("FRONTIER_AGENT_INPUTS_DIR", "inputs")
        outputs = os.environ.get("FRONTIER_AGENT_OUTPUTS_DIR", "outputs")
        filesystem_note = (
            "\n\nFILESYSTEM CONVENTION (native mode): Your current working "
            f"directory {workspace} is the workspace. Read task inputs from "
            f"{inputs}. Write final deliverables to {outputs}. Keep scratch "
            "and intermediate files in the workspace, not the outputs directory. "
            "Scientific and plotting packages are optional in native mode. "
            "Office writer packages used by create_file are already installed. "
            "Do not pre-install optional packages; if the task needs a missing package, "
            "install only that package with `python -m pip install <package>`. "
            "It will be placed in the workspace-local native dependency overlay."
        )
    if project:
        workspace = os.environ.get("FRONTIER_AGENT_WORKSPACE_DIR", "/workspace")
        if os.path.realpath(project) != os.path.realpath(workspace):
            filesystem_note += (
                f" The user's project/coding directory is {project}; read or "
                "edit repository files there, and use the workspace above only "
                "for clones, downloads, drafts, generated probes, and other "
                "run-private intermediate files."
            )
    return filesystem_note + CHART_NOTE + render_document_node_toolchain_note(
        sandbox_mode=sandbox_mode,
        tool_names=tool_names,
    )

# Graceful terminations — the loop ran out of its turn / context budget with
# real history to summarize. A forced final-answer is always worth attempting.
# ``response_truncated`` belongs here rather than with the infra errors: the run
# was mid-work with real history, and the LAST assistant text is a fragment cut
# mid-token. Without the rescue that fragment IS the answer — which is how a run
# whose visible output was "The shell ate my `" scored zero.
_GRACEFUL_FINAL_STOP_REASONS = frozenset({
    "max_turns", "context_limit_reached", "response_truncated", "no_tool",
})

# Abnormal / infra terminations: the LLM endpoint exhausted retries
# (``llm_error``), the wall-clock or token budget was hit (``wall_deadline`` /
# ``budget_exhausted``), or the runaway-rollback guard tripped
# (``max_attempts``). ``wall_deadline`` and ``max_attempts`` are emitted by the
# current engine (agent_loop.py:411,790) and did not exist on the source loop.
#
# Salvaging these is a best-effort fallback: try to coax a plain-text
# answer out of whatever history exists.  ``salvage_infra_errors=False`` skips
# the extra LLM rescue call for fail-fast benchmark scenarios, but the node
# still returns an explicit deterministic partial-status answer rather than an
# error sentinel.
_INFRA_FINAL_STOP_REASONS = frozenset(
    {"llm_error", "budget_exhausted", "wall_deadline", "max_attempts"},
)


def _forced_final_stop_reasons(salvage_infra_errors: bool) -> frozenset[str]:
    if salvage_infra_errors:
        return _GRACEFUL_FINAL_STOP_REASONS | _INFRA_FINAL_STOP_REASONS
    return _GRACEFUL_FINAL_STOP_REASONS


_FINAL_ANSWER_NUDGE = (
    "Please provide your final answer now based on all information gathered. "
    "Do not call any more tools; reply with plain text."
)
# Retrying a deterministic 400 against the same poisoned tool-call history used
# to waste 4m30s before returning ``<ANSWER_NOT_FOUND>``.  We now ask from a
# clean, single-user-message recovery context and retry only once for
# transient/empty responses.  The caller's finalization timeout is honoured as
# the *per fallback-leg* read timeout; a separate finite outer guard gives every
# configured fallback leg a chance to run without allowing a custom client that
# ignores ``timeout=`` to hang forever.
_FORCE_FINAL_MAX_RETRIES = 2
_FORCE_FINAL_WAIT_S = 1
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")
_RECOVERY_NUDGE_PREFIXES = (
    # This workflow's own summarize/report prompts, on top of the framework set.
    "Summarize the above conversation",
    "Please provide the final research summary",
    *COMMON_RECOVERY_NUDGE_PREFIXES,
)
_RECOVERY_EMPTY_FALLBACK = (
    "No reliable intermediate text was recoverable. Produce the most "
    "useful honest answer possible from the original task alone."
)

_LEAKED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*tool_call[^>]*>[\s\S]*?<\s*/\s*tool_call\s*>", re.IGNORECASE,
)
_LEAKED_TOOL_RESPONSE_BLOCK_RE = re.compile(
    r"<\s*tool_response[^>]*>[\s\S]*?<\s*/\s*tool_response\s*>", re.IGNORECASE,
)
_LEAKED_FUNCTION_BLOCK_RE = re.compile(
    r"<\s*function\s*=[\s\S]*?<\s*/\s*function\s*>", re.IGNORECASE,
)
_LEAKED_TAG_FRAGMENT_RE = re.compile(
    r"<\s*(tool_call|tool_response|function)\b[^>]*>?", re.IGNORECASE,
)

# CommonMark fence marker: backtick OR tilde, any run length >= 3. A fixed
# ```` ``` #```` (3-backtick) regex mismatches a legitimate 4-backtick
# ```` ````markdown ```` block — it "closes" on the first 3 of the 4 closing
# backticks, leaving a stray backtick that turns the whole thing into a
# single-backtick inline code span instead of unwrapping it.
_FENCE_LINE_RE = re.compile(r"^[ \t]{0,3}(?P<char>`|~)(?P<run>(?P=char){2,})")

# A single line that is ENTIRELY one image — no leading/trailing prose.
_IMAGE_ONLY_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\(([^)\n]+)\)[ \t]*$")

# ``[N] Title. PATH_OR_URL`` — the exact References-line shape
# ``get_report_prompt`` (prompts.py) mandates. Line-anchored so an in-text
# ``[3]`` citation mid-sentence never matches; only a full standalone line in
# this shape does.
_REFERENCE_LINE_RE = re.compile(r"^\[\d+\]\s+.+?\.\s+(\S+)\s*$", re.MULTILINE)

_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _fence_marker(line: str) -> tuple[str, int] | None:
    """Return ``(char, length)`` when ``line`` opens/closes a fence, else ``None``."""
    match = _FENCE_LINE_RE.match(line)
    if not match:
        return None
    return match.group("char"), len(match.group("run")) + 1


def _sandbox_ref_key(ref: str) -> tuple[str, str] | None:
    """``(dir_hint, basename)`` for a local sandbox ref, ``None`` for a URL.

    ``dir_hint`` is ``"outputs"``, ``"inputs"``, or ``""`` when the ref has no
    recognised sandbox prefix at all (a bare filename).
    """
    s = ref.strip()
    if _URL_SCHEME_RE.match(s):
        return None
    s = s.lstrip("./")
    match = re.match(r"^/?(outputs|inputs)/(.+)$", s)
    if match:
        return match.group(1), match.group(2).rsplit("/", 1)[-1]
    return "", s.rsplit("/", 1)[-1]


def _refs_match(cited: str, embedded: str) -> bool:
    """True when ``embedded`` (an image src) is the same source as ``cited``.

    A URL matches only by exact string — there is no local-execution
    analogue to the path drift handled below, so tolerating a mismatch there
    would be an unverified guess rather than a real signal. A sandbox path
    matches by basename, tolerant of either side using a bare filename
    instead of the full ``/outputs/...`` / ``/inputs/...`` form (e.g. a bash
    script that prints a cwd-relative save path rather than the absolute
    one) — but ``outputs`` and ``inputs`` never satisfy each other even when
    the basename collides, since they mean different things (generated vs.
    user-provided).
    """
    cited, embedded = cited.strip(), embedded.strip()
    if _URL_SCHEME_RE.match(cited) or _URL_SCHEME_RE.match(embedded):
        return cited == embedded
    key_c, key_e = _sandbox_ref_key(cited), _sandbox_ref_key(embedded)
    if key_c is None or key_e is None:
        return False
    dir_c, base_c = key_c
    dir_e, base_e = key_e
    if base_c != base_e:
        return False
    return dir_c == "" or dir_e == "" or dir_c == dir_e


def unwrap_fenced_images(text: str) -> str:
    """Strip a fence wrapping ONLY image markdown so it renders as an image.

    The reporter sometimes wraps a bare ``![alt](path)`` in a
    ```` ```markdown ```` / ```` ```md ```` / bare ```` ``` ```` fence, which
    renders as literal text instead of an image. Fence detection follows
    CommonMark's closing rule (same character, closer run length >= the
    opener's) rather than a fixed-length regex, so a legitimate 4-backtick
    (or tilde) fence is scanned correctly instead of mis-parsed.

    Unwrapping is gated on the report's OWN citation discipline, not just the
    fence's content shape: every image inside a candidate block must also
    match a ``[N] Title. PATH`` line elsewhere in the SAME report (via
    :func:`_refs_match`). ``get_report_prompt`` already requires any embedded
    image to be a cited source, so this is a self-consistency check, not a
    guess — and it is what stops a block from being unwrapped when it is
    actually a deliberate demonstration (e.g. the user asked "how do I write
    markdown image syntax" and the model answered with an example — that
    example has no reason to also appear in the References list, so it stays
    fenced). A block with no matching citation, or a report with no
    References section at all, is left untouched — the safe direction; this
    can only fail to unwrap a real image, never wrongly render a fake one. An
    unclosed trailing fence is left untouched too, for the same reason.
    """
    if not text:
        return text
    cited_refs = set(_REFERENCE_LINE_RE.findall(text))
    lines = text.split("\n")
    by_start: dict[int, tuple[int, str]] = {}
    # One Optional holding both halves: as two variables the checker could not
    # see that the index and its marker are always set and cleared together, so
    # every open_marker read looked possibly-None.
    open_block: tuple[int, tuple[str, int]] | None = None
    for i, line in enumerate(lines):
        marker = _fence_marker(line)
        if marker is None:
            continue
        if open_block is None:
            open_block = (i, marker)
            continue
        open_at, open_marker = open_block
        if marker[0] != open_marker[0] or marker[1] < open_marker[1]:
            # Inner marker (different char, or shorter run): part of the
            # block's content, not this block's delimiter.
            continue
        opener_match = _FENCE_LINE_RE.match(lines[open_at])
        assert opener_match is not None
        info = lines[open_at][opener_match.end():].strip().lower()
        body = lines[open_at + 1:i]
        # Collect the matches, then require every line to have matched: ``all``
        # on a list of Optionals does not narrow the elements for the reads below.
        image_matches = [_IMAGE_ONLY_LINE_RE.match(ln) for ln in body]
        matched = [m for m in image_matches if m is not None]
        if (
            info in ("", "markdown", "md")
            and body
            and len(matched) == len(image_matches)
            and all(
                any(_refs_match(ref, m.group(1)) for ref in cited_refs)
                for m in matched
            )
        ):
            by_start[open_at] = (i, "\n".join(ln.strip() for ln in body))
        open_block = None
    if not by_start:
        return text
    out: list[str] = []
    idx = 0
    n = len(lines)
    while idx < n:
        hit = by_start.get(idx)
        if hit is not None:
            end, replacement = hit
            out.append(replacement)
            idx = end + 1
        else:
            out.append(lines[idx])
            idx += 1
    return "\n".join(out)


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def scrub_leaked_tool_calls(text: str) -> str:
    """Remove qwen text-mode tool-call markup, WITHOUT the outer strip.

    Split out of :func:`_strip_leaked_tool_calls` so a streaming filter can apply
    the same removals to each settled segment (interior whitespace must survive
    a mid-report segment boundary) and only strip once at the end.
    """
    if not text:
        return text
    out = _LEAKED_TOOL_CALL_BLOCK_RE.sub("", text)
    out = _LEAKED_TOOL_RESPONSE_BLOCK_RE.sub("", out)
    out = _LEAKED_FUNCTION_BLOCK_RE.sub("", out)
    return _LEAKED_TAG_FRAGMENT_RE.sub("", out)


def _strip_leaked_tool_calls(text: str) -> str:
    """Remove qwen text-mode tool-call markup that leaked into content."""
    if not text:
        return text
    return scrub_leaked_tool_calls(text).strip()


def _build_recovery_messages(
    messages: list[Message],
    *,
    task_description: str,
    final_prompt: str,
    context_max_tokens: int | None = None,
) -> list[Message]:
    """Build a valid tool-free context for the finalization call.

    The original history may contain an empty/truncated
    ``function.arguments`` value or orphan tool messages — exactly the kind of
    protocol defect that caused the run supplied with this bug report to 400 on
    every retry.  Flatten useful visible/user/tool text into one user message
    instead of replaying any tool protocol fields.
    """
    recovery_context = build_recovery_context(
        messages,
        strip_thinking=_strip_thinking,
        strip_leaked_tool_calls=_strip_leaked_tool_calls,
        nudge_prefixes=_RECOVERY_NUDGE_PREFIXES,
        empty_fallback=_RECOVERY_EMPTY_FALLBACK,
        fixed_prompt_text=(
            f"{final_prompt}\n\nOriginal task:\n{task_description}\n\n"
            "## Clean recovery context\n"
        ),
        context_max_tokens=context_max_tokens,
    )
    prompt = (
        f"{final_prompt}\n\n"
        "## Clean recovery context\n"
        "The original tool protocol was intentionally removed because it may "
        "contain malformed or orphaned tool calls. Use the observations below "
        "only as partial progress; do not call tools.\n\n"
        f"Original task:\n{task_description}\n\n"
        f"{recovery_context}"
    )
    return [user_msg(prompt)]


def _minimal_best_effort_answer(
    task_description: str,
    stopped_by: str,
    *,
    language: str = "",
) -> str:
    """Always provide a user-facing answer even when the rescue LLM is down."""
    return minimal_best_effort_answer(
        task_description, stopped_by, language=language,
    )


def _is_non_retryable_finalize_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "error code: 400" in text
        or "badrequest" in text
        or "context length" in text
        or "longer than the model" in text
    )


async def force_final_answer(
    result: AgentLoopResult,
    llm: LLMClient,
    timeout: float,
    task_description: str = "",
    *,
    thinking_format: str = "tag",
    salvage_infra_errors: bool = True,
    final_prompt: str | None = None,
    language: str = "",
) -> AgentLoopResult:
    """Run a tool-free LLM call to extract a plain-text final answer.

    ``salvage_infra_errors`` (default ``True``) also makes an LLM rescue call on
    abnormal/infra exits (:data:`_INFRA_FINAL_STOP_REASONS`). Set ``False`` to
    skip that call; the node-level deterministic best-effort response still
    guarantees non-empty user output.
    """
    forced = _forced_final_stop_reasons(salvage_infra_errors)
    if result.metadata.get("final_answer") or result.stopped_by not in forced:
        return result

    prompt = (
        final_prompt
        or (
            get_summarize_prompt(task_description)
            if task_description
            else _FINAL_ANSWER_NUDGE
        )
    )
    recovery_messages = _build_recovery_messages(
        list(result.messages),
        task_description=task_description,
        final_prompt=prompt,
    )
    ask_timeout = max(float(timeout), 1.0)

    text = ""
    rescue_mode = "deterministic_fallback"
    last_exc: Exception | None = None
    for attempt in range(_FORCE_FINAL_MAX_RETRIES):
        try:
            resp = await chat_with_fallback_budget(
                llm,
                recovery_messages,
                per_leg_timeout_s=ask_timeout,
            )
            content = _strip_leaked_tool_calls(_strip_thinking(text_of(resp.content)))
            if content:
                result.messages.extend(recovery_messages)
                result.messages.append(
                    assistant_msg_with_reasoning(
                        content,
                        getattr(resp, "reasoning_content", "") or "",
                        thinking_format=thinking_format,
                    ),
                )
                text = content
                rescue_mode = "clean_context_llm"
                break
            last_exc = RuntimeError("LLM returned empty content")
            logger.warning(
                "force_final_answer attempt %d/%d returned empty content",
                attempt + 1,
                _FORCE_FINAL_MAX_RETRIES,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "force_final_answer attempt %d/%d failed: %s",
                attempt + 1,
                _FORCE_FINAL_MAX_RETRIES,
                exc,
            )
            if _is_non_retryable_finalize_error(exc):
                break

        if attempt < _FORCE_FINAL_MAX_RETRIES - 1:
            await asyncio.sleep(_FORCE_FINAL_WAIT_S)

    if not text and last_exc is not None:
        logger.warning("force_final_answer exhausted retries (%s)", last_exc)
    if not text:
        text = _strip_leaked_tool_calls(_strip_thinking(result.final_content))
        if text:
            rescue_mode = "existing_partial"
    if not text:
        text = _minimal_best_effort_answer(
            task_description, result.stopped_by, language=language,
        )
    result.metadata["final_answer"] = text
    result.metadata["final_answer_rescued"] = True
    result.metadata["final_answer_rescue_mode"] = rescue_mode
    result.metadata["final_answer_source"] = rescue_mode
    result.final_content = text
    logger.info(
        "force_final_answer injected after %s (%d chars)",
        result.stopped_by,
        len(text),
    )
    return result


class ReactToolResultPostProcessor(ToolResultPostProcessor):
    """Per-tool char budgets for fresh tool-result content."""

    # read_file already paginates itself and its trailing offset is part of the
    # recovery contract. A second head-only cut would either drop that footer
    # or create a silent gap before the next offset.
    _PASS_THROUGH = frozenset({
        "read_file", "web_fetch", "web_search",
        # Paginates itself and its trailing offset IS the recovery contract —
        # the same reason read_file is whitelisted. Head-capping it would cut the
        # continuation pointer off the content the call exists to fetch back, so
        # recovery would silently return less than it says it did.
        "recover_result",
    })

    # Uniform scheme for the high-volume tools: most cut at 8K upstream (gate ①
    # — regular read_file pages at max_chars=8K + "PARTIAL READ … offset=N" hint;
    # bash/run_python_code/grep/glob overflow to disk at meta.max_result_chars=8K
    # + "saved to <path>, use read_file" pointer), and this post-processor (gate
    # ②) gives 10K of headroom so the 8K body PLUS its trailing footer both
    # survive into history. A second cut at/below 8K would chop the footer and
    # lose the recovery path — exactly the bug that left results silently
    # truncated with no way to fetch the rest.
    _BUDGETS: ClassVar[dict[str, int]] = {
        "grep_search": 10_000,
        "glob_search": 10_000,
    }
    _CONFIGURED_EXEC_TOOLS = frozenset({"bash", "run_python_code"})
    _EXEC_FOOTER_HEADROOM = 2_000
    _BUDGET_DEFAULT = 6_000
    _BASH_STDERR_KEEP = 3_000
    _ELLIPSIS = (
        "\n\n[... body truncated for context budget; re-fetch or rerun with "
        "a focused query if you need the rest ...]"
    )

    def process(self, tool_result: ToolResult) -> str:
        content = (
            tool_result.result
            if isinstance(tool_result.result, str)
            else str(tool_result.result)
        )
        if tool_result.is_error:
            return content

        name = tool_result.name or ""
        if name in self._PASS_THROUGH:
            return content

        if name in self._CONFIGURED_EXEC_TOOLS:
            # Gate ① is configurable. Gate ② must track it or a deployment
            # raising TOOL_EXEC_RESULT_MAX_CHARS can lose the overflow pointer.
            budget = (
                get_config().tool_exec_result_max_chars
                + self._EXEC_FOOTER_HEADROOM
            )
        else:
            budget = self._BUDGETS.get(name, self._BUDGET_DEFAULT)
        if name == "bash":
            return self._compact_bash(content, budget)
        return self._head_cap(content, budget)

    def _head_cap(self, content: str, budget: int) -> str:
        if len(content) <= budget + len(self._ELLIPSIS):
            return content
        return content[:budget] + self._ELLIPSIS

    def _compact_bash(self, content: str, budget: int) -> str:
        if len(content) <= budget:
            return content

        if BASH_STDERR_SEPARATOR in content:
            stdout, stderr = content.split(BASH_STDERR_SEPARATOR, 1)
        else:
            stdout, stderr = content, ""

        stderr_keep = stderr[-self._BASH_STDERR_KEEP:] if len(stderr) > self._BASH_STDERR_KEEP else stderr
        overhead = len(BASH_STDERR_SEPARATOR) + 80 if stderr_keep else 0
        stdout_budget = max(500, budget - len(stderr_keep) - overhead)
        if len(stdout) > stdout_budget:
            half = stdout_budget // 2
            stdout_keep = (
                stdout[:half]
                + f"\n[... {len(stdout) - stdout_budget} stdout chars elided ...]\n"
                + stdout[-half:]
            )
        else:
            stdout_keep = stdout

        if stderr_keep:
            return stdout_keep + BASH_STDERR_SEPARATOR + stderr_keep
        return stdout_keep
