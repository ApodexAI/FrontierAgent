"""System prompt for the stateful ReAct agent."""

from __future__ import annotations

# The reference agent's main system prompt AS IT APPEARS IN THE TRAINING DATA,
# not as it appears in the reference source. Two byte-level deltas from that
# source are intentional and load-bearing — the model saw this exact form:
#  1. ASCII apostrophe ``'`` (not the source's U+2019 ``’``)
#  2. A trailing space after "given task." with only a single ``\n`` before
#     ``# General Objective`` (the source has no trailing space and ``\n\n``)
# Kept inline (no cross-workflow import) so this workflow stays self-contained.
# Unlike the reference, this prompt carries NO date — the node still appends the
# sandbox FS note + per-benchmark addenda after this base. The public builder
# deliberately appends the mandatory English ``web_search`` rule below.
_REACT_BASE = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n"
    "\n"
    "You only have access to the tools provided. You can use multiple tools per message, and will receive the results of those tools in the user's next response. You use tools step-by-step to accomplish a given task. \n"
    "# General Objective\n"
    "\n"
    "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."
)


_SEARCH_QUERY_LANGUAGE_NOTE = """

# Search Query Language for `web_search` — MANDATORY RULE

**When using the `web_search` tool, ALL search queries MUST be in English.** \
This is a hard rule, not a suggestion. This rule applies ONLY to `web_search` \
— other search tools (e.g., document search) should use whatever language \
matches the source content.

English web search queries return higher-quality, more comprehensive results \
for virtually all topics. Even when the user writes in Chinese or another \
non-English language, you MUST formulate your `web_search` query in English.
"""


# FrontierScience problem-solving guide, appended only when ``fs_mode=True``
# (off by default; not needed for officeqa/gdpval/onemillion). Kept local so
# this workflow has no cross-workflow import.
_FS_PROBLEM_GUIDE = """

# Scientific Problem-Solving Guide

## Answer Quality (Critical for Scoring)
- **Answer EVERY sub-question.** Missing a sub-question = losing points.
- **Be specific.** Vague answers score 0. Include exact numbers, exact \
names, exact formulas.
- Every derivation must include intermediate steps, not just the final \
result.
- Every protein/compound must be named with its standard identifier.
- Nothing should be left as "further investigation needed" — give your \
best answer with reasoning.
"""


def get_react_system_prompt(*, fs_mode: bool = False) -> str:
    """Build the base ReAct system prompt.

    ``date_str`` is accepted for API stability but intentionally unused — the
    aligned base prompt carries no date (matching the training
    distribution). ``fs_mode`` optionally appends the FrontierScience
    problem-solving guide (off by default; not needed for
    officeqa/gdpval/onemillion).
    """
    prompt = _REACT_BASE + _SEARCH_QUERY_LANGUAGE_NOTE
    if fs_mode:
        prompt += _FS_PROBLEM_GUIDE
    return prompt


# Direct-inference prompt — used by profiles with ``agent.direct: true``. The
# model answers from its own parametric knowledge in a single turn (no tools are
# bound, so a tool-free reply IS the final answer).
_DIRECT_BASE = (
    "Answer the user's question directly and completely, using your own "
    "knowledge and reasoning. Think carefully, then give your best final "
    "answer. If you are uncertain, make a well-reasoned best guess rather than "
    "refusing.\n\n"
    "Your final answer MUST strictly follow any formatting instructions in the "
    "question (units, rounding, ordering, exact names, etc.)."
)


def get_direct_system_prompt() -> str:
    """Direct-inference (no-tools) system prompt.

    The model answers from its own parametric knowledge in a single turn; used
    by profiles with ``agent.direct: true`` (no tools bound → the first
    tool-free reply is the final answer).
    """
    return _DIRECT_BASE


def get_summarize_prompt(task_description: str) -> str:
    """Final-answer extraction prompt for abnormal-exit fallback.

    Copied from the reference implementation's summarize prompt for the
    main agent. Used by :func:`force_final_answer` when the loop ends without
    a natural ``no_tool`` finish (max_turns / context guard / llm_error) — it
    coaxes a plain-text answer out of the accumulated history with tools
    explicitly forbidden.
    """
    return (
        "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
        "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
        "simply extract that answer.\n"
        "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
        "The original question is repeated here for reference:\n\n"
        f'"{task_description}"\n\n'
        "Your final answer MUST strictly follow any formatting instructions in the original question — "
        "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n\n"
        "You must absolutely not perform any MCP tool call, tool invocation, search, scrape, code execution, or similar actions.\n"
        "You can only answer the original question based on the information already retrieved and your own internal knowledge.\n"
        "If you attempt to call any tool, it will be considered a mistake."
    )


def get_report_prompt(task_description: str, language: str = "English") -> str:
    """Research-report synthesis prompt for the lightweight reporter.

    Distinct from :func:`get_summarize_prompt` (a terse final-answer *extraction*
    used by the salvage fallback): this asks the model to write a structured,
    cited report over the whole conversation. Copied from the reference
    implementation's reporter prompt — so the ``[ID]`` in-text citations + References
    section + markdown structure match that reference reporter.

    ONE deliberate addition to that port: a **Format Fidelity** bullet carrying
    :func:`get_summarize_prompt`'s formatting clause. The reporter REPLACES the
    format-faithful answer, so without it a question that dictates its answer
    shape (units / rounding / ordering / ``\\boxed{}``) would come back as a
    markdown report instead.
    """
    return f"""Please provide the final research summary based only on the information already gathered.
No further tool calls are allowed.

## Requirements
- **Language**: Write the entire response in **{language}**.
- **Focus**: Directly answer the original question above. Do not just summarize gathered information — provide a clear, actionable answer.
- **Response Length**: Match the complexity of your response to the question. For simple or short questions, provide a concise and direct answer without unnecessary elaboration. For complex questions, provide a detailed and structured report.
- **Format Fidelity**: If the original question specifies an answer format — alphabetization, sequencing, units, rounding, decimal places, exact names, or a required wrapper such as `\\boxed{{...}}` — the answer MUST follow it exactly, even when that means replying with just that value and no headings.
- Use clear and structured Markdown formatting when appropriate.
- Use appropriate Markdown headings (e.g., #, ##, ###) only when the content warrants structure.
- Present key findings in an organized, concise, and readable way.
- Use tables only when they genuinely improve clarity.
- **Currency Format**: Use `\\$` instead of `$` for currency amounts (e.g., `\\$100`, `\\$1,000`) to avoid conflicts with inline math syntax.
- **Citation Format**:
  - **In-Text**: Use the format `[ID]`, where `ID` is a **numeric identifier only** (digits 0-9), e.g. `[1]`, `[2]`.
  - **References Section (if has any sources)**: At the very end, add "References" (or equivalent in {language}), one reference per line, plain text — no bullets, no blank lines between entries, and no angle brackets around the URL/filename. Each line is exactly `[ID] TITLE. URL_OR_FILENAME` with ONE URL or filename only — never join multiple paths with `;`, `,`, "and", or similar (if you used several files, give each its own `[ID]`) — and the TITLE must be a description, not a repeat of that URL/filename (don't lead with the same token that appears at the end). Example:
    [1] NVDA: NVIDIA Corp - Stock Price, Quote and News. https://www.cnbc.com/quotes/NVDA
    [2] Jane Doe's personal homepage (printout). /inputs/jane_doe.pdf
    [3] Generated histogram of the input data. /outputs/histogram.png
  - **Sandbox File Citations**: Only cite a sandbox file if its path is under `/inputs/` or `/outputs/` (at any depth, e.g. `/inputs/apex/task_files/notes.pdf` or `/outputs/subdir/report.png` are both fine) — cite it with that full path, as in the examples above. Never cite an archive (`.zip`/`.tar`/etc.) itself, `/workspace/...`, `filesystem/...`, or any other path outside `/inputs/`/`/outputs/` — including a file you yourself extracted or copied there from an archive — since those don't persist; if your only source was such a path, do not cite it at all. Never cite the final report file itself.
  - **Inline Images**: To embed an image in the body (not just cite it in References), use plain Markdown image syntax `![alt text](path)` directly in the text — never wrap it in a ```` ```markdown ``` ```` (or any other) fenced code block, or it will render as literal text instead of an image. For example:
    ❌ Wrong: ```` ```markdown\n![histogram](histogram.png)\n``` ````
    ✅ Right: `![histogram](/outputs/histogram.png)`
    For a sandbox-generated image use the same full absolute path convention as sandbox file citations — `/outputs/histogram.png`, never a relative path like `outputs/histogram.png` or `./histogram.png` — since it may need to sit alongside an `/inputs/...` reference. You may also embed an external image directly by its URL, e.g. `![alt text](https://example.com/chart.png)`, but only if you've actually confirmed it loads as a plain image — if anything you saw while fetching that URL or its page (a login wall, paywall, or CAPTCHA/anti-bot challenge) suggests it won't load directly for a reader, don't embed it; cite the source page as a normal `[ID]` reference instead.
- **Source Quality**: Do NOT cite low-credibility content farms as references (e.g., CSDN, 今日头条, 搜狐, 百家号, 知乎专栏, 简书). Prefer authoritative primary sources, official documentation, and reputable publications. Exception: local/regional topics where these may be the only available source.
- **Brand/Publication Names**: Keep English names in their original form — write "Reuters" not "路透社", "Bloomberg" not "彭博社". This applies to all brand names, publication names, and institution names that are originally in English.
- Do NOT mention tools, tool calls, or internal reasoning steps.
- Focus solely on delivering a professional, easy-to-read response that answers the user's original question.

## Original Question (for reference)
{task_description}
"""


BOARD_PROMPT_ADDENDUM = """

# Task board (external memory)

You have a task board — `add_task` and `update_task` — that is your external \
memory and the source of truth for which sub-questions remain. It is a \
CHECKLIST, not a notebook: it tracks each sub-question's RESOLUTION, not your \
findings. Keep evidence and intermediate results in your own reasoning; the \
board only records what is still open.

## ⚠️ MANDATORY FIRST ACTION — decompose before you execute

Before you do any *work* on this question you MUST call `add_task` once to \
break the goal into the full set of concrete, checkable sub-questions. **Calling \
any EXECUTION tool — `web_fetch`, `bash`, `grep_search`, `glob_search`, etc. — \
before `add_task` is a FAILURE of your job, even if you would eventually get \
the right answer.** Planning is not something you do "in your head": if it is \
not on the board, it did not happen.

What IS allowed before `add_task`: heavy reasoning, and a few NECESSARY \
`web_search` calls only to clarify a basic concept or term you need just to \
STRUCTURE the plan. Default to ZERO searches; most plans need none. Everything \
else waits for the board.

## Lifecycle rules

1. At the start — as your first action — call `add_task` with the full list of \
sub-questions, each as its own object: \
`add_task(tasks=[{"description": "..."}, {"description": "..."}])` — pass several \
at once, each one concrete and independently checkable, not a restatement of the \
whole goal. Do NOT pass bare strings; every item must be a `{"description": ...}` \
object or it is dropped. All tasks start `open`.
2. A task's normal lifecycle is `open` → `in_progress` → `resolved`, and it must \
pass through `in_progress` first. The MOMENT you start working a task, \
`update_task` it to `in_progress` \
(`update_task(updates=[{"id": "t1", "resolution": "in_progress"}])`) — this is \
the truthful "I'm on this now" \
signal and keeps the board an accurate picture of where you are. Work one task at \
a time. Do NOT mark a task `resolved` just because you have started it: starting \
is `in_progress`, not `resolved`.
3. Resolve a task ONLY after it has been `in_progress` — in almost every case you \
should mark it `in_progress` when you start and `resolved` once it is done; \
jumping straight from `open` to `resolved` is wrong unless the task genuinely \
needed no work. The INSTANT an observation ANSWERS a task AND you have \
corroborated it, `update_task` it to `resolved` — before you reason about or \
start the next task. Do not let finished tasks pile up unmarked; flipping several \
to `resolved` in one burst at the end defeats the board's purpose. `resolved` is \
your verdict "answered and corroborated", not merely "I looked at it".
4. `update_task` takes a LIST, but that is ONLY to handle the case where two \
tasks genuinely changed state in the SAME turn. It is NOT a reason to batch up \
updates across turns. Default to updating each task on its own, as it happens.
5. If a new necessary sub-question appears, `add_task` it before working on it.
6. If a task turns out to be unnecessary or was created in error, set its \
resolution to `cancelled` — it stops counting toward remaining work but stays on \
the board for the trail. Do NOT leave dead sub-questions `open`.
7. Only use task ids returned by `add_task`; never invent one. Never re-open a \
`resolved` task.
8. You are finished only when every task is `resolved` or `cancelled` and the \
goal is answered. Give the final answer as a plain-text reply with no tool call.
"""
