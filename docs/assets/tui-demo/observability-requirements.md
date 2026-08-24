# Mock TUI observability requirements

> Demo asset only. This is a fictional product specification and is not a
> statement of FrontierAgent's implemented behavior.

## Product context

Developers use the terminal application for repository analysis and long-running
agent tasks. A single task may last 20–40 minutes and may involve several
background workers. Users must be able to understand progress and redirect work
without restarting the session.

## Requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| OBS-01 | Task progress | The UI exposes planned, active, completed, and cancelled work, plus an overall completion summary. |
| OBS-02 | Tool visibility | Users can see current and completed tool calls, state, elapsed time, and a compact detail view. |
| OBS-03 | Team visibility | Parallel workers are visually distinguishable from the coordinator and from one another. Queued and running workers must not look identical. |
| OBS-04 | Worker detail | A user can inspect a worker's recent thinking, messages, tool calls, results, and failures without leaving the main TUI. |
| OBS-05 | Mid-run steering | Plain-text guidance entered during a run is acknowledged immediately, queued visibly, and delivered at a safe coordinator or agent turn boundary. |
| OBS-06 | Steering boundary | Steering must not claim to cancel an in-flight LLM request, tool call, or already-dispatched worker. Immediate stopping is a separate action. |
| OBS-07 | Protected actions | File writes and protected commands show the target and a command or diff preview. Rejection must be the safe default. |
| OBS-08 | Dangerous actions | High-risk operations require stronger confirmation than an ordinary single-key approval. |
| OBS-09 | Deliverables | Session outputs are listed with relative names and sizes and can be previewed without modifying them. |
| OBS-10 | Preview coverage | Source, Markdown, tabular data, common office documents, images, and archives have either an in-terminal preview or a clear unsupported-state message. |
| OBS-11 | Responsive layout | The transcript remains usable in terminals narrower than the preferred desktop width. Hidden secondary UI must be recoverable. |
| OBS-12 | Recovery | An interrupted session retains enough completed state to resume, and the UI explains the recovery boundary accurately. |
| OBS-13 | Long-session review | Users can filter or search the transcript and jump to or copy the final report. |
| OBS-14 | Configuration safety | Provider diagnostics may show endpoint and model information but must never reveal the API key. |

## Priority policy

- P0: a behavior could cause unauthorized execution, secret disclosure, or loss
  of user work.
- P1: a core workflow is misleading, unavailable, or difficult to recover.
- P2: the behavior works but is inefficient, unclear, or lacks adequate tests.
