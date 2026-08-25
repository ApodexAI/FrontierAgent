# Framework architecture

[Documentation index](../docs/README.md) · [Workflow authoring](workflows.md)

FrontierAgent separates the reusable agent runtime from workflow plugins, tool
plugins, and benchmark evaluation. The framework layer has no dependency on
`benchmarks`; CI enforces that boundary with a framework-only import smoke.

```text
frontier_agent/   generic loop, scheduling, registries, AgentBus, observers
plugins/tools/   tool implementations and sandbox policy
workflows/       pipeline specs, profiles, prompts, workflow-owned observers
benchmarks/      public harness plus bundled FrontierSearchBench/FrontierChallenge
```

## Runtime flow

`PipelineSpec` describes nodes, state visibility, transitions, and terminal nodes.
Workflow plugins register specs and `AgentDefinition` objects. The scheduler builds
the selected graph, resolves each node function, applies its `ContextPolicy`, and
merges declared outputs back into pipeline state.

Agent nodes call `run_agent_loop`, the domain-neutral ReAct kernel:

```python
result = await run_agent_loop(
    system_prompt=system_prompt,
    user_message=question,
    llm=llm,
    tools=tools,
    config=loop_config,
    observers=observers,
    model_profile=model_profile,
)
```

The loop binds tools and session identity, calls the LLM, parses tool calls,
executes authorized tools, notifies observers, and applies compaction. Its
implementation is split into focused modules under
`frontier_agent/core/runtime/loop/`: `_bind`, `_call`, `_streaming`, `_response`,
`_runaway`, and `_tool`. `llm_client` remains the stable compatibility facade for
existing imports. Workflow semantics such as planning, terminal-tool behavior,
reporter routing, and recovery belong outside the kernel.

## Observer contract

Observers can implement only the callbacks they need:

| Callback | Purpose |
|---|---|
| `on_loop_start` | Initialize per-run state |
| `on_llm_attempt` | Observe retries, provider attempts, and failures |
| `on_llm_delta` | Inspect streaming chunks |
| `on_llm_response` | Validate or alter a completed assistant turn |
| `on_tool_call` | Approve, annotate, or interrupt a tool call |
| `on_tool_result` | Inspect and reshape tool outcomes |
| `on_turn_end` | Apply turn-level stopping or retry policy |
| `on_loop_end` | Final telemetry and cleanup |
| `on_loop_cancelled` | Cancellation-safe cleanup |

Callbacks return an `Intervention` when they need to stop, retry, replace content,
or continue without consuming the normal turn budget. Observer failures are handled
according to the loop contract; cleanup and telemetry should remain best-effort,
while authorization observers must fail closed.

## Agent teams

`AgentBus` provides task submission, messaging, report collection, cancellation,
and shared context. `SpawnGuard` limits nesting, parallelism, and task budgets. The
tool registry exposes only the explicit OSS allowlist; adding a Python module under
`plugins/tools/` does not automatically make it agent-accessible.

## Sandboxing

File and shell tools share a task-scoped sandbox. Supported backend values are:

- `auto`: probe bubblewrap and fail with guidance if isolation is unavailable.
- `bwrap`: require bubblewrap and Linux user namespaces.
- `container`: trust the surrounding task container as the isolation boundary.

There is no unisolated host fallback. `/inputs` is read-only, `/workspace` is the
working directory, and `/outputs` is the only persistent deliverable location for
file benchmarks. Network and path policies are applied before command execution,
and authorization or sandbox failures are fail-closed.

Output publication is manifest-aware: only declared publishers may write final
deliverables, and `/outputs/scratch` is reserved for persisted intermediate work.
Agent Team sub-agents receive scoped workspaces while sharing approved inputs and
outputs.

The interactive terminal adds a second layer on top of this — an approval gate,
hard denials that survive `--yes`, and a journal backing `/revert`. Those are
documented in [`apodex/README.md`](../apodex/README.md#safety), and the trace and
log paths each session writes in
[run artifacts and timestamps](run-artifacts.md).

## Package boundaries

Base dependencies run the framework and workflows. Install `plugins` for optional
third-party tool SDKs, `sandbox` and `document-readers` for file tools, and `eval`
for benchmark datasets, Harbor, and judges. This keeps framework consumers from
installing the evaluation stack.
