# Writing a workflow

[Documentation index](../docs/README.md) · [Framework architecture](framework.md)

A workflow is a plugin package under `workflows/`. It owns its topology, profiles,
prompts, node functions, observers, and terminal semantics. Generic behavior should
remain in `frontier_agent/`; public benchmark integrations belong in
`benchmarks/public/`.

## 1. Define agents

Create `AgentDefinition` objects with stable role IDs, prompts, and explicit tool
allowlists. Register them from the package's `register(context)` function. Resolve
tools through the shared registry rather than importing benchmark code.

## 2. Define the pipeline

Export a module-level `PipelineSpec` from `workflows/<name>/spec.py`:

```python
PIPELINE = PipelineSpec(
    pipeline_id="my-workflow",
    name="My Workflow",
    entry_point="run",
    terminal_nodes=["run"],
    nodes=[NodeDefinition(
        node_id="run",
        role_id="my_role",
        node_function="workflows.my_workflow.nodes.run_node",
        context_policy=ContextPolicy(
            include_fields=["original_question", "task_id", "metadata"],
        ),
        output_fields=["final_answer", "answer_status"],
    )],
    transitions=[TransitionSpec(from_phase="run", to_phase="__END__")],
)
```

Treat `PipelineSpec.pipeline_id` as the canonical runtime identifier and use it
exactly in benchmark defaults, CLI examples, and stored task metadata. Python
package names may use underscores independently. If compatibility aliases are
registered, document them explicitly; do not infer an alias by swapping dashes and
underscores.

## 3. Implement nodes

Node functions are async callables receiving pipeline state and `NodeContext`. Pass
workflow policy into `LoopConfig`, observers, compaction, and tool selection; do not
add workflow-specific phases or terminal rules to `run_agent_loop`.

Read the selected profile from `state["metadata"]["profile"]`. Compatibility keys
may be fallbacks, but `profile` is the runner contract. For file-aware workflows,
consume `_sandbox_mounts`, `_sys_prompt_addendum`, and related metadata prepared by
the benchmark adapter instead of importing `benchmarks.public`.

## 4. Add profiles

Store a small set of public YAML profiles under `profiles/`. Credentials and public
endpoints must use environment placeholders such as `${OPENAI_API_KEY}` and
`${OPENAI_BASE_URL}`. Do not commit internal endpoints, account IDs, model paths, or
benchmark results. Document profile differences in the workflow README.

## 5. Register the plugin

Expose `register(context)` from `workflows/<name>/__init__.py`. The workflow loader
discovers plugin packages, and benchmark bootstrap separately discovers module-level
specs. Registration should be idempotent and must not import the eval layer.

## 6. Test the contract

At minimum, test:

- canonical pipeline registration and any explicitly supported aliases;
- profile defaults and terminal routing;
- pure observer and parsing behavior;
- tool allowlists;
- a kernel bootstrap using the new pipeline;
- real functional paths for source-exec bundles or sandbox behavior.

Run the standard gates:

```bash
uv run ruff check frontier_agent benchmarks workflows plugins tests
uv run python tools/import_smoke.py --stage 1
uv run python tools/import_smoke.py --stage 2
uv run python tools/check_symbols.py
uv run pytest -q
```
