# TUI demo queries

[TUI user guide](tui-user-guide.md) · [Endpoint quickstart](install/tui-endpoint-quickstart.md)

These three English queries are designed for product screenshots and developer
demos. They use only the current repository and the included mock attachments;
web-search credentials are not required.

## Demo 1 — ReAct startup call-chain investigation

Use the `react` workflow. This demonstrates the Plan and Activity tabs, source
inspection, test discovery, and a Markdown deliverable.

```text
Analyze the complete execution path from invoking the FrontierAgent CLI to displaying the full-screen TUI and running the first agent task.

Requirements:
1. Identify the CLI entry point, argument parsing, runtime selection, session construction, TUI application initialization, input routing, and task execution path.
2. Explain each stage as “file path → class or function → responsibility.”
3. Cite at least eight concrete code locations. Do not infer behavior from filenames alone.
4. Identify the tests that verify the most important stages and call out meaningful coverage gaps.
5. Distinguish macOS and Linux behavior where the execution path differs.
6. Do not modify existing source files.
7. Save the final report to /outputs/tui-startup-call-chain.md.
```

Good screenshots: a populated Plan tab, sequential file/search operations in
Activity, and the report opened from Files with `Space`.

## Demo 2 — Agent Team parallel architecture audit

Use the `agent_team` workflow. This demonstrates task decomposition, parallel
workers, the `SUB-AGENTS` and `COORDINATOR` groups, cross-checking, and multiple
deliverables.

```text
Perform an evidence-based architecture audit of the current FrontierAgent repository using Agent Team.

Create and dispatch separate parallel investigations for at least these areas:
1. CLI configuration, runtime selection, and session lifecycle;
2. TUI layout, input routing, approvals, and deliverable previews;
3. The react and agent_team workflow execution models;
4. Native, bubblewrap, and Docker security boundaries;
5. Test coverage and agreement between maintained documentation and code.

Requirements:
- Give every sub-agent a narrow, non-overlapping assignment.
- Require concrete file or test evidence for every material conclusion.
- Cross-check high-impact claims with a second source or a verifier.
- Label findings as CODE-CONFIRMED, TEST-CONFIRMED, DOC-ONLY, or INFERRED.
- Do not modify existing source files.
- Save the synthesized report to /outputs/architecture-audit.md.
- Save a second deliverable to /outputs/risk-register.csv with the columns severity, area, finding, evidence, and recommendation.
```

Good screenshots: Plan with several tasks, Activity with multiple live workers,
one expanded sub-agent event stream, and both outputs in Files.

## Demo 3 — Agent Team attachment review with asynchronous intervention

This scenario uses the mock files in [`docs/assets/tui-demo/`](assets/tui-demo/).
Start in `agent_team`, then attach both files before submitting the main query:

```text
/attach docs/assets/tui-demo/observability-requirements.md
/attach docs/assets/tui-demo/support-incidents.md
```

You can type `@obs` and press `Tab` to demonstrate attachment-name completion.
Then submit this main query:

```text
Use Agent Team to evaluate whether the current TUI implementation satisfies @observability-requirements.md and adequately addresses the scenarios in @support-incidents.md.

Create parallel investigations for:
1. Plan and task-progress visibility;
2. Activity, coordinator, and sub-agent observability;
3. Mid-run steering and interruption semantics;
4. Approval and permission UX;
5. Deliverables, previews, responsive layout, and session recovery.

Requirements:
- Treat the attachments as fictional product requirements and support reports, not as proof of implementation.
- Verify every implementation claim against current code and, where possible, tests.
- For every requirement, classify the result as SATISFIED, PARTIAL, MISSING, or CONTRADICTED.
- Identify any support incident caused by a documentation or expectation mismatch rather than a code defect.
- Do not modify existing source files.
- Save the final review to /outputs/tui-observability-review.md.
- Save the requirement matrix to /outputs/requirements-traceability.csv.
```

After several workers appear as running in Activity, type this plain-text
intervention in the prompt and press Enter:

```text
Additional direction: for every PARTIAL, MISSING, or CONTRADICTED result, include a concrete failure scenario and determine whether an existing automated test would detect it. Do not interrupt work already delegated; apply this requirement during verification and synthesis.
```

The status bar should show a queued intervention while current workers continue.
After the first intervention has been consumed, optionally send a second one:

```text
Revise the final presentation: begin with a concise developer-facing verdict, then show the traceability matrix, and rank recommendations as P0, P1, or P2. Explicitly separate code defects from user-expectation and documentation problems.
```

This demonstrates the intended semantics: steering is delivered to the
coordinator at a safe turn boundary. It does not directly cancel sub-agents
that have already been dispatched.

Good screenshots: attachment chips, `@` completion, multiple active workers,
the queued notice and footer count, an expanded worker, and the final CSV or
Markdown preview.
