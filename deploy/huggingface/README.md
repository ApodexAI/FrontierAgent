# FrontierAgent — Hugging Face Space demo (deployment runbook)

[中文](README.zh-CN.md) · [Space landing page](README.space.md)

A Gradio web demo that runs the **real** FrontierAgent `react` workflow. The
Space hosts the UI, the agent runtime, sessions, queueing and output downloads;
the model is called over HTTP at an OpenAI-compatible endpoint.

```
Browser → Gradio (app.py) → FrontierAgentAdapter (adapter.py)
        → stateful-react-agent pipeline (workflows/stateful_react_agent)
        → OPENAI_BASE_URL  (external model serving)
```

No model weights are downloaded or served here. `OPENAI_BASE_URL` is the only
thing that decides which model answers.

---

## Contents

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI. Consumes structured events; contains no agent logic. |
| `adapter.py` | The only bridge to the runtime: events, queueing, timeout, cancel. |
| `config.py` | Every knob, from environment variables, plus preflight validation. |
| `security.py` | Secret redaction, demo-safe tool policy, download containment. |
| `containment.py` | Refuses tool calls naming a path outside the session. |
| `sessions.py` | Per-visitor session ids and directory trees. |
| `events.py` | The event vocabulary the UI consumes. |
| `errors.py` | Upstream failure → one actionable sentence. |
| `mock_llm.py` | A fake OpenAI-compatible endpoint for local runs and CI. |
| `poc.py` | Headless end-to-end check of runtime → tool → answer → artifact. |
| `Dockerfile` | The Space image. Build context is the **repository root**. |
| `README.space.md` | The README to publish *as the Space's own* README. |
| `.env.space.example` | Template for `--env-file` local runs. |
| `publish.sh` | Assembles a Space working tree (never pushes). |
| `README.zh-CN.md` | Chinese counterpart of *this* file, condensed around the HF-side launch. **This file is authoritative**; when they disagree, that one is out of date. |

---

## How it attaches to the runtime

Nothing in `frontier_agent/` was changed to make this demo exist, and nothing
here re-implements the agent loop. Three seams the runtime already exposed carry
the whole integration, which is why the CLI and TUI are unaffected by anything
in this directory:

| Seam | What it buys |
| --- | --- |
| `metadata['sdk_extra_observers']` | A plain `BaseObserver` receives streaming deltas, tool start/finish and the final answer — no stdin, no Textual, no ANSI parsing. |
| `metadata['pause_check']` | **Stop** is cooperative: the agent lands at the next turn boundary with whatever answer it has, instead of being killed mid-tool. |
| `metadata['profile_overrides']` | Model, endpoint, credentials and every demo bound are injected here, so replacing the model touches neither code nor the profile YAML. |

Two details in the observer are load-bearing, and both are commented at the
declaration — don't "clean them up":

* it must declare `wants_llm_delta`, or the loop decides nobody wants streaming
  and the answer appears in one lump at the end;
* it must set `critical = True`, or its hooks are dispatched as background tasks
  with no ordering guarantee and the streamed text interleaves out of order.

---

## 1. `docker build`

Build from the repository root, not from this directory:

```bash
docker build -f deploy/huggingface/Dockerfile -t frontier-agent-hf-space .
```

## 2. `docker run`

```bash
cp deploy/huggingface/.env.space.example .env.space.local
$EDITOR .env.space.local          # fill in endpoint + key + model

docker run --rm \
  -p 127.0.0.1:7860:7860 \
  --env-file .env.space.local \
  frontier-agent-hf-space
```

The app listens on `0.0.0.0:7860` **inside** the container; the `-p` above
publishes it only to the host's loopback, so a development run is never exposed
to the internet. On a Hugging Face Space, `7860` is the port HF expects.

`.env.space.local` is git-ignored (`.gitignore` covers `.env*`) and excluded
from the image (`.dockerignore`). Never bake a key into the image.

## 3. Required Variables

Set these as Space **Variables** (visible, non-secret):

| Variable | Required | Meaning |
| --- | --- | --- |
| `OPENAI_BASE_URL` | **yes** | OpenAI-compatible API base, ending in `/v1`. See §5. |
| `OPENAI_MODEL` | **yes** | The model name the *endpoint* serves — goes in the request body. |
| `HF_MODEL_ID` | no | Display label, default `apodex/Apodex-1.1-mini`. |
| `HF_MODEL_URL` | no | Display link; defaults to the model's HF page. |
| `DEMO_WORKFLOW` | no | Only `react` is supported in this release. |
| `DEMO_MAX_TURNS` | no | Agent turn budget per task (default `24`). |
| `DEMO_MAX_OUTPUT_TOKENS` | no | Per-call output cap (default `4096`). |
| `DEMO_TASK_TIMEOUT_SECONDS` | no | Wall clock per task (default `600`). |
| `DEMO_MAX_CONCURRENCY` | no | Accepted but **clamped to 1** — see §13. |
| `DEMO_QUEUE_SIZE` | no | Waiting runs before new ones are refused (default `4`). |
| `DEMO_SESSION_TTL_SECONDS` | no | Session directory lifetime (default `3600`). |
| `DEMO_MAX_PROMPT_CHARS` | no | Prompt length cap (default `4000`). |
| `DEMO_PUBLIC_MODE` | no | `true` (default) enforces the demo-safe toolset. |
| `DEMO_ALLOWED_TOOLS` | no | Comma list; can only *narrow* the safe set. |
| `DEMO_REPORTER` | no | `true` adds a final report-synthesis LLM call (slower). |
| `DEMO_RUNTIME_ROOT` | no | Session storage root; defaults to `/data` if mounted, else `$HOME`. |
| `DEMO_LOG_LEVEL` | no | `WARNING` (default), `INFO`, `DEBUG`. |
| `SANDBOX_BACKEND` | preset | `native` — set in the image; see §13. Do not change casually. |

`DEMO_TASK_TIMEOUT_SECONDS` is the only time knob you set. Everything else the
runtime needs is derived from it in `config.DemoConfig.profile_overrides()` —
the per-tool and per-LLM-call ceilings, the research/finalisation split, the
logical-call ceiling, and the reasoning-runaway guards. That derivation is not
cosmetic: the workflow profile these override (`workflows/stateful_react_agent/
profiles/tui.yaml`) is tuned for an interactive session with a 9000s research
wall, and several of its bounds are individually longer than a demo run is
allowed to be. Raising the task timeout scales all of them together; editing the
profile YAML instead would desynchronise them.

## 4. Required Secrets

Set these as Space **Secrets** (hidden, never echoed):

| Secret | Required | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | **yes** | Credential for `OPENAI_BASE_URL`. |
| `SERPER_API_KEY` | recommended | Without it `web_search` returns **zero results**, and the agent looks broken. |
| `SUMMARY_LLM_API_KEY` | recommended | Credential for the extraction endpoint below. |
| `JINA_API_KEY` | optional | Improves `web_fetch` page retrieval; it falls back to direct HTTP. |

`web_search` and `web_fetch` also accept `SERPER_BASE_URL` / `JINA_BASE_URL`
Variables when you route them through an internal proxy rather than the vendors'
public endpoints.

### Page extraction (`SUMMARY_LLM_*`) — needed for `web_fetch`

The `react` profile uses the *aligned* `web_fetch`, whose content extraction is
itself an LLM call. Without an extraction endpoint every fetch returns
`Extraction failed: SUMMARY_LLM_BASE_URL not set`, and the agent is reduced to
search snippets — it still answers, just noticeably worse, and nothing else
explains why. Startup warns about this.

| Variable | Notes |
| --- | --- |
| `SUMMARY_LLM_BASE_URL` | A **full** chat-completions URL — `…/v1/chat/completions`, *not* an API base. This differs from `OPENAI_BASE_URL` on purpose. |
| `SUMMARY_LLM_MODEL_NAME` | Model to extract with. A small, fast model is ideal. |
| `SUMMARY_LLM_API_KEY` | Credential for that endpoint (a **Secret**). |

Pointing all three at the same endpoint as `OPENAI_BASE_URL` works:

```
OPENAI_BASE_URL      = https://example.com/api/v1
SUMMARY_LLM_BASE_URL = https://example.com/api/v1/chat/completions
```

Startup refuses to serve when a required value is missing, and prints exactly
which variable to fix.

## 5. How to configure `OPENAI_BASE_URL`

It must be the **API base** of an OpenAI-compatible service — the app appends
`/chat/completions` itself:

```
✅  https://my-endpoint.example.com/v1
✅  https://router.huggingface.co/v1
✅  http://10.0.0.7:30000/v1                    (self-hosted vLLM / SGLang)
❌  https://huggingface.co/apodex/Apodex-1.1-mini    ← a *web page*, not an API
❌  https://my-endpoint.example.com/v1/chat/completions  ← too specific
```

Preflight rejects the model-page and full-route forms by name at startup rather
than letting the first prompt fail with a confusing 404.

The endpoint **must support tool/function calling.** FrontierAgent's `react`
workflow is an agent loop, not a chat completion; an endpoint without tool
support will produce a chat-like answer with no tool activity. Report that as an
endpoint capability gap — do not "fix" it by disabling tools.

## 6. How to replace the model

Change `OPENAI_MODEL` (and `OPENAI_BASE_URL` / `OPENAI_API_KEY` if the new model
lives elsewhere), then restart the Space. **No code change, no rebuild.** The
values flow to the runtime through the profile-override seam in
`config.DemoConfig.profile_overrides()`, so the profile YAML stays untouched.

Verify with the container log line printed at startup:

```
FrontierAgent Demo: workflow=react model=… served_model=… endpoint=… runtime_root=…
```

## 7. How to create the Docker Space

1. On Hugging Face: **New Space** → SDK **Docker** → *blank* template.
2. Choose hardware (CPU Basic is enough — no model runs here).
3. Optionally enable **persistent storage** to keep session outputs across
   restarts; the app uses `/data` automatically when it is writable.

## 8. How to publish the code

Hugging Face builds from a `Dockerfile` at the **root** of the Space repo, and
this repo's root `Dockerfile` is the CLI image. `publish.sh` assembles a correct
Space working tree for you:

```bash
./deploy/huggingface/publish.sh /tmp/space-tree
```

It copies the runtime packages, puts the Space `Dockerfile` at the root, and
installs `README.space.md` as the Space `README.md`. Then:

```bash
cd /tmp/space-tree
git init && git add -A && git commit -m "FrontierAgent react demo"
git remote add space https://huggingface.co/spaces/<org>/<space>
git push --force space HEAD:main
```

`publish.sh` never pushes by itself and never copies `.env*`, so a credential
cannot leave your machine by accident.

## 9. How to configure Space Variables / Secrets

Space **Settings** → *Variables and secrets*. Use §3 for Variables and §4 for
Secrets. Adding or changing either restarts the Space; a rebuild is not needed.

## 10. How to release

1. Push the code (§8) and wait for **Building → Running** in the Space UI.
2. Watch the logs for the `FrontierAgent Demo: …` startup line.
3. If it printed configuration errors instead, fix those variables — the app
   deliberately refuses to serve on a broken configuration.

## 11. How to smoke test

Locally, with no endpoint and no token at all:

```bash
# terminal 1 — a fake OpenAI-compatible endpoint that also scripts a tool call
uv run python -m deploy.huggingface.mock_llm --port 8018 --tool-demo

# terminal 2 — headless: runtime → tool → artifact → final answer
SANDBOX_BACKEND=native uv run python -m deploy.huggingface.poc
```

Through the browser, against the built image:

```bash
docker run --rm -p 127.0.0.1:7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8018/v1 \
  -e OPENAI_API_KEY=mock-key -e OPENAI_MODEL=Apodex-1.1-mini-mock \
  frontier-agent-hf-space
```

From your workstation, tunnel and open it — nothing is published publicly:

```bash
ssh -L 7860:127.0.0.1:7860 USER@SERVER
# then browse http://localhost:7860
```

Against a real Space, submit a task that needs a tool, e.g.
*"Search for the ReAct paper and save a two-paragraph summary to
outputs/react.md"*, and check that the page shows `running` → tool activity →
streamed answer → a downloadable file.

The automated suite covers all of this without any credential:

```bash
uv run pytest tests/test_hf_space_config.py tests/test_hf_space_runtime.py \
              tests/test_hf_space_ui.py -q
```

## 12. How to roll back

* **Configuration mistake** — restore the previous Variable/Secret value; the
  Space restarts without a rebuild.
* **Bad code** — in the Space repo, `git push --force space <previous-sha>:main`,
  or use *Settings → Factory rebuild* to rebuild the current commit cleanly.
* **Locally** — images are tagged, so `docker run` an earlier tag.

---

## 13. What this demo deliberately does *not* do

These are design decisions, not omissions. Changing them needs a matching
change to the safety story.

**One run at a time.** `DEMO_MAX_CONCURRENCY` is accepted and then clamped to 1,
with a warning. The FrontierAgent runtime keeps per-process global state — the
service registry that `BenchmarkSession` snapshots and restores, the bash-policy
and task-sandbox context variables, and the mount-directory environment read
per run — so two concurrent runs in one process would not be isolated from each
other. Scale with **replicas**, not with this value.

**No shell, no code execution, no downloads.** `bash`, `run_python_code`,
`download_file`, the file editors and every sub-agent tool are denied outright by
`security.HARD_DENIED_TOOLS`, and `DEMO_ALLOWED_TOOLS` can only narrow the set
further — never widen it. `security.demo_safe_tool_policy` installs this as the
runtime's process-wide tool policy before any turn runs.

**`create_file` is not in the toolset**, so the demo produces text deliverables
(via `write_file`) but not `.docx`/`.xlsx`/`.pptx`. This *is* a policy choice —
it stopped being a technical one. The tool used to fail unconditionally with
`E2BIG` (a ~131 KB base64 writer bundle inlined into a single `sh -c`
argument); that was fixed upstream in `773ece0`, which pipes the bundle over
stdin, and `openpyxl` / `python-docx` / `python-pptx` are now hard dependencies
of the package rather than an in-session `pip install`, so the image already
ships them. Two things would have to change before enabling it here:

* it is the only remaining tool that spawns a subprocess — the native writer
  runs a model-authored JSON program under this app's own interpreter — in a
  demo whose whole containment story is "no model-authored command executes";
* `containment.PathContainmentObserver` inspects top-level path arguments only,
  and `create_file` carries secondary destinations *inside* `ops` (an
  `image` target, `export_pdf`'s `out`). Those would be left to the tool's own
  `_write_roots` check, which also accepts the literal `/workspace` and
  `/outputs` — harmless on a Space where neither exists and the app user cannot
  create them, but not a boundary this demo asserts anywhere else.

**`SANDBOX_BACKEND=native`, not `container`.** In `native` mode the react prompt
names the session's *real* directories, which is what makes per-visitor
workspaces possible; `container` mode hard-codes `/workspace` and `/outputs` for
everyone. Neither mode is a weaker boundary here, because no model-authored
command can execute at all. Note that this value is cached by
`frontier_agent.infra.config` on first read, so it must be present in the
environment **before the process starts** — which is why the Dockerfile sets it.
Startup verifies the *effective* mode and refuses to serve if it disagrees.

**Session isolation is by unguessable id.** Each visitor gets
`<runtime-root>/sessions/<24-byte-random-id>/`, and `outputs/` is nested inside
`workspace/` because the react node authorises exactly one write root (the value
of `FRONTIER_AGENT_WORKSPACE_DIR`). `state/` and `inputs/` sit outside that root,
so the agent cannot alter its own trace files, and `containment` separates read
from write allowances so uploaded `inputs/` stay readable but immutable.
Downloads are served only from a session's own `outputs/`. Three things enforce
that, and it is worth being precise about which does what: the UI never turns
user input into a path — it hands Gradio a vetted list from
`list_output_files`, which excludes dotfiles, symlinks leaving the tree,
credential-shaped filenames and files containing a secret; Gradio's
`allowed_paths` / `blocked_paths` reject anything else asked for over HTTP
(verified: traversal, `/etc/passwd`, `/app`, `/proc/self/environ` and another
session's `state/` all return 403); and session ids are unguessable.
`security.resolve_download` is the equivalent guard for a *by-name* caller —
it is what any future download API should go through, and is covered by tests,
but no current code path needs it.

**Secrets never leave the process.** Three layers, because each covers a
different escape:

* `security.StreamRedactor` masks the answer stream *statefully*. Per-chunk
  matching is unsound — SSE boundaries are arbitrary, so a key split across two
  frames matches neither half and concatenating the deltas rebuilds it.
* `containment.SecretArgumentObserver` strips configured secrets out of tool
  arguments before the tool runs, so a hostile endpoint cannot have the agent
  write its API key into a deliverable that visitors then download.
* `list_output_files(..., redactor=…)` withholds any file whose *contents* carry
  a secret, as a second line behind the tool boundary.

The image contains no `.env` and no key.

**One caveat, recorded rather than papered over:** the runtime's own
`TrajectoryFileObserver` writes each raw assistant message — tool arguments
included — under `state/` *before* any observer can rewrite it, so a trace can
contain whatever the endpoint sent. That is contained, not scrubbed: `state/` is
outside the agent's authorised root, is never served to a browser, and is
deleted with the session.
