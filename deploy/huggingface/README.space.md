---
title: FrontierAgent Demo
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
models:
  - apodex/Apodex-1.1-mini-GPTQ-Int4
  - apodex/Apodex-1.1-mini
short_description: Run the FrontierAgent react agent workflow in your browser.
---

# FrontierAgent Demo

This Space runs the **real** [FrontierAgent](https://github.com/ApodexAI/FrontierAgent)
`react` agent workflow — the same `stateful-react-agent` pipeline the project's
CLI uses — in your browser.

Give it a task. It plans, searches the web, reads and writes files in a private
workspace, and streams its answer back as it works. If it produces a
deliverable, you can download it.

## Trying it

Type a task and press **Run**. Good first prompts:

* *Summarise what the ReAct pattern is and save it as `outputs/react.md`.*
* *Compare two open-source vector databases and list the trade-offs.*
* *Find the current recommended way to pin Python dependencies and write it up.*

While it runs you will see queue/running status, each tool the agent starts and
finishes, the answer streaming in, and any files it produced.

**Stop** asks the agent to finish its current step and return whatever it has —
it is not a kill switch, so you still get a partial answer. **Clear** empties
your workspace. **New session** gives you a fresh, empty one.

## What the model is

The agent's reasoning happens on an external OpenAI-compatible endpoint; no
model weights run inside this Space. The header shows which model is configured.

## Limits and privacy

* One task runs at a time, so you may briefly queue behind another visitor.
* Tasks have a turn budget and a wall-clock limit, then land with a best-effort
  answer.
* Your session gets its own private directory. Other visitors cannot see or
  download your files, and a new session starts empty.
* Sessions are temporary and are cleaned up automatically. Download anything you
  want to keep.
* Please don't paste secrets or personal data into a public demo.

## What the agent cannot do here

This is a public demo, so the agent runs with a deliberately narrow toolset: it
can search the web, fetch pages, and read and write files inside your own
session directory. It has **no shell**, cannot run code, cannot install
packages, cannot download arbitrary files, and cannot reach anything outside its
own workspace. Office-document deliverables (`.docx`/`.xlsx`/`.pptx`) are not
available in this demo; text formats such as Markdown and CSV are.

For the full toolset — including shell access inside a sandbox — run
FrontierAgent locally:

```bash
uv run frontier-agent --mode react --cwd /path/to/project
```

## Self-hosting this demo

The Space is built from `deploy/huggingface/` in the FrontierAgent repository.
`deploy/huggingface/README.md` there is the deployment runbook: build and run
commands, the full list of Variables and Secrets, how to point it at your own
endpoint, and how to roll back.
