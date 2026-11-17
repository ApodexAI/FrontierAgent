"""Local task commands; clients share SQLite, never the worker's process lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from frontier_agent.superagent.models import Budget
from frontier_agent.superagent.store import Store


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="frontier-agent tasks", description="Persistent local tasks"
    )
    result.add_argument("--state-dir", default=".apodex/tasks", help="Control state directory")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("serve", "start"):
        command = sub.add_parser(
            name, help="Run service in foreground" if name == "serve" else "Start detached service"
        )
        command.add_argument("--image", default="apodex:local")
        command.add_argument("--concurrency", type=int, default=2)
        command.add_argument("--forward-env", action="append", default=[], metavar="NAME")
    sub.add_parser("stop", help="Gracefully stop the service; preserve its tasks")
    sub.add_parser("list", help="List tasks as JSON")
    sub.add_parser("ui", help="Open the task dashboard")
    submit = sub.add_parser("submit")
    submit.add_argument("goal")
    submit.add_argument("--title", default="")
    submit.add_argument(
        "--input",
        action="append",
        default=[],
        type=Path,
        help="Snapshot a file/directory into read-only task inputs (repeatable)",
    )
    submit.add_argument("--request-key", default=None)
    submit.add_argument(
        "--workflow", choices=("stateful-react-agent", "agent_team"), default="stateful-react-agent"
    )
    _budget_args(submit, defaults=True)
    for name in ("show", "events", "pause", "cancel", "accept", "resume", "input"):
        command = sub.add_parser(name)
        command.add_argument("ticket_id")
        if name == "resume":
            command.add_argument(
                "--reconcile", default="", help="Findings after checking uncertain operations"
            )
            _budget_args(command, defaults=False)
        if name == "input":
            command.add_argument("text")
            command.add_argument("--request-key", default=None)
    return result


def _budget_args(command: argparse.ArgumentParser, *, defaults: bool) -> None:
    for key, default in asdict(Budget()).items():
        command.add_argument(
            "--" + key.replace("_", "-"), type=int, default=default if defaults else None
        )


def _budget(args: argparse.Namespace, previous: Budget | None = None) -> Budget:
    values = asdict(previous or Budget())
    for key in values:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return Budget(**values)


def _start(root: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "frontier_agent.superagent.cli",
        "--state-dir",
        str(root),
        "serve",
        "--image",
        args.image,
        "--concurrency",
        str(args.concurrency),
    ]
    for key in args.forward_env:
        command.extend(["--forward-env", key])
    with (root / "service.log").open("ab") as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service failed to start; see {root / 'service.log'}")
        try:
            if (root / "service.ready").read_text() == str(process.pid):
                print(f"Task service started. State: {root}")
                return 0
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    print(f"Service is still starting. Inspect {root / 'service.log'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        with Store(args.state_dir) as store:
            if args.command == "start":
                return _start(store.root, args)
            if args.command == "serve":
                from apodex.userenv import load_environment
                from frontier_agent.superagent.service import DockerExecutor, serve

                load_environment()
                names = {
                    key
                    for key in os.environ
                    if key.startswith(("OPENAI_", "ANTHROPIC_", "JINA_", "SERPER_", "TAVILY_"))
                }
                names.update(
                    {"LLM_PROVIDER", "WEB_SEARCH_PROVIDER", "REACT_NO_WEB", "SWARM_NO_WEB"}
                )
                reserved = {"SANDBOX_BACKEND", "APODEX_IN_CONTAINER", "PYTHONPATH", "PATH"}
                for key in args.forward_env:
                    if (
                        not key.isidentifier()
                        or key in reserved
                        or key.startswith("FRONTIER_AGENT_")
                    ):
                        raise ValueError(
                            f"Cannot forward reserved or invalid environment name: {key}"
                        )
                    names.add(key)
                asyncio.run(
                    serve(
                        store,
                        DockerExecutor(args.image, env_names=tuple(sorted(names))),
                        concurrency=args.concurrency,
                    )
                )
                return 0
            if args.command == "stop":
                (store.root / "stop.request").touch()
                print("Service stop requested; tasks are retained.")
                return 0
            if args.command == "ui":
                from apodex.tui.tasks import TaskDashboard

                TaskDashboard(store).run()
                return 0
            if args.command == "list":
                output = [asdict(task) for task in store.list()]
            elif args.command == "submit":
                output = asdict(
                    store.submit(
                        args.goal,
                        request_key=args.request_key or uuid.uuid4().hex,
                        workflow=args.workflow,
                        title=args.title,
                        budget=_budget(args),
                        inputs=args.input,
                    )
                )
            elif args.command == "show":
                output = store.context(args.ticket_id)
            elif args.command == "events":
                output = store.events(args.ticket_id)
            else:
                if args.command in {"pause", "cancel"}:
                    store.request_stop(args.ticket_id, cancel=args.command == "cancel")
                elif args.command == "resume":
                    store.resume(
                        args.ticket_id,
                        reconciliation=args.reconcile,
                        budget=_budget(args, store.get(args.ticket_id).budget),
                    )
                elif args.command == "accept":
                    store.accept(args.ticket_id)
                elif args.command == "input":
                    store.add_input(
                        args.ticket_id, args.text, request_key=args.request_key or uuid.uuid4().hex
                    )
                output = asdict(store.get(args.ticket_id))
            print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"Task service: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
