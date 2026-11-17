"""SQLite business state and durable inbox. All commands commit atomically.

The local OS account is the trust boundary. Workers never receive this database.
Queued tickets are durable launch intents; an assignment is the launch receipt.
Only a service holding the workspace lock may claim or recover assignments.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from frontier_agent.superagent.models import Assignment, Budget, Conflict, Fenced, Json, Ticket

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticket (
 id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
 goal TEXT NOT NULL, workflow TEXT NOT NULL, status TEXT NOT NULL,
 budget TEXT NOT NULL, seconds_used REAL NOT NULL DEFAULT 0,
 calls_used INTEGER NOT NULL DEFAULT 0, tokens_used INTEGER NOT NULL DEFAULT 0,
 revision INTEGER NOT NULL DEFAULT 1, attempts INTEGER NOT NULL DEFAULT 0,
 result TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
 created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS assignment (
 id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES ticket(id), owner TEXT NOT NULL,
 revision INTEGER NOT NULL, started_at REAL NOT NULL, lease_until REAL NOT NULL,
 deadline REAL NOT NULL, status TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_assignment ON assignment(ticket_id)
 WHERE status = 'running';
CREATE TABLE IF NOT EXISTS input (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL REFERENCES ticket(id),
 request_key TEXT NOT NULL, text TEXT NOT NULL, revision INTEGER NOT NULL,
 consumed_by TEXT, UNIQUE(ticket_id, request_key)
);
CREATE TABLE IF NOT EXISTS event (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT NOT NULL REFERENCES ticket(id),
 kind TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoint (
 ticket_id TEXT NOT NULL REFERENCES ticket(id), actor TEXT NOT NULL,
 assignment_id TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL,
 PRIMARY KEY(ticket_id, actor)
);
CREATE TABLE IF NOT EXISTS operation (
 assignment_id TEXT NOT NULL REFERENCES assignment(id), key TEXT NOT NULL,
 kind TEXT NOT NULL, payload TEXT NOT NULL, result TEXT,
 PRIMARY KEY(assignment_id, key)
);
CREATE TABLE IF NOT EXISTS receipt (
 assignment_id TEXT NOT NULL REFERENCES assignment(id), key TEXT NOT NULL,
 request TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(assignment_id, key)
);
CREATE INDEX IF NOT EXISTS events_by_ticket ON event(ticket_id, seq);
PRAGMA user_version = 1;
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Store:
    """One connection per caller, serialized writes, durable reads across processes."""

    def __init__(self, root: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.clock = clock
        self.db = sqlite3.connect(self.root / "tasks.sqlite", timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.db.close()
            raise RuntimeError(f"Unsupported task database version {version}")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _event(self, ticket_id: str, kind: str, payload: Json | None = None) -> None:
        self.db.execute(
            "INSERT INTO event(ticket_id,kind,payload,at) VALUES(?,?,?,?)",
            (ticket_id, kind, _json(payload or {}), self.clock()),
        )

    def _state(self, ticket_id: str, status: str, reason: str = "") -> None:
        self.db.execute(
            "UPDATE ticket SET status=?, reason=?, updated_at=? WHERE id=?",
            (status, reason, self.clock(), ticket_id),
        )
        self._event(ticket_id, "status", {"status": status, "reason": reason})

    def get(self, ticket_id: str) -> Ticket:
        row = self.db.execute("SELECT * FROM ticket WHERE id=?", (ticket_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown ticket {ticket_id}")
        values = dict(row)
        values.pop("request_key")
        values["budget"] = Budget(**json.loads(values["budget"]))
        return Ticket(**values)

    def list(self) -> list[Ticket]:
        return [
            self.get(row[0])
            for row in self.db.execute("SELECT id FROM ticket ORDER BY created_at DESC, id")
        ]

    def submit(
        self,
        goal: str,
        *,
        request_key: str,
        workflow: str = "stateful-react-agent",
        title: str = "",
        budget: Budget | None = None,
        inputs: list[Path] | None = None,
    ) -> Ticket:
        """A repeated identical request returns its original ticket, even after restart."""
        if not goal.strip() or not request_key.strip():
            raise ValueError("A nonempty goal and request key are required")
        if workflow not in {"stateful-react-agent", "agent_team"}:
            raise ValueError("Unsupported workflow")
        budget = budget or Budget()
        title = title.strip() or goal.strip()[:100]
        sources = [path.expanduser().absolute() for path in (inputs or [])]
        with self.transaction():
            row = self.db.execute(
                "SELECT id FROM ticket WHERE request_key=?", (request_key,)
            ).fetchone()
            if row:
                old = self.get(row[0])
                if (old.goal, old.workflow, old.title, old.budget) != (
                    goal,
                    workflow,
                    title,
                    budget,
                ):
                    raise Conflict("Request key was already used for a different task")
                original = self.events(old.id)[0]["payload"].get("inputs", [])
                if original != [str(path) for path in sources]:
                    raise Conflict("Request key was already used with different inputs")
                return old
            ticket_id = uuid.uuid4().hex
            if sources:
                from frontier_agent.superagent.inputs import snapshot_inputs

                snapshot_inputs(sources, self.root / "tasks" / ticket_id / "inputs")
            now = self.clock()
            self.db.execute(
                "INSERT INTO ticket(id,request_key,title,goal,workflow,status,budget,created_at,updated_at) "
                "VALUES(?,?,?,?,?,'queued',?,?,?)",
                (ticket_id, request_key, title, goal, workflow, _json(asdict(budget)), now, now),
            )
            self._event(
                ticket_id,
                "submitted",
                {"request_key": request_key, "inputs": [str(path) for path in sources]},
            )
        return self.get(ticket_id)

    def add_input(self, ticket_id: str, text: str, *, request_key: str) -> None:
        if not text.strip() or not request_key.strip():
            raise ValueError("A nonempty message and request key are required")
        with self.transaction():
            old = self.db.execute(
                "SELECT text FROM input WHERE ticket_id=? AND request_key=?",
                (ticket_id, request_key),
            ).fetchone()
            if old:
                if old[0] != text:
                    raise Conflict("Input request key was already used")
                return
            task = self.get(ticket_id)
            if task.status in {"completed", "cancelled", "cancelling"}:
                raise Conflict("Ended tasks cannot be woken by new input")
            revision = task.revision + 1
            self.db.execute(
                "INSERT INTO input(ticket_id,request_key,text,revision) VALUES(?,?,?,?)",
                (ticket_id, request_key, text, revision),
            )
            self.db.execute(
                "UPDATE ticket SET revision=?,updated_at=? WHERE id=?",
                (revision, self.clock(), ticket_id),
            )
            self._event(ticket_id, "input", {"revision": revision, "text": text})
            if task.status == "ready":
                self._state(ticket_id, "queued")

    def request_stop(self, ticket_id: str, *, cancel: bool = False) -> None:
        with self.transaction():
            task = self.get(ticket_id)
            if task.status in {"cancelled", "completed"}:
                return
            active = bool(
                self.db.execute(
                    "SELECT 1 FROM assignment WHERE ticket_id=? AND status='running'", (ticket_id,)
                ).fetchone()
            )
            state = (
                ("cancelling" if active else "cancelled")
                if cancel
                else ("pausing" if active else "paused")
            )
            if task.status == "cancelling" and not cancel:
                raise Conflict("Cancellation cannot be downgraded to a pause")
            self._state(ticket_id, state, "Requested by user")

    def resume(
        self, ticket_id: str, *, reconciliation: str = "", budget: Budget | None = None
    ) -> None:
        """Resume requires explicit reconciliation when a tool outcome is unknown."""
        with self.transaction():
            task = self.get(ticket_id)
            if task.status not in {"paused", "blocked", "failed"}:
                raise Conflict("Only paused, blocked or failed tasks can be resumed")
            if self.db.execute(
                "SELECT 1 FROM assignment WHERE ticket_id=? AND status='running'", (ticket_id,)
            ).fetchone():
                raise Conflict("Previous executor has not stopped")
            uncertain = self.unresolved(ticket_id)
            if uncertain and not reconciliation.strip():
                raise Conflict(
                    "Inspect uncertain operations; supply --reconcile with your findings"
                )
            if uncertain:
                self.db.execute(
                    "UPDATE operation SET result=? WHERE result IS NULL AND assignment_id IN "
                    "(SELECT id FROM assignment WHERE ticket_id=?)",
                    (_json({"human_reconciliation": reconciliation}), ticket_id),
                )
                self._event(ticket_id, "reconciled", {"findings": reconciliation})
            if budget is not None:
                self.db.execute(
                    "UPDATE ticket SET budget=? WHERE id=?", (_json(asdict(budget)), ticket_id)
                )
            task = self.get(ticket_id)
            if self._exhausted(task):
                raise Conflict("Budget exhausted; raise the total budget before resuming")
            self._state(
                ticket_id, "queued", "Resuming from recorded context; inspect existing work"
            )

    def accept(self, ticket_id: str) -> None:
        with self.transaction():
            if self.get(ticket_id).status != "ready":
                raise Conflict("Only a ready result can be accepted")
            self._state(ticket_id, "completed", "Accepted by user")

    @staticmethod
    def _exhausted(task: Ticket) -> bool:
        return (
            task.seconds_used >= task.budget.seconds
            or task.calls_used >= task.budget.model_calls
            or task.attempts >= task.budget.attempts
        )

    def claim(self, owner: str, *, lease_seconds: float = 30) -> Assignment | None:
        with self.transaction():
            for row in self.db.execute(
                "SELECT id FROM ticket WHERE status='queued' AND NOT EXISTS "
                "(SELECT 1 FROM assignment a WHERE a.ticket_id=ticket.id AND a.status='running') "
                "ORDER BY created_at,id"
            ).fetchall():
                task = self.get(row[0])
                if self._exhausted(task):
                    self._state(task.id, "paused", "Budget exhausted")
                    continue
                now = self.clock()
                assignment = Assignment(
                    uuid.uuid4().hex,
                    task.id,
                    owner,
                    task.revision,
                    now,
                    now + lease_seconds,
                    now + task.budget.seconds - task.seconds_used,
                    "running",
                )
                self.db.execute(
                    "INSERT INTO assignment VALUES(?,?,?,?,?,?,?,?)",
                    tuple(asdict(assignment).values()),
                )
                self.db.execute("UPDATE ticket SET attempts=attempts+1 WHERE id=?", (task.id,))
                self._state(task.id, "running")
                self._event(task.id, "assigned", asdict(assignment))
                return assignment
        return None

    def active(self) -> list[Assignment]:
        return [
            Assignment(**dict(row))
            for row in self.db.execute(
                "SELECT * FROM assignment WHERE status='running' ORDER BY started_at"
            )
        ]

    def _owned(self, assignment_id: str, owner: str) -> tuple[Assignment, Ticket]:
        row = self.db.execute("SELECT * FROM assignment WHERE id=?", (assignment_id,)).fetchone()
        if not row:
            raise Fenced("Unknown assignment")
        assignment = Assignment(**dict(row))
        task = self.get(assignment.ticket_id)
        if (
            assignment.owner != owner
            or assignment.status != "running"
            or assignment.lease_until <= self.clock()
            or assignment.deadline <= self.clock()
            or task.status != "running"
        ):
            raise Fenced("Executor lease, budget or task authorization ended")
        return assignment, task

    def renew(self, assignment_id: str, owner: str, *, lease_seconds: float = 30) -> None:
        with self.transaction():
            self._owned(assignment_id, owner)
            self.db.execute(
                "UPDATE assignment SET lease_until=? WHERE id=?",
                (self.clock() + lease_seconds, assignment_id),
            )

    def exchange(self, assignment_id: str, owner: str, key: str, kind: str, payload: Json) -> Json:
        """Durable worker RPC: acknowledge only after commit, replay receipts by key."""
        request = _json({"kind": kind, "payload": payload})
        with self.transaction():
            assignment, task = self._owned(assignment_id, owner)
            prior = self.db.execute(
                "SELECT request,response FROM receipt WHERE assignment_id=? AND key=?",
                (assignment_id, key),
            ).fetchone()
            if prior:
                if prior[0] != request:
                    raise Conflict("RPC key reused with different data")
                return json.loads(prior[1])
            response: Json = {}
            if kind == "heartbeat":
                return {}
            if kind == "model_call":
                if task.calls_used >= task.budget.model_calls:
                    raise Fenced("Model request budget exhausted")
                self.db.execute(
                    "UPDATE ticket SET calls_used=calls_used+1,updated_at=? WHERE id=?",
                    (self.clock(), task.id),
                )
            elif kind == "usage":
                tokens = int(payload.get("tokens", 0))
                if tokens < 0:
                    raise ValueError("Negative token usage")
                self.db.execute(
                    "UPDATE ticket SET tokens_used=tokens_used+? WHERE id=?", (tokens, task.id)
                )
            elif kind == "checkpoint":
                self.db.execute(
                    "INSERT INTO checkpoint VALUES(?,?,?,?,?) ON CONFLICT(ticket_id,actor) "
                    "DO UPDATE SET assignment_id=excluded.assignment_id,payload=excluded.payload,at=excluded.at",
                    (task.id, str(payload["actor"]), assignment_id, _json(payload), self.clock()),
                )
            elif kind == "inputs":
                rows = self.db.execute(
                    "SELECT seq,text,revision FROM input WHERE ticket_id=? "
                    "AND consumed_by IS NULL ORDER BY seq",
                    (task.id,),
                ).fetchall()
                response = {"inputs": [dict(row) for row in rows], "revision": task.revision}
                # Reading is not consuming. Consumption is coupled to the next checkpoint.
            elif kind == "consume":
                for seq in payload["seqs"]:
                    self.db.execute(
                        "UPDATE input SET consumed_by=? WHERE seq=? AND ticket_id=? "
                        "AND consumed_by IS NULL",
                        (assignment_id, int(seq), task.id),
                    )
            elif kind == "report":
                self._event(task.id, "report", {**payload, "assignment_id": assignment_id})
            elif kind in {"tool_start", "job_start"}:
                self.db.execute(
                    "INSERT INTO operation VALUES(?,?,?,?,NULL)",
                    (assignment_id, str(payload["operation_id"]), kind, _json(payload)),
                )
            elif kind in {"tool_end", "job_end"}:
                updated = self.db.execute(
                    "UPDATE operation SET result=? WHERE assignment_id=? AND key=?",
                    (_json(payload), assignment_id, str(payload["operation_id"])),
                )
                if updated.rowcount != 1:
                    raise Conflict("Operation completion has no recorded start")
            else:
                raise ValueError(f"Unknown worker message {kind}")
            if kind not in {"inputs", "usage", "checkpoint", "consume", "report"}:
                self._event(task.id, kind, payload)
            self.db.execute(
                "INSERT INTO receipt VALUES(?,?,?,?)",
                (assignment.id, key, request, _json(response)),
            )
            return response

    def context(self, ticket_id: str) -> Json:
        """Return authoritative goals, all follow-ups and per-actor recovery context."""
        return {
            "ticket": asdict(self.get(ticket_id)),
            "inputs": [
                dict(row)
                for row in self.db.execute(
                    "SELECT seq,text,revision,consumed_by FROM input WHERE ticket_id=? ORDER BY seq",
                    (ticket_id,),
                )
            ],
            "checkpoints": [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT payload FROM checkpoint WHERE ticket_id=? ORDER BY at", (ticket_id,)
                )
            ],
            "operations": self.operations(ticket_id),
            "reconciliations": [
                event["payload"]
                for event in self.events(ticket_id)
                if event["kind"] == "reconciled"
            ],
        }

    def operations(self, ticket_id: str) -> list[Json]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT o.* FROM operation o JOIN assignment a ON a.id=o.assignment_id "
                "WHERE a.ticket_id=?",
                (ticket_id,),
            )
        ]

    def unresolved(self, ticket_id: str) -> list[Json]:
        return [row for row in self.operations(ticket_id) if row["result"] is None]

    def events(self, ticket_id: str, *, after: int = 0) -> list[Json]:
        self.get(ticket_id)
        return [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in self.db.execute(
                "SELECT * FROM event WHERE ticket_id=? AND seq>? ORDER BY seq", (ticket_id, after)
            )
        ]

    def finish(
        self,
        assignment_id: str,
        owner: str,
        *,
        result: str = "",
        complete: bool = False,
        reason: str = "",
        recovered: bool = False,
        revision: int | None = None,
    ) -> None:
        """Called only after the execution environment is confirmed stopped.

        Recovery deliberately does not trust an expired lease as proof a process died.
        The service must stop its named container before invoking this method.
        """
        with self.transaction():
            row = self.db.execute(
                "SELECT * FROM assignment WHERE id=?", (assignment_id,)
            ).fetchone()
            if not row or row["owner"] != owner or row["status"] != "running":
                raise Fenced("Stale completion")
            assignment = Assignment(**dict(row))
            task = self.get(assignment.ticket_id)
            report_at = None
            # A report was accepted under a valid lease. Slow container cleanup
            # must not retroactively invalidate it or charge its time as inference.
            for event in self.events(task.id):
                if (
                    event["kind"] == "report"
                    and event["payload"].get("assignment_id") == assignment_id
                ):
                    saved = event["payload"]
                    result, complete = str(saved.get("result", "")), bool(saved.get("complete"))
                    revision, reason = saved.get("revision"), str(saved.get("reason", ""))
                    report_at = float(event["at"])
            elapsed = max(
                0, min(report_at or self.clock(), assignment.deadline) - assignment.started_at
            )
            self.db.execute("UPDATE assignment SET status='ended' WHERE id=?", (assignment_id,))
            self.db.execute(
                "UPDATE ticket SET seconds_used=seconds_used+? WHERE id=?", (elapsed, task.id)
            )
            task = self.get(task.id)
            expired = report_at is None and self.clock() >= min(
                assignment.lease_until, assignment.deadline
            )
            if task.status == "cancelling":
                status = "cancelled"
            elif self.unresolved(task.id):
                status, reason = "blocked", "Operation outcome uncertain; reconcile before retry"
            elif task.status == "pausing":
                status = "paused"
                reason = reason or "Paused or budget exhausted"
            elif complete and not expired:
                # Inputs arriving while the final answer was generated cannot disappear.
                unread = self.db.execute(
                    "SELECT 1 FROM input WHERE ticket_id=? AND consumed_by IS NULL", (task.id,)
                ).fetchone()
                status = "queued" if unread or revision != task.revision else "ready"
            elif self._exhausted(task):
                status, reason = "paused", reason or "Budget exhausted"
            elif recovered and report_at is None:
                status = "queued"
            else:
                status = "failed"
            if result:
                self.db.execute("UPDATE ticket SET result=? WHERE id=?", (result, task.id))
            self._event(
                task.id, "attempt_ended", {"assignment_id": assignment_id, "seconds": elapsed}
            )
            self._state(task.id, status, reason)
