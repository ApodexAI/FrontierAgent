"""A detachable Textual client for durable tasks; closing it never cancels work."""

from __future__ import annotations

import uuid
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from frontier_agent.superagent.store import Store


class TaskDashboard(App[None]):
    TITLE = "FrontierAgent Tasks"
    BINDINGS: ClassVar = [("q", "quit", "Close dashboard")]
    CSS = """
    DataTable { height: 40%; }
    #details { height: 1fr; overflow-y: auto; padding: 1 2; }
    #actions { height: 3; }
    Button { margin-right: 1; }
    """

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store
        self.selected: str | None = None
        self.snapshot: list[tuple[str, str, float]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="tasks", cursor_type="row")
        yield Static(
            "Select a task. Closing this dashboard leaves the service running.",
            id="details",
            markup=False,
        )
        yield Input(placeholder="New task goal, or a follow-up for the selected task", id="message")
        with Horizontal(id="actions"):
            yield Button("New task", id="submit", variant="primary")
            yield Button("Send input", id="input")
            yield Button("Pause", id="pause")
            yield Button("Resume", id="resume")
            yield Button("Cancel", id="cancel", variant="error")
            yield Button("Accept result", id="accept", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns("Task", "State", "Calls", "Seconds", "Title")
        self.refresh_tasks()
        self.set_interval(1, self.refresh_tasks)

    def refresh_tasks(self) -> None:
        tasks = self.store.list()
        snapshot = [(task.id, task.status, task.updated_at) for task in tasks]
        if snapshot != self.snapshot:
            self.snapshot = snapshot
            table = self.query_one(DataTable)
            table.clear()
            for task in tasks:
                table.add_row(
                    task.id[:8],
                    task.status,
                    str(task.calls_used),
                    str(round(task.seconds_used)),
                    task.title,
                    key=task.id,
                )
            if self.selected:
                for index, task in enumerate(tasks):
                    if task.id == self.selected:
                        table.move_cursor(row=index)
                        break
        self.show_selected()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.selected = str(event.row_key.value)
        self.show_selected()

    def show_selected(self) -> None:
        if self.selected:
            task = self.store.get(self.selected)
            self.query_one("#details", Static).update(
                f"{task.title}\n{task.id}\n\n{task.goal}\n\n"
                f"State: {task.status}  {task.reason}\n"
                f"Model loop calls: {task.calls_used}/{task.budget.model_calls}  "
                f"Tokens observed: {task.tokens_used}\n"
                f"Artifacts: {self.store.root / 'tasks' / task.id / 'outputs'}\n\n{task.result}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id
        field = self.query_one(Input)
        try:
            if action == "submit":
                self.selected = self.store.submit(field.value, request_key=uuid.uuid4().hex).id
                field.value = ""
            elif self.selected:
                if action == "input":
                    self.store.add_input(self.selected, field.value, request_key=uuid.uuid4().hex)
                    field.value = ""
                elif action in {"pause", "cancel"}:
                    self.store.request_stop(self.selected, cancel=action == "cancel")
                elif action == "resume":
                    self.store.resume(self.selected)
                elif action == "accept":
                    self.store.accept(self.selected)
            self.refresh_tasks()
        except (ValueError, KeyError) as exc:
            self.notify(str(exc), severity="error", timeout=8)
