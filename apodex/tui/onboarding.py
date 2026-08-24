"""Standalone Textual first-run setup shown before session construction."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, OptionList, Static
from textual.widgets.option_list import Option

from apodex.onboarding import (
    DEPLOYMENT_CHOICES,
    DEPLOYMENT_LABELS,
    OnboardingProbe,
    choice_guidance,
    format_probe,
)
from apodex.tui.themes import TUI_THEME_NAMES, register_themes


class OnboardingApp(App[str | None]):
    """Full-screen, keyboard and mouse friendly deployment selector."""

    CSS = """
    Screen { align: center middle; background: $background; color: $text; }
    #onboarding-box {
        width: 88%; max-width: 104; height: auto; max-height: 92%; padding: 1 2;
        border: round $border; background: $surface; color: $text;
    }
    #onboarding-title { height: auto; color: $primary; text-style: bold; }
    #onboarding-intro { height: auto; margin: 1 0; color: $text-muted; }
    #onboarding-probe {
        height: auto; padding: 1 2; margin-bottom: 1;
        border-left: solid $primary; background: $panel; color: $text-muted;
    }
    #onboarding-options { height: 8; margin-bottom: 1; }
    #onboarding-guidance {
        height: auto; min-height: 4; padding: 1 2;
        background: $panel; color: $text;
    }
    #onboarding-actions { height: 3; align-horizontal: center; margin-top: 1; }
    #onboarding-actions Button { min-width: 20; margin: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Quit setup", priority=True),
        Binding("ctrl+c", "cancel", "Quit setup", priority=True),
    ]

    def __init__(self, probe: OnboardingProbe, *, theme: str = "catppuccin") -> None:
        super().__init__()
        self.probe = probe
        self.selected_choice = probe.suggested_choice
        register_themes(self)
        self._ui_theme = theme if theme in TUI_THEME_NAMES else "catppuccin"

    def compose(self) -> ComposeResult:
        with Vertical(id="onboarding-box"):
            yield Static("Welcome to FrontierAgent", id="onboarding-title")
            yield Static(
                "First choose how this installation should reach an Apodex model. "
                "The choice is remembered and can be changed later with --setup.",
                id="onboarding-intro",
            )
            yield Static(format_probe(self.probe), id="onboarding-probe")
            yield OptionList(
                *(
                    Option(self._choice_label(choice), id=choice)
                    for choice in DEPLOYMENT_CHOICES
                ),
                id="onboarding-options",
            )
            yield Static(
                choice_guidance(self.selected_choice, self.probe),
                id="onboarding-guidance",
            )
            with Horizontal(id="onboarding-actions"):
                yield Button("Save & continue", id="onboarding-save", variant="primary")
                yield Button("Quit", id="onboarding-quit")
        yield Footer()

    def _choice_label(self, choice: str) -> str:
        suffix = " · recommended" if choice == self.probe.suggested_choice else ""
        if choice.startswith("nvidia_"):
            state = "ready" if self.probe.nvidia_container_ready else "setup needed"
            suffix += f" · {state}"
        elif self.probe.active_config_ok:
            suffix += " · .env detected"
        return DEPLOYMENT_LABELS[choice] + suffix

    def on_mount(self) -> None:
        self.theme = self._ui_theme
        options = self.query_one("#onboarding-options", OptionList)
        options.highlighted = DEPLOYMENT_CHOICES.index(self.selected_choice)
        options.focus()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted,
    ) -> None:
        if event.option.id in DEPLOYMENT_CHOICES:
            self.selected_choice = str(event.option.id)
            self.query_one("#onboarding-guidance", Static).update(
                choice_guidance(self.selected_choice, self.probe)
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id in DEPLOYMENT_CHOICES:
            self.selected_choice = str(event.option.id)
            self.query_one("#onboarding-save", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "onboarding-save":
            self.exit(self.selected_choice)
        elif event.button.id == "onboarding-quit":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.exit(None)


__all__ = ["OnboardingApp"]
