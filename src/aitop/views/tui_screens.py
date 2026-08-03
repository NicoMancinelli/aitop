"""Modal screens used by the aitop TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

# Match AiTopApp palette so modals don't flash the default Textual theme.
_CSS_SURFACE = """
    align: center middle;
"""
_BOX = """
        max-width: 90%;
        height: auto;
        border: thick #39d2c0;
        background: #161b22;
        padding: 1 2;
        color: #e6edf3;
"""


class ConfirmScreen(ModalScreen[bool]):
    """Yes/no confirmation for destructive actions."""

    BINDINGS = [
        Binding("y", "yes", "Yes", show=True),
        Binding("n", "no", "No", show=True),
        Binding("escape", "no", "Cancel", show=False),
        Binding("enter", "yes", "Yes", show=False),
    ]

    CSS = f"""
    ConfirmScreen {{
        {_CSS_SURFACE}
        background: rgba(14, 17, 22, 0.72);
    }}
    #confirm-box {{
        width: 60;
        {_BOX}
        border: thick #d29922;
    }}
    #confirm-title {{
        text-style: bold;
        color: #d29922;
        margin-bottom: 1;
    }}
    #confirm-body {{
        margin-bottom: 1;
    }}
    #confirm-hint {{
        color: #8b949e;
    }}
    """

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self._title, id="confirm-title")
            yield Static(self._body, id="confirm-body", markup=True)
            yield Static("[bold]y[/] yes   [bold]n[/] cancel", id="confirm-hint", markup=True)
        yield Footer()

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class ModelPickerScreen(ModalScreen[str | None]):
    """Pick a disk-resident model to load into an engine."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "select", "Load", show=True),
    ]

    CSS = f"""
    ModelPickerScreen {{
        {_CSS_SURFACE}
        background: rgba(14, 17, 22, 0.72);
    }}
    #picker-box {{
        width: 72;
        {_BOX}
        padding: 1 1;
        max-height: 80%;
    }}
    #picker-title {{
        text-style: bold;
        color: #39d2c0;
        margin: 0 1 1 1;
    }}
    OptionList {{
        height: auto;
        max-height: 18;
        border: none;
        background: #161b22;
    }}
    OptionList > .option-list--option-highlighted {{
        background: #1f6feb;
    }}
    """

    def __init__(self, engine_name: str, options: list[tuple[str, str]]) -> None:
        """`options` is a list of (model_id, label)."""
        super().__init__()
        self._engine_name = engine_name
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-box"):
            yield Label(f"Load model into {self._engine_name}", id="picker-title")
            yield OptionList(
                *[Option(label, id=model_id) for model_id, label in self._options],
                id="picker-list",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#picker-list", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        option_list = self.query_one("#picker-list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            self.dismiss(None)
            return
        option = option_list.get_option_at_index(highlighted)
        self.dismiss(str(option.id) if option.id is not None else None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id) if event.option.id is not None else None)


class FilterScreen(ModalScreen[str | None]):
    """Type a substring filter for the catalog pane. Empty string clears."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "apply", "Apply", show=True),
    ]

    CSS = f"""
    FilterScreen {{
        {_CSS_SURFACE}
        background: rgba(14, 17, 22, 0.72);
    }}
    #filter-box {{
        width: 56;
        {_BOX}
    }}
    #filter-title {{
        text-style: bold;
        color: #39d2c0;
        margin-bottom: 1;
    }}
    Input {{
        background: #0e1116;
        border: tall #30363d;
    }}
    Input:focus {{
        border: tall #39d2c0;
    }}
    """

    def __init__(self, current: str = "") -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="filter-box"):
            yield Label("Filter catalog", id="filter-title")
            yield Input(
                value=self._current,
                placeholder="name, quant, runtime…  (empty clears)",
                id="filter-input",
            )
            yield Static(
                "[dim]enter apply · esc cancel · clear field to show all[/]",
                markup=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply(self) -> None:
        value = self.query_one("#filter-input", Input).value.strip()
        self.dismiss(value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())


class PullScreen(ModalScreen[str | None]):
    """Ask for a model id/tag to pull into the selected engine."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "apply", "Pull", show=True),
    ]

    CSS = f"""
    PullScreen {{
        {_CSS_SURFACE}
        background: rgba(14, 17, 22, 0.72);
    }}
    #pull-box {{
        width: 56;
        {_BOX}
    }}
    #pull-title {{
        text-style: bold;
        color: #39d2c0;
        margin-bottom: 1;
    }}
    Input {{
        background: #0e1116;
        border: tall #30363d;
    }}
    Input:focus {{
        border: tall #39d2c0;
    }}
    """

    def __init__(self, engine_name: str) -> None:
        super().__init__()
        self._engine_name = engine_name

    def compose(self) -> ComposeResult:
        with Vertical(id="pull-box"):
            yield Label(f"Pull model into {self._engine_name}", id="pull-title")
            yield Input(placeholder="llama3.2:3b  /  mlx-community/…", id="pull-input")
            yield Static(
                "[dim]enter pull · esc cancel[/]",
                markup=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#pull-input", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_apply(self) -> None:
        value = self.query_one("#pull-input", Input).value.strip()
        self.dismiss(value or None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)


class HelpScreen(ModalScreen[None]):
    """Keybinding cheat sheet."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=True),
        Binding("q", "close", "Close", show=False),
        Binding("question_mark", "close", "Close", show=False),
    ]

    CSS = f"""
    HelpScreen {{
        {_CSS_SURFACE}
        background: rgba(14, 17, 22, 0.72);
    }}
    #help-box {{
        width: 68;
        {_BOX}
    }}
    #help-title {{
        text-style: bold;
        color: #39d2c0;
        margin-bottom: 1;
    }}
    """

    HELP = """\
[bold #39d2c0]Navigation[/]
  [bold]tab[/] / [bold]shift-tab[/]   cycle Engines · Catalog · Loaded · log
  [bold]1[/] [bold]2[/] [bold]3[/]              jump to Engines / Catalog / Loaded

[bold #39d2c0]Lifecycle[/]
  [bold]s[/]  start selected engine     [bold]e[/]  restart (confirm)
  [bold]x[/]  stop selected engine      [bold]l[/]  load (picker, or catalog row)
  [bold]u[/]  unload resident model     [bold]d[/]  delete catalog model
  [bold]p[/]  pull model into engine

[bold #39d2c0]View[/]
  [bold]space[/]  pause / resume (status shows age while paused)
  [bold]a[/]      toggle offline engines
  [bold]/[/]      filter catalog by name / quant / runtime
  [bold]o[/]      cycle catalog sort (name → size → state)
  [bold]r[/]      force refresh now
  [bold]j[/]      dump snapshot JSON snippet to log
  [bold]?[/]      this help
  [bold]q[/]      quit

Selection details (pid, scope, errors, format) appear in the strip
above the log as you move the cursor.
"""

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Label("aitop tui", id="help-title")
            yield Static(self.HELP, markup=True)
        yield Footer()

    def action_close(self) -> None:
        self.dismiss(None)
