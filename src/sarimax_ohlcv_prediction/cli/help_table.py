"""Help table UI component for CLI/REPL."""

from typing import Any

from rich.table import Table

from ..viz.theme import ICONS, get_console

console = get_console()


class HelpTable(Table):
    """Aligned help table: cmd | args | description."""

    def __init__(self, **kwargs: Any) -> None:  # pyrefly: ignore -- Rich passthrough kwargs
        super().__init__(
            show_header=False,
            box=None,
            padding=(0, 1),
            collapse_padding=True,
            show_edge=False,
            **kwargs,
        )
        self.add_column("cmd", style="brand", no_wrap=True, width=14)
        self.add_column("args", style="ui.text_dim", no_wrap=False, min_width=32, max_width=45)
        self.add_column("desc", style="ui.text", no_wrap=False)

    def add_section(self, title: str) -> None:
        """Add a section header."""
        self.add_row("", "", "")
        self.add_row(f"[bold ui.text]{ICONS['bullet']} {title.upper()}[/]", "", "")
        self.add_row("", "", "")

    def add_command(self, cmd: str, args: str, desc: str) -> None:
        """Add a command row."""
        self.add_row(
            f"[brand]{cmd}[/]" if cmd else "",
            f"[ui.text_dim]{args}[/]" if args else "",
            desc,
        )
