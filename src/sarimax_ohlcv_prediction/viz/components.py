"""Reusable styled components for REPL and CLI."""

import time
from typing import Any, Optional

import pandas as pd
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .theme import ICONS, PALETTE, REPL_THEME, get_console

console = get_console()

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
DATA_STYLES = {
    "open": "data.open",
    "high": "data.high",
    "low": "data.low",
    "close": "data.close",
    "volume": "data.volume",
}


class DataTable(Table):
    """Compact data table with semantic column colors and zebra striping."""

    def __init__(self, title: str = "", **kwargs: Any) -> None:
        super().__init__(
            title=title,
            title_style="panel.title",
            box=MINIMAL,
            padding=(0, 1),
            collapse_padding=True,
            show_header=True,
            header_style="table.header",
            row_styles=["table.row_even", "table.row_odd"],
            **kwargs,
        )

    def add_data_column(self, name: str, **kwargs: Any) -> None:
        """Add an OHLCV column with semantic styling."""
        style = DATA_STYLES.get(name.lower(), "ui.text")
        self.add_column(name.capitalize(), style=style, justify="right", **kwargs)

    def add_metric_row(self, metric: str, values: list[float | int]) -> None:
        """Add a row with metric name and formatted values."""
        row = [f"[ui.text_dim]{metric}[/]"]
        for i, v in enumerate(values):
            col = OHLCV_COLUMNS[i] if i < len(OHLCV_COLUMNS) else ""
            style = DATA_STYLES.get(col, "ui.text")
            if isinstance(v, float):
                row.append(f"[{style}]{v:.2f}[/]")
            else:
                row.append(f"[{style}]{v:,}[/]")
        self.add_row(*row)


class StatusPanel(Panel):
    """Status panel with icon and consistent styling."""

    def __init__(
        self,
        message: str,
        status: str = "info",
        title: str = "",
        **kwargs: Any,
    ) -> None:
        icon = ICONS.get(status, ICONS["info"])
        style_map = {
            "success": "success",
            "error": "error",
            "warning": "warning",
            "info": "info",
            "running": "status.running",
        }
        content = f"[{style_map.get(status, 'info')}]{icon}[/]  {message}"
        super().__init__(
            content,
            title=title,
            border_style="ui.border",
            padding=(0, 1),
            **kwargs,
        )


class MetricCard(Panel):
    """Compact KPI metric card."""

    def __init__(
        self,
        label: str,
        value: str,
        trend: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        trend_icon = ""
        trend_style = ""
        if trend == "up":
            trend_icon = f" [{PALETTE['success']}]{ICONS['chart_up']}[/]"
            trend_style = PALETTE["success"]
        elif trend == "down":
            trend_icon = f" [{PALETTE['error']}]{ICONS['chart_down']}[/]"
            trend_style = PALETTE["error"]

        content = (
            f"[ui.text_dim]{label}[/]\n"
            f"[ui.text]{value}[/]{trend_icon}"
        )
        super().__init__(
            content,
            border_style="ui.border",
            padding=(1, 2),
            **kwargs,
        )


class ProgressBar(Progress):
    """Branded progress bar with spinner and elapsed time."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            SpinnerColumn(spinner_name="dots", style="status.running"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="brand", finished_style="success"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            **kwargs,
        )


class HelpTable(Table):
    """Aligned help table: cmd | args | description."""

    def __init__(self, **kwargs: Any) -> None:
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


class LiveLayout:
    """Dual-area live layout: status (top) + progress (bottom)."""

    def __init__(self, header_text: str = "") -> None:
        self.header_text = Text(header_text, style="ui.text_dim")
        self.progress = ProgressBar()
        self.task_ids: dict[str, TaskID] = {}

    def add_tasks(self, task_names: list[str], totals: dict[str, int]) -> None:
        """Add progress tasks for each model."""
        for name in task_names:
            total = totals.get(name, 1)
            self.task_ids[name] = self.progress.add_task(
                f"[dim]{name} queued...[/]", total=total, start=False
            )

    def get_group(self) -> Group:
        """Get renderable group for Live."""
        return Group(self.header_text, self.progress)

    def start_task(self, name: str, description: str = "") -> None:
        """Start a task."""
        if name in self.task_ids:
            self.progress.start_task(self.task_ids[name])
            if description:
                self.progress.update(self.task_ids[name], description=description)

    def update_task(self, name: str, **kwargs: Any) -> None:
        """Update a task."""
        if name in self.task_ids:
            self.progress.update(self.task_ids[name], **kwargs)

    def complete_task(self, name: str, description: str = "") -> None:
        """Mark task complete."""
        if name in self.task_ids:
            total = self.progress.tasks[self.task_ids[name]].total or 1
            self.progress.update(
                self.task_ids[name],
                completed=total,
                description=f"[success]{description or f'{name} done'}[/]",
            )

    @property
    def live(self) -> Live:
        """Get Live context manager."""
        return Live(
            self.get_group(),
            console=console,
            refresh_per_second=12,
            transient=True,
        )


def print_data_summary(data: pd.DataFrame, title: str = "Data Summary") -> None:
    """Print OHLCV data summary using DataTable."""
    table = DataTable(title=title)
    table.add_column("Metric", style="brand", no_wrap=True, width=10)
    for col in OHLCV_COLUMNS:
        table.add_data_column(col)

    for col in OHLCV_COLUMNS:
        if col not in data.columns:
            continue
        desc = data[col].describe()
        table.add_metric_row("Count", [desc["count"]] * 5)
        table.add_metric_row("Mean", [desc["mean"]] * 5)
        table.add_metric_row("Std", [desc["std"]] * 5)
        table.add_metric_row("Min", [desc["min"]] * 5)
        table.add_metric_row("Max", [desc["max"]] * 5)
        break

    console.print(table)


def print_model_comparison(
    results: dict[str, dict],
    title: str = "Model Comparison",
) -> None:
    """Print model comparison table."""
    table = Table(
        title=title,
        title_style="panel.title",
        box=MINIMAL,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    table.add_column("Model", style="brand", no_wrap=True)
    table.add_column("Status", style="ui.text", justify="center")
    table.add_column("RMSE", style="warning", justify="right")
    table.add_column("MAE", style="warning", justify="right")
    table.add_column("MAPE", style="data.volume", justify="right")
    table.add_column("Info", style="ui.text_dim", max_width=50)
    table.add_column("Time", style="info", justify="right")

    for name, info in results.items():
        status = info.get("status", "N/A")
        status_style = "success" if "Success" in status else "error" if "Failed" in status else "warning"
        table.add_row(
            name,
            f"[{status_style}]{status}[/]",
            str(info.get("rmse", "N/A")),
            str(info.get("mae", "N/A")),
            str(info.get("mape", "N/A")),
            str(info.get("info", "N/A"))[:50],
            str(info.get("elapsed", "N/A")),
        )

    console.print(table)


def print_predictions_table(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
    title: str = "Predictions",
) -> None:
    """Print predictions in a table."""
    table = Table(
        title=title,
        title_style="panel.title",
        box=MINIMAL,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    table.add_column("Period", style="brand", justify="right", width=6)
    for col in OHLCV_COLUMNS:
        table.add_data_column(col)

    if actuals is not None:
        for col in OHLCV_COLUMNS:
            table.add_column(f"Actual {col.capitalize()}", style="warning", justify="right")

    for i in range(min(len(predictions), 20)):
        row = [str(i + 1)]
        for col in OHLCV_COLUMNS:
            row.append(f"{predictions[col].iloc[i]:.2f}")
        if actuals is not None and i < len(actuals):
            for col in OHLCV_COLUMNS:
                row.append(f"{actuals[col].iloc[i]:.2f}")
        table.add_row(*row)

    console.print(table)


def print_backtest_results(
    total_return: float,
    sharpe: float,
    max_dd: float,
    win_rate: float,
    total_trades: int,
    title: str = "Backtest Results",
) -> None:
    """Print backtest results as metric cards."""
    from rich.columns import Columns

    cards = [
        MetricCard("Total Return", f"{total_return:.2%}", "up" if total_return > 0 else "down"),
        MetricCard("Sharpe Ratio", f"{sharpe:.2f}", "up" if sharpe > 1 else "down"),
        MetricCard("Max Drawdown", f"{max_dd:.2%}", "down"),
        MetricCard("Win Rate", f"{win_rate:.2%}", "up" if win_rate > 0.5 else "down"),
        MetricCard("Total Trades", f"{total_trades:,}"),
    ]
    console.print(Columns(cards, equal=True, expand=True))


def status_spinner(message: str) -> Progress:
    """Create a status spinner context manager."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="status.running"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )


def print_inline_status(
    data_rows: int | str,
    model_name: str | None,
    model_path: str | None,
    pred_rows: int | str,
) -> None:
    """Print compact inline status line."""
    data_str = str(data_rows) if data_rows else "—"
    model_str = model_name or "—"
    path_str = model_path or "—"
    pred_str = str(pred_rows) if pred_rows else "—"

    console.print(
        f"[ui.text_dim]data[/] {data_str}  "
        f"[ui.text_dim]· model[/] {model_str}  "
        f"[ui.text_dim]· path[/] {path_str}  "
        f"[ui.text_dim]· preds[/] {pred_str}"
    )


def print_verbose_status(
    data_rows: int | str,
    model_name: str | None,
    model_path: str | None,
    pred_rows: int | str,
) -> None:
    """Print verbose status table."""
    table = Table(
        title="REPL Status",
        box=MINIMAL,
        show_header=False,
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("Item", style="brand", no_wrap=True, width=14)
    table.add_column("Value", style="ui.text")

    data_str = str(data_rows) if data_rows else "—"
    model_str = model_name or "—"
    path_str = model_path or "—"
    pred_str = str(pred_rows) if pred_rows else "—"

    table.add_row("Data", data_str)
    table.add_row("Model", model_str)
    table.add_row("Model Path", path_str)
    table.add_row("Predictions", pred_str)

    console.print(table)