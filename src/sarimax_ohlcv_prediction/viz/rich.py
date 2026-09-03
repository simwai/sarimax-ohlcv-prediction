"""Rich console visualizations for CLI/REPL."""

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def print_data_summary(data: pd.DataFrame, title: str = "Data Summary") -> None:
    """Print a summary of OHLCV data."""
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    for col in OHLCV_COLUMNS:
        table.add_column(col.capitalize(), style="green")

    for col in OHLCV_COLUMNS:
        if col in data.columns:
            _ = data[col].describe()
            table.add_row(
                "Count",
                *[f"{data[c].count():.0f}" for c in OHLCV_COLUMNS],
            )
            table.add_row(
                "Mean",
                *[f"{data[c].mean():.2f}" for c in OHLCV_COLUMNS],
            )
            table.add_row(
                "Std",
                *[f"{data[c].std():.2f}" for c in OHLCV_COLUMNS],
            )
            table.add_row(
                "Min",
                *[f"{data[c].min():.2f}" for c in OHLCV_COLUMNS],
            )
            table.add_row(
                "Max",
                *[f"{data[c].max():.2f}" for c in OHLCV_COLUMNS],
            )
            break

    console.print(table)


def print_model_comparison(
    results: dict[str, dict],
    title: str = "Model Comparison",
) -> None:
    """Print model comparison table."""
    table = Table(title=title)
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("AIC/Loss", style="yellow")
    table.add_column("Params", style="magenta")

    for name, info in results.items():
        table.add_row(
            name,
            info.get("status", "N/A"),
            str(info.get("score", "N/A")),
            str(info.get("params", "N/A")),
        )

    console.print(table)


def print_predictions_table(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame | None = None,
    title: str = "Predictions",
) -> None:
    """Print predictions in a table."""
    table = Table(title=title)
    table.add_column("Period", style="cyan")
    for col in OHLCV_COLUMNS:
        table.add_column(col.capitalize(), style="green")

    if actuals is not None:
        for col in OHLCV_COLUMNS:
            table.add_column(f"Actual {col.capitalize()}", style="yellow")

    for i in range(min(len(predictions), 20)):
        row = [str(i + 1)]
        for col in OHLCV_COLUMNS:
            row.append(f"{predictions[col].iloc[i]:.2f}")
        if actuals is not None and i < len(actuals):
            for col in OHLCV_COLUMNS:
                row.append(f"{actuals[col].iloc[i]:.2f}")
        table.add_row(*row)

    console.print(table)


class BacktestResults:
    """Container for backtest results."""

    def __init__(
        self,
        total_return: float,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        total_trades: int,
    ) -> None:
        self.total_return = total_return
        self.sharpe = sharpe
        self.max_dd = max_dd
        self.win_rate = win_rate
        self.total_trades = total_trades


def print_backtest_results(
    results: BacktestResults,
    title: str = "Backtest Results",
) -> None:
    """Print backtest results in a panel."""
    content = (
        f"[bold green]Total Return:[/bold green] {results.total_return:.2%}\n"
        f"[bold cyan]Sharpe Ratio:[/bold cyan] {results.sharpe:.2f}\n"
        f"[bold red]Max Drawdown:[/bold red] {results.max_dd:.2%}\n"
        f"[bold yellow]Win Rate:[/bold yellow] {results.win_rate:.2%}\n"
        f"[bold magenta]Total Trades:[/bold magenta] {results.total_trades}"
    )
    console.print(Panel(content, title=title, border_style="blue"))


def create_live_layout() -> Layout:
    """Create a live layout for REPL."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=5),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    return layout


def status_spinner(message: str):
    """Create a status spinner context manager."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )