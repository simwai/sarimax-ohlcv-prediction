"""Main CLI application with Typer and Rich REPL."""

import logging
from typing import Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from ..backtest import print_backtest_result, run_backtest
from ..config import SETTINGS
from ..data.fetcher import fetch_with_retry
from ..models import MODEL_REGISTRY, create_model
from ..viz.rich import print_data_summary, print_predictions_table

Mode = Literal["current", "historical"]


def _as_mode(mode: str) -> Mode:
    """Validate and cast mode string to Literal."""
    if mode not in ("current", "historical"):
        raise typer.BadParameter(f"Mode must be 'current' or 'historical', got: {mode}")  # noqa: TRY003
    return cast(Mode, mode)

app = typer.Typer(
    name="sarimax-ohlcv",
    help="Bitcoin price prediction with SARIMAX, Prophet, and LSTM",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

cli_console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Bitcoin OHLCV Prediction CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@app.command()
def fetch(
    mode: str = typer.Option("current", help="Mode: current or historical"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days for historical"),
    output: str | None = typer.Option(None, help="Save to CSV file"),
) -> None:
    """Fetch OHLCV data from Binance."""
    with cli_console.status(f"[bold green]Fetching {mode} data..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        cli_console.print("[red]Failed to fetch data[/red]")
        raise typer.Exit(1)

    print_data_summary(data, f"Fetched Data ({mode}, {lookback}d)")

    if output:
        data.to_csv(output, index=False)
        cli_console.print(f"[green]Saved to {output}[/green]")


@app.command()
def train(
    model: str = typer.Option("SARIMAX", help=f"Model: {', '.join(MODEL_REGISTRY.keys())}"),
    mode: str = typer.Option("historical", help="Data mode"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days"),
    iterations: int = typer.Option(SETTINGS.default_iterations, help="Optimization iterations"),
    save_path: str | None = typer.Option(None, help="Path to save model"),
) -> None:
    """Train a model on historical data."""
    # Fetch data
    with cli_console.status("[bold green]Fetching training data..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        cli_console.print("[red]Failed to fetch data[/red]")
        raise typer.Exit(1)

    # Create and train model
    with cli_console.status(f"[bold green]Training {model}..."):
        model_instance = create_model(model)
        model_instance.fit(data, iterations=iterations)

    cli_console.print(f"[green]{model} trained successfully[/green]")

    if save_path:
        model_instance.save(save_path)
        cli_console.print(f"[green]Model saved to {save_path}[/green]")


@app.command()
def predict(
    model: str = typer.Option("SARIMAX", help=f"Model: {', '.join(MODEL_REGISTRY.keys())}"),
    model_path: str | None = typer.Option(None, help="Path to load model from"),
    periods: int = typer.Option(SETTINGS.default_prediction_periods, help="Prediction periods"),
    mode: str = typer.Option("current", help="Data mode for context"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days"),
    save_csv: str | None = typer.Option(None, help="Save predictions to CSV"),
) -> None:
    """Make predictions using a trained model."""
    # Load or train model
    if model_path:
        with cli_console.status(f"[bold green]Loading {model} from {model_path}..."):
            model_class = MODEL_REGISTRY[model]
            model_instance = model_class.load(model_path)
    else:
        # Train on the fly
        with cli_console.status("[bold green]Fetching data and training..."):
            data = fetch_with_retry(_as_mode(mode), lookback)
            if data.empty:
                cli_console.print("[red]Failed to fetch data[/red]")
                raise typer.Exit(1)
            model_instance = create_model(model)
            model_instance.fit(data)

    # Make predictions
    with cli_console.status("[bold green]Generating predictions..."):
        if model == "LSTM":
            # LSTM needs recent data context
            data = fetch_with_retry(_as_mode(mode), lookback)
            predictions = model_instance.predict_with_context(data, periods)
        else:
            predictions = model_instance.predict(periods)

    print_predictions_table(predictions, title=f"{model} Predictions ({periods} periods)")

    if save_csv:
        predictions.to_csv(save_csv, index=False)
        cli_console.print(f"[green]Predictions saved to {save_csv}[/green]")


@app.command()
def backtest(
    model: str = typer.Option("SARIMAX", help=f"Model: {', '.join(MODEL_REGISTRY.keys())}"),
    model_path: str | None = typer.Option(None, help="Path to load model"),
    strategy: str = typer.Option("exit_after_n", help="Strategy: exit_after_n, exit_on_signal"),
    lookback: int = typer.Option(SETTINGS.backtest_default_lookback, help="Bars to backtest"),
    exit_bars: int = typer.Option(SETTINGS.backtest_default_exit_bars, help="Exit after N bars"),
    mode: str = typer.Option("historical", help="Data mode"),
    data_lookback: int = typer.Option(500, help="Data lookback for training"),
) -> None:
    """Run backtest on historical data."""
    # Load or train model
    if model_path:
        with cli_console.status(f"[bold green]Loading {model}..."):
            model_class = MODEL_REGISTRY[model]
            model_instance = model_class.load(model_path)
    else:
        with cli_console.status("[bold green]Fetching data and training..."):
            data = fetch_with_retry(_as_mode(mode), data_lookback)
            if data.empty:
                cli_console.print("[red]Failed to fetch data[/red]")
                raise typer.Exit(1)
            model_instance = create_model(model)
            model_instance.fit(data)

    # Fetch test data
    with cli_console.status("[bold green]Fetching test data..."):
        test_data = fetch_with_retry(_as_mode(mode), lookback)
        if test_data.empty:
            cli_console.print("[red]Failed to fetch test data[/red]")
            raise typer.Exit(1)

    # Run backtest
    with cli_console.status("[bold green]Running backtest..."):
        result = run_backtest(
            model_instance,
            test_data,
            strategy_name=strategy,
            exit_bars=exit_bars,
            lookback=lookback,
        )

    print_backtest_result(result)


@app.command()
def explore(
    mode: str = typer.Option("historical", help="Data mode"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days"),
) -> None:
    """Explore data statistics and correlations."""
    with cli_console.status("[bold green]Fetching data..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        cli_console.print("[red]Failed to fetch data[/red]")
        raise typer.Exit(1)

    print_data_summary(data, f"Data Exploration ({lookback}d)")

    # Correlation matrix
    corr_table = Table(title="Correlation Matrix")
    corr_table.add_column("", style="cyan")
    for col in ["open", "high", "low", "close", "volume"]:
        corr_table.add_column(col, style="green")

    corr = data[["open", "high", "low", "close", "volume"]].corr()
    for idx, row in corr.iterrows():
        corr_table.add_row(str(idx), *[f"{v:.3f}" for v in row])

    cli_console.print(corr_table)


@app.command()
def compare(
    periods: int = typer.Option(SETTINGS.default_prediction_periods, help="Prediction periods / holdout"),
    mode: str = typer.Option("current", help="Data mode for fetch"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days"),
    fast: bool = typer.Option(True, "--fast/--full", help="Fast preset (m 7-9, iter 10) vs full (7-50, iter 100)"),
    holdout: int | None = typer.Option(None, help="Holdout for scoring (0=unscored, default=periods)"),
    timeout: int | None = typer.Option(None, help="Timeout seconds per model (default 90 fast, 300 full)"),
) -> None:
    """Compare all models with progress, scoring, timeout - non-REPL version."""
    # lazy import to avoid circular
    from .repl import _print_model_comparison, _run_compare_core

    with cli_console.status(f"[bold green]Fetching {mode} data ({lookback}d)..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        cli_console.print("[red]Failed to fetch data[/red]")
        raise typer.Exit(1)

    print_data_summary(data, f"Fetched Data ({mode}, {lookback}d)")

    if timeout is None:
        timeout = SETTINGS.compare_timeout_seconds if fast else SETTINGS.compare_full_timeout_seconds

    # Determine holdout: if None auto=periods, fast flag already handled
    results = _run_compare_core(data, periods, fast, holdout, timeout)
    holdout_str = holdout if holdout is not None else periods
    title = f"Model Comparison (periods={periods} holdout={holdout_str} {'fast' if fast else 'full'})"
    _print_model_comparison(results, title=title)


@app.command()
def benchmark(
    holdout: int = typer.Option(SETTINGS.default_prediction_periods, help="Holdout periods"),
    mode: str = typer.Option("current", help="Data mode"),
    lookback: int = typer.Option(SETTINGS.default_lookback_days, help="Lookback days"),
    fast: bool = typer.Option(False, "--fast/--full", help="Fast preset"),
    timeout: int | None = typer.Option(None, help="Timeout seconds"),
) -> None:
    """Benchmark all models on held-out tail (alias for compare --full)."""
    from .repl import _print_model_comparison, _run_compare_core

    with cli_console.status(f"[bold green]Fetching {mode} data ({lookback}d)..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        cli_console.print("[red]Failed to fetch data[/red]")
        raise typer.Exit(1)

    if timeout is None:
        timeout = SETTINGS.compare_timeout_seconds if fast else SETTINGS.compare_full_timeout_seconds

    results = _run_compare_core(data, holdout, fast, holdout, timeout)
    _print_model_comparison(results, title=f"Benchmark (holdout={holdout} fast={fast})")


@app.command()
def repl() -> None:
    """Start interactive REPL."""
    from .repl import run_repl

    run_repl()


if __name__ == "__main__":
    app()