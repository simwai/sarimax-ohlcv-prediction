"""Interactive REPL for model training, prediction, and exploration."""

import contextlib
import logging
import time
from typing import Literal, cast

import numpy as np
import pandas as pd
from rich import box
from rich.console import Group
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
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ..backtest import print_backtest_result, run_backtest
from ..backtest.strategies import STRATEGY_REGISTRY
from ..config import SETTINGS
from ..data.fetcher import fetch_with_retry
from ..models import MODEL_REGISTRY, create_model
from ..viz.rich import console, print_data_summary, print_predictions_table

Mode = Literal["current", "historical"]


def _as_mode(mode: str) -> Mode:
    if mode not in ("current", "historical"):
        console.print(f"[yellow]Invalid mode: {mode}, using current[/yellow]")
        return cast(Mode, "current")
    return cast(Mode, mode)

# Global state for REPL
_state = {
    "data": None,
    "model": None,
    "model_name": None,
    "model_path": None,
    "predictions": None,
}

_MIN_ARGS_LOAD = 1
_MIN_ARGS_BACKTEST_LOOKBACK = 3
_MIN_HOLDOUT = 2


def _build_help_table() -> Table:
    """Build aligned help table: cmd | args | description (fixed columns)."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        show_edge=False,
    )
    table.add_column("cmd", style="cyan", no_wrap=True, width=14)
    table.add_column("args", style="dim", no_wrap=False, min_width=32, max_width=45)
    table.add_column("desc", style="white", no_wrap=False)

    def section(title: str) -> None:
        table.add_row("", "", "")  # blank line before section
        table.add_row(f"[bold cyan]{title}[/]", "", "")

    def row(cmd: str, args: str, desc: str) -> None:
        table.add_row(
            f"[cyan]{cmd}[/]" if cmd else "",
            f"[dim]{args}[/]" if args else "",
            desc,
        )

    section("Data")
    row("fetch", "[mode] [lookback]", "Fetch data (current/historical)")
    row("explore", "", "Show data statistics and correlations")
    row("show", "", "Show current data summary")

    section("Models")
    row("train", "<model> [iters]", "Train a model (SARIMAX/Prophet/LSTM)")
    row("load", "<model> [path]", "Load model from file (default: models/<model>_model.joblib)")
    row("save", "[path]", "Save current model")
    row("models", "", "List available models")

    section("Prediction")
    row("predict", "[periods]", "Make predictions")
    row("compare", "[periods] [--fast|--full] [--holdout N] [--timeout S]", "Compare all models (fast by default, scored on holdout)")
    row("benchmark", "[holdout]", "Train all models, score on held-out tail (alias for compare --full)")

    section("Backtesting")
    row("backtest", "[strategy] [params]", "Run backtest")
    row("strategies", "", "List available strategies")

    section("System")
    row("help", "", "Show this help")
    row("status", "", "Show current state")
    row("clear", "", "Clear screen")
    row("exit / quit", "", "Exit REPL")

    return table


def print_help(args: list[str]) -> None:
    """Print REPL help."""
    table = _build_help_table()
    console.print(Panel(table, title="REPL Help", border_style="blue", padding=(1, 1)))


def print_status(args: list[str]) -> None:
    """Print current REPL state compactly - inline status line, verbose table on --verbose."""
    verbose = "--verbose" in args or "-v" in args
    data_rows = len(_state["data"]) if _state["data"] is not None else 0
    pred_rows = len(_state["predictions"]) if _state["predictions"] is not None else 0
    data_str = str(data_rows) if _state["data"] is not None else "—"
    model_str = _state["model_name"] or "—"
    path_str = _state["model_path"] or "—"
    pred_str = str(pred_rows) if _state["predictions"] is not None else "—"

    if verbose:
        table = Table(title="REPL Status", box=box.MINIMAL, show_header=False, padding=(0, 1))
        table.add_column("Item", style="cyan", no_wrap=True, width=14)
        table.add_column("Value", style="green")
        table.add_row("Data", data_str)
        table.add_row("Model", model_str)
        table.add_row("Model Path", path_str)
        table.add_row("Predictions", pred_str)
        console.print(table)
    else:
        console.print(
            f"[dim]data[/] {data_str}  [dim]· model[/] {model_str}  [dim]· path[/] {path_str}  [dim]· preds[/] {pred_str}"
        )


def cmd_fetch(args: list[str]) -> None:
    """Fetch data command."""
    mode = args[0] if args else "current"
    lookback = int(args[1]) if len(args) > 1 else SETTINGS.default_lookback_days

    with console.status(f"[bold green]Fetching {mode} data ({lookback}d)..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        console.print("[red]Failed to fetch data[/red]")
        return

    _state["data"] = data
    print_data_summary(data, f"Fetched Data ({mode}, {lookback}d)")


def cmd_explore(args: list[str]) -> None:
    """Explore data command."""
    if _state["data"] is None:
        console.print("[yellow]No data loaded. Use 'fetch' first.[/yellow]")
        return

    print_data_summary(_state["data"], "Data Exploration")

    # Correlation matrix
    corr_table = Table(title="Correlation Matrix")
    corr_table.add_column("", style="cyan")
    for col in ["open", "high", "low", "close", "volume"]:
        corr_table.add_column(col, style="green")

    corr = _state["data"][["open", "high", "low", "close", "volume"]].corr()  # type: ignore[union-attr]
    for idx, row in corr.iterrows():
        corr_table.add_row(str(idx), *[f"{v:.3f}" for v in row])

    console.print(corr_table)


def cmd_show(args: list[str]) -> None:
    """Show current data."""
    if _state["data"] is None:
        console.print("[yellow]No data loaded.[/yellow]")
        return
    print_data_summary(_state["data"], "Current Data")


def cmd_train(args: list[str]) -> None:
    """Train model command."""
    if _state["data"] is None:
        console.print("[yellow]No data loaded. Use 'fetch' first.[/yellow]")
        return

    if not args:
        console.print("[red]Usage: train <model> [iterations][/red]")
        return

    model_name = args[0]
    if model_name not in MODEL_REGISTRY:
        console.print(
            f"[red]Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}[/red]"
        )
        return

    iterations = int(args[1]) if len(args) > 1 else SETTINGS.default_iterations

    with console.status(f"[bold green]Training {model_name}..."):
        model_instance = create_model(model_name)
        model_instance.fit(_state["data"], iterations=iterations)

    _state["model"] = model_instance
    _state["model_name"] = model_name

    default_path = f"models/{model_name.lower()}_model.joblib"
    model_instance.save(default_path)
    _state["model_path"] = default_path
    console.print(f"[green]{model_name} trained and saved to {default_path}[/green]")


def cmd_load(args: list[str]) -> None:
    """Load model command."""
    if len(args) < _MIN_ARGS_LOAD:
        console.print("[red]Usage: load <model> [path][/red]")
        return

    model_name = args[0]
    default_path = f"{SETTINGS.models_dir}/{model_name.lower()}_model.joblib"
    path = args[1] if len(args) > 1 else default_path

    if model_name not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model_name}[/red]")
        return

    try:
        with console.status(f"[bold green]Loading {model_name} from {path}..."):
            model_class = MODEL_REGISTRY[model_name]
            model_instance = model_class.load(path)

        _state["model"] = model_instance
        _state["model_name"] = model_name
        _state["model_path"] = path
        console.print(f"[green]{model_name} loaded from {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")


def cmd_save(args: list[str]) -> None:
    """Save model command."""
    if _state["model"] is None:
        console.print("[yellow]No model trained/loaded.[/yellow]")
        return

    model_name = _state["model_name"]
    default_path = f"models/{model_name.lower()}_model.joblib" if isinstance(model_name, str) else "models/model.joblib"
    path = args[0] if args else default_path

    try:
        _state["model"].save(path)
        _state["model_path"] = path
        console.print(f"[green]Model saved to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save model: {e}[/red]")


def cmd_models(args: list[str]) -> None:
    """List available models."""
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Description", style="yellow")

    descriptions = {
        "SARIMAX": "Seasonal ARIMA with exogenous variables (pmdarima)",
        "Prophet": "Facebook Prophet with custom seasonalities",
        "LSTM": "Long Short-Term Memory neural network (Keras/TF)",
    }

    for name, cls in MODEL_REGISTRY.items():
        table.add_row(name, cls.__name__, descriptions.get(name, ""))

    console.print(table)


def cmd_predict(args: list[str]) -> None:
    """Predict command."""
    if _state["model"] is None:
        console.print("[yellow]No model loaded. Use 'train' or 'load' first.[/yellow]")
        return

    periods = int(args[0]) if args else SETTINGS.default_prediction_periods

    try:
        with console.status("[bold green]Generating predictions..."):
            if _state["model_name"] == "LSTM":
                if _state["data"] is None:
                    console.print("[red]LSTM requires data context. Fetch data first.[/red]")
                    return
                predictions = _state["model"].predict_with_context(_state["data"], periods)
            else:
                predictions = _state["model"].predict(periods)

        _state["predictions"] = predictions
        print_predictions_table(
            predictions, title=f"{_state['model_name']} Predictions ({periods} periods)"
        )

    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")


def _print_model_comparison(results: dict, title: str = "Model Comparison") -> None:
    """Print model comparison table."""
    table = Table(title=title)
    table.add_column("Model", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("RMSE", style="yellow")
    table.add_column("MAE", style="yellow")
    table.add_column("MAPE", style="magenta")
    table.add_column("Info", style="dim")
    table.add_column("Time", style="blue")

    for name, info in results.items():
        table.add_row(
            name,
            info.get("status", "N/A"),
            str(info.get("rmse", "N/A")),
            str(info.get("mae", "N/A")),
            str(info.get("mape", "N/A")),
            str(info.get("info", "N/A"))[:60],
            str(info.get("elapsed", "N/A")),
        )

    console.print(table)


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE between actual and predicted series."""
    err = actual - predicted
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    nonzero = np.where(actual != 0, actual, np.nan)
    mape = float(np.nanmean(np.abs(err / nonzero)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def _make_progress(description: str) -> Progress:
    """Create a rich Progress with spinner+bar for model training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _silence_model_loggers() -> dict[str, int]:
    """Silence noisy model loggers during Live/Progress to avoid bar corruption."""
    names = [
        "sarimax_ohlcv_prediction.models.sarimax",
        "sarimax_ohlcv_prediction.models.prophet",
        "sarimax_ohlcv_prediction.models.lstm",
    ]
    prev: dict[str, int] = {}
    for n in names:
        lg = logging.getLogger(n)
        prev[n] = lg.level
        lg.setLevel(logging.WARNING)
    # also root pmdarima/prophet loggers could leak
    for n in ["pmdarima", "prophet"]:
        lg = logging.getLogger(n)
        prev[n] = lg.level
        lg.setLevel(logging.WARNING)
    return prev


def _restore_model_loggers(prev: dict[str, int]) -> None:
    for n, lvl in prev.items():
        logging.getLogger(n).setLevel(lvl)


def _parse_holdout_flag(args: list[str], idx: int, holdout: int | None) -> tuple[int | None, int]:
    """Parse --holdout <n> or --holdout=<n>. Returns (new_holdout, new_idx)."""
    a = args[idx]
    if a == "--holdout":
        if idx + 1 >= len(args):
            console.print("[red]--holdout requires value[/red]")
            return holdout, idx
        try:
            return int(args[idx + 1]), idx + 1
        except ValueError:
            console.print(f"[red]Invalid --holdout value: {args[idx + 1]}[/red]")
            return holdout, idx + 1
    try:
        return int(a.split("=", 1)[1]), idx
    except ValueError:
        console.print(f"[red]Invalid --holdout value: {a}[/red]")
        return holdout, idx


def _parse_timeout_flag(args: list[str], idx: int, timeout: int | None) -> tuple[int | None, int, bool]:
    """Parse --timeout <n> or --timeout=<n>. Returns (new_timeout, new_idx, explicit_set)."""
    a = args[idx]
    if a == "--timeout":
        if idx + 1 >= len(args):
            console.print("[red]--timeout requires value[/red]")
            return timeout, idx, False
        try:
            return int(args[idx + 1]), idx + 1, True
        except ValueError:
            console.print(f"[red]Invalid --timeout value: {args[idx + 1]}[/red]")
            return timeout, idx + 1, False
    try:
        return int(a.split("=", 1)[1]), idx, True
    except ValueError:
        console.print(f"[red]Invalid --timeout value: {a}[/red]")
        return timeout, idx, False


def _parse_compare_args(args: list[str]) -> tuple[int, bool, int | None, int | None]:
    """Parse compare/benchmark args.

    Returns: (periods, fast, holdout, timeout)
    holdout None means auto=periods; 0 means disable scoring.
    """
    periods = SETTINGS.default_prediction_periods
    fast = True
    holdout: int | None = None
    timeout: int | None = None
    timeout_explicit = False
    remaining = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--fast":
            fast = True
        elif a == "--full":
            fast = False
        elif a == "--holdout" or a.startswith("--holdout="):
            holdout, i = _parse_holdout_flag(args, i, holdout)
        elif a == "--timeout" or a.startswith("--timeout="):
            timeout, i, timeout_explicit = _parse_timeout_flag(args, i, timeout)
        elif a.startswith("--"):
            console.print(f"[yellow]Unknown flag: {a}[/yellow]")
        else:
            remaining.append(a)
        i += 1
    if remaining:
        with contextlib.suppress(ValueError):
            periods = int(remaining[0])
        if periods != SETTINGS.default_prediction_periods:
            console.print(f"[red]Invalid periods: {remaining[0]}[/red]")
        if len(remaining) > 1 and holdout is None:
            with contextlib.suppress(ValueError):
                holdout = int(remaining[1])
    if not timeout_explicit:
        timeout = SETTINGS.compare_timeout_seconds if fast else SETTINGS.compare_full_timeout_seconds
    return periods, fast, holdout, timeout


def _resolve_compare_params(
    data,
    periods: int,
    fast: bool,
    holdout: int | None,
) -> tuple[int, pd.DataFrame, np.ndarray | None, bool, int]:
    """Resolve final holdout/scored/train_df/actual_close/predict_periods."""
    if holdout is None:
        holdout = periods if len(data) > periods + 10 else 0
    if holdout is not None and holdout != 0 and len(data) <= holdout:
        console.print(f"[yellow]Not enough data for holdout={holdout}, disabling scoring[/yellow]")
        holdout = 0
    if holdout is not None and holdout != 0 and holdout < _MIN_HOLDOUT:
        console.print("[red]Holdout must be >=2, disabling scoring[/red]")
        holdout = 0

    scored = holdout is not None and holdout != 0
    if scored:
        assert holdout is not None
        train_df = data.iloc[:-holdout]
        test_df = data.iloc[-holdout:]
        actual_close: np.ndarray | None = test_df["close"].to_numpy(dtype=float)
        predict_periods = holdout
    else:
        train_df = data
        actual_close = None
        predict_periods = periods

    return holdout, train_df, actual_close, scored, predict_periods


def _resolve_mode_params(fast: bool) -> tuple[range, int, int]:
    """Pick SARIMAX and LSTM params for fast/full mode."""
    if fast:
        return (
            SETTINGS.compare_fast_m_range,
            SETTINGS.compare_fast_iterations,
            min(10, SETTINGS.lstm_epochs),
        )
    return (
        SETTINGS.compare_full_m_range,
        SETTINGS.compare_full_iterations,
        SETTINGS.lstm_epochs,
    )


def _make_progress_callback(
    progress: Progress, model_name: str, task_id: TaskID, total: int
):
    """Build a progress callback bound to (progress, model_name, task_id, total)."""
    def _cb(col_or_epoch: str, val: int, done: int) -> None:
        try:
            if model_name == "SARIMAX":
                progress.update(task_id, completed=done, description=f"[cyan]{model_name} {col_or_epoch} m={val}")
            elif model_name == "Prophet":
                progress.update(task_id, completed=done, description=f"[cyan]{model_name} {col_or_epoch}")
            elif model_name == "LSTM":
                progress.update(task_id, completed=val, description=f"[cyan]{model_name} epoch {val}/{total}")
        except Exception:
            pass
    return _cb


def _build_fit_kwargs(
    name: str,
    cb,
    timeout: int | None,
    sarimax_m_range: range,
    sarimax_iterations: int,
    lstm_epochs: int,
) -> dict:
    """Assemble kwargs for model.fit()."""
    fit_kwargs: dict = {"progress_callback": cb}
    if timeout is not None and timeout > 0:
        fit_kwargs["timeout"] = float(timeout)
        fit_kwargs["start_time"] = time.time()
    if name == "SARIMAX":
        fit_kwargs["m_range"] = sarimax_m_range
        fit_kwargs["iterations"] = sarimax_iterations
    elif name == "LSTM":
        fit_kwargs["epochs"] = lstm_epochs
    return fit_kwargs


def _score_predictions(
    model_instance,
    name: str,
    predictions,
    actual_close: np.ndarray | None,
    elapsed_str: str,
    pending_logs: list[tuple[str, str]],
) -> dict:
    """Score predictions against actual_close. Returns row dict for results table."""
    predicted_close = predictions["close"].to_numpy(dtype=float)
    if actual_close is None or len(predicted_close) == 0:
        info_str = _format_model_info(model_instance, name)
        pending_logs.append(("green", f"{name}: Done ({elapsed_str}) - {info_str}"))
        return {
            "status": "Success",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info_str,
            "elapsed": elapsed_str,
        }

    if len(predicted_close) != len(actual_close):
        min_len = min(len(predicted_close), len(actual_close))
        predicted_close = predicted_close[:min_len]
        actual_close_trim: np.ndarray = actual_close[:min_len]
    else:
        actual_close_trim = actual_close

    if np.isnan(predicted_close).any():
        info = _format_model_info(model_instance, name) + " (incomplete)"
        pending_logs.append(("yellow", f"{name}: Timeout/partial - {elapsed_str} - {info}"))
        return {
            "status": "Timeout (partial)",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info,
            "elapsed": elapsed_str,
        }

    metrics = _compute_metrics(actual_close_trim, predicted_close)
    info = _format_model_info(model_instance, name)
    if any(np.isnan(v) for v in metrics.values()):
        pending_logs.append(("yellow", f"{name}: Partial - no valid close prediction ({elapsed_str})"))
        return {
            "status": "Partial",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info,
            "elapsed": elapsed_str,
        }
    pending_logs.append(
        (
            "green",
            f"{name}: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} MAPE={metrics['mape']:.2f}% ({elapsed_str})",
        )
    )
    return {
        "status": "Success",
        "rmse": f"{metrics['rmse']:.4f}",
        "mae": f"{metrics['mae']:.4f}",
        "mape": f"{metrics['mape']:.2f}%",
        "info": info,
        "elapsed": elapsed_str,
    }


def _handle_interrupt(name: str, start: float, pending_logs: list[tuple[str, str]]) -> dict:
    """Build interrupted-row dict for results."""
    elapsed = time.time() - start
    pending_logs.append(("yellow", f"{name}: Interrupted ({elapsed:.1f}s)"))
    return {
        "status": "Interrupted",
        "rmse": "N/A",
        "mae": "N/A",
        "mape": "N/A",
        "info": "user interrupt",
        "elapsed": f"{elapsed:.1f}s",
    }


def _handle_failure(name: str, start: float, exc: Exception, pending_logs: list[tuple[str, str]]) -> dict:
    """Build failed-row dict for results."""
    elapsed = time.time() - start
    pending_logs.append(("red", f"{name}: Failed - {exc} ({elapsed:.1f}s)"))
    return {
        "status": f"Failed: {exc}",
        "rmse": "N/A",
        "mae": "N/A",
        "mape": "N/A",
        "info": "N/A",
        "elapsed": f"{elapsed:.1f}s",
    }


def _train_and_predict_one(
    name: str,
    train_df: pd.DataFrame,
    predict_periods: int,
    progress: Progress,
    task_ids: dict[str, TaskID],
    scored: bool,
    holdout: int | None,
    periods: int,
    timeout: int | None,
    mode_label: str,
    status_text: Text,
    pending_logs: list[tuple[str, str]],
    actual_close: np.ndarray | None,
    sarimax_m_range: range,
    sarimax_iterations: int,
    lstm_epochs: int,
) -> tuple[str, dict, float]:
    """Train and predict one model. Returns (name, result_row, elapsed_seconds)."""
    start = time.time()
    total = int(progress.tasks[task_ids[name]].total or 1)
    if scored:
        status_text.plain = f"train {len(train_df)} · holdout {holdout} · {mode_label} · {timeout}s | {name} training..."
    else:
        status_text.plain = f"train {len(train_df)} · predict {periods} · {mode_label} | {name} training..."
    progress.start_task(task_ids[name])
    progress.update(task_ids[name], description=f"[cyan]{name} training...")

    try:
        model_instance = create_model(name)
        cb = _make_progress_callback(progress, name, task_ids[name], total)
        fit_kwargs = _build_fit_kwargs(
            name, cb, timeout, sarimax_m_range, sarimax_iterations, lstm_epochs
        )
        try:
            model_instance.fit(train_df, **fit_kwargs)
        except KeyboardInterrupt:
            pending_logs.append(("yellow", f"{name}: interrupted by user, keeping best so far"))
        progress.update(task_ids[name], completed=total, description=f"[green]{name} training done")

        status_text.plain = (
            f"train {len(train_df)} · holdout {holdout} · {mode_label} | {name} predicting..."
        )
        if name == "LSTM":
            predictions = model_instance.predict_with_context(train_df, predict_periods)
        else:
            predictions = model_instance.predict(predict_periods)

        elapsed = time.time() - start
        elapsed_str = f"{elapsed:.1f}s"
        if scored:
            row = _score_predictions(model_instance, name, predictions, actual_close, elapsed_str, pending_logs)
        else:
            row = _score_predictions(model_instance, name, predictions, None, elapsed_str, pending_logs)
        return name, row, elapsed
    except Exception as exc:
        progress.stop_task(task_ids[name])
        return name, _handle_failure(name, start, exc, pending_logs), time.time() - start


def _run_compare_core(
    data,
    periods: int,
    fast: bool,
    holdout: int | None,
    timeout: int | None,
) -> dict[str, dict]:
    """Core compare logic with dual-area Live (status + progress bar).

    Status area (top) shows train/holdout/mode/timeout + current model.
    Progress area (bottom) shows a single transient Progress bar - never
    interleaves with status lines. INFO logs silenced to avoid bar corruption.

    If holdout is None, auto = periods (scored). If holdout==0, no scoring.
    Returns results dict for table.
    """
    holdout, train_df, actual_close, scored, predict_periods = _resolve_compare_params(
        data, periods, fast, holdout
    )
    sarimax_m_range, sarimax_iterations, lstm_epochs = _resolve_mode_params(fast)

    mode_label = "fast" if fast else "full"
    if scored:
        header_plain = f"train {len(train_df)} · holdout {holdout} · {mode_label} · {timeout}s timeout"
    else:
        header_plain = f"train {len(train_df)} · predict {periods} · {mode_label} · {timeout}s timeout"
    status_text = Text(header_plain, style="dim")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task_ids: dict[str, TaskID] = {}
    for _name in MODEL_REGISTRY:
        if _name == "SARIMAX":
            _total = len(sarimax_m_range) * 5
        elif _name == "Prophet":
            _total = 5
        else:
            _total = lstm_epochs
        task_ids[_name] = progress.add_task(f"[dim]{_name} queued...[/]", total=_total, start=False)

    group = Group(status_text, progress)
    pending_logs: list[tuple[str, str]] = []
    results: dict[str, dict] = {}
    prev_log_levels = _silence_model_loggers()

    try:
        with Live(group, console=console, refresh_per_second=12, transient=True):
            for name in MODEL_REGISTRY:
                _, row, _ = _train_and_predict_one(
                    name,
                    train_df,
                    predict_periods,
                    progress,
                    task_ids,
                    scored,
                    holdout,
                    periods,
                    timeout,
                    mode_label,
                    status_text,
                    pending_logs,
                    actual_close,
                    sarimax_m_range,
                    sarimax_iterations,
                    lstm_epochs,
                )
                results[name] = row
            status_text.plain = header_plain + " · done"
    finally:
        _restore_model_loggers(prev_log_levels)

    console.print(f"[dim]{header_plain}[/]")
    for style, msg in pending_logs:
        console.print(f"[{style}]{msg}[/]")

    return results


def _format_model_info(model_instance, name: str) -> str:
    """Format compact info string for table."""
    try:
        if name == "SARIMAX" and hasattr(model_instance, "best_m_values"):
            bm = getattr(model_instance, "best_m_values", {})
            if bm:
                # show best m per column compactly
                return "best_m=" + ",".join(f"{k}:{v}" for k, v in bm.items())
            return "no best_m"
        if name == "Prophet" and hasattr(model_instance, "daily_period"):
            return f"daily={model_instance.daily_period} weekly={model_instance.weekly_period}"
        if name == "LSTM" and hasattr(model_instance, "device"):
            return f"epochs={model_instance.epochs} device={model_instance.device}"
        # fallback
        return type(model_instance).__name__
    except Exception:
        return "N/A"


def cmd_benchmark(args: list[str]) -> None:
    """Train all models and score predictions on a held-out tail of the data."""
    if _state["data"] is None:
        console.print("[yellow]No data loaded. Use 'fetch' first.[/yellow]")
        return

    # benchmark defaults to full (not fast) unless --fast flag given
    # support both old style: benchmark 24  and new flags
    # if args contains --fast/--full use parse, else treat first arg as holdout and full=True
    has_flag = any(a.startswith("--") for a in args)
    if has_flag:
        periods, fast, holdout, timeout = _parse_compare_args(args)
        # benchmark typically scored, so holdout defaults to periods
        if holdout is None:
            holdout = periods
        # benchmark defaults to full regardless of parse unless explicitly --fast
        if "--fast" not in args and "--full" not in args:
            fast = False
        results = _run_compare_core(_state["data"], periods, fast, holdout, timeout)
        title = f"Benchmark (holdout={holdout} fast={fast})"
    else:
        holdout = int(args[0]) if args else SETTINGS.default_prediction_periods
        if holdout < _MIN_HOLDOUT:
            console.print("[red]Holdout must be >= 2 periods.[/red]")
            return
        if len(_state["data"]) <= holdout:
            console.print(f"[red]Need more than {holdout} rows; have {len(_state['data'])}.[/red]")
            return
        # legacy "benchmark N" path = --full style, use the full timeout preset
        results = _run_compare_core(
            _state["data"], holdout, False, holdout, SETTINGS.compare_full_timeout_seconds
        )
        title = f"Benchmark (holdout={holdout})"

    _print_model_comparison(results, title=title)


def cmd_compare(args: list[str]) -> None:
    """Compare all models - fast, scored, progress, cancellable, timeout."""
    if _state["data"] is None:
        console.print("[yellow]No data loaded. Use 'fetch' first.[/yellow]")
        return

    periods, fast, holdout, timeout = _parse_compare_args(args)
    results = _run_compare_core(_state["data"], periods, fast, holdout, timeout)
    holdout_str = holdout if holdout is not None else periods
    title = f"Model Comparison (periods={periods} holdout={holdout_str} {'fast' if fast else 'full'})"
    _print_model_comparison(results, title=title)
    console.print("[dim]Tip: compare --full --holdout 24 --timeout 300 for full sweep; compare --holdout 0 for unscored[/dim]")


def cmd_backtest(args: list[str]) -> None:
    """Backtest command."""
    if _state["model"] is None:
        console.print("[yellow]No model loaded. Use 'train' or 'load' first.[/yellow]")
        return

    if _state["data"] is None:
        console.print("[yellow]No data loaded. Use 'fetch' first.[/yellow]")
        return

    strategy = args[0] if args else "exit_after_n"
    exit_bars = int(args[1]) if len(args) > 1 else SETTINGS.backtest_default_exit_bars
    lookback = int(args[2]) if len(args) > _MIN_ARGS_BACKTEST_LOOKBACK - 1 else SETTINGS.backtest_default_lookback

    test_data = _state["data"].iloc[-lookback:]

    with console.status("[bold green]Running backtest..."):
        result = run_backtest(
            _state["model"],
            test_data,
            strategy_name=strategy,
            exit_bars=exit_bars,
            lookback=lookback,
        )

    print_backtest_result(result)


def cmd_strategies(args: list[str]) -> None:
    """List available strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Description", style="yellow")

    descriptions = {
        "exit_after_n": "Enter on prediction direction, exit after N bars",
        "exit_on_signal": "Enter on direction, exit when signal reverses",
    }

    for name, cls in STRATEGY_REGISTRY.items():
        table.add_row(name, cls.__name__, descriptions.get(name, ""))

    console.print(table)


def parse_command(line: str) -> tuple[str, list[str]]:
    """Parse command line into command and args."""
    parts = line.strip().split()
    if not parts:
        return "", []
    return parts[0].lower(), parts[1:]


_CMD_HANDLERS = {
    "help": print_help,
    "status": print_status,
    "clear": lambda _args: console.clear(),
    "fetch": cmd_fetch,
    "explore": cmd_explore,
    "show": cmd_show,
    "train": cmd_train,
    "load": cmd_load,
    "save": cmd_save,
    "models": cmd_models,
    "predict": cmd_predict,
    "compare": cmd_compare,
    "benchmark": cmd_benchmark,
    "backtest": cmd_backtest,
    "strategies": cmd_strategies,
}


def run_repl() -> None:
    """Run the interactive REPL loop."""
    console.print()
    console.print(
        Panel(
            "[bold bright_cyan]SARIMAX OHLCV Prediction REPL[/]\n"
            "[dim]Type 'help' for commands, 'exit' to quit[/]",
            border_style="bright_cyan",
            padding=(0, 1),
            style="on blue",
        )
    )

    while True:
        try:
            line = Prompt.ask("[bold bright_cyan]sarimax>[/]")
            cmd, args = parse_command(line)

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            handler = _CMD_HANDLERS.get(cmd)
            if handler:
                handler(args)
            else:
                console.print(f"[red]Unknown command: {cmd}. Type 'help' for commands.[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    run_repl()