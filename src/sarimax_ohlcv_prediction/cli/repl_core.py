"""Shared REPL command handlers and parsing logic."""

from typing import Any

from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..backtest import STRATEGY_REGISTRY, run_backtest
from ..config import SETTINGS
from ..data.fetcher import fetch_with_retry
from ..data.modes import Mode, as_mode
from ..models import MODEL_REGISTRY, create_model
from ..viz.components import DATA_STYLES, print_backtest_results
from ..viz.rich import console, print_data_summary, print_model_comparison, print_predictions_table
from .compare_core import _parse_compare_args, _run_compare_core

_INVALID_MODE_MSG = "Invalid mode: {mode}"


class HelpTable(Table):
    """Help table with fixed columns: Command | Args | Description."""

    def __init__(self):
        super().__init__(
            title="REPL Commands",
            title_style="panel.title",
            box=None,
            padding=(0, 1),
            collapse_padding=True,
            header_style="table.header",
            row_styles=["table.row_even", "table.row_odd"],
        )
        self.add_column("Command", style="brand", no_wrap=True, min_width=14)
        self.add_column("Args", style="warning", min_width=28)
        self.add_column("Description", style="success")

    def add_section(self, title: str) -> None:  # pyrefly: ignore[bad-override] -- intentional REPL helper
        self.add_row(f"[section]{title}[/]", "", "")

    def add_command(self, cmd: str, args: str, desc: str) -> None:
        self.add_row(cmd, args, desc)


def _as_mode(mode: str) -> Mode:
    """Validate and cast mode string to Literal."""
    try:
        return as_mode(mode)
    except ValueError as exc:
        raise ValueError(_INVALID_MODE_MSG.format(mode=mode)) from exc


def parse_command(line: str) -> tuple[str, list[str]]:
    """Parse command line into command and args."""
    parts = line.strip().split()
    if not parts:
        return "", []
    return parts[0].lower(), parts[1:]


_CMD_HANDLERS: dict[str, Any] = {}


def register_handler(name: str):
    """Decorator to register a command handler."""

    def decorator(fn):
        _CMD_HANDLERS[name] = fn
        return fn

    return decorator


def print_help(args: list[str]) -> None:
    """Print REPL help."""
    table = HelpTable()

    table.add_section("Data")
    table.add_command("fetch", "[mode] [lookback]", "Fetch data (current/historical)")
    table.add_command("explore", "", "Show data statistics and correlations")
    table.add_command("show", "", "Show current data summary")

    table.add_section("Models")
    table.add_command("train", "<model> [iters]", "Train a model (SARIMAX/Prophet/LSTM)")
    table.add_command(
        "load", "<model> [path]", "Load model from file (default: models/<model>_model.joblib)"
    )
    table.add_command("save", "[path]", "Save current model")
    table.add_command("models", "", "List available models")

    table.add_section("Prediction")
    table.add_command("predict", "[periods]", "Make predictions")
    table.add_command(
        "compare",
        "[periods] [--fast|--full] [--holdout N] [--timeout S]",
        "Compare all models (fast by default, scored on holdout)",
    )
    table.add_command(
        "benchmark",
        "[holdout]",
        "Train all models, score on held-out tail (alias for compare --full)",
    )

    table.add_section("Backtesting")
    table.add_command("backtest", "[strategy] [params]", "Run backtest")
    table.add_command("strategies", "", "List available strategies")

    table.add_section("System")
    table.add_command("help", "", "Show this help")
    table.add_command("status", "", "Show current state")
    table.add_command("clear", "", "Clear screen")
    table.add_command("exit / quit", "", "Exit REPL")

    console.print(Panel(table, title="REPL Help", border_style="ui.border", padding=(1, 1)))


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


def _parse_int(value: str, label: str) -> int | None:
    """Parse an int arg with a clean REPL error."""
    try:
        return int(value)
    except ValueError:
        console.print(f"[error]Invalid {label}: {value}[/error]")
        return None


@register_handler("help")
def cmd_help(args: list[str]) -> None:
    print_help(args)


@register_handler("status")
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
        print_verbose_status(data_str, model_str, path_str, pred_str)
    else:
        print_inline_status(data_str, model_str, path_str, pred_str)


def print_inline_status(data_str: str, model_str: str, path_str: str, pred_str: str) -> None:
    line = (
        f"[brand]data[/]: {data_str}  "
        f"[brand]model[/]: {model_str}  "
        f"[brand]path[/]: {path_str}  "
        f"[brand]pred[/]: {pred_str}"
    )
    console.print(line)


def print_verbose_status(data_str: str, model_str: str, path_str: str, pred_str: str) -> None:
    table = Table(
        title="REPL Status",
        title_style="panel.title",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    table.add_column("Key", style="brand")
    table.add_column("Value", style="ui.text")
    table.add_row("Data rows", data_str)
    table.add_row("Model", model_str)
    table.add_row("Model path", path_str)
    table.add_row("Predictions", pred_str)
    console.print(table)


@register_handler("clear")
def cmd_clear(args: list[str]) -> None:
    console.clear()


@register_handler("fetch")
def cmd_fetch(args: list[str]) -> None:
    """Fetch data command."""
    mode = args[0] if args else "current"
    lookback = SETTINGS.default_lookback_days
    if len(args) > 1:
        parsed = _parse_int(args[1], "lookback")
        if parsed is None:
            return
        lookback = parsed

    with console.status(f"[status.running]Fetching {mode} data ({lookback}d)..."):
        data = fetch_with_retry(_as_mode(mode), lookback)

    if data.empty:
        console.print("[error]Failed to fetch data[/error]")
        return

    _state["data"] = data
    print_data_summary(data, f"Fetched Data ({mode}, {lookback}d)")


@register_handler("explore")
def cmd_explore(args: list[str]) -> None:
    """Explore data command."""
    if _state["data"] is None:
        console.print("[warning]No data loaded. Use 'fetch' first.[/warning]")
        return

    print_data_summary(_state["data"], "Data Exploration")

    # Correlation matrix
    corr_table = Table(
        title="Correlation Matrix",
        title_style="panel.title",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    corr_table.add_column("", style="brand", no_wrap=True)
    for col in ["open", "high", "low", "close", "volume"]:
        corr_table.add_column(col, style=DATA_STYLES.get(col, "ui.text"), justify="right")

    corr = _state["data"][["open", "high", "low", "close", "volume"]].corr()  # type: ignore[union-attr]
    for idx, row in corr.iterrows():
        corr_table.add_row(str(idx), *[f"{v:.3f}" for v in row])

    console.print(corr_table)


@register_handler("show")
def cmd_show(args: list[str]) -> None:
    """Show current data."""
    if _state["data"] is None:
        console.print("[warning]No data loaded.[/warning]")
        return
    print_data_summary(_state["data"], "Current Data")


@register_handler("train")
def cmd_train(args: list[str]) -> None:
    """Train model command."""
    if _state["data"] is None:
        console.print("[warning]No data loaded. Use 'fetch' first.[/warning]")
        return

    if not args:
        console.print("[error]Usage: train <model> [iterations][/error]")
        return

    model_name = args[0]
    if model_name not in MODEL_REGISTRY:
        console.print(
            f"[error]Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}[/error]"
        )
        return

    iterations = SETTINGS.default_iterations
    if len(args) > 1:
        parsed = _parse_int(args[1], "iterations")
        if parsed is None:
            return
        iterations = parsed

    with console.status(f"[status.running]Training {model_name}..."):
        model_instance = create_model(model_name)
        model_instance.fit(_state["data"], iterations=iterations)

    _state["model"] = model_instance
    _state["model_name"] = model_name

    default_path = str(SETTINGS.models_dir / f"{model_name.lower()}_model.joblib")
    model_instance.save(default_path)
    _state["model_path"] = default_path
    console.print(f"[success]{model_name} trained and saved to {default_path}[/success]")


@register_handler("load")
def cmd_load(args: list[str]) -> None:
    """Load model command."""
    if len(args) < _MIN_ARGS_LOAD:
        console.print("[error]Usage: load <model> [path][/error]")
        return

    model_name = args[0]
    default_path = f"{SETTINGS.models_dir}/{model_name.lower()}_model.joblib"
    path = args[1] if len(args) > 1 else default_path

    if model_name not in MODEL_REGISTRY:
        console.print(f"[error]Unknown model: {model_name}[/error]")
        return

    try:
        with console.status(f"[status.running]Loading {model_name} from {path}..."):
            model_class = MODEL_REGISTRY[model_name]
            model_instance = model_class.load(path)

        _state["model"] = model_instance
        _state["model_name"] = model_name
        _state["model_path"] = path
        console.print(f"[success]{model_name} loaded from {path}[/success]")
    except Exception as e:
        console.print(f"[error]Failed to load model: {e}[/error]")


@register_handler("save")
def cmd_save(args: list[str]) -> None:
    """Save model command."""
    if _state["model"] is None:
        console.print("[warning]No model trained/loaded.[/warning]")
        return

    model_name = _state["model_name"]
    if isinstance(model_name, str):
        default_path = str(SETTINGS.models_dir / f"{model_name.lower()}_model.joblib")
    else:
        default_path = str(SETTINGS.models_dir / "model.joblib")
    path = args[0] if args else default_path

    try:
        _state["model"].save(path)
        _state["model_path"] = path
        console.print(f"[success]Model saved to {path}[/success]")
    except Exception as e:
        console.print(f"[error]Failed to save model: {e}[/error]")


@register_handler("models")
def cmd_models(args: list[str]) -> None:
    """List available models."""
    table = Table(
        title="Available Models",
        title_style="panel.title",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    table.add_column("Name", style="brand")
    table.add_column("Class", style="success")
    table.add_column("Description", style="warning")

    descriptions = {
        "SARIMAX": "Seasonal ARIMA with exogenous variables (pmdarima)",
        "Prophet": "Facebook Prophet with custom seasonalities",
        "LSTM": "Long Short-Term Memory neural network (Keras/TF)",
    }

    for name, cls in MODEL_REGISTRY.items():
        table.add_row(name, cls.__name__, descriptions.get(name, ""))

    console.print(table)


@register_handler("predict")
def cmd_predict(args: list[str]) -> None:
    """Predict command."""
    if _state["model"] is None:
        console.print("[warning]No model loaded. Use 'train' or 'load' first.[/warning]")
        return

    periods = SETTINGS.default_prediction_periods
    if args:
        parsed = _parse_int(args[0], "periods")
        if parsed is None:
            return
        periods = parsed

    try:
        with console.status("[status.running]Generating predictions..."):
            if _state["model_name"] == "LSTM":
                if _state["data"] is None:
                    console.print("[error]LSTM requires data context. Fetch data first.[/error]")
                    return
                predictions = _state["model"].predict_with_context(_state["data"], periods)
            else:
                predictions = _state["model"].predict(periods)

        _state["predictions"] = predictions
        print_predictions_table(
            predictions, title=f"{_state['model_name']} Predictions ({periods} periods)"
        )

    except Exception as e:
        console.print(f"[error]Prediction failed: {e}[/error]")


@register_handler("compare")
def cmd_compare(args: list[str]) -> None:
    """Compare all models - fast, scored, progress, cancellable, timeout."""
    if _state["data"] is None:
        console.print("[warning]No data loaded. Use 'fetch' first.[/warning]")
        return

    periods, fast, holdout, timeout = _parse_compare_args(args)
    results = _run_compare_core(_state["data"], periods, fast, holdout, timeout)
    holdout_str = holdout if holdout is not None else periods
    title = (
        f"Model Comparison (periods={periods} holdout={holdout_str} {'fast' if fast else 'full'})"
    )
    print_model_comparison(results, title=title)
    console.print(
        "[ui.text_dim]Tip: compare --full --holdout 24 --timeout 300 "
        "for full sweep; compare --holdout 0 for unscored[/]"
    )


@register_handler("benchmark")
def cmd_benchmark(args: list[str]) -> None:
    """Train all models and score predictions on a held-out tail of the data."""
    if _state["data"] is None:
        console.print("[warning]No data loaded. Use 'fetch' first.[/warning]")
        return

    has_flag = any(a.startswith("--") for a in args)
    if has_flag:
        periods, fast, holdout, timeout = _parse_compare_args(args)
        if holdout is None:
            holdout = periods
        if "--fast" not in args and "--full" not in args:
            fast = False
        results = _run_compare_core(_state["data"], periods, fast, holdout, timeout)
        title = f"Benchmark (holdout={holdout} fast={fast})"
    else:
        holdout = SETTINGS.default_prediction_periods
        if args:
            parsed = _parse_int(args[0], "holdout")
            if parsed is None:
                return
            holdout = parsed
        if holdout < _MIN_HOLDOUT:
            console.print("[error]Holdout must be >= 2 periods.[/error]")
            return
        if len(_state["data"]) <= holdout:
            console.print(
                f"[error]Need more than {holdout} rows; have {len(_state['data'])}.[/error]"
            )
            return
        results = _run_compare_core(
            _state["data"], holdout, False, holdout, SETTINGS.compare_full_timeout_seconds
        )
        title = f"Benchmark (holdout={holdout})"

    print_model_comparison(results, title=title)


@register_handler("backtest")
def cmd_backtest(args: list[str]) -> None:
    """Backtest command."""
    if _state["model"] is None:
        console.print("[warning]No model loaded. Use 'train' or 'load' first.[/warning]")
        return

    if _state["data"] is None:
        console.print("[warning]No data loaded. Use 'fetch' first.[/warning]")
        return

    strategy = args[0] if args else "exit_after_n"
    exit_bars = SETTINGS.backtest_default_exit_bars
    if len(args) > 1:
        parsed = _parse_int(args[1], "exit bars")
        if parsed is None:
            return
        exit_bars = parsed
    lookback = SETTINGS.backtest_default_lookback
    if len(args) > _MIN_ARGS_BACKTEST_LOOKBACK - 1:
        parsed = _parse_int(args[2], "lookback")
        if parsed is None:
            return
        lookback = parsed

    test_data = _state["data"].iloc[-lookback:]

    with console.status("[status.running]Running backtest..."):
        result = run_backtest(
            _state["model"],
            test_data,
            strategy_name=strategy,
            exit_bars=exit_bars,
            lookback=lookback,
        )

    print_backtest_results(
        total_return=result.total_return,
        sharpe=result.sharpe_ratio,
        max_dd=result.max_drawdown,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        title="Backtest Result",
    )


@register_handler("strategies")
def cmd_strategies(args: list[str]) -> None:
    """List available strategies."""
    table = Table(
        title="Available Strategies",
        title_style="panel.title",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        header_style="table.header",
        row_styles=["table.row_even", "table.row_odd"],
    )
    table.add_column("Name", style="brand")
    table.add_column("Class", style="success")
    table.add_column("Description", style="warning")

    descriptions = {
        "exit_after_n": "Enter on prediction direction, exit after N bars",
        "exit_on_signal": "Enter on direction, exit when signal reverses",
    }

    for name, cls in STRATEGY_REGISTRY.items():
        table.add_row(name, cls.__name__, descriptions.get(name, ""))

    console.print(table)


def run_repl() -> None:
    """Run the interactive REPL loop."""
    console.print()
    console.print(
        Panel(
            "[brand]SARIMAX OHLCV Prediction REPL[/]\n"
            "[ui.text_dim]Type 'help' for commands, 'exit' to quit[/]",
            border_style="brand",
            padding=(0, 1),
        )
    )

    while True:
        try:
            line = Prompt.ask("[prompt]▸ repl[/]")
            cmd, args = parse_command(line)

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                console.print("[warning]Goodbye![/warning]")
                break

            handler = _CMD_HANDLERS.get(cmd)
            if handler:
                handler(args)
            else:
                console.print(f"[error]Unknown command: {cmd}. Type 'help' for commands.[/error]")

        except KeyboardInterrupt:
            console.print("\n[warning]Use 'exit' to quit[/warning]")
        except EOFError:
            console.print("\n[warning]Goodbye![/warning]")
            break
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")
