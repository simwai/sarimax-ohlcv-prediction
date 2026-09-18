"""Core compare/benchmark logic extracted from REPL for reuse by CLI commands."""

import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Group
from rich.live import Live
from rich.progress import Progress as ProgressBar
from rich.progress import TaskID
from rich.text import Text

from ..config import SETTINGS
from ..models import MODEL_REGISTRY, create_model
from ..viz.rich import console

_MIN_HOLDOUT = 2


@dataclass(frozen=True)
class FitKwargsParams:
    """Parameters for assembling model.fit() kwargs."""

    name: str
    cb: Callable[[str, int, int], None] | None
    timeout: int | None
    sarimax_m_range: range
    sarimax_iterations: int
    lstm_epochs: int


@dataclass(frozen=True)
class ScoreParams:
    """Parameters for scoring predictions."""

    model_instance: Any  # pyrefly: ignore -- heterogeneous model types
    name: str
    predictions: pd.DataFrame
    actual_close: np.ndarray | None
    elapsed_str: str
    pending_logs: list[tuple[str, str]]


@dataclass(frozen=True)
class TrainPredictParams:
    """Parameters for training and predicting one model."""

    name: str
    train_df: pd.DataFrame
    predict_periods: int
    progress: ProgressBar
    task_ids: dict[str, TaskID]
    scored: bool
    holdout: int | None
    periods: int
    timeout: int | None
    mode_label: str
    status_text: Text
    pending_logs: list[tuple[str, str]]
    actual_close: np.ndarray | None
    sarimax_m_range: range
    sarimax_iterations: int
    lstm_epochs: int


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE between actual and predicted series."""
    err = actual - predicted
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    nonzero = np.where(actual != 0, actual, np.nan)
    mape = float(np.nanmean(np.abs(err / nonzero)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


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
            console.print("[error]--holdout requires value[/error]")
            return holdout, idx
        try:
            return int(args[idx + 1]), idx + 1
        except ValueError:
            console.print(f"[error]Invalid --holdout value: {args[idx + 1]}[/error]")
            return holdout, idx + 1
    try:
        return int(a.split("=", 1)[1]), idx
    except ValueError:
        console.print(f"[error]Invalid --holdout value: {a}[/error]")
        return holdout, idx


def _parse_timeout_flag(
    args: list[str], idx: int, timeout: int | None
) -> tuple[int | None, int, bool]:
    """Parse --timeout <n> or --timeout=<n>. Returns (new_timeout, new_idx, explicit_set)."""
    a = args[idx]
    if a == "--timeout":
        if idx + 1 >= len(args):
            console.print("[error]--timeout requires value[/error]")
            return timeout, idx, False
        try:
            return int(args[idx + 1]), idx + 1, True
        except ValueError:
            console.print(f"[error]Invalid --timeout value: {args[idx + 1]}[/error]")
            return timeout, idx + 1, False
    try:
        return int(a.split("=", 1)[1]), idx, True
    except ValueError:
        console.print(f"[error]Invalid --timeout value: {a}[/error]")
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
            console.print(f"[warning]Unknown flag: {a}[/warning]")
        else:
            remaining.append(a)
        i += 1
    if remaining:
        try:
            periods = int(remaining[0])
        except ValueError:
            console.print(f"[error]Invalid periods: {remaining[0]}[/error]")
        if len(remaining) > 1 and holdout is None:
            with contextlib.suppress(ValueError):
                holdout = int(remaining[1])
    if not timeout_explicit:
        timeout = (
            SETTINGS.compare_timeout_seconds if fast else SETTINGS.compare_full_timeout_seconds
        )
    return periods, fast, holdout, timeout


def _resolve_compare_params(
    data: pd.DataFrame,
    periods: int,
    fast: bool,
    holdout: int | None,
) -> tuple[int, pd.DataFrame, np.ndarray | None, bool, int]:
    """Resolve final holdout/scored/train_df/actual_close/predict_periods."""
    if holdout is None:
        holdout = periods if len(data) > periods + 10 else 0
    if holdout is not None and holdout != 0 and len(data) <= holdout:
        console.print(
            f"[warning]Not enough data for holdout={holdout}, disabling scoring[/warning]"
        )
        holdout = 0
    if holdout is not None and holdout != 0 and holdout < _MIN_HOLDOUT:
        console.print("[error]Holdout must be >=2, disabling scoring[/error]")
        holdout = 0

    scored = holdout is not None and holdout != 0
    if scored:
        assert holdout is not None
        train_df = data.iloc[:-holdout]
        test_df = data.iloc[-holdout:]
        actual_close: np.ndarray | None = test_df["close"].to_numpy(
            dtype=float
        )  # pyrefly: ignore[bad-assignment]
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
    progress: ProgressBar, model_name: str, task_id: TaskID, total: int
) -> Callable[[str, int, int], None]:
    """Build a progress callback bound to (progress, model_name, task_id, total)."""

    def _cb(col_or_epoch: str, val: int, done: int) -> None:
        try:
            if model_name == "SARIMAX":
                progress.update(
                    task_id,
                    completed=done,
                    description=f"[brand]{model_name} {col_or_epoch} m={val}[/]",
                )
            elif model_name == "Prophet":
                progress.update(
                    task_id, completed=done, description=f"[brand]{model_name} {col_or_epoch}[/]"
                )
            elif model_name == "LSTM":
                progress.update(
                    task_id,
                    completed=val,
                    description=f"[brand]{model_name} epoch {val}/{total}[/]",
                )
        except Exception:
            pass

    return _cb


def _build_fit_kwargs(
    params: FitKwargsParams,
) -> dict[str, Any]:  # pyrefly: ignore -- heterogeneous fit kwargs
    """Assemble kwargs for model.fit()."""
    fit_kwargs: dict = {"progress_callback": params.cb}
    if params.timeout is not None and params.timeout > 0:
        fit_kwargs["timeout"] = float(params.timeout)
        fit_kwargs["start_time"] = time.time()
    if params.name == "SARIMAX":
        fit_kwargs["m_range"] = params.sarimax_m_range
        fit_kwargs["iterations"] = params.sarimax_iterations
    elif params.name == "LSTM":
        fit_kwargs["epochs"] = params.lstm_epochs
    return fit_kwargs


def _format_model_info(
    model_instance: Any,  # pyrefly: ignore -- heterogeneous model types
    name: str,
) -> str:
    """Format compact info string for table."""
    try:
        if name == "SARIMAX" and hasattr(model_instance, "best_m_values"):
            bm = getattr(model_instance, "best_m_values", {})
            if bm:
                return "best_m=" + ",".join(f"{k}:{v}" for k, v in bm.items())
            return "no best_m"
        if name == "Prophet" and hasattr(model_instance, "daily_period"):
            return f"daily={model_instance.daily_period} weekly={model_instance.weekly_period}"
        if name == "LSTM" and hasattr(model_instance, "device"):
            return f"epochs={model_instance.epochs} device={model_instance.device}"
        return type(model_instance).__name__
    except Exception:
        return "N/A"


def _score_predictions(params: ScoreParams) -> dict[str, str]:
    """Score predictions against actual_close. Returns row dict for results table."""
    predicted_close = params.predictions["close"].to_numpy(dtype=float)
    if params.actual_close is None or len(predicted_close) == 0:
        info_str = _format_model_info(params.model_instance, params.name)
        params.pending_logs.append(
            (
                "success",
                f"{params.name}: Done ({params.elapsed_str}) - {info_str}",
            )
        )
        return {
            "status": "Success",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info_str,
            "elapsed": params.elapsed_str,
        }

    if len(predicted_close) != len(params.actual_close):
        min_len = min(len(predicted_close), len(params.actual_close))
        predicted_close = predicted_close[:min_len]
        actual_close_trim: np.ndarray = params.actual_close[:min_len]
    else:
        actual_close_trim = params.actual_close

    if np.isnan(predicted_close).any():
        info = _format_model_info(params.model_instance, params.name) + " (incomplete)"
        params.pending_logs.append(
            (
                "warning",
                f"{params.name}: Timeout/partial - {params.elapsed_str} - {info}",
            )
        )
        return {
            "status": "Timeout (partial)",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info,
            "elapsed": params.elapsed_str,
        }

    metrics = _compute_metrics(
        actual_close_trim, predicted_close
    )  # pyrefly: ignore[bad-argument-type]
    info = _format_model_info(params.model_instance, params.name)
    if any(np.isnan(v) for v in metrics.values()):
        params.pending_logs.append(
            (
                "warning",
                f"{params.name}: Partial - no valid close prediction ({params.elapsed_str})",
            )
        )
        return {
            "status": "Partial",
            "rmse": "N/A",
            "mae": "N/A",
            "mape": "N/A",
            "info": info,
            "elapsed": params.elapsed_str,
        }
    params.pending_logs.append(
        (
            "success",
            (
                f"{params.name}: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} "
                f"MAPE={metrics['mape']:.2f}% ({params.elapsed_str})"
            ),
        )
    )
    return {
        "status": "Success",
        "rmse": f"{metrics['rmse']:.4f}",
        "mae": f"{metrics['mae']:.4f}",
        "mape": f"{metrics['mape']:.2f}%",
        "info": info,
        "elapsed": params.elapsed_str,
    }


def _handle_interrupt(
    name: str, start: float, pending_logs: list[tuple[str, str]]
) -> dict[str, str]:
    """Build interrupted-row dict for results."""
    elapsed = time.time() - start
    pending_logs.append(("warning", f"{name}: Interrupted ({elapsed:.1f}s)"))
    return {
        "status": "Interrupted",
        "rmse": "N/A",
        "mae": "N/A",
        "mape": "N/A",
        "info": "user interrupt",
        "elapsed": f"{elapsed:.1f}s",
    }


def _handle_failure(
    name: str, start: float, exc: Exception, pending_logs: list[tuple[str, str]]
) -> dict[str, str]:
    """Build failed-row dict for results."""
    elapsed = time.time() - start
    pending_logs.append(("error", f"{name}: Failed - {exc} ({elapsed:.1f}s)"))
    return {
        "status": f"Failed: {exc}",
        "rmse": "N/A",
        "mae": "N/A",
        "mape": "N/A",
        "info": "N/A",
        "elapsed": f"{elapsed:.1f}s",
    }


def _train_and_predict_one(params: TrainPredictParams) -> tuple[str, dict[str, str], float]:
    """Train and predict one model. Returns (name, result_row, elapsed_seconds)."""
    start = time.time()
    total = int(params.progress.tasks[params.task_ids[params.name]].total or 1)
    if params.scored:
        params.status_text.plain = (
            f"train {len(params.train_df)} · holdout {params.holdout} · "
            f"{params.mode_label} · {params.timeout}s | {params.name} training..."
        )
    else:
        params.status_text.plain = (
            f"train {len(params.train_df)} · predict {params.periods} · "
            f"{params.mode_label} | {params.name} training..."
        )
    params.progress.start_task(params.task_ids[params.name])
    task_id = params.task_ids[params.name]
    params.progress.update(task_id, description=f"[brand]{params.name} training...[/]")

    try:
        model_instance = create_model(params.name)
        cb = _make_progress_callback(params.progress, params.name, task_id, total)
        fit_kwargs = _build_fit_kwargs(
            FitKwargsParams(
                name=params.name,
                cb=cb,
                timeout=params.timeout,
                sarimax_m_range=params.sarimax_m_range,
                sarimax_iterations=params.sarimax_iterations,
                lstm_epochs=params.lstm_epochs,
            )
        )
        try:
            model_instance.fit(params.train_df, **fit_kwargs)
        except KeyboardInterrupt:
            params.pending_logs.append(
                (
                    "warning",
                    f"{params.name}: interrupted by user, keeping best so far",
                )
            )
        params.progress.update(
            task_id, completed=total, description=f"[success]{params.name} training done[/]"
        )

        params.status_text.plain = (
            f"train {len(params.train_df)} · holdout {params.holdout} · "
            f"{params.mode_label} | {params.name} predicting..."
        )
        if params.name == "LSTM":
            predictions = model_instance.predict_with_context(
                params.train_df, params.predict_periods
            )
        else:
            predictions = model_instance.predict(params.predict_periods)

        elapsed = time.time() - start
        elapsed_str = f"{elapsed:.1f}s"
        if params.scored:
            row = _score_predictions(
                ScoreParams(
                    model_instance=model_instance,
                    name=params.name,
                    predictions=predictions,
                    actual_close=params.actual_close,
                    elapsed_str=elapsed_str,
                    pending_logs=params.pending_logs,
                )
            )
        else:
            row = _score_predictions(
                ScoreParams(
                    model_instance=model_instance,
                    name=params.name,
                    predictions=predictions,
                    actual_close=None,
                    elapsed_str=elapsed_str,
                    pending_logs=params.pending_logs,
                )
            )
    except Exception as exc:
        params.progress.stop_task(params.task_ids[params.name])
        return (
            params.name,
            _handle_failure(params.name, start, exc, params.pending_logs),
            time.time() - start,
        )
    else:
        return params.name, row, elapsed


def _run_compare_core(
    data: pd.DataFrame,
    periods: int,
    fast: bool,
    holdout: int | None,
    timeout: int | None,
) -> dict[str, dict]:
    """Core compare logic with dual-area Live (status + progress bar)."""
    holdout, train_df, actual_close, scored, predict_periods = _resolve_compare_params(
        data, periods, fast, holdout
    )
    sarimax_m_range, sarimax_iterations, lstm_epochs = _resolve_mode_params(fast)

    mode_label = "fast" if fast else "full"
    if scored:
        header_plain = (
            f"train {len(train_df)} · holdout {holdout} · {mode_label} · {timeout}s timeout"
        )
    else:
        header_plain = (
            f"train {len(train_df)} · predict {periods} · {mode_label} · {timeout}s timeout"
        )
    status_text = Text(header_plain, style="ui.text_dim")

    progress = ProgressBar()
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
                    TrainPredictParams(
                        name=name,
                        train_df=train_df,
                        predict_periods=predict_periods,
                        progress=progress,
                        task_ids=task_ids,
                        scored=scored,
                        holdout=holdout,
                        periods=periods,
                        timeout=timeout,
                        mode_label=mode_label,
                        status_text=status_text,
                        pending_logs=pending_logs,
                        actual_close=actual_close,
                        sarimax_m_range=sarimax_m_range,
                        sarimax_iterations=sarimax_iterations,
                        lstm_epochs=lstm_epochs,
                    )
                )
                results[name] = row
            status_text.plain = header_plain + " · done"
    finally:
        _restore_model_loggers(prev_log_levels)

    console.print(f"[ui.text_dim]{header_plain}[/]")
    for style, msg in pending_logs:
        console.print(f"[{style}]{msg}[/]")

    return results
