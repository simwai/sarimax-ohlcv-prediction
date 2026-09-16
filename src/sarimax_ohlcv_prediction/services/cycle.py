"""End-to-end prediction cycle service."""

import contextlib
import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config import SETTINGS
from ..data.fetcher import fetch_with_retry
from ..data.modes import as_mode
from ..data.processor import create_future_timestamps
from .artifacts import PredictionResult
from .prediction import make_predictions
from .training import train_model


def run_prediction_cycle(
    model_name: str,
    mode: Literal["current", "historical"],
    lookback: int,
    iterations: int,
    prediction_periods: int,
    save_csv: bool = True,
    csv_filename: str = SETTINGS.csv_filename,
) -> PredictionResult:
    """Run complete prediction cycle: fetch -> train -> predict -> save.

    Args:
        model_name: Model to use (SARIMAX, Prophet, LSTM)
        mode: Data mode ("current" or "historical")
        lookback: Lookback days
        iterations: Training iterations
        prediction_periods: Number of periods to predict
        save_csv: Whether to save predictions to CSV
        csv_filename: CSV filename

    Returns:
        PredictionResult with all outputs
    """
    data = fetch_with_retry(as_mode(mode), lookback)
    if data.empty:
        raise RuntimeError("Failed to fetch data")  # noqa: TRY003

    artifact = train_model(
        model_name=model_name,
        data=data,
        lookback=lookback,
        iterations=iterations,
    )

    predictions = make_predictions(artifact, prediction_periods, data)

    csv_path = None
    if save_csv:
        target = Path(csv_filename)
        anchor = pd.Timestamp(
            data["timestamp"].iloc[-1]
            if "timestamp" in data.columns
            else datetime.now(timezone.utc)
        )
        timestamps = create_future_timestamps(anchor, len(predictions))

        if target.exists() and target.stat().st_size > SETTINGS.csv_max_size_mb * 1024 * 1024:
            archive = target.with_name(
                f"{target.name}.archive_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
            )
            target.rename(archive)

        previous = target.read_text(encoding="utf-8") if target.exists() else None
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f"{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if previous:
                    f.write(previous)
                    if not previous.endswith("\n"):
                        f.write("\n")
                else:
                    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                for stamp, (_, row) in zip(timestamps, predictions.iterrows(), strict=True):
                    writer.writerow([
                        stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        row["open"], row["high"], row["low"], row["close"], row["volume"],
                    ])
            Path(tmp_name).replace(target)
            csv_path = target
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink()
            raise

    return PredictionResult(
        model_name=model_name,
        predictions=predictions,
        data=data,
        csv_path=csv_path,
    )
