"""Shared service layer for fetch, train, and predict operations.

Consolidates logic previously duplicated across CLI, Streamlit, and legacy scripts.
"""

import contextlib
import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..cli.compare_core import _run_compare_core
from ..config import SETTINGS
from ..data.fetcher import fetch_with_retry
from ..data.modes import as_mode
from ..data.processor import create_future_timestamps
from ..models import MODEL_REGISTRY, create_model


@dataclass(frozen=True)
class ModelKeyParams:  # noqa: PLR0913 -- dataclass replaces long parameter list
    """Identity and training parameters for a model cache key."""

    model_name: str
    symbol: str
    timeframe: str
    lookback: int
    iterations: int
    data_hash: str


@dataclass(frozen=True)
class TrainModelParams:  # noqa: PLR0913 -- dataclass replaces long parameter list
    """Parameters for training a model."""

    model_name: str
    data: pd.DataFrame
    lookback: int | None = None
    iterations: int | None = None
    save_path: str | Path | None = None
    use_cache: bool = True


@dataclass(frozen=True)
class RunPredictionCycleParams:  # noqa: PLR0913 -- dataclass replaces long parameter list
    """Parameters for running a complete prediction cycle."""

    model_name: str
    mode: Literal["current", "historical"]
    lookback: int
    iterations: int
    prediction_periods: int
    save_csv: bool = True
    csv_filename: str = SETTINGS.csv_filename


@dataclass
class PredictionResult:
    """Result of a prediction cycle."""
    model_name: str
    predictions: pd.DataFrame
    data: pd.DataFrame
    csv_path: Path | None = None


@dataclass
class ModelArtifact:
    """Trained model with metadata."""
    model_name: str
    model: Any
    data: pd.DataFrame | None
    lookback: int
    iterations: int
    saved_path: Path | None = None


def fetch_data(
    mode: Literal["current", "historical"],
    lookback: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance.

    Args:
        mode: "current" for last 24h, "historical" for lookback days
        lookback: Number of days to look back (for historical mode)
        use_cache: Whether to use cache (default True)

    Returns:
        DataFrame with OHLCV data
    """
    return fetch_with_retry(as_mode(mode), lookback, use_cache=use_cache)


def train_model(params: TrainModelParams) -> ModelArtifact:
    """Train a model on the provided data.

    Args:
        params: Bundled training parameters.

    Returns:
        ModelArtifact with trained model and metadata
    """
    if params.model_name not in MODEL_REGISTRY:
        raise ValueError(  # noqa: TRY003
            f"Unknown model: {params.model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    lookback = params.lookback or SETTINGS.default_lookback_days
    iterations = params.iterations or SETTINGS.default_iterations

    model = create_model(params.model_name)
    model.fit(params.data, iterations=iterations)

    saved_path = None
    if params.save_path:
        path = Path(params.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        saved_path = path
    else:
        default_path = Path(SETTINGS.models_dir) / f"{params.model_name.lower()}_model.joblib"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(default_path))
        saved_path = default_path

    return ModelArtifact(
        model_name=params.model_name,
        model=model,
        data=params.data,
        lookback=lookback,
        iterations=iterations,
        saved_path=saved_path,
    )


def load_model(
    model_name: str,
    path: str | Path,
) -> ModelArtifact:
    """Load a trained model from file.

    Args:
        model_name: Model type (SARIMAX, Prophet, LSTM)
        path: Path to model file

    Returns:
        ModelArtifact with loaded model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")  # noqa: TRY003

    model_class = MODEL_REGISTRY[model_name]
    model = model_class.load(str(path))

    return ModelArtifact(
        model_name=model_name,
        model=model,
        data=None,
        lookback=0,
        iterations=0,
        saved_path=Path(path),
    )


def make_predictions(
    model_artifact: ModelArtifact,
    periods: int | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions using a trained model.

    Args:
        model_artifact: Trained model with metadata
        periods: Number of periods to predict
        data: Context data (required for LSTM)

    Returns:
        DataFrame with predictions
    """
    periods = periods or SETTINGS.default_prediction_periods
    model = model_artifact.model

    if model_artifact.model_name == "LSTM":
        context = data if data is not None else model_artifact.data
        if context is None:
            raise ValueError(  # noqa: TRY003
                "LSTM requires data context. Provide data or train with data."
            )
        return model.predict_with_context(context, periods)

    return model.predict(periods)


def run_prediction_cycle(params: RunPredictionCycleParams) -> PredictionResult:
    """Run complete prediction cycle: fetch -> train -> predict -> save.

    Args:
        params: Bundled cycle parameters.

    Returns:
        PredictionResult with all outputs
    """
    # Fetch data
    data = fetch_data(params.mode, params.lookback)
    if data.empty:
        raise RuntimeError("Failed to fetch data")  # noqa: TRY003

    # Train model
    artifact = train_model(
        TrainModelParams(
            model_name=params.model_name,
            data=data,
            lookback=params.lookback,
            iterations=params.iterations,
        )
    )

    # Predict
    predictions = make_predictions(artifact, params.prediction_periods, data)

    # Save CSV
    csv_path = None
    if params.save_csv:
        target = Path(params.csv_filename)
        anchor = pd.Timestamp(
            data["timestamp"].iloc[-1] if "timestamp" in data.columns else datetime.now(UTC)
        )
        timestamps = create_future_timestamps(anchor, len(predictions))

        if target.exists() and target.stat().st_size > SETTINGS.csv_max_size_mb * 1024 * 1024:
            archive = target.with_name(
                f"{target.name}.archive_{datetime.now(UTC):%Y%m%d%H%M%S}"
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
        model_name=params.model_name,
        predictions=predictions,
        data=data,
        csv_path=csv_path,
    )


def compare_models(
    data: pd.DataFrame,
    periods: int,
    fast: bool = True,
    holdout: int | None = None,
    timeout: int | None = None,
) -> dict[str, dict]:
    """Compare all models on the same data.

    Args:
        data: Training/test data
        periods: Prediction periods / holdout
        fast: Fast preset vs full
        holdout: Holdout for scoring (None=auto, 0=disable)
        timeout: Timeout per model in seconds

    Returns:
        Dict of model results
    """
    return _run_compare_core(data, periods, fast, holdout, timeout)
