"""Service-layer result types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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
