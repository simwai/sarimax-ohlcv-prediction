"""Model training and loading helpers."""

from pathlib import Path

import pandas as pd

from ..config import SETTINGS
from ..models import MODEL_REGISTRY, create_model
from .artifacts import ModelArtifact


def train_model(
    model_name: str,
    data: pd.DataFrame,
    lookback: int | None = None,
    iterations: int | None = None,
    save_path: str | Path | None = None,
) -> ModelArtifact:
    """Train a model on the provided data.

    Args:
        model_name: Model type (SARIMAX, Prophet, LSTM)
        data: Training data
        lookback: Lookback days (default from SETTINGS)
        iterations: Training iterations (default from SETTINGS)
        save_path: Optional explicit save path

    Returns:
        ModelArtifact with trained model and metadata
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(  # noqa: TRY003
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    lookback = lookback or SETTINGS.default_lookback_days
    iterations = iterations or SETTINGS.default_iterations

    model = create_model(model_name)
    model.fit(data, iterations=iterations)

    saved_path = None
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        saved_path = path
    else:
        default_path = Path(SETTINGS.models_dir) / f"{model_name.lower()}_model.joblib"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(default_path))
        saved_path = default_path

    return ModelArtifact(
        model_name=model_name,
        model=model,
        data=data,
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
