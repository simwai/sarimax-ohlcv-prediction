"""Prediction helper service."""


import pandas as pd

from ..config import SETTINGS
from .artifacts import ModelArtifact


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
