"""Shared service layer for fetch, train, and predict operations.

Consolidates logic previously duplicated across CLI, Streamlit, and legacy scripts.
"""

from .artifacts import ModelArtifact, PredictionResult
from .cycle import run_prediction_cycle
from .prediction import make_predictions
from .training import load_model, train_model

__all__ = [
    "ModelArtifact",
    "PredictionResult",
    "train_model",
    "load_model",
    "make_predictions",
    "run_prediction_cycle",
]
