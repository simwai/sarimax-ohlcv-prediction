"""Models package - exports all model implementations."""

from .base import BaseModel, ModelBase
from .lstm import LSTMModel
from .prophet import ProphetModel
from .sarimax import SARIMAXModel

__all__ = [
    "BaseModel",
    "ModelBase",
    "SARIMAXModel",
    "ProphetModel",
    "LSTMModel",
]

# Model registry for CLI/REPL
MODEL_REGISTRY = {
    "SARIMAX": SARIMAXModel,
    "Prophet": ProphetModel,
    "LSTM": LSTMModel,
}

_UNKNOWN_MODEL_MSG = "Unknown model: {name}. Available: {available}"


def get_model_class(name: str) -> type:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(_UNKNOWN_MODEL_MSG.format(name=name, available=list(MODEL_REGISTRY.keys())))
    return MODEL_REGISTRY[name]


def create_model(name: str, **kwargs) -> BaseModel:
    """Create model instance by name."""
    return get_model_class(name)(**kwargs)