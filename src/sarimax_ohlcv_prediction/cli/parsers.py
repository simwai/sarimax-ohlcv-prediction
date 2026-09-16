"""Shared CLI parsers and validators."""


import typer

from ..data.modes import Mode, as_mode
from ..models import MODEL_REGISTRY, ModelBase


def _as_mode(mode: str) -> Mode:
    """Validate and cast mode string to Literal."""
    try:
        return as_mode(mode)
    except ValueError as exc:
        raise ValueError(f"Invalid mode: {mode}") from exc  # noqa: TRY003


def _model_class(model: str) -> type[ModelBase]:
    """Look up a model class by name with a clean CLI error."""
    if model not in MODEL_REGISTRY:
        raise typer.BadParameter(  # noqa: TRY003
            f"Model must be one of {', '.join(MODEL_REGISTRY.keys())}, got: {model}"
        )
    return MODEL_REGISTRY[model]
