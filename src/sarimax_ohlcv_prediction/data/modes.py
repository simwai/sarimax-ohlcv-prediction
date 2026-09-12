"""Shared data-mode validation."""

from typing import Literal, cast

Mode = Literal["current", "historical"]


def as_mode(mode: str) -> Mode:
    """Validate a data-mode string, raising ValueError on mismatch."""
    if mode not in ("current", "historical"):
        raise ValueError(f"Mode must be 'current' or 'historical', got: {mode}")  # noqa: TRY003
    return cast(Mode, mode)
