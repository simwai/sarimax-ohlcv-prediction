"""Pydantic models for data validation."""

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

_HIGH_GTE_LOW_MSG = "high must be >= low"
_LOW_LTE_HIGH_MSG = "low must be <= high"
_MONOTONIC_MSG = "Timestamps not monotonic at index {i}"


class OHLCVRow(BaseModel):
    """Single OHLCV row from exchange API."""

    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., ge=0, description="Volume")

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info: Any) -> float:
        if "low" in info.data and v < info.data["low"]:
            raise ValueError(_HIGH_GTE_LOW_MSG)
        return v

    @field_validator("low")
    @classmethod
    def low_lte_high(cls, v: float, info: Any) -> float:
        if "high" in info.data and v > info.data["high"]:
            raise ValueError(_LOW_LTE_HIGH_MSG)
        return v


class OHLCVResponse(BaseModel):
    """Validated OHLCV response from Binance."""

    symbol: str = Field(..., pattern=r"^[A-Z]+/[A-Z]+$", description="Trading pair symbol")
    timeframe: Literal[
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    ]
    data: list[OHLCVRow] = Field(..., min_length=1, description="OHLCV rows")

    @field_validator("data")
    @classmethod
    def timestamps_monotonic(cls, v: list[OHLCVRow]) -> list[OHLCVRow]:
        """Ensure timestamps are strictly increasing."""
        for i in range(1, len(v)):
            if v[i].timestamp <= v[i - 1].timestamp:
                raise ValueError(_MONOTONIC_MSG.format(i=i))
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with datetime index."""
        df = pd.DataFrame([row.model_dump() for row in self.data])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]]
