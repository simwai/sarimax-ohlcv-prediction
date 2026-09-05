"""Prophet model implementation."""

import contextlib
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from .base import ModelBase

logger = logging.getLogger(__name__)

_MODEL_NOT_FITTED = "Model not fitted. Call fit() first."
_BAD_ENVELOPE_MSG = "Unrecognized Prophet model file: {path}"
_REQUIRED_KEYS = ("models", "columns")
_PERIODS_MSG = "forecast periods must be an int, got {value!r}"


class ProphetModel(ModelBase):
    """Prophet model with custom seasonalities for 5-minute data."""

    def __init__(
        self,
        daily_seasonality: bool = False,
        yearly_seasonality: bool = False,
        daily_period: float = 24 * 60 / 5,  # 288 periods per day
        daily_fourier: int = 8,
        weekly_period: float = 7 * 24 * 60 / 5,  # 2016 periods per week
        weekly_fourier: int = 3,
    ) -> None:
        super().__init__()
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_period = daily_period
        self.daily_fourier = daily_fourier
        self.weekly_period = weekly_period
        self.weekly_fourier = weekly_fourier
        self.models: dict[str, Prophet] = {}
        self.forecasts: dict[str, pd.DataFrame] = {}

    def fit(self, data: pd.DataFrame, **kwargs) -> "ProphetModel":
        """Fit Prophet model for each column."""
        self._store_training_data(data)
        self.models = {}
        self.forecasts = {}

        raw_periods = kwargs.get("forecast_periods", kwargs.get("iterations", 100))
        if not isinstance(raw_periods, int):
            raise TypeError(_PERIODS_MSG.format(value=raw_periods))  # noqa: TRY003
        forecast_periods = raw_periods
        progress_cb = kwargs.get("progress_callback")

        for idx, column in enumerate(self.columns):
            logger.info("Fitting Prophet for %s", column)
            if progress_cb is not None:
                with contextlib.suppress(Exception):
                    progress_cb(column, idx, idx + 1)

            df = data[["timestamp", column]].rename(columns={"timestamp": "ds", column: "y"})

            model = (
                Prophet(
                    daily_seasonality=self.daily_seasonality,
                    yearly_seasonality=self.yearly_seasonality,
                )
                .add_seasonality(
                    name="daily",
                    period=self.daily_period,
                    fourier_order=self.daily_fourier,
                )
                .add_seasonality(
                    name="weekly",
                    period=self.weekly_period,
                    fourier_order=self.weekly_fourier,
                )
            )

            model.fit(df)
            self.models[column] = model

            # Generate forecast for later use
            future = model.make_future_dataframe(periods=forecast_periods, freq="5min")
            forecast = model.predict(future)
            self.forecasts[column] = forecast

        self.is_fitted = True
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict next `periods` steps for all columns."""
        if not self.is_fitted:
            raise RuntimeError(_MODEL_NOT_FITTED)

        predictions: dict[str, Any] = {}  # pyrefly: ignore -- forecast arrays
        for column in self.columns:
            if column in self.models:
                model = self.models[column]
                future = model.make_future_dataframe(periods=periods, freq="5min")
                forecast = model.predict(future)
                predictions[column] = forecast["yhat"].iloc[-periods:].to_numpy()
            else:
                predictions[column] = np.full(periods, float("nan"))

        return pd.DataFrame(predictions)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": "prophet-v1",
                "models": self.models,
                "forecasts": self.forecasts,
                "columns": self.columns,
                "daily_seasonality": self.daily_seasonality,
                "yearly_seasonality": self.yearly_seasonality,
                "daily_period": self.daily_period,
                "daily_fourier": self.daily_fourier,
                "weekly_period": self.weekly_period,
                "weekly_fourier": self.weekly_fourier,
            },
            path,
        )
        logger.info("Prophet model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ProphetModel":
        """Load model from disk, rejecting foreign envelopes."""
        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("format", "prophet-v0") not in (
            "prophet-v0",
            "prophet-v1",
        ):
            raise ValueError(_BAD_ENVELOPE_MSG.format(path=path))  # noqa: TRY003
        if any(key not in data for key in _REQUIRED_KEYS):
            raise ValueError(_BAD_ENVELOPE_MSG.format(path=path))  # noqa: TRY003
        model = cls(
            daily_seasonality=data.get("daily_seasonality", False),
            yearly_seasonality=data.get("yearly_seasonality", False),
            daily_period=data.get("daily_period", 288),
            daily_fourier=data.get("daily_fourier", 8),
            weekly_period=data.get("weekly_period", 2016),
            weekly_fourier=data.get("weekly_fourier", 3),
        )
        model.models = data["models"]
        model.forecasts = data.get("forecasts", {})
        model.columns = data["columns"]
        model.is_fitted = True
        logger.info("Prophet model loaded from %s", path)
        return model
