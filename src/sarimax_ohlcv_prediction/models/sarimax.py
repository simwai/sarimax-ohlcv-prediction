"""SARIMAX model implementation using pmdarima.auto_arima."""

import logging
import time
import warnings
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pmdarima import auto_arima

from .base import ModelBase

logger = logging.getLogger(__name__)

_MODEL_NOT_FITTED = "Model not fitted. Call fit() first."

_SMALL_M_MAX_PQ = 2
_MEDIUM_M_MAX_PQ = 3
_LARGE_M_MAX_PQ = 5
_SMALL_M_THRESHOLD = 3
_LARGE_M_THRESHOLD = 8
_EARLY_STOP_NO_IMPROVE = 6


def _resolve_max_pq(m_range: range) -> int:
    """Auto-tune max_p/q based on m_range length."""
    size = len(m_range)
    if size <= _SMALL_M_THRESHOLD:
        return _SMALL_M_MAX_PQ
    if size <= _LARGE_M_THRESHOLD:
        return _MEDIUM_M_MAX_PQ
    return _LARGE_M_MAX_PQ


def _is_timed_out(start_time: float | None, timeout: float | None) -> bool:
    """Return True if a timeout was configured and has elapsed."""
    return (
        timeout is not None
        and start_time is not None
        and time.time() - start_time > timeout
    )


@dataclass
class _ColumnFitConfig:
    """Parameters for fitting a single column."""

    column: str
    column_data: pd.Series
    m_range: range
    max_p: int
    max_q: int
    seasonal: bool
    iterations: int
    progress_cb: Callable[[str, int, int], None] | None
    start_time: float | None
    timeout: float | None


class SARIMAXModel(ModelBase):
    """SARIMAX model with automatic parameter optimization."""

    def __init__(
        self,
        seasonal: bool = True,
        m_range: range = range(7, 50),
        iterations: int = 100,
        information_criterion: str = "aic",
    ) -> None:
        super().__init__()
        self.seasonal = seasonal
        self.m_range = m_range
        self.iterations = iterations
        self.information_criterion = information_criterion
        self.models: dict[str, Any] = {}
        self.best_m_values: dict[str, int] = {}

    def _fit_column(
        self,
        cfg: _ColumnFitConfig,
    ) -> tuple[Any, int]:
        """Fit one column across the m_range. Returns (best_model, best_m)."""
        best_model = None
        best_score = float("inf")
        best_m: int | None = None
        no_improve = 0
        done = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m in cfg.m_range:
                if _is_timed_out(cfg.start_time, cfg.timeout):
                    logger.warning(
                        "SARIMAX fit timeout after %.1fs at %s m=%d",
                        time.time() - (cfg.start_time or 0.0),
                        cfg.column,
                        m,
                    )
                    break
                done += 1
                if cfg.progress_cb is not None:
                    with suppress(Exception):
                        cfg.progress_cb(cfg.column, m, done)
                try:
                    model = auto_arima(
                        cfg.column_data,
                        start_p=1,
                        start_q=1,
                        max_p=cfg.max_p,
                        max_q=cfg.max_q,
                        m=m,
                        start_P=0,
                        seasonal=cfg.seasonal,
                        d=1,
                        D=1,
                        trace=False,
                        error_action="ignore",
                        suppress_warnings=True,
                        stepwise=True,
                        information_criterion=self.information_criterion,
                        maxiter=cfg.iterations,
                    )
                    score = getattr(model, f"{self.information_criterion}")()
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_m = m
                        no_improve = 0
                    else:
                        no_improve += 1
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.debug("Failed for m=%d: %s", m, exc)
                    no_improve += 1
                    continue

                if no_improve >= _EARLY_STOP_NO_IMPROVE and best_model is not None:
                    logger.info(
                        "Early stopping SARIMAX for %s at m=%d (no improve %d steps)",
                        cfg.column,
                        m,
                        _EARLY_STOP_NO_IMPROVE,
                    )
                    break

        return best_model, best_m if best_m is not None else 0

    def fit(self, data: pd.DataFrame, **kwargs) -> "SARIMAXModel":
        """Optimize and fit SARIMAX for each column."""
        self.models = {}
        self.best_m_values = {}

        iterations = kwargs.get("iterations", self.iterations)
        seasonal = kwargs.get("seasonal", self.seasonal)
        m_range = kwargs.get("m_range", self.m_range)
        max_p = kwargs.get("max_p", _resolve_max_pq(m_range))
        max_q = kwargs.get("max_q", _resolve_max_pq(m_range))
        progress_cb: Callable[[str, int, int], None] | None = kwargs.get("progress_callback")
        timeout: float | None = kwargs.get("timeout")
        start_time: float | None = kwargs.get("start_time")
        if start_time is None and timeout is not None:
            start_time = time.time()

        for column in self.columns:
            logger.info("Optimizing SARIMAX for %s", column)
            cfg = _ColumnFitConfig(
                column=column,
                column_data=data[column],
                m_range=m_range,
                max_p=max_p,
                max_q=max_q,
                seasonal=seasonal,
                iterations=iterations,
                progress_cb=progress_cb,
                start_time=start_time,
                timeout=timeout,
            )
            best_model, best_m = self._fit_column(cfg)
            if best_model is not None and best_m:
                self.models[column] = best_model
                self.best_m_values[column] = best_m
                logger.info(
                    "Best %s for %s: m=%d",
                    self.information_criterion.upper(),
                    column,
                    best_m,
                )
            else:
                logger.warning("No valid model found for %s", column)
            if _is_timed_out(start_time, timeout):
                logger.warning("SARIMAX fit overall timeout, stopping after %s", column)
                break

        self.is_fitted = True
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """Predict next `periods` steps for all columns."""
        if not self.is_fitted:
            raise RuntimeError(_MODEL_NOT_FITTED)

        predictions = {}
        for column in self.columns:
            if column in self.models:
                pred = self.models[column].predict(n_periods=periods)
                predictions[column] = pred
            else:
                predictions[column] = [float("nan")] * periods

        return pd.DataFrame(predictions)

    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "best_m_values": self.best_m_values,
                "columns": self.columns,
                "seasonal": self.seasonal,
                "m_range": self.m_range,
                "iterations": self.iterations,
            },
            path,
        )
        logger.info("SARIMAX model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SARIMAXModel":
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(
            seasonal=data.get("seasonal", True),
            m_range=data.get("m_range", range(7, 50)),
            iterations=data.get("iterations", 100),
        )
        model.models = data["models"]
        model.best_m_values = data["best_m_values"]
        model.columns = data["columns"]
        model.is_fitted = True
        logger.info("SARIMAX model loaded from %s", path)
        return model