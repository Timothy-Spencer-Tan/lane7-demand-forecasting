"""
lane7_forecast.models
=====================
Trains one model per (segment, horizon) combination, not one model per SKU.

Model roster by segment and horizon
------------------------------------
                    H=1 (short)          H=3 (medium)         H=12 (long)
  REGULAR       XGBoost, LightGBM,    XGBoost, LightGBM,   Prophet,
                RandomForest, SARIMA, Prophet, NeuralProphet, NeuralProphet,
                Prophet, MA_3         SeasonalNaive         SeasonalAvg3Y,
                                                             SARIMA

  INTERMITTENT  SeasonalNaive,        SeasonalAvg3Y,        SeasonalAvg3Y,
                Croston               SeasonalNaive          SeasonalNaive

  DEAD          ZeroForecast          ZeroForecast           ZeroForecast

Design notes
------------
- ML models (XGBoost, LightGBM, RandomForest) receive the feature matrix from
  create_features() and are trained on the TRAIN split rows across ALL SKUs in
  the segment simultaneously (one global model per segment).
- Statistical models (SARIMA, Prophet, NeuralProphet) are fit per-SKU individually
  at inference/evaluation time because they are inherently single-series models.
  This module provides wrappers that make them look like sklearn estimators.
- Baseline models (MA, SeasonalNaive, SeasonalAvg3Y, Croston) are pure functions
  with no fit step — they compute predictions from the panel directly.

Public API
----------
    SEGMENT_MODEL_ROSTER  : dict mapping (segment, horizon) -> list[str]
    get_ml_feature_cols(horizon, panel) -> list[str]
    train_ml_model(model_name, X_train, y_train, **kwargs) -> fitted model
    predict_ml(model, X) -> np.ndarray
    predict_baseline(model_name, panel, sku, horizon_months) -> float | list[float]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"

# ---------------------------------------------------------------------------
# Model roster — segment × horizon -> candidate model names
# ---------------------------------------------------------------------------

SEGMENT_MODEL_ROSTER: dict[tuple[str, int], list[str]] = {
    # --- REGULAR ---
    # v5: focused on 3-month planning. Pruned Prophet, NeuralProphet, SARIMA, RandomForest, MA_3.
    # Rationale:
    #   Prophet/NeuralProphet: high runtime, inconsistent at SKU level, beat by LightGBM in CV
    #   SARIMA: per-SKU fitting is slow; RandomForest: no meaningful edge over XGB/LGB in panel CV
    #   MA_3: dominated by SeasonalNaive (which respects seasonality); kept as fallback only
    ("REGULAR", 1): [
        "XGBoost", "LightGBM",
        "SeasonalNaive",         # seasonal baseline: same month last year
    ],
    ("REGULAR", 3): [
        "XGBoost", "LightGBM",
        "SeasonalNaive",         # same 3 months last year
        "SeasonalAvg3Y",         # mean of same month over 3 prior years
    ],
    # H=12 kept minimal for fallback only — not part of v5 primary workflow
    ("REGULAR", 12): [
        "SeasonalAvg3Y",         # seasonal 3-year average; kept for backwards compatibility
    ],
    # --- INTERMITTENT ---
    # v5: removed raw Croston (SBA strictly dominates it with 10% bias reduction)
    ("INTERMITTENT", 1):  ["CrostonSBA", "SeasonalNaive"],
    ("INTERMITTENT", 3):  ["CrostonSBA", "SeasonalAvg3Y"],
    ("INTERMITTENT", 12): ["SeasonalAvg3Y", "SeasonalNaive"],
    # --- DEAD ---
    ("DEAD", 1):  ["ZeroForecast"],
    ("DEAD", 3):  ["ZeroForecast"],
    ("DEAD", 12): ["ZeroForecast"],
}

# ---------------------------------------------------------------------------
# ML model factory
# ---------------------------------------------------------------------------

def train_ml_model(
    model_name: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    **kwargs: Any,
) -> Any:
    """
    Fit an ML model and return the trained estimator.

    Supported model_name values: "XGBoost", "LightGBM", "RandomForest"
    Extra kwargs are forwarded to the underlying estimator constructor.

    The model is trained on all REGULAR SKUs in the training split simultaneously —
    this is a global panel model, not a per-SKU model.
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost not installed. Run: pip install xgboost") from e
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        model = XGBRegressor(**defaults)

    elif model_name_lower == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm") from e
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(kwargs)
        model = LGBMRegressor(**defaults)

    elif model_name_lower == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        defaults = dict(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        model = RandomForestRegressor(**defaults)

    else:
        raise ValueError(
            f"Unknown ML model: {model_name!r}. "
            f"Choose from: XGBoost, LightGBM, RandomForest"
        )

    model.fit(X_train, y_train)

    logger.info("Trained %s on %d rows, %d features", model_name, len(y_train),
                X_train.shape[1] if hasattr(X_train, "shape") else "?")
    return model


def predict_ml(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Generate predictions from an ML model, clipped to >= 0."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


# ---------------------------------------------------------------------------
# Baseline model predictors (stateless, pure-function style)
# ---------------------------------------------------------------------------

def predict_baseline(
    model_name: str,
    sku_history: pd.Series,   # indexed by MonthStart, sorted ascending
    horizon_months: int,
    target_dates: list[pd.Timestamp] | None = None,
) -> list[float]:
    """
    Compute baseline forecasts for a single SKU.

    Parameters
    ----------
    model_name      : one of "MA_3", "MA_12", "SeasonalNaive",
                      "SeasonalAvg3Y", "Croston", "CrostonSBA", "ZeroForecast"
    sku_history     : pd.Series of UnitsSold indexed by MonthStart (TRAIN period only)
    horizon_months  : 1, 3, or 12 — number of future months to forecast
    target_dates    : optional explicit list of target MonthStart values;
                      if None, forecast months are inferred from the last history date

    Returns
    -------
    list[float] of length horizon_months (non-negative)
    """
    model_name_lower = model_name.lower()

    if model_name_lower == "zeroforecast":
        return [0.0] * horizon_months

    if sku_history is None or len(sku_history) == 0:
        return [0.0] * horizon_months

    history = sku_history.sort_index()

    # ---- Moving Average ----
    if model_name_lower in ("ma_3", "ma_12"):
        window = 3 if model_name_lower == "ma_3" else 12
        ma_val = float(history.tail(window).mean())
        return [max(0.0, ma_val)] * horizon_months

    # ---- Seasonal Naive: same month last year ----
    if model_name_lower == "seasonalnaive":
        last_date = history.index.max()
        forecasts = []
        for h in range(1, horizon_months + 1):
            target   = last_date + pd.DateOffset(months=h)
            lookback = target - pd.DateOffset(years=1)
            if lookback in history.index:
                val = float(history[lookback])
            else:
                # Fall back to last observed same-month value
                same_month = history[history.index.month == target.month]
                val = float(same_month.iloc[-1]) if len(same_month) else 0.0
            forecasts.append(max(0.0, val))
        return forecasts

    # ---- Seasonal Average 3Y: mean of same month over past 3 years ----
    if model_name_lower == "seasonalavg3y":
        last_date = history.index.max()
        forecasts = []
        for h in range(1, horizon_months + 1):
            target     = last_date + pd.DateOffset(months=h)
            target_mon = target.month
            lookback_vals = []
            for yr_back in [1, 2, 3]:
                lkb = pd.Timestamp(target.year - yr_back, target_mon, 1)
                if lkb in history.index:
                    lookback_vals.append(float(history[lkb]))
            val = float(np.mean(lookback_vals)) if lookback_vals else 0.0
            forecasts.append(max(0.0, val))
        return forecasts

    # ---- Croston's method (intermittent demand) ----
    if model_name_lower == "croston":
        vals = history.values.astype(float)
        alpha = 0.2
        if len(vals) == 0:
            return [0.0] * horizon_months

        # Find demand intervals
        demand_periods = [(i, v) for i, v in enumerate(vals) if v > 0]
        if not demand_periods:
            return [0.0] * horizon_months

        # Smooth demand size and intervals separately
        smoothed_demand   = demand_periods[0][1]
        smoothed_interval = 1.0

        for i in range(1, len(demand_periods)):
            idx_curr = demand_periods[i][0]
            idx_prev = demand_periods[i - 1][0]
            interval = idx_curr - idx_prev
            d        = demand_periods[i][1]
            smoothed_demand   = alpha * d        + (1 - alpha) * smoothed_demand
            smoothed_interval = alpha * interval + (1 - alpha) * smoothed_interval

        croston_rate = smoothed_demand / smoothed_interval if smoothed_interval > 0 else 0.0
        return [max(0.0, croston_rate)] * horizon_months

    # ---- Croston-SBA (Syntetos-Boylan Approximation) ----
    # Bias-corrected Croston: multiplies the demand estimate by (1 - alpha/2).
    # SBA reduces the positive bias inherent in standard Croston's method and
    # consistently outperforms it on intermittent retail demand series.
    # Reference: Syntetos & Boylan (2005), International Journal of Forecasting.
    if model_name_lower == "crostonsha" or model_name_lower == "croston_sba" or model_name_lower == "croston sba" or model_name_lower == "crostonsba":
        vals = history.values.astype(float)
        alpha = 0.2
        if len(vals) == 0:
            return [0.0] * horizon_months

        demand_periods = [(i, v) for i, v in enumerate(vals) if v > 0]
        if not demand_periods:
            return [0.0] * horizon_months

        smoothed_demand   = demand_periods[0][1]
        smoothed_interval = 1.0

        for i in range(1, len(demand_periods)):
            idx_curr = demand_periods[i][0]
            idx_prev = demand_periods[i - 1][0]
            interval = idx_curr - idx_prev
            d        = demand_periods[i][1]
            smoothed_demand   = alpha * d        + (1 - alpha) * smoothed_demand
            smoothed_interval = alpha * interval + (1 - alpha) * smoothed_interval

        # SBA bias correction: multiply demand estimate by (1 - alpha/2)
        sba_correction  = 1.0 - (alpha / 2.0)
        croston_rate    = smoothed_demand / smoothed_interval if smoothed_interval > 0 else 0.0
        sba_rate        = croston_rate * sba_correction
        return [max(0.0, sba_rate)] * horizon_months

    raise ValueError(f"Unknown baseline model: {model_name!r}")


# ---------------------------------------------------------------------------
# Prophet / NeuralProphet wrapper (per-SKU, called during evaluation)
# ---------------------------------------------------------------------------

def fit_predict_prophet(
    sku_history: pd.Series,
    horizon_months: int,
    country_holidays: str | None = "US",
) -> tuple[list[float], list[float], list[float]]:
    """
    Fit Prophet on a single SKU's history and return (forecast, lower, upper).

    Returns three lists of length horizon_months.
    Requires: pip install prophet
    """
    try:
        from prophet import Prophet
    except ImportError as e:
        raise ImportError("prophet not installed. Run: pip install prophet") from e

    if len(sku_history) < 12:
        logger.debug("Prophet skipped — fewer than 12 months of history")
        fallback = [0.0] * horizon_months
        return fallback, fallback, fallback

    df_prophet = pd.DataFrame({
        "ds": sku_history.index,
        "y":  sku_history.values.astype(float),
    }).reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon_months, freq="MS")
    forecast = model.predict(future).tail(horizon_months)

    yhat  = np.clip(forecast["yhat"].values,       0, None).tolist()
    lower = np.clip(forecast["yhat_lower"].values,  0, None).tolist()
    upper = np.clip(forecast["yhat_upper"].values,  0, None).tolist()
    return yhat, lower, upper


def fit_predict_neuralprophet(
    sku_history: pd.Series,
    horizon_months: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Fit NeuralProphet on a single SKU's history.
    Requires: pip install neuralprophet

    Returns (forecast, lower, upper) — NeuralProphet does not natively produce
    prediction intervals so lower/upper are set to ±20% of forecast.
    """
    try:
        from neuralprophet import NeuralProphet
    except ImportError as e:
        raise ImportError("neuralprophet not installed. Run: pip install neuralprophet") from e

    if len(sku_history) < 12:
        fallback = [0.0] * horizon_months
        return fallback, fallback, fallback

    df_np = pd.DataFrame({
        "ds": sku_history.index,
        "y":  sku_history.values.astype(float),
    }).reset_index(drop=True)

    import logging as _logging
    _logging.getLogger("NP.forecaster").setLevel(_logging.ERROR)
    _logging.getLogger("NP.utils").setLevel(_logging.ERROR)

    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_forecasts=horizon_months,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
    )
    model.fit(df_np, freq="MS", progress=None)

    future  = model.make_future_dataframe(df_np, n_historic_predictions=False)
    forecast_df = model.predict(future)

    # NeuralProphet multi-step: columns yhat1..yhatN
    yhat_cols = [f"yhat{i}" for i in range(1, horizon_months + 1)
                 if f"yhat{i}" in forecast_df.columns]
    if yhat_cols:
        preds = forecast_df[yhat_cols].iloc[-1].values
    else:
        preds = np.zeros(horizon_months)

    preds = np.clip(preds, 0, None)
    lower = (preds * 0.80).tolist()
    upper = (preds * 1.20).tolist()
    return preds.tolist(), lower, upper


def fit_predict_sarima(
    sku_history: pd.Series,
    horizon_months: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Fit SARIMA(1,0,1)(1,1,1)_12 on a single SKU's history.
    Requires: pip install statsmodels

    Returns (forecast, lower, upper).
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise ImportError("statsmodels not installed. Run: pip install statsmodels") from e

    if len(sku_history) < 24:
        fallback = [0.0] * horizon_months
        return fallback, fallback, fallback

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            sku_history.values.astype(float),
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

    fc = result.get_forecast(steps=horizon_months)
    yhat  = np.clip(fc.predicted_mean,                    0, None).tolist()
    lower = np.clip(fc.conf_int(alpha=0.2).iloc[:, 0],   0, None).tolist()
    upper = np.clip(fc.conf_int(alpha=0.2).iloc[:, 1],   0, None).tolist()
    return yhat, lower, upper
