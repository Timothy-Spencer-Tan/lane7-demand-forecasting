"""
lane7_forecast.forecasting
==========================
Generates final monthly demand forecasts and writes gold_fact_forecasts.csv.

v5 fix: recursive multi-step ML forecasting
---------------------------------------------
The previous approach built all future feature rows at once from fixed
training history. This caused lags to collapse to NaN/0 after the first
forecast step, producing flat or erroneous long-range predictions.

The fix (_forecast_ml_recursive) works step-by-step per SKU:
  1. Build features for month t from running history (actuals + prior predictions)
  2. Predict month t
  3. Append prediction to running history
  4. Repeat for month t+1

Baseline models (SeasonalNaive, SeasonalAvg3Y, CrostonSBA) are unaffected —
they use their own internal logic and do not rely on lag features.

v6.2 additions: forecast_adjustments layer
--------------------------------------------
Four post-model corrections are applied per-SKU after predictions are produced:

  1. Intermittent cap        — prevents spike artefacts for INTERMITTENT SKUs
  2. Shrinkage               — reduces systematic overprediction bias
  3. ML vs Seasonal blend    — sanity-check against seasonal baseline for ML paths
  4. Recursive stabilization — reduces H=3 compounding drift via lag anchoring

All adjustments are controlled by FORECAST_ADJUSTMENT_CONFIG in
forecast_adjustments.py and can be disabled individually by setting the
relevant parameter to its no-op value (see forecast_adjustments.py docstring).
Pass adjustment_config=None to generate_forecasts to bypass all adjustments
(reverts to v6.1 behaviour).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"

FORECAST_SCHEMA_COLS = [
    "RunDate", "MonthStart", "Level", "Key",
    "ModelName", "HorizonMonths", "ForecastUnits",
    "Lower", "Upper", "ModelVersion",
]


# ---------------------------------------------------------------------------
# Future feature construction
# ---------------------------------------------------------------------------

def build_future_features(
    panel: pd.DataFrame,
    horizon_months: int,
    forecast_months: list[pd.Timestamp],
    label_encoders: dict | None = None,
) -> pd.DataFrame:
    """
    Thin wrapper kept for API compatibility.

    In v5 the recursive ML forecasting path does NOT call this function —
    it calls _build_one_step_features() per step inside _forecast_ml_recursive().
    This function still works for single-step or non-recursive callers (e.g.
    simulation runs that need only the first future month's features).
    """
    from lane7_forecast.features import ALL_LAG_PERIODS, ROLLING_WINDOWS, ROLLING_STD_WINDOWS

    all_rows: list[dict] = []
    skus = panel[SKU_COL].unique()

    for sku in skus:
        sku_panel = panel[panel[SKU_COL] == sku].sort_values(DATE_COL)
        sku_hist  = sku_panel.set_index(DATE_COL)[TARGET_COL].sort_index()
        if len(sku_hist) == 0:
            continue
        last_row = sku_panel.iloc[-1]
        n_hist   = len(sku_hist)

        for fmonth_idx, fmonth in enumerate(forecast_months):
            row = _build_one_step_features(
                fmonth=fmonth,
                sku_hist=sku_hist,
                last_row=last_row,
                n_hist=n_hist,
                fmonth_idx=fmonth_idx,
                lag_periods=ALL_LAG_PERIODS,
                rolling_windows_mean=ROLLING_WINDOWS,
                rolling_windows_std=ROLLING_STD_WINDOWS,
            )
            row[SKU_COL]    = sku
            row[DATE_COL]   = fmonth
            row[TARGET_COL] = 0.0
            row["IsTrain"]  = 0
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    future_df = pd.DataFrame(all_rows)
    _encode_future_categoricals(future_df, label_encoders)
    logger.info(
        "build_future_features: %d rows (%d SKUs x %d months)",
        len(future_df), len(skus), len(forecast_months),
    )
    return future_df


def _build_one_step_features(
    fmonth: pd.Timestamp,
    sku_hist: pd.Series,
    last_row: pd.Series,
    n_hist: int,
    fmonth_idx: int,
    lag_periods: list[int],
    rolling_windows_mean: list[int],
    rolling_windows_std: list[int],
) -> dict:
    """
    Compute ALL features for a SINGLE forecast month from a history series.

    Called once per forecast step inside _forecast_ml_recursive(). After each
    step, the predicted value is appended to sku_hist before this function is
    called again for the next step. This guarantees that lag_1 for step 2 is
    the prediction from step 1, not NaN.

    Parameters
    ----------
    fmonth               : the forecast month being computed
    sku_hist             : pd.Series indexed by MonthStart — GROWS after each step
    last_row             : most recent panel row (for SKU attributes)
    n_hist               : number of training months (fixed; used for trend_index)
    fmonth_idx           : 0-based position within the forecast horizon
    lag_periods          : list of lag distances to compute
    rolling_windows_mean : windows for rolling_mean_N
    rolling_windows_std  : windows for rolling_std_N

    Returns
    -------
    dict with all feature values (no SKU/date/target columns — caller adds those)
    """
    row: dict = {}

    # ── Lag features ────────────────────────────────────────────────────────
    # sku_hist already contains predicted values from previous steps, so lags
    # automatically pick up those predictions after step 1.
    for lag in lag_periods:
        offset = fmonth - pd.DateOffset(months=lag)
        lag_ts = pd.Timestamp(offset.year, offset.month, 1)
        row[f"lag_{lag}"] = float(sku_hist.get(lag_ts, np.nan))

    # ── Rolling mean / std ───────────────────────────────────────────────────
    # Window ends at fmonth - 1 month (no leakage).
    # Months within the window that are forecast steps (not actuals) are already
    # in sku_hist because we append after each prediction, so they are included
    # naturally.
    end_off  = fmonth - pd.DateOffset(months=1)
    end_ts   = pd.Timestamp(end_off.year, end_off.month, 1)

    for window in rolling_windows_mean:
        start_off = end_off - pd.DateOffset(months=window - 1)
        start_ts  = pd.Timestamp(start_off.year, start_off.month, 1)
        vals = sku_hist[(sku_hist.index >= start_ts) & (sku_hist.index <= end_ts)]
        min_p = max(1, window // 2)
        row[f"rolling_mean_{window}"] = float(vals.mean()) if len(vals) >= min_p else np.nan

    for window in rolling_windows_std:
        start_off = end_off - pd.DateOffset(months=window - 1)
        start_ts  = pd.Timestamp(start_off.year, start_off.month, 1)
        vals = sku_hist[(sku_hist.index >= start_ts) & (sku_hist.index <= end_ts)]
        min_p = max(1, window // 2)
        if len(vals) >= min_p:
            row[f"rolling_std_{window}"] = float(vals.std()) if len(vals) > 1 else 0.0
        else:
            row[f"rolling_std_{window}"] = np.nan

    # ── Momentum ─────────────────────────────────────────────────────────────
    l1 = row.get("lag_1", np.nan)
    l3 = row.get("lag_3", np.nan)
    if pd.notna(l1) and pd.notna(l3):
        row["momentum_3"] = float(np.clip((l1 - l3) / (l3 + 1.0), -5.0, 5.0))
    else:
        row["momentum_3"] = np.nan

    # ── YoY growth ───────────────────────────────────────────────────────────
    l12 = row.get("lag_12", np.nan)
    l24 = row.get("lag_24", np.nan)
    if pd.notna(l12) and pd.notna(l24) and l24 != 0:
        row["yoy_growth"] = float(np.clip(l12 / l24, 0.1, 10.0))
    else:
        row["yoy_growth"] = np.nan

    # ── Calendar ─────────────────────────────────────────────────────────────
    row["Month"]     = fmonth.month
    row["Quarter"]   = (fmonth.month - 1) // 3 + 1
    row["Year"]      = fmonth.year
    row["month_sin"] = float(np.sin(2 * np.pi * fmonth.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * fmonth.month / 12))

    # ── Trend index ──────────────────────────────────────────────────────────
    # n_hist is fixed (training history length); fmonth_idx advances per step
    row["trend_index"] = n_hist + fmonth_idx

    # ── SKU attributes (pass-through from last training row) ─────────────────
    for col in ["Category", "StyleCode", "ColorCode", "SizeCode", "StyleColor", "Segment"]:
        if hasattr(last_row, col) or (hasattr(last_row, '__contains__') and col in last_row):
            row[col] = last_row.get(col, np.nan) if hasattr(last_row, 'get') else getattr(last_row, col, np.nan)

    return row


def _encode_future_categoricals(
    future_df: pd.DataFrame,
    label_encoders: dict | None,
) -> None:
    """Apply categorical encoding in-place, matching the training feature space."""
    for col in ["Category", "StyleCode", "ColorCode"]:
        if col not in future_df.columns:
            continue
        enc_col = f"{col}_enc"
        le = (label_encoders or {}).get(col)
        if le is not None:
            known = set(le.classes_)
            future_df[enc_col] = future_df[col].fillna("UNKNOWN").astype(str).apply(
                lambda v: int(le.transform([v])[0]) if v in known else -1
            )
        else:
            future_df[enc_col] = (
                future_df[col].fillna("UNKNOWN").astype("category").cat.codes.astype(int)
            )


def _forecast_ml_recursive(
    fitted_model: Any,
    sku: str,
    sku_hist: pd.Series,
    last_row: pd.Series,
    n_hist: int,
    forecast_months: list[pd.Timestamp],
    feature_cols: list[str],
    label_encoders: dict,
    max_mom_growth: float | None = None,
    adjustment_config: dict | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """
    Recursive step-by-step ML forecasting for a single SKU.

    For each forecast month:
      1. Build features from sku_hist (which already contains previous predictions)
      2. Predict next value
      3. Apply growth guardrail (existing v5 logic)
      4. v6.2: Apply recursive stabilization — blend predicted lag with trailing
         mean before appending to running_hist (reduces H=3 compounding drift)
      5. Append stabilized value to running_hist
      6. Proceed to next month

    Growth guardrail (max_mom_growth)
    ----------------------------------
    Caps MoM growth per step. Applied BEFORE the stabilization blend so the
    guardrail still constrains explosive single-step growth.
      H=3  → max_mom_growth=0.15
      H!=3 → max_mom_growth=0.25
      None → disabled

    Recursive stabilization (v6.2)
    --------------------------------
    After the guardrail, the value appended to running_hist is blended:
        hist_val = alpha × pred + (1 − alpha) × trailing_mean_3
    This anchors subsequent lag features to recent trend rather than
    compounding the model's own over-prediction. The REPORTED prediction
    (yhat_list) uses the un-blended pred — only the history feed is blended.
    Set recursive_alpha=1.0 in adjustment_config to disable (pure v5 behaviour).

    Parameters
    ----------
    fitted_model      : trained sklearn-compatible estimator
    sku               : SKU identifier (for logging)
    sku_hist          : pd.Series indexed by MonthStart — MUTABLE, grows per step
    last_row          : most recent panel row (SKU attributes)
    n_hist            : number of training months (fixed)
    forecast_months   : list of pd.Timestamp forecast targets
    feature_cols      : ordered list of feature column names matching training
    label_encoders    : categorical encoder dict from feature_panel.attrs
    max_mom_growth    : max allowed MoM growth fraction per step, or None to disable
    adjustment_config : v6.2 adjustment config from forecast_adjustments.get_config()
                        Pass None to use defaults.

    Returns
    -------
    (yhat, lower, upper) — three lists of length n_forecast_months
    yhat   : raw (guardrail-capped) per-step predictions — used for output
    lower  : ±20% placeholder (replaced downstream by scenario bounds)
    upper  : ±20% placeholder
    """
    from lane7_forecast.features import ALL_LAG_PERIODS, ROLLING_WINDOWS, ROLLING_STD_WINDOWS
    from lane7_forecast.models import predict_ml
    from lane7_forecast.forecast_adjustments import (
        get_config as adj_get_config,
        stabilize_lag_for_recursion,
    )

    cfg          = adj_get_config(**(adjustment_config or {})) if adjustment_config is not None else adj_get_config()
    yhat_list    = []
    running_hist = sku_hist.copy()

    # Reference value for guardrail: last actual, updates to prediction each step
    prev_value = float(sku_hist.iloc[-1]) if len(sku_hist) > 0 else None

    for fmonth_idx, fmonth in enumerate(forecast_months):
        row = _build_one_step_features(
            fmonth=fmonth,
            sku_hist=running_hist,
            last_row=last_row,
            n_hist=n_hist,
            fmonth_idx=fmonth_idx,
            lag_periods=ALL_LAG_PERIODS,
            rolling_windows_mean=ROLLING_WINDOWS,
            rolling_windows_std=ROLLING_STD_WINDOWS,
        )

        row_df = pd.DataFrame([row])
        _encode_future_categoricals(row_df, label_encoders)

        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0.0

        X_step = row_df[feature_cols].fillna(0)

        try:
            raw_pred = float(predict_ml(fitted_model, X_step)[0])
        except Exception as exc:
            logger.warning(
                "Recursive ML predict failed at step %d for SKU %s: %s — using 0",
                fmonth_idx, sku, exc,
            )
            raw_pred = 0.0

        pred = max(0.0, raw_pred)

        # ── Growth guardrail (v5 logic, unchanged) ────────────────────────────
        # Caps MoM growth to suppress compounding errors in recursive steps.
        # Only applied when prev_value > 0 (no cap when base demand is zero).
        if max_mom_growth is not None and prev_value is not None and prev_value > 0:
            cap = prev_value * (1.0 + max_mom_growth)
            if pred > cap:
                logger.debug(
                    "Guardrail triggered: SKU=%s step=%d raw=%.1f → capped=%.1f "
                    "(prev=%.1f × (1 + %.2f))",
                    sku, fmonth_idx, pred, cap, prev_value, max_mom_growth,
                )
                pred = cap

        # pred is the reported prediction for this step
        yhat_list.append(pred)
        prev_value = pred  # guardrail reference updates to (guardrail-capped) pred

        # ── v6.2: Recursive stabilization — blend what we feed back into history
        # The REPORTED yhat uses pred (guardrail-capped model output).
        # The HISTORY feed uses a blend anchored to trailing mean so that lag_1
        # for the next step is not purely the model's potentially inflated output.
        hist_val = stabilize_lag_for_recursion(
            predicted_val=pred,
            sku_hist_so_far=running_hist,
            step_index=fmonth_idx,
            config=cfg,
        )

        # ── Append to running history ─────────────────────────────────────────
        running_hist = pd.concat([
            running_hist,
            pd.Series([hist_val], index=[fmonth]),
        ]).sort_index()

    # Lower/Upper are placeholder ±20% bands.
    # Use build_scenario_bounds_from_simulation() for calibrated planning bounds.
    lower = [v * 0.80 for v in yhat_list]
    upper = [v * 1.20 for v in yhat_list]
    return yhat_list, lower, upper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_forecast_rows(
    sku: str,
    model_name: str,
    horizon_months: int,
    forecast_months: list[pd.Timestamp],
    yhat: list[float],
    lower: list[float],
    upper: list[float],
    model_version: str,
    run_date: date,
) -> list[dict]:
    rows = []
    for i, month in enumerate(forecast_months):
        rows.append({
            "RunDate":       run_date,
            "MonthStart":    month,
            "Level":         "SKU",
            "Key":           sku,
            "ModelName":     model_name,
            "HorizonMonths": horizon_months,
            "ForecastUnits": round(max(0.0, yhat[i]), 4),
            "Lower":         round(max(0.0, lower[i]), 4),
            "Upper":         round(max(0.0, upper[i]), 4),
            "ModelVersion":  model_version,
        })
    return rows


def _get_best_model_for_segment(
    segment: str,
    horizon_months: int,
    best_models_df: pd.DataFrame,
) -> str:
    mask  = (best_models_df["Segment"] == segment) & \
            (best_models_df["HorizonMonths"] == horizon_months)
    match = best_models_df[mask]
    if match.empty:
        logger.warning(
            "No best model for segment=%s, horizon=%d -- using SeasonalNaive.",
            segment, horizon_months,
        )
        return "SeasonalNaive"
    return str(match.iloc[0]["BestModel"])


# ---------------------------------------------------------------------------
# Public: generate_forecasts
# ---------------------------------------------------------------------------

def generate_forecasts(
    panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str | pd.Timestamp,
    n_forecast_months: int = 24,
    phase: int = 1,
    model_version: str = "v1.0",
    ml_models_fitted: dict | None = None,
    adjustment_config: dict | None = None,
) -> pd.DataFrame:
    """
    Generate demand forecasts for every SKU for n_forecast_months months.

    ML models use _forecast_ml_recursive() for step-by-step lag-aware forecasting.
    Statistical models (Prophet, NeuralProphet, SARIMA) are fit per-SKU.
    Baseline models are pure functions over the SKU history series.

    v6.2 changes
    ------------
    A new optional parameter `adjustment_config` passes the v6.2 correction
    settings through to the per-SKU prediction paths:

    - Intermittent cap   (INTERMITTENT segment, all model types)
    - Shrinkage          (REGULAR + INTERMITTENT, after all model types)
    - ML/Seasonal blend  (REGULAR + INTERMITTENT ML paths only)
    - Recursive stab.    (ML paths, inside _forecast_ml_recursive)

    Pass adjustment_config=None to bypass all v6.2 adjustments entirely
    (this is the v6.1 / v5 fallback path).

    Parameters
    ----------
    panel              : full panel with Segment attached
    feature_panel      : training feature panel (used for column-name reference only)
    best_models_df     : output of select_best_model()
    horizon_months     : 1, 3, or 12
    forecast_start     : first forecast month (e.g. "2026-01-01")
    n_forecast_months  : months to generate (default 24)
    phase              : 1 or 2 (logged only)
    model_version      : tag written into ModelVersion column
    ml_models_fitted   : {segment_name: fitted_sklearn_model}
    adjustment_config  : v6.2 adjustment config dict from
                         forecast_adjustments.get_config(), or None to disable.
    """
    from lane7_forecast.models import (
        predict_ml, predict_baseline,
        fit_predict_prophet, fit_predict_neuralprophet, fit_predict_sarima,
    )
    from lane7_forecast.features import get_feature_columns
    from lane7_forecast.segmentation import SEG_DEAD

    # v6.2 adjustment imports — only activated when adjustment_config is not None
    _adj_enabled = adjustment_config is not None
    if _adj_enabled:
        from lane7_forecast.forecast_adjustments import (
            get_config as adj_get_config,
            apply_shrinkage,
            apply_intermittent_cap,
            blend_ml_with_seasonal,
        )
        _cfg = adj_get_config(**adjustment_config)
    else:
        _cfg = {}

    ML_MODELS            = {"XGBoost", "LightGBM", "RandomForest"}
    PROPHET_MODELS       = {"Prophet"}
    NEURALPROPHET_MODELS = {"NeuralProphet"}
    SARIMA_MODELS        = {"SARIMA"}

    forecast_start  = pd.Timestamp(forecast_start)
    forecast_months = [
        forecast_start + pd.DateOffset(months=i)
        for i in range(n_forecast_months)
    ]
    run_date     = date.today()
    feature_cols = get_feature_columns(horizon_months, feature_panel)
    skus         = panel[SKU_COL].unique()

    logger.info(
        "Generating %d-month forecasts for %d SKUs, %d months from %s "
        "(adjustments=%s)",
        horizon_months, len(skus), n_forecast_months,
        forecast_start.strftime("%Y-%m"),
        "enabled" if _adj_enabled else "disabled",
    )

    _label_encoders = feature_panel.attrs.get("label_encoders", {})
    future_feat_df  = pd.DataFrame()   # kept for API compatibility (non-ML paths)

    all_rows: list[dict] = []

    for sku in skus:
        sku_panel = panel[panel[SKU_COL] == sku].sort_values(DATE_COL)
        segment   = (
            str(sku_panel["Segment"].iloc[0])
            if "Segment" in sku_panel.columns else "REGULAR"
        )

        # DEAD -- zero forecast, no adjustments
        if segment == SEG_DEAD:
            all_rows.extend(_make_forecast_rows(
                sku=sku, model_name="ZeroForecast",
                horizon_months=horizon_months,
                forecast_months=forecast_months,
                yhat=[0.0] * n_forecast_months,
                lower=[0.0] * n_forecast_months,
                upper=[0.0] * n_forecast_months,
                model_version=model_version, run_date=run_date,
            ))
            continue

        model_name  = _get_best_model_for_segment(segment, horizon_months, best_models_df)
        sku_history = sku_panel.set_index(DATE_COL)[TARGET_COL].sort_index()

        # ---- ML (recursive, step-by-step) ------------------------------------
        if model_name in ML_MODELS:
            fitted_model = (ml_models_fitted or {}).get(segment)
            if fitted_model is None:
                # No trained model available — fall back to SeasonalNaive
                preds = predict_baseline("SeasonalNaive", sku_history, n_forecast_months)
                yhat, lower, upper = preds, [v * 0.80 for v in preds], [v * 1.20 for v in preds]
                model_name = "SeasonalNaive"
            else:
                last_row = sku_panel.iloc[-1]
                n_hist   = len(sku_history)
                # Guardrail: 15% for H=3 (3-month planning), 25% for other horizons.
                _guardrail = 0.15 if horizon_months == 3 else 0.25
                try:
                    yhat, lower, upper = _forecast_ml_recursive(
                        fitted_model=fitted_model,
                        sku=sku,
                        sku_hist=sku_history,
                        last_row=last_row,
                        n_hist=n_hist,
                        forecast_months=forecast_months,
                        feature_cols=feature_cols,
                        label_encoders=_label_encoders,
                        max_mom_growth=_guardrail,
                        # v6.2: pass adjustment config for recursive stabilization
                        adjustment_config=adjustment_config,
                    )
                except Exception as exc:
                    logger.warning("Recursive ML failed for SKU %s: %s -- using SeasonalNaive", sku, exc)
                    preds = predict_baseline("SeasonalNaive", sku_history, n_forecast_months)
                    yhat, lower, upper = preds, [v * 0.80 for v in preds], [v * 1.20 for v in preds]
                    model_name = "SeasonalNaive"

                # ── v6.2: ML/Seasonal blend ────────────────────────────────────
                # After ML forecast is produced, compute a seasonal baseline and
                # blend when ML diverges substantially from seasonal reality.
                if _adj_enabled and model_name in ML_MODELS:
                    try:
                        seasonal_preds = predict_baseline(
                            "SeasonalAvg3Y", sku_history, n_forecast_months
                        )
                        yhat = blend_ml_with_seasonal(yhat, seasonal_preds, _cfg)
                    except Exception as exc:
                        logger.debug("ML/Seasonal blend skipped for SKU %s: %s", sku, exc)

        # ---- Prophet ----------------------------------------------------------
        elif model_name in PROPHET_MODELS:
            try:
                yhat, lower, upper = fit_predict_prophet(sku_history, n_forecast_months)
            except Exception as exc:
                logger.warning("Prophet failed for SKU %s: %s -- using SeasonalNaive", sku, exc)
                preds = predict_baseline("SeasonalNaive", sku_history, n_forecast_months)
                yhat, lower, upper = preds, [v * 0.80 for v in preds], [v * 1.20 for v in preds]
                model_name = "SeasonalNaive"

        # ---- NeuralProphet ----------------------------------------------------
        elif model_name in NEURALPROPHET_MODELS:
            try:
                yhat, lower, upper = fit_predict_neuralprophet(sku_history, n_forecast_months)
            except Exception as exc:
                logger.warning("NeuralProphet failed for SKU %s: %s -- using SeasonalNaive", sku, exc)
                preds = predict_baseline("SeasonalNaive", sku_history, n_forecast_months)
                yhat, lower, upper = preds, [v * 0.80 for v in preds], [v * 1.20 for v in preds]
                model_name = "SeasonalNaive"

        # ---- SARIMA -----------------------------------------------------------
        elif model_name in SARIMA_MODELS:
            try:
                yhat, lower, upper = fit_predict_sarima(sku_history, n_forecast_months)
            except Exception as exc:
                logger.warning("SARIMA failed for SKU %s: %s -- using SeasonalNaive", sku, exc)
                preds = predict_baseline("SeasonalNaive", sku_history, n_forecast_months)
                yhat, lower, upper = preds, [v * 0.80 for v in preds], [v * 1.20 for v in preds]
                model_name = "SeasonalNaive"

        # ---- Baselines --------------------------------------------------------
        else:
            preds  = predict_baseline(model_name, sku_history, n_forecast_months)
            yhat   = preds
            lower  = [v * 0.80 for v in yhat]
            upper  = [v * 1.20 for v in yhat]

        # ── v6.2: Per-segment post-model corrections ──────────────────────────
        if _adj_enabled:
            # 1. Intermittent cap — INTERMITTENT only, before shrinkage
            if segment == "INTERMITTENT":
                yhat = apply_intermittent_cap(yhat, sku_history, _cfg)

            # 2. Shrinkage — all non-DEAD segments
            yhat = apply_shrinkage(yhat, sku_history, segment, _cfg)

            # Recompute lower/upper bounds from adjusted yhat
            lower = [v * 0.80 for v in yhat]
            upper = [v * 1.20 for v in yhat]

        all_rows.extend(_make_forecast_rows(
            sku=sku, model_name=model_name,
            horizon_months=horizon_months,
            forecast_months=forecast_months,
            yhat=yhat[:n_forecast_months],
            lower=lower[:n_forecast_months],
            upper=upper[:n_forecast_months],
            model_version=model_version, run_date=run_date,
        ))

    forecasts_df = pd.DataFrame(all_rows)[FORECAST_SCHEMA_COLS]
    logger.info("Forecast complete: %d rows, horizon=%d", len(forecasts_df), horizon_months)
    return forecasts_df


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_forecasts(forecasts_df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecasts_df.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(forecasts_df), path)


def append_forecasts(
    new_df: pd.DataFrame,
    existing_path: str | Path,
) -> pd.DataFrame:
    path = Path(existing_path)
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["RunDate", "MonthStart"])
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    dedup_keys = ["RunDate", "Key", "MonthStart", "HorizonMonths", "ModelName"]
    combined   = combined.drop_duplicates(subset=dedup_keys, keep="last")
    combined.to_csv(path, index=False)
    logger.info("Appended to %s -- total rows: %d", path, len(combined))
    return combined
