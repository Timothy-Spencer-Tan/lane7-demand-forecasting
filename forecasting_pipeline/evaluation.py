"""
lane7_forecast.evaluation
==========================
Walk-forward cross-validation and metric computation.

Evaluation strategy
-------------------
Phase 1 only evaluates within the TRAIN period (2017-05 to 2024-12) using an
expanding window.  The 2025 holdout is reserved for final Phase 1 reporting.

Walk-forward window logic
-------------------------
For each fold, the origin date advances by *step_months*:
    fold 0 : train through origin_0,   predict origin_0 + 1..H
    fold 1 : train through origin_0+step, predict origin_0+step + 1..H
    ...
The minimum training window is enforced to ensure enough history for lags.

Metrics
-------
    RMSE  : root mean squared error
    MAE   : mean absolute error
    MAPE  : mean absolute percentage error (only on non-zero actuals; can be inf)
    WMAPE : weighted MAPE = sum|actual-pred| / sum|actual|  ← PRIMARY METRIC

Public API
----------
    compute_metrics(actuals, predictions) -> dict
    walk_forward_cv(panel, feature_panel, horizon_months, segment,
                    model_names, n_folds, step_months) -> pd.DataFrame
    select_best_model(cv_results) -> str
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, MAPE, and WMAPE for a set of actual vs predicted values.

    MAPE is computed only on rows where actuals > 0 to avoid division by zero.
    WMAPE = sum(|actual - pred|) / sum(|actual|) -- handles zero actuals gracefully.

    Raises ValueError if actuals and predictions have different lengths, so the
    caller always gets an explicit error rather than a silent NumPy broadcast.
    """
    actuals     = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if actuals.shape != predictions.shape:
        raise ValueError(
            f"compute_metrics: actuals and predictions must have the same shape. "
            f"Got actuals={actuals.shape}, predictions={predictions.shape}. "
            f"Align on SKU + MonthStart before calling."
        )

    if len(actuals) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "WMAPE": np.nan}

    errors     = actuals - predictions
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae  = float(np.mean(abs_errors))

    # MAPE on non-zero actuals only
    nonzero_mask = actuals > 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(abs_errors[nonzero_mask] / actuals[nonzero_mask]) * 100)
    else:
        mape = np.nan

    # WMAPE
    total_actual = np.sum(np.abs(actuals))
    wmape = float(np.sum(abs_errors) / total_actual * 100) if total_actual > 0 else np.nan

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "WMAPE": wmape}


def score_holdout_for_model(
    actuals: np.ndarray,
    predictions: np.ndarray,
    segment: str,
    horizon: int,
    model_name: str,
    n_candidate: int,
    n_dropped: int,
) -> dict | None:
    """
    Validate alignment, compute metrics, and return a result dict.

    Logs a warning and returns None if lengths do not match or both arrays
    are empty, so the caller never crashes on a bad row.

    Parameters
    ----------
    actuals, predictions : aligned arrays of equal length
    segment, horizon, model_name : metadata written into the result row
    n_candidate : holdout rows before alignment (for logging)
    n_dropped   : rows dropped because required lags were unavailable

    Returns
    -------
    dict with HorizonMonths, Segment, BestModel, N_Obs, RMSE, MAE, MAPE, WMAPE
    or None if the arrays cannot be scored.
    """
    actuals     = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if actuals.shape != predictions.shape:
        logger.warning(
            "score_holdout_for_model: shape mismatch skipped "
            "(segment=%s horizon=%d model=%s actuals=%s preds=%s)",
            segment, horizon, model_name, actuals.shape, predictions.shape,
        )
        return None

    n_obs = len(actuals)
    if n_obs == 0:
        logger.warning(
            "score_holdout_for_model: zero rows after alignment "
            "(segment=%s horizon=%d model=%s)", segment, horizon, model_name,
        )
        return None

    logger.info(
        "Holdout scoring  segment=%-14s H=%2d  model=%-18s "
        "candidate=%d  aligned=%d  dropped=%d",
        segment, horizon, model_name, n_candidate, n_obs, n_dropped,
    )

    metrics = compute_metrics(actuals, predictions)
    return {
        "HorizonMonths": horizon,
        "Segment":       segment,
        "BestModel":     model_name,
        "N_Obs":         n_obs,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def _get_cv_origins(
    train_months: pd.DatetimeIndex,
    min_train_months: int,
    step_months: int,
    n_folds: int,
) -> list[pd.Timestamp]:
    """
    Generate fold origin dates (the last training month for each fold).

    Origins are spaced step_months apart, stepping backwards from the last
    eligible month.  At least one origin is always returned when there is
    enough history (>= min_train_months), even when step_months is large
    relative to the eligible window.

    Fix: the old version could return zero origins when step_months >= len(eligible),
    because end_idx - step_months went negative before a single origin was added.
    Now a fallback guarantees at least one fold when history is sufficient.
    """
    sorted_months = sorted(train_months)
    if len(sorted_months) <= min_train_months:
        return []

    eligible = sorted_months[min_train_months - 1:]
    if not eligible:
        return []

    origins = []
    end_idx = len(eligible) - 1
    while end_idx >= 0 and len(origins) < n_folds:
        origins.append(eligible[end_idx])
        end_idx -= step_months

    # Guarantee at least one fold when there is enough history
    if not origins:
        origins = [eligible[-1]]

    return list(reversed(origins))


def walk_forward_cv(
    panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    horizon_months: int,
    segment: str,
    model_names: Sequence[str],
    n_folds: int = 4,
    step_months: int | None = None,
    min_train_months: int = 24,
) -> pd.DataFrame:
    """
    Walk-forward cross-validation for a given segment × horizon combination.

    For ML models (XGBoost, LightGBM, RandomForest), one global model is trained
    on all SKUs in the segment up to the fold origin, then evaluated on all SKUs
    in the subsequent horizon window.

    For statistical / baseline models (Prophet, SARIMA, SeasonalNaive, etc.),
    the model is fit individually per SKU within the fold.

    Parameters
    ----------
    panel           : output of build_panel() with Segment column attached
    feature_panel   : output of create_features() for this horizon
    horizon_months  : 1, 3, or 12
    segment         : "REGULAR", "INTERMITTENT", or "DEAD"
    model_names     : list of model names to evaluate (from SEGMENT_MODEL_ROSTER)
    n_folds         : number of walk-forward folds
    step_months     : how far to advance the origin between folds
                      (defaults to horizon_months so folds don't overlap)
    min_train_months: minimum months of history required before the first fold

    Returns
    -------
    pd.DataFrame with columns:
        ModelName, HorizonMonths, Segment, Fold, Origin,
        RMSE, MAE, MAPE, WMAPE, N_SKUs, N_Obs
    """
    from forecasting_pipeline.models import (
        SEGMENT_MODEL_ROSTER,
        train_ml_model,
        predict_ml,
        predict_baseline,
        fit_predict_prophet,
        fit_predict_neuralprophet,
        fit_predict_sarima,
    )
    from forecasting_pipeline.features import get_feature_columns

    ML_MODELS         = {"XGBoost", "LightGBM", "RandomForest"}
    PROPHET_MODELS    = {"Prophet"}
    NEURALPROPHET_MODELS = {"NeuralProphet"}
    SARIMA_MODELS     = {"SARIMA"}
    BASELINE_MODELS   = {"MA_3", "MA_12", "SeasonalNaive", "SeasonalAvg3Y",
                         "Croston", "ZeroForecast"}

    if step_months is None:
        step_months = max(1, horizon_months)

    # Filter to the requested segment
    seg_panel   = panel[panel["Segment"] == segment].copy()
    seg_features = feature_panel[feature_panel[SKU_COL].isin(seg_panel[SKU_COL].unique())].copy()

    if len(seg_panel) == 0:
        logger.warning("No SKUs found for segment=%s", segment)
        return pd.DataFrame()

    train_months = seg_panel[seg_panel.get("IsTrain", pd.Series(1, index=seg_panel.index)) == 1][DATE_COL]
    unique_train_months = pd.DatetimeIndex(sorted(train_months.unique()))

    origins = _get_cv_origins(unique_train_months, min_train_months, step_months, n_folds)
    if not origins:
        logger.warning("Not enough training months for CV (need >%d)", min_train_months)
        return pd.DataFrame()

    logger.info(
        "CV: segment=%s, horizon=%d, models=%s, folds=%d",
        segment, horizon_months, model_names, len(origins),
    )

    feature_cols  = get_feature_columns(horizon_months, seg_features)
    all_fold_results = []

    for fold_idx, origin in enumerate(origins):
        # Prediction target window
        target_start = origin + pd.DateOffset(months=1)
        target_end   = origin + pd.DateOffset(months=horizon_months)

        train_mask  = seg_features[DATE_COL] <= origin
        target_mask = (seg_panel[DATE_COL] >= target_start) & (seg_panel[DATE_COL] <= target_end)

        X_train = seg_features[train_mask][feature_cols].fillna(0)
        y_train = seg_features[train_mask][TARGET_COL].values
        target_df = seg_panel[target_mask].copy()

        if len(X_train) == 0 or len(target_df) == 0:
            continue

        for model_name in model_names:
            try:
                all_actuals = []
                all_preds   = []

                # ---- ML models: global panel model ----
                if model_name in ML_MODELS:
                    fitted = train_ml_model(model_name, X_train, y_train)

                    eval_mask = (seg_features[DATE_COL] >= target_start) & \
                                (seg_features[DATE_COL] <= target_end)
                    eval_feat = seg_features[eval_mask][[SKU_COL, DATE_COL] + feature_cols].copy()

                    if len(eval_feat) == 0:
                        continue

                    # Align actuals and features on SKU+Date so row counts always match.
                    # feature_panel drops short-history SKUs (missing required lags),
                    # so target_df can have MORE rows than eval_feat.
                    aligned = target_df[[SKU_COL, DATE_COL, TARGET_COL]].merge(
                        eval_feat, on=[SKU_COL, DATE_COL], how="inner"
                    )
                    if aligned.empty:
                        continue

                    X_eval = aligned[feature_cols].fillna(0)
                    preds  = predict_ml(fitted, X_eval)

                    all_actuals = aligned[TARGET_COL].values
                    all_preds   = preds

                # ---- Statistical / baseline: per-SKU ----
                else:
                    skus = seg_panel[seg_panel[DATE_COL] <= origin][SKU_COL].unique()

                    for sku in skus:
                        sku_hist = (
                            seg_panel[
                                (seg_panel[SKU_COL] == sku) &
                                (seg_panel[DATE_COL] <= origin)
                            ]
                            .set_index(DATE_COL)[TARGET_COL]
                            .sort_index()
                        )
                        sku_actual = target_df[target_df[SKU_COL] == sku][TARGET_COL].values
                        if len(sku_actual) == 0:
                            continue

                        if model_name in PROPHET_MODELS:
                            preds_list, _, _ = fit_predict_prophet(sku_hist, horizon_months)
                        elif model_name in NEURALPROPHET_MODELS:
                            preds_list, _, _ = fit_predict_neuralprophet(sku_hist, horizon_months)
                        elif model_name in SARIMA_MODELS:
                            preds_list, _, _ = fit_predict_sarima(sku_hist, horizon_months)
                        else:
                            preds_list = predict_baseline(model_name, sku_hist, horizon_months)

                        # Align predictions to the actual evaluation window length
                        h = len(sku_actual)
                        all_actuals.extend(sku_actual[:h])
                        all_preds.extend(preds_list[:h])

                metrics = compute_metrics(
                    np.array(all_actuals, dtype=float),
                    np.array(all_preds, dtype=float),
                )
                all_fold_results.append({
                    "ModelName":     model_name,
                    "HorizonMonths": horizon_months,
                    "Segment":       segment,
                    "Fold":          fold_idx,
                    "Origin":        origin,
                    "N_SKUs":        len(set(target_df[SKU_COL])),
                    "N_Obs":         len(all_actuals),
                    **metrics,
                })
                logger.debug(
                    "  fold=%d origin=%s model=%s WMAPE=%.2f",
                    fold_idx, origin.strftime("%Y-%m"), model_name,
                    metrics.get("WMAPE", np.nan),
                )

            except Exception as exc:
                logger.warning("fold=%d model=%s FAILED: %s", fold_idx, model_name, exc)

    if not all_fold_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_fold_results)
    return results_df


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_model(cv_results: pd.DataFrame, metric: str = "WMAPE") -> pd.DataFrame:
    """
    Given the CV results DataFrame from walk_forward_cv(), select the best
    model per (Segment, HorizonMonths) combination based on *metric*.

    Returns a DataFrame with columns:
        Segment, HorizonMonths, BestModel, mean_WMAPE, mean_RMSE, mean_MAE
    """
    if cv_results.empty:
        return pd.DataFrame()

    agg = (
        cv_results
        .groupby(["Segment", "HorizonMonths", "ModelName"])
        [["WMAPE", "RMSE", "MAE", "MAPE"]]
        .mean()
        .reset_index()
    )

    # For each segment × horizon, pick the row with the lowest mean metric
    idx_best = (
        agg
        .sort_values(metric)
        .groupby(["Segment", "HorizonMonths"])
        .head(1)
        .index
    )
    best = agg.loc[idx_best].rename(columns={
        "ModelName": "BestModel",
        "WMAPE": "mean_WMAPE",
        "RMSE":  "mean_RMSE",
        "MAE":   "mean_MAE",
        "MAPE":  "mean_MAPE",
    }).reset_index(drop=True)

    logger.info("Best models per segment × horizon:\n%s", best.to_string(index=False))
    return best

def compute_scenario_bounds(
    cv_results: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    metric: str = "MAE",
    segments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Derive conservative / middle / aggressive scenario bounds from CV error.

    Business rationale
    ------------------
    Middle       = point forecast (model best estimate)
    Conservative = middle - expected_error  (plan for demand being lower)
    Aggressive   = middle + expected_error  (plan for demand being higher)

    Error bands are calibrated per Segment × HorizonMonths from walk-forward CV.
    This is strictly better than ±% multipliers because MAE is in the same units
    as UnitsSold and is already calibrated to each segment's demand scale.

    v5.2 bug fix: old version merged only on HorizonMonths, causing REGULAR
    (CV MAE ≈ 307) and INTERMITTENT (CV MAE ≈ 97) to share the same band.
    Now joins on Segment + HorizonMonths so each segment gets its own band.

    Parameters
    ----------
    cv_results      : fold-level CV results — must contain Segment, HorizonMonths,
                      and the metric column (MAE or RMSE)
    point_forecasts : gold_fact_forecasts.csv schema — must contain ForecastUnits,
                      HorizonMonths. Optionally Segment (or provide segments_df).
    metric          : "MAE" (default) or "RMSE"
    segments_df     : optional SKU→Segment mapping (columns: SKU, Segment).
                      Used to attach Segment to point_forecasts when it is absent.
                      Not needed if Segment is already a column in point_forecasts.

    Returns
    -------
    pd.DataFrame with columns:
        Key, MonthStart, HorizonMonths, Segment, ModelName,
        Conservative, Middle, Aggressive, ModelVersion
    All Conservative values are clipped to >= 0.
    """
    if cv_results.empty or point_forecasts.empty:
        return pd.DataFrame()

    fdf = point_forecasts.copy()

    # Attach Segment to forecasts if not already present
    if "Segment" not in fdf.columns:
        if segments_df is not None:
            seg_map = segments_df[["SKU", "Segment"]].drop_duplicates()
            fdf = fdf.merge(
                seg_map.rename(columns={"SKU": "Key"}),
                on="Key", how="left",
            )
        else:
            # Fallback: join on HorizonMonths only (degraded — logs warning)
            logger.warning(
                "compute_scenario_bounds: Segment column absent and no segments_df "
                "supplied. Falling back to HorizonMonths-only join. REGULAR and "
                "INTERMITTENT will share error bands."
            )
            error_h = (
                cv_results.groupby("HorizonMonths")[metric]
                .mean().reset_index()
                .rename(columns={metric: "expected_error"})
            )
            fdf = fdf.merge(error_h, on="HorizonMonths", how="left")
            fdf["expected_error"] = fdf["expected_error"].fillna(0.0)
            fdf["Middle"]         = fdf["ForecastUnits"]
            fdf["Conservative"]   = (fdf["ForecastUnits"] - fdf["expected_error"]).clip(lower=0.0)
            fdf["Aggressive"]     = fdf["ForecastUnits"] + fdf["expected_error"]
            out_cols = ["Key", "MonthStart", "HorizonMonths", "ModelName",
                        "Conservative", "Middle", "Aggressive", "ModelVersion"]
            return fdf[[c for c in out_cols if c in fdf.columns]].reset_index(drop=True)

    fdf["Segment"] = fdf["Segment"].fillna("REGULAR")

    # Expected error per Segment × HorizonMonths — exclude DEAD (always 0)
    cv_non_dead = cv_results[cv_results["Segment"] != "DEAD"].copy()
    error_by_seg = (
        cv_non_dead.groupby(["Segment", "HorizonMonths"])[metric]
        .mean().reset_index()
        .rename(columns={metric: "expected_error"})
    )
    # DEAD always gets 0 error (zero forecast has no planning band)
    dead_rows = pd.DataFrame([
        {"Segment": "DEAD", "HorizonMonths": h, "expected_error": 0.0}
        for h in cv_results["HorizonMonths"].unique()
    ])
    error_by_seg = pd.concat([error_by_seg, dead_rows], ignore_index=True)

    # Join on Segment + HorizonMonths — each segment gets its own calibrated band
    fdf = fdf.merge(error_by_seg, on=["Segment", "HorizonMonths"], how="left")
    fdf["expected_error"] = fdf["expected_error"].fillna(0.0)

    fdf["Middle"]       = fdf["ForecastUnits"]
    fdf["Conservative"] = (fdf["ForecastUnits"] - fdf["expected_error"]).clip(lower=0.0)
    fdf["Aggressive"]   = fdf["ForecastUnits"] + fdf["expected_error"]

    out_cols = ["Key", "MonthStart", "HorizonMonths", "Segment", "ModelName",
                "Conservative", "Middle", "Aggressive", "ModelVersion"]
    return fdf[[c for c in out_cols if c in fdf.columns]].reset_index(drop=True)


def build_scenario_bounds_from_simulation(
    sim_df: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    cv_results: pd.DataFrame | None = None,
    metric: str = "MAE",
) -> pd.DataFrame:
    """
    Derive scenario bounds using SIMULATION errors when available.

    Preferred over compute_scenario_bounds() when simulation_2026_predictions.csv
    exists, because simulation errors come from real out-of-sample actuals (Jan-Feb
    2026), not CV folds within the training period. They are a more honest estimate
    of how the model performs in a genuine future prediction context.

    Error source hierarchy (best → worst):
      1. Simulation AbsError per Segment × HorizonMonths  (HasActual == True)
      2. CV MAE per Segment × HorizonMonths               (fallback)
      3. 0                                                 (DEAD / no data)

    Requires at least 5 scored rows per (Segment, HorizonMonths) to trust the
    simulation estimate; otherwise falls back to CV.

    Parameters
    ----------
    sim_df          : simulation_2026_predictions.csv as a DataFrame.
                      Must contain: HasActual, AbsError, Segment, HorizonMonths.
    point_forecasts : gold_fact_forecasts.csv schema DataFrame.
                      Must contain: ForecastUnits, HorizonMonths.
                      Segment column preferred; attach via segments_df if absent.
    cv_results      : optional CV results for fallback. If None and simulation
                      has no actuals, expected_error defaults to 0.
    metric          : which CV column to use as fallback ("MAE" or "RMSE").

    Returns
    -------
    Same schema as compute_scenario_bounds():
        Key, MonthStart, HorizonMonths, Segment, ModelName,
        Conservative, Middle, Aggressive, ModelVersion
    All Conservative values are clipped to >= 0.
    """
    if sim_df.empty or point_forecasts.empty:
        return pd.DataFrame()

    # Normalise HasActual to boolean safely
    if sim_df["HasActual"].dtype == object:
        scored = sim_df[sim_df["HasActual"].astype(str).str.lower() == "true"].copy()
    else:
        scored = sim_df[sim_df["HasActual"].astype(bool)].copy()

    scored["AbsError"] = pd.to_numeric(scored.get("AbsError", pd.Series(dtype=float)), errors="coerce")

    # Build expected error table from simulation actuals
    sim_error: dict[tuple, float] = {}
    if not scored.empty and "AbsError" in scored.columns:
        for (seg, h), grp in scored.groupby(["Segment", "HorizonMonths"]):
            valid = grp["AbsError"].dropna()
            if len(valid) >= 5:  # need 5+ rows to trust the estimate
                sim_error[(seg, h)] = float(valid.mean())
                logger.info(
                    "Scenario bounds: %s H=%d using simulation MAE=%.1f (n=%d)",
                    seg, h, sim_error[(seg, h)], len(valid),
                )

    fdf = point_forecasts.copy()
    if "Segment" not in fdf.columns:
        fdf["Segment"] = "REGULAR"
    fdf["Segment"] = fdf["Segment"].fillna("REGULAR")

    segs_in_fc  = fdf["Segment"].unique()
    horizons_fc = fdf["HorizonMonths"].unique()

    # Build expected_error lookup: sim first, CV fallback, 0 for DEAD / no data
    error_rows = []
    for seg in segs_in_fc:
        for h in horizons_fc:
            if seg == "DEAD":
                err = 0.0
            elif (seg, h) in sim_error:
                err = sim_error[(seg, h)]
            elif cv_results is not None and not cv_results.empty:
                cv_grp = cv_results[
                    (cv_results["Segment"] == seg) &
                    (cv_results["HorizonMonths"] == h)
                ]
                err = float(cv_grp[metric].mean()) if not cv_grp.empty else 0.0
                if err > 0:
                    logger.info(
                        "Scenario bounds: %s H=%d using CV %s=%.1f (no sim actuals)",
                        seg, h, metric, err,
                    )
            else:
                err = 0.0
            error_rows.append({"Segment": seg, "HorizonMonths": h, "expected_error": err})

    error_df = pd.DataFrame(error_rows)
    fdf = fdf.merge(error_df, on=["Segment", "HorizonMonths"], how="left")
    fdf["expected_error"] = fdf["expected_error"].fillna(0.0)

    fdf["Middle"]       = fdf["ForecastUnits"]
    fdf["Conservative"] = (fdf["ForecastUnits"] - fdf["expected_error"]).clip(lower=0.0)
    fdf["Aggressive"]   = fdf["ForecastUnits"] + fdf["expected_error"]

    out_cols = ["Key", "MonthStart", "HorizonMonths", "Segment", "ModelName",
                "Conservative", "Middle", "Aggressive", "ModelVersion"]
    return fdf[[c for c in out_cols if c in fdf.columns]].reset_index(drop=True)
