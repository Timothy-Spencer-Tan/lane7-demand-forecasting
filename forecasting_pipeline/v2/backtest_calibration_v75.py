"""
lane7_forecast.backtest_calibration_v75
=========================================
v7.5 True Backtest-Based Bias Calibration.

Why v7.4 calibration was limited
----------------------------------
v7.4 had two calibration paths:
  A. CV fold predictions (if available)
  B. Seasonal-naive historical bias fallback (always WEAK evidence)

In practice, path A requires CV fold-level outputs to have been saved,
which the v7.x pipeline does not do automatically.  Path B only ever
produces WEAK evidence and never applies factors.  This meant v7.4 often
ran with no calibration actually applied.

v7.5 fix
--------
Run a dedicated rolling backtest at the StyleCode level, generating
true out-of-sample predictions at each origin month.  This produces
genuine backtest predictions that can then be used to learn calibration
factors with STRONG or MODERATE evidence for most StyleCodes.

Rolling backtest design
-----------------------
Origins are quarterly, starting 24 months before backtest_end:
    backtest_end = 2025-12-01  (last month before holdout)
    → origins: 2024-01, 2024-04, 2024-07, 2024-10,
               2025-01, 2025-04, 2025-07, 2025-10

For each origin month O:
    1. Build a StyleCode panel from data with MonthStart < O
    2. Call run_forecasts() with forecast_start = O
       n_forecast_months = horizon_months
    3. For each forecast month F = O + k months (k = 1..horizon):
       look up actual StyleCode demand for F from gold_df
       record (OriginMonth=O, ForecastMonth=F, HorizonMonths=H, ...)

Leakage guarantee
-----------------
All origin months are strictly < backtest_end.
Actuals are only joined for ForecastMonths ≤ backtest_end.
Jan–Feb 2026 actuals are NEVER loaded inside this module.

Calibration table rules
-----------------------
  raw_bias_ratio     = sum(PredictedUnits) / sum(ActualUnits)
  calibration_factor = sum(ActualUnits) / sum(PredictedUnits)
                     = 1 / raw_bias_ratio

Evidence tiers (v7.5, tighter than v7.4):
  STRONG   : n_obs ≥ 6  AND  total_actual ≥ 500
  MODERATE : n_obs ≥ 3  AND  total_actual ≥ 200
  WEAK     : some evidence but below moderate
  NONE     : no data

Safe factor bounds: [0.85, 1.15]
Only STRONG/MODERATE tiers apply the factor.

Public API
----------
    run_stylecode_backtest(
        gold_df, dim_product_df, dim_date_df,
        best_models_df, adjustment_config=None,
        backtest_end=None, horizon_months_list=[1,3],
        origin_step_months=3, n_origins=8,
        phase=1
    ) -> backtest_df

    build_v75_calibration_table(
        backtest_df, backtest_end,
        horizon_months_list=[1,3],
        strong_n=6, strong_units=500,
        moderate_n=3, moderate_units=200,
        min_factor=0.85, max_factor=1.15
    ) -> calibration_df

    apply_v75_calibration(
        scode_forecasts_df, calibration_df
    ) -> calibrated_df

    build_bias_analysis(
        actuals_df, dim_product_df,
        raw_scode_fc, calibrated_scode_fc,
        holdout_months
    ) -> bias_df

    validate_calibration_table(calibration_df, min_factor, max_factor) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"
SCODE_COL  = "StyleCodeDesc"
SC_COL     = "StyleColorDesc"

DEFAULT_MIN_FACTOR = 0.85
DEFAULT_MAX_FACTOR = 1.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_train_end(dim_date_df: pd.DataFrame, phase: int = 1) -> pd.Timestamp:
    """Return the last TRAIN date from dim_date_df for the given phase."""
    dd = dim_date_df.copy()
    dd[DATE_COL] = pd.to_datetime(dd[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    if "IsTrain" in dd.columns:
        train_rows = dd[dd["IsTrain"] == 1]
        if not train_rows.empty:
            return train_rows[DATE_COL].max()
    return dd[DATE_COL].max()


def _stylecode_actuals(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    forecast_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Aggregate gold demand to StyleCode level for the given forecast months.

    Returns DataFrame with columns: StyleCodeDesc, MonthStart, ActualUnits
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()
    gold = gold[gold[DATE_COL].isin(forecast_months)].copy()

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)

    merged = gold.merge(dp, on=SKU_COL, how="inner")
    agg = (
        merged.groupby([SCODE_COL, DATE_COL], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "ActualUnits", DATE_COL: "ForecastMonth"})
    )
    return agg


def _evidence_tier(n_obs: int, total_actual: float,
                   strong_n: int, strong_units: float,
                   moderate_n: int, moderate_units: float) -> str:
    if n_obs >= strong_n and total_actual >= strong_units:
        return "STRONG"
    if n_obs >= moderate_n and total_actual >= moderate_units:
        return "MODERATE"
    if n_obs >= 1:
        return "WEAK"
    return "NONE"


# ---------------------------------------------------------------------------
# Public: run_stylecode_backtest
# ---------------------------------------------------------------------------

def run_stylecode_backtest(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    dim_date_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    adjustment_config: dict | None = None,
    backtest_end: pd.Timestamp | None = None,
    horizon_months_list: list[int] | None = None,
    origin_step_months: int = 3,
    n_origins: int = 8,
    phase: int = 1,
) -> pd.DataFrame:
    """
    Run a rolling StyleCode-level backtest to produce true out-of-sample
    predictions for use in calibration.

    For each origin month O (stepping back origin_step_months at a time,
    starting n_origins * origin_step_months before backtest_end):
      1. Build a StyleCode panel using only data with MonthStart < O
      2. Segment and create features on that restricted panel
      3. Call run_forecasts() with forecast_start=O
      4. Join actuals for the forecast months F ≤ backtest_end

    Parameters
    ----------
    gold_df              : full gold demand (all dates)
    dim_product_df       : dim_product table
    dim_date_df          : dim_date table
    best_models_df       : output of run_cv() at StyleCode level
                           (uses the full-history CV results — these are
                           based on segment/model type, not on data after O)
    adjustment_config    : v6.2 adjustment config dict or None
    backtest_end         : last allowed backtest month (default: 2025-12)
    horizon_months_list  : list of horizons to backtest (default [1, 3])
    origin_step_months   : months between successive origins (default 3)
    n_origins            : number of origin months to evaluate (default 8)
    phase                : panel phase (default 1)

    Returns
    -------
    pd.DataFrame — backtest predictions with columns:
        OriginMonth, ForecastMonth, HorizonMonths,
        StyleCodeDesc, ModelName, PredictedUnits, ActualUnits,
        Error, AbsError, BiasRatio, WMAPE_Component
    """
    from lane7_forecast.data_prep    import build_stylecode_panel
    from lane7_forecast.segmentation import segment_skus, attach_segment
    from lane7_forecast.pipeline     import run_forecasts

    horizon_months_list = horizon_months_list or [1, 3]

    if backtest_end is None:
        backtest_end = pd.Timestamp("2025-12-01")
    else:
        backtest_end = pd.Timestamp(backtest_end)

    # Build origin months: step backwards from backtest_end
    origins = []
    for i in range(n_origins):
        offset = (i + 1) * origin_step_months
        o = backtest_end - pd.DateOffset(months=offset)
        o = pd.Timestamp(o.year, o.month, 1)
        if o > pd.Timestamp("2017-01-01"):   # never before data starts
            origins.append(o)
    origins = sorted(set(origins))

    logger.info(
        "[v7.5 backtest] %d origins × %d horizons  (backtest_end=%s)",
        len(origins), len(horizon_months_list), backtest_end.strftime("%Y-%m"),
    )

    all_rows: list[dict] = []

    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp_sc = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)

    # Build a synthetic dim_date that marks all months before each origin as TRAIN
    # (the real dim_date is used only for its date list)
    dd = dim_date_df.copy()
    dd[DATE_COL] = pd.to_datetime(dd[DATE_COL]).dt.to_period("M").dt.to_timestamp()

    for origin in origins:
        origin_str = origin.strftime("%Y-%m")
        logger.info("[v7.5 backtest] Origin %s", origin_str)

        # Data available at this origin: only months strictly before the origin
        gold_avail = gold[gold[DATE_COL] < origin].copy()
        if gold_avail.empty or gold_avail[DATE_COL].nunique() < 12:
            logger.debug("[v7.5 backtest] Skipping origin %s: insufficient history", origin_str)
            continue

        # Build synthetic dim_date: months < origin are IsTrain=1, others IsTrain=0
        dd_origin = dd.copy()
        dd_origin["IsTrain"] = (dd_origin[DATE_COL] < origin).astype(int)

        # Aggregate gold to StyleCode level
        gold_sc = gold_avail.merge(dp_sc, on=SKU_COL, how="inner")
        scode_demand = (
            gold_sc.groupby([SCODE_COL, DATE_COL], as_index=False)[TARGET_COL]
            .sum()
            .rename(columns={SCODE_COL: SKU_COL})   # treat StyleCode as SKU for build_panel
        )

        # Build panel and segment using this restricted history
        try:
            from lane7_forecast.data_prep import build_panel
            panel_origin = build_panel(
                demand_df=scode_demand,
                dim_date_df=dd_origin,
                phase=1,
                dim_product_df=None,
            )
            segs_origin   = segment_skus(panel_origin)
            panel_seg_org = attach_segment(panel_origin, segs_origin)
        except Exception as exc:
            logger.warning("[v7.5 backtest] Panel build failed for origin %s: %s", origin_str, exc)
            continue

        prep_origin = {
            "tables":    {
                "demand":    scode_demand.rename(columns={SKU_COL: SCODE_COL}),
                "dim_date":  dd_origin,
            },
            "panel":     panel_origin,
            "segments":  segs_origin,
            "panel_seg": panel_seg_org,
        }

        for horizon in horizon_months_list:
            # Number of forecast steps = horizon (capped at months available before backtest_end)
            max_steps = max(
                1,
                int((backtest_end.year - origin.year) * 12
                    + (backtest_end.month - origin.month)),
            )
            n_steps = min(horizon, max_steps)
            if n_steps < 1:
                continue

            bm_h = best_models_df[best_models_df["HorizonMonths"] == horizon]
            if bm_h.empty:
                continue

            try:
                fc_df = run_forecasts(
                    prep=prep_origin,
                    best_models_df=bm_h,
                    horizon_months=horizon,
                    forecast_start=origin.strftime("%Y-%m-%d"),
                    n_forecast_months=n_steps,
                    phase=1,
                    model_version=f"v7.5-backtest-O{origin_str}-H{horizon}",
                    output_path=None,
                    append=False,
                    adjustment_config=adjustment_config,
                )
            except Exception as exc:
                logger.warning(
                    "[v7.5 backtest] run_forecasts failed for origin=%s H=%d: %s",
                    origin_str, horizon, exc,
                )
                continue

            if fc_df.empty:
                continue

            # Rename Key back to StyleCodeDesc
            if "Key" in fc_df.columns:
                fc_df = fc_df.rename(columns={"Key": SCODE_COL})

            # Join actuals for each forecast month
            fc_months = fc_df["MonthStart"].apply(
                lambda m: pd.Timestamp(m).to_period("M").to_timestamp()
            ).unique().tolist()

            actuals_sc = _stylecode_actuals(gold_df, dim_product_df, fc_months)

            fc_df["MonthStart"] = pd.to_datetime(fc_df["MonthStart"]).dt.to_period("M").dt.to_timestamp()
            merged = fc_df.merge(
                actuals_sc.rename(columns={"ForecastMonth": "MonthStart"}),
                on=[SCODE_COL, "MonthStart"],
                how="left",
            )
            merged["ActualUnits"]    = merged["ActualUnits"].fillna(0.0)
            merged["PredictedUnits"] = merged["ForecastUnits"].clip(lower=0).fillna(0.0)
            merged["Error"]          = merged["ActualUnits"] - merged["PredictedUnits"]
            merged["AbsError"]       = merged["Error"].abs()
            merged["BiasRatio"]      = np.where(
                merged["ActualUnits"] > 0,
                merged["PredictedUnits"] / merged["ActualUnits"],
                np.nan,
            )
            merged["WMAPE_Component"] = np.where(
                merged["ActualUnits"] > 0,
                merged["AbsError"] / merged["ActualUnits"] * 100,
                np.nan,
            )

            for _, row in merged.iterrows():
                all_rows.append({
                    "OriginMonth":    origin_str,
                    "ForecastMonth":  row["MonthStart"].strftime("%Y-%m"),
                    "HorizonMonths":  int(horizon),
                    SCODE_COL:        row[SCODE_COL],
                    "ModelName":      row.get("ModelName", ""),
                    "PredictedUnits": round(float(row["PredictedUnits"]), 4),
                    "ActualUnits":    round(float(row["ActualUnits"]), 4),
                    "Error":          round(float(row["Error"]), 4),
                    "AbsError":       round(float(row["AbsError"]), 4),
                    "BiasRatio":      round(float(row["BiasRatio"]), 4)
                                      if not np.isnan(row["BiasRatio"]) else None,
                    "WMAPE_Component":round(float(row["WMAPE_Component"]), 4)
                                      if not np.isnan(row["WMAPE_Component"]) else None,
                })

    if not all_rows:
        logger.warning("[v7.5 backtest] No backtest rows produced — check origins and data.")
        return pd.DataFrame(columns=[
            "OriginMonth", "ForecastMonth", "HorizonMonths",
            SCODE_COL, "ModelName", "PredictedUnits", "ActualUnits",
            "Error", "AbsError", "BiasRatio", "WMAPE_Component",
        ])

    result = pd.DataFrame(all_rows)
    logger.info(
        "[v7.5 backtest] Complete: %d rows across %d StyleCodes",
        len(result), result[SCODE_COL].nunique(),
    )
    return result.sort_values(["OriginMonth", SCODE_COL, "HorizonMonths"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public: build_v75_calibration_table
# ---------------------------------------------------------------------------

def build_v75_calibration_table(
    backtest_df: pd.DataFrame,
    backtest_end: pd.Timestamp,
    horizon_months_list: list[int] | None = None,
    strong_n: int          = 6,
    strong_units: float    = 500.0,
    moderate_n: int        = 3,
    moderate_units: float  = 200.0,
    min_factor: float      = DEFAULT_MIN_FACTOR,
    max_factor: float      = DEFAULT_MAX_FACTOR,
) -> pd.DataFrame:
    """
    Build a StyleCode × HorizonMonths calibration table from backtest results.

    No leakage: only ForecastMonths ≤ backtest_end are used.

    Calibration factor = sum(ActualUnits) / sum(PredictedUnits)
    If the model over-forecasts → factor < 1.0 → reduces future forecasts.
    If the model under-forecasts → factor > 1.0 → increases future forecasts.

    Parameters
    ----------
    backtest_df          : output of run_stylecode_backtest()
    backtest_end         : last allowed ForecastMonth (inclusive)
    horizon_months_list  : horizons to calibrate (default [1, 3])
    strong_n / strong_units : STRONG tier thresholds
    moderate_n / moderate_units : MODERATE tier thresholds
    min_factor / max_factor : safe clamping range

    Returns
    -------
    pd.DataFrame — one row per (StyleCodeDesc, HorizonMonths)
    """
    horizon_months_list = horizon_months_list or [1, 3]
    backtest_end        = pd.Timestamp(backtest_end)

    df = backtest_df.copy()

    # Enforce leakage guard: only use forecast months ≤ backtest_end
    df["_fm_ts"] = pd.to_datetime(df["ForecastMonth"].astype(str) + "-01")
    df = df[df["_fm_ts"] <= backtest_end].drop(columns=["_fm_ts"])

    if df.empty:
        logger.warning("[v7.5 calib] No backtest rows within backtest_end — empty table.")
        return _empty_calib_df()

    rows = []

    for horizon in horizon_months_list:
        h_df = df[df["HorizonMonths"] == horizon]

        for scode in sorted(h_df[SCODE_COL].dropna().unique()):
            sc_df = h_df[h_df[SCODE_COL] == scode]

            # Only include rows where both sides have actual data
            scored = sc_df[sc_df["ActualUnits"] > 0].copy()
            n_obs       = len(scored)
            total_actual= float(scored["ActualUnits"].sum())
            total_pred  = float(scored["PredictedUnits"].sum())

            tier = _evidence_tier(
                n_obs, total_actual,
                strong_n, strong_units,
                moderate_n, moderate_units,
            )

            if tier in ("STRONG", "MODERATE") and total_pred > 0:
                raw_bias = round(total_pred / total_actual, 4)
                raw_factor = total_actual / total_pred
                clamped    = float(np.clip(raw_factor, min_factor, max_factor))
                applied    = True
                reason     = f"backtest_{tier.lower()}_evidence"
            else:
                raw_bias  = round(total_pred / total_actual, 4) if total_actual > 0 else None
                clamped   = 1.0
                applied   = False
                reason    = f"backtest_{tier.lower()}_evidence_no_factor"

            rows.append({
                SCODE_COL:                scode,
                "HorizonMonths":          int(horizon),
                "n_backtest_observations":n_obs,
                "total_actual_units":     round(total_actual, 2),
                "total_predicted_units":  round(total_pred, 2),
                "raw_bias_ratio":         raw_bias,
                "calibration_factor":     round(clamped, 4),
                "calibration_applied":    applied,
                "evidence_tier":          tier,
                "reason_code":            reason,
            })

    if not rows:
        return _empty_calib_df()

    result = pd.DataFrame(rows)

    n_strong   = (result["evidence_tier"] == "STRONG").sum()
    n_moderate = (result["evidence_tier"] == "MODERATE").sum()
    n_applied  = result["calibration_applied"].sum()
    logger.info(
        "[v7.5 calib] Table: %d rows | STRONG=%d MODERATE=%d WEAK=%d NONE=%d | applied=%d",
        len(result), n_strong, n_moderate,
        (result["evidence_tier"] == "WEAK").sum(),
        (result["evidence_tier"] == "NONE").sum(),
        n_applied,
    )
    return result.sort_values([SCODE_COL, "HorizonMonths"]).reset_index(drop=True)


def _empty_calib_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        SCODE_COL, "HorizonMonths",
        "n_backtest_observations", "total_actual_units", "total_predicted_units",
        "raw_bias_ratio", "calibration_factor", "calibration_applied",
        "evidence_tier", "reason_code",
    ])


# ---------------------------------------------------------------------------
# Public: apply_v75_calibration
# ---------------------------------------------------------------------------

def apply_v75_calibration(
    scode_forecasts_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply v7.5 calibration factors to StyleCode-level ForecastUnits.

    For each (StyleCodeDesc, HorizonMonths) in scode_forecasts_df, multiply
    ForecastUnits (and Lower/Upper) by the matching calibration_factor.
    Rows with no match or factor=1.0 are unchanged.

    Parameters
    ----------
    scode_forecasts_df : raw StyleCode forecasts (Key = StyleCodeDesc)
    calibration_df     : output of build_v75_calibration_table()

    Returns
    -------
    pd.DataFrame — same schema + CalibrationFactor, CalibrationApplied columns
    """
    fc = scode_forecasts_df.copy()
    if fc.empty:
        return fc

    key_col = "Key" if "Key" in fc.columns else SCODE_COL
    if key_col == "Key":
        fc = fc.rename(columns={"Key": SCODE_COL})

    if calibration_df.empty:
        fc["CalibrationFactor"]  = 1.0
        fc["CalibrationApplied"] = False
        if key_col == "Key":
            fc = fc.rename(columns={SCODE_COL: "Key"})
        return fc

    # Build vectorised lookup
    calib_lu = (
        calibration_df
        .set_index([SCODE_COL, "HorizonMonths"])[["calibration_factor", "calibration_applied"]]
    )

    factors = []
    applied = []
    for _, row in fc.iterrows():
        lk = (str(row.get(SCODE_COL, "")), int(row.get("HorizonMonths", -1)))
        if lk in calib_lu.index:
            factors.append(float(calib_lu.at[lk, "calibration_factor"]))
            applied.append(bool(calib_lu.at[lk, "calibration_applied"]))
        else:
            factors.append(1.0)
            applied.append(False)

    fc["CalibrationFactor"]  = factors
    fc["CalibrationApplied"] = applied

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in fc.columns:
            fc[col] = (fc[col] * fc["CalibrationFactor"]).clip(lower=0).round(4)

    if "ModelVersion" in fc.columns:
        fc["ModelVersion"] = fc.apply(
            lambda r: (r["ModelVersion"] + "+v75calib")
            if r["CalibrationApplied"] and "+v75calib" not in str(r.get("ModelVersion", ""))
            else r["ModelVersion"],
            axis=1,
        )

    if key_col == "Key":
        fc = fc.rename(columns={SCODE_COL: "Key"})

    n_applied = sum(applied)
    logger.info(
        "[v7.5 calib] Applied factors: %d / %d rows adjusted",
        n_applied, len(fc),
    )
    return fc


# ---------------------------------------------------------------------------
# Public: build_bias_analysis
# ---------------------------------------------------------------------------

def build_bias_analysis(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    raw_scode_fc: pd.DataFrame,
    calibrated_scode_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Compare bias before and after calibration at StyleCode level.

    Parameters
    ----------
    actuals_df          : gold_fact_monthly_demand (all actuals)
    dim_product_df      : dim_product table
    raw_scode_fc        : raw StyleCode forecasts (Key = StyleCodeDesc)
    calibrated_scode_fc : calibrated StyleCode forecasts
    holdout_months      : list of pd.Timestamps to evaluate

    Returns
    -------
    pd.DataFrame with columns:
        Level, HorizonMonths, MonthStart,
        TotalActual, TotalForecastRaw, TotalForecastCalibrated,
        RawBiasRatio, CalibratedBiasRatio, BiasImprovement
    """
    acts = actuals_df.copy()
    acts["MonthStart"] = pd.to_datetime(acts["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    acts["SKU"]        = acts["SKU"].astype(str).str.strip()
    acts = acts[acts["MonthStart"].isin([pd.Timestamp(m) for m in holdout_months])]

    dp = dim_product_df.copy()
    dp["SKU"] = dp["SKU"].astype(str).str.strip()
    dp_sc = dp[["SKU", SCODE_COL]].dropna().drop_duplicates("SKU")
    acts = acts.merge(dp_sc, on="SKU", how="left")

    def _prep(fc):
        f = fc.copy()
        if "Key" in f.columns:
            f = f.rename(columns={"Key": SCODE_COL})
        f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        return f

    raw_fc  = _prep(raw_scode_fc)
    cal_fc  = _prep(calibrated_scode_fc)

    rows = []

    for m in sorted(acts["MonthStart"].unique()):
        a_m = acts[acts["MonthStart"] == m]
        if SCODE_COL not in a_m.columns:
            continue

        a_sc = a_m.groupby(SCODE_COL)["UnitsSold"].sum().reset_index()

        raw_m = raw_fc[raw_fc["MonthStart"] == m]
        cal_m = cal_fc[cal_fc["MonthStart"] == m]

        for h in sorted(raw_m["HorizonMonths"].unique()):
            raw_h = raw_m[raw_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()
            cal_h = cal_m[cal_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()

            mg = (
                a_sc
                .merge(raw_h.rename(columns={"ForecastUnits": "RawFC"}), on=SCODE_COL, how="inner")
                .merge(cal_h.rename(columns={"ForecastUnits": "CalFC"}), on=SCODE_COL, how="inner")
            )
            if mg.empty:
                continue

            tot_act = float(mg["UnitsSold"].sum())
            tot_raw = float(mg["RawFC"].sum())
            tot_cal = float(mg["CalFC"].sum())

            raw_bias = round(tot_raw / tot_act, 4) if tot_act > 0 else np.nan
            cal_bias = round(tot_cal / tot_act, 4) if tot_act > 0 else np.nan

            # BiasImprovement = abs(raw_bias - 1) - abs(cal_bias - 1)
            # Positive = calibration moved closer to 1.0 (improved)
            improvement = round(
                (abs(raw_bias - 1.0) - abs(cal_bias - 1.0)) * 100, 4
            ) if not (np.isnan(raw_bias) or np.isnan(cal_bias)) else np.nan

            rows.append({
                "Level":                   "StyleCode",
                "HorizonMonths":           int(h),
                "MonthStart":              m.strftime("%Y-%m"),
                "TotalActual":             round(tot_act, 2),
                "TotalForecastRaw":        round(tot_raw, 2),
                "TotalForecastCalibrated": round(tot_cal, 2),
                "RawBiasRatio":            raw_bias,
                "CalibratedBiasRatio":     cal_bias,
                "BiasImprovement":         improvement,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Level","HorizonMonths","MonthStart",
            "TotalActual","TotalForecastRaw","TotalForecastCalibrated",
            "RawBiasRatio","CalibratedBiasRatio","BiasImprovement",
        ])

    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public: validate_calibration_table
# ---------------------------------------------------------------------------

def validate_calibration_table(
    calibration_df: pd.DataFrame,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
    backtest_end: pd.Timestamp | None = None,
    backtest_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run integrity checks on the v7.5 calibration table.

    Checks:
      - no duplicate (StyleCodeDesc, HorizonMonths) keys
      - all calibration_factor in [min_factor, max_factor]
      - no leakage (if backtest_df provided, ForecastMonths ≤ backtest_end)

    Returns dict with check results and any warning messages.
    """
    warnings: list[str] = []

    if calibration_df.empty:
        return {
            "n_rows": 0, "n_applied": 0, "n_by_tier": {},
            "factors_in_range": True, "has_duplicate_keys": False,
            "leakage_check_passed": True,
            "warnings": ["Calibration table is empty — all factors will be 1.0"],
        }

    n_rows    = len(calibration_df)
    n_applied = int(calibration_df["calibration_applied"].sum()) \
        if "calibration_applied" in calibration_df.columns else 0

    n_by_tier = {}
    if "evidence_tier" in calibration_df.columns:
        n_by_tier = calibration_df["evidence_tier"].value_counts().to_dict()

    # Factor range check
    in_range = True
    if "calibration_factor" in calibration_df.columns:
        out = (
            (calibration_df["calibration_factor"] < min_factor - 1e-6) |
            (calibration_df["calibration_factor"] > max_factor + 1e-6)
        )
        if out.any():
            in_range = False
            warnings.append(f"{out.sum()} factors outside [{min_factor}, {max_factor}]")

    # Duplicate key check
    has_dup = False
    if SCODE_COL in calibration_df.columns and "HorizonMonths" in calibration_df.columns:
        dup = calibration_df.duplicated(subset=[SCODE_COL, "HorizonMonths"], keep=False)
        if dup.any():
            has_dup = True
            warnings.append(f"{dup.sum()} duplicate (StyleCodeDesc, HorizonMonths) keys")

    # Leakage check
    leakage_ok = True
    if backtest_df is not None and backtest_end is not None and "ForecastMonth" in backtest_df.columns:
        fm_ts = pd.to_datetime(backtest_df["ForecastMonth"].astype(str) + "-01")
        future = (fm_ts > pd.Timestamp(backtest_end)).sum()
        if future > 0:
            leakage_ok = False
            warnings.append(f"{future} backtest rows have ForecastMonth > backtest_end — LEAKAGE!")

    return {
        "n_rows":              n_rows,
        "n_applied":           n_applied,
        "n_by_tier":           n_by_tier,
        "factors_in_range":    in_range,
        "has_duplicate_keys":  has_dup,
        "leakage_check_passed":leakage_ok,
        "warnings":            warnings,
    }
