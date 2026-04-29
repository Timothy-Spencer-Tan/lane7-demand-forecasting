"""
lane7_forecast.forecast_calibration_v74
=========================================
v7.4 StyleCode-level forecast calibration layer.

Purpose
-------
H3 February error analysis (v7.3) showed that the dominant error source for
February is upstream StyleCode forecast bias, not allocation.  Allocation
improvements alone cannot fix a biased parent forecast.

This module adds a light post-model, pre-allocation calibration step:
  1. Measure historical forecast bias at the StyleCode × HorizonMonths level
     using ONLY backtest/CV data available before the holdout window.
  2. Compute a calibration factor = sum(actual) / sum(predicted) within that
     evidence window.
  3. Apply the factor only when there is sufficient evidence.
  4. Cap the factor within [MIN_FACTOR, MAX_FACTOR] to prevent over-correction.

No leakage guarantee
--------------------
The calibration factors are built from data with MonthStart < holdout_start.
For the Jan–Feb 2026 holdout evaluation, all calibration data must have
MonthStart ≤ 2025-12.  This is enforced by the `backtest_end` parameter.

Evidence tiers
--------------
  STRONG   : n_observations ≥ STRONG_THRESHOLD  → factor applied
  MODERATE : n_observations ≥ MODERATE_THRESHOLD → factor applied
  WEAK     : n_observations ≥ WEAK_THRESHOLD     → factor = 1.0 (not applied)
  NONE     : n_observations < WEAK_THRESHOLD     → factor = 1.0 (not applied)

Default thresholds: STRONG=8, MODERATE=4, WEAK=2.

Safe calibration range
-----------------------
MIN_FACTOR = 0.80  (maximum downward correction: 20%)
MAX_FACTOR = 1.20  (maximum upward correction: 20%)
Factor is clamped to this range even for STRONG evidence.

Fallback calibration
--------------------
If CV fold predictions are not available, a fallback calibration is built
from the training actuals: for each StyleCode, we compute whether the
seasonal-naive 3-month-ahead forecast is systematically biased relative to
actuals over the last N months.  This is less precise than CV-based calibration
and always produces WEAK or NONE evidence.

Public API
----------
    build_stylecode_calibration_table(
        backtest_predictions_df,   # CV fold predictions (backtest only, no holdout)
        gold_df,                   # gold demand (for fallback path)
        dim_product_df,            # for StyleCode mapping
        backtest_end,              # pd.Timestamp — last allowed date
        horizon_months_list=[1,3], # horizons to calibrate
        **kwargs                   # threshold / factor overrides
    ) -> calibration_df

    apply_stylecode_calibration(
        scode_forecasts_df,        # raw StyleCode forecasts
        calibration_df,            # output of build_stylecode_calibration_table
    ) -> calibrated_df             # same schema, ForecastUnits adjusted

    validate_calibration_table(calibration_df) -> dict
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"
SCODE_COL  = "StyleCodeDesc"

MIN_FACTOR      = 0.80
MAX_FACTOR      = 1.20
STRONG_THRESHOLD   = 8
MODERATE_THRESHOLD = 4
WEAK_THRESHOLD     = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evidence_tier(n: int) -> str:
    if n >= STRONG_THRESHOLD:
        return "STRONG"
    if n >= MODERATE_THRESHOLD:
        return "MODERATE"
    if n >= WEAK_THRESHOLD:
        return "WEAK"
    return "NONE"


def _safe_factor(raw: float, tier: str) -> tuple[float, bool]:
    """Return (clamped_factor, calibration_applied)."""
    if tier not in ("STRONG", "MODERATE"):
        return 1.0, False
    clamped = float(np.clip(raw, MIN_FACTOR, MAX_FACTOR))
    return clamped, True


# ---------------------------------------------------------------------------
# Path A — CV-based calibration (preferred, no leakage)
# ---------------------------------------------------------------------------

def _calibrate_from_cv(
    backtest_df: pd.DataFrame,
    backtest_end: pd.Timestamp,
    horizon_months_list: list[int],
    scode_col: str,
) -> pd.DataFrame:
    """
    Build calibration factors from cross-validation fold predictions.

    Parameters
    ----------
    backtest_df         : DataFrame with columns [scode_col, MonthStart,
                          HorizonMonths, ActualUnits, PredictedUnits]
                          — these are fold-level predictions, never holdout rows.
    backtest_end        : all rows with MonthStart > backtest_end are dropped.
    horizon_months_list : horizons to compute factors for.
    scode_col           : parent key column (StyleCodeDesc).

    Returns
    -------
    pd.DataFrame with one row per (scode, horizon), columns as per spec.
    """
    df = backtest_df.copy()
    df["MonthStart"] = pd.to_datetime(df["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    df = df[df["MonthStart"] <= backtest_end]

    # Require both actual and predicted
    df = df.dropna(subset=["ActualUnits", "PredictedUnits"])
    df = df[df["PredictedUnits"] > 0]   # avoid division by zero rows

    rows = []
    for horizon in horizon_months_list:
        h_df = df[df["HorizonMonths"] == horizon] if "HorizonMonths" in df.columns else df

        for scode, grp in h_df.groupby(scode_col):
            n_obs       = len(grp)
            total_act   = float(grp["ActualUnits"].sum())
            total_pred  = float(grp["PredictedUnits"].sum())
            raw_factor  = total_act / total_pred if total_pred > 0 else 1.0
            raw_bias    = total_pred / total_act if total_act > 0 else np.nan
            tier        = _evidence_tier(n_obs)
            factor, applied = _safe_factor(raw_factor, tier)

            rows.append({
                SCODE_COL:              scode,
                "HorizonMonths":        int(horizon),
                "MonthStart":           "all_backtest",
                "raw_bias_ratio":       round(raw_bias, 4) if not np.isnan(raw_bias) else None,
                "calibration_factor":   round(factor, 4),
                "n_observations":       n_obs,
                "total_actual_units":   round(total_act, 2),
                "total_forecast_units": round(total_pred, 2),
                "evidence_tier":        tier,
                "calibration_applied":  applied,
                "reason_code":          "cv_backtest",
                "calibration_source":   "cv_fold_predictions",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Path B — Fallback calibration from training history
# ---------------------------------------------------------------------------

def _calibrate_from_history(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    backtest_end: pd.Timestamp,
    horizon_months_list: list[int],
    lookback_months: int = 24,
) -> pd.DataFrame:
    """
    Fallback calibration using seasonal-naive bias in the training window.

    For each StyleCode and HorizonMonths H, we compute:
        naive_forecast[t] = actual[t - H]      (H-step seasonal-naive)
        bias_ratio        = sum(actual[t]) / sum(naive_forecast[t])
                            for all t in the lookback window

    This is a weak signal (always WEAK or NONE tier) but prevents factor=1.0
    from being the only option when CV data is missing.
    """
    gold = gold_df.copy()
    gold["MonthStart"] = pd.to_datetime(gold["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]      = gold[SKU_COL].astype(str).str.strip()
    gold = gold[gold["MonthStart"] <= backtest_end].copy()

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)

    # Aggregate to StyleCode level
    gold = gold.merge(dp, on=SKU_COL, how="inner")
    scode_monthly = (
        gold.groupby([SCODE_COL, "MonthStart"])[TARGET_COL]
        .sum().reset_index().rename(columns={TARGET_COL: "actual"})
    )

    lookback_start = backtest_end - pd.DateOffset(months=lookback_months - 1)
    lookback_start = pd.Timestamp(lookback_start.year, lookback_start.month, 1)

    rows = []

    for horizon in horizon_months_list:
        for scode, grp in scode_monthly.groupby(SCODE_COL):
            grp = grp.set_index("MonthStart").sort_index()
            # Only use rows within the lookback window
            window = grp[grp.index >= lookback_start]["actual"]
            if len(window) < 2:
                rows.append({
                    SCODE_COL: scode, "HorizonMonths": horizon,
                    "MonthStart": "all_backtest",
                    "raw_bias_ratio": None, "calibration_factor": 1.0,
                    "n_observations": 0, "total_actual_units": 0.0,
                    "total_forecast_units": 0.0, "evidence_tier": "NONE",
                    "calibration_applied": False, "reason_code": "insufficient_history",
                    "calibration_source": "historical_fallback",
                })
                continue

            # Align actual[t] with naive_forecast[t] = actual[t - horizon months]
            pairs = []
            for ts, act_val in window.items():
                lag_ts = ts - pd.DateOffset(months=horizon)
                lag_ts = pd.Timestamp(lag_ts.year, lag_ts.month, 1)
                if lag_ts in grp.index:
                    pairs.append((act_val, grp.at[lag_ts, "actual"]))

            if len(pairs) < WEAK_THRESHOLD:
                tier = "NONE"
            elif len(pairs) < MODERATE_THRESHOLD:
                tier = "WEAK"
            else:
                tier = "WEAK"   # historical fallback is always capped at WEAK

            n_obs      = len(pairs)
            total_act  = sum(a for a, _ in pairs)
            total_pred = sum(p for _, p in pairs)
            raw_factor = total_act / total_pred if total_pred > 0 else 1.0
            raw_bias   = total_pred / total_act if total_act > 0 else np.nan
            factor, applied = _safe_factor(raw_factor, tier)  # WEAK → 1.0

            rows.append({
                SCODE_COL:              scode,
                "HorizonMonths":        int(horizon),
                "MonthStart":           "all_backtest",
                "raw_bias_ratio":       round(raw_bias, 4) if not np.isnan(raw_bias) else None,
                "calibration_factor":   round(factor, 4),
                "n_observations":       n_obs,
                "total_actual_units":   round(total_act, 2),
                "total_forecast_units": round(total_pred, 2),
                "evidence_tier":        tier,
                "calibration_applied":  applied,
                "reason_code":          "historical_naive_bias",
                "calibration_source":   "historical_fallback",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public: build_stylecode_calibration_table
# ---------------------------------------------------------------------------

def build_stylecode_calibration_table(
    backtest_predictions_df: pd.DataFrame | None,
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    backtest_end: pd.Timestamp,
    horizon_months_list: list[int] | None = None,
    lookback_months: int = 24,
    strong_threshold: int   = STRONG_THRESHOLD,
    moderate_threshold: int = MODERATE_THRESHOLD,
    weak_threshold: int     = WEAK_THRESHOLD,
    min_factor: float = MIN_FACTOR,
    max_factor: float = MAX_FACTOR,
) -> pd.DataFrame:
    """
    Build a StyleCode-level calibration table using backtest data only.

    No leakage: all data used must have MonthStart ≤ backtest_end.
    For the Jan–Feb 2026 holdout evaluation, backtest_end = 2025-12-01.

    Parameters
    ----------
    backtest_predictions_df : CV fold predictions DataFrame with columns:
        [StyleCodeDesc, MonthStart, HorizonMonths, ActualUnits, PredictedUnits]
        Pass None to fall through to the historical fallback path.
    gold_df          : gold_fact_monthly_demand_v2 (for fallback calibration)
    dim_product_df   : dim_product (for SKU→StyleCode mapping in fallback)
    backtest_end     : last allowed date for calibration evidence
    horizon_months_list : horizons to calibrate (default [1, 3])
    lookback_months  : fallback lookback window length (default 24)
    strong/moderate/weak_threshold : evidence tier cutoffs
    min_factor / max_factor : allowed calibration factor range

    Returns
    -------
    pd.DataFrame — one row per (StyleCodeDesc, HorizonMonths), columns:
        StyleCodeDesc, HorizonMonths, MonthStart, raw_bias_ratio,
        calibration_factor, n_observations, total_actual_units,
        total_forecast_units, evidence_tier, calibration_applied,
        reason_code, calibration_source
    """
    global STRONG_THRESHOLD, MODERATE_THRESHOLD, WEAK_THRESHOLD, MIN_FACTOR, MAX_FACTOR
    STRONG_THRESHOLD   = strong_threshold
    MODERATE_THRESHOLD = moderate_threshold
    WEAK_THRESHOLD     = weak_threshold
    MIN_FACTOR         = min_factor
    MAX_FACTOR         = max_factor

    horizon_months_list = horizon_months_list or [1, 3]
    backtest_end        = pd.Timestamp(backtest_end)

    cv_calib = pd.DataFrame()

    # ── Path A: CV backtest predictions ──────────────────────────────────
    if backtest_predictions_df is not None and not backtest_predictions_df.empty:
        bp = backtest_predictions_df.copy()

        # Detect schema variants
        if "Predicted" in bp.columns and "Actual" in bp.columns:
            bp = bp.rename(columns={"Predicted": "PredictedUnits", "Actual": "ActualUnits"})
        if "PredictedUnits" not in bp.columns or "ActualUnits" not in bp.columns:
            logger.warning(
                "[v7.4 calib] backtest_predictions_df missing PredictedUnits/ActualUnits "
                "— falling through to historical fallback."
            )
        elif SCODE_COL not in bp.columns:
            # Try joining via dim_product
            dp = dim_product_df.copy()
            dp["SKU"] = dp["SKU"].astype(str).str.strip()
            dp_sc = dp[[SCODE_COL, "SKU"]].dropna().drop_duplicates("SKU")
            if "Key" in bp.columns:
                bp = bp.rename(columns={"Key": "SKU"})
            elif "SKU" in bp.columns:
                pass
            else:
                logger.warning("[v7.4 calib] Cannot identify SKU column in backtest_df.")
                bp = pd.DataFrame()
            if not bp.empty:
                bp["SKU"] = bp["SKU"].astype(str).str.strip()
                bp = bp.merge(dp_sc, on="SKU", how="left")
                bp = bp[bp[SCODE_COL].notna()]

        if not bp.empty and SCODE_COL in bp.columns:
            cv_calib = _calibrate_from_cv(
                backtest_df=bp,
                backtest_end=backtest_end,
                horizon_months_list=horizon_months_list,
                scode_col=SCODE_COL,
            )
            if not cv_calib.empty:
                n_strong   = (cv_calib["evidence_tier"] == "STRONG").sum()
                n_moderate = (cv_calib["evidence_tier"] == "MODERATE").sum()
                n_applied  = cv_calib["calibration_applied"].sum()
                logger.info(
                    "[v7.4 calib] CV path: %d StyleCodes, %d STRONG, %d MODERATE, "
                    "%d calibration_applied",
                    len(cv_calib), n_strong, n_moderate, n_applied,
                )

    # ── Path B: Historical fallback ───────────────────────────────────────
    hist_calib = _calibrate_from_history(
        gold_df=gold_df,
        dim_product_df=dim_product_df,
        backtest_end=backtest_end,
        horizon_months_list=horizon_months_list,
        lookback_months=lookback_months,
    )

    # ── Merge: prefer CV where available, fill with fallback ───────────────
    if cv_calib.empty:
        logger.info(
            "[v7.4 calib] No CV-based calibration available — using historical fallback only."
        )
        result = hist_calib
    else:
        # Use CV for StyleCodes that have it; fallback for the rest
        cv_keys  = set(zip(cv_calib[SCODE_COL], cv_calib["HorizonMonths"]))
        hist_extra = hist_calib[
            ~hist_calib.apply(
                lambda r: (r[SCODE_COL], r["HorizonMonths"]) in cv_keys, axis=1
            )
        ]
        result = pd.concat([cv_calib, hist_extra], ignore_index=True)
        logger.info(
            "[v7.4 calib] Final table: %d rows (%d from CV, %d from fallback)",
            len(result), len(cv_calib), len(hist_extra),
        )

    if result.empty:
        logger.warning("[v7.4 calib] Calibration table is empty — no factors will be applied.")
        return pd.DataFrame(columns=[
            SCODE_COL, "HorizonMonths", "MonthStart",
            "raw_bias_ratio", "calibration_factor",
            "n_observations", "total_actual_units", "total_forecast_units",
            "evidence_tier", "calibration_applied", "reason_code", "calibration_source",
        ])

    return result.sort_values([SCODE_COL, "HorizonMonths"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public: apply_stylecode_calibration
# ---------------------------------------------------------------------------

def apply_stylecode_calibration(
    scode_forecasts_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply calibration factors to StyleCode-level ForecastUnits.

    For each (StyleCodeDesc, HorizonMonths) row in scode_forecasts_df, the
    matching calibration factor is looked up and applied.  Rows with no
    matching calibration entry are left unchanged (factor = 1.0).

    Lower and Upper bounds are scaled by the same factor.

    Parameters
    ----------
    scode_forecasts_df : raw StyleCode forecasts (Key = StyleCodeDesc)
    calibration_df     : output of build_stylecode_calibration_table()

    Returns
    -------
    pd.DataFrame — same schema as input with:
        ForecastUnits, Lower, Upper adjusted
        CalibrationFactor column added
        CalibrationApplied column added
        ModelVersion suffixed with "+calib" where applied
    """
    fc = scode_forecasts_df.copy()
    if fc.empty:
        return fc

    # Rename Key → StyleCodeDesc for join
    key_col = "Key" if "Key" in fc.columns else SCODE_COL
    if key_col == "Key":
        fc = fc.rename(columns={"Key": SCODE_COL})

    # Build lookup: (StyleCodeDesc, HorizonMonths) → (factor, applied)
    if calibration_df.empty:
        fc["CalibrationFactor"]  = 1.0
        fc["CalibrationApplied"] = False
        fc = fc.rename(columns={SCODE_COL: "Key"}) if key_col == "Key" else fc
        return fc

    calib_lookup = calibration_df.set_index([SCODE_COL, "HorizonMonths"])[
        ["calibration_factor", "calibration_applied"]
    ]

    factors  = []
    applied  = []

    for _, row in fc.iterrows():
        lookup_key = (row.get(SCODE_COL, ""), row.get("HorizonMonths", -1))
        if lookup_key in calib_lookup.index:
            f = float(calib_lookup.at[lookup_key, "calibration_factor"])
            a = bool(calib_lookup.at[lookup_key, "calibration_applied"])
        else:
            f = 1.0
            a = False
        factors.append(f)
        applied.append(a)

    fc["CalibrationFactor"]  = factors
    fc["CalibrationApplied"] = applied

    # Apply factors
    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in fc.columns:
            fc[col] = (fc[col] * fc["CalibrationFactor"]).clip(lower=0).round(4)

    # Tag ModelVersion where calibration was applied
    if "ModelVersion" in fc.columns:
        fc["ModelVersion"] = fc.apply(
            lambda r: (r["ModelVersion"] + "+calib")
            if r["CalibrationApplied"] and "+calib" not in str(r.get("ModelVersion",""))
            else r["ModelVersion"],
            axis=1,
        )

    # Rename back to Key
    if key_col == "Key":
        fc = fc.rename(columns={SCODE_COL: "Key"})

    n_applied = sum(applied)
    logger.info(
        "[v7.4 calib] Applied calibration: %d / %d rows had factor ≠ 1.0",
        n_applied, len(fc),
    )
    return fc


# ---------------------------------------------------------------------------
# Public: validate_calibration_table
# ---------------------------------------------------------------------------

def validate_calibration_table(calibration_df: pd.DataFrame) -> dict:
    """
    Run integrity checks on the calibration table.

    Returns
    -------
    dict with:
        n_rows, n_applied, n_by_tier (dict),
        factors_in_range (bool), has_duplicate_keys (bool),
        any_warnings (list[str])
    """
    warnings = []

    if calibration_df.empty:
        return {
            "n_rows": 0, "n_applied": 0,
            "n_by_tier": {}, "factors_in_range": True,
            "has_duplicate_keys": False,
            "any_warnings": ["Calibration table is empty — all factors will be 1.0"],
        }

    n_rows    = len(calibration_df)
    n_applied = int(calibration_df["calibration_applied"].sum()) \
        if "calibration_applied" in calibration_df.columns else 0

    n_by_tier = {}
    if "evidence_tier" in calibration_df.columns:
        n_by_tier = calibration_df["evidence_tier"].value_counts().to_dict()

    factors_in_range = True
    if "calibration_factor" in calibration_df.columns:
        out_of_range = (
            (calibration_df["calibration_factor"] < MIN_FACTOR - 1e-6) |
            (calibration_df["calibration_factor"] > MAX_FACTOR + 1e-6)
        )
        if out_of_range.any():
            factors_in_range = False
            warnings.append(
                f"{out_of_range.sum()} calibration factors outside [{MIN_FACTOR}, {MAX_FACTOR}]"
            )

    has_dup = False
    if SCODE_COL in calibration_df.columns and "HorizonMonths" in calibration_df.columns:
        dup_mask = calibration_df.duplicated(subset=[SCODE_COL, "HorizonMonths"], keep=False)
        if dup_mask.any():
            has_dup = True
            warnings.append(
                f"{dup_mask.sum()} duplicate (StyleCodeDesc, HorizonMonths) keys in calibration table"
            )

    logger.info(
        "[v7.4 calib] Validation: n_rows=%d  n_applied=%d  in_range=%s  dup=%s  tiers=%s",
        n_rows, n_applied, factors_in_range, has_dup, n_by_tier,
    )

    return {
        "n_rows":          n_rows,
        "n_applied":       n_applied,
        "n_by_tier":       n_by_tier,
        "factors_in_range":factors_in_range,
        "has_duplicate_keys": has_dup,
        "any_warnings":    warnings,
    }
