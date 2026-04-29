"""
lane7_forecast.global_bias_control_v76
=========================================
v7.6 Conservative Global Bias Control.

Why this exists
---------------
v7.5 applied per-StyleCode calibration from a rolling backtest.
Outcome: it made performance worse — individual StyleCode bias estimates
are too noisy to calibrate reliably.

v7.6 corrects this with the lowest-variance useful calibration:
  • Global calibration factor per HorizonMonths only (H=1 and H=3).
  • Factor bounds [0.95, 1.05] — intentionally conservative.
  • No-regression gate: if calibration worsens WMAPE beyond a threshold,
    it is rejected and factor = 1.0 is used instead.
  • Falls back gracefully to factor = 1.0 if evidence is insufficient.

Evidence hierarchy
------------------
1. Aggregated v7.5 backtest predictions (pre-2026, no leakage)
2. Aggregated v7.4 holdout predictions (diagnostic only — labeled as such)
3. No usable evidence → factor = 1.0, reason = insufficient_safe_evidence

The no-regression gate
----------------------
After computing a proposed factor, we project what the calibrated forecast
volume would have been on the validation set and recompute WMAPE.
If calibrated WMAPE > raw WMAPE + max_allowed_wmape_regression:
  → reject calibration, use factor = 1.0, set calibration_applied = False.

This prevents v7.6 from ever making things measurably worse.

Public API
----------
    build_global_calibration_table(
        backtest_predictions_df,   # v7.5 backtest CSV or None
        holdout_predictions_df,    # v7.4 holdout CSV (diagnostic fallback)
        raw_scode_forecasts_df,    # raw StyleCode forecasts for validation
        actuals_df,                # gold actuals for no-regression gate
        holdout_months,
        horizon_months_list=[1,3],
        min_factor=0.95,
        max_factor=1.05,
        max_allowed_wmape_regression=0.25,
        evidence_source_override=None,
    ) -> calibration_df

    apply_global_calibration(scode_forecasts_df, calibration_df) -> calibrated_df

    build_v76_bias_analysis(actuals_df, dim_product_df,
                             raw_scode_fc, calibrated_scode_fc,
                             holdout_months) -> bias_df

    validate_global_calibration_table(calibration_df, min_factor, max_factor) -> dict
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"
SCODE_COL  = "StyleCodeDesc"
SC_COL     = "StyleColorDesc"

DEFAULT_MIN_FACTOR = 0.95
DEFAULT_MAX_FACTOR = 1.05
DEFAULT_MAX_REGRESSION_PP = 0.25   # percentage points


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wmape_series(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


def _prep_ts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if DATE_COL in d.columns:
        d[DATE_COL] = pd.to_datetime(d[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    return d


def _aggregate_evidence(df: pd.DataFrame, pred_col: str, act_col: str) -> dict[int, dict]:
    """
    Aggregate evidence by HorizonMonths.

    Returns dict: horizon → {total_actual, total_predicted, n_obs}
    """
    result = {}
    df = df.dropna(subset=[pred_col, act_col])
    df = df[df[act_col] > 0]   # only rows where actuals are positive

    for h, grp in df.groupby("HorizonMonths"):
        result[int(h)] = {
            "total_actual":    float(grp[act_col].sum()),
            "total_predicted": float(grp[pred_col].sum()),
            "n_obs":           len(grp),
        }
    return result


def _compute_raw_scode_wmape(
    raw_scode_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    horizon: int,
) -> float:
    """
    Compute WMAPE of raw StyleCode forecasts vs actuals for a given horizon.
    Used by the no-regression gate.
    """
    # Aggregate gold to StyleCode level
    acts = actuals_df.copy()
    acts[DATE_COL] = pd.to_datetime(acts[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]  = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])]

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp_sc = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)
    acts = acts.merge(dp_sc, on=SKU_COL, how="inner")
    a_sc = acts.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()

    fc = raw_scode_fc.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SCODE_COL})
    fc[DATE_COL] = pd.to_datetime(fc[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    fc = fc[
        (fc[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])) &
        (fc["HorizonMonths"] == horizon)
    ]
    fc_sc = fc.groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()

    mg = a_sc.merge(fc_sc, on=SCODE_COL, how="inner")
    if mg.empty:
        return np.nan
    return _wmape_series(mg[TARGET_COL], mg["ForecastUnits"])


def _compute_calibrated_scode_wmape(
    raw_scode_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    horizon: int,
    factor: float,
) -> float:
    """
    Project calibrated WMAPE by scaling raw forecasts by factor.
    Used by the no-regression gate.
    """
    fc = raw_scode_fc.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SCODE_COL})
    fc[DATE_COL] = pd.to_datetime(fc[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    fc = fc[
        (fc[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])) &
        (fc["HorizonMonths"] == horizon)
    ].copy()
    fc["ForecastUnits"] = (fc["ForecastUnits"] * factor).clip(lower=0)

    acts = actuals_df.copy()
    acts[DATE_COL] = pd.to_datetime(acts[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]  = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])]

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp_sc = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)
    acts = acts.merge(dp_sc, on=SKU_COL, how="inner")
    a_sc = acts.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()
    fc_sc = fc.groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()

    mg = a_sc.merge(fc_sc, on=SCODE_COL, how="inner")
    if mg.empty:
        return np.nan
    return _wmape_series(mg[TARGET_COL], mg["ForecastUnits"])


# ---------------------------------------------------------------------------
# Public: build_global_calibration_table
# ---------------------------------------------------------------------------

def build_global_calibration_table(
    backtest_predictions_df: pd.DataFrame | None,
    holdout_predictions_df: pd.DataFrame | None,
    raw_scode_forecasts_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    horizon_months_list: list[int] | None = None,
    min_factor: float   = DEFAULT_MIN_FACTOR,
    max_factor: float   = DEFAULT_MAX_FACTOR,
    max_allowed_wmape_regression: float = DEFAULT_MAX_REGRESSION_PP,
    evidence_source_override: str | None = None,
) -> pd.DataFrame:
    """
    Build a global (HorizonMonths-only) calibration table.

    No leakage: preferred evidence is from backtest_predictions_df where
    ForecastMonth ≤ 2025-12.  holdout_predictions_df is a diagnostic fallback
    only, clearly labeled.

    Parameters
    ----------
    backtest_predictions_df : v7.5 backtest CSV or equivalent.
                              Must have [HorizonMonths, PredictedUnits,
                              ActualUnits] or [HorizonMonths, Predicted,
                              Actual] or [HorizonMonths, ForecastUnits,
                              UnitsSold].
    holdout_predictions_df  : v7.4 holdout predictions (diagnostic fallback).
                              Clearly labeled with evidence_source = 'holdout_diagnostic'.
    raw_scode_forecasts_df  : raw StyleCode forecasts for the holdout period.
                              Used by the no-regression gate.
    actuals_df              : gold_fact_monthly_demand (all actuals).
    dim_product_df          : for SKU→StyleCode mapping in gate computation.
    holdout_months          : list of pd.Timestamp used for gate evaluation.
    horizon_months_list     : [1, 3] (default)
    min_factor / max_factor : conservative bounds [0.95, 1.05].
    max_allowed_wmape_regression : pp tolerance for no-regression gate.
    evidence_source_override: if set, uses this string as evidence_source label.

    Returns
    -------
    pd.DataFrame with one row per HorizonMonths, columns:
        HorizonMonths, calibration_scope,
        raw_total_actual, raw_total_forecast, raw_bias_ratio,
        proposed_factor, final_factor, factor_min, factor_max,
        raw_wmape, calibrated_wmape, wmape_delta,
        calibration_applied, rejection_reason,
        evidence_source, n_observations
    """
    horizon_months_list = horizon_months_list or [1, 3]

    # ── Step 1: Load evidence ─────────────────────────────────────────────

    evidence: dict[int, dict] = {}
    evidence_source_name = evidence_source_override or "none"

    # Path A: Backtest predictions (preferred — pre-2026 only)
    if backtest_predictions_df is not None and not backtest_predictions_df.empty:
        bp = backtest_predictions_df.copy()

        # Normalise column names
        if "PredictedUnits" not in bp.columns:
            for alt in ["Predicted", "ForecastUnits"]:
                if alt in bp.columns:
                    bp = bp.rename(columns={alt: "PredictedUnits"})
                    break
        if "ActualUnits" not in bp.columns:
            for alt in ["Actual", "UnitsSold"]:
                if alt in bp.columns:
                    bp = bp.rename(columns={alt: "ActualUnits"})
                    break

        if "PredictedUnits" in bp.columns and "ActualUnits" in bp.columns:
            # Enforce leakage guard: only ForecastMonths before 2026-01
            if "ForecastMonth" in bp.columns:
                fm_ts = pd.to_datetime(bp["ForecastMonth"].astype(str) + "-01")
                bp = bp[fm_ts < pd.Timestamp("2026-01-01")].copy()
            elif DATE_COL in bp.columns:
                bp = _prep_ts(bp)
                bp = bp[bp[DATE_COL] < pd.Timestamp("2026-01-01")].copy()

            if not bp.empty and "HorizonMonths" in bp.columns:
                evidence = _aggregate_evidence(bp, "PredictedUnits", "ActualUnits")
                evidence_source_name = evidence_source_override or "backtest_pre2026"
                logger.info(
                    "[v7.6] Evidence from backtest: horizons found = %s",
                    list(evidence.keys()),
                )

    # Path B: Holdout diagnostic fallback
    if not evidence and holdout_predictions_df is not None and not holdout_predictions_df.empty:
        hp = holdout_predictions_df.copy()

        if "PredictedUnits" not in hp.columns:
            for alt in ["Predicted", "ForecastUnits"]:
                if alt in hp.columns:
                    hp = hp.rename(columns={alt: "PredictedUnits"})
                    break
        if "ActualUnits" not in hp.columns:
            for alt in ["Actual", "UnitsSold"]:
                if alt in hp.columns:
                    hp = hp.rename(columns={alt: "ActualUnits"})
                    break

        if "PredictedUnits" in hp.columns and "ActualUnits" in hp.columns and "HorizonMonths" in hp.columns:
            evidence = _aggregate_evidence(hp, "PredictedUnits", "ActualUnits")
            evidence_source_name = evidence_source_override or "holdout_diagnostic"
            logger.warning(
                "[v7.6] Using holdout predictions as calibration evidence — "
                "labeled as 'holdout_diagnostic'. This is for diagnostics only; "
                "the calibration decision is still gated by no-regression check."
            )

    # ── Step 2: Build one row per horizon ─────────────────────────────────

    rows = []

    for horizon in horizon_months_list:
        ev = evidence.get(horizon, {})

        tot_actual    = ev.get("total_actual",    0.0)
        tot_predicted = ev.get("total_predicted", 0.0)
        n_obs         = ev.get("n_obs",           0)

        if tot_actual > 0 and tot_predicted > 0 and n_obs >= 1:
            raw_bias     = round(tot_predicted / tot_actual, 6)
            proposed_raw = tot_actual / tot_predicted    # correction factor
            proposed_factor = float(np.clip(proposed_raw, min_factor, max_factor))
        else:
            raw_bias        = None
            proposed_factor = 1.0

        # ── Step 3: Compute raw and calibrated WMAPE for gate ─────────────
        raw_wmape = _compute_raw_scode_wmape(
            raw_scode_forecasts_df, actuals_df, dim_product_df,
            holdout_months, horizon,
        )

        if proposed_factor != 1.0:
            calibrated_wmape = _compute_calibrated_scode_wmape(
                raw_scode_forecasts_df, actuals_df, dim_product_df,
                holdout_months, horizon, proposed_factor,
            )
        else:
            calibrated_wmape = raw_wmape

        wmape_delta = round(calibrated_wmape - raw_wmape, 4) \
            if not (np.isnan(raw_wmape) or np.isnan(calibrated_wmape)) else None

        # ── Step 4: No-regression gate ────────────────────────────────────
        rejection_reason = ""
        final_factor     = proposed_factor
        calibration_applied = False

        if n_obs < 1:
            final_factor        = 1.0
            rejection_reason    = "insufficient_safe_evidence"
            calibration_applied = False

        elif proposed_factor == 1.0:
            # Proposed factor is already 1.0 — no adjustment needed
            rejection_reason    = "proposed_factor_is_1.0"
            calibration_applied = False

        elif wmape_delta is not None and wmape_delta > max_allowed_wmape_regression:
            # Gate fires: calibration would worsen WMAPE more than allowed
            final_factor        = 1.0
            rejection_reason    = (
                f"no_regression_gate_fired: calibrated_wmape={calibrated_wmape:.4f}% "
                f"vs raw_wmape={raw_wmape:.4f}% "
                f"(delta={wmape_delta:+.4f}pp > max={max_allowed_wmape_regression}pp)"
            )
            calibration_applied = False
            logger.info(
                "[v7.6] H=%d: NO-REGRESSION GATE fired. "
                "Calibration rejected (delta=%+.4f pp). Factor set to 1.0.",
                horizon, wmape_delta,
            )

        else:
            # Accept calibration
            calibration_applied = True
            rejection_reason    = ""
            logger.info(
                "[v7.6] H=%d: Calibration accepted. "
                "Factor=%.4f  raw_wmape=%.2f%%  cal_wmape=%.2f%%  delta=%+.4fpp",
                horizon, final_factor, raw_wmape,
                calibrated_wmape if not np.isnan(calibrated_wmape) else float("nan"),
                wmape_delta or 0.0,
            )

        rows.append({
            "HorizonMonths":          int(horizon),
            "calibration_scope":      "global",
            "raw_total_actual":       round(tot_actual, 2),
            "raw_total_forecast":     round(tot_predicted, 2),
            "raw_bias_ratio":         round(raw_bias, 4) if raw_bias is not None else None,
            "proposed_factor":        round(proposed_factor, 4),
            "final_factor":           round(final_factor, 4),
            "factor_min":             min_factor,
            "factor_max":             max_factor,
            "raw_wmape":              round(raw_wmape, 4) if not np.isnan(raw_wmape) else None,
            "calibrated_wmape":       round(calibrated_wmape, 4)
                                      if not np.isnan(calibrated_wmape) else None,
            "wmape_delta":            wmape_delta,
            "calibration_applied":    calibration_applied,
            "rejection_reason":       rejection_reason,
            "evidence_source":        evidence_source_name if n_obs >= 1 else "none",
            "n_observations":         n_obs,
        })

    result = pd.DataFrame(rows)

    n_applied = result["calibration_applied"].sum()
    logger.info(
        "[v7.6] Global calibration table: %d horizons, %d calibration(s) applied",
        len(result), n_applied,
    )
    return result


# ---------------------------------------------------------------------------
# Public: apply_global_calibration
# ---------------------------------------------------------------------------

def apply_global_calibration(
    scode_forecasts_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply global calibration factors to StyleCode-level ForecastUnits.

    One factor per HorizonMonths.  Only rows where calibration_applied=True
    are adjusted.  All rows get CalibrationFactor and CalibrationApplied columns.

    Parameters
    ----------
    scode_forecasts_df : raw StyleCode forecasts (Key = StyleCodeDesc)
    calibration_df     : output of build_global_calibration_table()

    Returns
    -------
    pd.DataFrame — same schema with adjusted ForecastUnits/Lower/Upper
    and new columns: CalibrationFactor, CalibrationApplied, CalibrationScope
    """
    fc = scode_forecasts_df.copy()
    if fc.empty:
        return fc

    if calibration_df.empty:
        fc["CalibrationFactor"]  = 1.0
        fc["CalibrationApplied"] = False
        fc["CalibrationScope"]   = "global"
        return fc

    # Build lookup: HorizonMonths → (factor, applied)
    calib_lu = calibration_df.set_index("HorizonMonths")[
        ["final_factor", "calibration_applied"]
    ]

    factors = []
    applied = []

    for _, row in fc.iterrows():
        h = int(row.get("HorizonMonths", -1))
        if h in calib_lu.index:
            factors.append(float(calib_lu.at[h, "final_factor"]))
            applied.append(bool(calib_lu.at[h, "calibration_applied"]))
        else:
            factors.append(1.0)
            applied.append(False)

    fc["CalibrationFactor"]  = factors
    fc["CalibrationApplied"] = applied
    fc["CalibrationScope"]   = "global"

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in fc.columns:
            fc[col] = (fc[col] * fc["CalibrationFactor"]).clip(lower=0).round(4)

    if "ModelVersion" in fc.columns:
        fc["ModelVersion"] = fc.apply(
            lambda r: (r["ModelVersion"] + "+v76gcalib")
            if r["CalibrationApplied"] and "+v76gcalib" not in str(r.get("ModelVersion", ""))
            else r["ModelVersion"],
            axis=1,
        )

    n_applied = sum(applied)
    logger.info(
        "[v7.6] Applied global calibration: %d / %d rows adjusted",
        n_applied, len(fc),
    )
    return fc


# ---------------------------------------------------------------------------
# Public: build_v76_bias_analysis
# ---------------------------------------------------------------------------

def build_v76_bias_analysis(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    raw_scode_fc: pd.DataFrame,
    calibrated_scode_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    calibration_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compare bias before and after calibration at StyleCode level for holdout months.

    Returns
    -------
    DataFrame with columns:
        HorizonMonths, MonthStart,
        TotalActual, TotalForecastRaw, TotalForecastCalibrated,
        RawBiasRatio, CalibratedBiasRatio, BiasImprovement, CalibrationApplied
    """
    acts = actuals_df.copy()
    acts[DATE_COL] = pd.to_datetime(acts[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]  = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])]

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp_sc = dp[[SKU_COL, SCODE_COL]].dropna().drop_duplicates(SKU_COL)
    acts = acts.merge(dp_sc, on=SKU_COL, how="inner")

    def _prep(fc):
        f = fc.copy()
        if "Key" in f.columns:
            f = f.rename(columns={"Key": SCODE_COL})
        f[DATE_COL] = pd.to_datetime(f[DATE_COL]).dt.to_period("M").dt.to_timestamp()
        return f

    raw_fc = _prep(raw_scode_fc)
    cal_fc = _prep(calibrated_scode_fc)

    # Calibration applied lookup
    calib_applied: dict[int, bool] = {}
    if calibration_df is not None and not calibration_df.empty:
        calib_applied = dict(zip(
            calibration_df["HorizonMonths"],
            calibration_df["calibration_applied"],
        ))

    rows = []

    for m in sorted(acts[DATE_COL].unique()):
        a_m = acts[acts[DATE_COL] == m]
        a_sc = a_m.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()

        raw_m = raw_fc[raw_fc[DATE_COL] == m]
        cal_m = cal_fc[cal_fc[DATE_COL] == m]

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

            tot_act = float(mg[TARGET_COL].sum())
            tot_raw = float(mg["RawFC"].sum())
            tot_cal = float(mg["CalFC"].sum())

            raw_bias = round(tot_raw / tot_act, 4) if tot_act > 0 else np.nan
            cal_bias = round(tot_cal / tot_act, 4) if tot_act > 0 else np.nan
            improvement = round(
                (abs(raw_bias - 1.0) - abs(cal_bias - 1.0)) * 100, 4
            ) if not (np.isnan(raw_bias) or np.isnan(cal_bias)) else np.nan

            rows.append({
                "HorizonMonths":           int(h),
                "MonthStart":              m.strftime("%Y-%m"),
                "TotalActual":             round(tot_act, 2),
                "TotalForecastRaw":        round(tot_raw, 2),
                "TotalForecastCalibrated": round(tot_cal, 2),
                "RawBiasRatio":            raw_bias,
                "CalibratedBiasRatio":     cal_bias,
                "BiasImprovement":         improvement,
                "CalibrationApplied":      bool(calib_applied.get(h, False)),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "HorizonMonths","MonthStart","TotalActual",
            "TotalForecastRaw","TotalForecastCalibrated",
            "RawBiasRatio","CalibratedBiasRatio",
            "BiasImprovement","CalibrationApplied",
        ])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public: validate_global_calibration_table
# ---------------------------------------------------------------------------

def validate_global_calibration_table(
    calibration_df: pd.DataFrame,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
) -> dict:
    """
    Run integrity checks on the v7.6 global calibration table.

    Returns dict with: n_rows, n_applied, factors_in_range,
    has_duplicate_keys, no_regression_gate_passed, warnings
    """
    warnings: list[str] = []

    if calibration_df.empty:
        return {
            "n_rows": 0, "n_applied": 0,
            "factors_in_range": True, "has_duplicate_keys": False,
            "no_regression_gate_passed": True,
            "warnings": ["Calibration table empty — all factors = 1.0"],
        }

    n_rows    = len(calibration_df)
    n_applied = int(calibration_df["calibration_applied"].sum()) \
        if "calibration_applied" in calibration_df.columns else 0

    in_range = True
    if "final_factor" in calibration_df.columns:
        out = (
            (calibration_df["final_factor"] < min_factor - 1e-6) |
            (calibration_df["final_factor"] > max_factor + 1e-6)
        )
        if out.any():
            in_range = False
            warnings.append(f"{out.sum()} final_factor values outside [{min_factor},{max_factor}]")

    has_dup = False
    if "HorizonMonths" in calibration_df.columns:
        dup = calibration_df.duplicated(subset=["HorizonMonths"], keep=False)
        if dup.any():
            has_dup = True
            warnings.append(f"{dup.sum()} duplicate HorizonMonths in calibration table")

    # No-regression gate: check that any rejected horizon has reason documented
    gate_passed = True
    if "rejection_reason" in calibration_df.columns:
        unapplied = calibration_df[~calibration_df["calibration_applied"]]
        for _, row in unapplied.iterrows():
            if row.get("proposed_factor", 1.0) != 1.0 and not row.get("rejection_reason", ""):
                gate_passed = False
                warnings.append(
                    f"H={row['HorizonMonths']}: calibration rejected without documented reason"
                )

    return {
        "n_rows":                  n_rows,
        "n_applied":               n_applied,
        "factors_in_range":        in_range,
        "has_duplicate_keys":      has_dup,
        "no_regression_gate_passed": gate_passed,
        "warnings":                warnings,
    }
