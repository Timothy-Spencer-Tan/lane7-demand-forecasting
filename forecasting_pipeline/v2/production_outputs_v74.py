"""
lane7_forecast.production_outputs_v74
========================================
v7.4 production output builders.

Produces all client-facing and diagnostic output files for the v7.4
Production Candidate.  All functions are pure (no side effects beyond
file writes when output_path is provided).

Public API
----------
    build_production_sku_table(sku_fc_df, dim_product_df,
                                calibration_df=None,
                                allocation_method="recency_only")
        -> pd.DataFrame  (v7_4_production_sku_forecasts schema)

    build_forecast_risk_flags(sku_fc_df, gold_df, dim_product_df,
                               calibration_df=None, holdout_months=None)
        -> pd.DataFrame  (v7_4_forecast_risk_flags schema)

    build_production_validation_report(scode_fc, scol_fc, sku_fc,
                                        calibration_df=None)
        -> pd.DataFrame  (v7_4_production_validation_report schema)

    build_error_decomposition(actuals_df, dim_product_df,
                               scode_fc, scol_fc, sku_fc, holdout_months)
        -> pd.DataFrame  (v7_4_error_decomposition schema)

    score_holdout(sku_fc, actuals_df, holdout_months)
        -> (eval_df, preds_df)

    build_version_comparison(v74_holdout_eval, v74_error_decomp,
                              prior_comparison_path=None,
                              v72_recency_holdout_eval=None,
                              v73_holdout_eval=None)
        -> pd.DataFrame  (v7_4_vs_prior_versions schema)
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
SIZE_COL   = "SizeDesc"
STANDALONE = "STANDALONE"


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


# ---------------------------------------------------------------------------
# 1. Production SKU table
# ---------------------------------------------------------------------------

def build_production_sku_table(
    sku_fc_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    allocation_method: str = "recency_only",
) -> pd.DataFrame:
    """
    Build the main client-facing SKU forecast table.

    Schema:
        MonthStart, HorizonMonths, SKU, StyleCodeDesc, StyleColorDesc,
        SizeDesc, ForecastUnits, Lower, Upper, ModelName, ModelVersion,
        AllocationMethod, CalibrationApplied, CalibrationFactor

    Parameters
    ----------
    sku_fc_df        : SKU-level allocated forecasts (Key = SKU)
    dim_product_df   : for joining StyleCodeDesc / StyleColorDesc / SizeDesc
    calibration_df   : calibration table (for CalibrationFactor column)
    allocation_method: string tag written into AllocationMethod column
    """
    fc = sku_fc_df.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SKU_COL})
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    fc[SKU_COL]      = fc[SKU_COL].astype(str).str.strip()

    # Join product attributes from dim_product (fill any missing from allocation)
    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    attr_cols = [c for c in [SCODE_COL, SC_COL, SIZE_COL] if c in dp.columns]
    dp_attrs = dp[[SKU_COL] + attr_cols].drop_duplicates(SKU_COL)

    # Some attr cols may already be in fc from the allocation step
    need_join = [c for c in attr_cols if c not in fc.columns or fc[c].isna().all()]
    if need_join:
        fc = fc.merge(dp_attrs[[SKU_COL] + need_join], on=SKU_COL, how="left")
    else:
        # Still merge to fill NaN values from allocation
        for col in attr_cols:
            if col in fc.columns:
                missing = fc[col].isna()
                if missing.any() and col in dp_attrs.columns:
                    filled = fc.merge(dp_attrs[[SKU_COL, col]], on=SKU_COL, how="left", suffixes=("","_dp"))
                    fc.loc[missing, col] = filled.loc[missing, f"{col}_dp"] if f"{col}_dp" in filled.columns else fc.loc[missing, col]

    # Calibration columns
    fc["AllocationMethod"] = allocation_method

    if "CalibrationFactor" not in fc.columns:
        fc["CalibrationFactor"] = 1.0
    if "CalibrationApplied" not in fc.columns:
        fc["CalibrationApplied"] = False

    # If calibration_df provided and CalibrationFactor not already on sku_fc,
    # join at StyleCode level (factor was applied at StyleCode before allocation)
    if calibration_df is not None and not calibration_df.empty:
        if SCODE_COL in fc.columns:
            calib_lu = calibration_df.set_index([SCODE_COL, "HorizonMonths"])[
                ["calibration_factor", "calibration_applied"]
            ]
            for idx, row in fc.iterrows():
                key = (row.get(SCODE_COL, ""), row.get("HorizonMonths", -1))
                if key in calib_lu.index:
                    fc.at[idx, "CalibrationFactor"]  = float(calib_lu.at[key, "calibration_factor"])
                    fc.at[idx, "CalibrationApplied"] = bool(calib_lu.at[key, "calibration_applied"])

    keep_cols = [c for c in [
        DATE_COL, "HorizonMonths", SKU_COL,
        SCODE_COL, SC_COL, SIZE_COL,
        "ForecastUnits", "Lower", "Upper",
        "ModelName", "ModelVersion",
        "AllocationMethod", "CalibrationApplied", "CalibrationFactor",
    ] if c in fc.columns]

    prod_table = (
        fc[keep_cols]
        .drop_duplicates(subset=[DATE_COL, "HorizonMonths", SKU_COL], keep="first")
        .sort_values([DATE_COL, "HorizonMonths", SKU_COL])
        .reset_index(drop=True)
    )

    logger.info(
        "[v7.4] Production SKU table: %d rows, %d SKUs, %d months",
        len(prod_table),
        prod_table[SKU_COL].nunique() if SKU_COL in prod_table.columns else 0,
        prod_table[DATE_COL].nunique() if DATE_COL in prod_table.columns else 0,
    )
    return prod_table


# ---------------------------------------------------------------------------
# 2. Forecast risk flags
# ---------------------------------------------------------------------------

def build_forecast_risk_flags(
    sku_fc_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    holdout_months: list[pd.Timestamp] | None = None,
    sparse_history_threshold: int   = 6,    # nonzero months < this → sparse
    intermittent_threshold: float   = 0.50,  # zero-rate > this → intermittent
    high_conc_threshold: float      = 0.85,  # max size share > this → concentration
    large_jump_multiplier: float    = 3.0,   # forecast > N× trailing mean → jump
    low_recent_units_threshold: int = 5,     # recent 3m units <= this → low recent
) -> pd.DataFrame:
    """
    Produce per-SKU-month risk flag rows for forecasts needing business review.

    Risk flags produced:
        sparse_history            : <6 nonzero months in training history
        intermittent_demand       : >50% of training months have 0 demand
        high_allocation_concentration : top-1 share >85%
        low_recent_demand         : last 3m actuals ≤ 5 units
        large_forecast_jump       : ForecastUnits > 3× trailing 3m mean
        calibration_weak_or_missing: CalibrationFactor=1.0 due to WEAK/NONE evidence
        parent_forecast_high_error: StyleCode WMAPE from prior versions was high
                                    (flagged when calibration raw_bias > 1.30)

    Parameters
    ----------
    sku_fc_df       : production SKU forecasts (from build_production_sku_table)
    gold_df         : gold demand (for history statistics)
    dim_product_df  : for attribute joining
    calibration_df  : calibration table (optional)
    holdout_months  : forecast months to flag (defaults to all months in sku_fc_df)
    """
    fc = sku_fc_df.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SKU_COL})
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    fc[SKU_COL]      = fc[SKU_COL].astype(str).str.strip()

    if holdout_months:
        fc = fc[fc["MonthStart"].isin([pd.Timestamp(m) for m in holdout_months])].copy()

    # Historical statistics per SKU
    gold = gold_df.copy()
    gold["MonthStart"] = pd.to_datetime(gold["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]      = gold[SKU_COL].astype(str).str.strip()

    # Training data cut: anything before first holdout month
    if holdout_months:
        train_cutoff = min(pd.Timestamp(m) for m in holdout_months) - pd.DateOffset(months=1)
        gold_train = gold[gold["MonthStart"] <= train_cutoff].copy()
    else:
        gold_train = gold.copy()

    recent_cutoff = gold_train["MonthStart"].max() - pd.DateOffset(months=2)
    recent_cutoff = pd.Timestamp(recent_cutoff.year, recent_cutoff.month, 1) if not gold_train.empty else pd.Timestamp("2025-10-01")

    sku_stats = {}
    for sku, grp in gold_train.groupby(SKU_COL):
        units      = grp.set_index("MonthStart")[TARGET_COL].sort_index()
        n_months   = len(units)
        n_nonzero  = int((units > 0).sum())
        zero_rate  = 1.0 - (n_nonzero / n_months) if n_months > 0 else 1.0
        trailing3  = float(units[units.index >= recent_cutoff].mean()) if n_nonzero > 0 else 0.0
        sku_stats[sku] = {
            "n_nonzero_months": n_nonzero,
            "zero_rate":        zero_rate,
            "trailing3_mean":   trailing3,
        }

    # Calibration lookup: StyleCode → (factor, tier)
    calib_lu = {}
    if calibration_df is not None and not calibration_df.empty and SCODE_COL in calibration_df.columns:
        for _, crow in calibration_df.iterrows():
            key = (crow[SCODE_COL], crow.get("HorizonMonths", -1))
            calib_lu[key] = {
                "factor": float(crow.get("calibration_factor", 1.0)),
                "tier":   str(crow.get("evidence_tier", "NONE")),
                "bias":   crow.get("raw_bias_ratio", None),
            }

    flag_rows = []

    for _, row in fc.iterrows():
        sku       = str(row.get(SKU_COL, ""))
        month     = row.get("MonthStart")
        horizon   = row.get("HorizonMonths", -1)
        fc_units  = float(row.get("ForecastUnits", 0.0))
        scode     = str(row.get(SCODE_COL, ""))
        sc        = str(row.get(SC_COL, ""))

        stats     = sku_stats.get(sku, {})
        n_nz      = stats.get("n_nonzero_months", 0)
        zero_rate = stats.get("zero_rate", 1.0)
        trail3    = stats.get("trailing3_mean", 0.0)
        c_info    = calib_lu.get((scode, horizon), {})

        def _flag(flag, severity, reason):
            flag_rows.append({
                SKU_COL:        sku,
                SCODE_COL:      scode,
                SC_COL:         sc,
                DATE_COL:       month,
                "HorizonMonths":horizon,
                "ForecastUnits":fc_units,
                "risk_flag":    flag,
                "risk_severity":severity,
                "reason_code":  reason,
            })

        if n_nz < sparse_history_threshold:
            _flag("sparse_history", "MEDIUM",
                  f"only {n_nz} nonzero training months")

        if zero_rate > intermittent_threshold:
            _flag("intermittent_demand", "LOW",
                  f"zero_rate={zero_rate:.2f}")

        if trail3 > 0 and fc_units > large_jump_multiplier * trail3:
            _flag("large_forecast_jump", "HIGH",
                  f"ForecastUnits={fc_units:.1f} > {large_jump_multiplier}× trailing_mean={trail3:.1f}")

        if trail3 <= low_recent_units_threshold:
            _flag("low_recent_demand", "LOW",
                  f"trailing_3m_mean={trail3:.1f}")

        if c_info.get("tier") in ("WEAK", "NONE", None):
            _flag("calibration_weak_or_missing", "LOW",
                  f"evidence_tier={c_info.get('tier','NONE')}")

        bias = c_info.get("bias")
        if bias is not None and not np.isnan(bias) and bias > 1.30:
            _flag("parent_forecast_high_error", "MEDIUM",
                  f"StyleCode raw_bias_ratio={bias:.2f} (over-prediction in backtest)")

    if not flag_rows:
        return pd.DataFrame(columns=[
            SKU_COL, SCODE_COL, SC_COL, DATE_COL, "HorizonMonths",
            "ForecastUnits", "risk_flag", "risk_severity", "reason_code",
        ])

    result = pd.DataFrame(flag_rows).sort_values(
        ["risk_severity", SKU_COL, DATE_COL]
    ).reset_index(drop=True)

    logger.info(
        "[v7.4] Risk flags: %d total across %d SKUs",
        len(result), result[SKU_COL].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Production validation report
# ---------------------------------------------------------------------------

def build_production_validation_report(
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    tolerance: float = 0.01,
    min_factor: float = 0.80,
    max_factor: float = 1.20,
) -> pd.DataFrame:
    """
    Run all production validation checks and return a report DataFrame.

    Checks performed:
        StyleCode totals == StyleColor totals
        StyleColor totals == SKU totals
        No negative forecasts
        No duplicate (SKU, MonthStart, HorizonMonths) rows
        No missing critical columns
        Calibration factors within allowed range
    """
    checks = []

    def _add(check, passed, detail=""):
        checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    def _prep_fc(fc, key_as=None):
        """
        Copy fc, normalise MonthStart, and rename Key → key_as only when:
          - key_as is given
          - "Key" column exists
          - key_as column does NOT already exist (prevents clobbering real columns)
        """
        f = fc.copy()
        if key_as is not None and "Key" in f.columns and key_as not in f.columns:
            f = f.rename(columns={"Key": key_as})
        f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        return f

    # Prepare each frame with its correct entity column
    scode_tmp = _prep_fc(scode_fc, key_as=SCODE_COL)   # Key → StyleCodeDesc
    scol_tmp  = _prep_fc(scol_fc,  key_as=SC_COL)      # Key → StyleColorDesc
    sku_tmp   = _prep_fc(sku_fc,   key_as=SKU_COL)     # Key → SKU

    # ── Check 1: StyleCode totals == StyleColor totals ────────────────────
    # Group both frames by (StyleCodeDesc, MonthStart, HorizonMonths).
    # scol_fc retains StyleCodeDesc from the allocation step; if it's absent
    # the check fails gracefully with an explanatory message.
    try:
        if SCODE_COL not in scode_tmp.columns:
            _add("StyleCode totals == StyleColor totals", False,
                 f"StyleCodeDesc missing from scode_fc (columns: {list(scode_fc.columns[:6])})")
        elif SCODE_COL not in scol_tmp.columns:
            _add("StyleCode totals == StyleColor totals", False,
                 f"StyleCodeDesc missing from scol_fc — needed to roll up StyleColor to StyleCode level "
                 f"(columns: {list(scol_fc.columns[:6])})")
        else:
            sc_roll = (
                scode_tmp.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SC_total"})
            )
            scol_by_sc = (
                scol_tmp.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SCOL_total"})
            )
            merged1 = sc_roll.merge(
                scol_by_sc, on=[SCODE_COL, "MonthStart", "HorizonMonths"], how="left"
            )
            if merged1.empty:
                _add("StyleCode totals == StyleColor totals", False, "no matching rows after merge")
            else:
                merged1["diff"] = (merged1["SC_total"] - merged1["SCOL_total"].fillna(0)).abs()
                max_diff = float(merged1["diff"].max())
                _add("StyleCode totals == StyleColor totals", max_diff < tolerance,
                     f"max_diff={max_diff:.4f}")
    except Exception as exc:
        _add("StyleCode totals == StyleColor totals", False, f"error: {exc}")

    # ── Check 2: StyleColor totals == SKU totals ──────────────────────────
    # Group scol_fc by (StyleColorDesc, MonthStart, HorizonMonths).
    # Group sku_fc  by (StyleColorDesc, MonthStart, HorizonMonths), excluding
    # STANDALONE rows that have no StyleColorDesc lineage.
    try:
        if SC_COL not in scol_tmp.columns:
            _add("StyleColor totals == SKU totals", False,
                 f"StyleColorDesc missing from scol_fc (columns: {list(scol_fc.columns[:6])})")
        elif SC_COL not in sku_tmp.columns:
            _add("StyleColor totals == SKU totals", False,
                 f"StyleColorDesc missing from sku_fc — needed to roll up SKU to StyleColor level "
                 f"(columns: {list(sku_fc.columns[:6])})")
        else:
            scol_roll = (
                scol_tmp.groupby([SC_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SCOL_total"})
            )
            # Exclude STANDALONE rows from the SKU rollup (they have no StyleColor parent)
            sku_hier = sku_tmp[sku_tmp[SC_COL].astype(str) != STANDALONE]
            if sku_hier.empty:
                _add("StyleColor totals == SKU totals", False, "no hierarchical SKU rows found")
            else:
                sku_by_sc = (
                    sku_hier.groupby([SC_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                    .sum().reset_index().rename(columns={"ForecastUnits": "SKU_total"})
                )
                merged2 = scol_roll.merge(
                    sku_by_sc, on=[SC_COL, "MonthStart", "HorizonMonths"], how="left"
                )
                if merged2.empty:
                    _add("StyleColor totals == SKU totals", False, "no matching rows after merge")
                else:
                    merged2["diff"] = (merged2["SCOL_total"] - merged2["SKU_total"].fillna(0)).abs()
                    max_diff2 = float(merged2["diff"].max())
                    _add("StyleColor totals == SKU totals", max_diff2 < tolerance,
                         f"max_diff={max_diff2:.4f}")
    except Exception as exc:
        _add("StyleColor totals == SKU totals", False, f"error: {exc}")

    # ── No negatives ──────────────────────────────────────────────────────
    no_neg    = bool((sku_fc["ForecastUnits"].fillna(0) >= 0).all())
    neg_count = int((sku_fc["ForecastUnits"].fillna(0) < 0).sum())
    _add("No negative ForecastUnits", no_neg, f"{neg_count} negative rows")

    # ── No duplicates ─────────────────────────────────────────────────────
    key_cols = [c for c in ["MonthStart", "HorizonMonths", SKU_COL] if c in sku_tmp.columns]
    if key_cols:
        dup_mask = sku_tmp.duplicated(subset=key_cols, keep=False)
        n_dups   = int(dup_mask.sum())
        _add("No duplicate (SKU, MonthStart, HorizonMonths) rows", n_dups == 0,
             f"{n_dups} duplicate rows")

    # ── Required columns ─────────────────────────────────────────────────
    required = ["ForecastUnits", "MonthStart", "HorizonMonths"]
    missing  = [c for c in required if c not in sku_fc.columns]
    _add("Required columns present in SKU forecasts", len(missing) == 0,
         f"missing: {missing}" if missing else "all present")

    # ── Calibration factors in range ──────────────────────────────────────
    if calibration_df is not None and not calibration_df.empty and "calibration_factor" in calibration_df.columns:
        f_min = float(calibration_df["calibration_factor"].min())
        f_max = float(calibration_df["calibration_factor"].max())
        in_range = (f_min >= min_factor - 1e-6) and (f_max <= max_factor + 1e-6)
        _add("Calibration factors within allowed range",
             in_range, f"min={f_min:.4f}, max={f_max:.4f}, allowed=[{min_factor},{max_factor}]")
    else:
        _add("Calibration factors within allowed range", True, "no calibration table provided")

    report = pd.DataFrame(checks)
    n_fail = int((~report["passed"]).sum())
    logger.info(
        "[v7.4] Production validation: %d checks, %d passed, %d failed",
        len(report), int(report["passed"].sum()), n_fail,
    )
    return report


# ---------------------------------------------------------------------------
# 4. Error decomposition
# ---------------------------------------------------------------------------

def build_error_decomposition(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    WMAPE at StyleCode / StyleColor / SKU for each (horizon, month).
    Mirrors the v7.2 / v7.3 decomposition for apples-to-apples comparison.
    """
    acts = actuals_df.copy()
    acts["MonthStart"] = pd.to_datetime(acts["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]      = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts["MonthStart"].isin([pd.Timestamp(m) for m in holdout_months])]

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    sc_cols = [c for c in [SCODE_COL, SC_COL] if c in dp.columns]
    dp = dp[[SKU_COL] + sc_cols].drop_duplicates(SKU_COL)
    acts = acts.merge(dp, on=SKU_COL, how="left")

    rows = []

    def _score(a, p, label, h, m):
        tot = a.sum()
        wmape = float((a - p).abs().sum() / tot * 100) if tot > 0 else np.nan
        rows.append({
            "Level": label, "HorizonMonths": int(h),
            "MonthStart": m.strftime("%Y-%m"),
            "TotalActual": round(float(tot), 2),
            "TotalPredicted": round(float(p.sum()), 2),
            "WMAPE": round(wmape, 4),
        })

    def _prep_fc(fc, key_col):
        f = fc.copy()
        if "Key" in f.columns:
            f = f.rename(columns={"Key": key_col})
        f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        return f

    for m in sorted(acts["MonthStart"].unique()):
        a_m = acts[acts["MonthStart"] == m]

        # SKU
        fc_sku = _prep_fc(sku_fc, SKU_COL)
        fc_m   = fc_sku[fc_sku["MonthStart"] == m]
        for h in sorted(fc_m["HorizonMonths"].unique()):
            fc_mh = fc_m[fc_m["HorizonMonths"] == h].drop_duplicates(SKU_COL)
            mg    = a_m[[SKU_COL, TARGET_COL]].merge(fc_mh[[SKU_COL, "ForecastUnits"]], on=SKU_COL)
            if not mg.empty:
                _score(mg[TARGET_COL], mg["ForecastUnits"], "SKU", h, m)

        # StyleColor
        if SC_COL in acts.columns:
            fc_sc = _prep_fc(scol_fc, SC_COL)
            fc2_m = fc_sc[fc_sc["MonthStart"] == m]
            a_sc  = a_m.groupby(SC_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc2_m["HorizonMonths"].unique()):
                fc_h = fc2_m[fc2_m["HorizonMonths"] == h].groupby(SC_COL)["ForecastUnits"].sum().reset_index()
                mg2  = a_sc.merge(fc_h, on=SC_COL)
                if not mg2.empty:
                    _score(mg2[TARGET_COL], mg2["ForecastUnits"], "StyleColor", h, m)

        # StyleCode
        if SCODE_COL in acts.columns:
            fc_scd = _prep_fc(scode_fc, SCODE_COL)
            fc3_m  = fc_scd[fc_scd["MonthStart"] == m]
            a_scd  = a_m.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc3_m["HorizonMonths"].unique()):
                fc_h3 = fc3_m[fc3_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()
                mg3   = a_scd.merge(fc_h3, on=SCODE_COL)
                if not mg3.empty:
                    _score(mg3[TARGET_COL], mg3["ForecastUnits"], "StyleCode", h, m)

    if not rows:
        return pd.DataFrame(columns=["Level","HorizonMonths","MonthStart","TotalActual","TotalPredicted","WMAPE"])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart","Level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Holdout scoring
# ---------------------------------------------------------------------------

def score_holdout(
    sku_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score SKU forecasts against actuals. Returns (eval_df, preds_df)."""
    fc = sku_fc.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SKU_COL})
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    fc[SKU_COL]      = fc[SKU_COL].astype(str).str.strip()

    acts = actuals_df.copy()
    acts["MonthStart"] = pd.to_datetime(acts["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]      = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts["MonthStart"].isin([pd.Timestamp(m) for m in holdout_months])]
    acts = acts[[SKU_COL, "MonthStart", TARGET_COL]].rename(columns={TARGET_COL: "ActualUnits"})

    scored = fc.merge(acts, on=[SKU_COL, "MonthStart"], how="inner")
    scored["PredictedUnits"] = scored["ForecastUnits"].clip(lower=0).fillna(0)
    scored["Error"]          = scored["ActualUnits"] - scored["PredictedUnits"]
    scored["AbsError"]       = scored["Error"].abs()
    scored["AbsPctError"]    = np.where(
        scored["ActualUnits"] > 0,
        scored["AbsError"] / scored["ActualUnits"] * 100, np.nan,
    )

    pred_cols = [SKU_COL, "MonthStart", "HorizonMonths", "ModelName", "ModelVersion",
                 "PredictedUnits", "ActualUnits", "Error", "AbsError", "AbsPctError",
                 "CalibrationApplied", "CalibrationFactor", "AllocationMethod"]
    preds = scored[[c for c in pred_cols if c in scored.columns]].copy()

    eval_rows = []
    for h in sorted(scored["HorizonMonths"].unique()):
        h_df = scored[scored["HorizonMonths"] == h]
        for m in sorted(h_df["MonthStart"].unique()):
            m_df = h_df[h_df["MonthStart"] == m]
            tot  = m_df["ActualUnits"].sum()
            wmape = m_df["AbsError"].sum() / max(1, tot) * 100
            eval_rows.append({
                "Level":          "SKU",
                "HorizonMonths":  int(h),
                "MonthStart":     m.strftime("%Y-%m"),
                "Segment":        "ALL",
                "N_SKUs":         m_df[SKU_COL].nunique(),
                "TotalActual":    round(tot, 2),
                "TotalPredicted": round(m_df["PredictedUnits"].sum(), 2),
                "AbsError":       round(m_df["AbsError"].sum(), 2),
                "WMAPE":          round(wmape, 4),
            })

    return pd.DataFrame(eval_rows), preds


# ---------------------------------------------------------------------------
# 6. Version comparison table
# ---------------------------------------------------------------------------

def build_version_comparison(
    v74_holdout_eval: pd.DataFrame,
    v74_error_decomp: pd.DataFrame,
    prior_comparison_path: str | Path | None = None,
    v72_recency_holdout_eval: pd.DataFrame | None = None,
    v73_holdout_eval: pd.DataFrame | None = None,
    v73_error_decomp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the v7_4_vs_prior_versions comparison table.

    Loads prior variant WMAPE from:
    1. prior_comparison_path (v7_2_variant_comparison.csv or v7_3_vs_v7_2_comparison.csv)
    2. Directly provided holdout eval DataFrames

    v7.4 row is always built from v74_holdout_eval and v74_error_decomp.

    Returns
    -------
    pd.DataFrame with columns:
        Variant, H1_Jan_WMAPE, H3_Jan_WMAPE, H3_Feb_WMAPE,
        StyleCode_H3_Jan_WMAPE, StyleColor_H3_Jan_WMAPE, SKU_H3_Jan_WMAPE,
        StyleCode_H3_Feb_WMAPE, StyleColor_H3_Feb_WMAPE, SKU_H3_Feb_WMAPE,
        ProductionCandidateFlag
    """
    jan = "2026-01"
    feb = "2026-02"

    def _ev(eval_df, h, m):
        if eval_df is None or eval_df.empty:
            return np.nan
        r = eval_df[(eval_df.get("HorizonMonths", pd.Series(dtype=int)) == h) &
                    (eval_df.get("MonthStart", pd.Series(dtype=str)) == m)]
        if r.empty and "MonthStart" in eval_df.columns:
            r = eval_df[(eval_df["HorizonMonths"] == h) & (eval_df["MonthStart"] == m)]
        return round(float(r["WMAPE"].iloc[0]), 4) if not r.empty else np.nan

    def _dc(dec_df, level, h, m):
        if dec_df is None or dec_df.empty:
            return np.nan
        r = dec_df[(dec_df.get("Level","") == level) &
                   (dec_df.get("HorizonMonths", pd.Series()) == h) &
                   (dec_df.get("MonthStart", pd.Series()) == m)]
        return round(float(r["WMAPE"].iloc[0]), 4) if not r.empty else np.nan

    # Build v7.4 row
    v74_row = {
        "Variant":                  "v7.4_production",
        "H1_Jan_WMAPE":             _ev(v74_holdout_eval, 1, jan),
        "H3_Jan_WMAPE":             _ev(v74_holdout_eval, 3, jan),
        "H3_Feb_WMAPE":             _ev(v74_holdout_eval, 3, feb),
        "StyleCode_H3_Jan_WMAPE":   _dc(v74_error_decomp, "StyleCode",  3, jan),
        "StyleColor_H3_Jan_WMAPE":  _dc(v74_error_decomp, "StyleColor", 3, jan),
        "SKU_H3_Jan_WMAPE":         _dc(v74_error_decomp, "SKU",        3, jan),
        "StyleCode_H3_Feb_WMAPE":   _dc(v74_error_decomp, "StyleCode",  3, feb),
        "StyleColor_H3_Feb_WMAPE":  _dc(v74_error_decomp, "StyleColor", 3, feb),
        "SKU_H3_Feb_WMAPE":         _dc(v74_error_decomp, "SKU",        3, feb),
        "ProductionCandidateFlag":  True,
    }

    rows = []

    # Load prior comparison if available
    if prior_comparison_path is not None:
        p = Path(prior_comparison_path)
        if p.exists():
            prior = pd.read_csv(p)
            if "ProductionCandidateFlag" not in prior.columns:
                prior["ProductionCandidateFlag"] = False
            # Avoid overwriting v7.4 if already in the file
            prior = prior[prior.get("Variant", pd.Series(dtype=str)) != "v7.4_production"]
            rows.extend(prior.to_dict("records"))

    # Add directly provided evaluations for versions not in prior_comparison_path
    existing_variants = {r.get("Variant","") for r in rows}

    if v72_recency_holdout_eval is not None and "v7.2_recency_only" not in existing_variants:
        rows.append({
            "Variant":               "v7.2_recency_only",
            "H1_Jan_WMAPE":          _ev(v72_recency_holdout_eval, 1, jan),
            "H3_Jan_WMAPE":          _ev(v72_recency_holdout_eval, 3, jan),
            "H3_Feb_WMAPE":          _ev(v72_recency_holdout_eval, 3, feb),
            "ProductionCandidateFlag": False,
        })

    if v73_holdout_eval is not None and "v7.3_segmented" not in existing_variants:
        rows.append({
            "Variant":               "v7.3_segmented",
            "H1_Jan_WMAPE":          _ev(v73_holdout_eval, 1, jan),
            "H3_Jan_WMAPE":          _ev(v73_holdout_eval, 3, jan),
            "H3_Feb_WMAPE":          _ev(v73_holdout_eval, 3, feb),
            "ProductionCandidateFlag": False,
        })

    rows.append(v74_row)

    result = pd.DataFrame(rows)

    col_order = [
        "Variant",
        "H1_Jan_WMAPE", "H3_Jan_WMAPE", "H3_Feb_WMAPE",
        "StyleCode_H3_Jan_WMAPE", "StyleColor_H3_Jan_WMAPE", "SKU_H3_Jan_WMAPE",
        "StyleCode_H3_Feb_WMAPE", "StyleColor_H3_Feb_WMAPE", "SKU_H3_Feb_WMAPE",
        "ProductionCandidateFlag",
    ]
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan

    return result[col_order].reset_index(drop=True)
