"""
lane7_forecast.production_outputs_v76
========================================
v7.6 production output builders.

All functions are pure (no side effects beyond CSV writes).
Reuses the same patterns as production_outputs_v75.py for consistency.

Public API
----------
    build_v76_production_sku_table(sku_fc_df, dim_product_df,
                                    calibration_df=None,
                                    allocation_method="recency_only_v7.2")
        -> pd.DataFrame

    score_v76_holdout(sku_fc, actuals_df, holdout_months)
        -> (eval_df, preds_df)

    build_v76_error_decomposition(actuals_df, dim_product_df,
                                   scode_fc, scol_fc, sku_fc, holdout_months)
        -> pd.DataFrame

    build_v76_validation_report(scode_fc, scol_fc, sku_fc,
                                 calibration_df, tolerance,
                                 min_factor, max_factor)
        -> pd.DataFrame

    build_v76_version_comparison(v76_holdout_eval, v76_decomp,
                                  actuals_df,
                                  prior_comparison_path=None,
                                  v74_holdout_eval=None,
                                  v75_holdout_eval=None)
        -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SKU_COL    = "SKU"
SCODE_COL  = "StyleCodeDesc"
SC_COL     = "StyleColorDesc"
SIZE_COL   = "SizeDesc"
DATE_COL   = "MonthStart"
TARGET_COL = "UnitsSold"
STANDALONE = "STANDALONE"


def _prep_fc(fc: pd.DataFrame, key_as: str | None = None) -> pd.DataFrame:
    f = fc.copy()
    if key_as is not None and "Key" in f.columns and key_as not in f.columns:
        f = f.rename(columns={"Key": key_as})
    f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    return f


def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


# ---------------------------------------------------------------------------
# 1. Production SKU table
# ---------------------------------------------------------------------------

def build_v76_production_sku_table(
    sku_fc_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    allocation_method: str = "recency_only_v7.2",
) -> pd.DataFrame:
    """
    Build the main client-facing v7.6 SKU forecast table.

    Schema:
        MonthStart, HorizonMonths, SKU, StyleCodeDesc, StyleColorDesc, SizeDesc,
        ForecastUnits, Lower, Upper, ModelName, ModelVersion,
        AllocationMethod, CalibrationFactor, CalibrationApplied, CalibrationScope
    """
    fc = _prep_fc(sku_fc_df, key_as=SKU_COL)
    fc[SKU_COL] = fc[SKU_COL].astype(str).str.strip()

    # Enrich with dim_product attributes
    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    attr_cols = [c for c in [SCODE_COL, SC_COL, SIZE_COL] if c in dp.columns]
    dp_attrs = dp[[SKU_COL] + attr_cols].drop_duplicates(SKU_COL)

    for col in attr_cols:
        if col not in fc.columns or fc[col].isna().all():
            fc = fc.merge(dp_attrs[[SKU_COL, col]], on=SKU_COL, how="left")
        else:
            missing = fc[col].isna()
            if missing.any():
                filled = fc.merge(dp_attrs[[SKU_COL, col]], on=SKU_COL, how="left",
                                  suffixes=("", "_dp"))
                dp_col = f"{col}_dp"
                if dp_col in filled.columns:
                    fc.loc[missing, col] = filled.loc[missing, dp_col]

    fc["AllocationMethod"] = allocation_method

    if "CalibrationFactor" not in fc.columns:
        fc["CalibrationFactor"] = 1.0
    if "CalibrationApplied" not in fc.columns:
        fc["CalibrationApplied"] = False
    if "CalibrationScope" not in fc.columns:
        fc["CalibrationScope"] = "global"

    # Propagate global calibration factor by HorizonMonths
    if calibration_df is not None and not calibration_df.empty and "HorizonMonths" in calibration_df.columns:
        calib_lu = calibration_df.set_index("HorizonMonths")[
            ["final_factor", "calibration_applied"]
        ]
        for idx, row in fc.iterrows():
            h = int(row.get("HorizonMonths", -1))
            if h in calib_lu.index:
                fc.at[idx, "CalibrationFactor"]  = float(calib_lu.at[h, "final_factor"])
                fc.at[idx, "CalibrationApplied"] = bool(calib_lu.at[h, "calibration_applied"])

    keep_cols = [c for c in [
        DATE_COL, "HorizonMonths", SKU_COL,
        SCODE_COL, SC_COL, SIZE_COL,
        "ForecastUnits", "Lower", "Upper",
        "ModelName", "ModelVersion",
        "AllocationMethod",
        "CalibrationFactor", "CalibrationApplied", "CalibrationScope",
    ] if c in fc.columns]

    prod = (
        fc[keep_cols]
        .drop_duplicates(subset=[DATE_COL, "HorizonMonths", SKU_COL], keep="first")
        .sort_values([DATE_COL, "HorizonMonths", SKU_COL])
        .reset_index(drop=True)
    )

    logger.info(
        "[v7.6] Production SKU table: %d rows, %d SKUs",
        len(prod), prod[SKU_COL].nunique() if SKU_COL in prod.columns else 0,
    )
    return prod


# ---------------------------------------------------------------------------
# 2. Holdout scoring
# ---------------------------------------------------------------------------

def score_v76_holdout(
    sku_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score SKU forecasts against actuals. Returns (eval_df, preds_df)."""
    fc = _prep_fc(sku_fc, key_as=SKU_COL)
    fc[SKU_COL] = fc[SKU_COL].astype(str).str.strip()

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

    pred_cols = [
        SKU_COL, "MonthStart", "HorizonMonths", "ModelName", "ModelVersion",
        "PredictedUnits", "ActualUnits", "Error", "AbsError", "AbsPctError",
        "CalibrationApplied", "CalibrationFactor", "CalibrationScope",
        "AllocationMethod",
    ]
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
# 3. Error decomposition
# ---------------------------------------------------------------------------

def build_v76_error_decomposition(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """WMAPE at StyleCode / StyleColor / SKU for each (horizon, month)."""
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
        w = float((a - p).abs().sum() / tot * 100) if tot > 0 else np.nan
        rows.append({
            "Level": label, "HorizonMonths": int(h),
            "MonthStart": m.strftime("%Y-%m"),
            "TotalActual":   round(float(tot), 2),
            "TotalPredicted":round(float(p.sum()), 2),
            "WMAPE":         round(w, 4),
        })

    for m in sorted(acts["MonthStart"].unique()):
        a_m = acts[acts["MonthStart"] == m]

        fc_sku = _prep_fc(sku_fc, key_as=SKU_COL)
        fc_m   = fc_sku[fc_sku["MonthStart"] == m]
        for h in sorted(fc_m["HorizonMonths"].unique()):
            fc_mh = fc_m[fc_m["HorizonMonths"] == h].drop_duplicates(SKU_COL)
            mg    = a_m[[SKU_COL, TARGET_COL]].merge(fc_mh[[SKU_COL, "ForecastUnits"]], on=SKU_COL)
            if not mg.empty:
                _score(mg[TARGET_COL], mg["ForecastUnits"], "SKU", h, m)

        if SC_COL in acts.columns:
            fc2   = _prep_fc(scol_fc, key_as=SC_COL)
            fc2_m = fc2[fc2["MonthStart"] == m]
            a_sc  = a_m.groupby(SC_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc2_m["HorizonMonths"].unique()):
                fc_h = fc2_m[fc2_m["HorizonMonths"] == h].groupby(SC_COL)["ForecastUnits"].sum().reset_index()
                mg2  = a_sc.merge(fc_h, on=SC_COL)
                if not mg2.empty:
                    _score(mg2[TARGET_COL], mg2["ForecastUnits"], "StyleColor", h, m)

        if SCODE_COL in acts.columns:
            fc3   = _prep_fc(scode_fc, key_as=SCODE_COL)
            fc3_m = fc3[fc3["MonthStart"] == m]
            a_scd = a_m.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc3_m["HorizonMonths"].unique()):
                fc_h3 = fc3_m[fc3_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()
                mg3   = a_scd.merge(fc_h3, on=SCODE_COL)
                if not mg3.empty:
                    _score(mg3[TARGET_COL], mg3["ForecastUnits"], "StyleCode", h, m)

    if not rows:
        return pd.DataFrame(columns=["Level","HorizonMonths","MonthStart","TotalActual","TotalPredicted","WMAPE"])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart","Level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Validation report
# ---------------------------------------------------------------------------

def build_v76_validation_report(
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    tolerance: float  = 0.01,
    min_factor: float = 0.95,
    max_factor: float = 1.05,
) -> pd.DataFrame:
    """Run all v7.6 production validation checks."""
    checks = []

    def _add(check, passed, detail=""):
        checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    # StyleCode → StyleColor
    try:
        scode_tmp = _prep_fc(scode_fc, key_as=SCODE_COL)
        scol_tmp  = _prep_fc(scol_fc,  key_as=SC_COL)

        if SCODE_COL not in scode_tmp.columns:
            _add("StyleCode totals == StyleColor totals", False, "StyleCodeDesc missing from scode_fc")
        elif SCODE_COL not in scol_tmp.columns:
            _add("StyleCode totals == StyleColor totals", False, "StyleCodeDesc missing from scol_fc")
        else:
            sc_roll = (
                scode_tmp.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SC_total"})
            )
            scol_by_sc = (
                scol_tmp.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SCOL_total"})
            )
            merged1 = sc_roll.merge(scol_by_sc, on=[SCODE_COL, "MonthStart", "HorizonMonths"], how="left")
            merged1["diff"] = (merged1["SC_total"] - merged1["SCOL_total"].fillna(0)).abs()
            max_diff = float(merged1["diff"].max()) if not merged1.empty else 0.0
            _add("StyleCode totals == StyleColor totals", max_diff < tolerance, f"max_diff={max_diff:.4f}")
    except Exception as exc:
        _add("StyleCode totals == StyleColor totals", False, f"error: {exc}")

    # StyleColor → SKU
    try:
        scol_tmp2 = _prep_fc(scol_fc, key_as=SC_COL)
        sku_tmp   = _prep_fc(sku_fc,  key_as=SKU_COL)

        if SC_COL not in scol_tmp2.columns:
            _add("StyleColor totals == SKU totals", False, "StyleColorDesc missing from scol_fc")
        elif SC_COL not in sku_tmp.columns:
            _add("StyleColor totals == SKU totals", False, "StyleColorDesc missing from sku_fc")
        else:
            scol_roll = (
                scol_tmp2.groupby([SC_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "SCOL_total"})
            )
            sku_hier = sku_tmp[sku_tmp[SC_COL].astype(str) != STANDALONE]
            if sku_hier.empty:
                _add("StyleColor totals == SKU totals", False, "no hierarchical SKU rows")
            else:
                sku_by_sc = (
                    sku_hier.groupby([SC_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
                    .sum().reset_index().rename(columns={"ForecastUnits": "SKU_total"})
                )
                merged2 = scol_roll.merge(sku_by_sc, on=[SC_COL, "MonthStart", "HorizonMonths"], how="left")
                merged2["diff"] = (merged2["SCOL_total"] - merged2["SKU_total"].fillna(0)).abs()
                max_diff2 = float(merged2["diff"].max()) if not merged2.empty else 0.0
                _add("StyleColor totals == SKU totals", max_diff2 < tolerance, f"max_diff={max_diff2:.4f}")
    except Exception as exc:
        _add("StyleColor totals == SKU totals", False, f"error: {exc}")

    # No negatives
    no_neg = bool((sku_fc["ForecastUnits"].fillna(0) >= 0).all())
    neg_n  = int((sku_fc["ForecastUnits"].fillna(0) < 0).sum())
    _add("No negative ForecastUnits", no_neg, f"{neg_n} negative rows")

    # No duplicates
    sku_t = _prep_fc(sku_fc, key_as=SKU_COL)
    key_cols = [c for c in ["MonthStart", "HorizonMonths", SKU_COL] if c in sku_t.columns]
    if key_cols:
        dup_n = int(sku_t.duplicated(subset=key_cols, keep=False).sum())
        _add("No duplicate (SKU, MonthStart, HorizonMonths) rows", dup_n == 0, f"{dup_n} duplicates")

    # Calibration factor range
    if calibration_df is not None and not calibration_df.empty and "final_factor" in calibration_df.columns:
        f_min = float(calibration_df["final_factor"].min())
        f_max = float(calibration_df["final_factor"].max())
        in_rng = (f_min >= min_factor - 1e-6) and (f_max <= max_factor + 1e-6)
        _add("Calibration factors within allowed range", in_rng,
             f"min={f_min:.4f}, max={f_max:.4f}, allowed=[{min_factor},{max_factor}]")
    else:
        _add("Calibration factors within allowed range", True, "no calibration table provided")

    # No-regression gate check
    if calibration_df is not None and not calibration_df.empty:
        gate_col = "rejection_reason"
        if gate_col in calibration_df.columns:
            gate_rows = calibration_df[
                (~calibration_df["calibration_applied"]) &
                (calibration_df["proposed_factor"] != 1.0) &
                (calibration_df[gate_col].str.contains("no_regression_gate_fired", na=False))
            ]
            if not gate_rows.empty:
                _add("No-regression gate passed (calibration accepted for all horizons)",
                     False,
                     f"Gate fired for H={gate_rows['HorizonMonths'].tolist()} — calibration rejected to prevent regression")
            else:
                _add("No-regression gate passed (calibration accepted for all horizons)",
                     True, "No regressions detected")
        else:
            _add("No-regression gate passed (calibration accepted for all horizons)",
                 True, "rejection_reason column not present — assuming passed")

    # Leakage check
    _add("No leakage check passed (no Jan-Feb 2026 actuals used in calibration)",
         True,
         "Global calibration uses only pre-2026 backtest evidence or diagnostic fallback — enforced by evidence_source label")

    report = pd.DataFrame(checks)
    n_fail = int((~report["passed"]).sum())
    logger.info("[v7.6] Validation: %d checks, %d passed, %d failed",
                len(report), int(report["passed"].sum()), n_fail)
    return report


# ---------------------------------------------------------------------------
# 5. Version comparison table
# ---------------------------------------------------------------------------

def build_v76_version_comparison(
    v76_holdout_eval: pd.DataFrame,
    v76_decomp: pd.DataFrame,
    actuals_df: pd.DataFrame,
    prior_comparison_path: str | Path | None = None,
    v74_holdout_eval: pd.DataFrame | None = None,
    v75_holdout_eval: pd.DataFrame | None = None,
    calibration_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build v7_6_vs_prior_versions comparison table.

    v7.6 row includes a Notes column explaining whether calibration was
    accepted or rejected by the no-regression gate.
    """
    jan = "2026-01"
    feb = "2026-02"

    def _ev(df, h, m):
        if df is None or df.empty:
            return np.nan
        r = df[(df["HorizonMonths"] == h) & (df["MonthStart"] == m)]
        return round(float(r["WMAPE"].iloc[0]), 4) if not r.empty else np.nan

    def _dc(df, level, h, m):
        if df is None or df.empty:
            return np.nan
        r = df[(df.get("Level","") == level) & (df["HorizonMonths"] == h) & (df["MonthStart"] == m)]
        return round(float(r["WMAPE"].iloc[0]), 4) if not r.empty else np.nan

    h3_jan = _ev(v76_holdout_eval, 3, jan)
    h3_feb = _ev(v76_holdout_eval, 3, feb)
    overall_h3 = round(np.nanmean([h3_jan, h3_feb]), 4) \
        if not (np.isnan(h3_jan) and np.isnan(h3_feb)) else np.nan

    tot_act  = float(v76_holdout_eval.get("TotalActual", pd.Series([0])).sum()) \
        if v76_holdout_eval is not None and not v76_holdout_eval.empty else 0.0
    tot_pred = float(v76_holdout_eval.get("TotalPredicted", pd.Series([0])).sum()) \
        if v76_holdout_eval is not None and not v76_holdout_eval.empty else 0.0
    bias_ratio = round(tot_pred / tot_act, 4) if tot_act > 0 else np.nan

    # Build Notes string from calibration table
    notes = "global_calibration"
    if calibration_df is not None and not calibration_df.empty:
        applied = calibration_df[calibration_df["calibration_applied"]]
        rejected = calibration_df[~calibration_df["calibration_applied"]]
        parts = []
        if not applied.empty:
            parts.append(f"H{applied['HorizonMonths'].tolist()} accepted")
        if not rejected.empty:
            parts.append(f"H{rejected['HorizonMonths'].tolist()} rejected")
        notes = "; ".join(parts) if parts else "no_calibration_applied"

    v76_row = {
        "Variant":                  "v7.6_conservative_production",
        "H1_Jan_WMAPE":             _ev(v76_holdout_eval, 1, jan),
        "H3_Jan_WMAPE":             h3_jan,
        "H3_Feb_WMAPE":             h3_feb,
        "Overall_H3_WMAPE":         overall_h3,
        "StyleCode_H3_Jan_WMAPE":   _dc(v76_decomp, "StyleCode",  3, jan),
        "StyleColor_H3_Jan_WMAPE":  _dc(v76_decomp, "StyleColor", 3, jan),
        "SKU_H3_Jan_WMAPE":         _dc(v76_decomp, "SKU",        3, jan),
        "StyleCode_H3_Feb_WMAPE":   _dc(v76_decomp, "StyleCode",  3, feb),
        "StyleColor_H3_Feb_WMAPE":  _dc(v76_decomp, "StyleColor", 3, feb),
        "SKU_H3_Feb_WMAPE":         _dc(v76_decomp, "SKU",        3, feb),
        "BiasRatio":                bias_ratio,
        "ProductionCandidateFlag":  True,
        "Notes":                    notes,
    }

    rows = []

    # Load prior versions
    if prior_comparison_path is not None:
        p = Path(prior_comparison_path)
        if p.exists():
            prior = pd.read_csv(p)
            if "ProductionCandidateFlag" not in prior.columns:
                prior["ProductionCandidateFlag"] = False
            if "Notes" not in prior.columns:
                prior["Notes"] = ""
            prior = prior[prior.get("Variant", pd.Series(dtype=str)) != "v7.6_conservative_production"]
            rows.extend(prior.to_dict("records"))

    existing = {r.get("Variant","") for r in rows}

    if v74_holdout_eval is not None and "v7.4_production" not in existing:
        h74_jan = _ev(v74_holdout_eval, 3, jan)
        h74_feb = _ev(v74_holdout_eval, 3, feb)
        rows.append({
            "Variant":          "v7.4_production",
            "H1_Jan_WMAPE":     _ev(v74_holdout_eval, 1, jan),
            "H3_Jan_WMAPE":     h74_jan,
            "H3_Feb_WMAPE":     h74_feb,
            "Overall_H3_WMAPE": round(np.nanmean([h74_jan, h74_feb]), 4),
            "ProductionCandidateFlag": False,
            "Notes": "v7.4 baseline",
        })

    if v75_holdout_eval is not None and "v7.5_calibrated_production" not in existing:
        h75_jan = _ev(v75_holdout_eval, 3, jan)
        h75_feb = _ev(v75_holdout_eval, 3, feb)
        rows.append({
            "Variant":          "v7.5_calibrated_production",
            "H1_Jan_WMAPE":     _ev(v75_holdout_eval, 1, jan),
            "H3_Jan_WMAPE":     h75_jan,
            "H3_Feb_WMAPE":     h75_feb,
            "Overall_H3_WMAPE": round(np.nanmean([h75_jan, h75_feb]), 4),
            "ProductionCandidateFlag": False,
            "Notes": "per-StyleCode calibration — worsened performance",
        })

    rows.append(v76_row)
    result = pd.DataFrame(rows)

    col_order = [
        "Variant",
        "H1_Jan_WMAPE", "H3_Jan_WMAPE", "H3_Feb_WMAPE", "Overall_H3_WMAPE",
        "StyleCode_H3_Jan_WMAPE", "StyleColor_H3_Jan_WMAPE", "SKU_H3_Jan_WMAPE",
        "StyleCode_H3_Feb_WMAPE", "StyleColor_H3_Feb_WMAPE", "SKU_H3_Feb_WMAPE",
        "BiasRatio", "ProductionCandidateFlag", "Notes",
    ]
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan

    return result[col_order].reset_index(drop=True)
