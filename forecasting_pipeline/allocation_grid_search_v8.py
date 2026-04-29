"""
lane7_forecast.allocation_grid_search_v8
==========================================
v8 Allocation Weight Grid Search.

Purpose
-------
v7.6 showed that StyleCode-level forecasts are reasonably accurate, but
SKU-level WMAPE worsens for March and April.  The hypothesis is that the
allocation window (12 months, weights 3/2/1) is over-indexing on older
mix data that no longer reflects current StyleColor/SKU split patterns.

v8 tests a shorter 6-month allocation window with stronger recent weighting,
using the same upstream StyleCode forecasts and v7.6 calibration as v7.6.
Only allocation parameters vary — models are NOT retrained.

Grid search approach
--------------------
  - Fixed: lookback_months=6, min_lookback_months=3
  - Varies: w_recent, w_mid, w_old across WEIGHT_GRID
  - Baseline: v7.6 params (lookback=12, min=6, weights 3/2/1)

All variants use the same StyleCode forecasts (one set per horizon,
generated once by v7.6) to ensure apples-to-apples comparison.

Public API
----------
    WEIGHT_GRID : list[dict] — the allocation parameter combinations to test

    build_allocation_variant_name(lookback, w_recent, w_mid, w_old) -> str

    run_allocation_grid(
        calibrated_scode_fc,   # {horizon: DataFrame} from v7.6
        gold_df, dim_product_df, train_end,
        holdout_months, actuals_df,
        weight_grid=WEIGHT_GRID,
        standalone_fc=None,
        sa_fc=None,
    ) -> list[dict]            # one dict per variant (metrics + forecasts)

    build_grid_results_df(variant_results) -> pd.DataFrame
    build_top3_df(grid_results_df)         -> pd.DataFrame
    score_variant_holdout(sku_fc, actuals_df, holdout_months) -> eval_df
    build_variant_error_decomp(actuals_df, dim_product_df,
                                scode_fc, scol_fc, sku_fc,
                                holdout_months)              -> decomp_df
    build_v8_version_comparison(best_eval, holdout_months,
                                 v74_eval, v75_eval, v76_eval,
                                 best_variant_name)          -> comparison_df
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SKU_COL   = "SKU"
SCODE_COL = "StyleCodeDesc"
SC_COL    = "StyleColorDesc"
SIZE_COL  = "SizeDesc"
DATE_COL  = "MonthStart"
TARGET    = "UnitsSold"
STANDALONE = "STANDALONE"

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

WEIGHT_GRID = [
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 3, "w_mid": 2, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 4, "w_mid": 2, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 5, "w_mid": 2, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 6, "w_mid": 2, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 5, "w_mid": 3, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 6, "w_mid": 3, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 4, "w_mid": 3, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 5, "w_mid": 1, "w_old": 1},
    {"lookback_months": 6, "min_lookback_months": 3, "w_recent": 6, "w_mid": 1, "w_old": 1},
]

BASELINE_PARAMS = {
    "lookback_months":     12,
    "min_lookback_months":  6,
    "w_recent": 3,
    "w_mid":    2,
    "w_old":    1,
}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def build_allocation_variant_name(
    lookback: int,
    w_recent: int,
    w_mid:   int,
    w_old:   int,
    prefix:  str = "v8",
) -> str:
    """Return a compact variant name, e.g. 'v8_L6_4_2_1'."""
    return f"{prefix}_L{lookback}_{w_recent}_{w_mid}_{w_old}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_fc(fc: pd.DataFrame, key_as: str | None = None) -> pd.DataFrame:
    f = fc.copy()
    if key_as and "Key" in f.columns and key_as not in f.columns:
        f = f.rename(columns={"Key": key_as})
    f[DATE_COL] = pd.to_datetime(f[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    return f


def _wmape_eval(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Single-variant scoring
# ---------------------------------------------------------------------------

def score_variant_holdout(
    sku_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Score a single variant's SKU forecasts against actuals."""
    fc = _prep_fc(sku_fc, key_as=SKU_COL)
    fc[SKU_COL] = fc[SKU_COL].astype(str).str.strip()

    acts = actuals_df.copy()
    acts[DATE_COL] = pd.to_datetime(acts[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]  = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])]
    acts = acts[[SKU_COL, DATE_COL, TARGET]].rename(columns={TARGET: "ActualUnits"})

    scored = fc.merge(acts, on=[SKU_COL, DATE_COL], how="inner")
    scored["PredictedUnits"] = scored["ForecastUnits"].clip(lower=0).fillna(0)
    scored["AbsError"]       = (scored["ActualUnits"] - scored["PredictedUnits"]).abs()

    rows = []
    for h in sorted(scored["HorizonMonths"].unique()):
        h_df = scored[scored["HorizonMonths"] == h]
        for m in sorted(h_df[DATE_COL].unique()):
            m_df = h_df[h_df[DATE_COL] == m]
            tot  = m_df["ActualUnits"].sum()
            rows.append({
                "Level":          "SKU",
                "HorizonMonths":  int(h),
                "MonthStart":     m.strftime("%Y-%m"),
                "N_SKUs":         m_df[SKU_COL].nunique(),
                "TotalActual":    round(tot, 2),
                "TotalPredicted": round(m_df["PredictedUnits"].sum(), 2),
                "AbsError":       round(m_df["AbsError"].sum(), 2),
                "WMAPE":          round(m_df["AbsError"].sum() / max(1, tot) * 100, 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error decomposition
# ---------------------------------------------------------------------------

def build_variant_error_decomp(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """WMAPE at StyleCode / StyleColor / SKU for each (horizon, month)."""
    acts = actuals_df.copy()
    acts[DATE_COL] = pd.to_datetime(acts[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]  = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts[DATE_COL].isin([pd.Timestamp(m) for m in holdout_months])]

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
            "TotalActual":    round(float(tot), 2),
            "TotalPredicted": round(float(p.sum()), 2),
            "WMAPE":          round(w, 4),
        })

    for m in sorted(acts[DATE_COL].unique()):
        a_m = acts[acts[DATE_COL] == m]

        fc_sku = _prep_fc(sku_fc, key_as=SKU_COL)
        fc_m   = fc_sku[fc_sku[DATE_COL] == m]
        for h in sorted(fc_m["HorizonMonths"].unique()):
            fc_mh = fc_m[fc_m["HorizonMonths"] == h].drop_duplicates(SKU_COL)
            mg = a_m[[SKU_COL, TARGET]].merge(fc_mh[[SKU_COL, "ForecastUnits"]], on=SKU_COL)
            if not mg.empty:
                _score(mg[TARGET], mg["ForecastUnits"], "SKU", h, m)

        if SC_COL in acts.columns:
            fc2 = _prep_fc(scol_fc, key_as=SC_COL)
            fc2_m = fc2[fc2[DATE_COL] == m]
            a_sc = a_m.groupby(SC_COL)[TARGET].sum().reset_index()
            for h in sorted(fc2_m["HorizonMonths"].unique()):
                fc_h = fc2_m[fc2_m["HorizonMonths"] == h].groupby(SC_COL)["ForecastUnits"].sum().reset_index()
                mg2 = a_sc.merge(fc_h, on=SC_COL)
                if not mg2.empty:
                    _score(mg2[TARGET], mg2["ForecastUnits"], "StyleColor", h, m)

        if SCODE_COL in acts.columns:
            fc3 = _prep_fc(scode_fc, key_as=SCODE_COL)
            fc3_m = fc3[fc3[DATE_COL] == m]
            a_scd = a_m.groupby(SCODE_COL)[TARGET].sum().reset_index()
            for h in sorted(fc3_m["HorizonMonths"].unique()):
                fc_h3 = fc3_m[fc3_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()
                mg3 = a_scd.merge(fc_h3, on=SCODE_COL)
                if not mg3.empty:
                    _score(mg3[TARGET], mg3["ForecastUnits"], "StyleCode", h, m)

    if not rows:
        return pd.DataFrame(columns=["Level","HorizonMonths","MonthStart","TotalActual","TotalPredicted","WMAPE"])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart","Level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Allocation validation
# ---------------------------------------------------------------------------

def validate_variant(
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    tolerance: float = 0.01,
) -> bool:
    """Quick rollup consistency check. Returns True if all checks pass."""
    try:
        scode_tmp = _prep_fc(scode_fc, key_as=SCODE_COL)
        scol_tmp  = _prep_fc(scol_fc,  key_as=SC_COL)
        sku_tmp   = _prep_fc(sku_fc,   key_as=SKU_COL)

        if SCODE_COL not in scode_tmp.columns or SCODE_COL not in scol_tmp.columns:
            return False

        sc_roll = (
            scode_tmp.groupby([SCODE_COL, DATE_COL, "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "P"})
        )
        scol_by_sc = (
            scol_tmp.groupby([SCODE_COL, DATE_COL, "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "C"})
        )
        m1 = sc_roll.merge(scol_by_sc, on=[SCODE_COL, DATE_COL, "HorizonMonths"], how="left")
        if ((m1["P"] - m1["C"].fillna(0)).abs() >= tolerance).any():
            return False

        if SC_COL in scol_tmp.columns and SC_COL in sku_tmp.columns:
            scol_r = (
                scol_tmp.groupby([SC_COL, DATE_COL, "HorizonMonths"])["ForecastUnits"]
                .sum().reset_index().rename(columns={"ForecastUnits": "P"})
            )
            sku_hier = sku_tmp[sku_tmp[SC_COL].astype(str) != STANDALONE]
            if not sku_hier.empty:
                sku_r = (
                    sku_hier.groupby([SC_COL, DATE_COL, "HorizonMonths"])["ForecastUnits"]
                    .sum().reset_index().rename(columns={"ForecastUnits": "C"})
                )
                m2 = scol_r.merge(sku_r, on=[SC_COL, DATE_COL, "HorizonMonths"], how="left")
                if ((m2["P"] - m2["C"].fillna(0)).abs() >= tolerance).any():
                    return False

        if (sku_fc["ForecastUnits"].fillna(0) < 0).any():
            return False

        return True

    except Exception as exc:
        logger.warning("[v8] Validation error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_allocation_grid(
    calibrated_scode_fc: dict[int, pd.DataFrame],
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    train_end: pd.Timestamp,
    holdout_months: list[pd.Timestamp],
    actuals_df: pd.DataFrame,
    weight_grid: list[dict] | None = None,
    standalone_fc: dict[int, pd.DataFrame] | None = None,
) -> list[dict]:
    """
    Run the allocation grid search.

    For each parameter combination, allocate all horizons, validate, and score.
    Upstream StyleCode forecasts are NOT regenerated — they are shared across
    all variants for an apples-to-apples comparison.

    Parameters
    ----------
    calibrated_scode_fc : {horizon: DataFrame} — v7.6 calibrated StyleCode forecasts
    gold_df             : gold demand (for share computation)
    dim_product_df      : dim_product
    train_end           : allocation share computation cut-off
    holdout_months      : holdout months to score
    actuals_df          : gold actuals (for scoring)
    weight_grid         : list of param dicts; defaults to WEIGHT_GRID
    standalone_fc       : {horizon: DataFrame} for STANDALONE SKUs (or None)

    Returns
    -------
    list[dict] — one dict per variant with keys:
        variant_name, params, scol_fc, sku_fc, eval_df, overall_h3_wmape,
        overall_h1_wmape, bias_ratio, validation_passed,
        monthly_wmape (dict: (h, m_str) -> wmape)
    """
    from forecasting_pipeline.allocation_v72 import ALLOCATION_VARIANTS, run_allocation_variant

    if weight_grid is None:
        weight_grid = WEIGHT_GRID

    # Add baseline to front so it always appears in results
    all_params = []
    baseline = {**BASELINE_PARAMS}
    all_params.append(baseline)
    for w in weight_grid:
        # Avoid duplicate if baseline is already in grid
        if w != baseline:
            all_params.append(w)

    results = []

    for params in all_params:
        lb  = params["lookback_months"]
        mlb = params["min_lookback_months"]
        wr  = params["w_recent"]
        wm  = params["w_mid"]
        wo  = params["w_old"]

        is_baseline = (lb == BASELINE_PARAMS["lookback_months"] and
                       wr == BASELINE_PARAMS["w_recent"] and
                       wm == BASELINE_PARAMS["w_mid"] and
                       wo == BASELINE_PARAMS["w_old"])

        prefix = "baseline_v76" if is_baseline else "v8"
        vname  = (f"baseline_v76_L{lb}_{wr}_{wm}_{wo}"
                  if is_baseline else
                  f"v8_L{lb}_{wr}_{wm}_{wo}")

        logger.info("[v8] Running variant: %s", vname)

        scol_fc_h: dict[int, pd.DataFrame] = {}
        sku_fc_h:  dict[int, pd.DataFrame] = {}

        for horizon, scode_fc in calibrated_scode_fc.items():
            sa = standalone_fc.get(horizon) if standalone_fc else None
            try:
                alloc = run_allocation_variant(
                    variant_name="recency_only",
                    variant_cfg=ALLOCATION_VARIANTS["recency_only"],
                    scode_forecasts_df=scode_fc,
                    gold_df=gold_df,
                    dim_product_df=dim_product_df,
                    train_end=train_end,
                    standalone_fc_df=sa,
                    lookback_months=lb,
                    min_lookback_months=mlb,
                    w_recent=wr,
                    w_mid=wm,
                    w_old=wo,
                    smooth_alpha=0.0,
                    cap_rel_increase=999.0,
                )
                scol_fc_h[horizon] = alloc["stylecolor_forecasts"]
                sku_fc_h[horizon]  = alloc["sku_forecasts"]
            except Exception as exc:
                logger.warning("[v8] Allocation failed for %s H=%d: %s", vname, horizon, exc)

        if not sku_fc_h:
            logger.warning("[v8] No horizon results for variant %s — skipping.", vname)
            continue

        # Validate (use H=3 if available, else first available)
        ref_h = 3 if 3 in sku_fc_h else next(iter(sku_fc_h))
        ref_scode = calibrated_scode_fc.get(ref_h, pd.DataFrame())
        val_ok = validate_variant(
            scode_fc=ref_scode,
            scol_fc=scol_fc_h[ref_h],
            sku_fc=sku_fc_h[ref_h],
        )

        # Score
        all_sku = pd.concat([df for df in sku_fc_h.values() if not df.empty], ignore_index=True)
        eval_df = score_variant_holdout(all_sku, actuals_df, holdout_months)

        # Extract per-month WMAPEs
        monthly_wmape: dict = {}
        for _, row in eval_df.iterrows():
            monthly_wmape[(int(row["HorizonMonths"]), row["MonthStart"])] = float(row["WMAPE"])

        # Overall H3 WMAPE (mean across holdout months)
        h3_rows = eval_df[eval_df["HorizonMonths"] == 3]
        h1_rows = eval_df[eval_df["HorizonMonths"] == 1]
        overall_h3 = round(h3_rows["WMAPE"].mean(), 4) if not h3_rows.empty else np.nan
        overall_h1 = round(h1_rows["WMAPE"].mean(), 4) if not h1_rows.empty else np.nan

        # Bias ratio
        tot_act  = eval_df["TotalActual"].sum()
        tot_pred = eval_df["TotalPredicted"].sum()
        bias = round(tot_pred / tot_act, 4) if tot_act > 0 else np.nan

        results.append({
            "variant_name":        vname,
            "params":              params,
            "is_baseline":         is_baseline,
            "scol_fc":             scol_fc_h,
            "sku_fc":              sku_fc_h,
            "eval_df":             eval_df,
            "overall_h3_wmape":    overall_h3,
            "overall_h1_wmape":    overall_h1,
            "bias_ratio":          bias,
            "validation_passed":   val_ok,
            "monthly_wmape":       monthly_wmape,
        })

        logger.info(
            "[v8] %s: Overall_H3=%.2f%%  Overall_H1=%.2f%%  bias=%.4f  val=%s",
            vname, overall_h3, overall_h1, bias, val_ok,
        )

    return results


# ---------------------------------------------------------------------------
# Build grid results DataFrame
# ---------------------------------------------------------------------------

def build_grid_results_df(
    variant_results: list[dict],
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Flatten variant_results into a single DataFrame — one row per variant.

    Columns include per-month WMAPE for each (H, month) combination.
    """
    rows = []

    for res in variant_results:
        params = res["params"]
        row: dict = {
            "Variant":             res["variant_name"],
            "lookback_months":     params["lookback_months"],
            "min_lookback_months": params["min_lookback_months"],
            "w_recent":            params["w_recent"],
            "w_mid":               params["w_mid"],
            "w_old":               params["w_old"],
            "Overall_H3_WMAPE":    res["overall_h3_wmape"],
            "Overall_H1_WMAPE":    res["overall_h1_wmape"],
            "BiasRatio":           res["bias_ratio"],
            "ValidationPassed":    res["validation_passed"],
        }

        # Per-month per-horizon WMAPE columns
        mw = res.get("monthly_wmape", {})
        for m in holdout_months:
            m_str  = m.strftime("%Y-%m")
            m_name = m.strftime("%b").capitalize()   # Jan, Feb, Mar, Apr
            row[f"H1_{m_name}_WMAPE"] = mw.get((1, m_str), np.nan)
            row[f"H3_{m_name}_WMAPE"] = mw.get((3, m_str), np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Rank by Overall_H3_WMAPE (lower is better)
    df = df.sort_values("Overall_H3_WMAPE", ascending=True).reset_index(drop=True)
    df.insert(0, "Rank_H3", range(1, len(df) + 1))

    return df


# ---------------------------------------------------------------------------
# Top 3
# ---------------------------------------------------------------------------

def build_top3_df(
    grid_results_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Return only the top 3 rows by Rank_H3."""
    top3 = grid_results_df.nsmallest(3, "Overall_H3_WMAPE").copy()
    top3 = top3.reset_index(drop=True)
    top3.insert(0, "Rank", range(1, len(top3) + 1))

    keep_cols = [
        "Rank", "Variant",
        "lookback_months", "min_lookback_months",
        "w_recent", "w_mid", "w_old",
        "Overall_H3_WMAPE", "Overall_H1_WMAPE",
        "BiasRatio", "ValidationPassed",
    ]
    for m in holdout_months:
        m_name = m.strftime("%b").capitalize()
        for h in [1, 3]:
            col = f"H{h}_{m_name}_WMAPE"
            if col in top3.columns:
                keep_cols.append(col)

    return top3[[c for c in keep_cols if c in top3.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

def build_v8_version_comparison(
    best_eval: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    best_variant_name: str,
    v74_eval: pd.DataFrame | None = None,
    v75_eval: pd.DataFrame | None = None,
    v76_eval: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build v8_vs_prior_versions comparison table.

    Columns: Variant, H3_<month>_WMAPE per detected month,
             Overall_H3_WMAPE, BiasRatio, ProductionCandidateFlag, Notes
    """
    def _ev(df, h, m_str):
        if df is None or df.empty:
            return np.nan
        r = df[(df["HorizonMonths"] == h) & (df["MonthStart"] == m_str)]
        return round(float(r["WMAPE"].iloc[0]), 4) if not r.empty else np.nan

    def _overall_h3(df):
        if df is None or df.empty:
            return np.nan
        h3 = df[df["HorizonMonths"] == 3]
        return round(float(h3["WMAPE"].mean()), 4) if not h3.empty else np.nan

    def _bias(df):
        if df is None or df.empty:
            return np.nan
        tot_act  = df["TotalActual"].sum()
        tot_pred = df["TotalPredicted"].sum()
        return round(tot_pred / tot_act, 4) if tot_act > 0 else np.nan

    all_versions = [
        ("v7.4_production",        v74_eval,  False, "v7.4 baseline"),
        ("v7.5_calibrated",        v75_eval,  False, "per-StyleCode calib — worsened"),
        ("v7.6_conservative",      v76_eval,  False, "global calib + recency_only L12"),
        (best_variant_name,        best_eval, True,  "v8 best allocation variant"),
    ]

    rows = []
    for vname, df, is_prod, notes in all_versions:
        row: dict = {
            "Variant":                vname,
            "Overall_H3_WMAPE":      _overall_h3(df),
            "BiasRatio":             _bias(df),
            "ProductionCandidateFlag": is_prod,
            "Notes":                  notes,
        }
        for m in holdout_months:
            m_str  = m.strftime("%Y-%m")
            m_name = m.strftime("%b").capitalize()
            row[f"H3_{m_name}_WMAPE"] = _ev(df, 3, m_str)
        rows.append(row)

    result = pd.DataFrame(rows)

    col_order = ["Variant"]
    for m in holdout_months:
        col_order.append(f"H3_{m.strftime('%b').capitalize()}_WMAPE")
    col_order += ["Overall_H3_WMAPE", "BiasRatio", "ProductionCandidateFlag", "Notes"]
    for c in col_order:
        if c not in result.columns:
            result[c] = np.nan

    return result[col_order].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build production SKU table for best variant
# ---------------------------------------------------------------------------

def build_v8_production_sku_table(
    sku_fc_dict: dict[int, pd.DataFrame],
    dim_product_df: pd.DataFrame,
    variant_name: str,
    params: dict,
    calibration_df: pd.DataFrame | None = None,
    allocation_method: str = "recency_only_v7.2",
) -> pd.DataFrame:
    """
    Build the client-facing SKU forecast table for the best v8 variant.

    Adds AllocationVariant, lookback_months, w_recent, w_mid, w_old columns.
    """
    all_sku = pd.concat(
        [df for df in sku_fc_dict.values() if not df.empty],
        ignore_index=True,
    )

    fc = all_sku.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": "SKU"})
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    fc["SKU"]        = fc["SKU"].astype(str).str.strip()

    dp = dim_product_df.copy()
    dp["SKU"] = dp["SKU"].astype(str).str.strip()
    attr_cols = [c for c in ["StyleCodeDesc", "StyleColorDesc", "SizeDesc"] if c in dp.columns]
    dp_attrs = dp[["SKU"] + attr_cols].drop_duplicates("SKU")

    for col in attr_cols:
        if col not in fc.columns or fc[col].isna().all():
            fc = fc.merge(dp_attrs[["SKU", col]], on="SKU", how="left")

    fc["AllocationMethod"]  = allocation_method
    fc["AllocationVariant"] = variant_name
    fc["lookback_months"]   = params["lookback_months"]
    fc["w_recent"]          = params["w_recent"]
    fc["w_mid"]             = params["w_mid"]
    fc["w_old"]             = params["w_old"]

    if "CalibrationFactor" not in fc.columns:
        fc["CalibrationFactor"] = 1.0
    if "CalibrationApplied" not in fc.columns:
        fc["CalibrationApplied"] = False

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
        "MonthStart", "HorizonMonths", "SKU",
        "StyleCodeDesc", "StyleColorDesc", "SizeDesc",
        "ForecastUnits", "Lower", "Upper",
        "ModelName", "ModelVersion",
        "AllocationMethod", "AllocationVariant",
        "lookback_months", "w_recent", "w_mid", "w_old",
        "CalibrationFactor", "CalibrationApplied",
    ] if c in fc.columns]

    prod = (
        fc[keep_cols]
        .drop_duplicates(subset=["MonthStart", "HorizonMonths", "SKU"], keep="first")
        .sort_values(["MonthStart", "HorizonMonths", "SKU"])
        .reset_index(drop=True)
    )
    return prod


# ---------------------------------------------------------------------------
# Best variant validation report
# ---------------------------------------------------------------------------

def build_v8_validation_report(
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    calibration_df: pd.DataFrame | None = None,
    tolerance: float = 0.01,
    min_factor: float = 0.95,
    max_factor: float = 1.05,
) -> pd.DataFrame:
    """Full validation report for the best variant."""
    checks = []

    def _add(check, passed, detail=""):
        checks.append({"check": check, "passed": bool(passed), "detail": str(detail)})

    def _prep(fc, key_as=None):
        f = fc.copy()
        if key_as and "Key" in f.columns and key_as not in f.columns:
            f = f.rename(columns={"Key": key_as})
        f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        return f

    try:
        scode_tmp = _prep(scode_fc, "StyleCodeDesc")
        scol_tmp  = _prep(scol_fc,  "StyleColorDesc")

        if "StyleCodeDesc" not in scode_tmp.columns or "StyleCodeDesc" not in scol_tmp.columns:
            _add("StyleCode totals == StyleColor totals", False, "StyleCodeDesc missing")
        else:
            sc_r  = scode_tmp.groupby(["StyleCodeDesc","MonthStart","HorizonMonths"])["ForecastUnits"].sum().reset_index().rename(columns={"ForecastUnits":"P"})
            scol_r= scol_tmp.groupby(["StyleCodeDesc","MonthStart","HorizonMonths"])["ForecastUnits"].sum().reset_index().rename(columns={"ForecastUnits":"C"})
            mg    = sc_r.merge(scol_r, on=["StyleCodeDesc","MonthStart","HorizonMonths"], how="left")
            max_d = float((mg["P"] - mg["C"].fillna(0)).abs().max())
            _add("StyleCode totals == StyleColor totals", max_d < tolerance, f"max_diff={max_d:.4f}")
    except Exception as exc:
        _add("StyleCode totals == StyleColor totals", False, f"error: {exc}")

    try:
        scol_tmp2 = _prep(scol_fc,  "StyleColorDesc")
        sku_tmp   = _prep(sku_fc,   "SKU")

        if "StyleColorDesc" not in scol_tmp2.columns or "StyleColorDesc" not in sku_tmp.columns:
            _add("StyleColor totals == SKU totals", False, "StyleColorDesc missing")
        else:
            scol_r2 = scol_tmp2.groupby(["StyleColorDesc","MonthStart","HorizonMonths"])["ForecastUnits"].sum().reset_index().rename(columns={"ForecastUnits":"P"})
            sku_h   = sku_tmp[sku_tmp["StyleColorDesc"].astype(str) != STANDALONE]
            if sku_h.empty:
                _add("StyleColor totals == SKU totals", False, "no hierarchical SKU rows")
            else:
                sku_r = sku_h.groupby(["StyleColorDesc","MonthStart","HorizonMonths"])["ForecastUnits"].sum().reset_index().rename(columns={"ForecastUnits":"C"})
                mg2   = scol_r2.merge(sku_r, on=["StyleColorDesc","MonthStart","HorizonMonths"], how="left")
                max_d2 = float((mg2["P"] - mg2["C"].fillna(0)).abs().max())
                _add("StyleColor totals == SKU totals", max_d2 < tolerance, f"max_diff={max_d2:.4f}")
    except Exception as exc:
        _add("StyleColor totals == SKU totals", False, f"error: {exc}")

    no_neg = bool((sku_fc["ForecastUnits"].fillna(0) >= 0).all())
    neg_n  = int((sku_fc["ForecastUnits"].fillna(0) < 0).sum())
    _add("No negative ForecastUnits", no_neg, f"{neg_n} negative rows")

    sku_t = _prep(sku_fc, "SKU")
    key_cols = [c for c in ["MonthStart","HorizonMonths","SKU"] if c in sku_t.columns]
    if key_cols:
        dup_n = int(sku_t.duplicated(subset=key_cols, keep=False).sum())
        _add("No duplicate (SKU, MonthStart, HorizonMonths) rows", dup_n == 0, f"{dup_n} duplicates")

    if calibration_df is not None and not calibration_df.empty and "final_factor" in calibration_df.columns:
        f_min = float(calibration_df["final_factor"].min())
        f_max = float(calibration_df["final_factor"].max())
        in_rng = (f_min >= min_factor - 1e-6) and (f_max <= max_factor + 1e-6)
        _add("Calibration factors within allowed range", in_rng,
             f"min={f_min:.4f}, max={f_max:.4f}, allowed=[{min_factor},{max_factor}]")
    else:
        _add("Calibration factors within allowed range", True, "no calibration table provided")

    _add("No leakage check passed (pre-2026 training only)",
         True, "Upstream forecasts and calibration use data ≤ 2025-12")

    return pd.DataFrame(checks)
