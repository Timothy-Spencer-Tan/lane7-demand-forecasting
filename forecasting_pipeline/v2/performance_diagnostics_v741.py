"""
lane7_forecast.performance_diagnostics_v741
=============================================
v7.4.1 Performance Diagnostics Layer.

Purpose
-------
Answers nine client-facing questions from existing v7.4 outputs, without
recomputing any forecasts:

  1. StyleCode performance table
  2. SKU performance table with error contribution
  3. Error concentration (cumulative error rank)
  4. Volume segmentation (Top 20 / Mid 30 / Bottom 50)
  5. StyleCode accuracy ranking with reliability tiers
  6. SKU accuracy ranking with accuracy tiers
  7. Bias analysis (Over / Under / Neutral at StyleCode level)
  8. High-impact error drivers (important SKUs we're getting wrong)
  9. Client summary table

All functions read from DataFrames — the caller is responsible for loading
the v7.4 CSV files.  No pipeline state is required.

Input DataFrames
----------------
The core input is ``holdout_preds_df``, which is ``v7_4_holdout_predictions.csv``.
Its expected columns (any subset works):
    SKU, MonthStart, HorizonMonths, PredictedUnits, ActualUnits,
    Error, AbsError, AbsPctError,
    StyleCodeDesc (optional — joined from dim_product if absent),
    StyleColorDesc (optional)

Alternatively, if only ``v7_4_production_sku_forecasts.csv`` is available and
actuals must be joined separately, pass actuals via ``actuals_df``.

Public API
----------
    load_and_merge(preds_path, actuals_path, dim_product_path,
                   holdout_months=None) -> scored_df

    build_stylecode_performance(scored_df, segments_df=None) -> df
    build_sku_performance(scored_df) -> df
    build_error_concentration(sku_perf_df) -> df
    build_volume_segmentation(sku_perf_df) -> df
    build_stylecode_accuracy_rank(stylecode_perf_df) -> df
    build_sku_accuracy_rank(sku_perf_df) -> df
    build_bias_analysis(stylecode_perf_df) -> df
    build_high_impact_errors(sku_perf_df) -> df
    build_client_summary(sku_perf_df, stylecode_perf_df) -> df

    build_all_diagnostics(scored_df, output_dir,
                          segments_df=None, horizon=3) -> dict[str, pd.DataFrame]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SKU_COL   = "SKU"
SCODE_COL = "StyleCodeDesc"
SC_COL    = "StyleColorDesc"
DATE_COL  = "MonthStart"

# Accuracy / reliability tier thresholds
SCODE_STRONG_WMAPE   = 40.0
SCODE_MODERATE_WMAPE = 70.0
SKU_STRONG_WMAPE     = 50.0
SKU_MODERATE_WMAPE   = 100.0

# Bias direction thresholds (forecast / actual ratio)
BIAS_OVER_THRESHOLD    = 1.05   # > 5% over-prediction
BIAS_UNDER_THRESHOLD   = 0.95   # < 5% under-prediction

# Volume segmentation splits
TOP_PCT    = 0.20
MIDDLE_PCT = 0.30
# BOTTOM    = 1 - TOP_PCT - MIDDLE_PCT = 0.50

# Error concentration: high-impact filter
ERROR_TOP_PCT  = 0.20    # top 20% by AbsoluteError
VOLUME_MEDIAN  = 0.50    # above median volume


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


def _bias_ratio(forecast: float, actual: float) -> float:
    return round(forecast / actual, 4) if actual > 0 else np.nan


def _reliability_tier_scode(wmape: float) -> str:
    if np.isnan(wmape):
        return "UNKNOWN"
    if wmape < SCODE_STRONG_WMAPE:
        return "STRONG"
    if wmape <= SCODE_MODERATE_WMAPE:
        return "MODERATE"
    return "WEAK"


def _accuracy_tier_sku(wmape: float) -> str:
    if np.isnan(wmape):
        return "UNKNOWN"
    if wmape < SKU_STRONG_WMAPE:
        return "STRONG"
    if wmape <= SKU_MODERATE_WMAPE:
        return "MODERATE"
    return "WEAK"


def _bias_direction(ratio: float) -> str:
    if np.isnan(ratio):
        return "Unknown"
    if ratio > BIAS_OVER_THRESHOLD:
        return "Over"
    if ratio < BIAS_UNDER_THRESHOLD:
        return "Under"
    return "Neutral"


def _normalise_scored(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names so the rest of the module sees a consistent schema:
        SKU, MonthStart, HorizonMonths, ActualUnits, PredictedUnits,
        AbsError, StyleCodeDesc, StyleColorDesc (last two optional)
    """
    d = df.copy()

    # Handle PredictedUnits vs ForecastUnits
    if "PredictedUnits" not in d.columns and "ForecastUnits" in d.columns:
        d["PredictedUnits"] = d["ForecastUnits"]
    if "ActualUnits" not in d.columns and "UnitsSold" in d.columns:
        d["ActualUnits"] = d["UnitsSold"]

    if "AbsError" not in d.columns:
        d["AbsError"] = (d["ActualUnits"] - d["PredictedUnits"]).abs()

    if DATE_COL in d.columns:
        d[DATE_COL] = pd.to_datetime(d[DATE_COL]).dt.to_period("M").dt.to_timestamp()

    if SKU_COL in d.columns:
        d[SKU_COL] = d[SKU_COL].astype(str).str.strip()

    return d


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_and_merge(
    preds_path: str | Path,
    actuals_path: str | Path | None = None,
    dim_product_path: str | Path | None = None,
    holdout_months: list[str] | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Load v7.4 holdout predictions and, if needed, join actuals and product attrs.

    Preferred input: ``v7_4_holdout_predictions.csv`` (already has ActualUnits).
    Fallback: ``v7_4_production_sku_forecasts.csv`` joined with gold actuals.

    Parameters
    ----------
    preds_path      : path to holdout predictions CSV
    actuals_path    : path to gold_fact_monthly_demand CSV (needed if preds
                      doesn't have ActualUnits column)
    dim_product_path: path to dim_product CSV (for StyleCodeDesc / StyleColorDesc
                      enrichment if missing from preds)
    holdout_months  : list of "YYYY-MM" strings to filter (default: keep all)
    horizon         : HorizonMonths to filter (default: keep all)

    Returns
    -------
    pd.DataFrame with consistent schema for all diagnostic functions
    """
    preds = pd.read_csv(preds_path, parse_dates=[DATE_COL])
    preds = _normalise_scored(preds)

    # Filter horizon
    if horizon is not None and "HorizonMonths" in preds.columns:
        preds = preds[preds["HorizonMonths"] == horizon].copy()

    # Filter holdout months
    if holdout_months:
        month_ts = [pd.Timestamp(m + "-01") for m in holdout_months]
        preds = preds[preds[DATE_COL].isin(month_ts)].copy()

    # Join actuals if not present
    if "ActualUnits" not in preds.columns and actuals_path is not None:
        acts = pd.read_csv(actuals_path, parse_dates=[DATE_COL])
        acts = _normalise_scored(acts)
        acts = acts[[SKU_COL, DATE_COL, "ActualUnits"]].drop_duplicates([SKU_COL, DATE_COL])
        preds = preds.merge(acts, on=[SKU_COL, DATE_COL], how="inner")

    # Enrich with StyleCodeDesc / StyleColorDesc from dim_product if missing
    if dim_product_path is not None:
        dp = pd.read_csv(dim_product_path)
        dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
        attr_cols = [c for c in [SCODE_COL, SC_COL] if c in dp.columns]
        if attr_cols:
            dp_slim = dp[[SKU_COL] + attr_cols].drop_duplicates(SKU_COL)
            missing_attrs = [c for c in attr_cols if c not in preds.columns or preds[c].isna().all()]
            if missing_attrs:
                preds = preds.merge(dp_slim[[SKU_COL] + missing_attrs], on=SKU_COL, how="left")

    logger.info(
        "[v7.4.1] Loaded %d scored rows, %d SKUs",
        len(preds), preds[SKU_COL].nunique() if SKU_COL in preds.columns else 0,
    )
    return preds.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostic 1 — StyleCode performance table
# ---------------------------------------------------------------------------

def build_stylecode_performance(
    scored_df: pd.DataFrame,
    segments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggregate holdout predictions to StyleCode level.

    Parameters
    ----------
    scored_df   : normalised scored DataFrame (from load_and_merge or direct)
    segments_df : v7_4_stylecode_segments.csv (optional — adds Segment column)

    Returns
    -------
    DataFrame sorted by TotalActual DESC with columns:
        StyleCodeDesc, TotalActual, TotalForecast, AbsoluteError,
        WMAPE, BiasRatio, N_SKUs, AvgSKUVolume, Segment
    """
    df = scored_df.copy()
    if SCODE_COL not in df.columns:
        raise ValueError(
            "StyleCodeDesc missing from scored_df. "
            "Pass dim_product_path to load_and_merge() to enrich."
        )

    grp = df.groupby(SCODE_COL, as_index=False).agg(
        TotalActual   =("ActualUnits",    "sum"),
        TotalForecast =("PredictedUnits", "sum"),
        AbsoluteError =("AbsError",       "sum"),
        N_SKUs        =(SKU_COL,          "nunique"),
    )
    grp["WMAPE"]       = grp.apply(
        lambda r: round(_wmape(pd.Series([r["TotalActual"]]),
                               pd.Series([r["TotalForecast"]])), 4), axis=1,
    )
    grp["WMAPE"]       = (grp["AbsoluteError"] / grp["TotalActual"].replace(0, np.nan) * 100).round(4)
    grp["BiasRatio"]   = (grp["TotalForecast"] / grp["TotalActual"].replace(0, np.nan)).round(4)
    grp["AvgSKUVolume"]= (grp["TotalActual"] / grp["N_SKUs"].replace(0, np.nan)).round(2)
    grp["AbsoluteError"] = grp["AbsoluteError"].round(2)

    if segments_df is not None and not segments_df.empty:
        seg_col = next(
            (c for c in ["Segment","segment"] if c in segments_df.columns), None
        )
        scode_col_seg = next(
            (c for c in [SCODE_COL,"SKU"] if c in segments_df.columns), None
        )
        if seg_col and scode_col_seg:
            seg_lu = segments_df[[scode_col_seg, seg_col]].rename(
                columns={scode_col_seg: SCODE_COL, seg_col: "Segment"}
            ).drop_duplicates(SCODE_COL)
            grp = grp.merge(seg_lu, on=SCODE_COL, how="left")
            grp["Segment"] = grp["Segment"].fillna("UNKNOWN")
    else:
        grp["Segment"] = "UNKNOWN"

    grp = grp.sort_values("TotalActual", ascending=False).reset_index(drop=True)

    col_order = [SCODE_COL, "TotalActual", "TotalForecast", "AbsoluteError",
                 "WMAPE", "BiasRatio", "N_SKUs", "AvgSKUVolume", "Segment"]
    return grp[[c for c in col_order if c in grp.columns]]


# ---------------------------------------------------------------------------
# Diagnostic 2 — SKU performance table
# ---------------------------------------------------------------------------

def build_sku_performance(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate holdout predictions to SKU level with error contribution.

    Returns
    -------
    DataFrame sorted by AbsoluteError DESC with columns:
        SKU, StyleCodeDesc, StyleColorDesc,
        TotalActual, TotalForecast, AbsoluteError, WMAPE, BiasRatio,
        ErrorContributionPct, Rank_By_ErrorContribution
    """
    df = scored_df.copy()

    agg_cols = {
        "TotalActual":   ("ActualUnits",    "sum"),
        "TotalForecast": ("PredictedUnits", "sum"),
        "AbsoluteError": ("AbsError",       "sum"),
    }
    group_cols = [SKU_COL]
    for c in [SCODE_COL, SC_COL]:
        if c in df.columns:
            group_cols.append(c)

    grp = df.groupby(group_cols, as_index=False).agg(**agg_cols)
    grp["WMAPE"]     = (grp["AbsoluteError"] / grp["TotalActual"].replace(0, np.nan) * 100).round(4)
    grp["BiasRatio"] = (grp["TotalForecast"] / grp["TotalActual"].replace(0, np.nan)).round(4)
    grp["AbsoluteError"] = grp["AbsoluteError"].round(2)

    total_abs_error = grp["AbsoluteError"].sum()
    grp["ErrorContributionPct"] = (
        grp["AbsoluteError"] / total_abs_error * 100
        if total_abs_error > 0 else 0.0
    ).round(4)

    grp = grp.sort_values("AbsoluteError", ascending=False).reset_index(drop=True)
    grp["Rank_By_ErrorContribution"] = range(1, len(grp) + 1)

    col_order = [SKU_COL, SCODE_COL, SC_COL,
                 "TotalActual", "TotalForecast", "AbsoluteError",
                 "WMAPE", "BiasRatio",
                 "ErrorContributionPct", "Rank_By_ErrorContribution"]
    return grp[[c for c in col_order if c in grp.columns]]


# ---------------------------------------------------------------------------
# Diagnostic 3 — Error concentration
# ---------------------------------------------------------------------------

def build_error_concentration(sku_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative error contribution ranked by AbsoluteError.

    Returns
    -------
    DataFrame with columns:
        Rank, SKU, AbsoluteError, CumulativeError, CumulativeErrorPct
    """
    df = sku_perf_df[[SKU_COL, "AbsoluteError"]].copy()
    df = df.sort_values("AbsoluteError", ascending=False).reset_index(drop=True)
    df["Rank"]             = range(1, len(df) + 1)
    df["CumulativeError"]  = df["AbsoluteError"].cumsum().round(2)
    total_err              = df["AbsoluteError"].sum()
    df["CumulativeErrorPct"] = (df["CumulativeError"] / total_err * 100
                                if total_err > 0 else 0.0).round(4)

    return df[["Rank", SKU_COL, "AbsoluteError", "CumulativeError", "CumulativeErrorPct"]]


# ---------------------------------------------------------------------------
# Diagnostic 4 — Volume segmentation
# ---------------------------------------------------------------------------

def build_volume_segmentation(sku_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split SKUs into Top-20% / Middle-30% / Bottom-50% by TotalActual
    and compute WMAPE, BiasRatio per segment.

    Returns
    -------
    DataFrame with columns:
        SegmentName, SKU_Count, TotalActual, TotalForecast,
        AbsoluteError, WMAPE, BiasRatio
    """
    df = sku_perf_df[["TotalActual", "TotalForecast", "AbsoluteError"]].copy()
    df = df.sort_values("TotalActual", ascending=False).reset_index(drop=True)

    n          = len(df)
    top_n      = max(1, int(np.ceil(n * TOP_PCT)))
    middle_n   = max(1, int(np.ceil(n * MIDDLE_PCT)))
    # Remainder goes to bottom segment

    def _label(i):
        if i < top_n:
            return "Top 20% (High Volume)"
        if i < top_n + middle_n:
            return "Middle 30% (Mid Volume)"
        return "Bottom 50% (Low Volume)"

    df["SegmentName"] = [_label(i) for i in range(n)]

    rows = []
    for seg in ["Top 20% (High Volume)", "Middle 30% (Mid Volume)", "Bottom 50% (Low Volume)"]:
        s = df[df["SegmentName"] == seg]
        if s.empty:
            continue
        tot_act  = s["TotalActual"].sum()
        tot_fc   = s["TotalForecast"].sum()
        abs_err  = s["AbsoluteError"].sum()
        wmape    = round(abs_err / tot_act * 100, 4) if tot_act > 0 else np.nan
        bias     = round(tot_fc / tot_act, 4) if tot_act > 0 else np.nan
        rows.append({
            "SegmentName":  seg,
            "SKU_Count":    len(s),
            "TotalActual":  round(tot_act, 2),
            "TotalForecast":round(tot_fc,  2),
            "AbsoluteError":round(abs_err, 2),
            "WMAPE":        wmape,
            "BiasRatio":    bias,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic 5 — StyleCode accuracy ranking
# ---------------------------------------------------------------------------

def build_stylecode_accuracy_rank(stylecode_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank StyleCodes by WMAPE and assign reliability tiers.

    Returns
    -------
    DataFrame sorted by WMAPE ASC with columns:
        Rank, StyleCodeDesc, TotalActual, WMAPE, BiasRatio,
        ForecastReliabilityTier
    """
    df = stylecode_perf_df[[SCODE_COL, "TotalActual", "WMAPE", "BiasRatio"]].copy()
    df = df.dropna(subset=["WMAPE"])
    df = df.sort_values("WMAPE", ascending=True).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    df["ForecastReliabilityTier"] = df["WMAPE"].apply(_reliability_tier_scode)

    col_order = ["Rank", SCODE_COL, "TotalActual", "WMAPE", "BiasRatio", "ForecastReliabilityTier"]
    return df[[c for c in col_order if c in df.columns]]


# ---------------------------------------------------------------------------
# Diagnostic 6 — SKU accuracy ranking
# ---------------------------------------------------------------------------

def build_sku_accuracy_rank(sku_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank SKUs by WMAPE and assign accuracy tiers.

    Returns
    -------
    DataFrame sorted by WMAPE ASC with columns:
        Rank, SKU, StyleCodeDesc, TotalActual, WMAPE, BiasRatio, AccuracyTier
    """
    keep = [SKU_COL, "TotalActual", "WMAPE", "BiasRatio"]
    if SCODE_COL in sku_perf_df.columns:
        keep.append(SCODE_COL)

    df = sku_perf_df[keep].copy()
    df = df.dropna(subset=["WMAPE"])
    df = df.sort_values("WMAPE", ascending=True).reset_index(drop=True)
    df["Rank"]         = range(1, len(df) + 1)
    df["AccuracyTier"] = df["WMAPE"].apply(_accuracy_tier_sku)

    col_order = ["Rank", SKU_COL, SCODE_COL, "TotalActual", "WMAPE", "BiasRatio", "AccuracyTier"]
    return df[[c for c in col_order if c in df.columns]]


# ---------------------------------------------------------------------------
# Diagnostic 7 — Bias analysis
# ---------------------------------------------------------------------------

def build_bias_analysis(stylecode_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each StyleCode by bias direction.

    Returns
    -------
    DataFrame sorted by BiasRatio DESC with columns:
        StyleCodeDesc, TotalActual, TotalForecast, BiasRatio, BiasDirection
    """
    df = stylecode_perf_df[[SCODE_COL, "TotalActual", "TotalForecast", "BiasRatio"]].copy()
    df["BiasDirection"] = df["BiasRatio"].apply(_bias_direction)
    df = df.sort_values("BiasRatio", ascending=False).reset_index(drop=True)
    return df[[SCODE_COL, "TotalActual", "TotalForecast", "BiasRatio", "BiasDirection"]]


# ---------------------------------------------------------------------------
# Diagnostic 8 — High-impact error drivers
# ---------------------------------------------------------------------------

def build_high_impact_errors(sku_perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify SKUs where error is high AND volume is significant.

    Filter criteria:
        AbsoluteError in top 20% of all SKUs by AbsoluteError
        AND TotalActual > median TotalActual across all SKUs

    Returns
    -------
    DataFrame sorted by AbsoluteError DESC with the same columns as sku_perf_df
    """
    df = sku_perf_df.copy()

    err_thresh    = df["AbsoluteError"].quantile(1.0 - ERROR_TOP_PCT)
    volume_median = df["TotalActual"].median()

    high_impact = df[
        (df["AbsoluteError"] >= err_thresh) &
        (df["TotalActual"]   > volume_median)
    ].sort_values("AbsoluteError", ascending=False).reset_index(drop=True)

    logger.info(
        "[v7.4.1] High-impact errors: %d SKUs (error >= %.1f, volume > %.1f)",
        len(high_impact), err_thresh, volume_median,
    )
    return high_impact


# ---------------------------------------------------------------------------
# Diagnostic 9 — Client summary table
# ---------------------------------------------------------------------------

def build_client_summary(
    sku_perf_df: pd.DataFrame,
    stylecode_perf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-metric client presentation summary.

    Returns
    -------
    DataFrame with columns: Metric, Value
    """
    total_skus    = int(sku_perf_df[SKU_COL].nunique())
    total_volume  = float(sku_perf_df["TotalActual"].sum())
    total_fc      = float(sku_perf_df["TotalForecast"].sum())
    total_abs_err = float(sku_perf_df["AbsoluteError"].sum())
    overall_wmape = round(total_abs_err / total_volume * 100, 2) if total_volume > 0 else np.nan

    # Volume segments
    df_sorted = sku_perf_df.sort_values("TotalActual", ascending=False).reset_index(drop=True)
    n         = len(df_sorted)
    top_n     = max(1, int(np.ceil(n * TOP_PCT)))

    top_df  = df_sorted.iloc[:top_n]
    bot_df  = df_sorted.iloc[top_n + max(1, int(np.ceil(n * MIDDLE_PCT))):]

    top_wmape = round(
        top_df["AbsoluteError"].sum() / top_df["TotalActual"].sum() * 100, 2
    ) if top_df["TotalActual"].sum() > 0 else np.nan

    bot_wmape = round(
        bot_df["AbsoluteError"].sum() / bot_df["TotalActual"].sum() * 100, 2
    ) if not bot_df.empty and bot_df["TotalActual"].sum() > 0 else np.nan

    # Error concentration: top 10 SKUs
    top10_err   = float(sku_perf_df.nlargest(10, "AbsoluteError")["AbsoluteError"].sum())
    top10_pct   = round(top10_err / total_abs_err * 100, 2) if total_abs_err > 0 else np.nan

    # Tier distribution
    if "AccuracyTier" in sku_perf_df.columns:
        tier_counts = sku_perf_df["AccuracyTier"].value_counts()
    else:
        # Recompute tiers inline if not pre-computed
        tiers = sku_perf_df["WMAPE"].apply(_accuracy_tier_sku)
        tier_counts = tiers.value_counts()

    def _tier_pct(tier):
        return round(tier_counts.get(tier, 0) / total_skus * 100, 2) if total_skus > 0 else 0.0

    rows = [
        {"Metric": "Total SKUs",                           "Value": total_skus},
        {"Metric": "Total Forecast Volume (units)",        "Value": round(total_fc, 0)},
        {"Metric": "Total Actual Volume (units)",          "Value": round(total_volume, 0)},
        {"Metric": "Overall WMAPE (%)",                    "Value": overall_wmape},
        {"Metric": "Top 20% SKU WMAPE (%)",                "Value": top_wmape},
        {"Metric": "Bottom 50% SKU WMAPE (%)",             "Value": bot_wmape},
        {"Metric": "% of error from Top 10 SKUs",          "Value": top10_pct},
        {"Metric": "% of SKUs — STRONG accuracy (<50 WMAPE)","Value": _tier_pct("STRONG")},
        {"Metric": "% of SKUs — MODERATE accuracy (50–100)", "Value": _tier_pct("MODERATE")},
        {"Metric": "% of SKUs — WEAK accuracy (>100)",       "Value": _tier_pct("WEAK")},
    ]

    # StyleCode-level summary
    if stylecode_perf_df is not None and not stylecode_perf_df.empty:
        n_sc = int(stylecode_perf_df[SCODE_COL].nunique())
        if "ForecastReliabilityTier" not in stylecode_perf_df.columns:
            stylecode_perf_df = stylecode_perf_df.copy()
            stylecode_perf_df["ForecastReliabilityTier"] = (
                stylecode_perf_df["WMAPE"].apply(_reliability_tier_scode)
            )
        sc_tiers = stylecode_perf_df["ForecastReliabilityTier"].value_counts()
        rows += [
            {"Metric": "Total StyleCodes",                                   "Value": n_sc},
            {"Metric": "% of StyleCodes — STRONG (<40 WMAPE)",              "Value": round(sc_tiers.get("STRONG",0)/n_sc*100,2)},
            {"Metric": "% of StyleCodes — MODERATE (40–70 WMAPE)",          "Value": round(sc_tiers.get("MODERATE",0)/n_sc*100,2)},
            {"Metric": "% of StyleCodes — WEAK (>70 WMAPE)",                "Value": round(sc_tiers.get("WEAK",0)/n_sc*100,2)},
        ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def build_all_diagnostics(
    scored_df: pd.DataFrame,
    output_dir: str | Path,
    segments_df: pd.DataFrame | None = None,
    horizon: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all nine diagnostic tables and write CSVs to output_dir.

    Parameters
    ----------
    scored_df  : normalised scored DataFrame (from load_and_merge or direct)
    output_dir : folder to write all CSVs
    segments_df: optional v7_4_stylecode_segments.csv for Segment column
    horizon    : if provided, filter scored_df to this HorizonMonths first

    Returns
    -------
    dict mapping short name → DataFrame, e.g. "stylecode_performance" → df
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scored_df.copy()
    if horizon is not None and "HorizonMonths" in df.columns:
        df = df[df["HorizonMonths"] == horizon].copy()
        logger.info("[v7.4.1] Filtered to H=%d: %d rows", horizon, len(df))

    results = {}

    logger.info("[v7.4.1] Building StyleCode performance table…")
    sc_perf = build_stylecode_performance(df, segments_df=segments_df)
    sc_perf.to_csv(output_dir / "v7_4_1_stylecode_performance.csv", index=False)
    results["stylecode_performance"] = sc_perf

    logger.info("[v7.4.1] Building SKU performance table…")
    sku_perf = build_sku_performance(df)
    sku_perf.to_csv(output_dir / "v7_4_1_sku_performance.csv", index=False)
    results["sku_performance"] = sku_perf

    logger.info("[v7.4.1] Building error concentration table…")
    err_conc = build_error_concentration(sku_perf)
    err_conc.to_csv(output_dir / "v7_4_1_error_concentration.csv", index=False)
    results["error_concentration"] = err_conc

    logger.info("[v7.4.1] Building volume segmentation table…")
    vol_seg = build_volume_segmentation(sku_perf)
    vol_seg.to_csv(output_dir / "v7_4_1_volume_segmentation.csv", index=False)
    results["volume_segmentation"] = vol_seg

    logger.info("[v7.4.1] Building StyleCode accuracy rank…")
    sc_rank = build_stylecode_accuracy_rank(sc_perf)
    sc_rank.to_csv(output_dir / "v7_4_1_stylecode_accuracy_rank.csv", index=False)
    results["stylecode_accuracy_rank"] = sc_rank

    logger.info("[v7.4.1] Building SKU accuracy rank…")
    sku_rank = build_sku_accuracy_rank(sku_perf)
    sku_rank.to_csv(output_dir / "v7_4_1_sku_accuracy_rank.csv", index=False)
    results["sku_accuracy_rank"] = sku_rank

    logger.info("[v7.4.1] Building bias analysis…")
    bias = build_bias_analysis(sc_perf)
    bias.to_csv(output_dir / "v7_4_1_bias_analysis.csv", index=False)
    results["bias_analysis"] = bias

    logger.info("[v7.4.1] Building high-impact errors…")
    high_err = build_high_impact_errors(sku_perf)
    high_err.to_csv(output_dir / "v7_4_1_high_impact_errors.csv", index=False)
    results["high_impact_errors"] = high_err

    logger.info("[v7.4.1] Building client summary…")
    client_summary = build_client_summary(sku_perf, sc_perf)
    client_summary.to_csv(output_dir / "v7_4_1_client_summary.csv", index=False)
    results["client_summary"] = client_summary

    logger.info(
        "[v7.4.1] All 9 diagnostic tables written to %s", output_dir,
    )
    return results
