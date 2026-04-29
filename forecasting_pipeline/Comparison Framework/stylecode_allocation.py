"""
lane7_forecast.stylecode_allocation
=====================================
v7 hierarchical allocation layer: StyleCodeDesc → StyleColorDesc → SKU.

Architecture
------------
v7 adds one more level above v6.  The full three-level hierarchy is:

    Level 1: StyleCodeDesc    ← model trains and forecasts here (v7 NEW)
    Level 2: StyleColorDesc   ← intermediate allocation (v7 NEW)
    Level 3: SizeDesc / SKU   ← final allocation (reuses v6 allocation.py)

v6 allocation.py is **unchanged**.  This module handles only the new
StyleCode → StyleColor step.  After this step, the resulting StyleColor
forecasts are fed into the existing ``allocate_to_sku()`` from allocation.py.

Allocation algorithm (StyleCode → StyleColor)
---------------------------------------------
For each (StyleCodeDesc, forecast_month):
    1. Compute color_share[StyleColorDesc] from look-back window.
       Priority order:
         (a) Configurable window (default 12 months) of training history
         (b) Full training history
         (c) Equal distribution across all dim_product StyleColors for the code
    2. Multiply: StyleColor_forecast = StyleCode_forecast × color_share
    3. Sum-consistency: sum(StyleColor) == StyleCode  (enforced by normalisation)

STANDALONE handling
-------------------
StyleCodes that cannot be identified from dim_product:
    - SKUs with no dim_product row → STANDALONE
    - SKUs with null StyleCodeDesc → STANDALONE
    - SKUs with null StyleColorDesc but valid StyleCodeDesc → routed through
      SKU-level fallback inside v7 pipeline

Public API
----------
    build_stylecode_demand(gold_df, dim_product_df)
        -> (stylecode_demand_df, standalone_skus_list)

    compute_stylecolor_shares(gold_df, dim_product_df,
                              lookback_months=12, min_lookback_months=6,
                              train_end=None)
        -> color_shares_df  (StyleCodeDesc, StyleColorDesc, share)

    allocate_to_stylecolor(stylecode_forecasts_df, color_shares_df,
                           dim_product_df)
        -> stylecolor_forecasts_df  (same schema, Key = StyleColorDesc)

    validate_stylecode_allocation(stylecode_forecasts_df, stylecolor_forecasts_df)
        -> dict  (sum_check_passed, max_abs_diff, share_check_passed, ...)

    get_v7_standalone_skus(gold_df, dim_product_df)
        -> list[str]   SKUs that must bypass the full hierarchy

    build_v7_coverage_report(gold_df, dim_product_df)
        -> dict   with counts of StyleCodes, StyleColors, SKUs, cold-starts
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
SCODE_COL  = "StyleCodeDesc"    # Level-1 key (NEW in v7)
SC_COL     = "StyleColorDesc"   # Level-2 key (was Level-1 in v6)
SIZE_COL   = "SizeDesc"         # Level-3 key (unchanged)
SKU_COL    = "SKU"              # Level-4 / final key
DATE_COL   = "MonthStart"
TARGET_COL = "UnitsSold"
STANDALONE = "STANDALONE"       # sentinel for unmappable items


# ---------------------------------------------------------------------------
# Step 1 — Build StyleCode-level demand table
# ---------------------------------------------------------------------------

def build_stylecode_demand(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate SKU-level monthly demand to StyleCodeDesc × Month.

    Parameters
    ----------
    gold_df        : gold_fact_monthly_demand_v2 (MonthStart, SKU, UnitsSold)
    dim_product_df : dim_product (SKU, StyleCodeDesc, StyleColorDesc, SizeDesc)

    Returns
    -------
    (stylecode_demand, standalone_skus)

    stylecode_demand : DataFrame with columns
        StyleCodeDesc, MonthStart, UnitsSold, [Revenue]
        One row per (StyleCodeDesc, MonthStart).

    standalone_skus : list[str]
        SKUs that cannot be mapped into the StyleCode hierarchy (no dim_product
        row, or null StyleCodeDesc).  These are handled separately.
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()

    # Attach StyleCodeDesc from dim_product
    dp = dim_product_df[[SKU_COL, SCODE_COL]].copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp = dp.drop_duplicates(subset=[SKU_COL], keep="first")

    merged = gold.merge(dp, on=SKU_COL, how="left")

    # STANDALONE = no dim_product row OR null StyleCodeDesc
    standalone_mask = merged[SCODE_COL].isna()
    standalone_skus = merged.loc[standalone_mask, SKU_COL].unique().tolist()

    n_sa_rows  = standalone_mask.sum()
    n_sa_units = merged.loc[standalone_mask, TARGET_COL].sum()
    logger.info(
        "[v7] STANDALONE (StyleCode level): %d SKUs, %d rows, %s units (%.2f%% of total)",
        len(standalone_skus),
        n_sa_rows,
        f"{n_sa_units:,.0f}",
        n_sa_units / max(1, merged[TARGET_COL].sum()) * 100,
    )

    # Aggregate hierarchical rows to StyleCodeDesc × Month
    hierarchical = merged[~standalone_mask].copy()
    agg_cols = {TARGET_COL: "sum"}
    if "Revenue" in hierarchical.columns:
        agg_cols["Revenue"] = "sum"

    scode_demand = (
        hierarchical
        .groupby([SCODE_COL, DATE_COL], as_index=False)
        .agg(agg_cols)
        .sort_values([SCODE_COL, DATE_COL])
        .reset_index(drop=True)
    )

    logger.info(
        "[v7] StyleCode demand table: %d rows, %d unique StyleCodes, %s → %s",
        len(scode_demand),
        scode_demand[SCODE_COL].nunique(),
        scode_demand[DATE_COL].min().strftime("%Y-%m"),
        scode_demand[DATE_COL].max().strftime("%Y-%m"),
    )
    return scode_demand, standalone_skus


# ---------------------------------------------------------------------------
# Step 2 — Compute StyleColor shares within each StyleCode
# ---------------------------------------------------------------------------

def compute_stylecolor_shares(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    lookback_months: int = 12,
    min_lookback_months: int = 6,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute the historical StyleColor share distribution within each StyleCode.

    These shares are used to allocate a StyleCode-level forecast down to the
    individual StyleColorDesc entries.

    Algorithm (applied in priority order):
        (a) Primary window: last ``lookback_months`` months ≤ train_end
        (b) Fallback A:     full history ≤ train_end
        (c) Fallback B:     equal split across all dim_product StyleColors

    Parameters
    ----------
    gold_df             : gold_fact_monthly_demand_v2
    dim_product_df      : dim_product with SKU, StyleCodeDesc, StyleColorDesc
    lookback_months     : primary look-back window (default 12)
    min_lookback_months : minimum months required to use primary window
    train_end           : cut-off date.  Defaults to max MonthStart in gold_df.

    Returns
    -------
    pd.DataFrame with columns:
        StyleCodeDesc, StyleColorDesc, units_in_window, share, fallback_used

    ``share`` values sum to 1.0 per StyleCodeDesc (normalised if needed).
    ``fallback_used`` ∈ {"primary", "full_history", "equal"}
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()

    if train_end is None:
        train_end = gold[DATE_COL].max()
    else:
        train_end = pd.Timestamp(train_end)

    gold = gold[gold[DATE_COL] <= train_end].copy()

    primary_start = train_end - pd.DateOffset(months=lookback_months - 1)
    primary_start = pd.Timestamp(primary_start.year, primary_start.month, 1)

    # dim_product: keep only rows with both StyleCodeDesc and StyleColorDesc
    dp = dim_product_df[[SKU_COL, SCODE_COL, SC_COL]].copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp = dp[dp[SCODE_COL].notna() & dp[SC_COL].notna()]
    dp = dp.drop_duplicates(subset=[SKU_COL], keep="first")

    # Map of StyleCode → set of StyleColors (from dim_product)
    scode_to_scs = (
        dp.groupby(SCODE_COL)[SC_COL]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )

    # Attach StyleCodeDesc and StyleColorDesc to gold
    gold_mapped = gold.merge(dp[[SKU_COL, SCODE_COL, SC_COL]], on=SKU_COL, how="inner")

    rows = []

    for scode, sc_list in scode_to_scs.items():
        sc_gold = gold_mapped[gold_mapped[SCODE_COL] == scode]

        # ── (a) Primary window ──────────────────────────────────────────────
        primary_gold      = sc_gold[sc_gold[DATE_COL] >= primary_start]
        n_primary_months  = primary_gold[DATE_COL].nunique()

        if n_primary_months >= min_lookback_months and primary_gold[TARGET_COL].sum() > 0:
            window_gold = primary_gold
            fallback    = "primary"
        elif sc_gold[TARGET_COL].sum() > 0:
            window_gold = sc_gold
            fallback    = "full_history"
            logger.debug(
                "[v7] StyleColor shares for '%s': primary had %d months — using full history",
                scode, n_primary_months,
            )
        else:
            window_gold = None
            fallback    = "equal"
            logger.debug(
                "[v7] StyleColor shares for '%s': no sales — using equal distribution",
                scode,
            )

        if fallback != "equal" and window_gold is not None:
            sc_units = (
                window_gold.groupby(SC_COL, as_index=False)[TARGET_COL]
                .sum()
                .rename(columns={TARGET_COL: "units_in_window"})
            )
            total = sc_units["units_in_window"].sum()
            sc_units["share"] = sc_units["units_in_window"] / total if total > 0 else 0.0

            # Ensure all dim_product StyleColors are present (0 share if missing)
            all_scs = pd.DataFrame({SC_COL: sc_list})
            sc_shares = all_scs.merge(
                sc_units[[SC_COL, "units_in_window", "share"]],
                on=SC_COL, how="left",
            )
            sc_shares["units_in_window"] = sc_shares["units_in_window"].fillna(0.0)
            sc_shares["share"]           = sc_shares["share"].fillna(0.0)
        else:
            # Equal distribution
            n = len(sc_list)
            sc_shares = pd.DataFrame({
                SC_COL:            sc_list,
                "units_in_window": [0.0] * n,
                "share":           [1.0 / n if n > 0 else 0.0] * n,
            })

        sc_shares[SCODE_COL]       = scode
        sc_shares["fallback_used"] = fallback
        rows.append(sc_shares)

    if not rows:
        logger.warning("[v7] No StyleCode→StyleColor shares computed — dim_product may be empty.")
        return pd.DataFrame(columns=[SCODE_COL, SC_COL, "units_in_window", "share", "fallback_used"])

    result = pd.concat(rows, ignore_index=True)
    result = result[[SCODE_COL, SC_COL, "units_in_window", "share", "fallback_used"]]

    # Normalise so shares sum exactly to 1.0 per StyleCode
    totals = result.groupby(SCODE_COL)["share"].transform("sum")
    off    = ((result.groupby(SCODE_COL)["share"].sum() - 1.0).abs() > 1e-6).sum()
    if off > 0:
        logger.warning(
            "[v7] %d StyleCodes have color shares not summing to 1.0 — normalising.", off
        )
        result["share"] = np.where(totals > 0, result["share"] / totals, 0.0)

    # Audit log
    fb_counts = result.drop_duplicates(SCODE_COL)["fallback_used"].value_counts()
    logger.info(
        "[v7] StyleColor shares — primary: %d, full_history: %d, equal: %d StyleCodes",
        fb_counts.get("primary", 0),
        fb_counts.get("full_history", 0),
        fb_counts.get("equal", 0),
    )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Allocate StyleCode forecasts → StyleColor forecasts
# ---------------------------------------------------------------------------

def allocate_to_stylecolor(
    stylecode_forecasts_df: pd.DataFrame,
    color_shares_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Disaggregate StyleCode-level forecasts to StyleColorDesc-level forecasts.

    The resulting DataFrame has the same schema as the input but with
    ``Key = StyleColorDesc`` (one row per StyleColor per forecast month).
    It can be fed directly into allocation.allocate_to_sku() for the
    second disaggregation step (StyleColor → SKU).

    Parameters
    ----------
    stylecode_forecasts_df : output of generate_forecasts() at StyleCode level.
                             Columns: Key (=StyleCodeDesc), MonthStart,
                             ForecastUnits, Lower, Upper, HorizonMonths,
                             ModelName, ModelVersion, RunDate.
    color_shares_df        : output of compute_stylecolor_shares().
                             Columns: StyleCodeDesc, StyleColorDesc, share.
    dim_product_df         : dim_product (used for schema validation only).

    Returns
    -------
    pd.DataFrame with Key = StyleColorDesc and same numeric schema.
    Sum-consistency: sum(StyleColor ForecastUnits) == StyleCode ForecastUnits
    per (StyleCodeDesc, MonthStart, HorizonMonths).
    """
    fc     = stylecode_forecasts_df.copy()
    shares = color_shares_df[[SCODE_COL, SC_COL, "share"]].copy()

    # Rename Key → StyleCodeDesc for the join
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SCODE_COL})

    # Join color shares onto forecasts
    allocated = fc.merge(shares, on=SCODE_COL, how="left")

    # Handle unmatched StyleCodes (e.g. cold-start codes with no dim_product)
    n_unmatched = allocated["share"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "[v7] allocate_to_stylecolor: %d rows have no color-share match "
            "(StyleCode not in color_shares_df — ForecastUnits set to 0).",
            n_unmatched,
        )
    allocated["share"] = allocated["share"].fillna(0.0)

    # Multiply ForecastUnits / Lower / Upper by the color share
    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (allocated[col] * allocated["share"]).fillna(0.0).clip(lower=0).round(4)

    # The output key is now StyleColorDesc — rename to "Key" for schema compatibility
    allocated = allocated.rename(columns={SC_COL: "Key"})
    allocated["Level"] = "StyleColor"

    # Drop the intermediate StyleCodeDesc and share columns; callers can add back if needed
    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        SCODE_COL,   # keep for traceability
    ] if c in allocated.columns]
    allocated = allocated[keep_cols].copy()

    # Drop rows where share==0 (StyleColors with ZeroForecast AND zero share)
    # BUT keep them if ForecastUnits==0 and the row is needed for downstream sum checks
    # Strategy: keep all rows — zero forecasts are valid outputs
    allocated = (
        allocated
        .sort_values(["Key", "MonthStart", "HorizonMonths"])
        .reset_index(drop=True)
    )

    logger.info(
        "[v7] allocate_to_stylecolor: %d StyleCode rows → %d StyleColor rows",
        len(fc), len(allocated),
    )
    return allocated


# ---------------------------------------------------------------------------
# Step 4 — Validation: StyleCode → StyleColor allocation integrity
# ---------------------------------------------------------------------------

def validate_stylecode_allocation(
    stylecode_forecasts_df: pd.DataFrame,
    stylecolor_forecasts_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """
    Check that the StyleCode → StyleColor allocation is consistent.

    Checks:
        1. Sum consistency: sum(StyleColor ForecastUnits) == StyleCode ForecastUnits
        2. Share sanity: ratio within tolerance of 1.0 for non-zero StyleCodes
        3. No negatives: all ForecastUnits ≥ 0

    Returns
    -------
    dict with keys:
        sum_check_passed, max_abs_diff, share_check_passed,
        no_negatives, n_stylecodes, n_stylecolors, n_groups_checked
    """
    sc_fc   = stylecode_forecasts_df.copy()
    scol_fc = stylecolor_forecasts_df.copy()

    # Normalise column names
    if "Key" in sc_fc.columns:
        sc_fc = sc_fc.rename(columns={"Key": SCODE_COL})
    if "Key" in scol_fc.columns:
        scol_fc = scol_fc.rename(columns={"Key": SC_COL})

    scode_totals = (
        sc_fc.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
        .sum().reset_index().rename(columns={"ForecastUnits": "SC_total"})
    )
    scol_totals = (
        scol_fc.groupby([SCODE_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
        .sum().reset_index().rename(columns={"ForecastUnits": "SCOL_total"})
    ) if SCODE_COL in scol_fc.columns else pd.DataFrame()

    if scol_totals.empty:
        return {
            "sum_check_passed":   False,
            "max_abs_diff":       None,
            "share_check_passed": False,
            "no_negatives":       bool((stylecolor_forecasts_df["ForecastUnits"] >= 0).all()),
            "n_stylecodes":       sc_fc[SCODE_COL].nunique(),
            "n_stylecolors":      0,
            "n_groups_checked":   0,
        }

    merged = scode_totals.merge(scol_totals, on=[SCODE_COL, "MonthStart", "HorizonMonths"], how="left")
    merged["abs_diff"] = (merged["SC_total"] - merged["SCOL_total"].fillna(0)).abs()
    max_diff  = float(merged["abs_diff"].max())
    sum_ok    = max_diff < tolerance
    n_groups  = len(merged)

    ratio_df  = merged[merged["SC_total"] > 0].copy()
    if not ratio_df.empty:
        ratio_df["ratio"] = ratio_df["SCOL_total"] / ratio_df["SC_total"]
        share_ok = bool(((ratio_df["ratio"] - 1.0).abs() < tolerance * 10).all())
    else:
        share_ok = True

    no_neg = bool((stylecolor_forecasts_df["ForecastUnits"] >= 0).all())

    result = {
        "sum_check_passed":   sum_ok,
        "max_abs_diff":       round(max_diff, 4),
        "share_check_passed": share_ok,
        "no_negatives":       no_neg,
        "n_stylecodes":       sc_fc[SCODE_COL].nunique(),
        "n_stylecolors":      scol_fc[SC_COL].nunique() if SC_COL in scol_fc.columns else 0,
        "n_groups_checked":   n_groups,
    }
    if sum_ok:
        logger.info(
            "[v7] validate_stylecode_allocation: ✓ sum check passed (max_diff=%.4f, %d groups)",
            max_diff, n_groups,
        )
    else:
        logger.warning(
            "[v7] validate_stylecode_allocation: ✗ sum check FAILED (max_diff=%.4f)",
            max_diff,
        )
    return result


# ---------------------------------------------------------------------------
# Utility — standalone SKU identification (v7 level)
# ---------------------------------------------------------------------------

def get_v7_standalone_skus(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
) -> list[str]:
    """
    Return SKUs that cannot participate in the v7 StyleCode hierarchy.

    A SKU is STANDALONE at the v7 level if:
        (a) No row in dim_product, OR
        (b) dim_product row has null StyleCodeDesc.

    These SKUs are forecasted by the existing v5.2/v6 SKU-level pipeline.
    """
    gold_skus = set(gold_df["SKU"].astype(str).str.strip().unique())
    dp = dim_product_df.copy()
    dp["SKU"] = dp["SKU"].astype(str).str.strip()

    unmapped       = gold_skus - set(dp["SKU"].unique())
    null_scode_skus = set(dp.loc[dp[SCODE_COL].isna(), "SKU"].unique())

    standalone = sorted(unmapped | (gold_skus & null_scode_skus))
    logger.info(
        "[v7] STANDALONE SKUs: %d total (%d unmapped, %d null StyleCodeDesc)",
        len(standalone), len(unmapped), len(gold_skus & null_scode_skus),
    )
    return standalone


# ---------------------------------------------------------------------------
# Utility — build v7 coverage report
# ---------------------------------------------------------------------------

def build_v7_coverage_report(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
) -> dict:
    """
    Return a summary dict of hierarchy coverage for auditing.

    Keys:
        n_stylecodes_in_dim        : StyleCodes present in dim_product
        n_stylecodes_with_demand   : StyleCodes that have ≥ 1 training row
        n_stylecolors_in_dim       : StyleColors present in dim_product
        n_stylecolors_with_demand  : StyleColors with demand ≥ 1 training row
        n_skus_in_gold             : unique SKUs in gold demand
        n_skus_mappable            : SKUs that map into the hierarchy
        n_skus_standalone          : SKUs with no hierarchy mapping
        n_cold_start_stylecodes    : StyleCodes in dim_product with 0 training demand
        n_cold_start_stylecolors   : StyleColors in dim_product with 0 training demand
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold["SKU"]    = gold["SKU"].astype(str).str.strip()

    if train_end is not None:
        gold = gold[gold[DATE_COL] <= pd.Timestamp(train_end)]

    dp = dim_product_df.copy()
    dp["SKU"] = dp["SKU"].astype(str).str.strip()

    n_sc_dim    = dp[SCODE_COL].dropna().nunique()
    n_scol_dim  = dp[SC_COL].dropna().nunique() if SC_COL in dp.columns else 0
    n_skus_gold = gold["SKU"].nunique()

    # Mappable = in gold AND in dim_product AND both SCODE and SC not null
    dp_valid   = dp[dp[SCODE_COL].notna()].drop_duplicates("SKU")
    gold_mapped = gold.merge(dp_valid[["SKU", SCODE_COL, SC_COL]], on="SKU", how="inner")

    n_mappable       = gold_mapped["SKU"].nunique()
    n_standalone     = n_skus_gold - n_mappable

    sc_with_demand   = gold_mapped[SCODE_COL].nunique()
    scol_with_demand = gold_mapped[SC_COL].nunique() if SC_COL in gold_mapped.columns else 0

    all_scodes   = set(dp[SCODE_COL].dropna().unique())
    active_codes = set(gold_mapped[SCODE_COL].unique())
    cold_codes   = len(all_scodes - active_codes)

    all_scols    = set(dp[SC_COL].dropna().unique()) if SC_COL in dp.columns else set()
    active_scols = set(gold_mapped[SC_COL].unique()) if SC_COL in gold_mapped.columns else set()
    cold_scols   = len(all_scols - active_scols)

    return {
        "n_stylecodes_in_dim":       n_sc_dim,
        "n_stylecodes_with_demand":  sc_with_demand,
        "n_stylecolors_in_dim":      n_scol_dim,
        "n_stylecolors_with_demand": scol_with_demand,
        "n_skus_in_gold":            n_skus_gold,
        "n_skus_mappable":           n_mappable,
        "n_skus_standalone":         n_standalone,
        "n_cold_start_stylecodes":   cold_codes,
        "n_cold_start_stylecolors":  cold_scols,
    }
