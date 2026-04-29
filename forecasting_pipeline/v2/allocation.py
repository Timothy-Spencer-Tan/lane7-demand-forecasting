"""
lane7_forecast.allocation
==========================
Hierarchical disaggregation layer for v6.

Architecture
------------
The v6 pipeline forecasts at the **StyleColorDesc** level, then disaggregates
down to **SKU** (StyleColorDesc + SizeDesc) using historical size shares.

Hierarchy
---------
    Level 1: StyleColorDesc   ← model is trained and forecast generated here
    Level 2: SizeDesc         ← allocation proportions derived here
    Level 3: SKU              ← final deliverable

STANDALONE SKUs
---------------
621 gold SKUs have no dim_product row (StyleColorDesc is unknown) and 17
dim_product rows carry a null StyleColorDesc.  These cannot participate in the
StyleColor hierarchy.  They are forecasted by the existing v5.2 SKU-level
pipeline unchanged and passed through this module as-is under the label
``StyleColorDesc = "STANDALONE"``.

Allocation algorithm
--------------------
For each (StyleColorDesc, forecast_month):

    1. Compute size_share[SizeDesc] from the look-back window.

       Look-back priority (applied in order until non-zero shares are found):
         a) Configurable window (default 12 months) of training history
         b) Full training history
         c) Equal distribution across all dim_product sizes for that StyleColor

    2. Multiply: SKU_forecast = StyleColor_forecast × size_share[SizeDesc]

    3. Sum-consistency check (internal):
       sum(SKU_forecast) == StyleColor_forecast   (up to floating-point tolerance)

Public API
----------
    build_stylecolor_demand(gold_df, dim_product_df)
        -> stylecolor_monthly_demand DataFrame

    compute_size_shares(gold_df, dim_product_df,
                        lookback_months=12, min_lookback_months=6,
                        train_end=None)
        -> size_shares DataFrame  (StyleColorDesc, SizeDesc, share)

    allocate_to_sku(stylecolor_forecasts_df, size_shares_df, dim_product_df)
        -> sku_forecasts_df  (gold_fact_forecasts schema)

    get_standalone_skus(gold_df, dim_product_df)
        -> list[str]   SKUs that must bypass the hierarchy

    validate_allocation(stylecolor_forecasts_df, sku_forecasts_df)
        -> dict   with keys 'sum_check_passed', 'max_abs_diff', 'share_check_passed'
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (mirrors data_prep conventions)
# ---------------------------------------------------------------------------
SC_COL      = "StyleColorDesc"   # Level-1 key
SIZE_COL    = "SizeDesc"         # Level-2 key
SKU_COL     = "SKU"              # Level-3 key
DATE_COL    = "MonthStart"
TARGET_COL  = "UnitsSold"
STANDALONE  = "STANDALONE"       # sentinel for unmappable SKUs


# ---------------------------------------------------------------------------
# Step 1 — Build StyleColor-level demand table
# ---------------------------------------------------------------------------

def build_stylecolor_demand(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate SKU-level monthly demand to StyleColorDesc × Month.

    Parameters
    ----------
    gold_df        : gold_fact_monthly_demand_v2 (MonthStart, SKU, UnitsSold, Revenue)
    dim_product_df : dim_product (SKU, StyleColorDesc, SizeDesc, ...)

    Returns
    -------
    (stylecolor_demand, standalone_skus)

    stylecolor_demand : DataFrame with columns
        StyleColorDesc, MonthStart, UnitsSold, Revenue
        One row per (StyleColorDesc, MonthStart). Null StyleColorDesc rows
        are labelled "STANDALONE".

    standalone_skus : list[str]
        SKUs that do not map into the hierarchy (no dim_product row, or
        dim_product row has null StyleColorDesc).  These must be handled
        by the v5.2 SKU-level pipeline and are NOT included in
        stylecolor_demand.
    """
    gold = gold_df.copy()
    gold[DATE_COL]   = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]    = gold[SKU_COL].astype(str).str.strip()

    # Attach StyleColorDesc from dim_product
    dp = dim_product_df[[SKU_COL, SC_COL, SIZE_COL]].copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    # dim_product may have duplicate (StyleColorDesc, SizeDesc) when
    # StyleColorDesc is null; de-dup on SKU (unique in dim_product)
    dp = dp.drop_duplicates(subset=[SKU_COL], keep="first")

    merged = gold.merge(dp[[SKU_COL, SC_COL]], on=SKU_COL, how="left")

    # Identify STANDALONE SKUs: no dim_product row OR null StyleColorDesc
    standalone_mask = merged[SC_COL].isna()
    standalone_skus = merged.loc[standalone_mask, SKU_COL].unique().tolist()

    n_standalone_rows = standalone_mask.sum()
    n_standalone_units = merged.loc[standalone_mask, TARGET_COL].sum()
    logger.info(
        "STANDALONE SKUs: %d SKUs, %d rows, %s units (%.2f%% of total)",
        len(standalone_skus),
        n_standalone_rows,
        f"{n_standalone_units:,.0f}",
        n_standalone_units / merged[TARGET_COL].sum() * 100,
    )

    # Aggregate non-standalone rows to StyleColorDesc × Month
    hierarchical = merged[~standalone_mask].copy()
    agg_cols = {TARGET_COL: "sum"}
    if "Revenue" in hierarchical.columns:
        agg_cols["Revenue"] = "sum"

    sc_demand = (
        hierarchical
        .groupby([SC_COL, DATE_COL], as_index=False)
        .agg(agg_cols)
        .sort_values([SC_COL, DATE_COL])
        .reset_index(drop=True)
    )

    logger.info(
        "StyleColor demand table: %d rows, %d unique StyleColors, date %s → %s",
        len(sc_demand),
        sc_demand[SC_COL].nunique(),
        sc_demand[DATE_COL].min().strftime("%Y-%m"),
        sc_demand[DATE_COL].max().strftime("%Y-%m"),
    )
    return sc_demand, standalone_skus


# ---------------------------------------------------------------------------
# Step 2 — Compute size shares
# ---------------------------------------------------------------------------

def compute_size_shares(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    lookback_months: int = 12,
    min_lookback_months: int = 6,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute the historical size-share distribution for every StyleColorDesc.

    Size shares are the proportional allocation weights used to disaggregate
    a StyleColor-level forecast back down to individual SKUs.

    Algorithm (applied in priority order until non-zero shares are found):
        (a) Primary window: last ``lookback_months`` months ≤ train_end
        (b) Fallback A:     full history ≤ train_end
        (c) Fallback B:     equal distribution across all dim_product sizes

    Parameters
    ----------
    gold_df          : gold_fact_monthly_demand_v2 with SKU, MonthStart, UnitsSold
    dim_product_df   : dim_product with SKU, StyleColorDesc, SizeDesc
    lookback_months  : primary look-back window length (default 12)
    min_lookback_months : minimum months of data required to use primary window;
                          if fewer months are present, falls back to full history.
    train_end        : cut-off date (inclusive).  Only data ≤ train_end is used.
                       Defaults to max MonthStart in gold_df.

    Returns
    -------
    pd.DataFrame with columns:
        StyleColorDesc, SizeDesc, SKU,
        units_in_window, share, fallback_used

    One row per (StyleColorDesc, SizeDesc) that exists in dim_product and has
    a non-null StyleColorDesc.

    ``share`` values sum to 1.0 per StyleColorDesc (up to floating-point tolerance).
    ``fallback_used`` ∈ {"primary", "full_history", "equal"}
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()

    if train_end is None:
        train_end = gold[DATE_COL].max()
    else:
        train_end = pd.Timestamp(train_end)

    # Restrict to training period (no HOLDOUT/FORECAST leakage into share calc)
    gold = gold[gold[DATE_COL] <= train_end].copy()

    # Build primary window (last lookback_months ≤ train_end)
    primary_start = train_end - pd.DateOffset(months=lookback_months - 1)
    primary_start = pd.Timestamp(primary_start.year, primary_start.month, 1)

    dp = dim_product_df[[SKU_COL, SC_COL, SIZE_COL]].copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    # Drop rows with null StyleColorDesc (STANDALONE bucket)
    dp = dp[dp[SC_COL].notna()].drop_duplicates(subset=[SKU_COL], keep="first")

    # Merge gold with dim to get (StyleColorDesc, SizeDesc) per row
    gold_mapped = gold.merge(dp, on=SKU_COL, how="inner")  # inner = only mappable SKUs

    rows = []
    sc_groups = dp.groupby(SC_COL, sort=False)

    for sc, sc_dp in sc_groups:
        # All (SizeDesc, SKU) pairs that belong to this StyleColor
        sc_sizes = sc_dp[[SIZE_COL, SKU_COL]].copy()

        # Subset of gold rows for this StyleColor
        sc_gold = gold_mapped[gold_mapped[SC_COL] == sc]

        # ── (a) Primary window ──────────────────────────────────────────────
        primary_gold = sc_gold[sc_gold[DATE_COL] >= primary_start]
        n_primary_months = primary_gold[DATE_COL].nunique()

        if n_primary_months >= min_lookback_months and primary_gold[TARGET_COL].sum() > 0:
            window_gold = primary_gold
            fallback    = "primary"
        # ── (b) Fallback A: full history ────────────────────────────────────
        elif sc_gold[TARGET_COL].sum() > 0:
            window_gold = sc_gold
            fallback    = "full_history"
            logger.debug(
                "Size shares for '%s': primary window had %d months or 0 units "
                "— using full history (%d months)",
                sc, n_primary_months, sc_gold[DATE_COL].nunique(),
            )
        # ── (c) Fallback B: equal distribution ─────────────────────────────
        else:
            window_gold = None
            fallback    = "equal"
            logger.debug(
                "Size shares for '%s': no historical sales at all — using equal distribution",
                sc,
            )

        if fallback != "equal" and window_gold is not None:
            # Guard: null SizeDesc rows are silently dropped by groupby.
            # Temporarily replace NaN with a sentinel so they are counted.
            _wg = window_gold.copy()
            _wg[SIZE_COL] = _wg[SIZE_COL].fillna("__ALL__")
            size_units = (
                _wg.groupby(SIZE_COL, as_index=False)[TARGET_COL]
                .sum()
                .rename(columns={TARGET_COL: "units_in_window"})
            )
            # Restore sentinel → NaN to match the dim_product SizeDesc value
            size_units[SIZE_COL] = size_units[SIZE_COL].replace("__ALL__", np.nan)
            total_units = size_units["units_in_window"].sum()
            size_units["share"] = size_units["units_in_window"] / total_units if total_units > 0 else 0.0
            # Merge onto the full size list from dim_product (some sizes may
            # have 0 units in the window — they get share=0 naturally).
            # Use a nan-safe merge key for the join.
            _sc_s   = sc_sizes.copy()
            _sc_s["_size_key"]   = _sc_s[SIZE_COL].fillna("__NAN__")
            _su     = size_units.copy()
            _su["_size_key"]     = _su[SIZE_COL].fillna("__NAN__")
            sc_shares = _sc_s.merge(_su[["_size_key","units_in_window","share"]], on="_size_key", how="left")
            sc_shares = sc_shares.drop(columns=["_size_key"])
            sc_shares["units_in_window"] = sc_shares["units_in_window"].fillna(0.0)
            sc_shares["share"]           = sc_shares["share"].fillna(0.0)
        else:
            # Equal distribution across all dim_product sizes
            n_sizes = len(sc_sizes)
            sc_shares = sc_sizes.copy()
            sc_shares["units_in_window"] = 0.0
            sc_shares["share"] = 1.0 / n_sizes if n_sizes > 0 else 0.0

        sc_shares[SC_COL]         = sc
        sc_shares["fallback_used"] = fallback

        rows.append(sc_shares)

    result = pd.concat(rows, ignore_index=True)

    # Final columns
    result = result[[SC_COL, SIZE_COL, SKU_COL, "units_in_window", "share", "fallback_used"]]

    # Audit log
    fallback_counts = result.drop_duplicates(subset=[SC_COL])["fallback_used"].value_counts()
    logger.info(
        "Size shares computed — primary: %d StyleColors, full_history: %d, equal: %d",
        fallback_counts.get("primary", 0),
        fallback_counts.get("full_history", 0),
        fallback_counts.get("equal", 0),
    )

    # Validate shares sum to 1 per StyleColor
    share_sums = result.groupby(SC_COL)["share"].sum()
    off = ((share_sums - 1.0).abs() > 1e-6).sum()
    if off > 0:
        logger.warning(
            "%d StyleColors have size shares that don't sum to 1.0 "
            "(largest deviation: %.6f) — normalising.",
            off, (share_sums - 1.0).abs().max(),
        )
        # Normalise: divide each share by the StyleColor's total share
        totals = result.groupby(SC_COL)["share"].transform("sum")
        result["share"] = np.where(totals > 0, result["share"] / totals, 0.0)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Allocate StyleColor forecasts → SKU forecasts
# ---------------------------------------------------------------------------

def allocate_to_sku(
    stylecolor_forecasts_df: pd.DataFrame,
    size_shares_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    standalone_sku_forecasts_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Disaggregate StyleColor-level forecasts to SKU-level forecasts.

    Parameters
    ----------
    stylecolor_forecasts_df    : output of generate_forecasts() run at SC level.
                                 Must have columns: Key (=StyleColorDesc),
                                 MonthStart, ForecastUnits, Lower, Upper,
                                 ModelName, HorizonMonths, ModelVersion.
    size_shares_df             : output of compute_size_shares().
                                 Columns: StyleColorDesc, SizeDesc, SKU, share.
    dim_product_df             : dim_product table.
    standalone_sku_forecasts_df: (optional) v5.2 SKU-level forecasts for
                                 STANDALONE SKUs (those outside the hierarchy).
                                 If provided, these are concatenated into the
                                 final output unchanged.

    Returns
    -------
    pd.DataFrame matching the gold_fact_forecasts schema:
        RunDate, MonthStart, Level, Key (=SKU), ModelName, HorizonMonths,
        ForecastUnits, Lower, Upper, ModelVersion, StyleColorDesc, SizeDesc

    Sum-consistency guarantee
    -------------------------
    For each (StyleColorDesc, MonthStart, HorizonMonths):
        sum(SKU ForecastUnits) == StyleColor ForecastUnits
        (subject to floating-point rounding; verified by validate_allocation)
    """
    fc = stylecolor_forecasts_df.copy()
    shares = size_shares_df[[SC_COL, SIZE_COL, SKU_COL, "share"]].copy()

    # Rename 'Key' → StyleColorDesc for clarity in the join
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SC_COL})

    # Join shares onto forecasts
    allocated = fc.merge(shares, on=SC_COL, how="left")

    # Multiply point forecast and intervals by size share.
    # NaN share arises when a StyleColorDesc in the forecast has no matching
    # row in size_shares_df (e.g. cold-start StyleColors with no training history).
    # These are treated as zero demand — the StyleColor should have been DEAD
    # in segmentation and produced a ZeroForecast already, so this is defensive.
    n_unmatched = allocated["share"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "allocate_to_sku: %d rows have no size-share match "
            "(StyleColor not in size_shares_df — setting ForecastUnits to 0).",
            n_unmatched,
        )
    allocated["share"] = allocated["share"].fillna(0.0)

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (allocated[col] * allocated["share"]).fillna(0.0)

    # Round to 4 decimal places (matching v5.2 precision)
    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = allocated[col].round(4).clip(lower=0)

    # Rename SKU_COL back to 'Key' to match forecast schema
    allocated = allocated.rename(columns={SKU_COL: "Key"})

    # Rebuild schema columns
    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        SC_COL, SIZE_COL,
    ] if c in allocated.columns]
    allocated = allocated[keep_cols].copy()
    allocated["Level"] = "SKU"

    logger.info(
        "allocate_to_sku: %d StyleColor rows → %d SKU rows",
        len(fc), len(allocated),
    )

    # Append STANDALONE SKU forecasts (pass-through, no allocation needed)
    if standalone_sku_forecasts_df is not None and not standalone_sku_forecasts_df.empty:
        sa = standalone_sku_forecasts_df.copy()
        if "Key" not in sa.columns and SKU_COL in sa.columns:
            sa = sa.rename(columns={SKU_COL: "Key"})
        sa[SC_COL]   = STANDALONE
        sa[SIZE_COL] = "N/A"
        sa["Level"]  = "SKU"
        allocated = pd.concat([allocated, sa], ignore_index=True)
        logger.info(
            "Appended %d STANDALONE SKU forecast rows", len(sa)
        )

    return allocated.sort_values(["Key", "MonthStart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 — Validation
# ---------------------------------------------------------------------------

def validate_allocation(
    stylecolor_forecasts_df: pd.DataFrame,
    sku_forecasts_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """
    Run three mandatory consistency checks on the allocation output.

    Check 1 — Sum consistency (primary integrity check):
        For each (StyleColorDesc, MonthStart, HorizonMonths):
            sum(SKU ForecastUnits) == StyleColor ForecastUnits
        Passes if max absolute difference < tolerance.

    Check 2 — Share sanity:
        Size shares (inferred from sku/sc ratio) should sum to ~1.0 per group.
        This is a secondary check — the primary guarantee is in compute_size_shares.

    Check 3 — No negative forecasts:
        All ForecastUnits values ≥ 0.

    Parameters
    ----------
    stylecolor_forecasts_df : StyleColor-level forecasts (Key = StyleColorDesc)
    sku_forecasts_df        : SKU-level output of allocate_to_sku()
    tolerance               : max allowed absolute unit difference for Check 1

    Returns
    -------
    dict with keys:
        sum_check_passed   (bool)
        max_abs_diff       (float)  — max |SC_total - SKU_total| per group
        share_check_passed (bool)
        no_negatives       (bool)
        n_stylecolors      (int)
        n_skus             (int)
        n_groups_checked   (int)    — (StyleColorDesc, MonthStart, Horizon) groups
    """
    sc_fc = stylecolor_forecasts_df.copy()
    sku_fc = sku_forecasts_df[
        sku_forecasts_df.get(SC_COL, pd.Series(dtype=str)) != STANDALONE
    ].copy() if SC_COL in sku_forecasts_df.columns else sku_forecasts_df.copy()

    if "Key" in sc_fc.columns:
        sc_fc = sc_fc.rename(columns={"Key": SC_COL})

    # ── Check 1: sum consistency ────────────────────────────────────────────
    sc_totals = (
        sc_fc.groupby([SC_COL, "MonthStart", "HorizonMonths"])["ForecastUnits"]
        .sum().reset_index().rename(columns={"ForecastUnits": "SC_total"})
    )
    sku_totals = (
        sku_fc[sku_fc[SC_COL] != STANDALONE].groupby(
            [SC_COL, "MonthStart", "HorizonMonths"]
        )["ForecastUnits"].sum().reset_index()
        .rename(columns={"ForecastUnits": "SKU_total"})
    ) if SC_COL in sku_fc.columns else pd.DataFrame()

    if sku_totals.empty:
        sum_check = False
        max_diff  = np.nan
    else:
        merged_check = sc_totals.merge(sku_totals, on=[SC_COL, "MonthStart", "HorizonMonths"], how="left")
        merged_check["abs_diff"] = (
            merged_check["SC_total"] - merged_check["SKU_total"].fillna(0)
        ).abs()
        max_diff   = float(merged_check["abs_diff"].max())
        sum_check  = max_diff < tolerance
        n_groups   = len(merged_check)

    # ── Check 2: share sanity ───────────────────────────────────────────────
    # Exclude groups where SC_total==0 (DEAD StyleColors with ZeroForecast):
    # their ratio = 0/0 = nan, which would always fail the all() check.
    if SC_COL in sku_fc.columns and not sku_totals.empty:
        ratio_check = merged_check[merged_check["SC_total"] > 0].copy()
        if not ratio_check.empty:
            ratio_check["ratio"] = ratio_check["SKU_total"] / ratio_check["SC_total"]
            share_ok = bool(((ratio_check["ratio"] - 1.0).abs() < tolerance * 10).all())
        else:
            share_ok = True   # All StyleColors are DEAD — no active shares to check
    else:
        share_ok = False

    # ── Check 3: no negatives ───────────────────────────────────────────────
    # Check the full input (including STANDALONE) for negative ForecastUnits.
    no_neg = bool((sku_forecasts_df["ForecastUnits"] >= 0).all())

    result = {
        "sum_check_passed":   sum_check,
        "max_abs_diff":       round(max_diff, 4) if not np.isnan(max_diff) else None,
        "share_check_passed": share_ok,
        "no_negatives":       no_neg,
        "n_stylecolors":      sc_fc[SC_COL].nunique(),
        "n_skus":             sku_fc["Key"].nunique() if "Key" in sku_fc else 0,
        "n_groups_checked":   n_groups if not sku_totals.empty else 0,
    }

    if sum_check:
        logger.info(
            "validate_allocation: ✓ sum check passed (max_abs_diff=%.4f, %d groups)",
            max_diff, n_groups,
        )
    else:
        logger.warning(
            "validate_allocation: ✗ sum check FAILED (max_abs_diff=%.4f, tolerance=%.4f)",
            max_diff, tolerance,
        )
    return result


# ---------------------------------------------------------------------------
# Utility: standalone SKU identification
# ---------------------------------------------------------------------------

def get_standalone_skus(
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
) -> list[str]:
    """
    Return the list of SKUs that cannot participate in the StyleColor hierarchy.

    A SKU is STANDALONE if:
        (a) It has no row in dim_product, OR
        (b) Its dim_product row has a null StyleColorDesc.

    These SKUs are forecasted by the existing v5.2 SKU-level pipeline and
    their forecasts are passed through allocate_to_sku() unchanged.
    """
    gold_skus = set(gold_df[SKU_COL].astype(str).str.strip().unique())
    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()

    # SKUs in gold but not in dim_product
    unmapped = gold_skus - set(dp[SKU_COL].unique())

    # SKUs in dim_product but with null StyleColorDesc
    null_sc_skus = set(dp.loc[dp[SC_COL].isna(), SKU_COL].unique())

    standalone = sorted(unmapped | (gold_skus & null_sc_skus))
    logger.info(
        "STANDALONE SKUs: %d total (%d unmapped from dim_product, "
        "%d with null StyleColorDesc in dim_product)",
        len(standalone), len(unmapped), len(gold_skus & null_sc_skus),
    )
    return standalone
