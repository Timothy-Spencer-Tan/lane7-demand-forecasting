"""
lane7_forecast.allocation_v72
==============================
v7.2 controlled allocation-ablation framework.

Purpose
-------
v7.1 combined recency weighting + smoothing + caps simultaneously, making it
impossible to determine which idea caused its regression vs v7.0. v7.2 tests
each idea in isolation using the same upstream StyleCode forecasts, so the
individual contribution of each technique can be measured cleanly.

Design
------
All four variants share:
  • the exact same StyleCode-level model forecasts (upstream is fixed)
  • the same two-level allocation pipeline (StyleCode→StyleColor→SKU)
  • the same validation checks
  • the same holdout evaluation framework
  • the same scoreable SKU population (apples-to-apples)

Only the share-computation logic varies between variants.

Allocation variants
-------------------
  baseline_v7    : plain v7.0 simple-sum shares (control)
  recency_only   : recency-weighted sums, no smoothing, no caps
  smoothing_only : equal-prior smoothing, no recency weighting, no caps
  caps_only      : share caps only, no recency weighting, no smoothing

Variant config keys
-------------------
Each variant is a dict with boolean flags:
  use_recency   : weight recent months more heavily
  use_smoothing : blend share toward equal prior
  use_caps      : cap maximum share relative to equal split

Numeric parameters (shared, configurable at call time):
  lookback_months      : primary window length
  min_lookback_months  : minimum months for primary window
  w_recent / w_mid / w_old : recency tier weights
  smooth_alpha         : smoothing blend weight (0=none, 0.3=default)
  cap_rel_increase     : relative cap vs equal split (0.5 = 1.5× equal)

Public API
----------
    ALLOCATION_VARIANTS  : dict of the four variant configs

    compute_shares_for_variant(gold_df, entity_col, child_col,
                               dim_map_df, train_end, variant_cfg,
                               **numeric_params)
        -> shares_df  (entity_col, child_col, share, + diagnostic cols)

    run_allocation_variant(variant_name, variant_cfg,
                           scode_forecasts_df,
                           gold_df, dim_product_df, train_end,
                           standalone_fc_df=None,
                           **numeric_params)
        -> dict with keys:
             stylecolor_forecasts, sku_forecasts,
             stylecolor_shares, size_shares,
             stylecolor_diag, sku_diag,
             validation_sc, validation_sku

    run_all_variants(scode_forecasts_df, gold_df, dim_product_df, train_end,
                     standalone_fc_df=None, output_dir=None,
                     actuals_df=None, holdout_months=None,
                     **numeric_params)
        -> dict[variant_name -> result_dict]
           (also writes output CSVs when output_dir is provided)

    build_variant_comparison(variant_results, actuals_df,
                             dim_product_df, holdout_months)
        -> v7_2_variant_comparison DataFrame

    validate_variant_allocation(sc_fc_df, scol_fc_df, sku_fc_df, tolerance)
        -> dict with sum_check_passed, max_abs_diff, no_negatives
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
# Variant configuration registry
# ---------------------------------------------------------------------------

ALLOCATION_VARIANTS: dict[str, dict] = {
    "baseline_v7": {
        "use_recency":   False,
        "use_smoothing": False,
        "use_caps":      False,
        "description":   "Plain v7.0 simple-sum shares (control)",
    },
    "recency_only": {
        "use_recency":   True,
        "use_smoothing": False,
        "use_caps":      False,
        "description":   "Recency-weighted sums only",
    },
    "smoothing_only": {
        "use_recency":   False,
        "use_smoothing": True,
        "use_caps":      False,
        "description":   "Equal-prior smoothing only",
    },
    "caps_only": {
        "use_recency":   False,
        "use_smoothing": False,
        "use_caps":      True,
        "description":   "Share cap only (no recency, no smoothing)",
    },
}

# Default numeric parameters (shared across all variants unless overridden)
_DEFAULT_PARAMS = dict(
    lookback_months      = 12,
    min_lookback_months  = 6,
    w_recent             = 3,
    w_mid                = 2,
    w_old                = 1,
    smooth_alpha         = 0.3,
    cap_rel_increase     = 0.5,
)


# ---------------------------------------------------------------------------
# Internal: recency weights per row
# ---------------------------------------------------------------------------

def _row_weights(gold_window: pd.DataFrame, train_end: pd.Timestamp,
                 w_recent: int, w_mid: int, w_old: int) -> pd.Series:
    months_ago = (
        (train_end.year  - gold_window[DATE_COL].dt.year)  * 12
        + (train_end.month - gold_window[DATE_COL].dt.month)
    )
    weights = np.where(
        months_ago <= 2, float(w_recent),
        np.where(months_ago <= 5, float(w_mid), float(w_old))
    )
    return pd.Series(weights, index=gold_window.index)


# ---------------------------------------------------------------------------
# Core: compute shares for a single (entity, child) level given variant flags
# ---------------------------------------------------------------------------

def compute_shares_for_variant(
    gold_df: pd.DataFrame,
    entity_col: str,
    child_col: str,
    dim_map_df: pd.DataFrame,
    train_end: pd.Timestamp,
    variant_cfg: dict,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float      = 0.3,
    cap_rel_increase: float  = 0.5,
) -> pd.DataFrame:
    """
    Compute allocation shares for one (entity→child) level with variant logic.

    This is the single code path used by all four variants. Boolean flags in
    ``variant_cfg`` enable or disable each improvement independently.

    Parameters
    ----------
    gold_df          : gold demand (MonthStart, SKU, UnitsSold)
    entity_col       : parent column in dim_map_df (e.g. StyleCodeDesc)
    child_col        : child column in dim_map_df  (e.g. StyleColorDesc or SKU)
    dim_map_df       : dimension table (must contain entity_col, child_col, SKU).
                       Already filtered to non-null, de-duped on SKU.
    train_end        : last date for share computation (inclusive)
    variant_cfg      : dict with keys use_recency, use_smoothing, use_caps
    lookback_months  : primary look-back window
    min_lookback_months : minimum months for primary window
    w_recent/mid/old : recency tier weights (only used when use_recency=True)
    smooth_alpha     : blend weight toward equal prior (use_smoothing only)
    cap_rel_increase : relative cap vs equal split (use_caps only)

    Returns
    -------
    pd.DataFrame with columns:
        entity_col, child_col,
        units_in_window,
        RawShare,           # simple proportional sum
        WeightedShare,      # after recency weighting (= RawShare if not used)
        SmoothedShare,      # after smoothing         (= WeightedShare if not used)
        FinalShare,         # after caps + normalise  (used for multiplication)
        use_recency,        # bool flag written for audit
        use_smoothing,      # bool flag written for audit
        use_caps,           # bool flag written for audit
        fallback_used       # primary / full_history / equal
    """
    use_recency   = bool(variant_cfg.get("use_recency",   False))
    use_smoothing = bool(variant_cfg.get("use_smoothing", False))
    use_caps      = bool(variant_cfg.get("use_caps",      False))

    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()
    gold = gold[gold[DATE_COL] <= train_end].copy()

    primary_start = train_end - pd.DateOffset(months=lookback_months - 1)
    primary_start = pd.Timestamp(primary_start.year, primary_start.month, 1)

    # Build (entity, child) map — join key is always SKU
    dm = dim_map_df.copy()
    dm[SKU_COL] = dm[SKU_COL].astype(str).str.strip()

    # entity→child universe from dim_map
    universe = (
        dm[[entity_col, child_col]]
        .drop_duplicates()
        .groupby(entity_col)[child_col]
        .apply(list)
        .to_dict()
    )

    # Attach entity and child to gold
    if child_col == SKU_COL:
        gold_mapped = gold.merge(dm[[SKU_COL, entity_col]], on=SKU_COL, how="inner")
        gold_mapped[child_col] = gold_mapped[SKU_COL]
    else:
        gold_mapped = gold.merge(dm[[SKU_COL, entity_col, child_col]], on=SKU_COL, how="inner")

    rows = []

    for entity, children in universe.items():
        children = sorted(set(children))
        n_ch     = len(children)
        equal_sh = 1.0 / n_ch if n_ch > 0 else 0.0

        ent_gold = gold_mapped[gold_mapped[entity_col] == entity].copy()
        pri_gold = ent_gold[ent_gold[DATE_COL] >= primary_start].copy()
        n_pri    = pri_gold[DATE_COL].nunique()

        if n_pri >= min_lookback_months and pri_gold[TARGET_COL].sum() > 0:
            window_gold = pri_gold
            fallback    = "primary"
        elif ent_gold[TARGET_COL].sum() > 0:
            window_gold = ent_gold
            fallback    = "full_history"
        else:
            window_gold = None
            fallback    = "equal"

        # ── Step 1: Raw share (simple proportional sum) ───────────────────
        if window_gold is not None and len(window_gold) > 0:
            raw_agg = (
                window_gold.groupby(child_col)[TARGET_COL].sum()
                .reset_index().rename(columns={TARGET_COL: "units_in_window"})
            )
            tot_raw = raw_agg["units_in_window"].sum()
            raw_agg["RawShare"] = (
                raw_agg["units_in_window"] / tot_raw if tot_raw > 0 else equal_sh
            )
        else:
            raw_agg = pd.DataFrame({
                child_col:         children,
                "units_in_window": [0.0] * n_ch,
                "RawShare":        [equal_sh] * n_ch,
            })

        # ── Step 2: Weighted share (recency) ─────────────────────────────
        if use_recency and window_gold is not None and len(window_gold) > 0:
            wg = window_gold.copy()
            wg["_w"] = _row_weights(wg, train_end, w_recent, w_mid, w_old)
            wg["_wu"] = wg[TARGET_COL] * wg["_w"]
            wt_agg = (
                wg.groupby(child_col)["_wu"].sum()
                .reset_index().rename(columns={"_wu": "_weighted_units"})
            )
            tot_wt = wt_agg["_weighted_units"].sum()
            wt_agg["WeightedShare"] = (
                wt_agg["_weighted_units"] / tot_wt if tot_wt > 0 else equal_sh
            )
        else:
            # No recency: WeightedShare == RawShare
            wt_agg = raw_agg[[child_col]].copy()
            wt_agg["WeightedShare"] = raw_agg["RawShare"].values

        # Merge raw + weighted onto full child universe
        all_children = pd.DataFrame({child_col: children})
        sh = (
            all_children
            .merge(raw_agg[[child_col, "units_in_window", "RawShare"]], on=child_col, how="left")
            .merge(wt_agg[[child_col, "WeightedShare"]], on=child_col, how="left")
        )
        sh["units_in_window"] = sh["units_in_window"].fillna(0.0)
        sh["RawShare"]        = sh["RawShare"].fillna(equal_sh)
        sh["WeightedShare"]   = sh["WeightedShare"].fillna(equal_sh)

        # ── Step 3: Smoothing ─────────────────────────────────────────────
        if use_smoothing:
            sh["SmoothedShare"] = (
                (1.0 - smooth_alpha) * sh["WeightedShare"]
                + smooth_alpha * equal_sh
            )
        else:
            sh["SmoothedShare"] = sh["WeightedShare"]

        # ── Step 4: Caps ──────────────────────────────────────────────────
        if use_caps and equal_sh > 0:
            cap_upper = equal_sh * (1.0 + cap_rel_increase)
            sh["SmoothedShare"] = sh["SmoothedShare"].clip(upper=cap_upper)

        # ── Step 5: Normalise to sum=1 ────────────────────────────────────
        total = sh["SmoothedShare"].sum()
        sh["FinalShare"] = sh["SmoothedShare"] / total if total > 0 else equal_sh

        sh[entity_col]       = entity
        sh["fallback_used"]  = fallback
        sh["use_recency"]    = use_recency
        sh["use_smoothing"]  = use_smoothing
        sh["use_caps"]       = use_caps

        rows.append(sh)

    if not rows:
        return pd.DataFrame(columns=[
            entity_col, child_col,
            "units_in_window", "RawShare", "WeightedShare",
            "SmoothedShare", "FinalShare",
            "use_recency", "use_smoothing", "use_caps", "fallback_used",
        ])

    result = pd.concat(rows, ignore_index=True)

    # Final audit normalisation
    totals = result.groupby(entity_col)["FinalShare"].transform("sum")
    off    = ((result.groupby(entity_col)["FinalShare"].sum() - 1.0).abs() > 1e-5).sum()
    if off > 0:
        logger.warning(
            "[v7.2] %d entities have FinalShare not summing to 1.0 — re-normalising.", off
        )
        result["FinalShare"] = np.where(totals > 0, result["FinalShare"] / totals, equal_sh)

    col_order = [
        entity_col, child_col,
        "units_in_window", "RawShare", "WeightedShare", "SmoothedShare", "FinalShare",
        "use_recency", "use_smoothing", "use_caps", "fallback_used",
    ]
    return result[[c for c in col_order if c in result.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Allocation step helpers
# ---------------------------------------------------------------------------

def _apply_shares_to_forecasts(
    fc_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    entity_col: str,
    child_col: str,
) -> pd.DataFrame:
    """
    Multiply ForecastUnits by FinalShare and return child-level forecasts.

    ``fc_df`` has Key = entity.  Returns DataFrame with Key = child.
    Preserves a ``share`` column (= FinalShare) for diagnostic tracing.
    """
    fc = fc_df.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": entity_col})

    sh = shares_df[[entity_col, child_col, "FinalShare"]].copy()
    sh = sh.rename(columns={"FinalShare": "share"})

    allocated = fc.merge(sh, on=entity_col, how="left")
    n_unmatched = allocated["share"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "[v7.2] _apply_shares: %d rows with no share match → ForecastUnits=0",
            n_unmatched,
        )
    allocated["share"] = allocated["share"].fillna(0.0)

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (allocated[col] * allocated["share"]).fillna(0.0).clip(lower=0).round(4)

    allocated = allocated.rename(columns={child_col: "Key"})
    if entity_col != "StyleColorDesc":
        # Keep the entity col for traceability (StyleCodeDesc in scol results)
        pass  # entity_col column already present

    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        entity_col,   # traceability
        "share",      # diagnostic
    ] if c in allocated.columns]
    return allocated[keep_cols].reset_index(drop=True)


def _sku_allocate(
    scol_fc: pd.DataFrame,
    size_shares: pd.DataFrame,
    standalone_fc: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Allocate StyleColor forecasts to SKU using size_shares (FinalShare col).

    Appends STANDALONE SKU pass-through forecasts if provided.
    """

    fc = scol_fc.copy()

    # v7.2 FIX:
    # scol_fc already contains a diagnostic 'share' column from the
    # StyleCode → StyleColor allocation stage.
    # Drop it before merging StyleColor → SKU shares, otherwise pandas creates
    # share_x / share_y and allocated["share"] fails.
    fc = fc.drop(columns=["share"], errors="ignore")

    if "Key" in fc.columns:
       fc = fc.rename(columns={"Key": SC_COL})

    sh = size_shares[[SC_COL, SIZE_COL, SKU_COL, "FinalShare"]].copy()
    sh = sh.rename(columns={"FinalShare": "share"})

    allocated = fc.merge(sh, on=SC_COL, how="left")
    allocated["share"] = allocated["share"].fillna(0.0)

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (allocated[col] * allocated["share"]).fillna(0.0).clip(lower=0).round(4)

    allocated = allocated.rename(columns={SKU_COL: "Key"})
    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        SC_COL, SIZE_COL,
        "share",
    ] if c in allocated.columns]
    allocated = allocated[keep_cols].copy()
    allocated["Level"] = "SKU"

    if standalone_fc is not None and not standalone_fc.empty:
        sa = standalone_fc.copy()
        if "Key" not in sa.columns and SKU_COL in sa.columns:
            sa = sa.rename(columns={SKU_COL: "Key"})
        sa[SC_COL]   = STANDALONE
        sa[SIZE_COL] = "N/A"
        sa["Level"]  = "SKU"
        if "share" not in sa.columns:
            sa["share"] = np.nan
        allocated = pd.concat([allocated, sa], ignore_index=True)

    return allocated.sort_values(["Key", "MonthStart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_variant_allocation(
    sc_fc_df: pd.DataFrame,
    scol_fc_df: pd.DataFrame,
    sku_fc_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """
    Run allocation integrity checks for one variant.

    Returns dict with:
        sc_to_scol_sum_ok  : bool
        sc_to_scol_max_diff: float
        scol_to_sku_sum_ok : bool
        scol_to_sku_max_diff: float
        no_negatives       : bool
    """
    def _check_level(parent_fc, child_fc, parent_key, child_key_on_child):
        pf = parent_fc.copy()
        cf = child_fc.copy()
        if "Key" in pf.columns:
            pf = pf.rename(columns={"Key": parent_key})
        if parent_key not in cf.columns:
            return True, 0.0

        p_totals = (
            pf.groupby([parent_key, "MonthStart", "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "P_total"})
        )
        c_totals = (
            cf[cf.get(parent_key, pd.Series(dtype=str)) != STANDALONE]
            .groupby([parent_key, "MonthStart", "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "C_total"})
        ) if parent_key in cf.columns else pd.DataFrame()

        if c_totals.empty:
            return False, np.nan

        merged = p_totals.merge(c_totals, on=[parent_key, "MonthStart", "HorizonMonths"], how="left")
        merged["abs_diff"] = (merged["P_total"] - merged["C_total"].fillna(0)).abs()
        max_diff = float(merged["abs_diff"].max())
        return max_diff < tolerance, max_diff

    sc_ok, sc_diff = _check_level(sc_fc_df, scol_fc_df, SCODE_COL, SC_COL)
    scol_ok, scol_diff = _check_level(scol_fc_df, sku_fc_df, SC_COL, SKU_COL)
    no_neg = bool((sku_fc_df["ForecastUnits"] >= 0).all())

    return {
        "sc_to_scol_sum_ok":    sc_ok,
        "sc_to_scol_max_diff":  round(sc_diff, 6) if not (sc_diff != sc_diff) else None,
        "scol_to_sku_sum_ok":   scol_ok,
        "scol_to_sku_max_diff": round(scol_diff, 6) if not (scol_diff != scol_diff) else None,
        "no_negatives":         no_neg,
    }


# ---------------------------------------------------------------------------
# Run one allocation variant
# ---------------------------------------------------------------------------

def run_allocation_variant(
    variant_name: str,
    variant_cfg: dict,
    scode_forecasts_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    train_end: pd.Timestamp,
    standalone_fc_df: pd.DataFrame | None = None,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float     = 0.3,
    cap_rel_increase: float = 0.5,
) -> dict:
    """
    Run the full two-level allocation for one variant.

    Uses the SAME scode_forecasts_df for all variants — only the share
    computation differs.

    Parameters
    ----------
    variant_name       : name string (for logging and output labelling)
    variant_cfg        : dict from ALLOCATION_VARIANTS
    scode_forecasts_df : upstream StyleCode forecasts (Key = StyleCodeDesc)
    gold_df            : gold demand (for share computation)
    dim_product_df     : dim_product table
    train_end          : share computation cut-off
    standalone_fc_df   : STANDALONE SKU pass-through forecasts (may be None)
    numeric params     : shared across all variants

    Returns
    -------
    dict with keys:
        stylecolor_shares    : shares DataFrame (StyleCode→StyleColor)
        size_shares          : shares DataFrame (StyleColor→SKU)
        stylecolor_forecasts : allocated StyleColor forecasts
        sku_forecasts        : allocated SKU forecasts
        validation           : dict from validate_variant_allocation()
        variant_name         : str
        variant_cfg          : dict
    """
    logger.info(
        "[v7.2] Running variant '%s': recency=%s  smoothing=%s  caps=%s",
        variant_name,
        variant_cfg.get("use_recency"),
        variant_cfg.get("use_smoothing"),
        variant_cfg.get("use_caps"),
    )

    numeric = dict(
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=smooth_alpha,
        cap_rel_increase=cap_rel_increase,
    )

    # ── Build dimension maps ──────────────────────────────────────────────
    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()

    dp_sc = dp[dp[SCODE_COL].notna() & dp[SC_COL].notna()].drop_duplicates(SKU_COL)
    dp_sz = dp[dp[SC_COL].notna()].drop_duplicates(SKU_COL)

    # ── Level 1: StyleCode → StyleColor shares ────────────────────────────
    scol_shares = compute_shares_for_variant(
        gold_df=gold_df,
        entity_col=SCODE_COL,
        child_col=SC_COL,
        dim_map_df=dp_sc,
        train_end=train_end,
        variant_cfg=variant_cfg,
        **numeric,
    )

    # ── Level 2: StyleColor → SKU (size) shares ───────────────────────────
    # For child_col=SKU, dim_map_df must have SIZE_COL too for output schema
    dp_sz_full = dp[dp[SC_COL].notna()].drop_duplicates(SKU_COL)
    size_shares_raw = compute_shares_for_variant(
        gold_df=gold_df,
        entity_col=SC_COL,
        child_col=SKU_COL,
        dim_map_df=dp_sz_full,
        train_end=train_end,
        variant_cfg=variant_cfg,
        **numeric,
    )
    # Attach SizeDesc for schema compatibility
    sku_size = dp_sz_full[[SKU_COL, SIZE_COL]].drop_duplicates(SKU_COL)
    size_shares = size_shares_raw.merge(sku_size, on=SKU_COL, how="left")

    # ── Allocate StyleCode → StyleColor ───────────────────────────────────
    scol_fc = _apply_shares_to_forecasts(
        fc_df=scode_forecasts_df,
        shares_df=scol_shares,
        entity_col=SCODE_COL,
        child_col=SC_COL,
    )
    scol_fc["Level"] = "StyleColor"

    # ── Allocate StyleColor → SKU ─────────────────────────────────────────
    sku_fc = _sku_allocate(
        scol_fc=scol_fc,
        size_shares=size_shares,
        standalone_fc=standalone_fc_df,
    )

    # ── Validation ────────────────────────────────────────────────────────
    val = validate_variant_allocation(scode_forecasts_df, scol_fc, sku_fc)
    logger.info(
        "[v7.2] Variant '%s' validation: sc→scol=%s (diff=%.4f)  scol→sku=%s (diff=%.4f)  no_neg=%s",
        variant_name,
        val["sc_to_scol_sum_ok"],   val.get("sc_to_scol_max_diff", float("nan")),
        val["scol_to_sku_sum_ok"],  val.get("scol_to_sku_max_diff", float("nan")),
        val["no_negatives"],
    )

    return {
        "variant_name":          variant_name,
        "variant_cfg":           variant_cfg,
        "stylecolor_shares":     scol_shares,
        "size_shares":           size_shares,
        "stylecolor_forecasts":  scol_fc,
        "sku_forecasts":         sku_fc,
        "validation":            val,
    }


# ---------------------------------------------------------------------------
# Run all four variants
# ---------------------------------------------------------------------------

def run_all_variants(
    scode_forecasts_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    train_end: pd.Timestamp,
    standalone_fc_df: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    actuals_df: pd.DataFrame | None = None,
    holdout_months: list[pd.Timestamp] | None = None,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float     = 0.3,
    cap_rel_increase: float = 0.5,
) -> dict[str, dict]:
    """
    Run all four allocation variants and optionally write output CSVs.

    Parameters
    ----------
    scode_forecasts_df : upstream StyleCode forecasts shared by all variants
    gold_df            : gold demand
    dim_product_df     : dim_product table
    train_end          : share computation cut-off
    standalone_fc_df   : STANDALONE pass-through (or None)
    output_dir         : if set, writes CSV files for each variant
    actuals_df         : gold actuals for holdout evaluation (optional)
    holdout_months     : list of Timestamps for holdout scoring (optional)
    numeric params     : forwarded to compute_shares_for_variant

    Returns
    -------
    dict mapping variant_name → result dict from run_allocation_variant()

    Output files written (when output_dir provided):
        v7_2_<variant>_stylecolor_shares.csv
        v7_2_<variant>_size_shares.csv
        v7_2_<variant>_stylecolor_forecasts.csv
        v7_2_<variant>_sku_forecasts.csv
        v7_2_<variant>_allocation_diagnostics.csv
        (holdout CSVs written only when actuals_df is provided)
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    numeric = dict(
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=smooth_alpha,
        cap_rel_increase=cap_rel_increase,
    )

    results: dict[str, dict] = {}

    for vname, vcfg in ALLOCATION_VARIANTS.items():
        result = run_allocation_variant(
            variant_name=vname,
            variant_cfg=vcfg,
            scode_forecasts_df=scode_forecasts_df,
            gold_df=gold_df,
            dim_product_df=dim_product_df,
            train_end=train_end,
            standalone_fc_df=standalone_fc_df,
            **numeric,
        )
        results[vname] = result

        if output_dir is not None:
            _write_variant_outputs(
                vname=vname,
                result=result,
                output_dir=output_dir,
                actuals_df=actuals_df,
                holdout_months=holdout_months,
                dim_product_df=dim_product_df,
                scode_forecasts_df=scode_forecasts_df,
            )

    return results


# ---------------------------------------------------------------------------
# Write per-variant output files
# ---------------------------------------------------------------------------

def _write_variant_outputs(
    vname: str,
    result: dict,
    output_dir: Path,
    actuals_df: pd.DataFrame | None,
    holdout_months: list[pd.Timestamp] | None,
    dim_product_df: pd.DataFrame,
    scode_forecasts_df: pd.DataFrame,
) -> None:
    """Write all CSV files for a single variant."""
    prefix = f"v7_2_{vname}"

    # Allocation diagnostics (merged share info for both levels)
    scol_sh = result["stylecolor_shares"].copy()
    sz_sh   = result["size_shares"].copy()

    diag_sc = scol_sh.assign(AllocationLevel="StyleCode→StyleColor")
    diag_sz = sz_sh.assign(AllocationLevel="StyleColor→SKU")

    # Unified diagnostics
    diag_cols_sc = [c for c in [
        "AllocationLevel", SCODE_COL, SC_COL, "units_in_window",
        "RawShare", "WeightedShare", "SmoothedShare", "FinalShare",
        "use_recency", "use_smoothing", "use_caps", "fallback_used",
    ] if c in diag_sc.columns]
    diag_cols_sz = [c for c in [
        "AllocationLevel", SC_COL, SKU_COL, SIZE_COL, "units_in_window",
        "RawShare", "WeightedShare", "SmoothedShare", "FinalShare",
        "use_recency", "use_smoothing", "use_caps", "fallback_used",
    ] if c in diag_sz.columns]

    diag = pd.concat([diag_sc[diag_cols_sc], diag_sz[diag_cols_sz]], ignore_index=True, sort=False)
    diag.to_csv(output_dir / f"{prefix}_allocation_diagnostics.csv", index=False)
    logger.info("[v7.2] Saved %s_allocation_diagnostics.csv (%d rows)", prefix, len(diag))

    # SKU forecasts
    result["sku_forecasts"].to_csv(output_dir / f"{prefix}_sku_forecasts.csv", index=False)

    # Holdout evaluation (if actuals provided)
    if actuals_df is not None and holdout_months is not None:
        hold_eval, hold_preds = _score_holdout(
            sku_fc=result["sku_forecasts"],
            actuals_df=actuals_df,
            holdout_months=holdout_months,
        )
        hold_preds.to_csv(output_dir / f"{prefix}_holdout_predictions.csv", index=False)
        hold_eval.to_csv(output_dir  / f"{prefix}_holdout_evaluation.csv",  index=False)
        logger.info("[v7.2] Saved holdout files for variant '%s'", vname)

        # Error decomposition
        decomp = _error_decomp(
            actuals_df=actuals_df,
            dim_product_df=dim_product_df,
            scode_fc=scode_forecasts_df,
            scol_fc=result["stylecolor_forecasts"],
            sku_fc=result["sku_forecasts"],
            holdout_months=holdout_months,
        )
        decomp.to_csv(output_dir / f"{prefix}_error_decomposition.csv", index=False)
        logger.info("[v7.2] Saved error_decomposition for variant '%s'", vname)


# ---------------------------------------------------------------------------
# Holdout scoring helper
# ---------------------------------------------------------------------------

def _score_holdout(
    sku_fc: pd.DataFrame,
    actuals_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score SKU forecasts against actuals. Returns (evaluation_df, predictions_df)."""
    fc = sku_fc.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SKU_COL})
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()

    acts = actuals_df.copy()
    acts["MonthStart"] = pd.to_datetime(acts["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]      = acts[SKU_COL].astype(str).str.strip()
    acts = acts[acts["MonthStart"].isin([pd.Timestamp(m) for m in holdout_months])]
    acts = acts[["SKU", "MonthStart", TARGET_COL]].rename(columns={TARGET_COL: "ActualUnits"})

    scored = fc.merge(acts, on=[SKU_COL, "MonthStart"], how="inner")
    scored["PredictedUnits"] = scored["ForecastUnits"].clip(lower=0).fillna(0)
    scored["Error"]          = scored["ActualUnits"] - scored["PredictedUnits"]
    scored["AbsError"]       = scored["Error"].abs()
    scored["AbsPctError"]    = np.where(
        scored["ActualUnits"] > 0,
        scored["AbsError"] / scored["ActualUnits"] * 100,
        np.nan,
    )

    pred_cols = [SKU_COL, "MonthStart", "HorizonMonths", "ModelName", "ModelVersion",
                 "PredictedUnits", "ActualUnits", "Error", "AbsError", "AbsPctError"]
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
# Error decomposition helper
# ---------------------------------------------------------------------------

def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


def _error_decomp(
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    scode_fc: pd.DataFrame,
    scol_fc: pd.DataFrame,
    sku_fc: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Build WMAPE at StyleCode / StyleColor / SKU for each holdout month."""
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

    for m in sorted(acts["MonthStart"].unique()):
        a_m = acts[acts["MonthStart"] == m]

        # SKU level
        fc = sku_fc.copy()
        if "Key" in fc.columns:
            fc = fc.rename(columns={"Key": SKU_COL})
        fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        fc_m = fc[fc["MonthStart"] == m]
        # Use best (lowest H) for each SKU
        if "HorizonMonths" in fc_m.columns:
            fc_m_dedup = fc_m.sort_values("HorizonMonths").drop_duplicates(SKU_COL)
        else:
            fc_m_dedup = fc_m.drop_duplicates(SKU_COL)
        merged = a_m[[SKU_COL, TARGET_COL]].merge(
            fc_m_dedup[[SKU_COL, "ForecastUnits", "HorizonMonths"]],
            on=SKU_COL, how="inner",
        )
        if not merged.empty:
            for h in sorted(merged["HorizonMonths"].unique()):
                h_m = merged[merged["HorizonMonths"] == h]
                rows.append({
                    "Level": "SKU", "HorizonMonths": int(h),
                    "MonthStart": m.strftime("%Y-%m"),
                    "TotalActual": round(h_m[TARGET_COL].sum(), 2),
                    "TotalPredicted": round(h_m["ForecastUnits"].sum(), 2),
                    "WMAPE": round(_wmape(h_m[TARGET_COL], h_m["ForecastUnits"]), 4),
                })

        # StyleColor level
        if SC_COL in acts.columns:
            fc2 = scol_fc.copy()
            if "Key" in fc2.columns:
                fc2 = fc2.rename(columns={"Key": SC_COL})
            fc2["MonthStart"] = pd.to_datetime(fc2["MonthStart"]).dt.to_period("M").dt.to_timestamp()
            fc2_m = fc2[fc2["MonthStart"] == m]
            a_sc = a_m.groupby(SC_COL)[TARGET_COL].sum().reset_index()
            fc_sc = fc2_m.groupby([SC_COL, "HorizonMonths"])["ForecastUnits"].sum().reset_index()
            merged2 = a_sc.merge(fc_sc, on=SC_COL, how="inner")
            for h in sorted(merged2["HorizonMonths"].unique()):
                h_m2 = merged2[merged2["HorizonMonths"] == h]
                rows.append({
                    "Level": "StyleColor", "HorizonMonths": int(h),
                    "MonthStart": m.strftime("%Y-%m"),
                    "TotalActual": round(h_m2[TARGET_COL].sum(), 2),
                    "TotalPredicted": round(h_m2["ForecastUnits"].sum(), 2),
                    "WMAPE": round(_wmape(h_m2[TARGET_COL], h_m2["ForecastUnits"]), 4),
                })

        # StyleCode level
        if SCODE_COL in acts.columns:
            fc3 = scode_fc.copy()
            if "Key" in fc3.columns:
                fc3 = fc3.rename(columns={"Key": SCODE_COL})
            fc3["MonthStart"] = pd.to_datetime(fc3["MonthStart"]).dt.to_period("M").dt.to_timestamp()
            fc3_m = fc3[fc3["MonthStart"] == m]
            a_scd = a_m.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()
            fc_scd = fc3_m.groupby([SCODE_COL, "HorizonMonths"])["ForecastUnits"].sum().reset_index()
            merged3 = a_scd.merge(fc_scd, on=SCODE_COL, how="inner")
            for h in sorted(merged3["HorizonMonths"].unique()):
                h_m3 = merged3[merged3["HorizonMonths"] == h]
                rows.append({
                    "Level": "StyleCode", "HorizonMonths": int(h),
                    "MonthStart": m.strftime("%Y-%m"),
                    "TotalActual": round(h_m3[TARGET_COL].sum(), 2),
                    "TotalPredicted": round(h_m3["ForecastUnits"].sum(), 2),
                    "WMAPE": round(_wmape(h_m3[TARGET_COL], h_m3["ForecastUnits"]), 4),
                })

    if not rows:
        return pd.DataFrame(columns=["Level","HorizonMonths","MonthStart","TotalActual","TotalPredicted","WMAPE"])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart","Level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

def build_variant_comparison(
    variant_results: dict[str, dict],
    actuals_df: pd.DataFrame,
    dim_product_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    scode_forecasts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the v7_2_variant_comparison.csv summary table.

    One row per variant, columns:
        Variant,
        H1_Jan_WMAPE, H3_Jan_WMAPE, H3_Feb_WMAPE,
        StyleCode_H3_Jan_WMAPE, StyleColor_H3_Jan_WMAPE, SKU_H3_Jan_WMAPE,
        StyleCode_H3_Feb_WMAPE, StyleColor_H3_Feb_WMAPE, SKU_H3_Feb_WMAPE,
        sc_to_scol_sum_ok, scol_to_sku_sum_ok, no_negatives,
        description

    Parameters
    ----------
    variant_results    : output of run_all_variants()
    actuals_df         : gold demand with actuals
    dim_product_df     : dim_product
    holdout_months     : list of Timestamps
    scode_forecasts_df : shared upstream StyleCode forecasts

    Returns
    -------
    pd.DataFrame — one row per variant
    """
    jan = pd.Timestamp("2026-01-01")
    feb = pd.Timestamp("2026-02-01")

    comp_rows = []

    for vname, result in variant_results.items():
        # Holdout scoring
        hold_eval, _ = _score_holdout(
            sku_fc=result["sku_forecasts"],
            actuals_df=actuals_df,
            holdout_months=holdout_months,
        )

        def _get_wmape(h, m):
            mstr = m.strftime("%Y-%m")
            row = hold_eval[(hold_eval["HorizonMonths"] == h) & (hold_eval["MonthStart"] == mstr)]
            return round(float(row["WMAPE"].iloc[0]), 4) if not row.empty else np.nan

        # Error decomposition for H=3
        decomp = _error_decomp(
            actuals_df=actuals_df,
            dim_product_df=dim_product_df,
            scode_fc=scode_forecasts_df,
            scol_fc=result["stylecolor_forecasts"],
            sku_fc=result["sku_forecasts"],
            holdout_months=holdout_months,
        )

        def _get_decomp(level, h, m):
            mstr = m.strftime("%Y-%m")
            row = decomp[
                (decomp["Level"] == level) &
                (decomp["HorizonMonths"] == h) &
                (decomp["MonthStart"] == mstr)
            ]
            return round(float(row["WMAPE"].iloc[0]), 4) if not row.empty else np.nan

        val = result["validation"]
        comp_rows.append({
            "Variant":               vname,
            "H1_Jan_WMAPE":          _get_wmape(1, jan),
            "H3_Jan_WMAPE":          _get_wmape(3, jan),
            "H3_Feb_WMAPE":          _get_wmape(3, feb),
            "StyleCode_H3_Jan_WMAPE":  _get_decomp("StyleCode", 3, jan),
            "StyleColor_H3_Jan_WMAPE": _get_decomp("StyleColor", 3, jan),
            "SKU_H3_Jan_WMAPE":        _get_decomp("SKU", 3, jan),
            "StyleCode_H3_Feb_WMAPE":  _get_decomp("StyleCode", 3, feb),
            "StyleColor_H3_Feb_WMAPE": _get_decomp("StyleColor", 3, feb),
            "SKU_H3_Feb_WMAPE":        _get_decomp("SKU", 3, feb),
            "sc_to_scol_sum_ok":       val.get("sc_to_scol_sum_ok"),
            "scol_to_sku_sum_ok":      val.get("scol_to_sku_sum_ok"),
            "no_negatives":            val.get("no_negatives"),
            "description":             ALLOCATION_VARIANTS[vname].get("description", ""),
        })

    return pd.DataFrame(comp_rows).reset_index(drop=True)
