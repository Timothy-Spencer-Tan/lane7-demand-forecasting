"""
lane7_forecast.allocation_strategy_selector
=============================================
v7.3 segmented allocation strategy framework.

Purpose
-------
v7.2 showed that recency_only globally beats baseline_v7, but smoothing hurts
when applied globally.  v7.3 improves on this by classifying each parent
allocation group (StyleCode for Level-1, StyleColor for Level-2) into one of
four behavioural profiles and applying the best strategy for that profile.

Profiles
--------
  HIGH_VOLUME_REGULAR  → recency_only
  STABLE_MIX           → recency_only (or baseline_v7 if mix is very stable)
  SPARSE_INTERMITTENT  → conditional_smoothing
  DOMINANT_CHILD       → recency_only (no cap applied)
  DEAD_OR_NEAR_DEAD    → zero_or_fallback

Strategies
----------
  baseline_v7           : plain proportional sum share
  recency_only          : recency-weighted sum (w_recent=3, w_mid=2, w_old=1)
  conditional_smoothing : light equal-prior smoothing for sparse groups
  zero_or_fallback      : equal-split (no meaningful share data available)

No global caps are applied.  Extreme share concentration is flagged in the
strategy map diagnostics but NOT automatically corrected.

Public API
----------
    ALLOCATION_STRATEGIES : dict of strategy configs (boolean flag dicts)

    classify_parent_allocation_strategy(
        gold_df, dim_map_df, entity_col, child_col, train_end,
        **thresholds) -> strategy_map_df

    compute_segmented_shares(
        gold_df, dim_map_df, entity_col, child_col,
        strategy_map_df, train_end, **numeric_params)
        -> shares_df

    run_segmented_allocation(
        scode_forecasts_df, gold_df, dim_product_df, train_end,
        standalone_fc_df=None, **params)
        -> dict  (stylecolor_forecasts, sku_forecasts, strategy_map,
                  stylecolor_shares, size_shares, validation)

    score_segmented_holdout(sku_fc, actuals_df, holdout_months)
        -> (eval_df, preds_df)

    build_segmented_error_decomp(actuals_df, dim_product_df,
        scode_fc, scol_fc, sku_fc, holdout_months) -> decomp_df

    build_parent_win_loss(strategy_map_df, sku_fc_v73, sku_fc_v72_recency,
        actuals_df, holdout_months) -> win_loss_df
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
# Strategy definitions — boolean flag dicts (same shape as v7.2 variants)
# ---------------------------------------------------------------------------

ALLOCATION_STRATEGIES: dict[str, dict] = {
    "baseline_v7": {
        "use_recency":   False,
        "use_smoothing": False,
        "description":   "Plain proportional sum share (v7.0 control)",
    },
    "recency_only": {
        "use_recency":   True,
        "use_smoothing": False,
        "description":   "Recency-weighted share (w_recent=3, w_mid=2, w_old=1)",
    },
    "conditional_smoothing": {
        "use_recency":   False,
        "use_smoothing": True,
        "smooth_alpha":  0.20,   # light smoothing — less aggressive than v7.1
        "description":   "Light equal-prior smoothing for sparse/intermittent groups",
    },
    "zero_or_fallback": {
        "use_recency":   False,
        "use_smoothing": False,
        "use_equal":     True,
        "description":   "Equal split — no meaningful share data available",
    },
}

# Profile → default strategy mapping
PROFILE_STRATEGY: dict[str, str] = {
    "HIGH_VOLUME_REGULAR": "recency_only",
    "STABLE_MIX":          "recency_only",
    "SPARSE_INTERMITTENT": "conditional_smoothing",
    "DOMINANT_CHILD":      "recency_only",
    "DEAD_OR_NEAR_DEAD":   "zero_or_fallback",
}

# ---------------------------------------------------------------------------
# Classification thresholds (all overridable)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = dict(
    # HIGH_VOLUME_REGULAR: total units in training history >= this
    hv_min_total_units    = 500,
    # HIGH_VOLUME_REGULAR: nonzero demand months >= this
    hv_min_nonzero_months = 12,

    # DEAD_OR_NEAR_DEAD: recent units (last 6 months) <= this
    dead_max_recent_units = 0,
    # DEAD_OR_NEAR_DEAD: total units in full history <= this
    dead_max_total_units  = 10,

    # SPARSE_INTERMITTENT: fraction of child × month cells with demand < this
    sparse_max_fill_rate  = 0.35,
    # SPARSE_INTERMITTENT: nonzero months per child on average < this
    sparse_max_avg_child_months = 6,

    # DOMINANT_CHILD: one child owns >= this share of parent demand
    dominant_child_min_share = 0.70,

    # STABLE_MIX: share volatility (std of child share across months) <= this
    stable_max_share_vol  = 0.10,

    # Recency window for classification
    recent_months_window  = 6,
    primary_window        = 12,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_wmape(actual: pd.Series, predicted: pd.Series) -> float:
    tot = actual.sum()
    return float((actual - predicted).abs().sum() / tot * 100) if tot > 0 else np.nan


def _row_weights(
    gold_window: pd.DataFrame,
    train_end: pd.Timestamp,
    w_recent: int, w_mid: int, w_old: int,
) -> pd.Series:
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
# Step 1 — Classify each parent group into a behavioural profile
# ---------------------------------------------------------------------------

def classify_parent_allocation_strategy(
    gold_df: pd.DataFrame,
    dim_map_df: pd.DataFrame,
    entity_col: str,
    child_col: str,
    train_end: pd.Timestamp,
    **thresholds,
) -> pd.DataFrame:
    """
    Classify each parent group into an allocation profile and assign a strategy.

    Parameters
    ----------
    gold_df      : gold demand (MonthStart, SKU, UnitsSold)
    dim_map_df   : dimension table with entity_col, child_col, SKU
    entity_col   : parent column (e.g. StyleCodeDesc)
    child_col    : child column  (e.g. StyleColorDesc or SKU)
    train_end    : cut-off for share computation
    **thresholds : override DEFAULT_THRESHOLDS keys

    Returns
    -------
    pd.DataFrame with one row per parent group, columns:
        entity_col, assigned_profile, selected_strategy,
        total_units, recent_units, nonzero_months,
        child_count, active_child_count,
        dominant_child_share, share_volatility, sparsity_rate,
        reason_code
    """
    thr = {**DEFAULT_THRESHOLDS, **thresholds}

    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()
    gold = gold[gold[DATE_COL] <= train_end].copy()

    recent_start = train_end - pd.DateOffset(months=thr["recent_months_window"] - 1)
    recent_start = pd.Timestamp(recent_start.year, recent_start.month, 1)

    dm = dim_map_df.copy()
    dm[SKU_COL] = dm[SKU_COL].astype(str).str.strip()

    # Join gold to entity + child via SKU
    if child_col == SKU_COL:
        gold_m = gold.merge(dm[[SKU_COL, entity_col]], on=SKU_COL, how="inner")
        gold_m[child_col] = gold_m[SKU_COL]
    else:
        gold_m = gold.merge(dm[[SKU_COL, entity_col, child_col]], on=SKU_COL, how="inner")

    # Universe of children per entity from dim_map
    universe = (
        dm[[entity_col, child_col]]
        .drop_duplicates()
        .groupby(entity_col)[child_col]
        .apply(list)
        .to_dict()
    )

    rows = []

    for entity, children in universe.items():
        children = sorted(set(children))
        n_children = len(children)

        ent_gold    = gold_m[gold_m[entity_col] == entity]
        recent_gold = ent_gold[ent_gold[DATE_COL] >= recent_start]

        total_units   = float(ent_gold[TARGET_COL].sum())
        recent_units  = float(recent_gold[TARGET_COL].sum())
        nonzero_months = int(ent_gold[ent_gold[TARGET_COL] > 0][DATE_COL].nunique())

        # Child-level stats
        child_units = (
            ent_gold.groupby(child_col)[TARGET_COL].sum().reindex(children, fill_value=0.0)
        )
        active_children = int((child_units > 0).sum())

        dominant_child_share = float(child_units.max() / total_units) if total_units > 0 else 0.0

        # Share volatility: std of per-child monthly share
        monthly_totals = ent_gold.groupby(DATE_COL)[TARGET_COL].sum()
        active_months  = monthly_totals[monthly_totals > 0].index
        if len(active_months) >= 3:
            per_month_shares = []
            for m in active_months:
                m_gold = ent_gold[ent_gold[DATE_COL] == m]
                c_units = m_gold.groupby(child_col)[TARGET_COL].sum()
                tot_m   = c_units.sum()
                if tot_m > 0:
                    per_month_shares.append((c_units / tot_m).reindex(children, fill_value=0.0))
            if per_month_shares:
                share_vol_df = pd.concat(per_month_shares, axis=1)
                share_volatility = float(share_vol_df.std(axis=1).mean())
            else:
                share_volatility = 0.0
        else:
            share_volatility = 0.0

        # Sparsity: fraction of (child × month) cells with demand
        n_months    = ent_gold[DATE_COL].nunique()
        max_cells   = n_children * n_months if n_months > 0 else 1
        filled_cells = int(
            ent_gold[ent_gold[TARGET_COL] > 0]
            .groupby([child_col, DATE_COL])
            .ngroups
        )
        sparsity_rate = 1.0 - (filled_cells / max_cells if max_cells > 0 else 0.0)

        avg_child_months = (
            ent_gold[ent_gold[TARGET_COL] > 0]
            .groupby(child_col)[DATE_COL]
            .nunique()
            .reindex(children, fill_value=0)
            .mean()
        )

        # ── Classification rules (priority order) ─────────────────────────
        if total_units <= thr["dead_max_total_units"] or (
            recent_units <= thr["dead_max_recent_units"] and nonzero_months <= 3
        ):
            profile  = "DEAD_OR_NEAR_DEAD"
            reason   = "no_recent_demand_or_near_dead"

        elif (
            sparsity_rate > (1.0 - thr["sparse_max_fill_rate"])
            or avg_child_months < thr["sparse_max_avg_child_months"]
        ):
            profile  = "SPARSE_INTERMITTENT"
            reason   = f"sparsity={sparsity_rate:.2f}_avg_child_months={avg_child_months:.1f}"

        elif dominant_child_share >= thr["dominant_child_min_share"]:
            profile  = "DOMINANT_CHILD"
            reason   = f"dominant_child_share={dominant_child_share:.2f}"

        elif share_volatility <= thr["stable_max_share_vol"]:
            profile  = "STABLE_MIX"
            reason   = f"share_volatility={share_volatility:.3f}"

        elif (
            total_units >= thr["hv_min_total_units"]
            and nonzero_months >= thr["hv_min_nonzero_months"]
        ):
            profile  = "HIGH_VOLUME_REGULAR"
            reason   = f"total_units={total_units:.0f}_nonzero_months={nonzero_months}"

        else:
            # Default: treat as regular — use recency
            profile  = "HIGH_VOLUME_REGULAR"
            reason   = "default_regular"

        strategy = PROFILE_STRATEGY[profile]

        rows.append({
            entity_col:              entity,
            "allocation_stage":      f"{entity_col}→{child_col}",
            "assigned_profile":      profile,
            "selected_strategy":     strategy,
            "total_units":           round(total_units, 2),
            "recent_units":          round(recent_units, 2),
            "nonzero_months":        nonzero_months,
            "child_count":           n_children,
            "active_child_count":    active_children,
            "dominant_child_share":  round(dominant_child_share, 4),
            "share_volatility":      round(share_volatility, 4),
            "sparsity_rate":         round(sparsity_rate, 4),
            "reason_code":           reason,
        })

    strat_df = pd.DataFrame(rows)

    # Summary log
    profile_counts = strat_df["assigned_profile"].value_counts()
    for p, n in profile_counts.items():
        logger.info(
            "[v7.3] %s→%s classification: %s = %d groups",
            entity_col, child_col, p, n,
        )

    return strat_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — Compute segmented shares (one strategy per parent group)
# ---------------------------------------------------------------------------

def compute_segmented_shares(
    gold_df: pd.DataFrame,
    dim_map_df: pd.DataFrame,
    entity_col: str,
    child_col: str,
    strategy_map_df: pd.DataFrame,
    train_end: pd.Timestamp,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float     = 0.20,
) -> pd.DataFrame:
    """
    Compute allocation shares using the per-parent strategy from strategy_map_df.

    Parameters
    ----------
    gold_df          : gold demand
    dim_map_df       : dimension table with entity_col, child_col, SKU
    entity_col       : parent column
    child_col        : child column
    strategy_map_df  : output of classify_parent_allocation_strategy()
    train_end        : share computation cut-off
    numeric params   : forwarded to individual strategy computation

    Returns
    -------
    pd.DataFrame with columns:
        entity_col, child_col,
        units_in_window, RawShare, FinalShare,
        strategy_applied, use_recency, use_smoothing, fallback_used
    """
    gold = gold_df.copy()
    gold[DATE_COL] = pd.to_datetime(gold[DATE_COL]).dt.to_period("M").dt.to_timestamp()
    gold[SKU_COL]  = gold[SKU_COL].astype(str).str.strip()
    gold = gold[gold[DATE_COL] <= train_end].copy()

    primary_start = train_end - pd.DateOffset(months=lookback_months - 1)
    primary_start = pd.Timestamp(primary_start.year, primary_start.month, 1)

    dm = dim_map_df.copy()
    dm[SKU_COL] = dm[SKU_COL].astype(str).str.strip()

    # Build entity→strategy lookup
    strat_lookup = dict(
        zip(strategy_map_df[entity_col], strategy_map_df["selected_strategy"])
    )

    # Join gold to entity + child via SKU
    if child_col == SKU_COL:
        gold_m = gold.merge(dm[[SKU_COL, entity_col]], on=SKU_COL, how="inner")
        gold_m[child_col] = gold_m[SKU_COL]
    else:
        gold_m = gold.merge(dm[[SKU_COL, entity_col, child_col]], on=SKU_COL, how="inner")

    universe = (
        dm[[entity_col, child_col]]
        .drop_duplicates()
        .groupby(entity_col)[child_col]
        .apply(list)
        .to_dict()
    )

    rows = []

    for entity, children in universe.items():
        children  = sorted(set(children))
        n_ch      = len(children)
        equal_sh  = 1.0 / n_ch if n_ch > 0 else 0.0

        strategy  = strat_lookup.get(entity, "recency_only")
        strat_cfg = ALLOCATION_STRATEGIES[strategy]

        use_recency   = bool(strat_cfg.get("use_recency",   False))
        use_smoothing = bool(strat_cfg.get("use_smoothing", False))
        use_equal     = bool(strat_cfg.get("use_equal",     False))
        alpha         = float(strat_cfg.get("smooth_alpha", smooth_alpha))

        ent_gold = gold_m[gold_m[entity_col] == entity].copy()
        pri_gold = ent_gold[ent_gold[DATE_COL] >= primary_start].copy()
        n_pri    = pri_gold[DATE_COL].nunique()

        # Window selection
        if use_equal or ent_gold[TARGET_COL].sum() == 0:
            window_gold = None
            fallback    = "equal"
        elif n_pri >= min_lookback_months and pri_gold[TARGET_COL].sum() > 0:
            window_gold = pri_gold
            fallback    = "primary"
        elif ent_gold[TARGET_COL].sum() > 0:
            window_gold = ent_gold
            fallback    = "full_history"
        else:
            window_gold = None
            fallback    = "equal"

        # ── Raw share (simple sum) ─────────────────────────────────────────
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

        # ── Recency weighting ─────────────────────────────────────────────
        if use_recency and window_gold is not None and len(window_gold) > 0:
            wg = window_gold.copy()
            wg["_w"]  = _row_weights(wg, train_end, w_recent, w_mid, w_old)
            wg["_wu"] = wg[TARGET_COL] * wg["_w"]
            wt_agg = (
                wg.groupby(child_col)["_wu"].sum()
                .reset_index().rename(columns={"_wu": "_wu"})
            )
            tot_wt = wt_agg["_wu"].sum()
            wt_agg["WeightedShare"] = (
                wt_agg["_wu"] / tot_wt if tot_wt > 0 else equal_sh
            )
            working_share = (
                pd.DataFrame({child_col: children})
                .merge(wt_agg[[child_col, "WeightedShare"]], on=child_col, how="left")
                ["WeightedShare"]
                .fillna(equal_sh)
                .values
            )
        else:
            # Use raw share as starting point
            working_share = (
                pd.DataFrame({child_col: children})
                .merge(raw_agg[[child_col, "RawShare"]], on=child_col, how="left")
                ["RawShare"]
                .fillna(equal_sh)
                .values
            )

        # ── Smoothing (conditional) ───────────────────────────────────────
        if use_smoothing:
            working_share = (1.0 - alpha) * working_share + alpha * equal_sh

        # ── Normalise ─────────────────────────────────────────────────────
        total_ws = working_share.sum()
        final_share = working_share / total_ws if total_ws > 0 else np.full(n_ch, equal_sh)

        # ── Dominant child flag (diagnostic — no correction applied) ──────
        dom_share = float(final_share.max()) if len(final_share) > 0 else 0.0
        dom_flag  = dom_share >= 0.70

        all_children = pd.DataFrame({child_col: children})
        sh = all_children.merge(
            raw_agg[[child_col, "units_in_window", "RawShare"]], on=child_col, how="left"
        )
        sh["units_in_window"] = sh["units_in_window"].fillna(0.0)
        sh["RawShare"]        = sh["RawShare"].fillna(equal_sh)
        sh["FinalShare"]      = final_share
        sh[entity_col]        = entity
        sh["strategy_applied"]  = strategy
        sh["use_recency"]       = use_recency
        sh["use_smoothing"]     = use_smoothing
        sh["fallback_used"]     = fallback
        sh["dominant_child_flagged"] = dom_flag

        rows.append(sh)

    if not rows:
        return pd.DataFrame(columns=[
            entity_col, child_col,
            "units_in_window", "RawShare", "FinalShare",
            "strategy_applied", "use_recency", "use_smoothing",
            "fallback_used", "dominant_child_flagged",
        ])

    result = pd.concat(rows, ignore_index=True)

    # Final normalisation audit
    share_sums = result.groupby(entity_col)["FinalShare"].sum()
    off = ((share_sums - 1.0).abs() > 1e-5).sum()
    if off > 0:
        logger.warning("[v7.3] %d entities have FinalShare not summing to 1.0 — re-normalising.", off)
        totals = result.groupby(entity_col)["FinalShare"].transform("sum")
        equal_fallback = 1.0 / result.groupby(entity_col)[child_col].transform("count")
        result["FinalShare"] = np.where(totals > 0, result["FinalShare"] / totals, equal_fallback)

    col_order = [
        entity_col, child_col,
        "units_in_window", "RawShare", "FinalShare",
        "strategy_applied", "use_recency", "use_smoothing",
        "fallback_used", "dominant_child_flagged",
    ]
    return result[[c for c in col_order if c in result.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3 — Apply shares to forecasts (two-level allocation)
# ---------------------------------------------------------------------------

def _apply_segmented_shares(
    fc_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    entity_col: str,
    child_col: str,
) -> pd.DataFrame:
    """Multiply ForecastUnits by FinalShare, return child-level forecasts."""
    fc = fc_df.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": entity_col})

    sh = shares_df[[entity_col, child_col, "FinalShare", "strategy_applied"]].copy()
    sh = sh.rename(columns={"FinalShare": "share"})

    allocated = fc.merge(sh, on=entity_col, how="left")
    n_unmatched = allocated["share"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "[v7.3] %d rows with no share match → ForecastUnits=0", n_unmatched
        )
    allocated["share"] = allocated["share"].fillna(0.0)

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (allocated[col] * allocated["share"]).fillna(0.0).clip(lower=0).round(4)

    allocated = allocated.rename(columns={child_col: "Key"})
    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        entity_col,
        "share",
        "strategy_applied",
    ] if c in allocated.columns]
    return allocated[keep_cols].reset_index(drop=True)


def _sku_allocate_segmented(
    scol_fc: pd.DataFrame,
    size_shares: pd.DataFrame,
    standalone_fc: pd.DataFrame | None,
) -> pd.DataFrame:
    """Allocate StyleColor forecasts to SKU using segmented size shares."""
    fc = scol_fc.copy()
    if "Key" in fc.columns:
        fc = fc.rename(columns={"Key": SC_COL})

    # Drop StyleCode→StyleColor allocation columns so they don't collide with
    # the SKU-level share columns during the merge (pandas would produce
    # share_x / share_y and strategy_applied_x / strategy_applied_y otherwise).
    fc = fc.drop(columns=["share", "strategy_applied"], errors="ignore")

    # Use prefixed names to avoid any residual collision after the drop.
    sh_cols = [SC_COL, SKU_COL, "FinalShare", "strategy_applied"]
    if SIZE_COL in size_shares.columns:
        sh_cols = [SC_COL, SIZE_COL, SKU_COL, "FinalShare", "strategy_applied"]
    sh = size_shares[sh_cols].copy()
    sh = sh.rename(columns={
        "FinalShare":       "sku_share",
        "strategy_applied": "sku_strategy_applied",
    })

    allocated = fc.merge(sh, on=SC_COL, how="left")
    allocated["sku_share"] = allocated["sku_share"].fillna(0.0)

    for col in ["ForecastUnits", "Lower", "Upper"]:
        if col in allocated.columns:
            allocated[col] = (
                allocated[col] * allocated["sku_share"]
            ).fillna(0.0).clip(lower=0).round(4)

    # Rename to clean final schema before selecting keep_cols
    allocated = allocated.rename(columns={
        SKU_COL:               "Key",
        "sku_share":           "share",
        "sku_strategy_applied":"strategy_applied",
    })

    keep_cols = [c for c in [
        "RunDate", "MonthStart", "Level", "Key",
        "ModelName", "HorizonMonths", "ForecastUnits",
        "Lower", "Upper", "ModelVersion",
        SC_COL, SIZE_COL,
        "share",
        "strategy_applied",
    ] if c in allocated.columns]
    allocated = allocated[keep_cols].copy()
    allocated["Level"] = "SKU"

    if standalone_fc is not None and not standalone_fc.empty:
        sa = standalone_fc.copy()
        if "Key" not in sa.columns and SKU_COL in sa.columns:
            sa = sa.rename(columns={SKU_COL: "Key"})
        sa[SC_COL]             = STANDALONE
        sa["Level"]            = "SKU"
        sa["share"]            = np.nan
        sa["strategy_applied"] = "standalone_passthrough"
        if SIZE_COL not in sa.columns:
            sa[SIZE_COL] = "N/A"
        allocated = pd.concat([allocated, sa], ignore_index=True)

    return allocated.sort_values(["Key", "MonthStart"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 — Validation
# ---------------------------------------------------------------------------

def validate_segmented_allocation(
    sc_fc_df: pd.DataFrame,
    scol_fc_df: pd.DataFrame,
    sku_fc_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """Check sum consistency and no-negatives for the segmented allocation."""

    def _check(parent_fc, child_fc, parent_key):
        pf = parent_fc.copy()
        cf = child_fc.copy()
        if "Key" in pf.columns:
            pf = pf.rename(columns={"Key": parent_key})
        if parent_key not in cf.columns:
            return True, 0.0
        cf_hier = cf[cf.get(parent_key, pd.Series(dtype=str)) != STANDALONE] \
            if parent_key in cf.columns else cf
        p_tot = (
            pf.groupby([parent_key, "MonthStart", "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "P"})
        )
        c_tot = (
            cf_hier.groupby([parent_key, "MonthStart", "HorizonMonths"])["ForecastUnits"]
            .sum().reset_index().rename(columns={"ForecastUnits": "C"})
        ) if parent_key in cf_hier.columns else pd.DataFrame()
        if c_tot.empty:
            return False, np.nan
        merged = p_tot.merge(c_tot, on=[parent_key, "MonthStart", "HorizonMonths"], how="left")
        merged["diff"] = (merged["P"] - merged["C"].fillna(0)).abs()
        max_d = float(merged["diff"].max())
        return max_d < tolerance, max_d

    sc_ok, sc_d     = _check(sc_fc_df, scol_fc_df, SCODE_COL)
    scol_ok, scol_d = _check(scol_fc_df, sku_fc_df, SC_COL)
    no_neg          = bool((sku_fc_df["ForecastUnits"] >= 0).all())

    return {
        "sc_to_scol_sum_ok":    sc_ok,
        "sc_to_scol_max_diff":  round(sc_d, 6) if not np.isnan(sc_d) else None,
        "scol_to_sku_sum_ok":   scol_ok,
        "scol_to_sku_max_diff": round(scol_d, 6) if not np.isnan(scol_d) else None,
        "no_negatives":         no_neg,
    }


# ---------------------------------------------------------------------------
# Main orchestration: run_segmented_allocation
# ---------------------------------------------------------------------------

def run_segmented_allocation(
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
    smooth_alpha: float     = 0.20,
    **classification_thresholds,
) -> dict:
    """
    Run the full two-level segmented allocation.

    Parameters
    ----------
    scode_forecasts_df : upstream StyleCode forecasts (Key = StyleCodeDesc)
    gold_df            : gold demand
    dim_product_df     : dim_product table
    train_end          : share computation cut-off
    standalone_fc_df   : STANDALONE pass-through forecasts (or None)
    numeric params     : forwarded to share computation
    **classification_thresholds : overrides for DEFAULT_THRESHOLDS

    Returns
    -------
    dict with keys:
        scode_strat_map       : strategy map for Level-1 (StyleCode→StyleColor)
        scol_strat_map        : strategy map for Level-2 (StyleColor→SKU)
        stylecolor_shares     : shares DataFrame (StyleCode→StyleColor)
        size_shares           : shares DataFrame (StyleColor→SKU)
        stylecolor_forecasts  : allocated StyleColor forecasts
        sku_forecasts         : allocated SKU forecasts
        validation            : dict from validate_segmented_allocation()
    """
    logger.info("[v7.3] Starting segmented allocation (train_end=%s)", train_end.strftime("%Y-%m"))

    dp = dim_product_df.copy()
    dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
    dp_sc  = dp[dp[SCODE_COL].notna() & dp[SC_COL].notna()].drop_duplicates(SKU_COL)
    dp_sz  = dp[dp[SC_COL].notna()].drop_duplicates(SKU_COL)

    numeric = dict(
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=smooth_alpha,
    )

    # ── Level 1: classify + compute StyleCode→StyleColor shares ───────────
    logger.info("[v7.3] Level 1: classifying StyleCode→StyleColor groups")
    scode_strat = classify_parent_allocation_strategy(
        gold_df=gold_df,
        dim_map_df=dp_sc,
        entity_col=SCODE_COL,
        child_col=SC_COL,
        train_end=train_end,
        **classification_thresholds,
    )

    scol_shares = compute_segmented_shares(
        gold_df=gold_df,
        dim_map_df=dp_sc,
        entity_col=SCODE_COL,
        child_col=SC_COL,
        strategy_map_df=scode_strat,
        train_end=train_end,
        **numeric,
    )

    # ── Allocate StyleCode → StyleColor ───────────────────────────────────
    scol_fc = _apply_segmented_shares(
        fc_df=scode_forecasts_df,
        shares_df=scol_shares,
        entity_col=SCODE_COL,
        child_col=SC_COL,
    )
    scol_fc["Level"] = "StyleColor"
    logger.info(
        "[v7.3] StyleColor allocation: %d rows, %d StyleColors",
        len(scol_fc), scol_fc["Key"].nunique(),
    )

    # ── Level 2: classify + compute StyleColor→SKU shares ────────────────
    logger.info("[v7.3] Level 2: classifying StyleColor→SKU groups")
    scol_strat = classify_parent_allocation_strategy(
        gold_df=gold_df,
        dim_map_df=dp_sz,
        entity_col=SC_COL,
        child_col=SKU_COL,
        train_end=train_end,
        **classification_thresholds,
    )

    # Attach SizeDesc for schema compatibility
    sku_size   = dp_sz[[SKU_COL, SIZE_COL]].drop_duplicates(SKU_COL)
    size_shr_raw = compute_segmented_shares(
        gold_df=gold_df,
        dim_map_df=dp_sz,
        entity_col=SC_COL,
        child_col=SKU_COL,
        strategy_map_df=scol_strat,
        train_end=train_end,
        **numeric,
    )
    size_shares = size_shr_raw.merge(sku_size, on=SKU_COL, how="left")

    # ── Allocate StyleColor → SKU ─────────────────────────────────────────
    sku_fc = _sku_allocate_segmented(
        scol_fc=scol_fc,
        size_shares=size_shares,
        standalone_fc=standalone_fc_df,
    )
    logger.info(
        "[v7.3] SKU allocation: %d rows, %d SKUs",
        len(sku_fc), sku_fc["Key"].nunique(),
    )

    # ── Validation ────────────────────────────────────────────────────────
    val = validate_segmented_allocation(scode_forecasts_df, scol_fc, sku_fc)
    logger.info(
        "[v7.3] Validation: sc→scol=%s (%.4f)  scol→sku=%s (%.4f)  no_neg=%s",
        val["sc_to_scol_sum_ok"],   val.get("sc_to_scol_max_diff", float("nan")),
        val["scol_to_sku_sum_ok"],  val.get("scol_to_sku_max_diff", float("nan")),
        val["no_negatives"],
    )

    return {
        "scode_strat_map":      scode_strat,
        "scol_strat_map":       scol_strat,
        "stylecolor_shares":    scol_shares,
        "size_shares":          size_shares,
        "stylecolor_forecasts": scol_fc,
        "sku_forecasts":        sku_fc,
        "validation":           val,
    }


# ---------------------------------------------------------------------------
# Holdout scoring helper
# ---------------------------------------------------------------------------

def score_segmented_holdout(
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

    pred_cols = [
        SKU_COL, "MonthStart", "HorizonMonths", "ModelName", "ModelVersion",
        "PredictedUnits", "ActualUnits", "Error", "AbsError", "AbsPctError",
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
# Error decomposition
# ---------------------------------------------------------------------------

def build_segmented_error_decomp(
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

    def _score(a, p, label, h, m):
        if a.empty:
            return
        tot = a.sum()
        wmape = float((a - p).abs().sum() / tot * 100) if tot > 0 else np.nan
        rows.append({
            "Level":          label,
            "HorizonMonths":  int(h),
            "MonthStart":     m.strftime("%Y-%m"),
            "TotalActual":    round(float(tot), 2),
            "TotalPredicted": round(float(p.sum()), 2),
            "WMAPE":          round(wmape, 4),
        })

    for m in sorted(acts["MonthStart"].unique()):
        a_m = acts[acts["MonthStart"] == m]

        # SKU
        fc = sku_fc.copy()
        if "Key" in fc.columns:
            fc = fc.rename(columns={"Key": SKU_COL})
        fc["MonthStart"] = pd.to_datetime(fc["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        fc_m = fc[fc["MonthStart"] == m]
        for h in sorted(fc_m["HorizonMonths"].unique()):
            fc_mh = fc_m[fc_m["HorizonMonths"] == h].drop_duplicates(SKU_COL)
            mg    = a_m[[SKU_COL, TARGET_COL]].merge(fc_mh[[SKU_COL, "ForecastUnits"]], on=SKU_COL)
            _score(mg[TARGET_COL], mg["ForecastUnits"], "SKU", h, m)

        # StyleColor
        if SC_COL in acts.columns:
            fc2 = scol_fc.copy()
            if "Key" in fc2.columns:
                fc2 = fc2.rename(columns={"Key": SC_COL})
            fc2["MonthStart"] = pd.to_datetime(fc2["MonthStart"]).dt.to_period("M").dt.to_timestamp()
            fc2_m = fc2[fc2["MonthStart"] == m]
            a_sc  = a_m.groupby(SC_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc2_m["HorizonMonths"].unique()):
                fc_h = fc2_m[fc2_m["HorizonMonths"] == h].groupby(SC_COL)["ForecastUnits"].sum().reset_index()
                mg2  = a_sc.merge(fc_h, on=SC_COL)
                _score(mg2[TARGET_COL], mg2["ForecastUnits"], "StyleColor", h, m)

        # StyleCode
        if SCODE_COL in acts.columns:
            fc3 = scode_fc.copy()
            if "Key" in fc3.columns:
                fc3 = fc3.rename(columns={"Key": SCODE_COL})
            fc3["MonthStart"] = pd.to_datetime(fc3["MonthStart"]).dt.to_period("M").dt.to_timestamp()
            fc3_m = fc3[fc3["MonthStart"] == m]
            a_scd = a_m.groupby(SCODE_COL)[TARGET_COL].sum().reset_index()
            for h in sorted(fc3_m["HorizonMonths"].unique()):
                fc_h3 = fc3_m[fc3_m["HorizonMonths"] == h].groupby(SCODE_COL)["ForecastUnits"].sum().reset_index()
                mg3   = a_scd.merge(fc_h3, on=SCODE_COL)
                _score(mg3[TARGET_COL], mg3["ForecastUnits"], "StyleCode", h, m)

    if not rows:
        return pd.DataFrame(columns=["Level","HorizonMonths","MonthStart","TotalActual","TotalPredicted","WMAPE"])
    return pd.DataFrame(rows).sort_values(["HorizonMonths","MonthStart","Level"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parent-level win/loss analysis vs v7.2 recency_only
# ---------------------------------------------------------------------------

def build_parent_win_loss(
    strategy_map_df: pd.DataFrame,
    sku_fc_v73: pd.DataFrame,
    sku_fc_v72_recency: pd.DataFrame,
    actuals_df: pd.DataFrame,
    holdout_months: list[pd.Timestamp],
    entity_col: str,
    horizon: int = 3,
    month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compare v7.3 vs v7.2 recency_only at the parent group level.

    Parameters
    ----------
    strategy_map_df    : scode_strat_map or scol_strat_map from run_segmented_allocation()
    sku_fc_v73         : v7.3 SKU forecasts
    sku_fc_v72_recency : v7.2 recency_only SKU forecasts (same upstream forecast)
    actuals_df         : gold actuals
    holdout_months     : months to score
    entity_col         : parent column for grouping (StyleCodeDesc or StyleColorDesc)
    horizon            : HorizonMonths to compare (default 3)
    month              : specific month to compare (None = Jan 2026)

    Returns
    -------
    pd.DataFrame  (one row per parent group)
    """
    if month is None:
        month = pd.Timestamp("2026-01-01")

    acts = actuals_df.copy()
    acts["MonthStart"] = pd.to_datetime(acts["MonthStart"]).dt.to_period("M").dt.to_timestamp()
    acts[SKU_COL]      = acts[SKU_COL].astype(str).str.strip()

    def _prep_fc(fc):
        f = fc.copy()
        if "Key" in f.columns:
            f = f.rename(columns={"Key": SKU_COL})
        f["MonthStart"] = pd.to_datetime(f["MonthStart"]).dt.to_period("M").dt.to_timestamp()
        return f[
            (f["MonthStart"] == month) &
            (f["HorizonMonths"] == horizon)
        ][["SKU", "ForecastUnits", entity_col]].copy() if entity_col in f.columns else \
               f[(f["MonthStart"] == month) & (f["HorizonMonths"] == horizon)][["SKU", "ForecastUnits"]].copy()

    fc_73 = _prep_fc(sku_fc_v73)
    fc_72 = _prep_fc(sku_fc_v72_recency)

    acts_m = acts[acts["MonthStart"] == month][["SKU", TARGET_COL]].rename(columns={TARGET_COL: "ActualUnits"})

    # Join entity_col from strategy_map
    strat_cols = [entity_col, "assigned_profile", "selected_strategy", "reason_code"]
    strat_cols = [c for c in strat_cols if c in strategy_map_df.columns]

    rows = []
    for _, strat_row in strategy_map_df.iterrows():
        entity = strat_row[entity_col]

        # Filter SKU forecasts to this entity group
        if entity_col in fc_73.columns:
            sku_73 = fc_73[fc_73[entity_col] == entity]
        else:
            sku_73 = fc_73

        if entity_col in fc_72.columns:
            sku_72 = fc_72[fc_72[entity_col] == entity]
        else:
            sku_72 = fc_72

        common_skus = set(sku_73[SKU_COL].unique()) & set(sku_72[SKU_COL].unique())
        if not common_skus:
            continue

        sku_73_c = sku_73[sku_73[SKU_COL].isin(common_skus)].set_index(SKU_COL)["ForecastUnits"]
        sku_72_c = sku_72[sku_72[SKU_COL].isin(common_skus)].set_index(SKU_COL)["ForecastUnits"]
        act_c    = acts_m[acts_m[SKU_COL].isin(common_skus)].set_index(SKU_COL)["ActualUnits"]

        shared_skus = sorted(common_skus & set(act_c.index))
        if not shared_skus:
            continue

        a    = act_c.reindex(shared_skus).fillna(0)
        p73  = sku_73_c.reindex(shared_skus).fillna(0)
        p72  = sku_72_c.reindex(shared_skus).fillna(0)

        tot = a.sum()
        wmape_73 = float((a - p73).abs().sum() / max(1, tot) * 100)
        wmape_72 = float((a - p72).abs().sum() / max(1, tot) * 100)

        rows.append({
            entity_col:             entity,
            "allocation_stage":     strat_row.get("allocation_stage", f"{entity_col}→SKU"),
            "selected_strategy":    strat_row.get("selected_strategy", ""),
            "assigned_profile":     strat_row.get("assigned_profile",  ""),
            "reason_code":          strat_row.get("reason_code",       ""),
            "v72_recency_wmape":    round(wmape_72, 4),
            "v73_wmape":            round(wmape_73, 4),
            "wmape_delta":          round(wmape_73 - wmape_72, 4),
            "actual_units":         round(float(tot), 2),
            "v73_forecast_units":   round(float(p73.sum()), 2),
            "v72_forecast_units":   round(float(p72.sum()), 2),
            "absolute_error_v73":   round(float((a - p73).abs().sum()), 2),
            "absolute_error_v72":   round(float((a - p72).abs().sum()), 2),
            "month":                month.strftime("%Y-%m"),
            "horizon":              int(horizon),
        })

    return pd.DataFrame(rows).sort_values("wmape_delta").reset_index(drop=True)
