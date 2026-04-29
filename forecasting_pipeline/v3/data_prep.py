"""
lane7_forecast.data_prep
========================
Builds the core monthly panel from gold_fact_monthly_demand.

Key design decisions:
  - Zero-filling starts at each SKU's own first sale date, not the global dataset start.
    This avoids treating pre-existence months as true zeros.
  - Training period end is read from dim_date (IsTrain == 1) so the cut-off is
    data-driven, not hard-coded.
  - 2026 HOLDOUT actuals present in the source file are excluded from the
    phase=1 training panel; they are reserved for out-of-sample evaluation.
  - SKU attributes (Category, StyleCode, ColorCode, SizeCode, StyleColor) are
    joined from dim_product because the v2 gold schema no longer carries them
    inline. SKUs with no dim_product match receive "UNKNOWN" so they remain in
    the panel and the downstream encoder treats them as a single bucket.

dim_date semantics (authoritative as of the v2 rebuild)
-------------------------------------------------------
    Split=TRAIN,    IsTrain=1  -> 2017-05 through 2025-12  (104 months)
    Split=HOLDOUT,  IsTrain=0  -> 2026-01, 2026-02          (2 months)
    Split=FORECAST, IsTrain=0  -> 2026-03 through 2026-12  (10 months)

Public API
----------
    load_gold_tables(gold_dir) -> dict[str, pd.DataFrame]
    build_panel(demand_df, dim_date_df, phase=1, dim_product_df=None) -> pd.DataFrame
    build_stylecolor_panel(gold_df, dim_date_df, dim_product_df, phase=1) -> pd.DataFrame  [v6]

    phase=1  => train_end = last month where IsTrain==1  (2025-12)
    phase=2  => train_end = last TRAIN or HOLDOUT month   (2026-02)
                (phase=2 includes HOLDOUT actuals in the training panel; use
                 only for production refits after holdout evaluation is done.)

v6 change — build_stylecolor_panel
    Aggregates SKU-level demand to StyleColorDesc × Month, then calls
    build_panel treating StyleColorDesc as the "SKU" equivalent.  This lets
    the entire v5.2 pipeline (segmentation, features, CV, forecasting) run
    unchanged at the StyleColor level.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"

DEMAND_REQUIRED_COLS = {DATE_COL, SKU_COL, TARGET_COL}
DIM_DATE_REQUIRED_COLS = {"MonthStart", "IsTrain"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _resolve_demand_path(gold_dir: Path) -> Path:
    """Prefer refreshed gold (v2). Fall back to legacy filename if v2 absent."""
    v2 = gold_dir / "gold_fact_monthly_demand_v2.csv"
    v1 = gold_dir / "gold_fact_monthly_demand.csv"

    if v2.exists():
        return v2
    if v1.exists():
        logger.warning("Using legacy gold_fact_monthly_demand.csv — v2 not found.")
        return v1

    raise FileNotFoundError(
        f"No gold demand file found. Expected one of "
        f"{v2.name} or {v1.name} in {gold_dir}"
    )


def load_gold_tables(gold_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all Gold-layer CSV files from *gold_dir*.

    Returns a dict with keys:
        "demand"    -> gold_fact_monthly_demand
        "dim_date"  -> dim_date
        "dim_product" -> dim_product  (may be None if missing)
        "dim_customer" -> dim_customer (may be None if missing)

    Raises FileNotFoundError for the two required tables.
    """
    gold_dir = Path(gold_dir)

    def _read(name: str, required: bool = True) -> pd.DataFrame | None:
        if name == "gold_fact_monthly_demand":
            path = _resolve_demand_path(gold_dir)
        else:
            path = gold_dir / f"{name}.csv"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            logger.warning("Optional file not found, skipping: %s", path)
            return None
        df = pd.read_csv(path)
        logger.info("Loaded %s: %s rows, %s cols", name, len(df), df.shape[1])
        return df

    return {
        "demand":       _read("gold_fact_monthly_demand", required=True),
        "dim_date":     _read("dim_date",                 required=True),
        "dim_product":  _read("dim_product",              required=False),
        "dim_customer": _read("dim_customer",             required=False),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_demand(df: pd.DataFrame) -> None:
    missing = DEMAND_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"gold_fact_monthly_demand is missing columns: {missing}")


def _validate_dim_date(df: pd.DataFrame) -> None:
    missing = DIM_DATE_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"dim_date is missing columns: {missing}")


def _coerce_types(demand: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and numeric columns; drop rows that fail."""
    df = demand.copy()
    df[DATE_COL]   = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[DATE_COL, SKU_COL, TARGET_COL])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with null MonthStart/SKU/UnitsSold", n_dropped)

    # Normalise MonthStart to first-of-month midnight
    df[DATE_COL] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()
    df[SKU_COL]  = df[SKU_COL].astype(str).str.strip()
    return df


def _resolve_training_end(dim_date: pd.DataFrame, phase: int) -> pd.Timestamp:
    """
    Return the last MonthStart that belongs to the training window.

    Phase 1: last month where IsTrain == 1
             Covers 2017-05 through 2025-12. HOLDOUT (2026-01, 2026-02) and
             FORECAST months are excluded so Jan-Feb 2026 remain true
             out-of-sample for evaluation.

    Phase 2: last month that is a real actual — i.e. Split in (TRAIN, HOLDOUT).
             Used ONLY for production refits after holdout evaluation has been
             completed and we want to feed 2026-01 and 2026-02 into the model
             before forecasting 2026-03 onwards. FORECAST-labelled months are
             always excluded so we never zero-fill into months without actuals.

    v2 rebuild: the legacy "TRAIN_V4" split label no longer exists in
    dim_date.csv. This function now keys off Split ∈ {TRAIN, HOLDOUT} directly.
    """
    dim_date = dim_date.copy()
    dim_date["MonthStart"] = pd.to_datetime(dim_date["MonthStart"]).dt.to_period("M").dt.to_timestamp()

    if phase == 1:
        train_rows = dim_date[dim_date["IsTrain"] == 1]
        if train_rows.empty:
            raise ValueError("dim_date has no rows with IsTrain == 1")
        end = train_rows["MonthStart"].max()
    elif phase == 2:
        # All "real actual" months: TRAIN or HOLDOUT (both have actuals loaded).
        # FORECAST months in dim_date exist only for the calendar spine and
        # must NOT be included — there are no actuals for them.
        if "Split" in dim_date.columns:
            actual_rows = dim_date[dim_date["Split"].isin(["TRAIN", "HOLDOUT"])]
        else:
            # Fallback when Split column is absent: use IsTrain==1 only
            logger.warning("dim_date has no Split column; phase=2 falling back to IsTrain==1")
            actual_rows = dim_date[dim_date["IsTrain"] == 1]
        if actual_rows.empty:
            raise ValueError("dim_date has no rows with Split in (TRAIN, HOLDOUT)")
        end = actual_rows["MonthStart"].max()
    else:
        raise ValueError(f"phase must be 1 or 2, got {phase}")

    logger.info("Training period end (phase %d): %s", phase, end.strftime("%Y-%m"))
    return end


# ---------------------------------------------------------------------------
# Public: build_panel
# ---------------------------------------------------------------------------

def build_panel(
    demand_df: pd.DataFrame,
    dim_date_df: pd.DataFrame,
    phase: int = 1,
    dim_product_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the complete SKU × Month panel used by all downstream steps.

    Parameters
    ----------
    demand_df      : gold_fact_monthly_demand (raw, as loaded from CSV).
                     The v2 schema carries only MonthStart, SKU, UnitsSold, Revenue.
    dim_date_df    : dim_date dimension table (must contain MonthStart, IsTrain;
                     Split column is used when present).
    dim_product_df : optional dim_product table with SKU and attribute columns
                     (Category, StyleCode, ColorCode, SizeCode, StyleColor).
                     When provided, these attributes are joined onto the panel
                     because the v2 gold schema no longer carries them inline.
                     SKUs with no match receive "UNKNOWN" for each attribute.
    phase          : 1 = train through 2025-12, evaluate on 2026-01/02 HOLDOUT
                     2 = train through 2026-02 (HOLDOUT folded in for production
                         refit — use AFTER holdout evaluation is complete)

    Returns
    -------
    pd.DataFrame with columns:
        MonthStart, SKU, UnitsSold,
        Category, StyleCode, ColorCode, SizeCode, StyleColor,
        Year, Month, Quarter,
        IsTrain (0/1 from dim_date), Split (from dim_date if present)

    Zero-fill logic
    ---------------
    For each SKU, months between its own first_sale_date and train_end with no
    recorded sales are filled with UnitsSold = 0. Months before the SKU's first
    sale are never created (the SKU did not exist yet).
    """
    _validate_demand(demand_df)
    _validate_dim_date(dim_date_df)

    df     = _coerce_types(demand_df)
    tr_end = _resolve_training_end(dim_date_df, phase)

    # ------------------------------------------------------------------
    # 1. Restrict to [global_start, train_end].
    #    Phase 1: excludes 2026-01/02 HOLDOUT and FORECAST months.
    #    Phase 2: includes 2026-01/02 but still excludes FORECAST months.
    # ------------------------------------------------------------------
    df = df[df[DATE_COL] <= tr_end].copy()
    logger.info("Rows after date filter (<= %s): %d", tr_end.strftime("%Y-%m"), len(df))

    # ------------------------------------------------------------------
    # 2. Aggregate: a SKU can appear more than once per month if the source
    #    data was not fully deduplicated. Sum to guarantee one row per key.
    #    v2 note: transaction_base has already been cleaned+deduped upstream,
    #    so this groupby is a safety net and should be a no-op in practice.
    # ------------------------------------------------------------------
    inline_attr_cols = [
        c for c in ["Category", "StyleCode", "ColorCode", "SizeCode", "StyleColor"]
        if c in df.columns
    ]
    agg_dict: dict = {TARGET_COL: "sum"}
    for c in inline_attr_cols:
        agg_dict[c] = "first"

    df = (
        df.groupby([SKU_COL, DATE_COL], as_index=False)
          .agg(agg_dict)
          .sort_values([SKU_COL, DATE_COL])
          .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 3. Per-SKU zero-fill between first_sale and train_end
    # ------------------------------------------------------------------
    sku_first_sale = df.groupby(SKU_COL)[DATE_COL].min().rename("first_sale")
    sku_attrs      = (
        df.groupby(SKU_COL)[inline_attr_cols].first()
        if inline_attr_cols else pd.DataFrame()
    )

    panel_rows = []

    for sku, group in df.groupby(SKU_COL, sort=False):
        first_sale = sku_first_sale[sku]
        # Full monthly spine from this SKU's first sale to train_end
        spine = pd.date_range(first_sale, tr_end, freq="MS")

        sku_df = group.set_index(DATE_COL).reindex(spine)
        sku_df.index.name = DATE_COL
        sku_df[SKU_COL]    = sku
        sku_df[TARGET_COL] = sku_df[TARGET_COL].fillna(0.0)

        # Forward-fill (then re-assign from known first row) attribute columns
        if inline_attr_cols:
            attrs = sku_attrs.loc[sku] if sku in sku_attrs.index else {}
            for c in inline_attr_cols:
                sku_df[c] = attrs.get(c, np.nan)

        panel_rows.append(sku_df.reset_index())

    panel = pd.concat(panel_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # 3b. v2 schema: if attribute columns were not in the demand file, join
    #     them from dim_product. This is the primary path under v2 because
    #     the new gold dataset only carries SKU / MonthStart / UnitsSold /
    #     Revenue. Without this join, the ML feature set would silently lose
    #     Category_enc / StyleCode_enc / ColorCode_enc and WMAPE would drop.
    # ------------------------------------------------------------------
    attr_cols_target = ["Category", "StyleCode", "ColorCode", "SizeCode", "StyleColor"]
    missing_attrs    = [c for c in attr_cols_target if c not in panel.columns]

    if missing_attrs and dim_product_df is not None:
        dp = dim_product_df[[SKU_COL] + [c for c in attr_cols_target if c in dim_product_df.columns]].copy()
        dp[SKU_COL] = dp[SKU_COL].astype(str).str.strip()
        # de-dup dim_product on SKU (should already be unique, safety net)
        dp = dp.drop_duplicates(subset=[SKU_COL], keep="first")

        n_before_merge = len(panel)
        panel = panel.merge(dp, on=SKU_COL, how="left")
        assert len(panel) == n_before_merge, "dim_product join caused row fan-out"

        # Fill unknowns with a single sentinel so the label encoder treats
        # them as one bucket rather than one-per-SKU. 621/3806 SKUs have no
        # dim_product row in the v2 data (verified at load time).
        n_unknown_per_col = {}
        for c in attr_cols_target:
            if c in panel.columns:
                n_unknown_per_col[c] = int(panel[c].isna().sum())
                panel[c] = panel[c].fillna("UNKNOWN").astype(str)
        logger.info(
            "dim_product joined onto panel. Unknown counts per attr: %s",
            n_unknown_per_col,
        )
    elif missing_attrs:
        # No dim_product provided and attrs missing — fill with UNKNOWN so
        # downstream code does not see NaN string columns.
        logger.warning(
            "Attribute columns missing from demand and no dim_product provided: %s. "
            "Filling with 'UNKNOWN'.", missing_attrs
        )
        for c in missing_attrs:
            panel[c] = "UNKNOWN"

    # ------------------------------------------------------------------
    # 4. Join dim_date to attach IsTrain, Year, Month, Quarter, Split
    # ------------------------------------------------------------------
    dim_date_clean = dim_date_df.copy()
    dim_date_clean["MonthStart"] = (
        pd.to_datetime(dim_date_clean["MonthStart"])
          .dt.to_period("M").dt.to_timestamp()
    )
    date_cols_to_join = [c for c in ["Year", "Month", "Quarter", "IsTrain", "Split"]
                         if c in dim_date_clean.columns]
    panel = panel.merge(
        dim_date_clean[["MonthStart"] + date_cols_to_join],
        on="MonthStart",
        how="left",
    )

    # For months zero-filled but not in dim_date (edge case), derive calendar cols
    if "Year" not in panel.columns or panel["Year"].isna().any():
        panel["Year"]    = panel[DATE_COL].dt.year
        panel["Month"]   = panel[DATE_COL].dt.month
        panel["Quarter"] = panel[DATE_COL].dt.quarter

    # ------------------------------------------------------------------
    # 5. Final sort and column ordering
    # ------------------------------------------------------------------
    core_cols  = [DATE_COL, SKU_COL, TARGET_COL]
    attr_cols  = [c for c in attr_cols_target if c in panel.columns]
    cal_cols   = [c for c in ["Year", "Month", "Quarter", "IsTrain", "Split"]
                  if c in panel.columns]
    ordered_cols = core_cols + attr_cols + cal_cols
    panel        = panel[ordered_cols].sort_values([SKU_COL, DATE_COL]).reset_index(drop=True)

    logger.info(
        "Panel built: %d rows, %d unique SKUs, date range %s to %s",
        len(panel),
        panel[SKU_COL].nunique(),
        panel[DATE_COL].min().strftime("%Y-%m"),
        panel[DATE_COL].max().strftime("%Y-%m"),
    )
    return panel


# ---------------------------------------------------------------------------
# v6: StyleColor-level panel (NEW — hierarchical forecasting entry point)
# ---------------------------------------------------------------------------

def build_stylecolor_panel(
    gold_df,
    dim_date_df,
    dim_product_df,
    phase: int = 1,
):
    """
    Build a StyleColorDesc x Month panel for the v6 hierarchical pipeline.

    Aggregates SKU-level gold demand to StyleColorDesc x Month, then calls
    build_panel() treating StyleColorDesc as the "SKU" equivalent.
    STANDALONE SKUs (no valid StyleColorDesc) are excluded; they are handled
    separately by the v5.2 SKU-level pipeline.

    Returns the same schema as build_panel() but 'SKU' column contains
    StyleColorDesc values. Attribute columns (Category_enc etc.) are absent
    because they are not meaningful at the StyleColor aggregate level — the
    ML feature set (lags, rolling, trend, calendar) applies unchanged.
    """
    from lane7_forecast.allocation import build_stylecolor_demand, get_standalone_skus

    # Aggregate demand to StyleColorDesc level (excludes STANDALONE SKUs)
    sc_demand, _ = build_stylecolor_demand(gold_df, dim_product_df)

    # build_panel expects a "SKU" column -- rename StyleColorDesc
    sc_demand_renamed = sc_demand.rename(columns={"StyleColorDesc": SKU_COL})

    # Call the unmodified build_panel -- no dim_product (not meaningful here)
    panel = build_panel(
        demand_df=sc_demand_renamed,
        dim_date_df=dim_date_df,
        phase=phase,
        dim_product_df=None,
    )

    import logging as _log
    _log.getLogger(__name__).info(
        "[v6] StyleColor panel: %d rows, %d StyleColors, phase=%d",
        len(panel), panel[SKU_COL].nunique(), phase,
    )
    return panel


# ---------------------------------------------------------------------------
# v7: StyleCode-level panel (NEW — two-level hierarchy entry point)
# ---------------------------------------------------------------------------

def build_stylecode_panel(
    gold_df,
    dim_date_df,
    dim_product_df,
    phase: int = 1,
):
    """
    Build a StyleCodeDesc × Month panel for the v7 hierarchical pipeline.

    Aggregates SKU-level gold demand to StyleCodeDesc × Month, then calls
    build_panel() treating StyleCodeDesc as the "SKU" equivalent.

    STANDALONE SKUs (null or missing StyleCodeDesc) are excluded; they are
    handled separately by the v5.2 / v6 SKU-level pipeline.

    Returns the same schema as build_panel() but 'SKU' column contains
    StyleCodeDesc values.  The full ML feature set (lags, rolling stats,
    trend index, calendar) applies unchanged because build_panel() knows
    nothing about the entity label.
    """
    from lane7_forecast.stylecode_allocation import build_stylecode_demand

    # Aggregate demand to StyleCodeDesc level
    scode_demand, _ = build_stylecode_demand(gold_df, dim_product_df)

    # build_panel expects a "SKU" column — rename StyleCodeDesc
    scode_demand_renamed = scode_demand.rename(columns={"StyleCodeDesc": SKU_COL})

    panel = build_panel(
        demand_df=scode_demand_renamed,
        dim_date_df=dim_date_df,
        phase=phase,
        dim_product_df=None,   # not meaningful at aggregated level
    )

    import logging as _log
    _log.getLogger(__name__).info(
        "[v7] StyleCode panel: %d rows, %d StyleCodes, phase=%d",
        len(panel), panel[SKU_COL].nunique(), phase,
    )
    return panel
