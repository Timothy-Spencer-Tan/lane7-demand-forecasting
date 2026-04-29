"""
lane7_forecast.segmentation
============================
Classifies every SKU into one of three demand segments:

    REGULAR      — consistently selling SKU; suitable for ML + statistical models
    INTERMITTENT — sparse demand (many zero months) but still alive; use lightweight
                   statistical methods (Croston, seasonal naive)
    DEAD         — zero sales in the trailing observation window; forecast = 0

Segment membership drives model selection in all three forecast horizons so that
training effort is concentrated where models can actually learn something useful.

Public API
----------
    segment_skus(panel, trailing_months=12, zero_ratio_threshold=0.40) -> pd.DataFrame

The returned DataFrame has one row per SKU with columns:
    SKU, Segment, zero_ratio, cv_nonzero, trailing_sum,
    first_sale, last_sale, n_active_months
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"

# Segment labels (use these constants everywhere to avoid typos)
SEG_REGULAR      = "REGULAR"
SEG_INTERMITTENT = "INTERMITTENT"
SEG_DEAD         = "DEAD"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sku_metrics(
    group: pd.DataFrame,
    trailing_months: int,
) -> dict:
    """Compute per-SKU summary metrics used for classification."""
    units = group[TARGET_COL].values

    total_months   = len(units)
    zero_months    = int((units == 0).sum())
    zero_ratio     = zero_months / total_months if total_months > 0 else 1.0

    nonzero_units  = units[units > 0]
    if len(nonzero_units) > 1:
        cv_nonzero = float(nonzero_units.std() / nonzero_units.mean())
    else:
        cv_nonzero = np.nan

    # Trailing window sum (default: last 12 months)
    sorted_units   = group.sort_values(DATE_COL)[TARGET_COL]
    trailing_sum   = float(sorted_units.tail(trailing_months).sum())

    return {
        "zero_ratio":      zero_ratio,
        "cv_nonzero":      cv_nonzero,
        "trailing_sum":    trailing_sum,
        "n_active_months": int((units > 0).sum()),
        "first_sale":      group[DATE_COL].min(),
        "last_sale":       group[DATE_COL].max(),
        "total_months":    total_months,
    }


def _classify(row: pd.Series, zero_ratio_threshold: float) -> str:
    """
    Apply segment rules.

    DEAD         : zero trailing sales (SKU has gone inactive)
    INTERMITTENT : trailing sales > 0 but too many zero months
    REGULAR      : everything else
    """
    if row["trailing_sum"] == 0:
        return SEG_DEAD
    if row["zero_ratio"] >= zero_ratio_threshold:
        return SEG_INTERMITTENT
    return SEG_REGULAR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_skus(
    panel: pd.DataFrame,
    trailing_months: int = 12,
    zero_ratio_threshold: float = 0.40,
) -> pd.DataFrame:
    """
    Classify every SKU in the panel into REGULAR / INTERMITTENT / DEAD.

    Parameters
    ----------
    panel               : output of build_panel(); must contain MonthStart, SKU, UnitsSold
    trailing_months     : look-back window (in months) used to determine if a SKU is DEAD
    zero_ratio_threshold: fraction of months with zero sales above which a SKU is
                          considered INTERMITTENT (not DEAD, since trailing_sum > 0)

    Returns
    -------
    pd.DataFrame — one row per SKU, sorted by Segment then SKU. Columns:
        SKU, Segment, zero_ratio, cv_nonzero, trailing_sum,
        first_sale, last_sale, n_active_months, total_months
    """
    required = {DATE_COL, SKU_COL, TARGET_COL}
    missing  = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    records = []
    for sku, group in panel.groupby(SKU_COL, sort=False):
        metrics = _compute_sku_metrics(group, trailing_months)
        metrics[SKU_COL] = sku
        records.append(metrics)

    segments_df = pd.DataFrame(records)
    segments_df["Segment"] = segments_df.apply(
        _classify, axis=1, zero_ratio_threshold=zero_ratio_threshold
    )

    # Reorder columns
    col_order = [
        SKU_COL, "Segment",
        "zero_ratio", "cv_nonzero", "trailing_sum",
        "n_active_months", "total_months",
        "first_sale", "last_sale",
    ]
    segments_df = segments_df[col_order].sort_values(["Segment", SKU_COL]).reset_index(drop=True)

    # Summary log
    counts = segments_df["Segment"].value_counts()
    logger.info(
        "SKU segmentation complete — REGULAR: %d | INTERMITTENT: %d | DEAD: %d",
        counts.get(SEG_REGULAR, 0),
        counts.get(SEG_INTERMITTENT, 0),
        counts.get(SEG_DEAD, 0),
    )
    return segments_df


def attach_segment(panel: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the Segment column from segments_df back onto the panel.

    Useful downstream so that feature engineering and model training can
    filter the panel by segment without re-running segmentation.
    """
    return panel.merge(segments_df[[SKU_COL, "Segment"]], on=SKU_COL, how="left")


def get_segment_skus(segments_df: pd.DataFrame, segment: str) -> list[str]:
    """Return the list of SKUs belonging to *segment*."""
    return segments_df.loc[segments_df["Segment"] == segment, SKU_COL].tolist()
