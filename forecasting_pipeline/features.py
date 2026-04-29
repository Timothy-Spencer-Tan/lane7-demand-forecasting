"""
lane7_forecast.features
========================
Generates time-series features for the three forecast horizons.

All lag and rolling features are computed strictly within each SKU's own history
using groupby + shift/rolling so there is ZERO leakage across SKUs or across time.

Feature groups
--------------
    Calendar        : month, quarter, year, month_sin/cos (cyclical encoding)
    Lag features    : UnitsSold at t-1, t-2, t-3, t-6, t-12
                      (only lags relevant to each horizon are guaranteed non-null)
    Rolling stats   : 3/6/12-month rolling mean; 3/6/12-month rolling std [v5: added 6m mean]
    Time trend      : per-SKU integer index (months since first sale) [v4]
    Momentum        : 3-month demand growth rate (lag_1 - lag_3)/(lag_3+1) [v5]
    YoY growth      : (t-12 units) / (t-24 units) -- trend signal for long horizon
    SKU attributes  : Category, StyleCode encoded as integers (for tree models)

Public API
----------
    create_features(panel, horizon_months) -> pd.DataFrame

    horizon_months : 1 | 3 | 12
        Controls which lag columns are required to be non-null for a row to be
        included in the training set (rows with null required lags are dropped).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

DATE_COL   = "MonthStart"
SKU_COL    = "SKU"
TARGET_COL = "UnitsSold"

# Lag distances computed for all horizons; downstream code filters as needed
ALL_LAG_PERIODS = [1, 2, 3, 6, 12, 24]

# Rolling windows for mean features (over months ending at t-1 so no leakage)
# v5: added 6-month mean window — fills gap between 3m and 12m for quarterly planning
ROLLING_WINDOWS = [3, 6, 12]
# Rolling windows for std/volatility features (adds 6-month window)
ROLLING_STD_WINDOWS = [3, 6, 12]

# Lags that MUST be non-null to include a row, per horizon
# v5: H=3 now requires lag_1 as a short-term anchor (recent actuals help 3m planning)
REQUIRED_LAGS = {
    1:  ["lag_1", "lag_2", "lag_3", "rolling_mean_3"],
    3:  ["lag_1", "lag_3", "lag_6", "lag_12", "rolling_mean_6"],
    12: ["lag_12", "lag_24", "rolling_mean_12"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag_N columns per SKU with no cross-SKU contamination."""
    df = panel.copy()
    for lag in ALL_LAG_PERIODS:
        col = f"lag_{lag}"
        df[col] = df.groupby(SKU_COL, sort=False)[TARGET_COL].shift(lag)
    return df


def _add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling_mean_N (windows 3, 12) and rolling_std_N (windows 3, 6, 12).

    The rolling window is computed over the N months *ending at t-1* to avoid
    leakage (we shift by 1 before rolling so the window never includes row t).

    rolling_std_6 is a new v4 feature — medium-horizon volatility signal that
    captures demand variability better than rolling_std_3 (too noisy) or
    rolling_std_12 (too slow to react).
    """
    df = panel.copy()
    shifted = df.groupby(SKU_COL, sort=False)[TARGET_COL].shift(1)

    # Rolling mean: windows [3, 12]
    for window in ROLLING_WINDOWS:
        roll = shifted.groupby(df[SKU_COL], sort=False).transform(
            lambda s: s.rolling(window, min_periods=max(1, window // 2)).mean()
        )
        df[f"rolling_mean_{window}"] = roll

    # Rolling std: windows [3, 6, 12] — adds 6-month volatility signal
    for window in ROLLING_STD_WINDOWS:
        roll_std = shifted.groupby(df[SKU_COL], sort=False).transform(
            lambda s: s.rolling(window, min_periods=max(1, window // 2)).std()
        )
        df[f"rolling_std_{window}"] = roll_std

    return df


def _add_trend_index(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add a per-SKU integer time trend index: months since the SKU's first sale.

    This gives tree models a direct trend signal. Without it, models must
    infer trend from Year alone, which is coarser and harder to generalise.

    trend_index = 0 for the SKU's first month, 1 for the second, etc.
    At inference time this is extrapolated from the panel max.
    """
    df = panel.copy()
    df["trend_index"] = df.groupby(SKU_COL, sort=False).cumcount()
    return df


def _add_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum_3: 3-month demand growth rate.

    momentum_3 = (lag_1 - lag_3) / (lag_3 + 1)

    Captures whether demand is accelerating or decelerating over the recent
    3-month window. Critical for 3-month order planning: a positive value
    suggests upward momentum, negative suggests pullback.

    The +1 denominator avoids division by zero on zero-demand months.
    Cap at [-5, 5] to suppress outliers from near-zero denominators.

    v5: added as a primary H=3 feature.
    """
    df = panel.copy()
    if "lag_1" not in df.columns or "lag_3" not in df.columns:
        # Lags may not exist yet if called before _add_lag_features — skip gracefully
        df["momentum_3"] = np.nan
        return df
    numerator   = df["lag_1"] - df["lag_3"]
    denominator = df["lag_3"] + 1.0
    df["momentum_3"] = np.clip(numerator / denominator, -5.0, 5.0)
    return df


def _add_yoy_growth(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Year-over-year growth rate: lag_12 / lag_24 (capped to avoid blow-ups).
    Used primarily for the 12-month horizon.
    """
    df = panel.copy()
    lag12 = df.get("lag_12", df.groupby(SKU_COL, sort=False)[TARGET_COL].shift(12))
    lag24 = df.get("lag_24", df.groupby(SKU_COL, sort=False)[TARGET_COL].shift(24))

    # Safe division: 0/0 -> nan, x/0 -> nan
    yoy = np.where(
        (lag24.isna()) | (lag24 == 0),
        np.nan,
        lag12 / lag24,
    )
    # Cap at 10x growth / 0.1x decline to suppress outlier influence
    df["yoy_growth"] = np.clip(yoy, 0.1, 10.0)
    return df


def _add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Month, quarter, year, plus cyclical sin/cos encoding for month so
    tree models can learn the periodic nature of the calendar.
    """
    df = panel.copy()

    if "Month" not in df.columns:
        df["Month"] = df[DATE_COL].dt.month
    if "Quarter" not in df.columns:
        df["Quarter"] = df[DATE_COL].dt.quarter
    if "Year" not in df.columns:
        df["Year"] = df[DATE_COL].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df


def _encode_categoricals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Integer-encode string categorical columns so tree models can use them.
    New unseen values at inference time get -1.
    """
    df      = panel.copy()
    cat_cols = [c for c in ["Category", "StyleCode", "ColorCode"] if c in df.columns]

    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].fillna("UNKNOWN").astype(str))
        encoders[col] = le

    # Store encoders on the DataFrame as metadata (accessible downstream if needed)
    df.attrs["label_encoders"] = encoders
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_features(
    panel: pd.DataFrame,
    horizon_months: int,
    drop_null_required: bool = True,
) -> pd.DataFrame:
    """
    Add all features to the panel and return the feature-enriched DataFrame.

    Parameters
    ----------
    panel            : output of build_panel() (with Segment column attached)
    horizon_months   : 1, 3, or 12 — controls which lag nulls are treated as errors
    drop_null_required: if True, drop rows where required lags for this horizon are null.
                        Set False when building the inference panel (future months).

    Returns
    -------
    pd.DataFrame — all original columns plus:
        lag_1, lag_2, lag_3, lag_6, lag_12, lag_24
        rolling_mean_3, rolling_std_3
        rolling_mean_12, rolling_std_12
        yoy_growth
        month_sin, month_cos  (Month, Quarter, Year already present from build_panel)
        Category_enc, StyleCode_enc, ColorCode_enc  (where source cols exist)
    """
    if horizon_months not in (1, 3, 12):
        raise ValueError(f"horizon_months must be 1, 3, or 12; got {horizon_months}")

    df = panel.copy()
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_trend_index(df)          # v4: per-SKU time trend
    df = _add_momentum(df)             # v5: 3-month demand momentum for planning
    df = _add_yoy_growth(df)
    df = _add_calendar_features(df)
    df = _encode_categoricals(df)

    if drop_null_required:
        required = REQUIRED_LAGS[horizon_months]
        available_required = [c for c in required if c in df.columns]
        n_before = len(df)
        df = df.dropna(subset=available_required).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.info(
                "Horizon %d-month: dropped %d rows with null required lags (%s)",
                horizon_months, n_dropped, available_required,
            )

    logger.info(
        "Features created for horizon=%d — panel shape: %s",
        horizon_months, df.shape,
    )
    return df


def get_feature_columns(horizon_months: int, panel: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names for *horizon_months* that are
    actually present in *panel* (useful for constructing X arrays).

    Excludes the target, date, and ID columns.
    """
    exclude = {DATE_COL, SKU_COL, TARGET_COL, "Split", "IsTrain",
               "Category", "StyleCode", "ColorCode", "SizeCode", "StyleColor",
               "Segment"}

    # Base feature pools per horizon
    # v4 additions: rolling_std_6 (volatility), trend_index (time trend)
    lag_features = {
        1:  ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3",
             "rolling_std_6", "rolling_mean_12", "month_sin", "month_cos",
             "Month", "Quarter", "Year", "trend_index"],
        3:  ["lag_1", "lag_3", "lag_6", "lag_12",
             "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
             "rolling_std_3", "rolling_std_6",
             "momentum_3",
             "month_sin", "month_cos", "Month", "Quarter", "Year", "trend_index"],
        12: ["lag_12", "lag_24", "rolling_mean_12", "rolling_std_12",
             "rolling_std_6", "yoy_growth", "month_sin", "month_cos",
             "Month", "Quarter", "Year", "trend_index"],
    }

    candidates = lag_features[horizon_months]
    # Add encoded categorical columns if present
    candidates += [c for c in panel.columns if c.endswith("_enc")]

    # Return only columns that actually exist in the panel
    return [c for c in candidates if c in panel.columns and c not in exclude]
