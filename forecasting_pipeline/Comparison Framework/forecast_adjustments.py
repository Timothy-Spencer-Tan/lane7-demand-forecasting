"""
lane7_forecast.forecast_adjustments
=====================================
v6.2 post-model forecast correction and stabilization layer.

All adjustments operate on raw model predictions AFTER the model has
produced them and BEFORE they are written to the output schema.
No changes are made to feature engineering, segmentation, or model training.

Adjustment pipeline (applied in this order):
--------------------------------------------
1. Intermittent cap
   For INTERMITTENT SKUs: cap prediction relative to trailing sum and last
   non-zero value to prevent large spikes from sparse-history noise.

2. Shrinkage (anti-overprediction)
   For any SKU: if the prediction significantly exceeds recent trailing mean,
   shrink it toward the mean using a configurable weight.
   DEAD SKUs are exempt.

3. ML vs Seasonal blend (sanity check)
   For REGULAR and INTERMITTENT ML forecasts: if the ML prediction diverges
   substantially from a seasonal baseline, blend them to damp extremes.
   Only applied when an ML model was used (not for baseline-only SKUs).

4. Recursive stabilization (H=3 lag anchor)
   For multi-step recursive forecasting: blend the predicted lag value with
   the trailing mean before feeding it as lag_1 into the next step.
   This prevents compounding drift across steps.
   Applied inside _forecast_ml_recursive (called from forecasting.py).

Centralized config
------------------
    FORECAST_ADJUSTMENT_CONFIG  — default values for all parameters
    get_config(**overrides)     — merge caller overrides into defaults

Public API
----------
    get_config(**overrides) -> dict
    apply_shrinkage(yhat, sku_hist, segment, config) -> list[float]
    apply_intermittent_cap(yhat, sku_hist, config) -> list[float]
    blend_ml_with_seasonal(yhat, seasonal_yhat, config) -> list[float]
    stabilize_lag_for_recursion(predicted_val, sku_hist_so_far, config) -> float
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralized configuration
# ---------------------------------------------------------------------------

FORECAST_ADJUSTMENT_CONFIG: dict = {
    # ── Shrinkage (anti-overprediction) ──────────────────────────────────────
    # If forecast > shrink_threshold × trailing_mean_3, blend toward the mean.
    # shrink_factor: weight given to the model prediction (0=full mean, 1=no shrink)
    # E.g. 0.8 → final = 0.8 * forecast + 0.2 * trailing_mean_3
    # Set shrink_factor=1.0 to disable shrinkage entirely.
    "shrink_threshold":  1.5,    # trigger multiplier above trailing 3-month mean
    "shrink_factor":     0.8,    # weight on model prediction when shrinkage fires

    # ── Recursive stabilization (H=3 lag anchor) ─────────────────────────────
    # Blend the predicted value with the trailing mean before using it as
    # lag_1 in the next recursive step. Reduces compounding drift.
    # recursive_alpha: weight on predicted value (0=full mean, 1=no blend)
    # Only applies when step_index > 0 (not the first forecast step).
    # Set recursive_alpha=1.0 to disable (pure predicted lag, original behaviour).
    "recursive_alpha":   0.7,    # weight on prediction vs trailing mean in lag feed

    # ── ML vs Seasonal blend ─────────────────────────────────────────────────
    # If |ML - Seasonal| / max(1, Seasonal) > blend_threshold, blend them.
    # blend_weight: weight given to the ML forecast
    # E.g. 0.7 → final = 0.7 * ML + 0.3 * Seasonal
    # Set blend_weight=1.0 to disable blending entirely.
    "blend_threshold":   0.25,   # fractional divergence trigger (25%)
    "blend_weight":      0.7,    # weight on ML forecast when blending fires

    # ── Intermittent cap ─────────────────────────────────────────────────────
    # Cap INTERMITTENT predictions at:
    #   max(intermittent_cap_multiplier × trailing_sum_12, last_nonzero × 3)
    # Prevents large spikes from Croston/Seasonal models on sparse history.
    # Set intermittent_cap_multiplier=inf to disable.
    "intermittent_cap_multiplier":  2.0,

    # ── Trailing mean windows ─────────────────────────────────────────────────
    # Windows used internally for all adjustments above.
    "trailing_window_3":   3,
    "trailing_window_12":  12,
}


def get_config(**overrides) -> dict:
    """
    Return a config dict that merges FORECAST_ADJUSTMENT_CONFIG with any
    caller-supplied overrides.

    Usage
    -----
        cfg = get_config()                          # pure defaults
        cfg = get_config(shrink_factor=0.9)         # override one key
        cfg = get_config(**my_dict)                 # override many keys

    Unknown keys in overrides are silently accepted (forward-compatible).
    """
    cfg = dict(FORECAST_ADJUSTMENT_CONFIG)
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trailing_mean(sku_hist: pd.Series, window: int) -> float:
    """Return the mean of the last *window* values from a sorted history series.
    Returns 0.0 if the series is empty or all-zero in the window."""
    if sku_hist is None or len(sku_hist) == 0:
        return 0.0
    tail = sku_hist.sort_index().tail(window)
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def _last_nonzero(sku_hist: pd.Series) -> float:
    """Return the last non-zero value in the history, or 0.0 if none."""
    if sku_hist is None or len(sku_hist) == 0:
        return 0.0
    nonzero = sku_hist[sku_hist > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(nonzero.iloc[-1])


# ---------------------------------------------------------------------------
# Adjustment 1 — Intermittent cap
# ---------------------------------------------------------------------------

def apply_intermittent_cap(
    yhat: list[float],
    sku_hist: pd.Series,
    config: dict,
) -> list[float]:
    """
    Cap INTERMITTENT predictions to prevent spike artefacts.

    The cap is:
        cap = max(
            cap_multiplier × trailing_sum_12_per_month,
            last_nonzero × 3
        )

    where trailing_sum_12_per_month = sum(last 12 months) / 12.

    Any prediction above the cap is clipped to the cap.
    Predictions of 0 are left at 0.

    Parameters
    ----------
    yhat     : list of float predictions (non-negative)
    sku_hist : pd.Series indexed by MonthStart, sorted ascending (training history)
    config   : adjustment config dict from get_config()

    Returns
    -------
    list[float] — capped predictions, same length as yhat
    """
    multiplier = float(config.get("intermittent_cap_multiplier", 2.0))
    if np.isinf(multiplier) or multiplier <= 0:
        return yhat

    window_12      = int(config.get("trailing_window_12", 12))
    trailing_sum   = float(sku_hist.sort_index().tail(window_12).sum()) if sku_hist is not None and len(sku_hist) > 0 else 0.0
    monthly_avg    = trailing_sum / max(1, window_12)
    last_nz        = _last_nonzero(sku_hist)

    cap = max(
        multiplier * monthly_avg,
        last_nz * 3.0,
        1.0,   # never cap below 1 unit (avoids capping everything to 0 when history is 0)
    )

    capped = [min(v, cap) if v > 0 else 0.0 for v in yhat]
    n_capped = sum(1 for orig, new in zip(yhat, capped) if new < orig)
    if n_capped > 0:
        logger.debug(
            "Intermittent cap applied: %d/%d steps capped at %.1f",
            n_capped, len(yhat), cap,
        )
    return capped


# ---------------------------------------------------------------------------
# Adjustment 2 — Shrinkage (anti-overprediction)
# ---------------------------------------------------------------------------

def apply_shrinkage(
    yhat: list[float],
    sku_hist: pd.Series,
    segment: str,
    config: dict,
) -> list[float]:
    """
    Shrink predictions toward recent trailing mean when they are too far above it.

    Trigger: forecast > shrink_threshold × trailing_mean_3
    Correction: final = shrink_factor × forecast + (1 − shrink_factor) × trailing_mean_3

    DEAD SKUs are exempt. If trailing_mean_3 == 0 (no recent sales), shrinkage
    is not applied (we don't want to suppress legitimate restarts).

    Parameters
    ----------
    yhat     : list of float predictions (non-negative)
    sku_hist : pd.Series indexed by MonthStart (training history only)
    segment  : "REGULAR", "INTERMITTENT", or "DEAD"
    config   : adjustment config dict from get_config()

    Returns
    -------
    list[float] — shrinkage-adjusted predictions
    """
    if segment == "DEAD":
        return yhat

    threshold    = float(config.get("shrink_threshold",  1.5))
    shrink_f     = float(config.get("shrink_factor",     0.8))
    window_3     = int(config.get("trailing_window_3",   3))

    if shrink_f >= 1.0:
        return yhat

    trail_3 = _trailing_mean(sku_hist, window_3)

    if trail_3 <= 0:
        # No recent trend to anchor against — skip shrinkage
        return yhat

    adjusted = []
    for v in yhat:
        if v > threshold * trail_3:
            new_v = shrink_f * v + (1.0 - shrink_f) * trail_3
            logger.debug(
                "Shrinkage: %.1f → %.1f (trail_3=%.1f, threshold=%.2f×)",
                v, new_v, trail_3, threshold,
            )
            adjusted.append(max(0.0, new_v))
        else:
            adjusted.append(v)
    return adjusted


# ---------------------------------------------------------------------------
# Adjustment 3 — ML vs Seasonal blend
# ---------------------------------------------------------------------------

def blend_ml_with_seasonal(
    yhat_ml: list[float],
    yhat_seasonal: list[float],
    config: dict,
) -> list[float]:
    """
    Blend ML predictions with a seasonal baseline when they diverge substantially.

    Trigger (per step): |ML - Seasonal| / max(1, Seasonal) > blend_threshold
    Correction:         final = blend_weight × ML + (1 − blend_weight) × Seasonal

    Steps where the models agree (within threshold) are left unchanged.

    Parameters
    ----------
    yhat_ml       : list of float ML predictions
    yhat_seasonal : list of float seasonal baseline predictions (same length)
    config        : adjustment config dict from get_config()

    Returns
    -------
    list[float] — blended predictions, same length as yhat_ml
    """
    threshold    = float(config.get("blend_threshold", 0.25))
    blend_w      = float(config.get("blend_weight",    0.7))

    if blend_w >= 1.0:
        return yhat_ml

    if len(yhat_ml) != len(yhat_seasonal):
        logger.warning(
            "blend_ml_with_seasonal: length mismatch ML=%d vs Seasonal=%d — skipping blend",
            len(yhat_ml), len(yhat_seasonal),
        )
        return yhat_ml

    blended = []
    n_blended = 0
    for ml_v, sea_v in zip(yhat_ml, yhat_seasonal):
        denom = max(1.0, sea_v)
        if abs(ml_v - sea_v) / denom > threshold:
            new_v = blend_w * ml_v + (1.0 - blend_w) * sea_v
            blended.append(max(0.0, new_v))
            n_blended += 1
        else:
            blended.append(ml_v)

    if n_blended > 0:
        logger.debug(
            "ML/Seasonal blend applied: %d/%d steps blended",
            n_blended, len(yhat_ml),
        )
    return blended


# ---------------------------------------------------------------------------
# Adjustment 4 — Recursive stabilization
# ---------------------------------------------------------------------------

def stabilize_lag_for_recursion(
    predicted_val: float,
    sku_hist_so_far: pd.Series,
    step_index: int,
    config: dict,
) -> float:
    """
    Blend the predicted lag value with the trailing mean before feeding it
    as lag_1 into the next recursive step.

    This reduces compounding drift in H=3 multi-step recursive forecasting
    by anchoring each step's lag_1 partly to the known recent trend.

    Only applied when step_index > 0 (the first step uses the last actual,
    not a predicted value — no stabilization needed).

    Formula:
        stabilized = alpha × predicted_val + (1 − alpha) × trailing_mean_3

    Parameters
    ----------
    predicted_val     : the predicted value to be appended to history
    sku_hist_so_far   : running history including all predictions up to this step
    step_index        : 0-based index of the current forecast step
    config            : adjustment config dict from get_config()

    Returns
    -------
    float — the stabilized value to append to running_hist
    """
    if step_index == 0:
        # First step: lag_1 is the last ACTUAL value, no blending needed
        return predicted_val

    alpha    = float(config.get("recursive_alpha", 0.7))
    window_3 = int(config.get("trailing_window_3",  3))

    if alpha >= 1.0:
        return predicted_val

    trail_3 = _trailing_mean(sku_hist_so_far, window_3)

    if trail_3 <= 0:
        # No trailing mean to blend with — use prediction as-is
        return predicted_val

    stabilized = alpha * predicted_val + (1.0 - alpha) * trail_3
    stabilized = max(0.0, stabilized)

    logger.debug(
        "Recursive stabilization step=%d: pred=%.1f trail=%.1f → stabilized=%.1f",
        step_index, predicted_val, trail_3, stabilized,
    )
    return stabilized
