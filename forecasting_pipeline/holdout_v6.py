"""
lane7_forecast.holdout_v6
==========================
Holdout evaluation for the v6 hierarchical pipeline.

Why a separate module
----------------------
The pre-computed v6_gold_fact_forecasts.csv starts at 2026-03-01 (it is the
forward planning file, not a backtesting file).  Evaluating holdout performance
on Jan–Feb 2026 therefore requires re-running the v6 forecast machinery with
forecast_start='2026-01-01', training exclusively on data ≤ 2025-12.

This module provides two public entry points:

    run_v6_holdout(data_dir, sc_prep, best_models_v6, output_dir, ...)
        Train the v6 hierarchical model on 2017-05–2025-12, predict Jan–Feb
        2026 at StyleColor level, allocate to SKU via size shares, join
        Jan–Feb 2026 actuals from gold_v2, compute per-row error metrics.

        Writes:
            v6_holdout_predictions.csv   (per-row SKU predictions vs actuals)
            v6_holdout_evaluation.csv    (aggregated WMAPE by Horizon × Month × Segment)

    compare_v6_vs_v52(v6_preds_df, v52_source, output_dir, ...)
        Align the v6 holdout predictions with v5.2 predictions on the same
        scoreable population, then compute side-by-side WMAPE metrics.

        v52_source may be:
            - outputs/holdout_predictions.csv  (if it covers Jan-Feb 2026; check
              MonthStart values before using — the v5.2 rebuild session wrote this
              with Jan-Feb 2026 actuals)
            - outputs/simulation_2026_predictions.csv  (preferred fallback — covers
              Jan-Apr 2026 at H=3 from run_simulation())

        Writes:
            v6_vs_v52_holdout_comparison.csv

Population rule for apples-to-apples comparison
-------------------------------------------------
"Scoreable" = a SKU that satisfies ALL of:
    (a) Has actual sales in Jan 2026 or Feb 2026 (appears in gold_v2 HOLDOUT)
    (b) Has training history ≤ 2025-12 (appears in the Phase-1 panel)
    (c) Has a v6 prediction for that month (not cold-start)
    (d) Has a v5.2 prediction for that month (present in the v5.2 source file)

This guarantees that the WMAPE comparison is not distorted by coverage
differences between models.

Public API
----------
    run_v6_holdout(data_dir, sc_prep, best_models_v6, output_dir,
                   size_shares_df=None, holdout_months=None) -> pd.DataFrame

    compare_v6_vs_v52(v6_preds_df, v52_source_path, output_dir,
                      horizon_months=None) -> pd.DataFrame
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

HOLDOUT_MONTHS_DEFAULT = [
    pd.Timestamp("2026-01-01"),
    pd.Timestamp("2026-02-01"),
]

_GOLD_V2 = "gold_fact_monthly_demand_v2.csv"
_GOLD_V1 = "gold_fact_monthly_demand.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_actuals(data_dir: Path, holdout_months: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Load Jan–Feb 2026 actuals from gold_fact_monthly_demand_v2.csv.
    Falls back to v1 if v2 is absent.
    """
    v2 = data_dir / _GOLD_V2
    v1 = data_dir / _GOLD_V1
    path = v2 if v2.exists() else v1
    if not path.exists():
        raise FileNotFoundError(f"No gold demand file found in {data_dir}")

    raw = pd.read_csv(path, parse_dates=["MonthStart"])
    raw["MonthStart"] = raw["MonthStart"].dt.to_period("M").dt.to_timestamp()
    raw["SKU"]        = raw["SKU"].astype(str).str.strip()

    actuals = raw[raw["MonthStart"].isin(holdout_months)][
        ["SKU", "MonthStart", "UnitsSold"]
    ].rename(columns={"UnitsSold": "ActualUnits"}).copy()

    logger.info(
        "Loaded actuals from %s: %d rows across %d SKUs in holdout window",
        path.name, len(actuals), actuals["SKU"].nunique(),
    )
    return actuals


def _wmape(actual: pd.Series, predicted: pd.Series) -> float:
    """WMAPE = sum|actual-pred| / sum(actual) × 100. Returns nan if sum(actual)==0."""
    total_actual = actual.sum()
    if total_actual <= 0:
        return np.nan
    return float((actual - predicted).abs().sum() / total_actual * 100)


# ---------------------------------------------------------------------------
# Step 1 — Generate v6 holdout predictions (Jan–Feb 2026)
# ---------------------------------------------------------------------------

def run_v6_holdout(
    data_dir: str | Path,
    sc_prep: dict,
    best_models_v6: pd.DataFrame,
    output_dir: str | Path,
    size_shares_df: pd.DataFrame | None = None,
    holdout_months: list[pd.Timestamp] | None = None,
    adjustment_config: dict | None = None,
) -> pd.DataFrame:
    """
    Train v6 on 2017-05–2025-12, predict Jan–Feb 2026 at StyleColor level,
    allocate to SKU, join actuals, and write evaluation files.

    Parameters
    ----------
    data_dir        : folder containing gold_fact_monthly_demand_v2.csv,
                      dim_date.csv, dim_product.csv
    sc_prep         : dict output of run_hierarchical_prep(phase=1).
                      Must contain keys: 'tables', 'panel_seg', 'segments',
                      'standalone_skus', 'size_shares', 'dim_product'.
    best_models_v6  : v6_best_models.csv as a DataFrame (BestModel per Segment × Horizon)
    output_dir      : folder to write v6_holdout_predictions.csv and
                      v6_holdout_evaluation.csv
    size_shares_df  : optional pre-computed size shares DataFrame.
                      If None, uses sc_prep['size_shares'].
    holdout_months  : list of pd.Timestamp to evaluate.
                      Defaults to [2026-01-01, 2026-02-01].
    adjustment_config : v6.2 forecast correction config dict from
                        forecast_adjustments.get_config(), or None to disable.
                        Pass an empty dict ({}) to use all v6.2 defaults.

    Returns
    -------
    pd.DataFrame — v6_holdout_predictions.csv contents (per-row SKU predictions
                   vs actuals). Saved to output_dir automatically.
    """
    # Inline import to avoid circular imports at module level
    from forecasting_pipeline.pipeline    import run_forecasts
    from forecasting_pipeline.allocation  import allocate_to_sku, get_standalone_skus
    from forecasting_pipeline.data_prep   import build_panel
    from forecasting_pipeline.segmentation import segment_skus, attach_segment

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_months = holdout_months or HOLDOUT_MONTHS_DEFAULT
    n_holdout      = len(holdout_months)

    logger.info(
        "=== v6 holdout evaluation: training on phase=1 panel, predicting %d months ===",
        n_holdout,
    )

    # ── 1. Extract key objects from sc_prep ───────────────────────────────────
    tables       = sc_prep["tables"]
    dp           = sc_prep.get("dim_product")
    if dp is None:
        raise ValueError("sc_prep must contain 'dim_product' (set by run_hierarchical_prep)")

    standalone_skus = sc_prep.get("standalone_skus", [])
    shares          = size_shares_df if size_shares_df is not None else sc_prep.get("size_shares", pd.DataFrame())
    if shares.empty:
        raise ValueError(
            "size_shares_df is empty and sc_prep['size_shares'] is empty. "
            "Run run_hierarchical_prep() first or pass size_shares_df explicitly."
        )

    # ── 2. StyleColor-level holdout forecasts ─────────────────────────────────
    # forecast_start = first holdout month; n_forecast_months = 2 (Jan + Feb 2026)
    forecast_start = holdout_months[0].strftime("%Y-%m-%d")

    sc_fc_all   = []
    sa_fc_all   = []

    # Build STANDALONE panel once
    gold_raw  = tables["demand"]
    sa_gold   = gold_raw[gold_raw["SKU"].isin(standalone_skus)].copy()
    dim_date  = tables["dim_date"]

    sa_prep = None
    if standalone_skus and not sa_gold.empty:
        sa_panel     = build_panel(sa_gold, dim_date, phase=1, dim_product_df=dp)
        sa_segments  = segment_skus(sa_panel)
        sa_panel_seg = attach_segment(sa_panel, sa_segments)
        sa_prep = {
            "tables":    tables,
            "panel":     sa_panel,
            "segments":  sa_segments,
            "panel_seg": sa_panel_seg,
        }

    for horizon in sorted(best_models_v6["HorizonMonths"].unique()):
        bm_h = best_models_v6[best_models_v6["HorizonMonths"] == horizon]
        if bm_h.empty:
            continue

        # Number of holdout months to predict.
        # H=1 → predict exactly 1 step (Jan 2026, scoreable)
        # H=3 → predict 2 steps (Jan + Feb 2026, both scoreable with H=3 lags)
        # For H=3, the model uses Dec 2025 lag_1 for Jan, and Jan prediction
        # (via recursive step) for Feb — matching the production behaviour.
        n_predict = min(horizon, n_holdout)

        logger.info("v6 holdout: generating H=%d forecast (%d months from %s)",
                    horizon, n_predict, forecast_start)

        # StyleColor-level forecast
        sc_fc = run_forecasts(
            prep=sc_prep,
            best_models_df=bm_h,
            horizon_months=horizon,
            forecast_start=forecast_start,
            n_forecast_months=n_predict,
            phase=1,
            model_version=f"v6-holdout-H{horizon}",
            output_path=None,
            append=False,
            adjustment_config=adjustment_config,
        )
        sc_fc_all.append(sc_fc)

        # STANDALONE SKU forecast
        if sa_prep is not None:
            sa_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=bm_h,
                horizon_months=horizon,
                forecast_start=forecast_start,
                n_forecast_months=n_predict,
                phase=1,
                model_version=f"v6-holdout-standalone-H{horizon}",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )
        else:
            sa_fc = pd.DataFrame()
        sa_fc_all.append(sa_fc)

    if not sc_fc_all:
        logger.warning("No holdout forecasts generated — best_models_v6 may be empty.")
        return pd.DataFrame()

    # ── 3. Allocate SC → SKU for each horizon ────────────────────────────────
    sku_fc_parts = []
    for sc_fc, sa_fc in zip(sc_fc_all, sa_fc_all):
        sku_fc = allocate_to_sku(
            stylecolor_forecasts_df=sc_fc,
            size_shares_df=shares,
            dim_product_df=dp,
            standalone_sku_forecasts_df=sa_fc if not sa_fc.empty else None,
        )
        sku_fc_parts.append(sku_fc)

    sku_combined = pd.concat(sku_fc_parts, ignore_index=True)
    # Normalise Key → SKU for downstream joins
    sku_combined = sku_combined.rename(columns={"Key": "SKU"})
    sku_combined["MonthStart"] = pd.to_datetime(sku_combined["MonthStart"])

    logger.info(
        "v6 holdout allocated: %d rows, %d SKUs",
        len(sku_combined), sku_combined["SKU"].nunique(),
    )

    # ── 4. Attach segment labels ──────────────────────────────────────────────
    # SC-level segments are in sc_prep['segments'] (SKU col = StyleColorDesc).
    # Map StyleColorDesc → SC segment for hierarchical SKUs;
    # STANDALONE SKUs carry their own segment from sa_prep['segments'].
    sc_seg_map = (
        sc_prep["segments"][["SKU", "Segment"]]
        .rename(columns={"SKU": "StyleColorDesc", "Segment": "SC_Segment"})
    )

    # Start from the StyleColorDesc column already in sku_combined
    if "StyleColorDesc" in sku_combined.columns:
        sku_combined = sku_combined.merge(sc_seg_map, on="StyleColorDesc", how="left")
        # For STANDALONE rows (StyleColorDesc=='STANDALONE'), fill from sa_prep if available
        if sa_prep is not None and not sa_prep["segments"].empty:
            sa_seg_map = (
                sa_prep["segments"][["SKU", "Segment"]]
                .rename(columns={"Segment": "SA_Segment"})
            )
            sku_combined = sku_combined.merge(sa_seg_map, on="SKU", how="left")
            # STANDALONE rows: use SA_Segment; others: use SC_Segment
            standalone_mask = sku_combined["StyleColorDesc"] == "STANDALONE"
            sku_combined["Segment"] = np.where(
                standalone_mask,
                sku_combined["SA_Segment"].fillna("DEAD"),
                sku_combined["SC_Segment"].fillna("DEAD"),
            )
            sku_combined = sku_combined.drop(columns=["SC_Segment","SA_Segment"], errors="ignore")
        else:
            sku_combined = sku_combined.rename(columns={"SC_Segment": "Segment"})
            sku_combined["Segment"] = sku_combined["Segment"].fillna("DEAD")
    else:
        sku_combined["Segment"] = "UNKNOWN"

    # ── 5. Join actuals ───────────────────────────────────────────────────────
    actuals = _load_actuals(data_dir, holdout_months)
    scored  = sku_combined.merge(
        actuals, on=["SKU", "MonthStart"], how="inner"
    )
    logger.info(
        "Joined actuals: %d scoreable (SKU, MonthStart) pairs across %d SKUs",
        len(scored), scored["SKU"].nunique(),
    )

    if scored.empty:
        logger.warning("No matched rows after joining actuals. Check holdout_months alignment.")
        return pd.DataFrame()

    # ── 6. Compute per-row error metrics ──────────────────────────────────────
    scored["PredictedUnits"] = scored["ForecastUnits"].clip(lower=0).fillna(0)
    scored["Error"]          = scored["ActualUnits"]   - scored["PredictedUnits"]
    scored["AbsError"]       = scored["Error"].abs()
    scored["AbsPctError"]    = np.where(
        scored["ActualUnits"] > 0,
        scored["AbsError"] / scored["ActualUnits"] * 100,
        np.nan,
    )

    # ── 7. Build per-row output ───────────────────────────────────────────────
    pred_cols = [
        "SKU", "MonthStart", "Segment", "HorizonMonths",
        "PredictedUnits", "ActualUnits",
        "Error", "AbsError", "AbsPctError",
        "ModelName", "ModelVersion",
        "StyleColorDesc", "SizeDesc",
    ]
    pred_cols = [c for c in pred_cols if c in scored.columns]
    predictions_df = (
        scored[pred_cols]
        .sort_values(["HorizonMonths", "SKU", "MonthStart"])
        .reset_index(drop=True)
    )

    pred_path = output_dir / "v6_holdout_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info("Saved %s (%d rows)", pred_path.name, len(predictions_df))

    # ── 8. Aggregated evaluation table ────────────────────────────────────────
    eval_rows = []
    for horizon in predictions_df["HorizonMonths"].unique():
        h_df = predictions_df[predictions_df["HorizonMonths"] == horizon]
        for month in sorted(h_df["MonthStart"].unique()):
            m_df = h_df[h_df["MonthStart"] == month]
            for seg in sorted(m_df["Segment"].unique()):
                s_df = m_df[m_df["Segment"] == seg]
                if s_df.empty:
                    continue
                eval_rows.append({
                    "Level":          "SKU",
                    "HorizonMonths":  int(horizon),
                    "MonthStart":     pd.Timestamp(month).strftime("%Y-%m"),
                    "Segment":        seg,
                    "N_SKUs":         s_df["SKU"].nunique(),
                    "TotalActual":    round(s_df["ActualUnits"].sum(), 2),
                    "TotalPredicted": round(s_df["PredictedUnits"].sum(), 2),
                    "AbsError":       round(s_df["AbsError"].sum(), 2),
                    "WMAPE":          round(_wmape(s_df["ActualUnits"], s_df["PredictedUnits"]), 4),
                })
        # Overall (all segments) per horizon × month
        for month in sorted(h_df["MonthStart"].unique()):
            m_df = h_df[h_df["MonthStart"] == month]
            eval_rows.append({
                "Level":          "SKU",
                "HorizonMonths":  int(horizon),
                "MonthStart":     pd.Timestamp(month).strftime("%Y-%m"),
                "Segment":        "ALL",
                "N_SKUs":         m_df["SKU"].nunique(),
                "TotalActual":    round(m_df["ActualUnits"].sum(), 2),
                "TotalPredicted": round(m_df["PredictedUnits"].sum(), 2),
                "AbsError":       round(m_df["AbsError"].sum(), 2),
                "WMAPE":          round(_wmape(m_df["ActualUnits"], m_df["PredictedUnits"]), 4),
            })

    eval_df = pd.DataFrame(eval_rows).sort_values(
        ["HorizonMonths", "MonthStart", "Segment"]
    ).reset_index(drop=True)

    eval_path = output_dir / "v6_holdout_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)
    logger.info("Saved %s (%d rows)", eval_path.name, len(eval_df))

    # Quick print summary
    print("\n=== v6 HOLDOUT EVALUATION — Jan–Feb 2026 (final SKU level) ===")
    print(eval_df[eval_df["Segment"] == "ALL"][
        ["HorizonMonths", "MonthStart", "N_SKUs",
         "TotalActual", "TotalPredicted", "WMAPE"]
    ].to_string(index=False))

    return predictions_df


# ---------------------------------------------------------------------------
# Step 2 — Apples-to-apples comparison: v6 vs v5.2
# ---------------------------------------------------------------------------

def compare_v6_vs_v52(
    v6_preds_df: pd.DataFrame,
    v52_source_path: str | Path,
    output_dir: str | Path,
    horizon_months: list[int] | None = None,
    holdout_months: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    Compare v6 vs v5.2 on the same scoreable SKU population.

    Scoreable population rule (all four conditions must hold):
        (a) SKU has actuals in gold_v2 for the relevant month
        (b) SKU has a v6 holdout prediction for that month
        (c) SKU has a v5.2 prediction for that month
        (d) The month is in the HOLDOUT window (Jan–Feb 2026)

    v5.2 source mapping
    -------------------
    Two possible source files:

    1. outputs/holdout_predictions.csv
       Schema: SKU, MonthStart, Actual, Predicted, Segment, ModelName, HorizonMonths
       Coverage: depends on when this was generated.
           - If generated by the v5.2 rebuild session (this project), it covers
             Jan-Feb 2026 with columns Actual/Predicted.
           - If generated by the original v4/v5.1 notebook, it covers 2025 actuals only.
       Detection: if MonthStart values include 2026-01 or 2026-02, use this file.

    2. outputs/simulation_2026_predictions.csv
       Schema: SKU, MonthStart, Segment, ModelName, HorizonMonths,
               PredictedUnits, ActualUnits, HasActual, Error, AbsError, ...
       Coverage: Jan–Apr 2026 at H=3 only.
       This is the preferred fallback for H=3.

    Parameters
    ----------
    v6_preds_df      : output of run_v6_holdout() — per-row v6 predictions
    v52_source_path  : path to holdout_predictions.csv or
                       simulation_2026_predictions.csv
    output_dir       : folder to write v6_vs_v52_holdout_comparison.csv
    horizon_months   : list of horizons to compare (default [1, 3])
    holdout_months   : list of pd.Timestamps to compare (default Jan–Feb 2026)

    Returns
    -------
    pd.DataFrame — v6_vs_v52_holdout_comparison.csv contents
    """
    output_dir     = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v52_source_path = Path(v52_source_path)
    horizon_months  = horizon_months  or [1, 3]
    holdout_months  = holdout_months  or HOLDOUT_MONTHS_DEFAULT

    if not v52_source_path.exists():
        raise FileNotFoundError(
            f"v5.2 source file not found: {v52_source_path}\n"
            f"Expected one of:\n"
            f"  outputs/holdout_predictions.csv\n"
            f"  outputs/simulation_2026_predictions.csv"
        )

    # ── Load v5.2 source ─────────────────────────────────────────────────────
    v52_raw = pd.read_csv(v52_source_path, parse_dates=["MonthStart"])
    v52_raw["MonthStart"] = v52_raw["MonthStart"].dt.to_period("M").dt.to_timestamp()
    v52_raw["SKU"] = v52_raw["SKU"].astype(str).str.strip()

    # Detect schema variant
    if "Predicted" in v52_raw.columns and "Actual" in v52_raw.columns:
        # holdout_predictions.csv schema
        v52_raw = v52_raw.rename(columns={
            "Predicted": "PredictedUnits",
            "Actual":    "ActualUnits",
        })
        logger.info("v5.2 source: holdout_predictions.csv schema detected")
    elif "PredictedUnits" in v52_raw.columns and "ActualUnits" in v52_raw.columns:
        # simulation_2026_predictions.csv schema — filter to HasActual rows only
        if "HasActual" in v52_raw.columns:
            if v52_raw["HasActual"].dtype == object:
                v52_raw = v52_raw[v52_raw["HasActual"].astype(str).str.lower() == "true"].copy()
            else:
                v52_raw = v52_raw[v52_raw["HasActual"].astype(bool)].copy()
        logger.info("v5.2 source: simulation_2026_predictions.csv schema detected")
    else:
        raise ValueError(
            f"Cannot recognise schema of {v52_source_path.name}. "
            "Expected columns 'Predicted'+'Actual' or 'PredictedUnits'+'ActualUnits'."
        )

    # Filter to holdout months only
    v52_raw = v52_raw[v52_raw["MonthStart"].isin(holdout_months)].copy()
    logger.info(
        "v5.2 source after filtering to holdout months: %d rows, %d SKUs",
        len(v52_raw), v52_raw["SKU"].nunique(),
    )

    # Prepare v6 predictions
    v6 = v6_preds_df.copy()
    v6["MonthStart"] = pd.to_datetime(v6["MonthStart"])
    v6 = v6[v6["MonthStart"].isin(holdout_months)].copy()

    comparison_rows = []

    for horizon in horizon_months:
        # v5.2 rows for this horizon
        v52_h = v52_raw[v52_raw["HorizonMonths"] == horizon].copy() if "HorizonMonths" in v52_raw.columns else v52_raw.copy()
        # If simulation file (H=3 only) and we're asking for H=1, skip
        if "HorizonMonths" in v52_raw.columns and horizon not in v52_raw["HorizonMonths"].values:
            logger.info("v5.2 source has no H=%d rows — skipping horizon comparison", horizon)
            continue

        # v6 rows for this horizon
        v6_h = v6[v6["HorizonMonths"] == horizon].copy()

        if v6_h.empty:
            logger.info("v6 has no H=%d holdout predictions — skipping", horizon)
            continue

        for month in sorted(holdout_months):
            v52_m = v52_h[v52_h["MonthStart"] == month]
            v6_m  = v6_h[v6_h["MonthStart"] == month]

            if v52_m.empty or v6_m.empty:
                continue

            # ── Scoreable population: intersection of SKUs present in BOTH ──
            # Rule:
            #   (a) SKU in v6 predictions for this (horizon, month) AND
            #   (b) SKU in v5.2 predictions for this (horizon, month) AND
            #   (c) Both have non-null ActualUnits (both were scored against real actuals)
            v52_skus = set(v52_m.dropna(subset=["ActualUnits"])["SKU"].unique())
            v6_skus  = set(v6_m.dropna(subset=["ActualUnits"])["SKU"].unique())
            common   = v52_skus & v6_skus

            if not common:
                logger.warning(
                    "H=%d month=%s: no common scoreable SKUs between v5.2 (%d SKUs) "
                    "and v6 (%d SKUs) — skipping.",
                    horizon, month.strftime("%Y-%m"), len(v52_skus), len(v6_skus),
                )
                continue

            # Restrict both to common SKUs
            v52_c = v52_m[v52_m["SKU"].isin(common)].copy()
            v6_c  = v6_m[v6_m["SKU"].isin(common)].copy()

            v52_actual    = v52_c.set_index("SKU")["ActualUnits"]
            v52_predicted = v52_c.set_index("SKU")["PredictedUnits"]
            v6_actual     = v6_c.set_index("SKU")["ActualUnits"]
            v6_predicted  = v6_c.set_index("SKU")["PredictedUnits"]

            # Align on SKU order
            shared_skus = sorted(common)
            v52_a = v52_actual.reindex(shared_skus).fillna(0)
            v52_p = v52_predicted.reindex(shared_skus).fillna(0)
            v6_a  = v6_actual.reindex(shared_skus).fillna(0)
            v6_p  = v6_predicted.reindex(shared_skus).fillna(0)

            # Use v6 actual for both (they come from the same gold source,
            # but v6_a is the reference since v6 was evaluated against gold_v2)
            common_actual = v6_a

            v52_wmape = _wmape(common_actual, v52_p)
            v6_wmape  = _wmape(common_actual, v6_p)
            delta     = round(v6_wmape - v52_wmape, 4) if not (np.isnan(v52_wmape) or np.isnan(v6_wmape)) else np.nan

            comparison_rows.append({
                "HorizonMonths":    int(horizon),
                "MonthStart":       month.strftime("%Y-%m"),
                # Population rule documented in the column so the CSV is self-explanatory
                "PopulationRule":   "SKUs with actuals in both v5.2 and v6 predictions",
                "N_SKUs":           len(shared_skus),
                "v52_TotalActual":  round(common_actual.sum(), 2),
                "v52_TotalPredicted": round(v52_p.sum(), 2),
                "v52_WMAPE":        round(v52_wmape, 4) if not np.isnan(v52_wmape) else None,
                "v6_TotalActual":   round(common_actual.sum(), 2),
                "v6_TotalPredicted": round(v6_p.sum(), 2),
                "v6_WMAPE":         round(v6_wmape, 4) if not np.isnan(v6_wmape) else None,
                # WMAPE_Delta < 0 means v6 improved on v5.2
                "WMAPE_Delta":      round(delta, 4) if not np.isnan(delta) else None,
            })

    if not comparison_rows:
        logger.warning("No comparison rows generated. Check that both v5.2 and v6 cover the same holdout months and horizons.")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["HorizonMonths", "MonthStart"]
    ).reset_index(drop=True)

    out_path = output_dir / "v6_vs_v52_holdout_comparison.csv"
    comparison_df.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path.name, len(comparison_df))

    print("\n=== v5.2 vs v6 HOLDOUT COMPARISON (same scoreable population) ===")
    print("(WMAPE_Delta < 0 means v6 improved on v5.2)")
    print(comparison_df[[
        "HorizonMonths", "MonthStart", "N_SKUs",
        "v52_WMAPE", "v6_WMAPE", "WMAPE_Delta",
    ]].to_string(index=False))

    return comparison_df
