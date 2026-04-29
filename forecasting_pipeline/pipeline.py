"""
lane7_forecast.pipeline
========================
Top-level orchestration — wires all modules together in the correct order.

Three-step usage
----------------
    from lane7_forecast.pipeline import run_data_prep, run_cv, run_forecasts

    prep = run_data_prep(data_dir="data/", phase=1)

    cv_results, best_models = run_cv(prep, horizon_months=1)
    cv_results_3,  best_3  = run_cv(prep, horizon_months=3)
    cv_results_12, best_12 = run_cv(prep, horizon_months=12)

    forecasts = run_forecasts(
        prep, best_models,
        horizon_months=1,
        forecast_start="2026-01-01",
        output_path="outputs/gold_fact_forecasts.csv",
    )

Forecasting goals
-----------------
  Horizon 1  → 1-month-ahead  (short-term,  operational)
  Horizon 3  → 3-month-ahead  (mid-term,    quarterly planning)
  Horizon 12 → 12-month-ahead (long-term,   annual / budget)

Segmentation is applied BEFORE modeling so each segment gets the
model class best suited to its demand pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_pipeline.data_prep    import load_gold_tables, build_panel
from forecasting_pipeline.segmentation import segment_skus, attach_segment
from forecasting_pipeline.features     import create_features, get_feature_columns
from forecasting_pipeline.evaluation   import walk_forward_cv, select_best_model
from forecasting_pipeline.models       import SEGMENT_MODEL_ROSTER, train_ml_model
from forecasting_pipeline.forecasting  import generate_forecasts, write_forecasts, append_forecasts

logger = logging.getLogger(__name__)

ML_MODELS = {"XGBoost", "LightGBM", "RandomForest"}

# Default CV config per horizon (can be overridden in run_cv)
_CV_DEFAULTS: dict[int, dict] = {
    1:  {"n_folds": 6, "step_months": 3,  "min_train_months": 24},
    3:  {"n_folds": 4, "step_months": 3,  "min_train_months": 24},
    12: {"n_folds": 3, "step_months": 12, "min_train_months": 36},
}


# ---------------------------------------------------------------------------
# Step 1 — Data preparation
# ---------------------------------------------------------------------------

def run_data_prep(
    data_dir: str | Path = "data/",
    phase: int = 1,
    trailing_months: int = 12,
    zero_ratio_threshold: float = 0.40,
) -> dict:
    """
    Load Gold tables, build the SKU × Month panel, and segment all SKUs.

    Parameters
    ----------
    data_dir              : folder containing the Gold CSV files
    phase                 : 1 → train 2017-05 through 2025-12, evaluate on
                                2026-01/02 HOLDOUT (used for CV + holdout scoring)
                            2 → train 2017-05 through 2026-02 (HOLDOUT folded in),
                                forecast 2026-03 onwards (production refit)
    trailing_months       : look-back window (months) for DEAD classification
    zero_ratio_threshold  : fraction of zero-sales months above which a SKU
                            is classified INTERMITTENT (default 0.40 = 40 %)

    Returns dict with keys
    ----------------------
    "tables"     → raw loaded DataFrames
    "panel"      → clean monthly panel, zero-filled per-SKU from first sale
    "segments"   → one row per SKU with Segment label
    "panel_seg"  → panel with Segment column attached (primary working table)
    """
    logger.info("=== Step 1: Data preparation (phase=%d) ===", phase)

    tables = load_gold_tables(data_dir)

    panel = build_panel(
        demand_df=tables["demand"],
        dim_date_df=tables["dim_date"],
        phase=phase,
        dim_product_df=tables.get("dim_product"),
    )

    segments = segment_skus(
        panel,
        trailing_months=trailing_months,
        zero_ratio_threshold=zero_ratio_threshold,
    )

    panel_seg = attach_segment(panel, segments)

    logger.info(
        "Data prep complete — %d rows, %d SKUs | REGULAR %d  INTERMITTENT %d  DEAD %d",
        len(panel_seg),
        panel_seg["SKU"].nunique(),
        (segments["Segment"] == "REGULAR").sum(),
        (segments["Segment"] == "INTERMITTENT").sum(),
        (segments["Segment"] == "DEAD").sum(),
    )
    return {
        "tables":    tables,
        "panel":     panel,
        "segments":  segments,
        "panel_seg": panel_seg,
    }


# ---------------------------------------------------------------------------
# Step 2 — Walk-forward cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    prep: dict,
    horizon_months: int,
    n_folds: int | None = None,
    step_months: int | None = None,
    min_train_months: int | None = None,
    segments_to_run: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward CV for one horizon across all (or selected) segments.

    Segmentation is already embedded in prep["panel_seg"]; models in
    SEGMENT_MODEL_ROSTER are chosen per (segment, horizon) automatically.

    Parameters
    ----------
    prep             : output of run_data_prep()
    horizon_months   : 1, 3, or 12
    n_folds          : CV folds   (default: 6 / 4 / 3 for h=1 / 3 / 12)
    step_months      : fold-advance step (default: 3 / 3 / 12)
    min_train_months : minimum training months before first fold
                       (default: 24 / 24 / 36)
    segments_to_run  : subset of ["REGULAR","INTERMITTENT","DEAD"]
                       (default: all three)

    Returns
    -------
    cv_results_df  : fold-level metrics for every model × segment × fold
    best_models_df : one row per (segment, horizon) — lowest mean WMAPE wins
    """
    logger.info("=== Step 2: CV for horizon=%d months ===", horizon_months)

    defaults = _CV_DEFAULTS.get(horizon_months, {"n_folds": 4, "step_months": horizon_months, "min_train_months": 24})
    n_folds          = n_folds          if n_folds          is not None else defaults["n_folds"]
    step_months      = step_months      if step_months      is not None else defaults["step_months"]
    min_train_months = min_train_months if min_train_months is not None else defaults["min_train_months"]

    panel_seg = prep["panel_seg"]
    segments  = segments_to_run or ["REGULAR", "INTERMITTENT", "DEAD"]

    # Build features once for this horizon (shared across all segments)
    feature_panel = create_features(panel_seg, horizon_months, drop_null_required=True)

    all_cv_rows: list[pd.DataFrame] = []

    for segment in segments:
        # DEAD SKUs always get ZeroForecast — no CV needed.
        # Their "target" months fall outside the training panel so
        # walk_forward_cv would always return empty.
        if segment == "DEAD":
            dead_row = pd.DataFrame([{
                "ModelName":     "ZeroForecast",
                "HorizonMonths": horizon_months,
                "Segment":       "DEAD",
                "Fold":          0,
                "Origin":        pd.NaT,
                "N_SKUs":        (panel_seg["Segment"] == "DEAD").sum(),
                "N_Obs":         0,
                "RMSE":          0.0,
                "MAE":           0.0,
                "MAPE":          np.nan,
                "WMAPE":         np.nan,
            }])
            all_cv_rows.append(dead_row)
            continue

        roster = SEGMENT_MODEL_ROSTER.get((segment, horizon_months), [])
        if not roster:
            logger.info("No models defined for (%s, %d) — skipping", segment, horizon_months)
            continue

        cv_df = walk_forward_cv(
            panel=panel_seg,
            feature_panel=feature_panel,
            horizon_months=horizon_months,
            segment=segment,
            model_names=roster,
            n_folds=n_folds,
            step_months=step_months,
            min_train_months=min_train_months,
        )
        if not cv_df.empty:
            all_cv_rows.append(cv_df)

    if not all_cv_rows:
        logger.warning("CV produced no results for horizon=%d", horizon_months)
        return pd.DataFrame(), pd.DataFrame()

    cv_results  = pd.concat(all_cv_rows, ignore_index=True)
    best_models = select_best_model(cv_results)

    return cv_results, best_models


# ---------------------------------------------------------------------------
# Step 3 — Final forecast generation
# ---------------------------------------------------------------------------

def run_forecasts(
    prep: dict,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-01-01",
    n_forecast_months: int = 24,
    phase: int = 1,
    model_version: str = "v1.0",
    output_path: str | Path | None = None,
    append: bool = True,
    adjustment_config: dict | None = None,
) -> pd.DataFrame:
    """
    Retrain the winning model(s) on the full training set, then generate
    monthly demand forecasts for every SKU for *n_forecast_months* months.

    Parameters
    ----------
    prep               : output of run_data_prep()
    best_models_df     : output of run_cv() — BestModel per (segment, horizon)
    horizon_months     : 1, 3, or 12
    forecast_start     : first forecast month, e.g. "2026-01-01"
    n_forecast_months  : months to generate (default 24 = 2026 + 2027)
    phase              : 1 or 2 (recorded in ModelVersion column)
    model_version      : string tag written into gold_fact_forecasts
    output_path        : if provided, write / append forecasts CSV here
    append             : True → append to existing file; False → overwrite
    adjustment_config  : v6.2 forecast correction config dict from
                         forecast_adjustments.get_config(), or None to disable
                         all v6.2 adjustments (reverts to v6.1 behaviour).
                         Pass an empty dict ({}) to use all v6.2 defaults.

    Returns
    -------
    pd.DataFrame matching the gold_fact_forecasts schema
    """
    logger.info(
        "=== Step 3: Forecast generation (horizon=%d, phase=%d) ===",
        horizon_months, phase,
    )

    panel_seg     = prep["panel_seg"]
    feature_panel = create_features(panel_seg, horizon_months, drop_null_required=True)
    feature_cols  = get_feature_columns(horizon_months, feature_panel)

    # Retrain ML models on full training data for each segment that needs one
    ml_models_fitted: dict[str, object] = {}

    for segment in ["REGULAR", "INTERMITTENT"]:
        match = best_models_df[
            (best_models_df["Segment"] == segment) &
            (best_models_df["HorizonMonths"] == horizon_months)
        ]
        if match.empty:
            continue
        best_model_name = match.iloc[0]["BestModel"]

        if best_model_name in ML_MODELS:
            seg_skus   = panel_seg[panel_seg["Segment"] == segment]["SKU"].unique()
            seg_feat   = feature_panel[feature_panel["SKU"].isin(seg_skus)]
            # Training rows = all months where IsTrain==1 (dim_date is authoritative).
            # v2 rebuild note: the legacy TRAIN_V4 fallback has been removed
            # because that split label no longer exists in dim_date. For phase=2
            # the build_panel step includes HOLDOUT months in the panel with
            # IsTrain=0, so we also include Split==HOLDOUT rows as training data
            # (phase=2 is the "fold HOLDOUT into training" production refit).
            if phase == 2 and "Split" in seg_feat.columns:
                _is_train_mask = (
                    (seg_feat["IsTrain"] == 1) |
                    (seg_feat["Split"] == "HOLDOUT")
                )
            else:
                _is_train_mask = (seg_feat["IsTrain"] == 1)
            train_feat = seg_feat[_is_train_mask]
            X_train    = train_feat[feature_cols].fillna(0)
            y_train    = train_feat["UnitsSold"].values

            if len(X_train) > 0:
                fitted = train_ml_model(best_model_name, X_train, y_train)
                ml_models_fitted[segment] = fitted
                logger.info(
                    "Retrained %s for segment=%s on %d rows",
                    best_model_name, segment, len(X_train),
                )

    forecasts_df = generate_forecasts(
        panel=panel_seg,
        feature_panel=feature_panel,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=model_version,
        ml_models_fitted=ml_models_fitted,
        adjustment_config=adjustment_config,
    )

    if output_path is not None:
        if append:
            append_forecasts(forecasts_df, output_path)
        else:
            write_forecasts(forecasts_df, output_path)

    return forecasts_df

# ---------------------------------------------------------------------------
# Step 4 — v5 Simulation: train through 2025-12, predict Jan-Apr 2026
# ---------------------------------------------------------------------------

def run_simulation(
    data_dir: str | Path,
    best_models_df: pd.DataFrame,
    horizon_months: int = 3,
    sim_start: str = "2026-01-01",
    n_sim_months: int = 4,
    demand_actuals_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_version: str = "v5.0-sim",
) -> pd.DataFrame:
    """
    v5 simulation: train on all actuals through 2025-12, generate predictions
    for Jan-Apr 2026, then join available actuals to compute per-row errors.

    This is the primary validation framework for v5. It answers:
    "How well would our model have done if we had ordered in Dec 2025 for Q1 2026?"

    Parameters
    ----------
    data_dir          : folder containing Gold CSV files
    best_models_df    : output of run_cv() — BestModel per (segment, horizon)
    horizon_months    : 1 or 3 (3 is primary for v5)
    sim_start         : first month to predict (default "2026-01-01")
    n_sim_months      : how many months to predict (default 4 = Jan-Apr 2026)
    demand_actuals_path : path to gold_fact_monthly_demand.csv (for joining actuals).
                         If None, uses data_dir/gold_fact_monthly_demand.csv
    output_path       : if set, saves simulation_2026_predictions.csv here
    model_version     : tag written into the output

    Returns
    -------
    pd.DataFrame with columns:
        SKU, MonthStart, Segment, ModelName, HorizonMonths,
        PredictedUnits, ActualUnits, HasActual,
        Error, AbsError, AbsPctError,
        Conservative, Middle, Aggressive,
        ModelVersion

    Error = Actual - Predicted  (positive = under-forecast, negative = over-forecast)
    AbsPctError = |Error| / max(Actual, 1) * 100
    """
    logger.info("=== v5 Simulation: train 2025-12, predict %s + %d months ===",
                sim_start, n_sim_months)

    data_dir = Path(data_dir)

    # ── 1. Load tables and build Phase 1 panel (train through 2025-12) ─────────
    tables = load_gold_tables(data_dir)
    # Phase 1 trains through 2025-12 (last IsTrain==1 month in dim_date).
    # Jan-Feb 2026 (HOLDOUT) are excluded from training and will be predicted
    # as true out-of-sample, then joined back as actuals for scoring below.
    panel = build_panel(
        tables["demand"], tables["dim_date"], phase=1,
        dim_product_df=tables.get("dim_product"),
    )
    segments = segment_skus(panel)
    panel_seg = attach_segment(panel, segments)

    logger.info("Simulation panel: %d rows, %d SKUs, through %s",
                len(panel_seg), panel_seg["SKU"].nunique(),
                panel_seg["MonthStart"].max().strftime("%Y-%m"))

    # ── 2. Generate predictions ───────────────────────────────────────────────
    sim_forecasts = run_forecasts(
        prep={
            "tables":    tables,
            "panel":     panel,
            "segments":  segments,
            "panel_seg": panel_seg,
        },
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=sim_start,
        n_forecast_months=n_sim_months,
        phase=1,
        model_version=model_version,
        output_path=None,   # don't write intermediate
        append=False,
    )

    if sim_forecasts.empty:
        logger.warning("Simulation produced no forecasts.")
        return pd.DataFrame()

    # Rename for clarity
    sim_df = sim_forecasts.rename(columns={
        "Key": "SKU",
        "ForecastUnits": "PredictedUnits",
        "Lower": "Conservative_raw",
        "Upper": "Aggressive_raw",
    })[["SKU", "MonthStart", "ModelName", "HorizonMonths",
        "PredictedUnits", "Conservative_raw", "Aggressive_raw", "ModelVersion"]].copy()

    # Round to whole units for planning readability
    sim_df["PredictedUnits"]  = sim_df["PredictedUnits"].round(0)
    sim_df["Conservative_raw"] = sim_df["Conservative_raw"].round(0)
    sim_df["Aggressive_raw"]   = sim_df["Aggressive_raw"].round(0)

    # Rename scenario columns
    sim_df = sim_df.rename(columns={
        "Conservative_raw": "Conservative",
        "Aggressive_raw":   "Aggressive",
    })
    sim_df["Middle"] = sim_df["PredictedUnits"]

    # Add segment label
    seg_map = segments[["SKU", "Segment"]].set_index("SKU")["Segment"].to_dict()
    sim_df["Segment"] = sim_df["SKU"].map(seg_map).fillna("UNKNOWN")

    # ── 3. Join actuals where available ──────────────────────────────────────
    # Prefer v2 gold (the trusted source per the v2 rebuild); fall back to v1
    # only if v2 is absent. Same resolution order as data_prep._resolve_demand_path.
    if demand_actuals_path is None:
        v2_path = data_dir / "gold_fact_monthly_demand_v2.csv"
        v1_path = data_dir / "gold_fact_monthly_demand.csv"
        actual_path = v2_path if v2_path.exists() else v1_path
    else:
        actual_path = Path(demand_actuals_path)
    try:
        actuals_raw = pd.read_csv(actual_path, parse_dates=["MonthStart"])
        actuals_raw["MonthStart"] = actuals_raw["MonthStart"].dt.to_period("M").dt.to_timestamp()

        sim_months = pd.to_datetime(sim_df["MonthStart"].unique())
        actual_window = actuals_raw[actuals_raw["MonthStart"].isin(sim_months)][
            ["SKU", "MonthStart", "UnitsSold"]
        ].rename(columns={"UnitsSold": "ActualUnits"})

        sim_df["MonthStart"] = pd.to_datetime(sim_df["MonthStart"])
        sim_df = sim_df.merge(actual_window, on=["SKU", "MonthStart"], how="left")
        sim_df["HasActual"] = sim_df["ActualUnits"].notna()
        sim_df["ActualUnits"] = sim_df["ActualUnits"].fillna(np.nan)
        logger.info("Joined actuals from %s (%d rows in sim window)",
                    actual_path.name, len(actual_window))

    except FileNotFoundError:
        logger.warning("Actuals file not found at %s — simulation has no actuals.", actual_path)
        sim_df["ActualUnits"] = np.nan
        sim_df["HasActual"]   = False

    # ── 4. Compute error metrics ──────────────────────────────────────────────
    has = sim_df["HasActual"]
    sim_df["Error"]       = np.where(has, sim_df["ActualUnits"] - sim_df["PredictedUnits"], np.nan)
    sim_df["AbsError"]    = np.where(has, np.abs(sim_df["Error"]),                          np.nan)
    sim_df["AbsPctError"] = np.where(
        has & (sim_df["ActualUnits"] > 0),
        np.abs(sim_df["Error"]) / sim_df["ActualUnits"] * 100,
        np.nan,
    )

    # ── 5. Final column order ─────────────────────────────────────────────────
    col_order = [
        "SKU", "MonthStart", "Segment", "ModelName", "HorizonMonths",
        "PredictedUnits", "ActualUnits", "HasActual",
        "Error", "AbsError", "AbsPctError",
        "Conservative", "Middle", "Aggressive",
        "ModelVersion",
    ]
    sim_df = sim_df[[c for c in col_order if c in sim_df.columns]]
    sim_df = sim_df.sort_values(["SKU", "MonthStart"]).reset_index(drop=True)

    n_scored = sim_df["HasActual"].sum()
    logger.info(
        "Simulation complete: %d prediction rows, %d with actuals (%d without)",
        len(sim_df), n_scored, len(sim_df) - n_scored,
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(out, index=False)
        logger.info("Saved simulation to %s", out)

    return sim_df


# ---------------------------------------------------------------------------
# Step 5 — v5.2 Simulation summary CSVs
# ---------------------------------------------------------------------------

def build_simulation_summaries(
    sim_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Build and save three simulation summary CSV files from run_simulation() output.

    These are Power BI-ready aggregates of the Jan-Apr 2026 simulation results.
    Only rows where HasActual==True contribute to error metrics (WMAPE, AbsError).
    Rows without actuals (e.g. Mar-Apr 2026) appear in predicted totals only.

    Outputs written to output_dir
    ------------------------------
    simulation_summary_by_month.csv
        Columns: MonthStart, HorizonMonths, N_SKUs, PredictedUnits,
                 ActualUnits, Error, AbsError, WMAPE, N_Scored

    simulation_summary_by_segment.csv
        Columns: Segment, HorizonMonths, N_SKUs, PredictedUnits,
                 ActualUnits, Error, AbsError, WMAPE, N_Scored

    simulation_summary_by_model.csv
        Columns: ModelName, HorizonMonths, N_SKUs, PredictedUnits,
                 ActualUnits, Error, AbsError, WMAPE, N_Scored

    Parameters
    ----------
    sim_df     : DataFrame from run_simulation() or loaded from simulation CSV
    output_dir : folder to write the three CSV files (created if absent)

    Returns
    -------
    dict with keys "by_month", "by_segment", "by_model" — each a pd.DataFrame
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim = sim_df.copy()
    sim["MonthStart"]     = pd.to_datetime(sim["MonthStart"])
    sim["PredictedUnits"] = pd.to_numeric(sim["PredictedUnits"], errors="coerce").fillna(0)
    sim["ActualUnits"]    = pd.to_numeric(sim["ActualUnits"],    errors="coerce")
    sim["AbsError"]       = pd.to_numeric(sim["AbsError"],       errors="coerce")

    # Normalise HasActual to bool
    if sim["HasActual"].dtype == object:
        sim["HasActual"] = sim["HasActual"].astype(str).str.lower() == "true"
    else:
        sim["HasActual"] = sim["HasActual"].astype(bool)

    scored = sim[sim["HasActual"]].copy()

    def _summarise(group_cols: list[str]) -> pd.DataFrame:
        all_agg = (
            sim.groupby(group_cols)
            .agg(
                N_SKUs        =("SKU", "nunique"),
                PredictedUnits=("PredictedUnits", "sum"),
            )
            .reset_index()
        )
        if scored.empty:
            all_agg["ActualUnits"] = np.nan
            all_agg["Error"]       = np.nan
            all_agg["AbsError"]    = np.nan
            all_agg["WMAPE"]       = np.nan
            all_agg["N_Scored"]    = 0
            return all_agg

        scored_agg = (
            scored.groupby(group_cols)
            .agg(
                ActualUnits_sum=("ActualUnits", "sum"),
                AbsError_sum   =("AbsError",    "sum"),
                N_Scored       =("SKU",         "count"),
            )
            .reset_index()
        )
        df = all_agg.merge(scored_agg, on=group_cols, how="left")
        df["ActualUnits"] = df["ActualUnits_sum"]
        df["AbsError"]    = df["AbsError_sum"]
        df["Error"]       = df["ActualUnits"] - df["PredictedUnits"]
        df["WMAPE"]       = np.where(
            df["ActualUnits"] > 0,
            df["AbsError"] / df["ActualUnits"] * 100,
            np.nan,
        )
        df["N_Scored"] = df["N_Scored"].fillna(0).astype(int)
        return df.drop(columns=["ActualUnits_sum", "AbsError_sum"], errors="ignore")

    rnd = {"PredictedUnits": 0, "ActualUnits": 0, "Error": 0, "AbsError": 0, "WMAPE": 2}

    by_month = _summarise(["MonthStart", "HorizonMonths"])
    by_month["MonthStart"] = by_month["MonthStart"].dt.strftime("%Y-%m-%d")
    by_month = by_month.round(rnd)
    by_month.to_csv(out / "simulation_summary_by_month.csv", index=False)
    logger.info("Saved simulation_summary_by_month.csv (%d rows)", len(by_month))

    by_seg = _summarise(["Segment", "HorizonMonths"])
    by_seg = by_seg.round(rnd)
    by_seg.to_csv(out / "simulation_summary_by_segment.csv", index=False)
    logger.info("Saved simulation_summary_by_segment.csv (%d rows)", len(by_seg))

    by_model = _summarise(["ModelName", "HorizonMonths"])
    by_model = by_model.round(rnd)
    by_model.to_csv(out / "simulation_summary_by_model.csv", index=False)
    logger.info("Saved simulation_summary_by_model.csv (%d rows)", len(by_model))

    return {"by_month": by_month, "by_segment": by_seg, "by_model": by_model}


# ===========================================================================
# v6 HIERARCHICAL PIPELINE — StyleColor-level forecasting + SKU allocation
# ===========================================================================
# These functions extend the pipeline with hierarchical forecasting.
# ALL existing v5.2 functions (run_data_prep, run_cv, run_forecasts,
# run_simulation) are unchanged. v6 adds two new orchestration functions:
#
#   run_hierarchical_prep(data_dir, phase)
#       -> Builds the StyleColor-level prep dict. Structurally identical to
#          the dict returned by run_data_prep() so that run_cv() and
#          run_forecasts() can be called on it unmodified.
#
#   run_hierarchical_forecasts(sc_prep, standalone_prep, best_models_df, ...)
#       -> Generates StyleColor-level forecasts, disaggregates to SKU via
#          size shares, and merges in STANDALONE SKU forecasts.
#          Produces both stylecolor_forecasts.csv and gold_fact_forecasts.csv.
# ===========================================================================


def run_hierarchical_prep(
    data_dir,
    phase: int = 1,
    trailing_months: int = 12,
    zero_ratio_threshold: float = 0.40,
):
    """
    v6: Build the StyleColor-level prep dict for the hierarchical pipeline.

    This is the v6 counterpart of run_data_prep(). The returned dict is
    structurally identical so that run_cv() and run_forecasts() can consume
    it without modification.

    In addition to the standard keys, returns:
        "standalone_skus"  -> list[str]  SKUs outside the hierarchy
        "size_shares"      -> DataFrame  (StyleColorDesc, SizeDesc, SKU, share)
        "dim_product"      -> the raw dim_product table (needed for allocation)

    Parameters
    ----------
    data_dir              : folder containing the Gold CSV files
    phase                 : 1 = train through 2025-12, 2 = include HOLDOUT
    trailing_months       : look-back for DEAD classification
    zero_ratio_threshold  : INTERMITTENT threshold

    Returns
    -------
    dict with keys:
        "tables", "panel", "segments", "panel_seg",  <- same as run_data_prep
        "standalone_skus", "size_shares", "dim_product"  <- v6 additions
    """
    from forecasting_pipeline.data_prep import (
        load_gold_tables, build_stylecolor_panel, build_panel, _resolve_training_end,
    )
    from forecasting_pipeline.segmentation import segment_skus, attach_segment
    from forecasting_pipeline.allocation import (
        get_standalone_skus, compute_size_shares,
    )

    logger.info("=== v6 Step 1: Hierarchical data prep (phase=%d) ===", phase)

    tables = load_gold_tables(data_dir)
    gold_df      = tables["demand"]
    dim_date_df  = tables["dim_date"]
    dim_prod_df  = tables.get("dim_product")

    if dim_prod_df is None:
        raise ValueError(
            "dim_product.csv is required for hierarchical forecasting (v6). "
            "Place it in the data directory."
        )

    # Determine train_end for size-share computation
    train_end = _resolve_training_end(dim_date_df, phase=1)  # always use phase=1 cutoff for shares

    # ── StyleColor-level panel ────────────────────────────────────────────
    panel = build_stylecolor_panel(
        gold_df=gold_df,
        dim_date_df=dim_date_df,
        dim_product_df=dim_prod_df,
        phase=phase,
    )

    segments  = segment_skus(panel, trailing_months=trailing_months,
                              zero_ratio_threshold=zero_ratio_threshold)
    panel_seg = attach_segment(panel, segments)

    # ── STANDALONE SKUs and size shares ───────────────────────────────────
    standalone_skus = get_standalone_skus(gold_df, dim_prod_df)

    size_shares = compute_size_shares(
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        lookback_months=12,
        min_lookback_months=6,
        train_end=train_end,
    )

    logger.info(
        "v6 data prep complete — %d StyleColors | REGULAR %d  INTERMITTENT %d  DEAD %d",
        panel_seg["SKU"].nunique(),
        (segments["Segment"] == "REGULAR").sum(),
        (segments["Segment"] == "INTERMITTENT").sum(),
        (segments["Segment"] == "DEAD").sum(),
    )
    logger.info(
        "Size shares: %d (StyleColor, Size) pairs; %d STANDALONE SKUs",
        len(size_shares), len(standalone_skus),
    )

    return {
        "tables":          tables,
        "panel":           panel,
        "segments":        segments,
        "panel_seg":       panel_seg,
        # v6 additions
        "standalone_skus": standalone_skus,
        "size_shares":     size_shares,
        "dim_product":     dim_prod_df,
    }


def run_hierarchical_forecasts(
    sc_prep,
    best_models_df,
    horizon_months: int,
    forecast_start: str = "2026-03-01",
    n_forecast_months: int = 12,
    phase: int = 1,
    model_version: str = "v6.0",
    sc_output_path=None,
    sku_output_path=None,
    append: bool = False,
    lookback_months: int = 12,
    adjustment_config: dict | None = None,
):
    """
    v6: Generate hierarchical forecasts — StyleColor level → allocate → SKU level.

    Steps
    -----
    1. Run run_forecasts() on the StyleColor-level prep dict to produce
       StyleColor-level ForecastUnits (Key = StyleColorDesc).
    2. Disaggregate via allocation.allocate_to_sku() using pre-computed
       size shares from sc_prep["size_shares"].
    3. For STANDALONE SKUs (those outside the hierarchy), run run_forecasts()
       with the standard v5.2 SKU-level prep dict and concatenate.
    4. Validate allocation consistency.
    5. Write stylecolor_forecasts.csv (SC level) and gold_fact_forecasts.csv
       (SKU level, drop-in replacement for the v5.2 output).

    Parameters
    ----------
    sc_prep             : output of run_hierarchical_prep()
    best_models_df      : output of run_cv() run on sc_prep
    horizon_months      : 1 or 3
    forecast_start      : first forecast month string
    n_forecast_months   : number of months to forecast
    phase               : 1 or 2 (passed to run_forecasts)
    model_version       : tag written into ModelVersion column
    sc_output_path      : path for stylecolor_forecasts.csv (optional)
    sku_output_path     : path for gold_fact_forecasts.csv (optional)
    append              : if True, append to sku_output_path if it exists
    lookback_months     : size-share look-back window (default 12, matches prep)
    adjustment_config   : v6.2 forecast correction config dict from
                          forecast_adjustments.get_config(), or None to disable.
                          Passed through to both the SC-level and STANDALONE
                          run_forecasts() calls.

    Returns
    -------
    tuple (stylecolor_df, sku_df)
        stylecolor_df : StyleColor-level forecasts (Key = StyleColorDesc)
        sku_df        : SKU-level forecasts (Key = SKU) — drop-in for v5.2
    """
    from forecasting_pipeline.forecasting import write_forecasts, append_forecasts
    from forecasting_pipeline.allocation  import allocate_to_sku, validate_allocation

    logger.info(
        "=== v6 Hierarchical forecasts: H=%d, start=%s, months=%d (adjustments=%s) ===",
        horizon_months, forecast_start, n_forecast_months,
        "enabled" if adjustment_config is not None else "disabled",
    )

    # ── Step 1: StyleColor-level forecasts ───────────────────────────────
    sc_forecasts = run_forecasts(
        prep=sc_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=model_version,
        output_path=sc_output_path,
        append=False,
        adjustment_config=adjustment_config,
    )
    logger.info(
        "StyleColor forecasts: %d rows for %d StyleColors",
        len(sc_forecasts),
        sc_forecasts["Key"].nunique() if "Key" in sc_forecasts.columns else 0,
    )

    # ── Step 2: STANDALONE SKU forecasts (v5.2 path, with v6.2 adjustments) ──
    standalone_skus = sc_prep.get("standalone_skus", [])
    standalone_fc = pd.DataFrame()

    if standalone_skus:
        logger.info(
            "Running SKU-level forecasts for %d STANDALONE SKUs", len(standalone_skus)
        )
        tables     = sc_prep["tables"]
        dim_date   = tables["dim_date"]
        gold_df    = tables["demand"]
        dim_prod   = sc_prep.get("dim_product")

        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        # Build a SKU-level panel restricted to STANDALONE SKUs only
        standalone_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not standalone_gold.empty:
            sa_panel = build_panel(
                demand_df=standalone_gold,
                dim_date_df=dim_date,
                phase=phase,
                dim_product_df=dim_prod,
            )
            sa_segments  = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segments)
            sa_prep = {
                "tables":    tables,
                "panel":     sa_panel,
                "segments":  sa_segments,
                "panel_seg": sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=model_version + "-standalone",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )
            logger.info(
                "STANDALONE forecasts: %d rows for %d SKUs",
                len(standalone_fc), standalone_fc["Key"].nunique(),
            )

    # ── Step 3: Disaggregate SC → SKU ─────────────────────────────────────
    size_shares = sc_prep.get("size_shares", pd.DataFrame())
    sku_forecasts = allocate_to_sku(
        stylecolor_forecasts_df=sc_forecasts,
        size_shares_df=size_shares,
        dim_product_df=sc_prep.get("dim_product", pd.DataFrame()),
        standalone_sku_forecasts_df=standalone_fc if not standalone_fc.empty else None,
    )

    # ── Step 4: Validate allocation ───────────────────────────────────────
    val = validate_allocation(sc_forecasts, sku_forecasts)
    logger.info(
        "Allocation validation — sum_check: %s, max_diff: %s, no_negatives: %s",
        val["sum_check_passed"],
        val["max_abs_diff"],
        val["no_negatives"],
    )
    if not val["sum_check_passed"]:
        logger.warning(
            "Allocation sum check failed (max_abs_diff=%.4f). "
            "Review size_shares for StyleColors with sparse history.",
            val.get("max_abs_diff", float("nan")),
        )

    # ── Step 5: Write outputs ─────────────────────────────────────────────
    if sku_output_path is not None:
        p = Path(sku_output_path)
        if append and p.exists():
            append_forecasts(sku_forecasts, p)
        else:
            write_forecasts(sku_forecasts, p)
        logger.info("Saved SKU-level forecasts: %s (%d rows)", p, len(sku_forecasts))

    return sc_forecasts, sku_forecasts


# ===========================================================================
# v7 HIERARCHICAL PIPELINE — StyleCode-level forecasting + two-level allocation
# ===========================================================================
# Adds one more hierarchy level above v6:
#
#   StyleCodeDesc → StyleColorDesc → SizeDesc → SKU
#
# Two new orchestration functions are added; ALL v6 and v5.2 functions remain
# unchanged.
#
#   run_v7_prep(data_dir, phase)
#       -> Builds the StyleCode-level prep dict (structurally identical to
#          run_hierarchical_prep so run_cv() / run_forecasts() work unchanged).
#          Adds v7-specific keys: "stylecolor_shares", "stylecode_standalone_skus".
#
#   run_v7_forecasts(scode_prep, sc_prep, best_models_df, ...)
#       -> StyleCode forecasts → allocate to StyleColor → allocate to SKU.
#          Produces: stylecode_forecasts.csv, stylecolor_allocations.csv,
#                    gold_fact_forecasts.csv (final SKU output).
# ===========================================================================


def run_v7_prep(
    data_dir,
    phase: int = 1,
    trailing_months: int = 12,
    zero_ratio_threshold: float = 0.40,
    lookback_months: int = 12,
    min_lookback_months: int = 6,
):
    """
    v7: Build the StyleCode-level prep dict for the two-level hierarchy.

    This mirrors run_hierarchical_prep() but operates at the StyleCodeDesc
    level.  The returned dict is structurally compatible with run_cv() and
    run_forecasts() (they use 'panel_seg', 'segments', 'tables').

    Additional v7 keys returned:
        "stylecolor_shares"        : StyleCode→StyleColor share weights
        "size_shares"              : StyleColor→SKU size shares (v6 reused)
        "stylecode_standalone_skus": SKUs excluded from the StyleCode hierarchy
        "dim_product"              : raw dim_product table
        "sc_prep"                  : v6-style StyleColor prep dict (for
                                     intermediate allocation and comparison)

    Parameters
    ----------
    data_dir              : folder containing Gold CSV files
    phase                 : 1 = train through 2025-12 (default)
    trailing_months       : look-back for DEAD classification
    zero_ratio_threshold  : INTERMITTENT threshold
    lookback_months       : share look-back window (for both allocation levels)
    min_lookback_months   : minimum months required for primary window

    Returns
    -------
    dict with keys as described above plus the standard prep keys.
    """
    from forecasting_pipeline.data_prep import (
        load_gold_tables, build_stylecode_panel, build_stylecolor_panel,
        build_panel, _resolve_training_end,
    )
    from forecasting_pipeline.segmentation import segment_skus, attach_segment
    from forecasting_pipeline.stylecode_allocation import (
        build_stylecode_demand,
        compute_stylecolor_shares,
        get_v7_standalone_skus,
    )
    from forecasting_pipeline.allocation import (
        get_standalone_skus,
        compute_size_shares,
    )

    logger.info("=== v7 Step 1: StyleCode-level data prep (phase=%d) ===", phase)

    tables      = load_gold_tables(data_dir)
    gold_df     = tables["demand"]
    dim_date_df = tables["dim_date"]
    dim_prod_df = tables.get("dim_product")

    if dim_prod_df is None:
        raise ValueError(
            "dim_product.csv is required for v7 hierarchical forecasting. "
            "Place it in the data directory."
        )

    # Always use phase=1 cut-off for share computation (no leakage)
    train_end = _resolve_training_end(dim_date_df, phase=1)

    # ── StyleCode-level panel (modeling grain) ────────────────────────────
    panel = build_stylecode_panel(
        gold_df=gold_df,
        dim_date_df=dim_date_df,
        dim_product_df=dim_prod_df,
        phase=phase,
    )
    segments  = segment_skus(panel, trailing_months=trailing_months,
                              zero_ratio_threshold=zero_ratio_threshold)
    panel_seg = attach_segment(panel, segments)

    # ── Level-1 → Level-2 shares: StyleCode → StyleColor ────────────────
    stylecolor_shares = compute_stylecolor_shares(
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        train_end=train_end,
    )

    # ── Level-2 → Level-3 shares: StyleColor → SKU  (reuse v6) ─────────
    size_shares = compute_size_shares(
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        train_end=train_end,
    )

    # ── STANDALONE SKUs (cannot enter the StyleCode hierarchy) ───────────
    stylecode_standalone_skus = get_v7_standalone_skus(gold_df, dim_prod_df)

    logger.info(
        "v7 prep complete — %d StyleCodes | REGULAR %d  INTERMITTENT %d  DEAD %d",
        panel_seg["SKU"].nunique(),
        (segments["Segment"] == "REGULAR").sum(),
        (segments["Segment"] == "INTERMITTENT").sum(),
        (segments["Segment"] == "DEAD").sum(),
    )
    logger.info(
        "StyleColor shares: %d pairs | Size shares: %d pairs | STANDALONE: %d SKUs",
        len(stylecolor_shares), len(size_shares), len(stylecode_standalone_skus),
    )

    return {
        # Standard prep keys (consumed by run_cv / run_forecasts unchanged)
        "tables":    tables,
        "panel":     panel,
        "segments":  segments,
        "panel_seg": panel_seg,
        # v7 additions
        "stylecolor_shares":         stylecolor_shares,
        "size_shares":               size_shares,
        "stylecode_standalone_skus": stylecode_standalone_skus,
        "dim_product":               dim_prod_df,
    }


def run_v7_forecasts(
    scode_prep: dict,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-03-01",
    n_forecast_months: int = 12,
    phase: int = 1,
    model_version: str = "v7.0",
    scode_output_path=None,
    scol_output_path=None,
    sku_output_path=None,
    append: bool = False,
    adjustment_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    v7: Generate full two-level hierarchical forecasts.

    Pipeline:
        1. run_forecasts() at StyleCode level → scode_forecasts
        2. allocate_to_stylecolor()           → stylecolor_forecasts
        3. allocate_to_sku()                  → sku_forecasts
        4. STANDALONE SKU pass-through        → appended to sku_forecasts
        5. Validate both allocation steps
        6. Write outputs

    Parameters
    ----------
    scode_prep          : output of run_v7_prep()
    best_models_df      : output of run_cv() on scode_prep
    horizon_months      : 1 or 3
    forecast_start      : first forecast month string (e.g. "2026-03-01")
    n_forecast_months   : number of months to produce
    phase               : 1 or 2 (passed through to run_forecasts)
    model_version       : tag written into ModelVersion column
    scode_output_path   : optional path for v7_stylecode_forecasts.csv
    scol_output_path    : optional path for v7_stylecolor_allocations.csv
    sku_output_path     : optional path for v7_gold_fact_forecasts.csv
    append              : if True, append to sku_output_path if it exists
    adjustment_config   : v6.2 adjustment config dict or None to disable.

    Returns
    -------
    (scode_df, stylecolor_df, sku_df)
    """
    from forecasting_pipeline.forecasting import write_forecasts, append_forecasts
    from forecasting_pipeline.stylecode_allocation import (
        allocate_to_stylecolor,
        validate_stylecode_allocation,
    )
    from forecasting_pipeline.allocation import allocate_to_sku, validate_allocation

    logger.info(
        "=== v7 Forecasts: H=%d, start=%s, months=%d (adjustments=%s) ===",
        horizon_months, forecast_start, n_forecast_months,
        "enabled" if adjustment_config is not None else "disabled",
    )

    # ── Step 1: StyleCode-level forecasts ────────────────────────────────
    scode_forecasts = run_forecasts(
        prep=scode_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=model_version,
        output_path=scode_output_path,
        append=False,
        adjustment_config=adjustment_config,
    )
    logger.info(
        "StyleCode forecasts: %d rows for %d StyleCodes",
        len(scode_forecasts),
        scode_forecasts["Key"].nunique() if "Key" in scode_forecasts.columns else 0,
    )

    # ── Step 2: StyleCode → StyleColor allocation ─────────────────────────
    stylecolor_shares = scode_prep.get("stylecolor_shares", pd.DataFrame())
    stylecolor_forecasts = allocate_to_stylecolor(
        stylecode_forecasts_df=scode_forecasts,
        color_shares_df=stylecolor_shares,
        dim_product_df=scode_prep.get("dim_product", pd.DataFrame()),
    )
    logger.info(
        "StyleColor allocations: %d rows for %d StyleColors",
        len(stylecolor_forecasts),
        stylecolor_forecasts["Key"].nunique() if "Key" in stylecolor_forecasts.columns else 0,
    )

    # Validate step 1 → step 2
    val_sc = validate_stylecode_allocation(scode_forecasts, stylecolor_forecasts)
    logger.info(
        "StyleCode→StyleColor validation: sum_check=%s, max_diff=%s",
        val_sc["sum_check_passed"], val_sc["max_abs_diff"],
    )

    # Optionally save StyleColor intermediate file
    if scol_output_path is not None:
        write_forecasts(stylecolor_forecasts, scol_output_path)

    # ── Step 3: STANDALONE SKU forecasts (bypass hierarchy) ──────────────
    standalone_skus = scode_prep.get("stylecode_standalone_skus", [])
    standalone_fc   = pd.DataFrame()

    if standalone_skus:
        tables   = scode_prep["tables"]
        gold_df  = tables["demand"]
        dim_prod = scode_prep.get("dim_product")

        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        standalone_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not standalone_gold.empty:
            sa_panel     = build_panel(
                demand_df=standalone_gold,
                dim_date_df=tables["dim_date"],
                phase=phase,
                dim_product_df=dim_prod,
            )
            sa_segs      = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segs)
            sa_prep = {
                "tables":    tables,
                "panel":     sa_panel,
                "segments":  sa_segs,
                "panel_seg": sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=model_version + "-standalone",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )
            logger.info(
                "STANDALONE forecasts: %d rows for %d SKUs",
                len(standalone_fc), standalone_fc["Key"].nunique(),
            )

    # ── Step 4: StyleColor → SKU allocation ──────────────────────────────
    # Feed the intermediate StyleColor forecasts into the existing allocate_to_sku()
    # which handles the SizeDesc split.  stylecolor_forecasts has Key=StyleColorDesc.
    size_shares = scode_prep.get("size_shares", pd.DataFrame())
    sku_forecasts = allocate_to_sku(
        stylecolor_forecasts_df=stylecolor_forecasts,
        size_shares_df=size_shares,
        dim_product_df=scode_prep.get("dim_product", pd.DataFrame()),
        standalone_sku_forecasts_df=standalone_fc if not standalone_fc.empty else None,
    )

    # Validate step 2 → step 3
    val_sku = validate_allocation(stylecolor_forecasts, sku_forecasts)
    logger.info(
        "StyleColor→SKU validation: sum_check=%s, max_diff=%s, no_neg=%s",
        val_sku["sum_check_passed"], val_sku["max_abs_diff"], val_sku["no_negatives"],
    )

    # ── Step 5: Write SKU output ──────────────────────────────────────────
    if sku_output_path is not None:
        p = Path(sku_output_path)
        if append and p.exists():
            append_forecasts(sku_forecasts, p)
        else:
            write_forecasts(sku_forecasts, p)
        logger.info("Saved v7 SKU forecasts: %s (%d rows)", p, len(sku_forecasts))

    return scode_forecasts, stylecolor_forecasts, sku_forecasts


# ===========================================================================
# v7.2 ABLATION PIPELINE — controlled allocation experiment
# ===========================================================================
# run_v7_2_ablation() runs the shared upstream StyleCode forecasting once,
# then applies all four allocation variants to the same forecast DataFrame.
# Each variant produces its own diagnostics, holdout evaluation, and error
# decomposition files.  A cross-variant comparison table is also written.
# ===========================================================================


def run_v7_2_ablation(
    data_dir,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-01-01",
    n_forecast_months: int = 2,
    phase: int = 1,
    adjustment_config: dict | None = None,
    output_dir: str | Path | None = None,
    actuals_path: str | Path | None = None,
    holdout_months: list | None = None,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float     = 0.3,
    cap_rel_increase: float = 0.5,
) -> dict:
    """
    v7.2: Run controlled allocation ablation using one shared upstream forecast.

    All four allocation variants (baseline_v7, recency_only, smoothing_only,
    caps_only) receive the same StyleCode-level forecasts.  Only the share
    computation differs.  This isolates the marginal effect of each technique.

    Steps
    -----
    1. run_v7_prep() — build StyleCode-level panel (for panel/segment/tables)
    2. run_forecasts() at StyleCode level — generate upstream forecasts once
    3. run_all_variants() — apply all four allocation variants to those forecasts
    4. build_variant_comparison() — write summary CSV
    5. Write per-variant output CSVs

    Parameters
    ----------
    data_dir             : Gold CSV folder
    best_models_df       : output of run_cv() (StyleCode-level best models)
    horizon_months       : 1 or 3
    forecast_start       : first forecast month (e.g. "2026-01-01" for holdout)
    n_forecast_months    : number of months to forecast (2 for Jan-Feb holdout)
    phase                : 1 (train through 2025-12)
    adjustment_config    : v6.2 adjustment config dict or None
    output_dir           : folder for output CSVs (None = skip file writes)
    actuals_path         : path to gold_fact_monthly_demand_v2.csv for scoring
    holdout_months       : list of pd.Timestamp to score (default Jan-Feb 2026)
    numeric params       : forwarded to allocation share computation

    Returns
    -------
    dict with keys:
        "scode_forecasts"    : shared upstream StyleCode forecasts DataFrame
        "variant_results"    : dict[str → result_dict] from run_all_variants()
        "comparison_table"   : v7_2_variant_comparison DataFrame
    """
    from forecasting_pipeline.data_prep import _resolve_training_end
    from forecasting_pipeline.allocation_v72 import (
        run_all_variants,
        build_variant_comparison,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if holdout_months is None:
        holdout_months = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]

    # ── Step 1: v7.0 upstream prep ────────────────────────────────────────
    logger.info("=== v7.2 Ablation: upstream prep (phase=%d) ===", phase)
    scode_prep = run_v7_prep(
        data_dir=data_dir,
        phase=phase,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
    )

    tables      = scode_prep["tables"]
    dim_date_df = tables["dim_date"]
    gold_df     = tables["demand"]
    dim_prod_df = scode_prep["dim_product"]
    train_end   = _resolve_training_end(dim_date_df, phase=1)

    # ── Step 2: StyleCode-level forecasts (shared) ────────────────────────
    logger.info(
        "=== v7.2 Ablation: generating shared StyleCode forecasts (H=%d) ===",
        horizon_months,
    )
    scode_forecasts = run_forecasts(
        prep=scode_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=f"v7.2-upstream-H{horizon_months}",
        output_path=None,
        append=False,
        adjustment_config=adjustment_config,
    )
    logger.info(
        "Shared StyleCode forecasts: %d rows for %d StyleCodes",
        len(scode_forecasts),
        scode_forecasts["Key"].nunique() if "Key" in scode_forecasts.columns else 0,
    )

    # Optional: save shared upstream forecast
    if output_dir is not None:
        scode_forecasts.to_csv(
            output_dir / f"v7_2_shared_stylecode_forecasts_H{horizon_months}.csv",
            index=False,
        )

    # ── Step 3: STANDALONE pass-through (shared across variants) ─────────
    standalone_skus = scode_prep.get("stylecode_standalone_skus", [])
    standalone_fc   = pd.DataFrame()

    if standalone_skus:
        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        standalone_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not standalone_gold.empty:
            sa_panel     = build_panel(
                demand_df=standalone_gold,
                dim_date_df=tables["dim_date"],
                phase=phase,
                dim_product_df=dim_prod_df,
            )
            sa_segs      = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segs)
            sa_prep = {
                "tables": tables, "panel": sa_panel,
                "segments": sa_segs, "panel_seg": sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=f"v7.2-standalone-H{horizon_months}",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )

    # ── Step 4: Load actuals for scoring ──────────────────────────────────
    actuals_df = None
    if actuals_path is not None:
        ap = Path(actuals_path)
        if ap.exists():
            actuals_df = pd.read_csv(ap, parse_dates=["MonthStart"])
            actuals_df["MonthStart"] = (
                actuals_df["MonthStart"].dt.to_period("M").dt.to_timestamp()
            )
        else:
            logger.warning("actuals_path not found: %s — holdout scoring skipped", ap)

    # ── Step 5: Run all four allocation variants ───────────────────────────
    logger.info("=== v7.2 Ablation: running all 4 allocation variants ===")
    numeric = dict(
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=smooth_alpha,
        cap_rel_increase=cap_rel_increase,
    )

    variant_results = run_all_variants(
        scode_forecasts_df=scode_forecasts,
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        train_end=train_end,
        standalone_fc_df=standalone_fc if not standalone_fc.empty else None,
        output_dir=output_dir,
        actuals_df=actuals_df,
        holdout_months=holdout_months,
        **numeric,
    )

    # ── Step 6: Build and write comparison table ───────────────────────────
    comparison_table = pd.DataFrame()
    if actuals_df is not None:
        comparison_table = build_variant_comparison(
            variant_results=variant_results,
            actuals_df=actuals_df,
            dim_product_df=dim_prod_df,
            holdout_months=holdout_months,
            scode_forecasts_df=scode_forecasts,
        )
        if output_dir is not None:
            comparison_table.to_csv(
                output_dir / "v7_2_variant_comparison.csv", index=False
            )
            logger.info(
                "[v7.2] Saved v7_2_variant_comparison.csv (%d rows)", len(comparison_table)
            )

    return {
        "scode_forecasts":  scode_forecasts,
        "variant_results":  variant_results,
        "comparison_table": comparison_table,
    }
# ===========================================================================
# run_v7_1_prep() and run_v7_1_forecasts() are drop-in replacements for
# run_v7_prep() and run_v7_forecasts() with:
#   • recency-weighted, smoothed, capped shares at both allocation levels
#   • four diagnostic CSV outputs written automatically
#   • segment-aware window selection
# ===========================================================================


def run_v7_1_prep(
    data_dir,
    phase: int = 1,
    trailing_months: int = 12,
    zero_ratio_threshold: float = 0.40,
    lookback_months_regular: int = 12,
    lookback_months_intermittent: int = 3,
    min_months_regular: int = 6,
    min_months_intermittent: int = 2,
    smooth_alpha: float = 0.3,
    cap_rel_increase: float = 0.5,
    min_share_floor: float = 0.0,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
) -> dict:
    """
    v7.1: Build the StyleCode-level prep dict with improved allocation shares.

    This is the v7.1 replacement for ``run_v7_prep()``.  The returned dict
    is structurally identical (same keys) but the ``stylecolor_shares`` and
    ``size_shares`` entries use the recency-weighted, smoothed, capped shares
    from the v7.1 allocation improvements.

    Additional key returned vs run_v7_prep:
        "segment_map_scode"   : dict StyleCodeDesc → Segment  (for diagnostic use)
        "segment_map_scol"    : dict StyleColorDesc → Segment (for diagnostic use)

    All other parameters match run_v7_prep().
    """
    from forecasting_pipeline.data_prep import (
        load_gold_tables, build_stylecode_panel, _resolve_training_end,
    )
    from forecasting_pipeline.segmentation import segment_skus, attach_segment
    from forecasting_pipeline.stylecode_allocation import (
        get_v7_standalone_skus,
        compute_stylecolor_shares_v71,
    )
    from forecasting_pipeline.allocation import compute_size_shares_v71

    logger.info("=== v7.1 Step 1: StyleCode-level data prep (phase=%d) ===", phase)

    tables      = load_gold_tables(data_dir)
    gold_df     = tables["demand"]
    dim_date_df = tables["dim_date"]
    dim_prod_df = tables.get("dim_product")

    if dim_prod_df is None:
        raise ValueError(
            "dim_product.csv is required for v7.1 hierarchical forecasting."
        )

    train_end = _resolve_training_end(dim_date_df, phase=1)

    panel = build_stylecode_panel(
        gold_df=gold_df,
        dim_date_df=dim_date_df,
        dim_product_df=dim_prod_df,
        phase=phase,
    )
    segments  = segment_skus(panel, trailing_months=trailing_months,
                              zero_ratio_threshold=zero_ratio_threshold)
    panel_seg = attach_segment(panel, segments)

    # Build segment maps for both levels (used for segment-aware windows)
    # StyleCode → Segment
    seg_map_scode = dict(zip(segments["SKU"], segments["Segment"]))

    # For StyleColor-level segment map: propagate StyleCode segment down.
    # StyleColors inherit the segment of their parent StyleCode.
    dp_sc = dim_prod_df[["StyleColorDesc","StyleCodeDesc"]].dropna().drop_duplicates("StyleColorDesc")
    seg_map_scol = {
        row["StyleColorDesc"]: seg_map_scode.get(row["StyleCodeDesc"], "REGULAR")
        for _, row in dp_sc.iterrows()
    }

    # ── v7.1 StyleCode → StyleColor shares ──────────────────────────────
    stylecolor_shares = compute_stylecolor_shares_v71(
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        train_end=train_end,
        segment_map=seg_map_scode,
        smooth_alpha=smooth_alpha,
        cap_rel_increase=cap_rel_increase,
        min_share_floor=min_share_floor,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        window_regular=lookback_months_regular,
        window_intermittent=lookback_months_intermittent,
        min_months_regular=min_months_regular,
        min_months_intermittent=min_months_intermittent,
    )

    # ── v7.1 StyleColor → SKU (size) shares ─────────────────────────────
    size_shares = compute_size_shares_v71(
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        train_end=train_end,
        segment_map=seg_map_scol,
        smooth_alpha=smooth_alpha,
        cap_rel_increase=cap_rel_increase,
        min_share_floor=min_share_floor,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        window_regular=lookback_months_regular,
        window_intermittent=lookback_months_intermittent,
        min_months_regular=min_months_regular,
        min_months_intermittent=min_months_intermittent,
    )

    stylecode_standalone_skus = get_v7_standalone_skus(gold_df, dim_prod_df)

    logger.info(
        "v7.1 prep complete — %d StyleCodes | REGULAR %d  INTERMITTENT %d  DEAD %d",
        panel_seg["SKU"].nunique(),
        (segments["Segment"] == "REGULAR").sum(),
        (segments["Segment"] == "INTERMITTENT").sum(),
        (segments["Segment"] == "DEAD").sum(),
    )
    logger.info(
        "v7.1 StyleColor shares: %d pairs | Size shares: %d pairs | STANDALONE: %d",
        len(stylecolor_shares), len(size_shares), len(stylecode_standalone_skus),
    )

    return {
        # Standard prep keys
        "tables":    tables,
        "panel":     panel,
        "segments":  segments,
        "panel_seg": panel_seg,
        # v7.1 allocation
        "stylecolor_shares":          stylecolor_shares,
        "size_shares":                size_shares,
        "stylecode_standalone_skus":  stylecode_standalone_skus,
        "dim_product":                dim_prod_df,
        # segment maps for diagnostic use
        "segment_map_scode":          seg_map_scode,
        "segment_map_scol":           seg_map_scol,
    }


def run_v7_1_forecasts(
    scode_prep: dict,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-03-01",
    n_forecast_months: int = 12,
    phase: int = 1,
    model_version: str = "v7.1",
    scode_output_path=None,
    scol_output_path=None,
    sku_output_path=None,
    append: bool = False,
    adjustment_config: dict | None = None,
    diag_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    v7.1: Two-level hierarchical forecasts with improved allocation and diagnostics.

    Uses the v7.1 allocation functions (recency-weighted, smoothed, capped)
    at both the StyleCode→StyleColor and StyleColor→SKU steps.

    Writes four diagnostic CSV files when ``diag_output_dir`` is provided:
        v7_1_stylecode_allocation_diagnostics.csv
        v7_1_stylecolor_share_analysis.csv
        v7_1_sku_allocation_diagnostics.csv
        v7_1_error_decomposition.csv  (only when actuals are available)

    Parameters
    ----------
    scode_prep         : output of run_v7_1_prep()
    best_models_df     : output of run_cv() on scode_prep
    horizon_months     : 1 or 3
    forecast_start     : first forecast month (e.g. "2026-03-01")
    n_forecast_months  : number of months to produce
    phase              : 1 or 2
    model_version      : tag for ModelVersion column
    scode_output_path  : optional path for v7_1_stylecode_forecasts.csv
    scol_output_path   : optional path for v7_1_stylecolor_allocations.csv
    sku_output_path    : optional path for v7_1_gold_fact_forecasts.csv
    append             : append to sku_output_path if it exists
    adjustment_config  : v6.2 adjustment config dict or None
    diag_output_dir    : folder to write diagnostic CSVs (None = skip)

    Returns
    -------
    (scode_df, stylecolor_df, sku_df)
    """
    from forecasting_pipeline.forecasting import write_forecasts, append_forecasts
    from forecasting_pipeline.stylecode_allocation import (
        allocate_to_stylecolor_v71,
        validate_stylecode_allocation,
    )
    from forecasting_pipeline.allocation import (
        allocate_to_sku_v71,
        validate_allocation,
    )
    from forecasting_pipeline.allocation_v71_utils import (
        build_stylecode_alloc_diagnostics,
        build_stylecolor_share_analysis,
        build_sku_alloc_diagnostics,
    )

    diag_dir = Path(diag_output_dir) if diag_output_dir is not None else None
    if diag_dir is not None:
        diag_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== v7.1 Forecasts: H=%d, start=%s, months=%d (adjustments=%s) ===",
        horizon_months, forecast_start, n_forecast_months,
        "enabled" if adjustment_config is not None else "disabled",
    )

    # ── Step 1: StyleCode-level forecasts (unchanged from v7) ─────────────
    scode_forecasts = run_forecasts(
        prep=scode_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=model_version,
        output_path=scode_output_path,
        append=False,
        adjustment_config=adjustment_config,
    )
    logger.info(
        "StyleCode forecasts: %d rows for %d StyleCodes",
        len(scode_forecasts),
        scode_forecasts["Key"].nunique() if "Key" in scode_forecasts.columns else 0,
    )

    # ── Step 2: StyleCode → StyleColor (v7.1 improved shares) ─────────────
    stylecolor_shares = scode_prep.get("stylecolor_shares", pd.DataFrame())
    stylecolor_forecasts = allocate_to_stylecolor_v71(
        stylecode_forecasts_df=scode_forecasts,
        color_shares_df=stylecolor_shares,
        dim_product_df=scode_prep.get("dim_product", pd.DataFrame()),
    )
    logger.info(
        "StyleColor allocations: %d rows for %d StyleColors",
        len(stylecolor_forecasts),
        stylecolor_forecasts["Key"].nunique() if "Key" in stylecolor_forecasts.columns else 0,
    )

    # Validate step 1 → step 2
    val_sc = validate_stylecode_allocation(scode_forecasts, stylecolor_forecasts)
    logger.info(
        "[v7.1] StyleCode→StyleColor validation: sum_check=%s, max_diff=%s",
        val_sc["sum_check_passed"], val_sc["max_abs_diff"],
    )

    if scol_output_path is not None:
        write_forecasts(stylecolor_forecasts, scol_output_path)

    # ── Diagnostic A: StyleCode allocation diagnostics ────────────────────
    if diag_dir is not None:
        diag_a = build_stylecode_alloc_diagnostics(scode_forecasts, stylecolor_forecasts)
        diag_a.to_csv(diag_dir / "v7_1_stylecode_allocation_diagnostics.csv", index=False)
        logger.info("[v7.1 diag] Saved v7_1_stylecode_allocation_diagnostics.csv (%d rows)", len(diag_a))

    # ── Diagnostic B: StyleColor share analysis ────────────────────────────
    if diag_dir is not None and not stylecolor_shares.empty:
        diag_b = build_stylecolor_share_analysis(
            shares_df=stylecolor_shares,
            entity_col="StyleCodeDesc",
            child_col="StyleColorDesc",
        )
        diag_b.to_csv(diag_dir / "v7_1_stylecolor_share_analysis.csv", index=False)
        logger.info("[v7.1 diag] Saved v7_1_stylecolor_share_analysis.csv (%d rows)", len(diag_b))

    # ── Step 3: STANDALONE SKU forecasts ──────────────────────────────────
    standalone_skus = scode_prep.get("stylecode_standalone_skus", [])
    standalone_fc   = pd.DataFrame()

    if standalone_skus:
        tables   = scode_prep["tables"]
        gold_df  = tables["demand"]
        dim_prod = scode_prep.get("dim_product")

        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        standalone_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not standalone_gold.empty:
            sa_panel     = build_panel(
                demand_df=standalone_gold,
                dim_date_df=tables["dim_date"],
                phase=phase,
                dim_product_df=dim_prod,
            )
            sa_segs      = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segs)
            sa_prep = {
                "tables": tables, "panel": sa_panel,
                "segments": sa_segs, "panel_seg": sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=model_version + "-standalone",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )
            logger.info(
                "[v7.1] STANDALONE: %d rows for %d SKUs",
                len(standalone_fc), standalone_fc["Key"].nunique(),
            )

    # ── Step 4: StyleColor → SKU (v7.1 improved size shares) ──────────────
    size_shares = scode_prep.get("size_shares", pd.DataFrame())
    sku_forecasts = allocate_to_sku_v71(
        stylecolor_forecasts_df=stylecolor_forecasts,
        size_shares_df=size_shares,
        dim_product_df=scode_prep.get("dim_product", pd.DataFrame()),
        standalone_sku_forecasts_df=standalone_fc if not standalone_fc.empty else None,
    )

    # Validate step 2 → step 3
    val_sku = validate_allocation(stylecolor_forecasts, sku_forecasts)
    logger.info(
        "[v7.1] StyleColor→SKU validation: sum_check=%s, max_diff=%s, no_neg=%s",
        val_sku["sum_check_passed"], val_sku["max_abs_diff"], val_sku["no_negatives"],
    )

    # ── Diagnostic C: SKU allocation diagnostics ──────────────────────────
    if diag_dir is not None:
        diag_c = build_sku_alloc_diagnostics(
            sku_fc_df=sku_forecasts,
            allocation_stage="StyleColor→SKU",
        )
        diag_c.to_csv(diag_dir / "v7_1_sku_allocation_diagnostics.csv", index=False)
        logger.info("[v7.1 diag] Saved v7_1_sku_allocation_diagnostics.csv (%d rows)", len(diag_c))

    # ── Step 5: Write SKU output ──────────────────────────────────────────
    if sku_output_path is not None:
        p = Path(sku_output_path)
        if append and p.exists():
            append_forecasts(sku_forecasts, p)
        else:
            write_forecasts(sku_forecasts, p)
        logger.info("[v7.1] Saved SKU forecasts: %s (%d rows)", p, len(sku_forecasts))

    return scode_forecasts, stylecolor_forecasts, sku_forecasts


# ===========================================================================
# v7.3 PIPELINE — segmented allocation strategy
# ===========================================================================
# run_v7_3_segmented() uses the same upstream StyleCode forecasts as v7.2 but
# selects allocation strategy per parent group instead of applying one method
# globally.  No model retraining; only allocation logic changes.
# ===========================================================================


def run_v7_3_segmented(
    data_dir,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-01-01",
    n_forecast_months: int = 2,
    phase: int = 1,
    adjustment_config: dict | None = None,
    output_dir: str | Path | None = None,
    actuals_path: str | Path | None = None,
    holdout_months: list | None = None,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
    smooth_alpha: float     = 0.20,
    **classification_thresholds,
) -> dict:
    """
    v7.3: Segmented allocation — one strategy per parent group.

    Steps
    -----
    1. run_v7_prep() — build StyleCode-level panel
    2. run_forecasts() at StyleCode level — generate upstream forecasts once
    3. run_segmented_allocation() — classify each parent group and allocate
    4. Write output CSVs

    Parameters
    ----------
    data_dir              : Gold CSV folder
    best_models_df        : output of run_cv() (StyleCode-level)
    horizon_months        : 1 or 3
    forecast_start        : first forecast month (e.g. "2026-01-01")
    n_forecast_months     : number of months to forecast
    phase                 : 1 (train through 2025-12)
    adjustment_config     : v6.2 adjustment config dict or None
    output_dir            : folder for output CSVs (None = skip)
    actuals_path          : path to gold actuals for holdout scoring
    holdout_months        : list of pd.Timestamp (default Jan–Feb 2026)
    numeric params        : forwarded to allocation
    **classification_thresholds : override DEFAULT_THRESHOLDS in classifier

    Returns
    -------
    dict with keys:
        scode_forecasts, segmented_result, holdout_eval, holdout_preds,
        error_decomp, strategy_map_combined
    """
    from forecasting_pipeline.data_prep import _resolve_training_end
    from forecasting_pipeline.allocation_strategy_selector import (
        run_segmented_allocation,
        score_segmented_holdout,
        build_segmented_error_decomp,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if holdout_months is None:
        holdout_months = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]

    # ── Step 1: v7.0 upstream prep ────────────────────────────────────────
    logger.info("=== v7.3 Segmented: upstream prep (phase=%d) ===", phase)
    scode_prep = run_v7_prep(
        data_dir=data_dir,
        phase=phase,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
    )

    tables      = scode_prep["tables"]
    dim_date_df = tables["dim_date"]
    gold_df     = tables["demand"]
    dim_prod_df = scode_prep["dim_product"]
    train_end   = _resolve_training_end(dim_date_df, phase=1)

    # ── Step 2: StyleCode-level forecasts ────────────────────────────────
    logger.info("[v7.3] Generating upstream StyleCode forecasts H=%d", horizon_months)
    scode_forecasts = run_forecasts(
        prep=scode_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=f"v7.3-upstream-H{horizon_months}",
        output_path=None,
        append=False,
        adjustment_config=adjustment_config,
    )

    if output_dir is not None:
        scode_forecasts.to_csv(
            output_dir / f"v7_3_shared_stylecode_forecasts_H{horizon_months}.csv",
            index=False,
        )

    # ── Step 3: STANDALONE pass-through ───────────────────────────────────
    standalone_skus = scode_prep.get("stylecode_standalone_skus", [])
    standalone_fc   = pd.DataFrame()

    if standalone_skus:
        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        sa_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not sa_gold.empty:
            sa_panel     = build_panel(sa_gold, tables["dim_date"], phase=phase, dim_product_df=dim_prod_df)
            sa_segs      = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segs)
            sa_prep = {
                "tables": tables, "panel": sa_panel,
                "segments": sa_segs, "panel_seg": sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=f"v7.3-standalone-H{horizon_months}",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )

    # ── Step 4: Segmented allocation ──────────────────────────────────────
    logger.info("[v7.3] Running segmented allocation (H=%d)", horizon_months)
    seg_result = run_segmented_allocation(
        scode_forecasts_df=scode_forecasts,
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        train_end=train_end,
        standalone_fc_df=standalone_fc if not standalone_fc.empty else None,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=smooth_alpha,
        **classification_thresholds,
    )

    # ── Step 5: Write strategy map ────────────────────────────────────────
    strategy_map_combined = pd.concat(
        [seg_result["scode_strat_map"], seg_result["scol_strat_map"]],
        ignore_index=True,
    )

    if output_dir is not None:
        strategy_map_combined.to_csv(
            output_dir / "v7_3_allocation_strategy_map.csv", index=False
        )
        seg_result["stylecolor_shares"].to_csv(
            output_dir / "v7_3_segmented_allocation_diagnostics_scol.csv", index=False
        )
        seg_result["size_shares"].to_csv(
            output_dir / "v7_3_segmented_allocation_diagnostics_sku.csv", index=False
        )
        seg_result["sku_forecasts"].to_csv(
            output_dir / f"v7_3_sku_forecasts_H{horizon_months}.csv", index=False
        )
        logger.info("[v7.3] Strategy map and diagnostics saved to %s", output_dir)

    # ── Step 6: Load actuals + holdout evaluation ─────────────────────────
    actuals_df    = None
    holdout_eval  = pd.DataFrame()
    holdout_preds = pd.DataFrame()
    error_decomp  = pd.DataFrame()

    if actuals_path is not None:
        ap = Path(actuals_path)
        if ap.exists():
            actuals_df = pd.read_csv(ap, parse_dates=["MonthStart"])
            actuals_df["MonthStart"] = (
                actuals_df["MonthStart"].dt.to_period("M").dt.to_timestamp()
            )

            holdout_eval, holdout_preds = score_segmented_holdout(
                sku_fc=seg_result["sku_forecasts"],
                actuals_df=actuals_df,
                holdout_months=holdout_months,
            )

            error_decomp = build_segmented_error_decomp(
                actuals_df=actuals_df,
                dim_product_df=dim_prod_df,
                scode_fc=scode_forecasts,
                scol_fc=seg_result["stylecolor_forecasts"],
                sku_fc=seg_result["sku_forecasts"],
                holdout_months=holdout_months,
            )

            if output_dir is not None:
                holdout_eval.to_csv(
                    output_dir / "v7_3_segmented_holdout_evaluation.csv", index=False
                )
                holdout_preds.to_csv(
                    output_dir / "v7_3_segmented_holdout_predictions.csv", index=False
                )
                error_decomp.to_csv(
                    output_dir / "v7_3_segmented_error_decomposition.csv", index=False
                )
                logger.info("[v7.3] Holdout evaluation and error decomposition saved.")

    return {
        "scode_forecasts":       scode_forecasts,
        "segmented_result":      seg_result,
        "holdout_eval":          holdout_eval,
        "holdout_preds":         holdout_preds,
        "error_decomp":          error_decomp,
        "strategy_map_combined": strategy_map_combined,
        "actuals_df":            actuals_df,
    }


# ===========================================================================
# v7.4 PRODUCTION CANDIDATE PIPELINE
# ===========================================================================
# run_v7_4_production() orchestrates the full v7.4 pipeline:
#   1. StyleCode-level prep and forecasting (v7.0 base)
#   2. Optional StyleCode calibration layer (v7.4 NEW)
#   3. Production allocation using v7.2 recency_only (proven champion)
#   4. Build all production output files
# ===========================================================================


def run_v7_4_production(
    data_dir,
    best_models_df: pd.DataFrame,
    horizon_months: int,
    forecast_start: str = "2026-01-01",
    n_forecast_months: int = 2,
    phase: int = 1,
    adjustment_config: dict | None = None,
    output_dir: str | Path | None = None,
    actuals_path: str | Path | None = None,
    holdout_months: list | None = None,
    backtest_predictions_df: pd.DataFrame | None = None,
    apply_calibration: bool = True,
    lookback_months: int     = 12,
    min_lookback_months: int = 6,
    w_recent: int = 3,
    w_mid: int    = 2,
    w_old: int    = 1,
) -> dict:
    """
    v7.4 Production Candidate pipeline.

    Steps
    -----
    1. run_v7_prep() — StyleCode-level panel
    2. run_forecasts() — raw StyleCode forecasts
    3. build_stylecode_calibration_table() — calibration from backtest data
    4. apply_stylecode_calibration() — adjust raw forecasts
    5. run_allocation_variant("recency_only") — production allocation
    6. Build all production output files

    Parameters
    ----------
    data_dir                 : Gold CSV folder
    best_models_df           : output of run_cv() (StyleCode-level)
    horizon_months           : 1 or 3
    forecast_start           : first forecast month
    n_forecast_months        : months to forecast
    phase                    : 1 (train through 2025-12)
    adjustment_config        : v6.2 adjustment config or None
    output_dir               : folder for output CSVs (None = skip writes)
    actuals_path             : path to gold actuals for holdout scoring
    holdout_months           : list of pd.Timestamp (default Jan–Feb 2026)
    backtest_predictions_df  : CV fold-level predictions for calibration.
                               Must contain [StyleCodeDesc or SKU, MonthStart,
                               HorizonMonths, ActualUnits, PredictedUnits].
                               Pass None to use historical fallback calibration.
    apply_calibration        : if False, calibration is built but not applied
    lookback_months / min_lookback_months / w_recent / w_mid / w_old :
                               forwarded to allocation share computation

    Returns
    -------
    dict with keys:
        scode_raw_fc, scode_calibrated_fc, calibration_table,
        calibration_validation,
        stylecolor_fc, sku_fc,
        production_sku_table, validation_report,
        holdout_eval, holdout_preds, error_decomp,
        risk_flags
    """
    from forecasting_pipeline.data_prep import _resolve_training_end
    from forecasting_pipeline.forecast_calibration_v74 import (
        build_stylecode_calibration_table,
        apply_stylecode_calibration,
        validate_calibration_table,
    )
    from forecasting_pipeline.allocation_v72 import (
        ALLOCATION_VARIANTS,
        run_allocation_variant,
    )
    from forecasting_pipeline.production_outputs_v74 import (
        build_production_sku_table,
        build_forecast_risk_flags,
        build_production_validation_report,
        build_error_decomposition,
        score_holdout,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if holdout_months is None:
        holdout_months = [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]

    # ── Step 1: v7.0 prep ────────────────────────────────────────────────
    logger.info("=== v7.4 Production: upstream prep (phase=%d, H=%d) ===", phase, horizon_months)
    scode_prep = run_v7_prep(
        data_dir=data_dir,
        phase=phase,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
    )

    tables      = scode_prep["tables"]
    gold_df     = tables["demand"]
    dim_prod_df = scode_prep["dim_product"]
    train_end   = _resolve_training_end(tables["dim_date"], phase=1)

    # ── Step 2: Raw StyleCode forecasts ──────────────────────────────────
    scode_raw_fc = run_forecasts(
        prep=scode_prep,
        best_models_df=best_models_df,
        horizon_months=horizon_months,
        forecast_start=forecast_start,
        n_forecast_months=n_forecast_months,
        phase=phase,
        model_version=f"v7.4-raw-H{horizon_months}",
        output_path=None,
        append=False,
        adjustment_config=adjustment_config,
    )
    logger.info(
        "[v7.4] Raw StyleCode forecasts: %d rows, %d StyleCodes",
        len(scode_raw_fc), scode_raw_fc["Key"].nunique(),
    )

    if output_dir is not None:
        scode_raw_fc.to_csv(
            output_dir / f"v7_4_stylecode_forecasts_raw_H{horizon_months}.csv", index=False
        )

    # ── Step 3: Calibration table ─────────────────────────────────────────
    backtest_end = min(pd.Timestamp(m) for m in holdout_months) - pd.DateOffset(months=1)
    backtest_end = pd.Timestamp(backtest_end.year, backtest_end.month, 1)

    calibration_table = build_stylecode_calibration_table(
        backtest_predictions_df=backtest_predictions_df,
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        backtest_end=backtest_end,
        horizon_months_list=[horizon_months],
    )

    calib_val = validate_calibration_table(calibration_table)
    logger.info(
        "[v7.4] Calibration: %d rows, %d applied, tiers=%s",
        calib_val["n_rows"], calib_val["n_applied"], calib_val["n_by_tier"],
    )

    if output_dir is not None:
        calibration_table.to_csv(
            output_dir / "v7_4_stylecode_calibration_table.csv", index=False
        )

    # ── Step 4: Apply calibration ─────────────────────────────────────────
    if apply_calibration and not calibration_table.empty:
        scode_calibrated_fc = apply_stylecode_calibration(
            scode_forecasts_df=scode_raw_fc,
            calibration_df=calibration_table,
        )
    else:
        scode_calibrated_fc = scode_raw_fc.copy()
        scode_calibrated_fc["CalibrationFactor"]  = 1.0
        scode_calibrated_fc["CalibrationApplied"] = False
        logger.info("[v7.4] Calibration skipped (apply_calibration=%s)", apply_calibration)

    if output_dir is not None:
        scode_calibrated_fc.to_csv(
            output_dir / f"v7_4_stylecode_forecasts_calibrated_H{horizon_months}.csv", index=False
        )

    # ── Step 5: STANDALONE pass-through ───────────────────────────────────
    standalone_skus = scode_prep.get("stylecode_standalone_skus", [])
    standalone_fc   = pd.DataFrame()

    if standalone_skus:
        from forecasting_pipeline.data_prep    import build_panel
        from forecasting_pipeline.segmentation import segment_skus, attach_segment

        sa_gold = gold_df[gold_df["SKU"].isin(standalone_skus)].copy()
        if not sa_gold.empty:
            sa_panel     = build_panel(sa_gold, tables["dim_date"], phase=phase, dim_product_df=dim_prod_df)
            sa_segs      = segment_skus(sa_panel)
            sa_panel_seg = attach_segment(sa_panel, sa_segs)
            sa_prep = {
                "tables":tables, "panel":sa_panel,
                "segments":sa_segs, "panel_seg":sa_panel_seg,
            }
            standalone_fc = run_forecasts(
                prep=sa_prep,
                best_models_df=best_models_df,
                horizon_months=horizon_months,
                forecast_start=forecast_start,
                n_forecast_months=n_forecast_months,
                phase=phase,
                model_version=f"v7.4-standalone-H{horizon_months}",
                output_path=None,
                append=False,
                adjustment_config=adjustment_config,
            )

    # ── Step 6: Production allocation — v7.2 recency_only ────────────────
    logger.info("[v7.4] Running production allocation: recency_only")
    recency_cfg = ALLOCATION_VARIANTS["recency_only"]

    alloc_result = run_allocation_variant(
        variant_name="recency_only",
        variant_cfg=recency_cfg,
        scode_forecasts_df=scode_calibrated_fc,
        gold_df=gold_df,
        dim_product_df=dim_prod_df,
        train_end=train_end,
        standalone_fc_df=standalone_fc if not standalone_fc.empty else None,
        lookback_months=lookback_months,
        min_lookback_months=min_lookback_months,
        w_recent=w_recent,
        w_mid=w_mid,
        w_old=w_old,
        smooth_alpha=0.0,
        cap_rel_increase=999.0,  # effectively disabled for recency_only
    )

    scol_fc = alloc_result["stylecolor_forecasts"]
    sku_fc  = alloc_result["sku_forecasts"]

    logger.info(
        "[v7.4] Allocation complete: %d StyleColor rows, %d SKU rows",
        len(scol_fc), len(sku_fc),
    )

    # ── Step 7: Production SKU table ──────────────────────────────────────
    prod_sku_table = build_production_sku_table(
        sku_fc_df=sku_fc,
        dim_product_df=dim_prod_df,
        calibration_df=calibration_table if not calibration_table.empty else None,
        allocation_method="recency_only_v7.2",
    )

    if output_dir is not None:
        prod_sku_table.to_csv(
            output_dir / "v7_4_production_sku_forecasts.csv", index=False
        )
        logger.info("[v7.4] Saved v7_4_production_sku_forecasts.csv (%d rows)", len(prod_sku_table))

    # ── Step 8: Validation report ─────────────────────────────────────────
    val_report = build_production_validation_report(
        scode_fc=scode_calibrated_fc,
        scol_fc=scol_fc,
        sku_fc=sku_fc,
        calibration_df=calibration_table if not calibration_table.empty else None,
    )
    if output_dir is not None:
        val_report.to_csv(
            output_dir / "v7_4_production_validation_report.csv", index=False
        )

    # ── Step 9: Holdout evaluation ────────────────────────────────────────
    actuals_df   = None
    holdout_eval = pd.DataFrame()
    holdout_preds= pd.DataFrame()
    error_decomp = pd.DataFrame()
    risk_flags   = pd.DataFrame()

    if actuals_path is not None:
        ap = Path(actuals_path)
        if ap.exists():
            actuals_df = pd.read_csv(ap, parse_dates=["MonthStart"])
            actuals_df["MonthStart"] = (
                actuals_df["MonthStart"].dt.to_period("M").dt.to_timestamp()
            )

            holdout_eval, holdout_preds = score_holdout(
                sku_fc=prod_sku_table,
                actuals_df=actuals_df,
                holdout_months=holdout_months,
            )

            error_decomp = build_error_decomposition(
                actuals_df=actuals_df,
                dim_product_df=dim_prod_df,
                scode_fc=scode_calibrated_fc,
                scol_fc=scol_fc,
                sku_fc=sku_fc,
                holdout_months=holdout_months,
            )

            risk_flags = build_forecast_risk_flags(
                sku_fc_df=prod_sku_table,
                gold_df=gold_df,
                dim_product_df=dim_prod_df,
                calibration_df=calibration_table if not calibration_table.empty else None,
                holdout_months=holdout_months,
            )

            if output_dir is not None:
                holdout_eval.to_csv(
                    output_dir / "v7_4_holdout_evaluation.csv", index=False
                )
                holdout_preds.to_csv(
                    output_dir / "v7_4_holdout_predictions.csv", index=False
                )
                error_decomp.to_csv(
                    output_dir / "v7_4_error_decomposition.csv", index=False
                )
                risk_flags.to_csv(
                    output_dir / "v7_4_forecast_risk_flags.csv", index=False
                )
                logger.info("[v7.4] All holdout and risk outputs saved.")

    return {
        "scode_raw_fc":          scode_raw_fc,
        "scode_calibrated_fc":   scode_calibrated_fc,
        "calibration_table":     calibration_table,
        "calibration_validation":calib_val,
        "stylecolor_fc":         scol_fc,
        "sku_fc":                sku_fc,
        "production_sku_table":  prod_sku_table,
        "validation_report":     val_report,
        "holdout_eval":          holdout_eval,
        "holdout_preds":         holdout_preds,
        "error_decomp":          error_decomp,
        "risk_flags":            risk_flags,
        "actuals_df":            actuals_df,
    }
