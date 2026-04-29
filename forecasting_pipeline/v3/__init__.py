"""
lane7_forecast
==============
Modular demand forecasting pipeline for Lane Seven Apparel.
v7.4 — Production Candidate.
"""

# Core stable modules
from lane7_forecast.data_prep import (
    load_gold_tables, build_panel, build_stylecolor_panel, build_stylecode_panel,
)
from lane7_forecast.segmentation import segment_skus, attach_segment, get_segment_skus
from lane7_forecast.features     import create_features, get_feature_columns
from lane7_forecast.models import (
    SEGMENT_MODEL_ROSTER, train_ml_model, predict_ml, predict_baseline,
    fit_predict_prophet, fit_predict_neuralprophet, fit_predict_sarima,
)
from lane7_forecast.evaluation import (
    walk_forward_cv, compute_metrics, select_best_model,
    score_holdout_for_model, compute_scenario_bounds,
    build_scenario_bounds_from_simulation,
)
from lane7_forecast.forecasting import (
    generate_forecasts, write_forecasts, append_forecasts, build_future_features,
)
from lane7_forecast.pipeline import (
    run_data_prep, run_cv, run_forecasts, run_simulation, build_simulation_summaries,
)

# v6
from lane7_forecast.pipeline import run_hierarchical_prep, run_hierarchical_forecasts
from lane7_forecast.allocation import (
    build_stylecolor_demand, compute_size_shares, allocate_to_sku,
    get_standalone_skus, validate_allocation,
)
try:
    from lane7_forecast.holdout_v6 import run_v6_holdout, compare_v6_vs_v52
except Exception:
    pass

# v6.2
from lane7_forecast.forecast_adjustments import (
    FORECAST_ADJUSTMENT_CONFIG, get_config as get_adjustment_config,
    apply_shrinkage, apply_intermittent_cap,
    blend_ml_with_seasonal, stabilize_lag_for_recursion,
)

# v7
from lane7_forecast.pipeline import run_v7_prep, run_v7_forecasts
from lane7_forecast.stylecode_allocation import (
    build_stylecode_demand, compute_stylecolor_shares, allocate_to_stylecolor,
    validate_stylecode_allocation, get_v7_standalone_skus, build_v7_coverage_report,
)

# v7.1 (guarded)
try:
    from lane7_forecast.pipeline import run_v7_1_prep, run_v7_1_forecasts
except Exception: pass
try:
    from lane7_forecast.allocation import compute_size_shares_v71, allocate_to_sku_v71
except Exception: pass
try:
    from lane7_forecast.stylecode_allocation import (
        compute_stylecolor_shares_v71, allocate_to_stylecolor_v71,
    )
except Exception: pass
try:
    from lane7_forecast.allocation_v71_utils import (
        compute_recency_weighted_shares, build_stylecode_alloc_diagnostics,
        build_stylecolor_share_analysis, build_sku_alloc_diagnostics,
        build_error_decomposition,
    )
except Exception: pass

# v7.2 (guarded)
try:
    from lane7_forecast.pipeline import run_v7_2_ablation
except Exception: pass
try:
    from lane7_forecast.allocation_v72 import (
        ALLOCATION_VARIANTS, compute_shares_for_variant, run_allocation_variant,
        run_all_variants, build_variant_comparison, validate_variant_allocation,
    )
except Exception: pass

# v7.3 (guarded)
try:
    from lane7_forecast.pipeline import run_v7_3_segmented
except Exception: pass
try:
    from lane7_forecast.allocation_strategy_selector import (
        ALLOCATION_STRATEGIES, PROFILE_STRATEGY, DEFAULT_THRESHOLDS,
        classify_parent_allocation_strategy, compute_segmented_shares,
        run_segmented_allocation, score_segmented_holdout,
        build_segmented_error_decomp, build_parent_win_loss,
        validate_segmented_allocation,
    )
except Exception: pass

# v7.4 Production Candidate
try:
    from lane7_forecast.pipeline import run_v7_4_production
except Exception: pass
try:
    from lane7_forecast.forecast_calibration_v74 import (
        build_stylecode_calibration_table,
        apply_stylecode_calibration,
        validate_calibration_table,
    )
except Exception: pass
try:
    from lane7_forecast.production_outputs_v74 import (
        build_production_sku_table,
        build_forecast_risk_flags,
        build_production_validation_report,
        build_error_decomposition as build_v74_error_decomposition,
        score_holdout as score_v74_holdout,
        build_version_comparison,
    )
except Exception: pass

__version__ = "7.4.0"
# v7.5 additions
try:
    from lane7_forecast.backtest_calibration_v75 import (
        run_stylecode_backtest,
        build_v75_calibration_table,
        apply_v75_calibration,
        build_bias_analysis,
        validate_calibration_table as validate_v75_calibration_table,
    )
except Exception:
    pass
try:
    from lane7_forecast.production_outputs_v75 import (
        build_v75_production_sku_table,
        score_v75_holdout,
        build_v75_error_decomposition,
        build_v75_validation_report,
        build_v75_version_comparison,
    )
except Exception:
    pass
# v7.6 additions
try:
    from lane7_forecast.global_bias_control_v76 import (
        build_global_calibration_table,
        apply_global_calibration,
        build_v76_bias_analysis,
        validate_global_calibration_table,
    )
except Exception:
    pass
try:
    from lane7_forecast.production_outputs_v76 import (
        build_v76_production_sku_table,
        score_v76_holdout,
        build_v76_error_decomposition,
        build_v76_validation_report,
        build_v76_version_comparison,
    )
except Exception:
    pass
