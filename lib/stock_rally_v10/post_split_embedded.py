"""
Training, Meta-Modelle, Kalibrierung, Plots und HTML-Export.

Implementierung: ``training_phases`` (benannte Funktionen). Dieses Modul re-exportiert
dieselbe API für bestehende Imports (``from post_split_embedded import run_training_...``).
"""
from __future__ import annotations

from lib.stock_rally_v10.training_phases import (
    run_phase_daily_scoring_html,
    run_phase_holdout_visualization,
    run_phase_meta_learner_and_threshold,
    run_phase_optuna_base_models,
    run_phase_regime_benchmark_report,
    run_phase_threshold_pr_and_filters,
    run_training_scoring_and_export,
)

# Kurz-Aliasse (optional)
run_optuna_base_models = run_phase_optuna_base_models
run_meta_learner_and_threshold = run_phase_meta_learner_and_threshold
run_regime_benchmark_report = run_phase_regime_benchmark_report
run_threshold_pr_and_filters = run_phase_threshold_pr_and_filters
run_holdout_visualization = run_phase_holdout_visualization
run_daily_scoring_html = run_phase_daily_scoring_html

__all__ = [
    "run_training_scoring_and_export",
    "run_phase_optuna_base_models",
    "run_phase_meta_learner_and_threshold",
    "run_phase_regime_benchmark_report",
    "run_phase_threshold_pr_and_filters",
    "run_phase_holdout_visualization",
    "run_phase_daily_scoring_html",
    "run_optuna_base_models",
    "run_meta_learner_and_threshold",
    "run_regime_benchmark_report",
    "run_threshold_pr_and_filters",
    "run_holdout_visualization",
    "run_daily_scoring_html",
]
