"""
Nach-Split-Training: benannte Phasen (12–17) als echte Python-Funktionen.

Module: ``optuna_base_models``, ``meta_learner``, ``regime``, ``threshold_pr_filters``,
``holdout``, ``daily_scoring_html``.
"""
from __future__ import annotations

from typing import Any

from lib.stock_rally_v10.training_phases.daily_scoring_html import run_phase_daily_scoring_html
from lib.stock_rally_v10.training_phases.holdout import run_phase_holdout_visualization
from lib.stock_rally_v10.training_phases.meta_learner import run_phase_meta_learner_and_threshold
from lib.stock_rally_v10.training_phases.optuna_base_models import run_phase_optuna_base_models
from lib.stock_rally_v10.training_phases.regime import run_phase_regime_benchmark_report
from lib.stock_rally_v10.training_phases.threshold_pr_filters import run_phase_threshold_pr_and_filters


def _cfg_module(cfg_mod: Any | None):
    if cfg_mod is not None:
        return cfg_mod
    from lib.stock_rally_v10 import config as cfg

    return cfg


def run_training_scoring_and_export(cfg_mod: Any | None = None) -> None:
    """Volle Kette: Phase 12 → 17."""
    c = _cfg_module(cfg_mod)
    run_phase_optuna_base_models(c)
    run_phase_meta_learner_and_threshold(c)
    run_phase_regime_benchmark_report(c)
    run_phase_threshold_pr_and_filters(c)
    run_phase_holdout_visualization(c)
    run_phase_daily_scoring_html(c)


__all__ = [
    "run_training_scoring_and_export",
    "run_phase_optuna_base_models",
    "run_phase_meta_learner_and_threshold",
    "run_phase_regime_benchmark_report",
    "run_phase_threshold_pr_and_filters",
    "run_phase_holdout_visualization",
    "run_phase_daily_scoring_html",
    "_cfg_module",
]
