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


def _log_training_partition_calendar(c: Any) -> None:
    """Min/Max-Datum und Zeilenzahl pro Partition — zu Beginn des Trainings."""
    import pandas as pd

    rows = (
        ("BASE (Base-Optuna & Base-Modelle)", getattr(c, "df_train", None)),
        ("META (Meta-Learner)", getattr(c, "df_test", None)),
        ("THRESHOLD (Schwellen-Kalibrierung)", getattr(c, "df_threshold", None)),
        ("FINAL (OOS-Eval)", getattr(c, "df_final", None)),
    )
    print("=" * 72, flush=True)
    print(
        "Trainings-/Kalibrierungs-Fenster (Wiederholung vor Phase 12 / Base-Optuna)",
        flush=True,
    )
    print("=" * 72, flush=True)
    _sm = getattr(c, "SPLIT_MODE", None)
    if _sm is not None:
        print(f"  SPLIT_MODE={_sm}", flush=True)
    any_rows = False
    for label, df in rows:
        if df is None or not hasattr(df, "columns") or "Date" not in df.columns or len(df) == 0:
            print(f"  {label}: — (keine Daten am cfg)", flush=True)
            continue
        any_rows = True
        d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        d0, d1 = d.min().date(), d.max().date()
        n_days = int(d.nunique())
        print(f"  {label}: {d0} … {d1}  |  {n_days} Handelstage  |  {len(df):,} Zeilen", flush=True)
    if not any_rows:
        print(
            "  Hinweis: Keine Split-DataFrames — typisch bei SCORING_ONLY ohne vorherigen Split.",
            flush=True,
        )
    print("=" * 72, flush=True)


def run_training_scoring_and_export(cfg_mod: Any | None = None) -> None:
    """Volle Kette: Phase 12 → 17."""
    c = _cfg_module(cfg_mod)
    _log_training_partition_calendar(c)
    retrain_meta_only = bool(getattr(c, "RETRAIN_META_ONLY", False))
    scoring_only = bool(getattr(c, "SCORING_ONLY", False))
    if scoring_only and retrain_meta_only:
        print(
            "Hinweis: SCORING_ONLY=True hat Vorrang; RETRAIN_META_ONLY wird ignoriert.",
            flush=True,
        )
        retrain_meta_only = False
    if retrain_meta_only:
        print(
            "Phase 12 übersprungen: RETRAIN_META_ONLY=True — lade Base-Modelle/Parameter aus Artefakt.",
            flush=True,
        )
        c.load_scoring_artifacts()
    else:
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
