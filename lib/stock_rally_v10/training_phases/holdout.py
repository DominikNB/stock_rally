"""Phase 16: Holdout-Plots auf dem FINAL-Set."""
from __future__ import annotations

from typing import Any


def run_phase_holdout_visualization(cfg_mod: Any | None = None) -> None:
    if cfg_mod is None:
        from lib.stock_rally_v10 import config as cfg_mod

    if getattr(cfg_mod, "SCORING_ONLY", False):
        print("[SCORING_ONLY] Training-Zelle übersprungen.")
        return

    df_final = cfg_mod.df_final
    final_tickers = cfg_mod.final_tickers
    best_threshold = cfg_mod.best_threshold
    apply_signal_filters = cfg_mod.apply_signal_filters
    plot_holdout_results = cfg_mod.plot_holdout_results
    summarize_filtered_signals_vs_target = cfg_mod.summarize_filtered_signals_vs_target

    filtered_signals_final = {}
    for ticker in final_tickers:
        sub = df_final[df_final["ticker"] == ticker]
        filtered_signals_final[ticker] = apply_signal_filters(sub, best_threshold)

    cfg_mod.signal_target_diag = summarize_filtered_signals_vs_target(
        df_final, filtered_signals_final, tickers=final_tickers
    )

    plot_holdout_results(
        df_final,
        final_tickers,
        filtered_signals_final,
        title=f"FINAL Holdout — Threshold={best_threshold:.2f}",
    )
    print(
        "Forward-Return-/Qualitätsanalyse: signals_holdout_final in signals.json "
        '(nicht die volle Historie "signals").',
        flush=True,
    )
