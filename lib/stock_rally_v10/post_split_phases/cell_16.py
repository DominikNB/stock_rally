if globals().get('SCORING_ONLY', False):
    print('[SCORING_ONLY] Training-Zelle übersprungen.')
else:
    # Cell 15 — 8d: Holdout plots (FINAL tickers)
    # Unbiased OOS: nur FINAL — df_test ist TRAIN_META (Meta-Training, kein echter Holdout für Performance).

    # Build filtered signals dict for FINAL tickers
    filtered_signals_final = {}
    for ticker in final_tickers:
        sub = df_final[df_final['ticker'] == ticker]
        filtered_signals_final[ticker] = apply_signal_filters(sub, best_threshold)

    signal_target_diag = summarize_filtered_signals_vs_target(
        df_final, filtered_signals_final, tickers=final_tickers)

    # Visualise
    plot_holdout_results(
        df_final,
        final_tickers,
        filtered_signals_final,
        title=f'FINAL Holdout — Threshold={best_threshold:.2f}'
    )
    print('Forward-Return-/Qualitätsanalyse: signals_holdout_final in signals.json (nicht die volle Historie "signals").')
