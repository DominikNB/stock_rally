"""stock_rally_v10 — Holdout-Plots und Signal-Diagnose (Pipeline-Modul)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from lib.stock_rally_v10 import config as cfg

def plot_holdout_results(df_final, final_tickers, filtered_signals, title='FINAL Holdout'):
    """
    For each ticker in final_tickers: plot close price, rally periods,
    target days, and model buy signals.
    """
    n = len(final_tickers)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, ticker in zip(axes, final_tickers):
        sub = df_final[df_final['ticker'] == ticker].sort_values('Date')
        if sub.empty:
            ax.set_title(f'{ticker} — no data')
            continue

        dates = sub['Date'].values
        close = sub['close'].values

        # Close price
        ax.plot(dates, close, color='black', linewidth=0.8, label='Close')

        # Rally shading
        rally_mask = sub['rally'].values == 1
        in_rally = False
        rally_start = None
        for i, (d, r) in enumerate(zip(dates, rally_mask)):
            if r and not in_rally:
                rally_start = d
                in_rally = True
            elif not r and in_rally:
                ax.axvspan(rally_start, dates[i - 1], alpha=0.15, color='green')
                in_rally = False
        if in_rally:
            ax.axvspan(rally_start, dates[-1], alpha=0.15, color='green')

        # Target days (orange dots)
        target_mask = sub['target'].values == 1
        ax.scatter(dates[target_mask], close[target_mask],
                   color='orange', s=20, zorder=3, label='Target day')

        # Model signals (red triangles)
        sig_dates = filtered_signals.get(ticker, np.array([], dtype='datetime64[ns]'))
        if len(sig_dates) > 0:
            sig_days = {pd.Timestamp(x).normalize().date() for x in sig_dates}
            row_dates = sub['Date'].map(lambda x: pd.Timestamp(x).normalize().date())
            sig_sub = sub[row_dates.isin(sig_days)]
            ax.scatter(sig_sub['Date'].values, sig_sub['close'].values,
                       marker='^', color='red', s=60, zorder=4, label='Buy signal')

        ax.set_title(f"{ticker} — {cfg.COMPANY_NAMES.get(ticker, ticker)}", fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(loc='upper left', fontsize=7)
        ax.set_ylabel('Price', fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.001)
    plt.tight_layout()
    plt.show()


def _rows_for_signal_calendar_day(sub, d):
    """Zeilen zu Signaldatum d (Kalendertag; robust datetime64 vs Timestamp)."""
    sig_day = pd.Timestamp(d).normalize().date()
    m = sub['Date'].map(lambda x: pd.Timestamp(x).normalize().date()) == sig_day
    return sub.loc[m]


def _offset_from_rally_start(rally):
    """Handelstage seit Segmentbeginn (0 = erster Rally-Tag). NaN außerhalb Rally."""
    rally = np.asarray(rally, dtype=np.int8)
    n = len(rally)
    off = np.full(n, np.nan)
    i = 0
    while i < n:
        if rally[i] != 1:
            i += 1
            continue
        s = i
        while i < n and rally[i] == 1:
            off[i] = i - s
            i += 1
    return off


def _next_rally_start_idx(rally):
    """Index des nächsten Rally-Starts (erster grüner Tag ab inkl. i), sonst -1."""
    rally = np.asarray(rally, dtype=np.int8)
    n = len(rally)
    nxt = np.full(n, -1, dtype=np.int32)
    next_s = -1
    for i in range(n - 1, -1, -1):
        if rally[i] == 1 and (i == 0 or rally[i - 1] == 0):
            next_s = i
        nxt[i] = next_s
    return nxt


def summarize_filtered_signals_vs_target(df, filtered_signals, tickers=None):
    """
    Gefilterte Modell-Signale vs. target (positives Label) und Lage zur Rally.
    target==1: Vorlauf + Einstiegszone nur mit genug Rally-Rest (cfg.MIN_RALLY_TAIL_DAYS); sonst target=0.
    """
    if tickers is None:
        tickers = list(filtered_signals.keys())
    rows = []
    for ticker in tickers:
        sig_dates = filtered_signals.get(ticker)
        if sig_dates is None or len(sig_dates) == 0:
            continue
        sub = df[df['ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        if sub.empty:
            continue
        rally = sub['rally'].values.astype(np.int8)
        tgt = sub['target'].values.astype(np.int8)
        off = _offset_from_rally_start(rally)
        nxt = _next_rally_start_idx(rally)
        for d in sig_dates:
            hit = _rows_for_signal_calendar_day(sub, d)
            if hit.empty:
                continue
            j = int(hit.index[0])
            r0, t0 = int(rally[j]), int(tgt[j])
            row = dict(
                ticker=ticker,
                date=str(pd.Timestamp(sub['Date'].iloc[j]).date()),
                target=t0,
                rally=r0,
            )
            if r0 == 1 and not np.isnan(off[j]):
                row['days_from_rally_start'] = int(off[j])
            else:
                row['days_from_rally_start'] = None
            if r0 == 0 and t0 == 1 and nxt[j] >= 0:
                row['days_to_rally_start'] = int(nxt[j] - j)
            else:
                row['days_to_rally_start'] = None
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        print('summarize_filtered_signals_vs_target: keine Signalzeilen.')
        return out
    n_one = int((out['target'] == 1).sum())
    n_zero = int((out['target'] == 0).sum())
    print(f'\nGefilterte Signale vs. target:  target=1 → {n_one}  target=0 → {n_zero}  (n={len(out)})')
    if len(out):
        print(f'  Anteil auf Label-Positiv (target=1): {n_one / len(out):.1%}')
    g = out.groupby('ticker', as_index=False).agg(
        n=('target', 'size'),
        on_target=('target', lambda s: int((s == 1).sum())),
    )
    g['on_target_pct'] = (g['on_target'] / g['n'] * 100).round(1)
    print('\nPro Ticker (gefilterte Signale):')
    print(g.to_string(index=False))
    late = out[(out['target'] == 0) & (out['rally'] == 1)]
    if len(late) > 0 and late['days_from_rally_start'].notna().any():
        print('\nSignale mit target=0, rally=1 (spät in grüner Phase) — Tage seit Rally-Start:')
        print(late[['ticker', 'date', 'days_from_rally_start']].to_string(index=False))
    return out


def apply_signal_filters(df_ticker_prob, threshold,
                         consecutive_days=None, signal_cooldown_days=None):
    """
    Apply consecutive filter then cooldown filter for a single ticker.
    consecutive_days / signal_cooldown_days default to the global constants
    (which are updated after Optuna with the optimised values).
    Optional: skip signals at local price peaks (near N-day high) and/or very high RSI.
    Returns array of Date values where filtered signals fire.
    """
    cd  = cfg.CONSECUTIVE_DAYS     if consecutive_days     is None else consecutive_days
    scd = cfg.SIGNAL_COOLDOWN_DAYS if signal_cooldown_days is None else signal_cooldown_days

    df_s = df_ticker_prob.sort_values('Date').reset_index(drop=True)
    raw  = (df_s['prob'].values >= threshold).astype(np.int8)
    n    = len(raw)

    consec = np.zeros(n, dtype=np.int8)
    for i in range(2, n):
        if raw[i-2] + raw[i-1] + raw[i] >= cd:
            consec[i] = 1

    final = np.zeros(n, dtype=np.int8)
    last_signal = -999
    for i in range(n):
        if consec[i] == 1 and (i - last_signal) >= scd:
            final[i] = 1
            last_signal = i

    if 'close' in df_s.columns:
        rw = cfg.__dict__.get('rsi_w')
        rsi_series = _rsi_from_close_1d(df_s['close'].values, rw)
        mask_ok = _peak_rsi_mask_1d(
            df_s['close'].values,
            rsi_series,
            bool(cfg.__dict__.get('cfg.SIGNAL_SKIP_NEAR_PEAK', False)),
            int(cfg.__dict__.get('cfg.PEAK_LOOKBACK_DAYS', 20)),
            float(cfg.__dict__.get('cfg.PEAK_MIN_DIST_FROM_HIGH_PCT', 0.012)),
            cfg.__dict__.get('cfg.SIGNAL_MAX_RSI', None),
        )
        for i in range(n):
            if final[i] == 1 and not mask_ok[i]:
                final[i] = 0

    return df_s.loc[final == 1, 'Date'].values