"""stock_rally_v10 — Target-Aufbau (Pipeline-Modul)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg

def _create_target_one_ticker(df_ticker, lead_days=None, entry_days=None,
                              return_window=None, rally_threshold=None,
                              min_rally_tail_days=None):
    """
    For a single ticker DataFrame (sorted by Date), compute:
      - rally[]: 1 where the return_window-day cumulative return >= rally_threshold
      - target[]: 1 in Vorlauf + Einstiegszone nur, wenn die Rally-Segmentlänge es hergibt:
        ab dem jeweiligen Tag müssen noch mindestens min_rally_tail_days Rally-Handelstage
        bis zum Segmentende folgen (damit nach Einstieg noch genug grüne Phase übrig ist).
    Params default to globals when None.
    """
    ld = cfg.LEAD_DAYS       if lead_days       is None else lead_days
    ed = cfg.ENTRY_DAYS      if entry_days      is None else entry_days
    mt = cfg.MIN_RALLY_TAIL_DAYS if min_rally_tail_days is None else int(min_rally_tail_days)
    rw = cfg.RETURN_WINDOW   if return_window   is None else return_window
    rt = cfg.RALLY_THRESHOLD if rally_threshold is None else rally_threshold

    close = df_ticker['close'].values.astype(np.float64)
    n = len(close)
    window = rw

    # Rolling cumulative return (product of (1+daily_ret) over window)
    daily_ret = np.full(n, np.nan)
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    cum_ret = np.full(n, np.nan)
    for i in range(window - 1, n):
        product = 1.0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(daily_ret[j]):
                product *= (1.0 + daily_ret[j])
        cum_ret[i] = product - 1.0

    # Mark rally periods (backwards from qualifying end-day)
    rally = np.zeros(n, dtype=np.int8)
    for end_idx in range(n):
        if not np.isnan(cum_ret[end_idx]) and cum_ret[end_idx] >= rt:
            start_idx = max(0, end_idx - window + 1)
            rally[start_idx:end_idx + 1] = 1

    # Vorlauf + Einstiegszone nur, wenn genug Rally „Restlauf“ (kein positives Label am Rally-Ende)
    target = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        if rally[i] != 1:
            i += 1
            continue
        if i > 0 and rally[i - 1] == 1:
            i += 1
            continue
        start = i
        j = i
        while j < n and rally[j] == 1:
            j += 1
        end = j - 1
        seg_len = end - start + 1
        pre_start = max(0, start - ld)
        if seg_len >= mt:
            for k in range(pre_start, start):
                target[k] = 1
        for k in range(start, min(n, end + 1, start + ed)):
            if end - k + 1 >= mt:
                target[k] = 1
        i = j

    return rally, target


def _create_target_one_ticker_fixed_bands(df_ticker):
    """
    Feste Label-Regel (wenn cfg.OPT_OPTIMIZE_Y_TARGETS False):
      Grüner Bereich: Es gibt ein Fenster w in [cfg.FIXED_Y_WINDOW_MIN, cfg.FIXED_Y_WINDOW_MAX],
      über das die kumulierte Rendite (Produkt der Tagesrenditen) >= cfg.FIXED_Y_RALLY_THRESHOLD ist;
      alle Tage dieses Fensters gehören zur grünen Phase (Vereinigung wie bisher).

      Pro zusammenhängendem grünen Segment [start, end] mit Länge L:
        L < cfg.FIXED_Y_SEGMENT_SPLIT: Target = 3 Tage vor start + erste 2 grüne Tage.
        L >= cfg.FIXED_Y_SEGMENT_SPLIT: Target = 3 Tage vor start + alle grünen Tage außer den letzten 3.
    """
    w_lo, w_hi, rt, split = cfg.fixed_y_rule_params()

    close = df_ticker["close"].values.astype(np.float64)
    n = len(close)
    daily_ret = np.full(n, np.nan)
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    rally = np.zeros(n, dtype=np.int8)
    for end_idx in range(n):
        for w in range(w_lo, w_hi + 1):
            if end_idx < w - 1:
                continue
            product = 1.0
            bad = False
            for j in range(end_idx - w + 1, end_idx + 1):
                dr = daily_ret[j]
                if np.isnan(dr):
                    bad = True
                    break
                product *= 1.0 + dr
            if bad:
                continue
            if product - 1.0 >= rt:
                start_idx = max(0, end_idx - w + 1)
                rally[start_idx : end_idx + 1] = 1

    target = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        if rally[i] != 1:
            i += 1
            continue
        if i > 0 and rally[i - 1] == 1:
            i += 1
            continue
        start = i
        j = i
        while j < n and rally[j] == 1:
            j += 1
        end = j - 1
        L = end - start + 1

        for k in range(max(0, start - 3), start):
            target[k] = 1

        if L < split:
            for k in range(start, min(n, start + 2)):
                target[k] = 1
        else:
            last_ok = end - 3
            if last_ok >= start:
                for k in range(start, last_ok + 1):
                    target[k] = 1
        i = j

    return rally, target


def rebuild_target_for_train(df, lead_days, entry_days,
                             return_window=None, rally_threshold=None,
                             min_rally_tail_days=None):
    """
    Re-compute 'target' and 'rally' columns for every ticker in df.
    Wenn cfg.OPT_OPTIMIZE_Y_TARGETS False: feste Band-Regel (_create_target_one_ticker_fixed_bands),
    lead_days/entry_days/return_window-Argumente werden ignoriert.
    Sonst: parametrisierte _create_target_one_ticker.
    """
    df = df.copy()
    use_opt_y = cfg.opt_optimize_y_targets()
    for ticker, sub in df.groupby('ticker'):
        sub_r = sub.reset_index(drop=True)
        if not use_opt_y:
            r, t = _create_target_one_ticker_fixed_bands(sub_r)
        else:
            r, t = _create_target_one_ticker(
                sub_r,
                lead_days=lead_days, entry_days=entry_days,
                return_window=return_window, rally_threshold=rally_threshold,
                min_rally_tail_days=min_rally_tail_days,
            )
        df.loc[sub.index, 'rally']  = r
        df.loc[sub.index, 'target'] = t
    return df


def create_target(df):
    """Add 'rally' and 'target' columns to full DataFrame, parallelised per ticker."""
    df = df.sort_values(['ticker', 'Date']).copy()
    rally_col  = np.zeros(len(df), dtype=np.int8)
    target_col = np.zeros(len(df), dtype=np.int8)

    def process(args):
        idx, sub = args
        use_opt_y = cfg.opt_optimize_y_targets()
        if use_opt_y:
            r, t = _create_target_one_ticker(sub.reset_index(drop=True))
        else:
            r, t = _create_target_one_ticker_fixed_bands(sub.reset_index(drop=True))
        return idx, r, t

    groups = list(df.groupby('ticker'))
    with ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
        results = list(ex.map(process, groups))

    for ticker, sub in groups:
        pass  # just to reuse the loop below

    # Re-assign in correct row order
    for ticker_name, sub in df.groupby('ticker'):
        use_opt_y = cfg.opt_optimize_y_targets()
        if use_opt_y:
            r, t = _create_target_one_ticker(sub.reset_index(drop=True))
        else:
            r, t = _create_target_one_ticker_fixed_bands(sub.reset_index(drop=True))
        df.loc[sub.index, 'rally']  = r
        df.loc[sub.index, 'target'] = t

    pos_rate = df['target'].mean()
    print(f'Target created. Positive rate: {pos_rate:.1%}')
    return df