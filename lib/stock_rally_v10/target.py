"""stock_rally_v10 — Target-Aufbau (Pipeline-Modul)."""
from __future__ import annotations

import contextlib
from typing import Any, Iterator
import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg


def _segment_meets_constraints(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    start: int,
    end: int,
    *,
    min_return: float,
    min_len: int = 1,
    require_strict_daily_up: bool = False,
) -> bool:
    """Finales Rally-Segment muss auch nach dem Mergen die Mindestkriterien erfüllen."""
    if start < 0 or end < start or end >= len(close_arr):
        return False
    seg_len = end - start + 1
    if seg_len < int(min_len):
        return False
    o = float(open_arr[start])
    c = float(close_arr[end])
    if not np.isfinite(o) or not np.isfinite(c) or o == 0.0:
        return False
    if (c / o - 1.0) < float(min_return):
        return False
    if require_strict_daily_up:
        for j in range(start + 1, end + 1):
            prev_c = float(close_arr[j - 1])
            cur_c = float(close_arr[j])
            if (
                not np.isfinite(prev_c)
                or not np.isfinite(cur_c)
                or prev_c == 0.0
                or (cur_c / prev_c - 1.0) <= 0.0
            ):
                return False
    return True


def _create_target_one_ticker(df_ticker, lead_days=None, entry_days=None,
                              return_window=None, rally_threshold=None,
                              min_rally_tail_days=None):
    """
    For a single ticker DataFrame (sorted by Date), compute:
      - rally[]: 1 auf den Handelstagen der Rally-Phase, wenn die Trade-Return
        (Entry am Open t_1 nach Signal-Tag t_0, Exit am Close t_(1+rw))
        >= rally_threshold ist:
            ret_trade = close[t_1+rw] / open[t_1] - 1
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
    if "open" in df_ticker.columns:
        open_ = pd.to_numeric(df_ticker["open"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    else:
        open_ = close.copy()
    n = len(close)
    window = rw

    # Trade-basierte Rally:
    # Signal auf t_0 -> Entry am Open t_1 -> Exit am Close t_(1+window).
    # Bei erfolgreicher Rally werden die aktiven Haltetage [t_1, t_(1+window)] als rally=1 markiert.
    rally = np.zeros(n, dtype=np.int8)
    for signal_idx in range(n):
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + window
        if entry_idx >= n or exit_idx >= n:
            continue
        o = open_[entry_idx]
        c = close[exit_idx]
        if not np.isfinite(o) or not np.isfinite(c) or o == 0.0:
            continue
        ret_trade = c / o - 1.0
        if ret_trade >= rt:
            rally[entry_idx : exit_idx + 1] = 1

    # Vorlauf + Einstiegszone nur, wenn genug Rally „Restlauf“ (kein positives Label am Rally-Ende)
    # und das zusammenhängende Segment als Ganzes weiterhin die Zielrendite erreicht.
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
        if not _segment_meets_constraints(
            open_,
            close,
            start,
            end,
            min_return=rt,
            min_len=max(1, mt),
            require_strict_daily_up=False,
        ):
            rally[start : end + 1] = 0
            i = j
            continue
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
      so dass für Signal-Tag t_0 gilt:
        Entry am Open t_1, Exit am Close t_(1+w), Trade-Return >= FIXED_Y_RALLY_THRESHOLD.
      Bei Treffer gehören die Haltetage [t_1, t_(1+w)] zur grünen Phase (Vereinigung wie bisher).

      Pro zusammenhängendem grünen Segment [start, end] mit Länge L (Parameter aus cfg.FIXED_Y_*):
        L < split: Target = FIXED_Y_LEAD_DAYS vor start + erste FIXED_Y_ENTRY_DAYS grüne Tage.
        L >= split:
          - mode "tail_exclude": alle grünen außer den letzten FIXED_Y_TAIL_EXCLUDE_DAYS.
          - mode "early_only": nur die ersten FIXED_Y_LONG_ENTRY_DAYS grünen Tage.
    """
    w_lo, w_hi, rt, split, ld, ed, tail_ex = cfg.fixed_y_rule_params()
    label_mode = cfg.fixed_y_label_mode()
    strict_daily_up = bool(cfg.fixed_y_require_strict_daily_up_in_rally())
    max_dip = float(cfg.fixed_y_max_dip_below_entry_fraction())
    max_dip = max(0.0, min(float(max_dip), 1.0))
    long_mode, long_entry_days = cfg.fixed_y_long_segment_label_params()

    close = df_ticker["close"].values.astype(np.float64)
    if "open" in df_ticker.columns:
        open_ = pd.to_numeric(df_ticker["open"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    else:
        open_ = close.copy()
    n = len(close)

    rally = np.zeros(n, dtype=np.int8)
    entry_valid = np.zeros(n, dtype=np.int8)  # target-Signal auf t0: Einstieg morgen (t1) wäre valide.
    for signal_idx in range(n):
        entry_idx = signal_idx + 1
        if entry_idx >= n:
            continue
        has_valid_window = False
        for w in range(w_lo, w_hi + 1):
            exit_idx = entry_idx + int(w)
            if exit_idx >= n:
                continue
            o = open_[entry_idx]
            c = close[exit_idx]
            if not np.isfinite(o) or not np.isfinite(c) or o == 0.0:
                continue
            ret_trade = c / o - 1.0
            if ret_trade < rt:
                continue
            # Zulässiger Mindest-Schlusskurs: Entry-Open * (1 - max_dip). max_dip=0: streng wie früher
            # (kein Close unter o); max_dip=0.01: bis 1% unter o erlaubt; max_dip=1.0: keine untere Dipschranke.
            floor = float(o) * (1.0 - max_dip)
            c_path = close[entry_idx : exit_idx + 1]
            if (len(c_path) == 0) or bool(np.any(c_path < floor - 1e-9)):
                continue
            if strict_daily_up:
                bad = False
                for j in range(entry_idx + 1, exit_idx + 1):
                    prev_c = close[j - 1]
                    cur_c = close[j]
                    if (
                        not np.isfinite(prev_c)
                        or not np.isfinite(cur_c)
                        or prev_c == 0.0
                        or (cur_c / prev_c - 1.0) <= 0.0
                    ):
                        bad = True
                        break
                if bad:
                    continue
            has_valid_window = True
            rally[entry_idx : exit_idx + 1] = 1
        if has_valid_window:
            entry_valid[signal_idx] = 1

    target = np.zeros(n, dtype=np.int8)
    if label_mode == "entry_direct":
        target[:] = entry_valid
        return rally, target

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
        if not _segment_meets_constraints(
            open_,
            close,
            start,
            end,
            min_return=rt,
            min_len=max(1, int(w_lo) + 1),
            require_strict_daily_up=strict_daily_up,
        ):
            rally[start : end + 1] = 0
            i = j
            continue
        L = end - start + 1

        for k in range(max(0, start - ld), start):
            target[k] = 1

        if L < split:
            for k in range(start, min(n, start + ed)):
                target[k] = 1
        else:
            if long_mode == "early_only":
                for k in range(start, min(n, start + int(long_entry_days))):
                    target[k] = 1
            else:
                last_ok = end - tail_ex
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


def create_target(df, *, quiet: bool = False):
    """Add 'rally' and 'target' columns to full DataFrame, parallelised per ticker."""
    df = df.sort_values(['ticker', 'Date']).copy()

    use_opt_y = cfg.opt_optimize_y_targets()

    def _process(item: tuple[Any, pd.DataFrame]) -> tuple[Any, pd.Index, np.ndarray, np.ndarray]:
        ticker_name, sub = item
        sub_r = sub.reset_index(drop=False)
        if use_opt_y:
            r, t = _create_target_one_ticker(sub_r)
        else:
            r, t = _create_target_one_ticker_fixed_bands(sub_r)
        return ticker_name, sub.index, r, t

    groups = list(df.groupby('ticker'))
    # Threading hilft hier kaum (GIL + reine Python/NumPy-Schleifen). Prozessparallelität wäre
    # teuer wegen Pickle-Kosten der Sub-DataFrames; daher bewusst seriell und einmalig.
    for _ticker_name, idx, r, t in map(_process, groups):
        df.loc[idx, 'rally'] = r
        df.loc[idx, 'target'] = t

    pos_rate = df["target"].mean()
    if not quiet:
        if cfg.opt_optimize_y_targets():
            print(
                f"Target created. Positive rate: {pos_rate:.1%} "
                "(Baseline vor Optuna: cfg RETURN_WINDOW, RALLY_THRESHOLD, LEAD_DAYS, "
                "ENTRY_DAYS, MIN_RALLY_TAIL_DAYS — in Phase 1 setzt jedes Trial "
                "rebuild_target_for_train mit gesampelten Label-Parametern neu).",
                flush=True,
            )
        else:
            print(f"Target created. Positive rate: {pos_rate:.1%}", flush=True)
        if not cfg.opt_optimize_y_targets():
            _w_lo, _w_hi, _rt, _sp, _ld, _ed, _tex = cfg.fixed_y_rule_params()
            _strict = bool(cfg.fixed_y_require_strict_daily_up_in_rally())
            _md = float(cfg.fixed_y_max_dip_below_entry_fraction())
            _label_mode = cfg.fixed_y_label_mode()
            _long_mode, _long_ed = cfg.fixed_y_long_segment_label_params()
            print(
                f"  Feste Band-Regel: Fenster w in [{_w_lo}, {_w_hi}] Handelstage, "
                f"kum. Rendite >= {_rt:.2%}, Segment-Split {_sp}d, "
                f"label_mode={_label_mode} lead={_ld} entry={_ed} tail_excl={_tex} strict_up={_strict} "
                f"max_dip_below_entry={_md:.4f} long_mode={_long_mode} long_entry={_long_ed} (cfg.FIXED_Y_*)",
                flush=True,
            )
    return df


@contextlib.contextmanager
def override_fixed_y_config_for_grid(**overrides: Any) -> Iterator[None]:
    """
    Temporär ``config``-Attribute setzen (typisch: ``FIXED_Y_*``), danach wiederherstellen.
    Für Gittersuchen/Reports ohne dauerhafte cfg-Änderung.
    """
    from lib.stock_rally_v10 import config as _m

    if not overrides:
        yield
        return
    saved: dict[str, Any] = {k: getattr(_m, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(_m, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(_m, k, v)