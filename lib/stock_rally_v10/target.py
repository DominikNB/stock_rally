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

      Pro zusammenhängendem grünen Segment [start, end] (nur für rally=1 / Segment-Prüfung):
        rally_plus_entry: rally wie bisher als Vereinigung aller Fenster; target=1 je
          **qualifizierendem Einzelfenster** [entry, exit] (vor dem Merge) auf den ersten
          ceil(Lw·f) Tagen dieses Fensters (Lw = Fensterlänge) plus FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS
          vor entry — nur wenn nach Segment-Validierung alle Tage des Fensters noch rally=1 sind.
        segment_based (Legacy): FIXED_Y_LEAD_DAYS vor start + …
        entry_direct: target = entry_valid (nur Signal-Tag t0).
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
    # Jedes qualifizierende Haltefenster [entry, exit] (ein w aus w_lo..w_hi) — für rally_plus_entry
    # werden Targets pro Fenster gesetzt, nicht auf dem merged Segment.
    raw_rally_windows: list[tuple[int, int]] = []
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
            raw_rally_windows.append((int(entry_idx), int(exit_idx)))
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
        if label_mode == "rally_plus_entry":
            # Nur Segment-Validierung / rally-Bereinigung; target folgt aus raw_rally_windows.
            pass
        else:
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

    if label_mode == "rally_plus_entry":
        n_pre = max(0, int(getattr(cfg, "FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS", 2)))
        head_frac = float(getattr(cfg, "FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION", 0.1))
        head_frac = max(0.0, min(1.0, head_frac))
        _ov = str(getattr(cfg, "FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE", "greedy_first")).strip().lower()
        if _ov not in {"greedy_first", "union"}:
            _ov = "greedy_first"

        wins: list[tuple[int, int]] = []
        for e, ex in raw_rally_windows:
            if e < 0 or ex >= n or e > ex:
                continue
            if not bool(np.all(rally[e : ex + 1] == 1)):
                continue
            wins.append((int(e), int(ex)))

        if _ov == "union":
            for e, ex in wins:
                lw = int(ex - e + 1)
                n_head = int(np.ceil(lw * head_frac)) if head_frac > 0.0 else 0
                n_head = min(max(n_head, 0), lw)
                for k in range(max(0, e - n_pre), e):
                    if 0 <= k < n:
                        target[k] = 1
                for k in range(e, min(n, e + n_head)):
                    target[k] = 1
        else:
            wins.sort(key=lambda t: (t[0], t[1] - t[0]))
            claimed = np.zeros(n, dtype=np.bool_)
            for e, ex in wins:
                lw = int(ex - e + 1)
                n_head = int(np.ceil(lw * head_frac)) if head_frac > 0.0 else 0
                n_head = min(max(n_head, 0), lw)
                for k in range(max(0, e - n_pre), e):
                    if 0 <= k < n and not bool(claimed[k]):
                        target[k] = 1
                        claimed[k] = True
                for k in range(e, min(n, e + n_head)):
                    if k <= ex and 0 <= k < n and int(rally[k]) == 1 and not bool(claimed[k]):
                        target[k] = 1
                        claimed[k] = True

    return rally, target


def apply_cross_sectional_top_q_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setzt ``target`` und ``rally`` nach Querschnitts-Top-q-Regel (nur bei cfg.Y_LABEL_RULE).

    Pro Gruppe (Kalendertag oder Sektor×Tag): unter allen Zeilen mit gültigem Forward-Return
    ``close[t+H]/close[t]-1`` die obersten ``CS_TARGET_TOP_Q`` nach r erhalten ``target=1``.
    ``rally`` wird gleich ``target`` gesetzt.
    """
    df = df.sort_values(["ticker", "Date"]).copy()
    H = int(getattr(cfg, "CS_TARGET_FORWARD_HORIZON", 20))
    q = float(getattr(cfg, "CS_TARGET_TOP_Q", 0.1))
    q = float(min(max(q, 1e-9), 0.499999999))
    min_n = max(1, int(getattr(cfg, "CS_TARGET_MIN_TICKERS_PER_GROUP", 5)))
    gb = str(getattr(cfg, "CS_TARGET_GROUPBY", "calendar_day")).strip().lower()
    if gb not in {"calendar_day", "sector_day"}:
        gb = "calendar_day"

    tick_to_sec: dict[str, str] = dict(getattr(cfg, "TICKER_TO_SECTOR", {}) or {})
    d_norm = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["__d"] = d_norm
    close_num = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
    df["__fwd"] = df.groupby("ticker", sort=False)["close"].transform(
        lambda s: s.shift(-H).astype(np.float64) / pd.to_numeric(s, errors="coerce").astype(np.float64) - 1.0
    )
    valid = df["__fwd"].replace([np.inf, -np.inf], np.nan).notna() & np.isfinite(close_num.to_numpy()) & (
        close_num.to_numpy() > 0.0
    )

    if gb == "sector_day":
        sec = df["ticker"].map(lambda t: str(tick_to_sec.get(t, "unknown")))
        df["__grp"] = df["__d"].dt.strftime("%Y-%m-%d").astype(str) + "\x00" + sec.astype(str)
    else:
        df["__grp"] = df["__d"].dt.strftime("%Y-%m-%d")

    df["target"] = np.zeros(len(df), dtype=np.int8)
    df["rally"] = np.zeros(len(df), dtype=np.int8)
    sub = df.loc[valid]
    for _, part in sub.groupby("__grp", sort=False):
        n = int(len(part))
        if n < min_n:
            continue
        k = max(1, min(n, int(np.ceil(n * q))))
        top_ix = part["__fwd"].nlargest(k).index
        df.loc[top_ix, "target"] = 1
    df["rally"] = df["target"].astype(np.int8, copy=False)
    df.drop(columns=["__d", "__fwd", "__grp"], inplace=True, errors="ignore")
    return df


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
    if not use_opt_y and cfg.y_label_rule() == "cross_sectional_top_q":
        return apply_cross_sectional_top_q_labels(df)
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
    if not use_opt_y and cfg.y_label_rule() == "cross_sectional_top_q":
        df = apply_cross_sectional_top_q_labels(df)
    else:

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
            if cfg.y_label_rule() == "cross_sectional_top_q":
                print(
                    f"  Cross-sectional Top-q: H={int(getattr(cfg, 'CS_TARGET_FORWARD_HORIZON', 20))} "
                    f"q={float(getattr(cfg, 'CS_TARGET_TOP_Q', 0.1)):.4f} "
                    f"groupby={getattr(cfg, 'CS_TARGET_GROUPBY', 'calendar_day')} "
                    f"min_per_group={int(getattr(cfg, 'CS_TARGET_MIN_TICKERS_PER_GROUP', 5))}",
                    flush=True,
                )
            else:
                _w_lo, _w_hi, _rt, _sp, _ld, _ed, _tex = cfg.fixed_y_rule_params()
                _strict = bool(cfg.fixed_y_require_strict_daily_up_in_rally())
                _md = float(cfg.fixed_y_max_dip_below_entry_fraction())
                _label_mode = cfg.fixed_y_label_mode()
                _long_mode, _long_ed = cfg.fixed_y_long_segment_label_params()
                _sig_pre = int(getattr(cfg, "FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS", 2))
                _head_f = float(getattr(cfg, "FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION", 0.1))
                _ovm = str(getattr(cfg, "FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE", "greedy_first")).strip().lower()
                _extra = (
                    f" rally_signal_pre_days={_sig_pre} rally_plus_head_frac={_head_f:.2f} overlap={_ovm}"
                    if _label_mode == "rally_plus_entry"
                    else ""
                )
                print(
                    f"  Feste Band-Regel: Fenster w in [{_w_lo}, {_w_hi}] Handelstage, "
                    f"kum. Rendite >= {_rt:.2%}, Segment-Split {_sp}d, "
                    f"label_mode={_label_mode} lead={_ld} entry={_ed} tail_excl={_tex} strict_up={_strict} "
                    f"max_dip_below_entry={_md:.4f} long_mode={_long_mode} long_entry={_long_ed}{_extra} (cfg.FIXED_Y_*)",
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