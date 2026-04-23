"""
Datenpipeline bis zur zeitlichen bzw. Ticker-basierten Aufteilung.

Lädt Kurse, baut Targets, Indikatoren, News-Features, assembliert die Matrix und erzeugt
disjunkte Zeitfenster (``SPLIT_MODE=time``):

  **BASE** → Purge → **META** → Purge → **THRESHOLD** → Purge → **FINAL**

Base-Optuna und Base-Modelle nutzen nur ``df_train`` (= ``df_train_base``).
Meta-Learner nutzt ``df_test`` (= ``df_train_meta``, strikt nach BASE+Purge).
Kalibrierung der Schwelle nur auf ``df_threshold``; echtes OOS bleibt ``df_final``.

``SPLIT_MODE=ticker`` (Legacy): THRESHOLD/FINAL weiter nach Ticker getrennt; BASE und META
werden innerhalb des TRAIN-Kalenders zeitlich getrennt (gleiche TRAIN-Ticker).
Alle Ergebnisse werden als Attribute auf ``config`` geschrieben.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.config import load_scoring_artifacts
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.features import assemble_features
from lib.stock_rally_v10.indicators import add_technical_indicators
from lib.stock_rally_v10.news import fetch_news_sentiment
from lib.stock_rally_v10.target import create_target


def _df_on_trading_days(df, tickers, date_array):
    """Zeilen zu exakt diesen Kalendertagen (normalisiert), gleiche Ticker."""
    ds = {pd.Timestamp(d).normalize() for d in date_array}
    dn = pd.to_datetime(df["Date"]).dt.normalize()
    return df.loc[df["ticker"].isin(tickers) & dn.isin(ds)].copy()


def _split_calendar_four_way(
    uniq: np.ndarray,
    fb: float,
    fm: float,
    ft: float,
    purge: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Teilt sortierte Handelstage in BASE, META, THRESHOLD, FINAL mit ``purge`` Lücken dazwischen.
    Anteile beziehen sich auf (n − 3·purge) „Inhalts“-Tage; der FINAL-Block erhält den Rest.
    """
    n = int(len(uniq))
    p = int(purge)
    if min(fb, fm, ft) <= 0.0:
        raise ValueError(
            "Zeit-Split: TIME_SPLIT_FRAC_BASE, TIME_SPLIT_FRAC_META und "
            "TIME_SPLIT_FRAC_THRESHOLD müssen jeweils > 0 sein."
        )
    if fb + fm + ft >= 1.0:
        raise ValueError(
            "Zeit-Split: TIME_SPLIT_FRAC_BASE + META + THRESHOLD muss < 1 sein "
            "(verbleibender Anteil = FINAL)."
        )
    if p < 0:
        raise ValueError("TIME_PURGE_TRADING_DAYS darf nicht negativ sein.")
    available = n - 3 * p
    if available < 4:
        raise ValueError(
            f"Zeit-Split: zu wenig Handelstage (n={n}) für 4 Blöcke und {p} Purge-Tage "
            "zwischen BASE–META–THRESHOLD–FINAL — Anteile oder Purge verkleinern."
        )
    c_base = max(1, int(round(available * fb)))
    c_meta = max(1, int(round(available * fm)))
    c_thr = max(1, int(round(available * ft)))
    while c_base + c_meta + c_thr > available - 1:
        if c_base >= c_meta and c_base >= c_thr and c_base > 1:
            c_base -= 1
        elif c_meta >= c_thr and c_meta > 1:
            c_meta -= 1
        elif c_thr > 1:
            c_thr -= 1
        else:
            raise ValueError(
                "Zeit-Split: BASE/META/THRESHOLD lassen sich nicht so legen, dass FINAL ≥ 1 Tag bleibt."
            )
    i0 = 0
    i1 = i0 + c_base
    i2 = i1 + p
    i3 = i2 + c_meta
    i4 = i3 + p
    i5 = i4 + c_thr
    i6 = i5 + p
    if i6 >= n or not len(uniq[i6:]):
        raise ValueError(
            "Zeit-Split: FINAL-Fenster leer — TIME_SPLIT_FRAC_* oder Purge anpassen."
        )
    base_dates = uniq[i0:i1]
    meta_dates = uniq[i2:i3]
    thr_dates = uniq[i4:i5]
    final_dates = uniq[i6:]
    return base_dates, meta_dates, thr_dates, final_dates


def _split_train_calendar_base_meta(
    uniq_tr: np.ndarray,
    fb: float,
    fm: float,
    purge: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Teilt den TRAIN-Kalender (Legacy-Ticker-Modus) in BASE- und META-Zeilenblöcke."""
    n_tr = int(len(uniq_tr))
    p = int(purge)
    if min(fb, fm) <= 0.0:
        raise ValueError(
            "Ticker-Split: TIME_SPLIT_FRAC_BASE und TIME_SPLIT_FRAC_META müssen jeweils > 0 sein."
        )
    ratio = fb + fm
    if n_tr < 2:
        raise ValueError("Ticker-Split: zu wenig Handelstage im TRAIN-Fenster.")
    inner = n_tr - p
    if inner < 2:
        raise ValueError(
            "Ticker-Split: TRAIN-Kalender zu kurz für Purge zwischen BASE und META — "
            "TIME_PURGE_TRADING_DAYS senken."
        )
    w_base = fb / ratio
    c_base = max(1, int(round(inner * w_base)))
    c_meta = inner - c_base
    if c_meta < 1:
        c_meta = 1
        c_base = inner - 1
    if c_base < 1:
        raise ValueError("Ticker-Split: kann BASE/META nicht aufteilen.")
    base_dates = uniq_tr[:c_base]
    meta_dates = uniq_tr[c_base + p : c_base + p + c_meta]
    if len(meta_dates) < 1:
        raise ValueError("Ticker-Split: META-Fenster leer nach Aufteilung.")
    return base_dates, meta_dates


def _print_training_windows_summary(
    *,
    split_mode: str,
    df_train_base: pd.DataFrame,
    df_train_meta: pd.DataFrame,
    df_threshold: pd.DataFrame,
    df_final: pd.DataFrame,
) -> None:
    """Gut sichtbare Übersicht direkt nach dem Split (nicht beim Yahoo-Download)."""
    bar = "=" * 72
    print(f"\n{bar}", flush=True)
    print(
        "ZEITSPANNEN: Base-Train | Meta-Train | Threshold-Kalibrierung | Final (OOS)",
        flush=True,
    )
    print(bar, flush=True)
    print(f"  SPLIT_MODE={split_mode}", flush=True)
    rows = (
        ("BASE (Base-Optuna & Base-Modelle)", df_train_base),
        ("META (Meta-Learner)", df_train_meta),
        ("THRESHOLD (Schwellen-Kalibrierung)", df_threshold),
        ("FINAL (OOS-Eval)", df_final),
    )
    for label, df in rows:
        if df is None or len(df) == 0 or "Date" not in df.columns:
            print(f"  {label}: — (leer)", flush=True)
            continue
        d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        print(
            f"  {label}: {d.min().date()} … {d.max().date()}  |  "
            f"{int(d.nunique())} Handelstage  |  {len(df):,} Zeilen",
            flush=True,
        )
    print(
        f"{bar}\n"
        "(Diese Zeilen stehen hier, weil der Kalender erst nach Target/Indikatoren/News/Features "
        "festliegt — nicht direkt nach dem Kurs-Download.)\n",
        flush=True,
    )


def run_data_download_and_split() -> None:
    """Download → Target → Indikatoren → Features → Split (Zeit oder Legacy-Ticker)."""
    cfg.log_pipeline_mode_banner()
    _retrain_meta_only = bool(getattr(cfg, "RETRAIN_META_ONLY", False))
    # ── 1. Load data ────────────────────────────────────────────────────────
    if getattr(cfg, "SCORING_ONLY", False) or _retrain_meta_only:
        load_scoring_artifacts()
        if _retrain_meta_only and not getattr(cfg, "SCORING_ONLY", False):
            print(
                "RETRAIN_META_ONLY: Artefakt für Base-Modelle/Best-Params vor Feature-Build geladen.",
                flush=True,
            )

    if getattr(cfg, "SCORING_ONLY", False) and getattr(cfg, "_tickers_for_run", None):
        _tickers_for_run = list(cfg._tickers_for_run)
        print(
            f"SCORING_ONLY: {len(_tickers_for_run)} Ticker aus Artefakt "
            "(UNIVERSE_FRACTION nicht erneut angewandt)."
        )
    else:
        _tickers_for_run = list(cfg.ALL_TICKERS)
        _uf = float(getattr(cfg, "UNIVERSE_FRACTION", 1.0))
        if _uf > 1.0:
            raise ValueError(
                "UNIVERSE_FRACTION muss ≤ 1.0 sein (Anteil des Universums). "
                "Für 2.5 % bitte 0.025 setzen — nicht 2.5 (sonst wird nicht untergeteilt)."
            )
        if _uf < 1.0:
            _rng_u = np.random.RandomState(
                int(getattr(cfg, "UNIVERSE_SAMPLE_SEED", cfg.RANDOM_STATE))
            )
            _rng_u.shuffle(_tickers_for_run)
            _n_keep = max(1, int(round(len(_tickers_for_run) * _uf)))
            _tickers_for_run = _tickers_for_run[:_n_keep]
            print(
                f"UNIVERSE_FRACTION={_uf} (Ticker-Anteil): "
                f"{len(_tickers_for_run)}/{len(cfg.ALL_TICKERS)} Symbole "
                "(vor Download, Target, Indikatoren, News, Features). "
                "Zeit-Split-Anteile sind TIME_SPLIT_FRAC_* — nicht dasselbe."
            )
    cfg._tickers_for_run = _tickers_for_run

    df_raw = load_stock_data(tickers=_tickers_for_run)
    df_with_target = create_target(df_raw)
    df_with_indicators = add_technical_indicators(df_with_target)
    sentiment_df = fetch_news_sentiment(df_with_indicators)
    df_features = assemble_features(
        df_with_indicators,
        sentiment_df,
        meta_only=_retrain_meta_only,
    )

    cfg.df_raw = df_raw
    cfg.df_with_target = df_with_target
    cfg.df_with_indicators = df_with_indicators
    cfg.sentiment_df = sentiment_df
    cfg.df_features = df_features
    print(
        "\n[Kalender-Split] Target, Indikatoren, News und Features fertig — "
        "berechne jetzt BASE / META / THRESHOLD / FINAL (erscheint unten im Kasten).\n",
        flush=True,
    )

    # ── 6. Split ─────────────────────────────────────────────────────────────
    _train_cutoff = pd.Timestamp(cfg.TRAIN_END_DATE)
    available_tickers = set(df_features["ticker"].unique())

    _split_mode = getattr(cfg, "SPLIT_MODE", "time")

    if _split_mode == "time":
        fb = float(cfg.TIME_SPLIT_FRAC_BASE)
        fm = float(getattr(cfg, "TIME_SPLIT_FRAC_META", 0.0))
        ft = float(cfg.TIME_SPLIT_FRAC_THRESHOLD)
        p = int(getattr(cfg, "TIME_PURGE_TRADING_DAYS", 0))
        dc = pd.to_datetime(df_features["Date"])
        uniq = np.sort(dc[dc <= _train_cutoff].unique())
        n = len(uniq)
        if n < 30:
            raise ValueError(f"Zu wenig Handelstage für Zeit-Split (n={n}).")
        base_dates, meta_dates, thr_dates, final_dates = _split_calendar_four_way(
            uniq, fb, fm, ft, p
        )
        universe_tickers = sorted(available_tickers)
        train_base_tickers = universe_tickers
        train_meta_tickers = universe_tickers
        threshold_tickers = universe_tickers
        final_tickers = universe_tickers
        df_train_base = _df_on_trading_days(df_features, universe_tickers, base_dates)
        df_train_meta = _df_on_trading_days(df_features, universe_tickers, meta_dates)
        df_threshold = _df_on_trading_days(df_features, universe_tickers, thr_dates)
        df_final = _df_on_trading_days(df_features, universe_tickers, final_dates)
        _ff = 1.0 - fb - fm - ft
        print(
            f"SPLIT_MODE=time — {len(universe_tickers)} Ticker; "
            f"BASE / META / THRESHOLD / FINAL + je {p} Handelstage Purge dazwischen "
            f"(Anteile der Inhalts-Tage ≈ {fb:.0%} / {fm:.0%} / {ft:.0%} / {_ff:.0%})"
        )
        print(
            f"  BASE:      {pd.Timestamp(base_dates[0]).date()} … "
            f"{pd.Timestamp(base_dates[-1]).date()}  ({len(base_dates)} Tage)"
        )
        print(
            f"  META:      {pd.Timestamp(meta_dates[0]).date()} … "
            f"{pd.Timestamp(meta_dates[-1]).date()}  ({len(meta_dates)} Tage)"
        )
        print(
            f"  THRESHOLD: {pd.Timestamp(thr_dates[0]).date()} … "
            f"{pd.Timestamp(thr_dates[-1]).date()}  ({len(thr_dates)} Tage)"
        )
        print(
            f"  FINAL:     {pd.Timestamp(final_dates[0]).date()} … "
            f"{pd.Timestamp(final_dates[-1]).date()}  ({len(final_dates)} Tage)"
        )
    else:
        TARGET_META = max(15, round(cfg.MAX_TRAIN_TICKERS * 0.6))
        TARGET_THRESHOLD = max(10, round(cfg.MAX_TRAIN_TICKERS * 0.4))
        TARGET_FINAL = max(10, round(cfg.MAX_TRAIN_TICKERS * 0.4))
        print(
            f"SPLIT_MODE=ticker (Legacy) — Ziel: TRAIN_BASE≤{cfg.MAX_TRAIN_TICKERS}  "
            f"META≥{TARGET_META}  THRESHOLD≥{TARGET_THRESHOLD}  FINAL≥{TARGET_FINAL}"
        )
        rng_split = np.random.RandomState(42)
        train_base_tickers = []
        train_meta_tickers = []
        threshold_tickers = []
        final_tickers = []
        assigned_so_far = set()
        for sector, tickers in cfg.TICKERS_BY_SECTOR.items():
            avail = [t for t in tickers if t in available_tickers and t not in assigned_so_far]
            rng_split.shuffle(avail)
            n = len(avail)
            if n >= 4:
                train_meta_tickers.append(avail[0])
                threshold_tickers.append(avail[1])
                final_tickers.append(avail[2])
                train_base_tickers.extend(avail[3:])
            elif n == 3:
                train_meta_tickers.append(avail[0])
                threshold_tickers.append(avail[1])
                train_base_tickers.append(avail[2])
            elif n == 2:
                train_meta_tickers.append(avail[0])
                train_base_tickers.append(avail[1])
            elif n == 1:
                train_base_tickers.extend(avail)
            assigned_so_far.update(avail)

        def _fill_partition(target_list, source_list, target_size, exclude_sets):
            pool = [t for t in source_list if not any(t in s for s in exclude_sets)]
            rng_split.shuffle(pool)
            for t in pool:
                if len(target_list) >= target_size:
                    break
                if t in source_list:
                    target_list.append(t)
                    source_list.remove(t)

        _fill_partition(
            train_meta_tickers,
            train_base_tickers,
            TARGET_META,
            [set(threshold_tickers), set(final_tickers)],
        )
        _fill_partition(
            threshold_tickers,
            train_base_tickers,
            TARGET_THRESHOLD,
            [set(train_meta_tickers), set(final_tickers)],
        )
        _fill_partition(
            final_tickers,
            train_base_tickers,
            TARGET_FINAL,
            [set(train_meta_tickers), set(threshold_tickers)],
        )
        if len(train_base_tickers) > cfg.MAX_TRAIN_TICKERS:
            rng_split.shuffle(train_base_tickers)
            overflow = train_base_tickers[cfg.MAX_TRAIN_TICKERS :]
            train_base_tickers = train_base_tickers[: cfg.MAX_TRAIN_TICKERS]
            already_assigned = set(train_meta_tickers) | set(threshold_tickers) | set(final_tickers)
            overflow_clean = [t for t in overflow if t not in already_assigned]
            train_meta_tickers.extend(overflow_clean[0::3])
            threshold_tickers.extend(overflow_clean[1::3])
            final_tickers.extend(overflow_clean[2::3])
        all_assigned = train_base_tickers + train_meta_tickers + threshold_tickers + final_tickers
        assert len(all_assigned) == len(set(all_assigned)), "Ticker-Überlappung zwischen Partitionen!"
        _seen_m = set()
        _merged_tb: list[str] = []
        for t in train_base_tickers + train_meta_tickers:
            if t not in _seen_m:
                _merged_tb.append(t)
                _seen_m.add(t)
        train_base_tickers = _merged_tb
        train_meta_tickers = train_base_tickers
        print(f"TRAIN:       {len(train_base_tickers):3d} tickers (BASE+META — gleiche Ticker, zeitlich getrennt)")
        print(f"THRESHOLD:   {len(threshold_tickers):3d} tickers — {threshold_tickers}")
        print(f"FINAL:       {len(final_tickers):3d} tickers — {final_tickers}")
        _p_tm = int(getattr(cfg, "TIME_PURGE_TRADING_DAYS", 0))
        _fb_tm = float(cfg.TIME_SPLIT_FRAC_BASE)
        _fm_tm = float(getattr(cfg, "TIME_SPLIT_FRAC_META", 0.0))
        df_train_pool = df_features[
            (df_features["ticker"].isin(train_base_tickers)) & (df_features["Date"] <= _train_cutoff)
        ].copy()
        uniq_tr = np.sort(pd.to_datetime(df_train_pool["Date"]).dt.normalize().unique())
        base_dates_tm, meta_dates_tm = _split_train_calendar_base_meta(
            uniq_tr, _fb_tm, _fm_tm, _p_tm
        )
        df_train_base = _df_on_trading_days(df_train_pool, train_base_tickers, base_dates_tm)
        df_train_meta = _df_on_trading_days(df_train_pool, train_base_tickers, meta_dates_tm)
        print(
            f"  TRAIN zeitlich: BASE {pd.Timestamp(base_dates_tm[0]).date()} … "
            f"{pd.Timestamp(base_dates_tm[-1]).date()} ({len(base_dates_tm)} Tage) | "
            f"Purge {_p_tm} | META {pd.Timestamp(meta_dates_tm[0]).date()} … "
            f"{pd.Timestamp(meta_dates_tm[-1]).date()} ({len(meta_dates_tm)} Tage)"
        )
        df_threshold = df_features[
            (df_features["ticker"].isin(threshold_tickers)) & (df_features["Date"] <= _train_cutoff)
        ].copy()
        df_final = df_features[
            (df_features["ticker"].isin(final_tickers)) & (df_features["Date"] <= _train_cutoff)
        ].copy()

    cfg.train_base_tickers = train_base_tickers
    cfg.train_meta_tickers = train_meta_tickers
    cfg.threshold_tickers = threshold_tickers
    cfg.final_tickers = final_tickers
    cfg.df_train_base = df_train_base
    cfg.df_train_meta = df_train_meta
    cfg.df_threshold = df_threshold
    cfg.df_final = df_final
    cfg.df_train = df_train_base
    cfg.df_test = df_train_meta

    if len(df_final) > 0 and len(final_tickers) == 0:
        raise ValueError(
            "Split: df_final ist nicht leer, aber final_tickers leer — Ticker-Zuweisung prüfen (SPLIT_MODE)."
        )

    print(
        f"\nZeilenanzahl — BASE: {len(df_train_base):,}  META: {len(df_train_meta):,}  "
        f"THRESHOLD: {len(df_threshold):,}  FINAL: {len(df_final):,}"
    )
    _print_training_windows_summary(
        split_mode=str(_split_mode),
        df_train_base=df_train_base,
        df_train_meta=df_train_meta,
        df_threshold=df_threshold,
        df_final=df_final,
    )
