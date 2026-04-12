"""
Datenpipeline bis zur zeitlichen bzw. Ticker-basierten Aufteilung.

Lädt Kurse, baut Targets, Indikatoren, News-Features, assembliert die Matrix und erzeugt
``df_train_base`` / ``df_train_meta`` / ``df_threshold`` / ``df_final`` sowie Aliase ``df_train`` / ``df_test``.
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


def run_data_download_and_split() -> None:
    """Download → Target → Indikatoren → Features → Split (Zeit oder Legacy-Ticker)."""
    # ── 1. Load data ────────────────────────────────────────────────────────
    if getattr(cfg, "SCORING_ONLY", False):
        load_scoring_artifacts()

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
                f"UNIVERSE_FRACTION={_uf}: nutze {len(_tickers_for_run)}/{len(cfg.ALL_TICKERS)} Ticker "
                "(vor Download, Target, Indikatoren, News, Features)."
            )
    cfg._tickers_for_run = _tickers_for_run

    df_raw = load_stock_data(tickers=_tickers_for_run)
    df_with_target = create_target(df_raw)
    df_with_indicators = add_technical_indicators(df_with_target)
    sentiment_df = fetch_news_sentiment(df_with_indicators)
    df_features = assemble_features(df_with_indicators, sentiment_df)

    cfg.df_raw = df_raw
    cfg.df_with_target = df_with_target
    cfg.df_with_indicators = df_with_indicators
    cfg.sentiment_df = sentiment_df
    cfg.df_features = df_features

    # ── 6. Split ─────────────────────────────────────────────────────────────
    _train_cutoff = pd.Timestamp(cfg.TRAIN_END_DATE)
    available_tickers = set(df_features["ticker"].unique())

    _split_mode = getattr(cfg, "SPLIT_MODE", "time")

    if _split_mode == "time":
        fb = float(cfg.TIME_SPLIT_FRAC_BASE)
        fm = float(cfg.TIME_SPLIT_FRAC_META)
        ft = float(cfg.TIME_SPLIT_FRAC_THRESHOLD)
        ff = 1.0 - fb - fm - ft
        if ff < 0.0:
            raise ValueError(
                "TIME_SPLIT_FRAC_BASE+META+THRESHOLD darf 1.0 nicht überschreiten."
            )
        p = int(getattr(cfg, "TIME_PURGE_TRADING_DAYS", 0))
        dc = pd.to_datetime(df_features["Date"])
        uniq = np.sort(dc[dc <= _train_cutoff].unique())
        n = len(uniq)
        if n < 30:
            raise ValueError(f"Zu wenig Handelstage für Zeit-Split (n={n}).")
        c1 = max(1, int(round(n * fb)))
        meta_start = c1 + p
        if meta_start >= n - 2:
            raise ValueError(
                "Zeit-Split: Purge zu groß oder BASE-Anteil zu groß — TIME_PURGE_TRADING_DAYS "
                "oder TIME_SPLIT_FRAC_BASE verkleinern."
            )
        c2 = max(meta_start + 1, int(round(n * (fb + fm))))
        c3 = max(c2 + 1, int(round(n * (fb + fm + ft))))
        c2 = min(c2, n - 1)
        c3 = min(c3, n - 1)
        if c2 <= meta_start or c3 <= c2:
            raise ValueError("Zeit-Split: ungültige Grenzen — Anteile anpassen.")
        base_dates = uniq[:c1]
        meta_dates = uniq[meta_start:c2]
        thr_dates = uniq[c2:c3]
        final_dates = uniq[c3:]
        if not len(meta_dates) or not len(thr_dates) or not len(final_dates):
            raise ValueError("Zeit-Split: eine Partition ist leer — Fraktionen anpassen.")
        universe_tickers = sorted(available_tickers)
        train_base_tickers = universe_tickers
        train_meta_tickers = universe_tickers
        threshold_tickers = universe_tickers
        final_tickers = universe_tickers
        df_train_base = _df_on_trading_days(df_features, universe_tickers, base_dates)
        df_train_meta = _df_on_trading_days(df_features, universe_tickers, meta_dates)
        df_threshold = _df_on_trading_days(df_features, universe_tickers, thr_dates)
        df_final = _df_on_trading_days(df_features, universe_tickers, final_dates)
        print(
            f"SPLIT_MODE=time — {len(universe_tickers)} Ticker in allen Stufen; "
            f"Purge Base→Meta: {p} Handelstage"
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
        print(f"TRAIN_BASE:  {len(train_base_tickers):3d} tickers")
        print(f"TRAIN_META:  {len(train_meta_tickers):3d} tickers — {train_meta_tickers}")
        print(f"THRESHOLD:   {len(threshold_tickers):3d} tickers — {threshold_tickers}")
        print(f"FINAL:       {len(final_tickers):3d} tickers — {final_tickers}")
        df_train_base = df_features[
            (df_features["ticker"].isin(train_base_tickers)) & (df_features["Date"] <= _train_cutoff)
        ].copy()
        df_train_meta = df_features[
            (df_features["ticker"].isin(train_meta_tickers)) & (df_features["Date"] <= _train_cutoff)
        ].copy()
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

    print(
        f"\nZeilenanzahl — TRAIN_BASE: {len(df_train_base):,}  TRAIN_META: {len(df_train_meta):,}  "
        f"THRESHOLD: {len(df_threshold):,}  FINAL: {len(df_final):,}"
    )
