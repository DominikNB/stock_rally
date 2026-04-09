"""
Zusätzliche Filter-Spalten für Signale (über News/Makro/Chart hinaus):

  • Liquidität: 20-Tage-Durchschnitt Umsatz (Close * Volume), Perzentil pro Signaltag
  • Meta-Kalibrierung: prob − threshold_used
  • Cluster: Anzahl Signale am gleichen Tag, gleiche Branche, Sektor-Anteil
  • Korrelation: mittlere paarweise Korrelation der Tagesrenditen (Lookback) unter
    allen Titeln mit gleichem Signaltag (Diversifikations-/Cluster-Risiko)
  • Termine: nächstes Quartalsdatum laut yfinance calendar, Handelstage bis dahin,
    Flag ob innerhalb eines typischen Swing-Fensters (3–15 Handelstage)

Keine harten externen Abhängigkeiten außer yfinance/pandas/numpy.
"""
from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from datetime import date, datetime

warnings.filterwarnings(
    "ignore", message=".*fill_method.*pct_change.*", category=FutureWarning
)

import numpy as np
import pandas as pd
import yfinance as yf

# --- Liquidität -----------------------------------------------------------------

LIQ_VERY_THIN_PCT = 15.0  # unter diesem Perzentil am Signaltag
LIQ_THIN_PCT = 35.0


def _ohlc_for_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            o = raw["Open"][ticker]
            c = raw["Close"][ticker]
            v = raw["Volume"][ticker]
        else:
            o, c, v = raw["Open"], raw["Close"], raw["Volume"]
    except Exception:
        return None
    df = pd.DataFrame({"Open": o, "Close": c, "Volume": v})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index().dropna(how="all")


def adv_currency_20d(raw: pd.DataFrame, ticker: str, signal_d: pd.Timestamp) -> float:
    """Mittlerer Tagesumsatz (Close*Volume) über die letzten 20 Handelsschlüsse bis einschl. signal_d."""
    df = _ohlc_for_ticker(raw, ticker)
    if df is None or df.empty:
        return float("nan")
    sig = pd.Timestamp(signal_d).normalize()
    hist = df[df.index <= sig]
    if hist.empty:
        return float("nan")
    last20 = hist.tail(20)
    if last20["Close"].isna().all() or last20["Volume"].isna().all():
        return float("nan")
    turn = (last20["Close"].astype(float) * last20["Volume"].astype(float)).dropna()
    if turn.empty:
        return float("nan")
    return float(turn.mean())


def liquidity_label_from_pctile(pct: float) -> str:
    if not np.isfinite(pct):
        return "unknown"
    if pct < LIQ_VERY_THIN_PCT:
        return "very_thin"
    if pct < LIQ_THIN_PCT:
        return "thin"
    return "ok"


# --- Meta -----------------------------------------------------------------------


def meta_prob_margin(prob: float, threshold: float) -> float:
    return float(prob) - float(threshold)


# --- Cluster / Korrelation ------------------------------------------------------


def cluster_counts(sig: pd.DataFrame) -> pd.DataFrame:
    """Pro Zeile: signals_same_day, signals_same_sector_same_day, sector_share_same_day."""
    sig = sig.copy()
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    day_ct = sig.groupby("Date")["ticker"].transform("count")
    sec_ct = sig.groupby(["Date", "sector"])["ticker"].transform("count")
    sig["signals_same_day"] = day_ct
    sig["signals_same_sector_same_day"] = sec_ct
    sig["sector_share_same_day"] = (sec_ct / day_ct).astype(float)
    return sig[
        ["signals_same_day", "signals_same_sector_same_day", "sector_share_same_day"]
    ]


def _return_matrix_up_to(
    closes: pd.DataFrame, end_d: pd.Timestamp, lookback: int
) -> pd.DataFrame | None:
    end_d = pd.Timestamp(end_d).normalize()
    sub = closes[closes.index <= end_d].tail(lookback + 1)
    if len(sub) < 10:
        return None
    r = sub.pct_change(fill_method=None).iloc[1:]
    return r.dropna(axis=1, how="all")


def mean_pairwise_corr(
    closes: pd.DataFrame, tickers: list[str], signal_d: pd.Timestamp, lookback: int = 60
) -> float:
    """Mittlere paarweise Korrelation der Tagesrenditen (gleiches Kalenderfenster)."""
    cols = [c for c in tickers if c in closes.columns]
    if len(cols) < 2:
        return float("nan")
    r = _return_matrix_up_to(closes[cols], signal_d, lookback)
    if r is None or r.shape[1] < 2:
        return float("nan")
    r = r.dropna(axis=0, how="any")
    if len(r) < 10:
        return float("nan")
    cm = r.corr()
    tri = cm.values[np.triu_indices_from(cm.values, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return float("nan")
    return float(np.mean(tri))


def cluster_mean_corr_by_date(
    sig: pd.DataFrame, closes: pd.DataFrame, lookback: int = 60
) -> pd.Series:
    """Pro Signaltag: mittlere Korrelation aller Titelpaare an diesem Tag (gleicher Wert pro Zeile)."""
    sig = sig.copy()
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    by_date: dict[pd.Timestamp, float] = {}
    for d, g in sig.groupby("Date"):
        tickers = g["ticker"].unique().tolist()
        by_date[d] = mean_pairwise_corr(closes, tickers, d, lookback)
    return sig["Date"].map(by_date).rename("cluster_mean_corr_60d")


# --- Earnings (yfinance calendar) -----------------------------------------------


def _parse_earnings_date(cal: dict | None) -> date | None:
    if not cal or "Earnings Date" not in cal:
        return None
    ed = cal["Earnings Date"]
    if ed is None:
        return None
    if isinstance(ed, (list, tuple)) and len(ed):
        ed = ed[0]
    if isinstance(ed, datetime):
        return ed.date()
    if isinstance(ed, date):
        return ed
    return None


@contextmanager
def _suppress_stderr():
    dev = open(os.devnull, "w", encoding="utf-8")
    old = sys.stderr
    try:
        sys.stderr = dev
        yield
    finally:
        sys.stderr = old
        dev.close()


def next_earnings_date(ticker: str) -> date | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with _suppress_stderr():
                cal = yf.Ticker(ticker).calendar
        except Exception:
            return None
    if not cal:
        return None
    return _parse_earnings_date(cal if isinstance(cal, dict) else None)


def bdays_from_signal_to(
    signal_d: pd.Timestamp, target_d: date | None
) -> float:
    """Anzahl Handelstage vom ersten Tag nach signal_d bis einschl. target_d."""
    if target_d is None:
        return float("nan")
    start = pd.Timestamp(signal_d).normalize()
    end = pd.Timestamp(target_d).normalize()
    if end <= start:
        return float("nan")
    b = pd.bdate_range(
        start=start + pd.Timedelta(days=1), end=end, freq="B", inclusive="both"
    )
    return float(len(b))


def earnings_flag_in_swing_window(
    bdays_to_earnings: float, hold_lo: int = 3, hold_hi: int = 15
) -> bool:
    if not np.isfinite(bdays_to_earnings):
        return False
    return hold_lo <= bdays_to_earnings <= hold_hi


# --- Alles in einem DataFrame ---------------------------------------------------


def enrich_signal_frame(
    sig: pd.DataFrame,
    raw: pd.DataFrame,
    corr_lookback: int = 60,
) -> pd.DataFrame:
    """
    sig: mind. ticker, Date, prob, threshold_used, sector
    raw: Multi-Index yfinance download wie in build_holdout_signals_master
    """
    out = sig.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()

    out["meta_prob_margin"] = [
        meta_prob_margin(p, t) for p, t in zip(out["prob"], out["threshold_used"])
    ]

    cc = cluster_counts(out[["ticker", "Date", "sector"]].copy())
    out = pd.concat([out.reset_index(drop=True), cc.reset_index(drop=True)], axis=1)

    # Close-Matrix für Korrelationen
    tickers = sorted(out["ticker"].unique())
    closes_list = []
    for t in tickers:
        df = _ohlc_for_ticker(raw, t)
        if df is None:
            continue
        s = df["Close"].rename(t)
        closes_list.append(s)
    if closes_list:
        closes = pd.concat(closes_list, axis=1).sort_index()
        out["cluster_mean_corr_60d"] = cluster_mean_corr_by_date(
            out[["ticker", "Date"]], closes, lookback=corr_lookback
        ).values
    else:
        out["cluster_mean_corr_60d"] = np.nan

    adv = []
    for _, r in out.iterrows():
        adv.append(adv_currency_20d(raw, r["ticker"], r["Date"]))
    out["adv_20d_local"] = adv

    # Perzentil-Rang des Umsatzes pro Signaltag (höher = liquid)
    out["adv_pctile_same_day"] = (
        out.groupby("Date")["adv_20d_local"]
        .rank(pct=True, method="average")
        .mul(100.0)
    )
    out["liquidity_tier"] = out["adv_pctile_same_day"].apply(liquidity_label_from_pctile)

    # Earnings (pro Ticker cachen)
    cache: dict[str, date | None] = {}

    def _ed(t: str) -> date | None:
        if t not in cache:
            cache[t] = next_earnings_date(t)
        return cache[t]

    out["next_earnings_date"] = [_ed(t) for t in out["ticker"]]
    bd = [
        bdays_from_signal_to(sd, ed)
        for sd, ed in zip(out["Date"], out["next_earnings_date"])
    ]
    out["bdays_to_next_earnings"] = bd
    out["earnings_in_3_15_bday_window"] = [
        earnings_flag_in_swing_window(b) for b in bd
    ]

    return out
