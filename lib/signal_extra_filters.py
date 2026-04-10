"""
Zusätzliche Filter-Spalten für Signale (über News/Makro/Chart hinaus) — alles für
data/master_complete.csv / master_daily_update.csv vorrechenbar, damit ein LLM die Werte direkt aus der CSV lesen kann:

  • Liquidität: adv_20d_local, adv_pctile_same_day, adv_rank_same_day, liquidity_tier
  • Meta: meta_prob_margin
  • Cluster/Sektor: signals_same_day, signals_same_sector_same_day, sector_share_same_day,
    n_sectors_same_day, sector_hhi_same_day
  • Korrelation: cluster_mean_corr_60d (60 Handelstage Lookback), cluster_corr_pairwise_valid
  • Kurs/OHLCV (yfinance): volatility_20d/60d/ann, momentum_20d/60d, dist_from_20d_high/low_pct,
    ret_1d_signal_day, avg_hl_range_pct_14d, volume_zscore_20d
  • Cross-Section je Signaltag: rank_prob_same_day, pct_rank_prob_same_day, prob_zscore_same_day
  • Termine (yfinance calendar): next_earnings_date, bdays_to_next_earnings,
    earnings_in_3_15_bday_window, earnings_date_known, earnings_beyond_swing_15b, earnings_too_soon_lt3b
  • Relative Stärke: ret_vs_spy_5d (5 Handelstage vs. SPY), ret_vs_sector_5d (vs. grobem US-Sektor-ETF, falls gemappt)
  • Gap: open_gap_pct (Open vs. Vortages-Close am Signaltag)
  • yfinance info (Snapshot): short_float_pct, short_days_to_cover, inst_own_pct (jeweils % 0–100, kein Trend — nur Stand)
  • Markt/Sektor (yfinance Index/ETF, dynamisch zur Notierung): market_bench_symbol, sector_bench_symbol,
    market_ret_1d/2d/3d, sector_ret_1d/2d/3d (kumulativ über 1/2/3 Handelstage bis Signaltag)
  • Platzhalter: news_sentiment (derzeit nicht befüllt — NaN)

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

# Vollständige Header für docs/website_analysis_prompt.txt (LLM liest nur vorhandene Spaltennamen).
_OHLC_LLM_COLS = (
    "volatility_20d",
    "volatility_60d",
    "volatility_20d_ann",
    "momentum_20d",
    "momentum_60d",
    "dist_from_20d_high_pct",
    "dist_from_20d_low_pct",
    "ret_1d_signal_day",
    "avg_hl_range_pct_14d",
    "volume_zscore_20d",
)
_LLM_EXTRA_COLS = (
    "meta_prob_margin",
    "signals_same_day",
    "signals_same_sector_same_day",
    "sector_share_same_day",
    "n_sectors_same_day",
    "sector_hhi_same_day",
    "cluster_mean_corr_60d",
    "cluster_corr_pairwise_valid",
    "adv_20d_local",
    "adv_pctile_same_day",
    "adv_rank_same_day",
    "liquidity_tier",
    "rank_prob_same_day",
    "pct_rank_prob_same_day",
    "prob_zscore_same_day",
    "next_earnings_date",
    "bdays_to_next_earnings",
    "earnings_in_3_15_bday_window",
    "earnings_date_known",
    "earnings_beyond_swing_15b",
    "earnings_too_soon_lt3b",
    "ret_vs_spy_5d",
    "ret_vs_sector_5d",
    "open_gap_pct",
    "short_float_pct",
    "short_days_to_cover",
    "inst_own_pct",
    "market_ret_1d",
    "market_ret_2d",
    "market_ret_3d",
    "sector_ret_1d",
    "sector_ret_2d",
    "sector_ret_3d",
    "market_bench_symbol",
    "sector_bench_symbol",
    "news_sentiment",
)

# Grobe US-Sektor-ETFs (Vergleich „eigene Story vs. Sektor“); unbekannte Sektoren → NaN in ret_vs_sector_5d
_SECTOR_TO_BENCH_ETF: dict[str, str] = {
    "tech": "XLK",
    "finance": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "industrial": "XLI",
    "materials": "XLB",
    "consumer": "XLY",
    "automotive": "CARZ",
    "real_estate": "XLRE",
    "telecom": "VOX",
    "media": "XLC",
    "crypto": "BITO",
}
_BENCH_SPY = "SPY"

# Suffix → Leitindex, wenn yfinance `info` keine eindeutige Region liefert
_LEAD_INDEX_BY_TICKER_SUFFIX: list[tuple[tuple[str, ...], str]] = [
    ((".DE", ".F"), "^GDAXI"),
    ((".PA",), "^FCHI"),
    ((".L",), "^FTSE"),
    ((".T",), "^N225"),
    ((".MI",), "^STOXX50E"),
    ((".SW",), "^SSMI"),
    ((".AS",), "^AEX"),
    ((".BR",), "^BFX"),
    ((".VI",), "^ATX"),
    ((".AX",), "^AXJO"),
    ((".SA",), "^BVSP"),
    ((".TO",), "^GSPTSE"),
    ((".HK",), "^HSI"),
    ((".KS",), "^KS11"),
    ((".TW",), "^TWII"),
    ((".ST",), "^OMX"),
]

# EU: iShares STOXX Europe 600 Sector UCITS (DE-Listing) — grobe Branchen-Zuordnung
_EU_SECTOR_TO_BENCH_ETF: dict[str, str] = {
    "tech": "EXV3.DE",
    "finance": "EXH1.DE",
    "healthcare": "EXV2.DE",
    "energy": "EXV4.DE",
    "industrial": "EXV1.DE",
    "materials": "EXV5.DE",
    "consumer": "EXV6.DE",
    "automotive": "EXV1.DE",
    "real_estate": "EXV7.DE",
    "telecom": "EXV8.DE",
    "media": "EXV9.DE",
    "crypto": "BITO",
}


def _lead_index_from_suffix_only(ticker: str) -> str:
    """Fallback: Leitindex nur aus Ticker-Suffix."""
    t = str(ticker).upper().strip()
    for suffixes, sym in _LEAD_INDEX_BY_TICKER_SUFFIX:
        if any(t.endswith(s) for s in suffixes):
            return sym
    return "^GSPC"


def _lead_index_from_info(ticker: str, info: dict | None) -> str:
    """
    Leitindex passend zur Notierung: zuerst Land/Börse aus yfinance ``info``,
    sonst Suffix-Heuristik (nicht immer DAX — z. B. US → S&P 500).
    """
    inf = info or {}
    t = str(ticker).upper().strip()
    c = (inf.get("country") or "").strip().lower()
    ex = (inf.get("exchange") or "").upper()

    if c in ("united states", "usa"):
        return "^GSPC"
    if c in ("germany", "deutschland"):
        return "^GDAXI"
    if c in ("france",):
        return "^FCHI"
    if c in ("united kingdom", "uk"):
        return "^FTSE"
    if c in ("japan",):
        return "^N225"
    if c in ("switzerland",):
        return "^SSMI"
    if c in ("netherlands",):
        return "^AEX"
    if c in ("italy",):
        return "^STOXX50E"
    if c in ("belgium",):
        return "^BFX"
    if c in ("austria",):
        return "^ATX"
    if c in ("australia",):
        return "^AXJO"
    if c in ("canada",):
        return "^GSPTSE"
    if c in ("hong kong", "hong kong sar"):
        return "^HSI"
    if c in ("sweden",):
        return "^OMX"
    if c in ("south korea", "korea", "republic of korea"):
        return "^KS11"
    if c in ("taiwan",):
        return "^TWII"
    if c in ("brazil",):
        return "^BVSP"

    if any(x in ex for x in ("NMS", "NYSE", "NAS", "PCX", "NGM")):
        return "^GSPC"
    if any(x in ex for x in ("BER", "FRA", "XETRA", "HAN", "MUN", "DUS", "STU")):
        return "^GDAXI"
    if any(x in ex for x in ("PAR", "EPA")):
        return "^FCHI"
    if any(x in ex for x in ("LSE", "LON")):
        return "^FTSE"
    if "JPX" in ex or "TOKYO" in ex:
        return "^N225"

    return _lead_index_from_suffix_only(ticker)


def _use_eu_stoxx_sector_bench(ticker: str, info: dict | None) -> bool:
    """Europäische STOXX-600-Sektor-ETFs statt US-SPDR, wenn Notierung EU/CH/UK-nah."""
    inf = info or {}
    t = str(ticker).upper().strip()
    if any(
        t.endswith(s)
        for s in (".DE", ".PA", ".MI", ".AS", ".BR", ".VI", ".F", ".SW", ".L")
    ):
        return True
    c = (inf.get("country") or "").strip().lower()
    eu = {
        "germany",
        "france",
        "italy",
        "spain",
        "netherlands",
        "belgium",
        "austria",
        "ireland",
        "finland",
        "portugal",
        "greece",
        "luxembourg",
    }
    if c in eu or c in ("switzerland", "united kingdom", "uk"):
        return True
    return False


def _sector_bench_etf(
    sector: str | float | None, ticker: str, info: dict | None
) -> str | None:
    """
    Sektor-Benchmark-ETF zur Aktie: EU → STOXX Europe 600 Sector UCITS (DE),
    sonst US-Sektor-ETF (XLK, …). Unbekanntes Label → None.
    """
    if sector is None or (isinstance(sector, float) and np.isnan(sector)):
        return None
    key = str(sector).strip().lower().replace(" ", "_")
    if not key or key not in _SECTOR_TO_BENCH_ETF:
        return None
    if _use_eu_stoxx_sector_bench(ticker, info):
        return _EU_SECTOR_TO_BENCH_ETF.get(key) or _SECTOR_TO_BENCH_ETF.get(key)
    return _SECTOR_TO_BENCH_ETF.get(key)


def _warm_ticker_info_cache(tickers: list[str], cache: dict[str, dict]) -> None:
    """Lädt ``yf.Ticker(t).info`` je Ticker höchstens einmal (für Short, Benchmarks)."""
    for t in tickers:
        ts = str(t)
        if ts in cache:
            continue
        try:
            cache[ts] = yf.Ticker(ts).info or {}
        except Exception:
            cache[ts] = {}


def ensure_llm_signal_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt sicher, dass jede im Website-Prompt genannte Struktur-Spalte im Export existiert.
    Fehlende Spalten (z. B. wenn OHLC-Merge leer blieb) werden mit NaN/False/unknown gefüllt,
    damit LLMs nicht fälschlich von „fehlenden Spalten“ ausgehen.
    """
    o = out.copy()
    for c in _LLM_EXTRA_COLS:
        if c not in o.columns:
            if c == "liquidity_tier":
                o[c] = "unknown"
            elif c in (
                "cluster_corr_pairwise_valid",
                "earnings_in_3_15_bday_window",
                "earnings_date_known",
                "earnings_beyond_swing_15b",
                "earnings_too_soon_lt3b",
            ):
                o[c] = False
            elif c == "next_earnings_date":
                o[c] = pd.NaT
            elif c in ("market_bench_symbol", "sector_bench_symbol"):
                o[c] = ""
            else:
                o[c] = np.nan
    for c in _OHLC_LLM_COLS:
        if c not in o.columns:
            o[c] = np.nan
    return o


def _ohlc_for_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            o = raw["Open"][ticker]
            c = raw["Close"][ticker]
            v = raw["Volume"][ticker]
            h = raw["High"][ticker]
            l = raw["Low"][ticker]
        else:
            o, c, v = raw["Open"], raw["Close"], raw["Volume"]
            h, l = raw["High"], raw["Low"]
    except Exception:
        return None
    df = pd.DataFrame({"Open": o, "Close": c, "Volume": v, "High": h, "Low": l})
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


def _sector_metrics_by_date(sig: pd.DataFrame) -> pd.DataFrame:
    """n_sectors_same_day, sector_hhi_same_day (Herfindahl der Sektoranteile am Signaltag)."""
    s = sig.copy()
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    s["_sec"] = s["sector"].astype(str).str.strip().replace("", "unknown")

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n == 0:
            return pd.Series({"sector_hhi_same_day": np.nan, "n_sectors_same_day": np.nan})
        vc = g["_sec"].value_counts()
        hhi = float(((vc / n) ** 2).sum())
        return pd.Series(
            {"sector_hhi_same_day": hhi, "n_sectors_same_day": int(g["_sec"].nunique())}
        )

    return s.groupby("Date", sort=False).apply(_agg).reset_index()


def _merge_ticker_date_features(raw: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """
    Volatilität, Momentum, Abstand zu 20d-Hoch/Tief, Intraday-Range, Volumen-Z-Score — alles aus OHLCV.
    """
    tickers = sorted(out["ticker"].unique())
    parts: list[pd.DataFrame] = []
    for t in tickers:
        df = _ohlc_for_ticker(raw, t)
        if df is None or len(df) < 10:
            continue
        c = df["Close"].astype(float)
        ret = c.pct_change()
        vol = df["Volume"].astype(float)
        hi = df["High"].astype(float)
        lo = df["Low"].astype(float)
        rm20 = c.rolling(20, min_periods=5).max()
        rmi20 = c.rolling(20, min_periods=5).min()
        hl_pct = (hi - lo) / c.replace(0, np.nan)
        fe = pd.DataFrame(
            {
                "Date": pd.to_datetime(df.index).normalize(),
                "ticker": t,
                "volatility_20d": ret.rolling(20, min_periods=10).std(),
                "volatility_60d": ret.rolling(60, min_periods=20).std(),
                "volatility_20d_ann": ret.rolling(20, min_periods=10).std() * np.sqrt(252.0),
                "momentum_20d": c / c.shift(20) - 1.0,
                "momentum_60d": c / c.shift(60) - 1.0,
                "dist_from_20d_high_pct": (c - rm20) / rm20.replace(0, np.nan),
                "dist_from_20d_low_pct": (c - rmi20) / rmi20.replace(0, np.nan),
                "ret_1d_signal_day": ret,
                "avg_hl_range_pct_14d": hl_pct.rolling(14, min_periods=5).mean(),
                "volume_zscore_20d": (vol - vol.rolling(20, min_periods=10).mean())
                / vol.rolling(20, min_periods=10).std().replace(0, np.nan),
            }
        )
        parts.append(fe)
    if not parts:
        return out
    feat = pd.concat(parts, ignore_index=True)
    feat["Date"] = pd.to_datetime(feat["Date"]).dt.normalize()
    o = out.copy()
    o["Date"] = pd.to_datetime(o["Date"]).dt.normalize()
    return o.merge(feat, on=["ticker", "Date"], how="left")


def _pct_ret_last_n_trading_days(close: pd.Series | None, end: pd.Timestamp, n: int = 5) -> float:
    """Kursrendite über die letzten n Handelstage bis einschließlich ``end`` (Close/PrevClose_n - 1)."""
    if close is None:
        return float("nan")
    end = pd.Timestamp(end).normalize()
    s = close.dropna().sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    sub = s[s.index <= end]
    if len(sub) < n + 1:
        return float("nan")
    c_now = float(sub.iloc[-1])
    c_prev = float(sub.iloc[-(n + 1)])
    if not np.isfinite(c_now) or not np.isfinite(c_prev) or c_prev == 0:
        return float("nan")
    return c_now / c_prev - 1.0


def _yf_close_series_single(ticker: str, start_d: str, end_d: str) -> pd.Series | None:
    try:
        df = yf.download(
            ticker,
            start=start_d,
            end=end_d,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        if df is None or len(df) == 0:
            return None
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c.sort_index().astype(float)
    except Exception:
        return None


def _sector_etf_for_label(sector: str | float | None) -> str | None:
    if sector is None or (isinstance(sector, float) and np.isnan(sector)):
        return None
    key = str(sector).strip().lower().replace(" ", "_")
    return _SECTOR_TO_BENCH_ETF.get(key)


def _open_gap_pct_ohlc(df: pd.DataFrame | None, d: pd.Timestamp) -> float:
    """Open vs. Vortages-Close am Tag ``d`` (Anteil)."""
    if df is None or len(df) < 2:
        return float("nan")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    dn = pd.Timestamp(d).normalize()
    if dn not in df.index:
        return float("nan")
    pos = df.index.get_loc(dn)
    if isinstance(pos, slice):
        return float("nan")
    if pos == 0:
        return float("nan")
    o = float(df.iloc[pos]["Open"])
    prev_c = float(df.iloc[pos - 1]["Close"])
    if not np.isfinite(o) or not np.isfinite(prev_c) or prev_c == 0:
        return float("nan")
    return (o - prev_c) / prev_c


def _open_gap_pct_raw(raw: pd.DataFrame, ticker: str, d: pd.Timestamp) -> float:
    return _open_gap_pct_ohlc(_ohlc_for_ticker(raw, ticker), d)


def _yf_info_short_snapshot(ticker: str, cache: dict[str, dict]) -> tuple[float, float, float]:
    """(short_float_pct 0–100, short_days_to_cover, inst_own_pct). Fehler → NaN."""
    if ticker not in cache:
        try:
            cache[ticker] = yf.Ticker(ticker).info or {}
        except Exception:
            cache[ticker] = {}
    inf = cache[ticker]
    sf = inf.get("shortPercentOfFloat")
    if sf is not None:
        sf = float(sf)
        if sf <= 1.0:
            sf *= 100.0
    else:
        sf = float("nan")
    sdc = inf.get("shortRatio")
    sdc = float(sdc) if sdc is not None else float("nan")
    inst = inf.get("heldPercentInstitutions")
    if inst is not None:
        inst = float(inst)
        if inst <= 1.0:
            inst *= 100.0
    else:
        inst = float("nan")
    return sf, sdc, inst


def _add_rs_gap_short_columns(
    out: pd.DataFrame, raw: pd.DataFrame, info_cache: dict[str, dict]
) -> pd.DataFrame:
    """Relative Stärke vs. SPY/Sektor-ETF, Gap, Short Interest / Inst.-Quote (Snapshot)."""
    o = out.copy()
    o["Date"] = pd.to_datetime(o["Date"]).dt.normalize()
    d_min = o["Date"].min() - pd.Timedelta(days=45)
    d_max = o["Date"].max() + pd.Timedelta(days=5)
    start_s = d_min.strftime("%Y-%m-%d")
    end_s = d_max.strftime("%Y-%m-%d")

    spy = _yf_close_series_single(_BENCH_SPY, start_s, end_s)
    sector_etfs: set[str] = set()
    for _, r in o.iterrows():
        etf = _sector_bench_etf(
            r.get("sector"), r["ticker"], info_cache.get(str(r["ticker"]), {})
        )
        if etf:
            sector_etfs.add(etf)
    bench_series: dict[str, pd.Series | None] = {_BENCH_SPY: spy}
    for etf in sector_etfs:
        if etf not in bench_series:
            bench_series[etf] = _yf_close_series_single(etf, start_s, end_s)

    ohlc_cache: dict[str, pd.DataFrame | None] = {}

    def _st(tk: str) -> pd.DataFrame | None:
        if tk not in ohlc_cache:
            ohlc_cache[tk] = _ohlc_for_ticker(raw, tk)
        return ohlc_cache[tk]

    ret_vs_spy: list[float] = []
    ret_vs_sec: list[float] = []
    gap_pct: list[float] = []
    for _, r in o.iterrows():
        t = r["ticker"]
        d = pd.Timestamp(r["Date"]).normalize()
        st = _st(t)
        rs = float("nan")
        rspy = float("nan")
        if st is not None and len(st) >= 2:
            c = st["Close"].astype(float)
            c.index = pd.to_datetime(st.index).normalize()
            rs = _pct_ret_last_n_trading_days(c, d, 5)
        if spy is not None:
            rspy = _pct_ret_last_n_trading_days(spy, d, 5)
        if np.isfinite(rs) and np.isfinite(rspy):
            ret_vs_spy.append(float(rs - rspy))
        else:
            ret_vs_spy.append(float("nan"))

        etf = _sector_bench_etf(
            r.get("sector"), t, info_cache.get(str(t), {})
        )
        rsec = float("nan")
        if etf:
            ser = bench_series.get(etf)
            if ser is not None and st is not None:
                c = st["Close"].astype(float)
                c.index = pd.to_datetime(c.index).normalize()
                rs2 = _pct_ret_last_n_trading_days(c, d, 5)
                rb = _pct_ret_last_n_trading_days(ser, d, 5)
                if np.isfinite(rs2) and np.isfinite(rb):
                    rsec = float(rs2 - rb)
        ret_vs_sec.append(rsec)

        gap_pct.append(_open_gap_pct_ohlc(st, d))

    o["ret_vs_spy_5d"] = ret_vs_spy
    o["ret_vs_sector_5d"] = ret_vs_sec
    o["open_gap_pct"] = gap_pct

    sf_l: list[float] = []
    sd_l: list[float] = []
    inst_l: list[float] = []
    for t in o["ticker"]:
        sf, sd, inst = _yf_info_short_snapshot(str(t), info_cache)
        sf_l.append(sf)
        sd_l.append(sd)
        inst_l.append(inst)
    o["short_float_pct"] = sf_l
    o["short_days_to_cover"] = sd_l
    o["inst_own_pct"] = inst_l

    return o


def _add_market_sector_bench_columns(
    out: pd.DataFrame, info_cache: dict[str, dict]
) -> pd.DataFrame:
    """
    Leitindex und Sektor-ETF **passend zur Notierung** (``info`` + Fallback Suffix);
    kumulierte Renditen über 1/2/3 Handelstage bis Signaltag (wie ``_pct_ret_last_n_trading_days``).
    """
    o = out.copy()
    o["Date"] = pd.to_datetime(o["Date"]).dt.normalize()
    d_min = o["Date"].min() - pd.Timedelta(days=21)
    d_max = o["Date"].max() + pd.Timedelta(days=5)
    start_s = d_min.strftime("%Y-%m-%d")
    end_s = d_max.strftime("%Y-%m-%d")

    lead_syms: set[str] = set()
    sec_syms: set[str] = set()
    for _, r in o.iterrows():
        inf = info_cache.get(str(r["ticker"]), {})
        lead_syms.add(_lead_index_from_info(r["ticker"], inf))
        etf = _sector_bench_etf(r.get("sector"), r["ticker"], inf)
        if etf:
            sec_syms.add(etf)

    all_syms = lead_syms | sec_syms
    series_cache: dict[str, pd.Series | None] = {}
    for sym in all_syms:
        series_cache[sym] = _yf_close_series_single(sym, start_s, end_s)

    m1: list[float] = []
    m2: list[float] = []
    m3: list[float] = []
    s1: list[float] = []
    s2: list[float] = []
    s3: list[float] = []
    mlab: list[str] = []
    slab: list[str] = []
    for _, r in o.iterrows():
        d = pd.Timestamp(r["Date"]).normalize()
        inf = info_cache.get(str(r["ticker"]), {})
        li = _lead_index_from_info(r["ticker"], inf)
        mlab.append(li)
        ser_li = series_cache.get(li)
        m1.append(_pct_ret_last_n_trading_days(ser_li, d, 1))
        m2.append(_pct_ret_last_n_trading_days(ser_li, d, 2))
        m3.append(_pct_ret_last_n_trading_days(ser_li, d, 3))

        etf = _sector_bench_etf(r.get("sector"), r["ticker"], inf)
        if etf:
            slab.append(etf)
            ser_etf = series_cache.get(etf)
            s1.append(_pct_ret_last_n_trading_days(ser_etf, d, 1))
            s2.append(_pct_ret_last_n_trading_days(ser_etf, d, 2))
            s3.append(_pct_ret_last_n_trading_days(ser_etf, d, 3))
        else:
            slab.append("")
            s1.append(float("nan"))
            s2.append(float("nan"))
            s3.append(float("nan"))

    o["market_bench_symbol"] = mlab
    o["sector_bench_symbol"] = slab
    o["market_ret_1d"] = m1
    o["market_ret_2d"] = m2
    o["market_ret_3d"] = m3
    o["sector_ret_1d"] = s1
    o["sector_ret_2d"] = s2
    o["sector_ret_3d"] = s3
    o["news_sentiment"] = np.nan
    return o


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

    info_cache: dict[str, dict] = {}
    _warm_ticker_info_cache(sorted(sig["ticker"].astype(str).unique().tolist()), info_cache)

    out["meta_prob_margin"] = [
        meta_prob_margin(p, t) for p, t in zip(out["prob"], out["threshold_used"])
    ]

    cc = cluster_counts(out[["ticker", "Date", "sector"]].copy())
    out = pd.concat([out.reset_index(drop=True), cc.reset_index(drop=True)], axis=1)

    _sec_agg = _sector_metrics_by_date(out[["Date", "sector"]])
    out = out.merge(_sec_agg, on="Date", how="left")

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

    out["cluster_corr_pairwise_valid"] = out["signals_same_day"].fillna(0).astype(int) >= 2

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
    out["adv_rank_same_day"] = out.groupby("Date")["adv_20d_local"].rank(
        ascending=False, method="min"
    )
    out["rank_prob_same_day"] = out.groupby("Date")["prob"].rank(ascending=False, method="min")
    out["pct_rank_prob_same_day"] = (
        out.groupby("Date")["prob"].rank(pct=True, method="average").mul(100.0)
    )

    def _z(s: pd.Series) -> pd.Series:
        m = s.mean()
        st = s.std(ddof=0)
        if not np.isfinite(st) or st == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / st

    out["prob_zscore_same_day"] = out.groupby("Date")["prob"].transform(_z)

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
    out["earnings_date_known"] = out["next_earnings_date"].notna()
    out["earnings_beyond_swing_15b"] = [
        bool(np.isfinite(b) and b > 15.0) for b in bd
    ]
    out["earnings_too_soon_lt3b"] = [bool(np.isfinite(b) and b < 3.0) for b in bd]

    out = _merge_ticker_date_features(raw, out)
    out = _add_rs_gap_short_columns(out, raw, info_cache)
    out = _add_market_sector_bench_columns(out, info_cache)

    return ensure_llm_signal_columns(out)
