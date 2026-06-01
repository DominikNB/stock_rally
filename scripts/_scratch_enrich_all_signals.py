"""
Reichert alle OOS-Scratch-Signale an (yfinance + News-Shard + abgeleitete Kontext-Spalten).

Ausgabe: data/_scratch_signals_enriched.parquet

  python scripts/_scratch_enrich_all_signals.py
  python scripts/_scratch_enrich_all_signals.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*fill_method.*pct_change.*", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.ERROR)

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.features import merge_news_shard_from_best_params

SIGNALS_CSV = ROOT / "data" / "_scratch_meta_thr_final_signals.csv"
OUT_PATH = ROOT / "data" / "_scratch_signals_enriched.parquet"
SHARD_DIR = ROOT / "data" / "feature_shards_news"
YF_START = "2018-01-01"
SMA_BREADTH = 50
CROSS_ASSETS = {
    "gld_ret_5d": "GLD",
    "oil_ret_5d": "CL=F",
    "dxy_ret_5d": "DX-Y.NYB",
    "eurusd_ret_5d": "EURUSD=X",
}
VIX_SYM = "^VIX"
VIX3M_SYM = "^VIX3M"


def _load_macro_event_days() -> pd.DatetimeIndex:
    """US FOMC + CPI (naherungsweise) 2018–2026 — für Makro-Fenster-Tests."""
    path = ROOT / "data" / "_scratch_us_macro_event_days.json"
    if path.is_file():
        obj = json.loads(path.read_text(encoding="utf-8"))
        days = [pd.Timestamp(d).normalize() for d in obj.get("dates", [])]
        return pd.DatetimeIndex(days)
    # Fallback: nur FOMC-typische Mittwoch-Cluster aus bekannten Jahren (kompakt)
    fomc = pd.to_datetime(
        [
            "2018-03-21", "2018-06-13", "2018-09-26", "2018-12-19",
            "2019-03-20", "2019-06-19", "2019-09-18", "2019-12-11",
            "2020-03-15", "2020-03-23", "2020-04-29", "2020-06-10", "2020-07-29",
            "2020-09-16", "2020-11-05", "2020-12-16",
            "2021-03-17", "2021-06-16", "2021-09-22", "2021-12-15",
            "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21",
            "2022-11-02", "2022-12-14",
            "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
            "2023-09-20", "2023-11-01", "2023-12-13",
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
            "2024-09-18", "2024-11-07", "2024-12-18",
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
            "2025-09-17", "2025-12-10",
            "2026-01-28", "2026-03-18",
        ]
    ).normalize()
    cpi = pd.date_range("2018-01-01", "2026-12-01", freq="MS") + pd.offsets.BDay(11)
    days = fomc.union(cpi).normalize().unique()
    path.write_text(
        json.dumps({"dates": [d.strftime("%Y-%m-%d") for d in days]}, indent=2),
        encoding="utf-8",
    )
    return pd.DatetimeIndex(days)


def _bdays_to_event(signal_dates: pd.Series, event_days: pd.DatetimeIndex, window: int = 2) -> pd.Series:
    ev = pd.DatetimeIndex(event_days).sort_values()
    out = []
    for d in pd.to_datetime(signal_dates).dt.normalize():
        if pd.isna(d):
            out.append(np.nan)
            continue
        diffs = (ev - d).days
        near = np.abs(diffs) <= window
        out.append(int(near.any()) if len(diffs) else 0)
    return pd.Series(out, index=signal_dates.index, dtype=float)


def _ret_n_bdays(close: pd.Series, n: int = 5) -> pd.Series:
    return close.pct_change(n)


def _download_yf(tickers: list[str], end_d: str) -> pd.DataFrame:
    print(f"yfinance: {len(tickers)} Symbole ({YF_START} … {end_d}) …", flush=True)
    return yf.download(
        list(tickers),
        start=YF_START,
        end=end_d,
        auto_adjust=True,
        threads=True,
        progress=False,
    )


def _sector_breadth_ma50(raw: pd.DataFrame, universe: dict[str, str], sma: int = SMA_BREADTH) -> pd.DataFrame:
    """Anteil Universe-Ticker je (Date, sector) mit Close > SMA{sma}."""
    from lib.signal_extra_filters import _ohlc_for_ticker

    rows: list[dict] = []
    by_sec: dict[str, list[str]] = {}
    for t, s in universe.items():
        by_sec.setdefault(s, []).append(t)

    for sec, tickers in by_sec.items():
        above_list: list[pd.Series] = []
        for t in tickers:
            ohlc = _ohlc_for_ticker(raw, t)
            if ohlc is None or len(ohlc) < sma + 5:
                continue
            c = ohlc["Close"]
            sma_s = c.rolling(sma, min_periods=sma).mean()
            above = (c > sma_s).astype(float)
            above.name = t
            above_list.append(above)
        if not above_list:
            continue
        mat = pd.concat(above_list, axis=1)
        breadth = mat.mean(axis=1)
        for dt, val in breadth.items():
            rows.append({"Date": pd.Timestamp(dt).normalize(), "sector": sec, "sector_breadth_ma50": float(val)})
    if not rows:
        return pd.DataFrame(columns=["Date", "sector", "sector_breadth_ma50"])
    out = pd.DataFrame(rows)
    return out.groupby(["Date", "sector"], as_index=False)["sector_breadth_ma50"].mean()


def _merge_news_shard(sig: pd.DataFrame, best_params: dict) -> pd.DataFrame:
    cfg._FEATURE_NEWS_SHARDS_ACTIVE = True
    cfg.FEATURE_SHARD_DIR = str(SHARD_DIR.resolve())
    manifest_path = SHARD_DIR / "news_shards_manifest.json"
    if manifest_path.is_file():
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg.NEWS_SHARD_MANIFEST = {str(k): str(SHARD_DIR / v) for k, v in (obj.get("tags") or {}).items()}
    keys = sig[["Date", "ticker", "sector"]].copy()
    keys["Date"] = pd.to_datetime(keys["Date"]).dt.normalize()
    merged = merge_news_shard_from_best_params(keys, best_params)
    tone_macro = [c for c in merged.columns if c.startswith("news_macro") and c.endswith("_tone")]
    tone_sec = [c for c in merged.columns if c.startswith("news_sec") and c.endswith("_tone")]
    keep = ["Date", "ticker"] + tone_macro[:1] + tone_sec[:1]
    for pat in ("macro_sec_diff", "tone_z_w", "x_volz"):
        keep += [c for c in merged.columns if pat in c and c not in keep]
    keep = list(dict.fromkeys(keep))
    slim = merged[keep].drop_duplicates(subset=["Date", "ticker"])
    out = sig.merge(slim, on=["Date", "ticker"], how="left")
    if tone_macro and tone_sec:
        mc, sc = tone_macro[0], tone_sec[0]
        out["news_sec_minus_macro_tone"] = pd.to_numeric(out[sc], errors="coerce") - pd.to_numeric(
            out[mc], errors="coerce"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Cache ignorieren")
    args = ap.parse_args()

    if OUT_PATH.is_file() and not args.force:
        print(f"Vorhanden: {OUT_PATH}  (nutze --force zum Neuaufbau)")
        return

    if not SIGNALS_CSV.is_file():
        print(f"Fehlt: {SIGNALS_CSV}", file=sys.stderr)
        sys.exit(1)

    sig = pd.read_csv(SIGNALS_CSV)
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["sector"] = sig["ticker"].astype(str).map(cfg.TICKER_TO_SECTOR).fillna("unknown")
    if "prob" not in sig.columns:
        sig["prob"] = 0.85
    if "threshold_used" not in sig.columns:
        sig["threshold_used"] = 0.80

    end_d = (sig["Date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    tickers = sorted(sig["ticker"].astype(str).unique())
    raw = _download_yf(tickers, end_d)

    from lib.signal_extra_filters import enrich_signal_frame

    print("enrich_signal_frame …", flush=True)
    out = enrich_signal_frame(sig, raw)

    # Sektor-Breadth (Universum)
    universe = {t: cfg.TICKER_TO_SECTOR.get(t, "unknown") for t in cfg.TICKER_TO_SECTOR}
    uni_tickers = sorted(set(universe) | set(tickers))
    if len(uni_tickers) > len(tickers):
        print(f"Sektor-Breadth: lade {len(uni_tickers)} Universe-Ticker …", flush=True)
        raw_uni = _download_yf(uni_tickers, end_d)
    else:
        raw_uni = raw
    br = _sector_breadth_ma50(raw_uni, universe)
    out = out.merge(br, on=["Date", "sector"], how="left")

    # News-Shard
    try:
        import joblib

        art = joblib.load(ROOT / "models" / "scoring_artifacts.joblib")
        bp = art.get("best_params", {})
        print("News-Shard merge …", flush=True)
        out = _merge_news_shard(out, bp)
    except Exception as exc:
        print(f"News-Shard übersprungen: {exc}", flush=True)

    # Cross-Asset + VIX-Termstruktur
    cross_syms = list(CROSS_ASSETS.values()) + [VIX_SYM, VIX3M_SYM]
    print(f"Cross-Asset / VIX: {cross_syms}", flush=True)
    raw_x = _download_yf(cross_syms, end_d)

    def _close_for(sym: str) -> pd.Series | None:
        if isinstance(raw_x.columns, pd.MultiIndex):
            if ("Close", sym) in raw_x.columns:
                s = raw_x[("Close", sym)]
            elif sym in raw_x.columns.get_level_values(1, dropna=False):
                s = raw_x["Close"][sym]
            else:
                return None
        else:
            s = raw_x["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index).normalize()
        return s

    asset_rets: dict[str, pd.Series] = {}
    for col_name, sym in {**CROSS_ASSETS, "vix_level": VIX_SYM, "vix3m_level": VIX3M_SYM}.items():
        s = _close_for(sym)
        if s is None:
            continue
        asset_rets[col_name] = s if col_name.endswith("_level") else _ret_n_bdays(s, 5)

    xdf = pd.DataFrame(asset_rets)
    xdf.index = pd.to_datetime(xdf.index).normalize()
    xdf = xdf.reset_index().rename(columns={"index": "Date"})
    xdf["Date"] = pd.to_datetime(xdf["Date"]).dt.normalize()
    out = out.merge(xdf, on="Date", how="left")
    out["vix3m_vix_ratio"] = pd.to_numeric(out["vix3m_level"], errors="coerce") / (
        pd.to_numeric(out["vix_level"], errors="coerce") + 1e-9
    )

    # Makro-Kalender
    ev_days = _load_macro_event_days()
    out["macro_event_within_2bd"] = _bdays_to_event(out["Date"], ev_days, window=2)

    # Abgeleitete Flags
    out["is_red"] = pd.to_numeric(out["vix"], errors="coerce") < 20.0
    out["is_eu"] = out["ticker"].astype(str).str.contains(
        r"\.(?:DE|AS|PA|MI|L|SW|ST|HE|F)$", regex=True
    )
    out["rot_tech"] = out["is_red"] & (out["sector"] == "technology")
    out["rot_lag_sector_20d"] = out["is_red"] & (pd.to_numeric(out.get("ret_vs_sector_20d"), errors="coerce") < 0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Geschrieben: {OUT_PATH}  ({len(out)} Zeilen, {len(out.columns)} Spalten)")


if __name__ == "__main__":
    main()
