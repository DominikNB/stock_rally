"""
Branchenklassifikation aus Yahoo Finance (yfinance): nur die zwei Felder, die Yahoo zuverlässig liefert:

 • ``sector`` → ``gics_sector``
  • ``industry`` → ``gics_industry`` (ein String, wie von Yahoo gesetzt)

Zusätzlich ``gics_sector_key`` / ``gics_industry_key``: normalisiert für Merges (z. B. GDELT).
``classification_standard`` kennzeichnet US vs. EU-Suffix grob (ICB_or_GICS_Yahoo vs. GICS_Yahoo).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASSIFICATION_DISPLAY_KEYS = (
    "classification_standard",
    "gics_sector",
    "gics_industry",
)
CLASSIFICATION_MERGE_KEYS = (
    "gics_sector_key",
    "gics_industry_key",
)
CLASSIFICATION_COLUMN_KEYS = CLASSIFICATION_DISPLAY_KEYS + CLASSIFICATION_MERGE_KEYS
_DEFAULT_CACHE_PATH = Path("data") / "equity_classification_cache.json"


def normalize_taxonomy_label(s: str) -> str:
    """Yahoo-/ICB-ähnliche Branchenstrings für Joins bereinigen (GDELT, Lookup-Tabellen)."""
    if not s:
        return ""
    t = str(s).strip()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _ticker_symbol_str(ticker: Any) -> str:
    if ticker is None:
        return ""
    try:
        if pd.isna(ticker):
            return ""
    except (ValueError, TypeError):
        pass
    s = str(ticker).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _infer_classification_standard(ticker: str) -> str:
    ts = _ticker_symbol_str(ticker)
    if not ts:
        return "GICS_Yahoo"
    u = ts.upper()
    if u.endswith(".L") or u.endswith(".PA") or u.endswith(".AS") or u.endswith(".BR"):
        return "ICB_or_GICS_Yahoo"
    if u.endswith(".DE") or u.endswith(".F") or u.endswith(".SW"):
        return "ICB_or_GICS_Yahoo"
    return "GICS_Yahoo"


def equity_classification_from_info(ticker: str, info: dict[str, Any] | None) -> dict[str, str]:
    inf = info or {}
    ts = _ticker_symbol_str(ticker)
    sector = str(inf.get("sector") or inf.get("sectorDisp") or "").strip()
    industry = str(inf.get("industry") or inf.get("industryDisp") or "").strip()
    return {
        "classification_standard": _infer_classification_standard(ts),
        "gics_sector": sector,
        "gics_industry": industry,
        "gics_sector_key": normalize_taxonomy_label(sector),
        "gics_industry_key": normalize_taxonomy_label(industry),
    }


def get_equity_classification(ticker: str, info: dict[str, Any] | None = None) -> dict[str, str]:
    ts = _ticker_symbol_str(ticker)
    if not ts:
        return equity_classification_from_info("", info)
    if info is not None:
        return equity_classification_from_info(ts, info)
    import yfinance as yf

    try:
        raw = yf.Ticker(ts).info or {}
    except Exception:
        raw = {}
    return equity_classification_from_info(ts, raw)


def _read_classification_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    if not isinstance(payload, dict):
        return out
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, dict):
        return out
    for tk, row in raw_rows.items():
        ts = _ticker_symbol_str(tk)
        if not ts or not isinstance(row, dict):
            continue
        out[ts] = {
            "classification_standard": str(row.get("classification_standard") or ""),
            "gics_sector": str(row.get("gics_sector") or ""),
            "gics_industry": str(row.get("gics_industry") or ""),
            "gics_sector_key": str(row.get("gics_sector_key") or ""),
            "gics_industry_key": str(row.get("gics_industry_key") or ""),
        }
    return out


def _write_classification_cache(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at_epoch_s": int(time.time()),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_classification_cache(
    tickers: list[str],
    *,
    progress_every: int = 0,
    cache_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """``ticker`` → Klassifikations-Dict mit persistentem Cache (nur neue Ticker werden nachgeladen)."""
    import yfinance as yf

    p = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE_PATH
    persisted = _read_classification_cache(p)
    out: dict[str, dict[str, str]] = {}
    uniq_tickers = []
    seen: set[str] = set()
    for t in tickers:
        ts = _ticker_symbol_str(t)
        if not ts or ts in seen:
            continue
        seen.add(ts)
        uniq_tickers.append(ts)

    missing = [ts for ts in uniq_tickers if ts not in persisted]
    n = len(missing)
    step = progress_every if progress_every > 0 else max(1, n // 10) if n > 0 else 1
    t0 = time.perf_counter()
    for i, ts in enumerate(missing, 1):
        try:
            inf = yf.Ticker(ts).info or {}
        except Exception:
            inf = {}
        persisted[ts] = equity_classification_from_info(ts, inf)
        if i == 1 or i % step == 0 or i == n:
            print(
                f"  … Branchenklassifikation (yfinance, neu) {i}/{n} Ticker "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    if n > 0:
        _write_classification_cache(p, persisted)
        print(
            f"  Branchenklassifikation-Cache aktualisiert: {n} neue Ticker "
            f"(gesamt {len(persisted)}) -> {p}",
            flush=True,
        )
    else:
        print(
            f"  Branchenklassifikation aus Cache: 0 neue Ticker "
            f"(gesamt {len(persisted)})",
            flush=True,
        )

    for ts in uniq_tickers:
        row = persisted.get(ts)
        if row is not None:
            out[ts] = row
    return out


def gics_label_maps_from_cache(
    cache: dict[str, dict[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Aus Yahoo-Strings im Cache stabile 0..n-Labels für Modell-Features (pro Lauf)."""
    secs = sorted(
        {
            (v.get("gics_sector") or "").strip()
            for v in cache.values()
            if (v.get("gics_sector") or "").strip()
        }
    )
    inds = sorted(
        {
            (v.get("gics_industry") or "").strip()
            for v in cache.values()
            if (v.get("gics_industry") or "").strip()
        }
    )
    return {s: i for i, s in enumerate(secs)}, {s: i for i, s in enumerate(inds)}


def add_yahoo_gics_feature_columns(df: pd.DataFrame, cache: dict[str, dict[str, str]]) -> None:
    """
    Zwei Yahoo-Hierarchiestufen (yfinance ``sector`` / ``industry``) als Strings + float-IDs.
    Unbekannt/leer → NaN in den *_id-Spalten. In-place.
    """
    empty = empty_classification_row()
    smap, imap = gics_label_maps_from_cache(cache)

    def gsec_raw(t):
        ts = _ticker_symbol_str(t)
        v = cache.get(ts, empty) if ts else empty
        return (v.get("gics_sector") or "").strip()

    def gind_raw(t):
        ts = _ticker_symbol_str(t)
        v = cache.get(ts, empty) if ts else empty
        return (v.get("gics_industry") or "").strip()

    gs = df["ticker"].map(gsec_raw)
    gi = df["ticker"].map(gind_raw)
    df["gics_sector"] = gs
    df["gics_industry"] = gi
    df["gics_sector_id"] = gs.map(lambda s: float(smap[s]) if s in smap else np.nan)
    df["gics_industry_id"] = gi.map(lambda s: float(imap[s]) if s in imap else np.nan)


def empty_classification_row() -> dict[str, str]:
    return {k: "" for k in CLASSIFICATION_COLUMN_KEYS}
