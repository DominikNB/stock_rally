"""
Branchenklassifikation aus Yahoo Finance (yfinance): nur die zwei Felder, die Yahoo zuverlässig liefert:

 • ``sector`` → ``gics_sector``
  • ``industry`` → ``gics_industry`` (ein String, wie von Yahoo gesetzt)

Zusätzlich ``gics_sector_key`` / ``gics_industry_key``: normalisiert für Merges (z. B. GDELT).
``classification_standard`` kennzeichnet US vs. EU-Suffix grob (ICB_or_GICS_Yahoo vs. GICS_Yahoo).
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

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


def _infer_classification_standard(ticker: str) -> str:
    u = ticker.upper()
    if u.endswith(".L") or u.endswith(".PA") or u.endswith(".AS") or u.endswith(".BR"):
        return "ICB_or_GICS_Yahoo"
    if u.endswith(".DE") or u.endswith(".F") or u.endswith(".SW"):
        return "ICB_or_GICS_Yahoo"
    return "GICS_Yahoo"


def equity_classification_from_info(ticker: str, info: dict[str, Any] | None) -> dict[str, str]:
    inf = info or {}
    sector = str(inf.get("sector") or inf.get("sectorDisp") or "").strip()
    industry = str(inf.get("industry") or inf.get("industryDisp") or "").strip()
    return {
        "classification_standard": _infer_classification_standard(ticker),
        "gics_sector": sector,
        "gics_industry": industry,
        "gics_sector_key": normalize_taxonomy_label(sector),
        "gics_industry_key": normalize_taxonomy_label(industry),
    }


def get_equity_classification(ticker: str, info: dict[str, Any] | None = None) -> dict[str, str]:
    if info is not None:
        return equity_classification_from_info(ticker, info)
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).info or {}
    except Exception:
        raw = {}
    return equity_classification_from_info(ticker, raw)


def build_classification_cache(
    tickers: list[str],
    *,
    progress_every: int = 0,
) -> dict[str, dict[str, str]]:
    """``ticker`` → Klassifikations-Dict; optional Fortschritt auf stdout."""
    import yfinance as yf

    out: dict[str, dict[str, str]] = {}
    n = len(tickers)
    step = progress_every if progress_every > 0 else max(1, n // 10)
    t0 = time.perf_counter()
    for i, t in enumerate(tickers, 1):
        try:
            inf = yf.Ticker(t).info or {}
        except Exception:
            inf = {}
        out[t] = equity_classification_from_info(t, inf)
        if i == 1 or i % step == 0 or i == n:
            print(
                f"  … Branchenklassifikation (yfinance) {i}/{n} Ticker "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return out


def empty_classification_row() -> dict[str, str]:
    return {k: "" for k in CLASSIFICATION_COLUMN_KEYS}
