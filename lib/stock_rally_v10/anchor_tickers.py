"""
Rollierende Sektor-Anker (Top-N nach Marktkapitalisierung) für News-Spillover.

Zeitplan als JSON (siehe ``scripts/build_sector_anchor_schedule.py``): pro Halbjahr/Quartal
welche Ticker den Sektor in GDELT ``V2Organizations`` repräsentieren. Die Pipeline
verknüpft das weiterhin mit allen Aktien des Sektors in ``features.py`` (bereits
sektoral gemergt); hier geht es um die BigQuery-Filterung (weniger Rauschen, weniger OR-Tokens).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, MutableSequence

SCHEDULE_VERSION = 1


def _like_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("'", "''")


def load_anchor_schedule(path: str | Path) -> list[dict[str, Any]] | None:
    """Lädt ``anchors`` aus JSON; fehlt die Datei → ``None``."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rows = obj.get("anchors")
    if not isinstance(rows, list):
        return None
    return [r for r in rows if isinstance(r, dict)]


def tickers_for_sector_on_date(
    schedule: list[dict[str, Any]],
    sector: str,
    on: date,
) -> tuple[str, ...]:
    """Ticker für ``sector`` am Kalendertag ``on`` (letzte passende Periode gewinnt)."""
    best: tuple[date, tuple[str, ...]] | None = None
    for row in schedule:
        if str(row.get("sector") or "") != sector:
            continue
        ps = row.get("period_start")
        pe = row.get("period_end_exclusive")
        if not ps or not pe:
            continue
        try:
            d0 = datetime.fromisoformat(str(ps)[:10]).date()
            d1 = datetime.fromisoformat(str(pe)[:10]).date()
        except ValueError:
            continue
        if not (d0 <= on < d1):
            continue
        raw = row.get("tickers") or []
        if not isinstance(raw, list):
            continue
        tks = tuple(str(t).strip() for t in raw if str(t).strip())
        if not tks:
            continue
        if best is None or d0 >= best[0]:
            best = (d0, tks)
    return best[1] if best else tuple()


def _search_tokens_for_ticker(
    ticker: str,
    company_names: Mapping[str, str],
) -> list[str]:
    t = str(ticker).strip()
    if not t:
        return []
    out: list[str] = []
    if t in company_names:
        out.append(str(company_names[t]).strip())
    base = t.split(".")[0].split("-")[0].upper()
    if base and base not in (x.upper() for x in out):
        out.append(base)
    return [x for x in out if x]


def v2organizations_or_clause(
    tickers: tuple[str, ...] | list[str],
    company_names: Mapping[str, str],
) -> str:
    """``(V2Organizations LIKE '%%…%%' OR …)`` — leer wenn keine sinnvollen Tokens."""
    seen: set[str] = set()
    parts: list[str] = []
    for tk in tickers:
        for tok in _search_tokens_for_ticker(tk, company_names):
            key = tok.casefold()
            if key in seen:
                continue
            seen.add(key)
            esc = _like_escape(tok)
            parts.append(f"V2Organizations LIKE '%{esc}%'")
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def upsert_schedule_rows(
    path: str | Path,
    new_rows: list[dict[str, Any]],
) -> None:
    """Merge ``new_rows`` nach (period_start, sector); schreibt JSON."""
    p = Path(path)
    prev = load_anchor_schedule(p) or []

    def _row_key(r: dict[str, Any]) -> tuple[str, str]:
        return (str(r.get("period_start")), str(r.get("sector")))

    by_k = {_row_key(r): r for r in prev}
    for r in new_rows:
        by_k[_row_key(r)] = r
    merged = sorted(by_k.values(), key=_row_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": SCHEDULE_VERSION, "anchors": merged}
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)


def quarter_bounds(d: date) -> tuple[date, date]:
    """Kalenderquartal: ``[start, end_exclusive)``."""
    q = (d.month - 1) // 3
    start_month = q * 3 + 1
    start = date(d.year, start_month, 1)
    end_month = start_month + 3
    if end_month > 12:
        end = date(d.year + 1, end_month - 12, 1)
    else:
        end = date(d.year, end_month, 1)
    return start, end


def build_quarter_snapshot_rows(
    tickers_by_sector: Mapping[str, MutableSequence[str]],
    company_names: Mapping[str, str],
    *,
    quarter_end: date,
    top_n: int,
    fetch_cap: Any,
) -> list[dict[str, Any]]:
    """
    Eine Zeile pro Sektor für das Quartal von ``quarter_end`` (via ``fetch_cap`` = yfinance).
    ``fetch_cap(sym)`` → Marktkapitalisierung (float) oder None.
    """
    q0, q1 = quarter_bounds(quarter_end)
    rows: list[dict[str, Any]] = []
    for sector, tickers in tickers_by_sector.items():
        scored: list[tuple[float, str]] = []
        for sym in tickers:
            s = str(sym).strip()
            if not s or s.upper().endswith("-USD"):
                continue
            cap = fetch_cap(s)
            if cap is not None and cap > 0:
                scored.append((float(cap), s))
        scored.sort(key=lambda x: -x[0])
        top = [t for _, t in scored[: max(1, int(top_n))]]
        if not top and tickers:
            top = [str(tickers[0])]
        rows.append(
            {
                "period_start": q0.isoformat(),
                "period_end_exclusive": q1.isoformat(),
                "sector": sector,
                "tickers": top,
                "note": "market_cap_snapshot",
            }
        )
    return rows
