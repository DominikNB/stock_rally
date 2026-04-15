"""
Quartals-Anker pro Sektor (Top-N nach Yahoo-``marketCap``) → JSON für cfg.NEWS_ANCHOR_SCHEDULE_PATH.

Hinweis: ``marketCap`` ist ein Snapshot zum Abrufzeitpunkt — echte historische
Marktkapitalisierung pro Quartalsende braucht externe Fundamentals (nicht yfinance).

Beispiel:
  python scripts/build_sector_anchor_schedule.py --quarter-end 2024-12-31
  python scripts/build_sector_anchor_schedule.py --date 2025-04-04
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10.anchor_tickers import (
    build_quarter_snapshot_rows,
    quarter_bounds,
    upsert_schedule_rows,
)
from lib.stock_rally_v10 import config as cfg


def _fetch_cap(sym: str):
    try:
        import yfinance as yf
    except ImportError:
        print("Bitte installieren: pip install yfinance", file=sys.stderr)
        raise SystemExit(2) from None
    try:
        t = yf.Ticker(sym)
        cap = t.fast_info.get("market_cap")
        if cap is None or cap <= 0:
            info = t.info or {}
            cap = info.get("marketCap")
        return float(cap) if cap else None
    except Exception:
        return None


def _quarter_start(d: date) -> date:
    q = (d.month - 1) // 3
    m0 = q * 3 + 1
    return date(d.year, m0, 1)


def _next_quarter(d: date) -> date:
    m = d.month + 3
    y = d.year
    if m > 12:
        y += 1
        m -= 12
    return date(y, m, 1)


def _quarter_points(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        return []
    cur = _quarter_start(start_date)
    out: list[date] = []
    while cur <= end_date:
        out.append(cur)
        cur = _next_quarter(cur)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sektor-Anker-Zeitplan (Market-Cap Top-N) erzeugen.")
    p.add_argument(
        "--quarter-end",
        type=str,
        default=None,
        help="Datum im Zielquartal (YYYY-MM-DD), z. B. 2024-03-31",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Beliebiges Datum; Quartal wird abgeleitet (Alternative zu --quarter-end)",
    )
    p.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Startdatum für alle Quartale (YYYY-MM-DD).",
    )
    p.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="Enddatum für alle Quartale (YYYY-MM-DD). Default: heute.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(cfg.NEWS_ANCHOR_SCHEDULE_PATH),
        help="Ausgabe-JSON",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=int(cfg.NEWS_ANCHOR_TOP_N),
        help="Anker pro Sektor",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Sekunden zwischen yfinance-Calls (Rate-Limit)",
    )
    args = p.parse_args()

    if args.from_date:
        d0 = date.fromisoformat(args.from_date[:10])
        d1 = date.fromisoformat(args.to_date[:10]) if args.to_date else date.today()
        q_dates = _quarter_points(d0, d1)
        if not q_dates:
            raise SystemExit(f"Keine Quartale im Bereich {d0}..{d1}")
        print(
            f"Quartals-Range {q_dates[0]} .. {q_dates[-1]} "
            f"({len(q_dates)} Quartale) — Top {args.top_n} pro Sektor",
            flush=True,
        )
    else:
        if args.quarter_end:
            qd = date.fromisoformat(args.quarter_end[:10])
        elif args.date:
            qd = date.fromisoformat(args.date[:10])
        else:
            qd = date.today()
        q_dates = [qd]
        q0, q1 = quarter_bounds(qd)
        print(f"Quartal {q0} … (exkl.) {q1} — Top {args.top_n} pro Sektor", flush=True)

    cap_cache: dict[str, float | None] = {}

    def fetch_cap(sym: str):
        s = str(sym).strip()
        if s in cap_cache:
            return cap_cache[s]
        time.sleep(max(0.0, float(args.sleep)))
        cap = _fetch_cap(s)
        cap_cache[s] = cap
        return cap

    rows = []
    n_q = len(q_dates)
    for i, qd in enumerate(q_dates, start=1):
        q0, q1 = quarter_bounds(qd)
        print(f"[{i}/{n_q}] Quartal {q0} … (exkl.) {q1}", flush=True)
        rows.extend(
            build_quarter_snapshot_rows(
                cfg.TICKERS_BY_SECTOR,
                cfg.COMPANY_NAMES,
                quarter_end=qd,
                top_n=args.top_n,
                fetch_cap=fetch_cap,
            )
        )
    out_path = Path(args.out)
    upsert_schedule_rows(out_path, rows)
    print(f"Geschrieben: {out_path}  ({len(rows)} Zeilen)", flush=True)
    for r in rows[-min(len(rows), 20):]:
        print(f"  {r['sector']}: {r['tickers']}", flush=True)


if __name__ == "__main__":
    main()
