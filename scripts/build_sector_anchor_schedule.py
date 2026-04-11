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

    if args.quarter_end:
        qd = date.fromisoformat(args.quarter_end[:10])
    elif args.date:
        qd = date.fromisoformat(args.date[:10])
    else:
        qd = date.today()

    q0, q1 = quarter_bounds(qd)
    print(f"Quartal {q0} … (exkl.) {q1} — Top {args.top_n} pro Sektor", flush=True)

    def fetch_cap(sym: str):
        time.sleep(max(0.0, float(args.sleep)))
        return _fetch_cap(sym)

    rows = build_quarter_snapshot_rows(
        cfg.TICKERS_BY_SECTOR,
        cfg.COMPANY_NAMES,
        quarter_end=qd,
        top_n=args.top_n,
        fetch_cap=fetch_cap,
    )
    out_path = Path(args.out)
    upsert_schedule_rows(out_path, rows)
    print(f"Geschrieben: {out_path}  ({len(rows)} Sektoren)", flush=True)
    for r in rows:
        print(f"  {r['sector']}: {r['tickers']}", flush=True)


if __name__ == "__main__":
    main()
