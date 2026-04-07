#!/usr/bin/env python3
"""
Forward returns for signals in docs/signals.json (stdlib + Yahoo chart API v8).
10 Handelstage nach Signaltag: Kurs vs. Benchmark (^GSPC).

Standard ist --source holdout: nur ``signals_holdout_final`` (FINAL-Zeitfenster, OOS).
Die volle Liste ``signals`` enthält In-Sample-Termine und ist für Performance-Analyse verzerrt.

Usage:
  python scripts/analyze_signals_forward.py [--json path] [--source holdout|all] [--bench ^GSPC]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

USER_AGENT = "Mozilla/5.0 (compatible; stock_rally/1.0; +local script)"


def _epoch_utc(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _calendar_buffer_days(trading_days: int) -> int:
    """Grober Kalenderpuffer für genug 1d-Bars."""
    return max(28, trading_days * 2 + 10)


def yahoo_chart(symbol: str, period1: int, period2: int) -> dict:
    sym = urllib.parse.quote(symbol, safe="")
    q = urllib.parse.urlencode(
        {"period1": period1, "period2": period2, "interval": "1d"}
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


def _closes_and_dates(payload: dict) -> tuple[list[date], list[float]]:
    r = payload["chart"]["result"][0]
    ts = r["timestamp"]
    q = r["indicators"]["quote"][0]
    closes = q.get("close") or []
    out_d: list[date] = []
    out_c: list[float] = []
    for t, c in zip(ts, closes):
        if c is None or c != c:
            continue
        out_d.append(datetime.fromtimestamp(t, tz=timezone.utc).date())
        out_c.append(float(c))
    return out_d, out_c


def _idx_on_or_after(dates: list[date], d0: date) -> int:
    for i, d in enumerate(dates):
        if d >= d0:
            return i
    return -1


def forward_return_trading_days(
    symbol: str, signal_day: date, horizon: int
) -> float | None:
    """Einfaches Rückgabeformat: (close[T+h] / close[T] - 1), T = erster Handelstag >= signal_day."""
    p1 = _epoch_utc(signal_day - timedelta(days=5))
    p2 = _epoch_utc(signal_day + timedelta(days=_calendar_buffer_days(horizon)))
    try:
        data = yahoo_chart(symbol, p1, p2)
    except Exception:
        return None
    dates, closes = _closes_and_dates(data)
    if len(dates) < 2:
        return None
    i0 = _idx_on_or_after(dates, signal_day)
    if i0 < 0:
        return None
    i1 = i0 + horizon
    if i1 >= len(closes):
        return None
    c0, c1 = closes[i0], closes[i1]
    if c0 <= 0:
        return None
    return c1 / c0 - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default="docs/signals.json",
        help="Pfad zu signals.json",
    )
    ap.add_argument(
        "--source",
        choices=("holdout", "all"),
        default="holdout",
        help="holdout = signals_holdout_final (FINAL, OOS); all = volle Historie (inkl. In-Sample)",
    )
    ap.add_argument("--bench", default="^GSPC", help="Yahoo-Symbol Benchmark")
    ap.add_argument("--horizon", type=int, default=10, help="Handelstage nach Einstieg")
    args = ap.parse_args()

    try:
        with open(args.json, encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        print(e, file=sys.stderr)
        return 1

    thr = doc.get("threshold")
    if args.source == "holdout":
        sigs = doc.get("signals_holdout_final")
        if sigs is None:
            print(
                "Fehlender Schlüssel signals_holdout_final — Notebook Cell 17 neu ausführen.",
                file=sys.stderr,
            )
            return 2
        label = "signals_holdout_final (FINAL OOS)"
        if len(sigs) == 0:
            print(
                "Keine Signale im FINAL-Holdout (signals_holdout_final leer). "
                "Schwellenwert/Filter lassen im letzten Zeitfenster keine Treffer zu — "
                "keine unbiased Auswertung möglich. Mit --source all die volle Liste prüfen (nicht OOS).",
                file=sys.stderr,
            )
            return 3
    else:
        sigs = doc.get("signals") or []
        label = "signals (volle Historie, inkl. In-Sample — nur QA, keine OOS-Performance)"
        print(
            "Hinweis: --source all nutzt In-Sample-Termine; für Modell-Performance nur --source holdout.",
            file=sys.stderr,
        )

    print(f"Datei: {args.json}  |  Quelle: {label}")
    print(f"Signale: {len(sigs)}  |  threshold: {thr}")
    print(f"Horizont: {args.horizon} Handelstage  |  Benchmark: {args.bench}\n")

    stock_rets: list[float] = []
    bench_rets: list[float] = []
    rel: list[float] = []
    skipped: list[str] = []

    for s in sigs:
        t = s.get("ticker")
        ds = s.get("date")
        if not t or not ds:
            skipped.append(f"{t} {ds} (fehlt)")
            continue
        d0 = datetime.strptime(ds, "%Y-%m-%d").date()
        rs = forward_return_trading_days(t, d0, args.horizon)
        rb = forward_return_trading_days(args.bench, d0, args.horizon)
        if rs is None:
            skipped.append(f"{t} {ds} (Kurs)")
            continue
        stock_rets.append(rs)
        if rb is not None:
            bench_rets.append(rb)
            rel.append(rs - rb)
        else:
            skipped.append(f"{args.bench} {ds} (Benchmark)")

    n = len(stock_rets)
    if n == 0:
        print("Keine vollständigen Returns.")
        return 0

    def pct(x: float) -> str:
        return f"{100.0 * x:+.2f}%"

    pos = sum(1 for x in stock_rets if x > 0)
    print(f"Aktie: n={n}  positiv: {pos} ({100 * pos / n:.1f}%)")
    print(f"  Mittel: {pct(statistics.mean(stock_rets))}  Median: {pct(statistics.median(stock_rets))}")
    if bench_rets and len(bench_rets) == n:
        print(
            f"Benchmark ({args.bench}): Mittel {pct(statistics.mean(bench_rets))}  "
            f"Median {pct(statistics.median(bench_rets))}"
        )
        print(
            f"Relativ (Aktie − Bench): Mittel {pct(statistics.mean(rel))}  "
            f"Median {pct(statistics.median(rel))}"
        )
    if skipped:
        print(f"\nAusgelassen / unvollständig: {len(skipped)}")
        for line in skipped[:15]:
            print(f"  - {line}")
        if len(skipped) > 15:
            print("  ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
