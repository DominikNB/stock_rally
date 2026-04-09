"""
Reichert Holdout-Signale mit zusätzlichen Filtern an (signal_extra_filters).

  python enrich_holdout_filters.py
  python enrich_holdout_filters.py --input data/holdout_signals.csv --output data/holdout_signals_enriched.csv
  python enrich_holdout_filters.py --merge-master   # merged in data/meta_holdout_signals.csv

Benötigt einen vorherigen Download derselben Ticker wie build_holdout_signals_master (yfinance).
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings(
    "ignore", message=".*fill_method.*pct_change.*", category=FutureWarning
)
logging.getLogger("yfinance").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "data" / "holdout_signals.csv"
DEFAULT_OUT = ROOT / "data" / "holdout_signals_enriched.csv"
MASTER = ROOT / "data" / "meta_holdout_signals.csv"
YF_START = "2018-01-01"


def main() -> None:
    ap = argparse.ArgumentParser(description="Holdout-Signale um Filter-Spalten anreichern")
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_IN,
        help="CSV mit mind. ticker, Date, prob, threshold_used, sector",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Ausgabe-CSV (alle Originalspalten + Filter)",
    )
    ap.add_argument(
        "--merge-master",
        action="store_true",
        help=f"Nach Anreicherung mit {MASTER.name} per ticker+Date mergen und Master überschreiben",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Fehlt: {args.input}", file=sys.stderr)
        sys.exit(1)

    from signal_extra_filters import enrich_signal_frame

    sig = pd.read_csv(args.input)
    for col in ("ticker", "Date", "prob", "threshold_used", "sector"):
        if col not in sig.columns:
            print(f"Spalte fehlt: {col}", file=sys.stderr)
            sys.exit(1)
    if "company" not in sig.columns:
        sig["company"] = ""

    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    end_d = (sig["Date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    tickers = sorted(sig["ticker"].unique())

    print(f"yfinance: {len(tickers)} Ticker ({YF_START} … {end_d}) …")
    raw = yf.download(
        list(tickers),
        start=YF_START,
        end=end_d,
        auto_adjust=True,
        threads=False,
        progress=False,
    )

    out = enrich_signal_frame(sig, raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Geschrieben: {args.output}  ({len(out)} Zeilen)")

    extra = [
        "meta_prob_margin",
        "signals_same_day",
        "signals_same_sector_same_day",
        "sector_share_same_day",
        "cluster_mean_corr_60d",
        "adv_20d_local",
        "adv_pctile_same_day",
        "liquidity_tier",
        "next_earnings_date",
        "bdays_to_next_earnings",
        "earnings_in_3_15_bday_window",
    ]
    print("Neue Spalten:", ", ".join(extra))

    if args.merge_master:
        if not MASTER.is_file():
            print(f"Hinweis: {MASTER} fehlt — --merge-master übersprungen.", file=sys.stderr)
            return
        m = pd.read_csv(MASTER)
        m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        key = ["ticker", "Date"]
        enrich_cols = [c for c in extra if c in out.columns]
        slim = out[key + enrich_cols].drop_duplicates(subset=key)
        merged = m.merge(slim, on=key, how="left")
        merged.to_csv(MASTER, index=False)
        print(f"Gemerged: {MASTER}")


if __name__ == "__main__":
    main()
