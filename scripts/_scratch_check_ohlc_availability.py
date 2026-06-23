"""Einmalige Diagnose: wie oft fehlen High/Low bei yfinance-Downloads?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from holdout.build_holdout_signals_master import _ohlc_field, _ticker_ohlc


def main() -> None:
    ho = ROOT / "data" / "holdout_signals.csv"
    if ho.exists():
        sig = pd.read_csv(ho)
        tickers = sorted(sig["ticker"].unique())
        src = "holdout_signals.csv"
    else:
        from lib.stock_rally_v10 import config as cfg

        tickers = list(cfg.ALL_TICKERS[:80])
        src = "ALL_TICKERS[:80]"

    start, end = "2015-01-01", "2026-07-18"
    print(f"Source: {src}, n={len(tickers)} tickers")

    raw_ho = yf.download(
        tickers, start=start, end=end, auto_adjust=True, threads=False, progress=False
    )
    if isinstance(raw_ho.columns, pd.MultiIndex):
        top = sorted(set(raw_ho.columns.get_level_values(0)))
    else:
        top = sorted(raw_ho.columns.tolist())
    print(f"Holdout-style raw top-level fields: {top}")

    no_oc: list[str] = []
    no_hl: list[str] = []
    ok_full: list[str] = []
    for t in tickers:
        o = _ohlc_field(raw_ho, "Open", t)
        c = _ohlc_field(raw_ho, "Close", t)
        h = _ohlc_field(raw_ho, "High", t)
        l = _ohlc_field(raw_ho, "Low", t)
        if o is None or c is None:
            no_oc.append(t)
        elif h is None or l is None:
            no_hl.append(t)
        else:
            ok_full.append(t)

    print(f"Holdout-style: yfinance liefert volles OHLC: {len(ok_full)}/{len(tickers)}")
    print(f"Holdout-style: ohne Open/Close: {len(no_oc)}")
    print(f"Holdout-style: ohne High/Low (aber O/C da): {len(no_hl)}")
    if no_hl:
        print(f"  Beispiele fehlend H/L: {no_hl[:10]}")
    if no_oc:
        print(f"  Beispiele fehlend O/C: {no_oc[:10]}")

    # Training-style
    bsz = 20
    train_ok: list[str] = []
    train_skip: list[str] = []
    for bi in range(0, len(tickers), bsz):
        batch = tickers[bi : bi + bsz]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            threads=False,
            progress=False,
            group_by="ticker",
        )
        for t in batch:
            cols = raw.columns
            ok = False
            if isinstance(cols, pd.MultiIndex):
                if (t, "Close") in cols and (t, "High") in cols and (t, "Low") in cols:
                    ok = True
                elif ("Close", t) in cols and ("High", t) in cols and ("Low", t) in cols:
                    ok = True
            else:
                ok = all(x in raw.columns for x in ["Open", "High", "Low", "Close"])
            if ok:
                train_ok.append(t)
            else:
                train_skip.append(t)

    print(
        f"Training-style (group_by=ticker): volles OHLC: {len(train_ok)}/{len(tickers)}"
    )
    print(f"Training-style: würde überspringen: {len(train_skip)}")
    if train_skip:
        print(f"  übersprungen: {train_skip[:10]}")

    # NaN-Anteil innerhalb erfolgreicher Serien (Training-Pfad)
    nan_high_rows = 0
    total_rows = 0
    for t in ok_full[: min(30, len(ok_full))]:
        ohlc = _ticker_ohlc(raw_ho, t)
        if ohlc is None:
            continue
        total_rows += len(ohlc)
        nan_high_rows += int(ohlc["High"].isna().sum())
    if total_rows:
        print(
            f"NaN in High (Stichprobe {min(30, len(ok_full))} Ticker): "
            f"{nan_high_rows}/{total_rows} Zeilen ({100*nan_high_rows/total_rows:.2f}%)"
        )


if __name__ == "__main__":
    main()
