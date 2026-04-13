"""
Vergleich Holdout-Signale: Rendite vom Einstiegs-Open bis zum letzten Schlusskurs
≤ (heute − ``ASOF_CALENDAR_LAG_DAYS``) — nicht die festen Horizonte ``ret_*d`` in der CSV.

Voraussetzung: ``data/master_complete.csv`` (wie von ``build_holdout_signals_master``).

  python -m holdout.analyze_holdout_return_to_asof
  python -m holdout.analyze_holdout_return_to_asof --lag-days 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "master_complete.csv"
YF_START = "2018-01-01"


def next_trading_day_after(signal_d: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    s = pd.Timestamp(signal_d).normalize()
    after = dates[dates > s]
    if len(after) == 0:
        return None
    return pd.Timestamp(after[0]).normalize()


def _ticker_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            o = raw["Open"][ticker]
            c = raw["Close"][ticker]
        else:
            # Einzel-Ticker-Download: keine zweite Spalten-Ebene
            o = raw["Open"]
            c = raw["Close"]
    except Exception:
        return None
    df = pd.DataFrame({"Open": o, "Close": c})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index().dropna(how="all")
    return df if len(df) else None


def ret_to_asof(
    sig_d: pd.Timestamp,
    entry_date_str: str,
    ohlc: pd.DataFrame,
    asof: pd.Timestamp,
) -> tuple[float | None, str]:
    dates = ohlc.index
    entry: pd.Timestamp | None = None
    if entry_date_str and str(entry_date_str).strip() not in ("", "nan", "NaT"):
        entry = pd.Timestamp(entry_date_str).normalize()
        if entry not in ohlc.index:
            entry = None
    if entry is None:
        entry = next_trading_day_after(sig_d, dates)
    if entry is None:
        return None, "kein_entry"
    if entry not in ohlc.index:
        return None, "entry_nicht_im_index"
    o_ent = float(ohlc.loc[entry, "Open"])
    if not np.isfinite(o_ent) or o_ent <= 0:
        return None, "open_ungueltig"
    eligible = ohlc.index[ohlc.index <= asof]
    if len(eligible) == 0:
        return None, "kein_kurs_bis_asof"
    last_d = eligible[-1]
    if last_d <= entry:
        return None, "asof_vor_oder_gleich_entry"
    c_last = float(ohlc.loc[last_d, "Close"])
    if not np.isfinite(c_last) or c_last <= 0:
        return None, "close_ungueltig"
    return c_last / o_ent - 1.0, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Holdout: Rendite bis As-of vs. heute-minus-lag vergleichen.")
    ap.add_argument(
        "--lag-days",
        type=int,
        default=3,
        help="Letzter Kurs: höchstens dieser Kalenderabstand vor heute (Standard: 3).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=MASTER,
        help="Pfad zu master_complete.csv",
    )
    args = ap.parse_args()
    lag = max(0, int(args.lag_days))
    path = Path(args.csv)
    if not path.is_file():
        print(f"Datei fehlt: {path}", file=sys.stderr)
        sys.exit(1)

    mc = pd.read_csv(path)
    need = {"ticker", "Date"}
    miss = need - set(mc.columns)
    if miss:
        print(f"CSV fehlt Spalten: {miss}", file=sys.stderr)
        sys.exit(1)

    mc = mc.copy()
    mc["Date"] = pd.to_datetime(mc["Date"]).dt.normalize()
    asof = pd.Timestamp.now().normalize() - pd.Timedelta(days=lag)
    tickers = sorted(mc["ticker"].astype(str).unique())
    end_d = (max(mc["Date"].max(), asof) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(
        f"Analyse: {path}  ({len(mc)} Zeilen)\n"
        f" As-of (letzter Schlusskurs): letzter Handelstag ≤ {asof.date()} "
        f"(heute − {lag} Kalendertage)\n"
        f"  Yahoo-Download {len(tickers)} Ticker …",
        flush=True,
    )
    raw = yf.download(
        list(tickers),
        start=YF_START,
        end=end_d,
        auto_adjust=True,
        threads=False,
        progress=False,
    )

    dfs: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _ticker_ohlc(raw, t)
        if df is not None and len(df) >= 5:
            dfs[t] = df

    rets: list[float | None] = []
    errs: list[str] = []
    for _, r in mc.iterrows():
        t = str(r["ticker"])
        sig_d = pd.Timestamp(r["Date"]).normalize()
        ed = r["entry_date"] if "entry_date" in mc.columns else ""
        if t not in dfs:
            rets.append(None)
            errs.append("keine_ohlc")
            continue
        rv, err = ret_to_asof(sig_d, str(ed) if pd.notna(ed) else "", dfs[t], asof)
        rets.append(rv)
        errs.append(err if rv is None else "")

    mc["ret_to_asof"] = rets
    mc["_err"] = errs
    usable = mc[mc["ret_to_asof"].notna()].copy()
    bad = mc[mc["ret_to_asof"].isna()]
    print(
        f"  Auswertbar: {len(usable)} / {len(mc)} "
        f"(ausgeschlossen: {len(bad)} — fwd_error/As-of/Entry siehe Spalte _err)\n",
        flush=True,
    )
    if usable.empty:
        print("Keine Zeile mit berechneter ret_to_asof — Abbruch.", file=sys.stderr)
        sys.exit(2)

    usable["win"] = (usable["ret_to_asof"] > 0).astype(int)
    pos = usable[usable["win"] == 1]
    neg = usable[usable["win"] == 0]

    def _block(name: str, sub: pd.DataFrame) -> None:
        print(f"── {name} (n={len(sub)}) ──")
        if sub.empty:
            print("  (leer)\n")
            return
        r = sub["ret_to_asof"]
        print(
            f"  ret_to_asof: Mittel={r.mean():+.2%}  Median={r.median():+.2%}  "
            f"Min={r.min():+.2%}  Max={r.max():+.2%}"
        )
        if "prob" in sub.columns:
            p = pd.to_numeric(sub["prob"], errors="coerce").dropna()
            if len(p):
                print(
                    f"  prob:        Mittel={p.mean():.4f}  Median={p.median():.4f}  "
                    f"Std={p.std():.4f}"
                )
        if "ret_10d" in sub.columns:
            h = pd.to_numeric(sub["ret_10d"], errors="coerce").dropna()
            if len(h):
                print(
                    f"  ret_10d(CSV): Mittel={h.mean():+.2%}  Median={h.median():+.2%} "
                    f"(fester Horizont, nicht As-of)"
                )
        print()

    _block("Positive Rendite bis As-of (>0)", pos)
    _block("Negative oder Null bis As-of (≤ 0)", neg)

    if "sector" in usable.columns and len(usable):
        print("── Anteil positiver ret_to_asof nach Sektor (n≥3) ──")
        g = usable.groupby("sector", dropna=False).agg(
            n=("ret_to_asof", "count"),
            anteil_pos=("win", "mean"),
        )
        g = g[g["n"] >= 3].sort_values("anteil_pos", ascending=True)
        print(g.to_string(), end="\n\n")

    if "prob" in usable.columns:
        a = pd.to_numeric(pos["prob"], errors="coerce").dropna()
        b = pd.to_numeric(neg["prob"], errors="coerce").dropna()
        if len(a) >= 5 and len(b) >= 5:
            from statistics import mean, stdev

            # Welch-ähnlich: einfacher Mittelwertvergleich + gemeinsame Übersicht
            print(
                f"── prob (grobe Trennung) ──\n"
                f"  positiv: n={len(a)}  Mittel={mean(a):.4f}\n"
                f"  negativ: n={len(b)}  Mittel={mean(b):.4f}\n"
            )


if __name__ == "__main__":
    main()
