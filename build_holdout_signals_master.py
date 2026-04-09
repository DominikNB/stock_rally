"""
Baut eine einzige CSV mit allen Holdout-Signal-Daten (keine Joins nötig):

  • Meta/Metadaten (ticker, Date, prob, threshold_used, company, sector) aus CSV oder DataFrame
  • Forward-Renditen (2/4/6/8/10 Handelstage, ret_mean_5) — gleiche Timing-Logik wie bisher
  • Trainings-Label-Eval (train_target, rally, eval_note) aus models/scoring_artifacts.joblib, falls vorhanden
  • Zusätzliche Filter (Liquidität, Cluster, Korrelation, Earnings-Fenster) — signal_extra_filters.py
    Standard: **an**; mit `--no-filters` abschaltbar (schneller, schlankere CSV).

Ausgabe: **data/meta_holdout_signals.csv** (eine Datei: Meta + Forward + Filter + optional Rally-Labels)

Eingabe CLI: data/holdout_signals.csv — oder Aufruf ``main(holdout_df=...)`` (Notebook nach Meta-Scoring).

python build_holdout_signals_master.py
python build_holdout_signals_master.py --no-filters
"""
from __future__ import annotations

import sys
from pathlib import Path

# Anreicherung standardmäßig aktiv (auch wenn main() von analyze_holdout_*.py importiert wird).
WITH_FILTERS = "--no-filters" not in sys.argv

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
HOLDOUT_CSV = ROOT / "data" / "holdout_signals.csv"
META_EXPORT_CSV = ROOT / "data" / "meta_holdout_signals.csv"
ARTIFACT = ROOT / "models" / "scoring_artifacts.joblib"
HORIZONS = (2, 4, 6, 8, 10)
YF_START = "2018-01-01"


def _ticker_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            o = raw["Open"][ticker]
            c = raw["Close"][ticker]
        else:
            o = raw["Open"]
            c = raw["Close"]
    except Exception:
        return None
    df = pd.DataFrame({"Open": o, "Close": c})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index().dropna(how="all")
    return df if len(df) else None


def next_trading_day_after(signal_d: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    s = pd.Timestamp(signal_d).normalize()
    after = dates[dates > s]
    if len(after) == 0:
        return None
    return pd.Timestamp(after[0]).normalize()


def create_target_one_ticker(close: np.ndarray, rw: int, rt: float, ld: int, ed: int, mt: int):
    """Gleiche Logik wie analyze_holdout_signal_quality / Notebook."""
    n = len(close)
    daily_ret = np.full(n, np.nan)
    daily_ret[1:] = close[1:] / close[:-1] - 1.0
    cum_ret = np.full(n, np.nan)
    window = int(rw)
    for i in range(window - 1, n):
        product = 1.0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(daily_ret[j]):
                product *= 1.0 + daily_ret[j]
        cum_ret[i] = product - 1.0
    rally = np.zeros(n, dtype=np.int8)
    for end_idx in range(n):
        if not np.isnan(cum_ret[end_idx]) and cum_ret[end_idx] >= rt:
            start_idx = max(0, end_idx - window + 1)
            rally[start_idx : end_idx + 1] = 1
    target = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        if rally[i] != 1:
            i += 1
            continue
        if i > 0 and rally[i - 1] == 1:
            i += 1
            continue
        start = i
        j = i
        while j < n and rally[j] == 1:
            j += 1
        end = j - 1
        pre_start = max(0, start - ld)
        if end - start + 1 >= mt:
            for k in range(pre_start, start):
                target[k] = 1
        for k in range(start, min(n, end + 1, start + ed)):
            if end - k + 1 >= mt:
                target[k] = 1
        i = j
    return rally, target


def _close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"][ticker]
    else:
        s = raw["Close"]
    return s.dropna()


_MIN_META_COLS = ("ticker", "Date", "prob", "threshold_used", "company", "sector")


def main(holdout_df: pd.DataFrame | None = None) -> None:
    """
    holdout_df: optional DataFrame mit Spalten ticker, Date, prob, threshold_used, company, sector
                (wie nach Meta-Classifier). Wenn None: zuerst data/holdout_signals.csv, sonst
                schmale Spalten aus data/meta_holdout_signals.csv (Rebuild ohne Notebook).
    """
    if holdout_df is None:
        if HOLDOUT_CSV.is_file():
            sig = pd.read_csv(HOLDOUT_CSV)
        elif META_EXPORT_CSV.is_file():
            sig = pd.read_csv(META_EXPORT_CSV)
            miss = [c for c in _MIN_META_COLS if c not in sig.columns]
            if miss:
                print(
                    f"Fehlt {META_EXPORT_CSV} Spalten {miss}; alternativ {HOLDOUT_CSV} anlegen.",
                    file=sys.stderr,
                )
                sys.exit(1)
            sig = sig[list(_MIN_META_COLS)].copy()
        else:
            print(
                f"Fehlt {HOLDOUT_CSV} oder {META_EXPORT_CSV}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        sig = holdout_df.copy()

    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["signal_date"] = sig["Date"]

    end_d = (sig["signal_date"].max() + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    tickers = sorted(sig["ticker"].unique())

    print(f"Download {len(tickers)} Ticker ({YF_START} … {end_d}) …")
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
        if df is not None and len(df) >= 20:
            dfs[t] = df

    ret_cols = [f"ret_{h}d" for h in HORIZONS]
    rows_fwd: list[dict] = []

    for _, r in sig.iterrows():
        t = r["ticker"]
        sig_d = r["signal_date"]
        df = dfs.get(t)
        base = {
            "ticker": t,
            "Date": str(sig_d.date()),
            "prob": r.get("prob", np.nan),
            "threshold_used": r.get("threshold_used", np.nan),
            "company": r.get("company", ""),
            "sector": r.get("sector", ""),
        }
        if df is None:
            row = {**base, "entry_date": "", "fwd_error": "keine Kursreihe"}
            for h in HORIZONS:
                row[f"ret_{h}d"] = np.nan
            row["ret_mean_5"] = np.nan
            rows_fwd.append(row)
            continue

        dates = df.index
        entry = next_trading_day_after(sig_d, dates)
        if entry is None:
            row = {**base, "entry_date": "", "fwd_error": "kein entry nach signal"}
            for h in HORIZONS:
                row[f"ret_{h}d"] = np.nan
            row["ret_mean_5"] = np.nan
            rows_fwd.append(row)
            continue

        if entry not in df.index:
            row = {
                **base,
                "entry_date": str(entry.date()),
                "fwd_error": "entry nicht im Index",
            }
            for h in HORIZONS:
                row[f"ret_{h}d"] = np.nan
            row["ret_mean_5"] = np.nan
            rows_fwd.append(row)
            continue

        o_ent = float(df.loc[entry, "Open"])
        if not np.isfinite(o_ent) or o_ent <= 0:
            row = {
                **base,
                "entry_date": str(entry.date()),
                "fwd_error": "open ungültig",
            }
            for h in HORIZONS:
                row[f"ret_{h}d"] = np.nan
            row["ret_mean_5"] = np.nan
            rows_fwd.append(row)
            continue

        pos = int(dates.get_loc(entry))
        row = {**base, "entry_date": str(entry.date()), "fwd_error": ""}
        for h in HORIZONS:
            j = pos + h
            if j >= len(df):
                row[f"ret_{h}d"] = np.nan
            else:
                c_ex = float(df.iloc[j]["Close"])
                if not np.isfinite(c_ex) or c_ex <= 0:
                    row[f"ret_{h}d"] = np.nan
                else:
                    row[f"ret_{h}d"] = c_ex / o_ent - 1.0

        vals = [row[c] for c in ret_cols]
        if all(np.isfinite(v) for v in vals):
            row["ret_mean_5"] = float(np.mean(vals))
        else:
            row["ret_mean_5"] = np.nan
        rows_fwd.append(row)

    out = pd.DataFrame(rows_fwd)

    # Trainings-Label (optional)
    out["train_target"] = np.nan
    out["rally"] = np.nan
    out["eval_note"] = ""

    if ARTIFACT.is_file():
        bp = joblib.load(ARTIFACT)["best_params"]
        rw = int(bp["return_window"])
        rt = float(bp["rally_threshold"])
        ld = int(bp["lead_days"])
        ed = int(bp["entry_days"])
        mt = int(bp["min_rally_tail_days"])
        print(
            f"Train-Label-Eval: return_window={rw}, rally_threshold={rt:.4f}, "
            f"lead_days={ld}, entry_days={ed}, min_rally_tail_days={mt}"
        )

        per_ticker: dict[str, tuple[dict, dict]] = {}
        for t in tickers:
            try:
                close = _close_series(raw, t)
            except Exception:
                continue
            if len(close) < 30:
                continue
            dates = pd.to_datetime(close.index).tz_localize(None).normalize()
            c = close.values.astype(np.float64)
            rally_arr, target_arr = create_target_one_ticker(c, rw, rt, ld, ed, mt)
            date_to_r = {dates[j]: int(rally_arr[j]) for j in range(len(dates))}
            date_to_t = {dates[j]: int(target_arr[j]) for j in range(len(dates))}
            per_ticker[t] = (date_to_t, date_to_r)

        sig_r = sig.reset_index(drop=True)
        for i in range(len(out)):
            t = out.at[i, "ticker"]
            d = pd.Timestamp(sig_r.at[i, "Date"]).normalize()
            if t not in per_ticker:
                out.at[i, "eval_note"] = "kein close"
                continue
            date_to_t, date_to_r = per_ticker[t]
            if d not in date_to_t:
                out.at[i, "eval_note"] = "kein Handelstag in Serie"
                continue
            out.at[i, "train_target"] = date_to_t[d]
            out.at[i, "rally"] = date_to_r[d]
    else:
        print(f"Hinweis: {ARTIFACT} fehlt — train_target/rally bleiben leer.")

    # Spaltenreihenfolge
    meta = ["ticker", "Date", "prob", "threshold_used", "company", "sector"]
    fwd_meta = ["entry_date", "fwd_error"] + ret_cols + ["ret_mean_5"]
    lab = ["train_target", "rally", "eval_note"]
    for c in meta + fwd_meta + lab:
        if c not in out.columns:
            out[c] = np.nan if c != "eval_note" else ""
    base_cols = meta + fwd_meta + lab
    out = out[base_cols]

    if WITH_FILTERS:
        try:
            from signal_extra_filters import enrich_signal_frame

            fil = enrich_signal_frame(sig, raw)
            for c in fil.columns:
                if c not in out.columns:
                    out[c] = fil[c].values
            tail = [c for c in out.columns if c not in base_cols]
            out = out[base_cols + tail]
            print(f"Zusätzliche Filter-Spalten: {', '.join(tail)}")
        except Exception as e:
            print(f"Warnung: Zusatzfilter fehlgeschlagen ({e}). Export ohne Filter-Spalten.", file=sys.stderr)

    META_EXPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(META_EXPORT_CSV, index=False)

    n_ok = int(out[ret_cols].notna().all(axis=1).sum())
    print(f"\nGeschrieben: {META_EXPORT_CSV}  ({len(out)} Zeilen, davon {n_ok} mit allen Forward-Renditen)")
    for h in HORIZONS:
        col = f"ret_{h}d"
        s = pd.to_numeric(out[col], errors="coerce")
        nn = int(s.notna().sum())
        if nn == 0:
            print(f"  H={h}d: keine Werte")
            continue
        print(
            f"  H={h}d: n={nn}  Mittel={s.mean():+.2%}  Median={s.median():+.2%}  "
            f"Anteil>0={(s > 0).sum() / nn:.1%}"
        )


if __name__ == "__main__":
    main()
