"""
Baut eine einzige CSV mit allen Holdout-Signal-Daten (keine Joins nötig):

  • Meta/Metadaten inkl. Yahoo sector/industry + Merge-Keys (equity_classification.CLASSIFICATION_COLUMN_KEYS)
  • Forward-Renditen (2/4/6/8/10 Handelstage, ret_mean_5) — gleiche Timing-Logik wie bisher
  • Trainings-Label-Eval (train_target, rally, eval_note) aus models/scoring_artifacts.joblib, falls vorhanden
  • Zusätzliche Filter (Liquidität, Cluster, Sektor-HHI, Korrelation, Cross-Section-Ranks,
    OHLCV-Technik, Earnings-Fenster) — signal_extra_filters.py
  • **News-Töne** aus ``data/feature_shards_news`` (Sektor/Makro für Rot-Chip) — immer,
    unabhängig von ``--no-filters`` (nur Zusatzfilter lassen sich abschalten).

Ausgabe:
  • **data/master_complete.csv** — volle Historie: Meta + Forward-Renditen + Trainingslabels + Zusatzfilter
  • **data/master_daily_update.csv** — nur **letzter Signaltag**; schlanke Spalten für LLM/Website (keine Forward-/Labels)

Eingabe CLI: data/holdout_signals.csv — oder Aufruf ``main(holdout_df=...)`` (Notebook nach Meta-Scoring).

**Classifier unberührt:** ``enrich_signal_frame`` (Zusatzspalten) läuft **nach** dem Meta-Scoring
und schreibt nur die Export-CSV; es lädt **kein** Classifier-Training neu und ändert keine
``build_feature_cols``-Logik in ``config.py``. Zusatzspalten dienen Analyse und LLM.

python -m holdout.build_holdout_signals_master
python -m holdout.build_holdout_signals_master --no-filters
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Anreicherung standardmäßig aktiv (auch wenn main() von analyze_holdout_*.py importiert wird).
WITH_FILTERS = "--no-filters" not in sys.argv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import json
import numpy as np
import pandas as pd
import yfinance as yf

from lib.stock_rally_v10.equity_classification import CLASSIFICATION_COLUMN_KEYS as _CLASSIFICATION_META_COLS

HOLDOUT_CSV = ROOT / "data" / "holdout_signals.csv"
MASTER_COMPLETE_CSV = ROOT / "data" / "master_complete.csv"
MASTER_DAILY_CSV = ROOT / "data" / "master_daily_update.csv"
ARTIFACT = ROOT / "models" / "scoring_artifacts.joblib"
NEWS_SHARD_DIR = ROOT / "data" / "feature_shards_news"
HORIZONS = (2, 4, 6, 8, 10)
try:
    from lib.stock_rally_v10 import config as _sr_cfg_for_yf

    _sd = getattr(_sr_cfg_for_yf, "START_DATE", "2015-01-01")
    YF_START = _sd.strftime("%Y-%m-%d") if hasattr(_sd, "strftime") else str(_sd)[:10]
except Exception:
    YF_START = "2015-01-01"


def _progress_step(n: int, parts: int = 10) -> int:
    """Schrittweite für Logs: etwa ``parts`` Meldungen über ``n`` Elemente."""
    if n <= 0:
        return 1
    return max(1, n // parts)


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


def _fixed_y_labels_per_ticker(
    raw: pd.DataFrame, tickers: list[str]
) -> dict[str, tuple[dict[pd.Timestamp, int], dict[pd.Timestamp, int]]]:
    """Trainings-Labels wie Pipeline: ``cfg.fixed_y_label_mode()`` (z. B. rally_plus_entry)."""
    from lib.stock_rally_v10 import config as _cfg
    from lib.stock_rally_v10.target import _create_target_one_ticker_fixed_bands

    per_ticker: dict[str, tuple[dict[pd.Timestamp, int], dict[pd.Timestamp, int]]] = {}
    mode = _cfg.fixed_y_label_mode()
    for t in tickers:
        ohlc = _ticker_ohlc(raw, t)
        if ohlc is None or len(ohlc) < 30:
            continue
        sub = pd.DataFrame(
            {
                "close": ohlc["Close"].astype(np.float64).values,
                "open": ohlc["Open"].astype(np.float64).values,
            },
            index=pd.to_datetime(ohlc.index).tz_localize(None).normalize(),
        )
        rally_arr, target_arr = _create_target_one_ticker_fixed_bands(sub)
        dates = sub.index
        per_ticker[t] = (
            {dates[j]: int(target_arr[j]) for j in range(len(dates))},
            {dates[j]: int(rally_arr[j]) for j in range(len(dates))},
        )
    if per_ticker:
        print(
            f"Train-Label-Eval: feste Band-Regel ({mode}) — {len(per_ticker)} Ticker-Serien",
            flush=True,
        )
    return per_ticker


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


def _columns_to_drop_for_daily(ret_cols: list[str]) -> list[str]:
    """Forward-Renditen, Entry/Fehler, Trainingslabels — für tagesaktuelle Exporte ohne Blick nach vorne."""
    return (
        ["entry_date", "fwd_error"]
        + ret_cols
        + ["ret_mean_5", "train_target", "rally", "eval_note"]
    )


def _build_daily_update(full: pd.DataFrame, ret_cols: list[str]) -> pd.DataFrame:
    """Nur Zeilen mit maximalem Signaltag; ohne Forward-/Label-Spalten."""
    dts = pd.to_datetime(full["Date"])
    latest = dts.max()
    sub = full.loc[dts == latest].copy()
    drop = [c for c in _columns_to_drop_for_daily(ret_cols) if c in sub.columns]
    return sub.drop(columns=drop, errors="ignore")


def _merge_news_shard_for_holdout(sig: pd.DataFrame) -> pd.DataFrame:
    """
    Sektor-/Makro-News-Töne aus dem News-Shard (wie Training/Scoring) — für CSV/LLM-Export.
    """
    manifest_path = NEWS_SHARD_DIR / "news_shards_manifest.json"
    if not manifest_path.is_file():
        print(
            "Hinweis: News-Shard fehlt (data/feature_shards_news) — News-Chip bleibt grau.",
            flush=True,
        )
        return sig
    if not ARTIFACT.is_file():
        print(
            "Hinweis: models/scoring_artifacts.joblib fehlt — News-Merge übersprungen.",
            flush=True,
        )
        return sig

    try:
        from lib.stock_rally_v10 import config as cfg
        from lib.stock_rally_v10.features import merge_news_shard_from_best_params

        artifact = joblib.load(ARTIFACT)
        best_params = artifact.get("best_params") or {}
        for req in ("news_mom_w", "news_vol_ma", "news_tone_roll"):
            if req not in best_params:
                print(
                    f"Hinweis: best_params ohne {req} — News-Merge übersprungen.",
                    flush=True,
                )
                return sig

        cfg._FEATURE_NEWS_SHARDS_ACTIVE = True
        cfg.FEATURE_SHARD_DIR = str(NEWS_SHARD_DIR.resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg.NEWS_SHARD_MANIFEST = {
            str(k): str(NEWS_SHARD_DIR / v) for k, v in (manifest.get("tags") or {}).items()
        }

        frame = sig.copy()
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
        if "sector" not in frame.columns:
            frame["sector"] = ""
        frame["sector"] = frame["sector"].astype(str).replace({"": np.nan, "nan": np.nan})
        frame["sector"] = frame["sector"].fillna(
            frame["ticker"].astype(str).map(cfg.TICKER_TO_SECTOR).fillna("unknown")
        )

        keys = frame[["Date", "ticker", "sector"]].copy()
        merged = merge_news_shard_from_best_params(keys, best_params)
        tag = cfg.news_feat_tag(
            int(best_params["news_mom_w"]),
            int(best_params["news_vol_ma"]),
            int(best_params["news_tone_roll"]),
        )
        mc = f"news_macro_{tag}_tone"
        sc = f"news_sec_{tag}_tone"
        if mc not in merged.columns or sc not in merged.columns:
            print(
                f"Hinweis: News-Shard ohne {mc}/{sc} — News-Chip grau.",
                flush=True,
            )
            return sig

        keep = ["Date", "ticker", mc, sc]
        slim = merged[keep].drop_duplicates(subset=["Date", "ticker"])
        out = frame.merge(slim, on=["Date", "ticker"], how="left")
        out["news_sec_minus_macro_tone"] = pd.to_numeric(out[sc], errors="coerce") - pd.to_numeric(
            out[mc], errors="coerce"
        )
        nn = int(out["news_sec_minus_macro_tone"].notna().sum())
        print(
            f"  … News-Shard ({mc}, {sc}): {nn}/{len(out)} Zeilen mit Sektor−Makro-Ton",
            flush=True,
        )
        return out
    except FileNotFoundError as exc:
        print(f"Hinweis: News-Shard nicht gefunden ({exc}) — News-Chip bleibt grau.", flush=True)
        return sig
    except Exception as exc:
        print(f"Warnung: News-Shard-Merge fehlgeschlagen ({exc}).", file=sys.stderr, flush=True)
        return sig


def main(holdout_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """
    holdout_df: optional DataFrame inkl. Klassifikationsspalten (s. _CLASSIFICATION_META_COLS).
                Wenn None: data/holdout_signals.csv oder Rebuild aus master/meta CSV.
    """

    def _ensure_classification_columns(frame: pd.DataFrame) -> pd.DataFrame:
        for c in _CLASSIFICATION_META_COLS:
            if c not in frame.columns:
                frame[c] = ""
        return frame

    if holdout_df is None:
        if HOLDOUT_CSV.is_file():
            sig = pd.read_csv(HOLDOUT_CSV)
            miss = [c for c in _MIN_META_COLS if c not in sig.columns]
            if miss:
                print(
                    f"Fehlt {HOLDOUT_CSV} Spalten {miss}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            sig = _ensure_classification_columns(sig)
            sig = sig[list(_MIN_META_COLS) + list(_CLASSIFICATION_META_COLS)].copy()
        elif MASTER_COMPLETE_CSV.is_file():
            sig = pd.read_csv(MASTER_COMPLETE_CSV)
            miss = [c for c in _MIN_META_COLS if c not in sig.columns]
            if miss:
                print(
                    f"Fehlt {MASTER_COMPLETE_CSV} Spalten {miss}; alternativ {HOLDOUT_CSV} anlegen.",
                    file=sys.stderr,
                )
                sys.exit(1)
            sig = _ensure_classification_columns(sig)
            sig = sig[list(_MIN_META_COLS) + list(_CLASSIFICATION_META_COLS)].copy()
        else:
            print(
                f"Fehlt {HOLDOUT_CSV} oder {MASTER_COMPLETE_CSV}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        sig = _ensure_classification_columns(holdout_df.copy())

    if sig.empty:
        print(
            "build_holdout_signals_master: 0 Signale — kein Yahoo-Download, "
            "master_complete unverändert.",
            flush=True,
        )
        return None
    if "Date" not in sig.columns:
        print("build_holdout_signals_master: Spalte 'Date' fehlt.", file=sys.stderr)
        sys.exit(1)

    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["signal_date"] = sig["Date"]

    sig = _merge_news_shard_for_holdout(sig)

    end_d = (sig["signal_date"].max() + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    tickers = sorted(sig["ticker"].unique())

    print(f"Download {len(tickers)} Ticker ({YF_START} … {end_d}) …", flush=True)
    _t_dl = time.perf_counter()
    raw = yf.download(
        list(tickers),
        start=YF_START,
        end=end_d,
        auto_adjust=True,
        threads=False,
        progress=False,
    )
    _n_bar = int(len(raw)) if raw is not None and hasattr(raw, "__len__") else 0
    print(
        f"  … Download fertig ({time.perf_counter() - _t_dl:.1f}s, {_n_bar} Handelszeilen gesamt). "
        "OHLC pro Ticker …",
        flush=True,
    )

    dfs: dict[str, pd.DataFrame] = {}
    _nt = len(tickers)
    _step_t = _progress_step(_nt, 10)
    for _ti, t in enumerate(tickers, 1):
        df = _ticker_ohlc(raw, t)
        if df is not None and len(df) >= 20:
            dfs[t] = df
        if _ti % _step_t == 0 or _ti == _nt:
            print(
                f"  … Ticker-Serien {_ti}/{_nt} (davon {len(dfs)} mit >=20 Tagen)",
                flush=True,
            )

    ret_cols = [f"ret_{h}d" for h in HORIZONS]
    rows_fwd: list[dict] = []

    _nsig = len(sig)
    _step_s = _progress_step(_nsig, 10)
    for _si, (_, r) in enumerate(sig.iterrows(), 1):
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
            **{k: r.get(k, "") for k in _CLASSIFICATION_META_COLS},
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
        if _si % _step_s == 0 or _si == _nsig:
            print(f"  … Forward-Renditen {_si}/{_nsig} Signale", flush=True)

    out = pd.DataFrame(rows_fwd)

    # Trainings-Label (optional)
    out["train_target"] = np.nan
    out["rally"] = np.nan
    out["eval_note"] = ""

    print(
        "  … Rally/Target je Ticker (feste cfg.FIXED_Y_*-Regel, wie Training) …",
        flush=True,
    )
    per_ticker = _fixed_y_labels_per_ticker(raw, tickers)

    if per_ticker:
        sig_r = sig.reset_index(drop=True)
        print("  … Labels den Holdout-Zeilen zuordnen (vektorisiert) …", flush=True)
        _no = len(out)
        _step_o = _progress_step(_no, 10)
        _tick = out["ticker"].to_numpy()
        _dts = pd.to_datetime(sig_r["Date"]).dt.normalize().to_numpy()
        _ev = np.array([""] * _no, dtype=object)
        _tt = np.full(_no, np.nan, dtype=float)
        _rv = np.full(_no, np.nan, dtype=float)
        for i in range(_no):
            t = _tick[i]
            d = pd.Timestamp(_dts[i]).normalize()
            if t not in per_ticker:
                _ev[i] = "kein close"
            else:
                date_to_t, date_to_r = per_ticker[t]
                if d not in date_to_t:
                    _ev[i] = "kein Handelstag in Serie"
                else:
                    _tt[i] = float(date_to_t[d])
                    _rv[i] = float(date_to_r[d])
            if (i + 1) % _step_o == 0 or i + 1 == _no:
                print(f"  … Label-Zuordnung {i + 1}/{_no} Zeilen", flush=True)
        out["eval_note"] = _ev
        out["train_target"] = _tt
        out["rally"] = _rv
    else:
        print("Hinweis: Keine Label-Serien — train_target/rally bleiben leer.")

    # Spaltenreihenfolge
    meta = ["ticker", "Date", "prob", "threshold_used", "company", "sector"] + list(
        _CLASSIFICATION_META_COLS
    )
    fwd_meta = ["entry_date", "fwd_error"] + ret_cols + ["ret_mean_5"]
    lab = ["train_target", "rally", "eval_note"]
    for c in meta + fwd_meta + lab:
        if c not in out.columns:
            out[c] = np.nan if c != "eval_note" else ""
    base_cols = meta + fwd_meta + lab
    out = out[base_cols]

    _news_extra = [
        c
        for c in sig.columns
        if str(c).startswith("news_") or c == "news_sec_minus_macro_tone"
    ]
    for c in _news_extra:
        if c not in out.columns:
            out[c] = sig[c].values

    if WITH_FILTERS:
        try:
            from lib.signal_extra_filters import (
                enrich_signal_frame,
                ensure_llm_signal_columns,
                get_last_enrich_diagnostics,
            )

            print("  … Zusatzfilter (Cross-Section, Technik, …) …", flush=True)
            _t_f = time.perf_counter()
            fil = enrich_signal_frame(sig, raw)
            print(f"  … Zusatzfilter fertig ({time.perf_counter() - _t_f:.1f}s)", flush=True)
            for c in fil.columns:
                if c not in out.columns:
                    out[c] = fil[c].values
            tail = [c for c in out.columns if c not in base_cols]
            out = out[base_cols + tail]
            out = ensure_llm_signal_columns(out)
            from lib.red_regime_summary import attach_red_regime_llm_columns

            out = attach_red_regime_llm_columns(out)
            print(f"Zusätzliche Filter-Spalten: {', '.join(tail)}")
            _diag = get_last_enrich_diagnostics()
            if _diag:
                print("\nZusatzfilter-Diagnose (fehlende Kennzahlen):", flush=True)
                _mx = _diag.get("metrics", [])[:15]
                if not _mx:
                    print("  keine fehlenden Kern-Kennzahlen erkannt.", flush=True)
                else:
                    for m in _mx:
                        if m.get("status") == "missing_column":
                            print(
                                f"  - {m.get('metric')}: Spalte fehlt ({m.get('reason')})",
                                flush=True,
                            )
                        else:
                            print(
                                f"  - {m.get('metric')}: missing={m.get('missing_count')}/{_diag.get('row_count', 0)} "
                                f"({m.get('missing_pct', 0.0):.1f}%) | {m.get('reason')}",
                                flush=True,
                            )
                _ss = _diag.get("short_metrics_source_counts") or {}
                if _ss:
                    _src = ", ".join(f"{k}={v}" for k, v in _ss.items())
                    print(f"  short_metrics_source: {_src}", flush=True)
        except Exception as e:
            print(f"Warnung: Zusatzfilter fehlgeschlagen ({e}). Export mit leeren LLM-Strukturspalten.", file=sys.stderr)
            from lib.signal_extra_filters import ensure_llm_signal_columns

            out = ensure_llm_signal_columns(out)
            from lib.red_regime_summary import attach_red_regime_llm_columns

            out = attach_red_regime_llm_columns(out)

    MASTER_COMPLETE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MASTER_COMPLETE_CSV, index=False)

    daily = _build_daily_update(out, list(ret_cols))
    daily.to_csv(MASTER_DAILY_CSV, index=False)

    n_ok = int(out[ret_cols].notna().all(axis=1).sum())
    print(
        f"\nGeschrieben: {MASTER_COMPLETE_CSV}  ({len(out)} Zeilen, davon {n_ok} mit allen Forward-Renditen)"
    )
    print(
        f"Geschrieben: {MASTER_DAILY_CSV}  ({len(daily)} Zeilen, letzter Signaltag, LLM-Spalten)"
    )
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

    return out


SIGNALS_JSON = ROOT / "docs" / "signals.json"


def holdout_rows_from_signals_json(path: Path | None = None) -> pd.DataFrame:
    """Meta-Zeilen für alle Einträge in ``docs/signals.json`` (Website/OOS-Keys)."""
    p = path or SIGNALS_JSON
    if not p.is_file():
        raise FileNotFoundError(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    signals: list[dict] = []
    if isinstance(payload, list):
        signals = payload
    elif isinstance(payload, dict):
        for key in ("signals", "signals_holdout_final"):
            block = payload.get(key)
            if isinstance(block, list) and block:
                signals = block
                break
    if not signals:
        return pd.DataFrame(columns=list(_MIN_META_COLS) + list(_CLASSIFICATION_META_COLS))
    rows: list[dict] = []
    for s in signals:
        rows.append(
            {
                "ticker": str(s.get("ticker", "")).strip(),
                "Date": str(s.get("date", s.get("Date", "")))[:10],
                "prob": s.get("prob", np.nan),
                "threshold_used": s.get("threshold_used", np.nan),
                "company": s.get("company", s.get("ticker", "")),
                "sector": s.get("sector", "—"),
                **{k: s.get(k, "") for k in _CLASSIFICATION_META_COLS},
            }
        )
    return pd.DataFrame(rows)


def rebuild_master_from_signals_json(path: Path | None = None) -> pd.DataFrame | None:
    """``master_complete.csv`` für alle Keys in signals.json neu aufbauen."""
    sig = holdout_rows_from_signals_json(path)
    if sig.empty:
        print("rebuild_master_from_signals_json: 0 Signale — Abbruch.", flush=True)
        return None
    print(
        f"rebuild_master_from_signals_json: {len(sig)} Signale aus {path or SIGNALS_JSON}",
        flush=True,
    )
    return main(holdout_df=sig)


if __name__ == "__main__":
    if "--from-signals-json" in sys.argv:
        rebuild_master_from_signals_json()
    else:
        main()
