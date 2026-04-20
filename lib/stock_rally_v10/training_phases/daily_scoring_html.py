"""Phase 17: Daily Scoring, Charts, HTML/JSON, optional Gemini & Git."""
from __future__ import annotations

import base64
import html as _html_std
import io
import json as _json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import average_precision_score, precision_recall_curve

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.features import merge_news_shard_from_best_params

warnings.filterwarnings("ignore")


def _phase17_apply_signals_one(
    ticker: str,
    sub: pd.DataFrame,
    threshold: float,
    fkw: dict[str, Any],
) -> tuple[str, np.ndarray]:
    """Pickelbar für joblib/loky: alle Filterkonstanten in ``fkw`` (nicht globales cfg im Worker)."""
    from lib.stock_rally_v10.holdout_plot import apply_signal_filters

    sig_dates = apply_signal_filters(sub, threshold, **fkw)
    return ticker, sig_dates


def run_phase_daily_scoring_html(cfg_mod: Any) -> None:
    _run_phase17(cfg_mod)


def _run_phase17(c: Any) -> None:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d %H:%M")

    _root_smoke = Path.cwd()
    if str(_root_smoke) not in sys.path:
        sys.path.insert(0, str(_root_smoke))
    from config.load_env import load_project_env

    load_project_env(_root_smoke)
    if os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            from google import genai as _genai_smoke
            from google.genai import types as _gtypes_smoke

            _client_smoke = _genai_smoke.Client(
                api_key=os.environ["GEMINI_API_KEY"].strip(),
                http_options=_gtypes_smoke.HttpOptions(timeout=600_000),
            )
            _mraw = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash").strip()
            _model_smoke = _mraw[len("models/") :] if _mraw.startswith("models/") else _mraw
            _r_smoke = _client_smoke.models.generate_content(
                model=_model_smoke,
                contents="Antworte ausschließlich mit einem einzigen Wort: OK",
            )
            try:
                _txt_smoke = (_r_smoke.text or "").strip()[:500]
            except ValueError:
                _parts = []
                for _cand in _r_smoke.candidates or []:
                    if not _cand.content or not _cand.content.parts:
                        continue
                    for p in _cand.content.parts:
                        if hasattr(p, "text") and p.text:
                            _parts.append(p.text)
                _txt_smoke = "".join(_parts).strip()[:500]
            if not _txt_smoke:
                print("[Gemini] Smoke-Test: keine Text-Antwort — Phase 17 abgebrochen.", flush=True)
                raise SystemExit(1)
            print(f"[Gemini] Smoke-Test OK ({_model_smoke}): {_txt_smoke!r}", flush=True)
        except SystemExit:
            raise
        except Exception as _e_smoke:
            print(
                f"[Gemini] Smoke-Test fehlgeschlagen ({type(_e_smoke).__name__}: {_e_smoke}) — Phase 17 abgebrochen.",
                flush=True,
            )
            raise SystemExit(1)
    else:
        print("[Gemini] Kein GEMINI_API_KEY — Smoke-Test übersprungen.", flush=True)

    print("Preparing full-history feature matrix …")
    if c.df_features is None or len(c.df_features) == 0:
        print(
            "  FEHLER Phase17: c.df_features ist leer — Full-History-Scoring und HTML werden übersprungen.",
            flush=True,
        )
        return
    df_s = c.df_features.copy()
    df_s = merge_news_shard_from_best_params(df_s, c.best_params)
    if len(df_s) == 0:
        print(
            "  FEHLER Phase17: df_features nach News-Merge leer — Abbruch Phase17.",
            flush=True,
        )
        return
    feat_arr = df_s[c.FEAT_COLS].values.astype(np.float32)
    _arr_nan = np.isnan(feat_arr)
    _arr_inf = np.isinf(feat_arr)
    _nan_cells = int(_arr_nan.sum())
    _inf_cells = int(_arr_inf.sum())
    _bad_rows = int((_arr_nan | _arr_inf).any(axis=1).sum())
    if _nan_cells or _inf_cells:
        feat_df = df_s[c.FEAT_COLS]
        bad_cols = []
        for col in c.FEAT_COLS:
            s = pd.to_numeric(feat_df[col], errors="coerce")
            n_nan = int(s.isna().sum())
            n_inf = int(np.isinf(s.to_numpy(dtype=np.float64, copy=False)).sum())
            n_bad = n_nan + n_inf
            if n_bad > 0:
                bad_cols.append((col, n_bad, n_nan, n_inf))
        bad_cols.sort(key=lambda x: x[1], reverse=True)
        _top = bad_cols[:15]
        print(
            "  Phase17: Problematische FEAT_COLS erkannt "
            f"(NaN={_nan_cells}, Inf={_inf_cells}, Zeilen mit ≥1 Problemwert: {_bad_rows}/{len(df_s)}).",
            flush=True,
        )
        n_rows = max(1, len(df_s))
        print("  Top-Spalten nach Problemwerten (count = NaN+Inf):", flush=True)
        for col, n_bad, n_nan, n_inf in _top:
            pct_bad = 100.0 * float(n_bad) / float(n_rows)
            print(
                f"    - {col}: {n_bad} ({pct_bad:.1f}%) (NaN={n_nan}, Inf={n_inf})",
                flush=True,
            )
        if len(bad_cols) > len(_top):
            print(f"    ... +{len(bad_cols) - len(_top)} weitere Spalten", flush=True)
        _nan_s = getattr(c, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8)
        print(f"  Für Phase17-Scoring werden diese Werte auf Sentinel {_nan_s} gesetzt.", flush=True)
    _nan_sentinel = np.float32(getattr(c, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    feat_arr = np.nan_to_num(
        feat_arr,
        nan=_nan_sentinel,
        posinf=_nan_sentinel,
        neginf=_nan_sentinel,
        copy=False,
    ).astype(np.float32, copy=False)
    df_s = df_s.reset_index(drop=True)
    print(
        f'  {len(df_s):,} Zeilen, {df_s["ticker"].nunique()} Ticker — building meta features …'
    )

    X_meta_all = c.build_meta_features(feat_arr, dataset_label="FULL HISTORY")
    probs_all = c.meta_clf.predict_proba(X_meta_all)[:, 1]
    df_s["prob"] = probs_all
    print("  Scoring done.", flush=True)

    _rows_for_signal_calendar_day = c._rows_for_signal_calendar_day
    best_threshold = c.best_threshold
    COMPANY_NAMES = c.COMPANY_NAMES
    TICKER_TO_SECTOR = c.TICKER_TO_SECTOR
    df_final = c.df_final

    _tickers_sorted = sorted(
        {
            s
            for s in df_s["ticker"].dropna().astype(str).str.strip().unique()
            if s and s.lower() != "nan" and s not in ("0.0", "0")
        }
    )
    if not _tickers_sorted:
        _raw = sorted(df_s["ticker"].dropna().astype(str).str.strip().unique())
        print(
            "  WARNUNG Phase17: Keine gültigen Ticker nach Filter (0.0/0 ausgeschlossen). "
            f"Roh-Unique (max. 15): {_raw[:15]} — Ticker-Spalte in Features prüfen.",
            flush=True,
        )
        _tickers_sorted = [x for x in _raw if x and x.lower() != "nan"]
    _nt = len(_tickers_sorted)
    print(
        "  Branchenklassifikation (Yahoo sector + industry via yfinance) für Website & Holdout-CSV …",
        flush=True,
    )
    from lib.stock_rally_v10.equity_classification import (
        CLASSIFICATION_COLUMN_KEYS,
        build_classification_cache,
    )

    _class_cache = build_classification_cache(_tickers_sorted)
    _rw = getattr(c, "rsi_w", None)
    if _rw is None:
        _rw = int(c.SEED_PARAMS.get("rsi_window", 14))
    else:
        _rw = int(_rw)
    _fkw: dict[str, Any] = {
        "consecutive_days": c.CONSECUTIVE_DAYS,
        "signal_cooldown_days": c.SIGNAL_COOLDOWN_DAYS,
        "rsi_window": _rw,
        "signal_skip_near_peak": bool(c.SIGNAL_SKIP_NEAR_PEAK),
        "peak_lookback_days": int(c.PEAK_LOOKBACK_DAYS),
        "peak_min_dist_from_high_pct": float(c.PEAK_MIN_DIST_FROM_HIGH_PCT),
        "signal_max_rsi": getattr(c, "SIGNAL_MAX_RSI", None),
    }
    _ts_norm = df_s["ticker"].astype(str).str.strip()
    _tasks = [
        (t, df_s.loc[_ts_norm == str(t)].sort_values("Date").reset_index(drop=True))
        for t in _tickers_sorted
    ]
    _n_jobs = int(getattr(c, "PHASE17_SIGNAL_FILTER_JOBS", -1))
    _pbatch = int(getattr(c, "PHASE17_SIGNAL_FILTER_PROGRESS_BATCH", 0) or 0)
    _step = _pbatch if _pbatch > 0 else max(1, _nt // 10)
    if _n_jobs != 1 and len(_tasks) >= 4:
        print(
            f"  Signale filtern: {_nt} Ticker parallel (joblib loky, n_jobs={_n_jobs}, "
            f"Fortschritt alle {_step}) …",
            flush=True,
        )
        _pairs: list[tuple[str, np.ndarray]] = []
        for i in range(0, len(_tasks), _step):
            batch = _tasks[i : i + _step]
            chunk = Parallel(n_jobs=_n_jobs, backend="loky", verbose=0)(
                delayed(_phase17_apply_signals_one)(t, sub, float(best_threshold), _fkw)
                for t, sub in batch
            )
            _pairs.extend(chunk)
            done = min(i + len(batch), len(_tasks))
            print(f"    … {done}/{_nt} Ticker (Signal-Filter)", flush=True)
    else:
        print(
            f"  Signale filtern: {_nt} Ticker seriell "
            f"(PHASE17_SIGNAL_FILTER_JOBS=1 oder <4 Ticker), Fortschritt alle {_step} …",
            flush=True,
        )
        _pairs = []
        for i in range(0, len(_tasks), _step):
            batch = _tasks[i : i + _step]
            for t, sub in batch:
                _pairs.append(_phase17_apply_signals_one(t, sub, float(best_threshold), _fkw))
            done = min(i + len(batch), len(_tasks))
            print(f"    … {done}/{_nt} Ticker (Signal-Filter)", flush=True)

    all_hist_signals = []
    for ticker, sig_dates in _pairs:
        sub = df_s.loc[_ts_norm == str(ticker)].sort_values("Date").reset_index(drop=True)
        for d in sig_dates:
            match = _rows_for_signal_calendar_day(sub, d)
            if match.empty:
                continue
            _cl = _class_cache.get(ticker) or {}
            _cls = {k: _cl.get(k, "") for k in CLASSIFICATION_COLUMN_KEYS}
            all_hist_signals.append(
                {
                    "ticker": ticker,
                    "company": COMPANY_NAMES.get(ticker, ticker),
                    "sector": TICKER_TO_SECTOR.get(ticker, "—"),
                    **_cls,
                    "date": str(d)[:10],
                    "prob": float(match["prob"].values[0]),
                }
            )

    holdout_keys = set(
        zip(df_final["ticker"], pd.to_datetime(df_final["Date"]).dt.strftime("%Y-%m-%d"))
    )
    signals_holdout_final = [s for s in all_hist_signals if (s["ticker"], s["date"]) in holdout_keys]
    signals_holdout_final.sort(key=lambda x: x["date"], reverse=True)
    all_hist_signals.sort(key=lambda x: x["date"], reverse=True)
    print(f"\n{len(all_hist_signals)} historical signals across all tickers.")
    print(f"{len(signals_holdout_final)} signals in FINAL holdout (unbiased / OOS analysis).")

    Path("data").mkdir(parents=True, exist_ok=True)
    _ho_rows = [
        {
            "ticker": s["ticker"],
            "Date": s["date"],
            "prob": s["prob"],
            "threshold_used": float(best_threshold),
            "company": s["company"],
            "sector": s["sector"],
            **{k: s.get(k, "") for k in CLASSIFICATION_COLUMN_KEYS},
        }
        for s in signals_holdout_final
    ]
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    from holdout.build_holdout_signals_master import main as _build_holdout_master

    if _ho_rows:
        _exported_ho = _build_holdout_master(holdout_df=pd.DataFrame(_ho_rows))
        if _exported_ho is not None and len(_exported_ho) > 0:
            _cls_keys = list(CLASSIFICATION_COLUMN_KEYS)
            signals_holdout_final = []
            for _, _r in _exported_ho.iterrows():
                signals_holdout_final.append(
                    {
                        "ticker": str(_r["ticker"]),
                        "company": str(_r.get("company", _r["ticker"])),
                        "sector": str(_r.get("sector", "—")),
                        **{k: str(_r.get(k, "")) for k in _cls_keys},
                        "date": str(_r["Date"])[:10],
                        "prob": float(_r["prob"]),
                    }
                )
            signals_holdout_final.sort(key=lambda x: x["date"], reverse=True)
            print(
                f"wrote data/master_complete.csv & master_daily_update.csv (LLM-Spalten) "
                f"({len(_exported_ho)} holdout rows)"
            )
        else:
            signals_holdout_final = []
            print(
                "Holdout-CSV: 0 Zeilen nach build_holdout_signals_master.",
                flush=True,
            )
    else:
        print(
            "Holdout-CSV übersprungen: 0 Signale im FINAL-Zeitfenster (Schwelle + Filter + Datenlage). "
            "Kein KeyError mehr — master_complete wird nicht überschrieben.",
            flush=True,
        )

    # Website + öffentliches signals.json: nur echte OOS — keine Signale aus TRAIN/THRESHOLD-Kalender.
    website_signals = list(signals_holdout_final)
    website_signals.sort(key=lambda x: x["date"], reverse=True)
    print(
        f"Website/OOS-Export: {len(website_signals)} Signale "
        f"(intern für Diagnose: {len(all_hist_signals)} über alle Zeiten mit Schwelle+Filter).",
        flush=True,
    )

    recent_cutoff = (pd.Timestamp(c.END_DATE) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    recent_signals = [s for s in website_signals if s["date"] >= recent_cutoff]
    print(f"{len(recent_signals)} OOS-Signale in den letzten 30 Tagen (Website).")

    _chart_yf_failures: list[tuple[Any, ...]] = []

    def _yf_close_rows_from_series(ser):
        rows = []
        if ser is None or len(ser) == 0:
            return rows
        s = ser.dropna()
        for _ti, _px in s.items():
            rows.append({"Date": pd.Timestamp(pd.Timestamp(_ti).date()), "close": float(_px)})
        return rows

    def _extend_chart_close_yfinance(ticker, win, win_lo, win_hi):
        _last_feat = pd.to_datetime(win["Date"]).max().normalize()
        _chart_end = min(
            pd.Timestamp(win_hi).normalize(), pd.Timestamp(datetime.today().date()).normalize()
        )
        if _last_feat >= _chart_end:
            return win
        _d0 = (_last_feat + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        _d1 = (_chart_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        _rows = []
        _last_err = None
        for _attempt in range(3):
            try:
                _hist = yf.Ticker(ticker).history(
                    start=_d0,
                    end=_d1,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                )
                if _hist is not None and len(_hist) and "Close" in _hist.columns:
                    _rows = _yf_close_rows_from_series(_hist["Close"])
                if not _rows:
                    _ext = yf.download(ticker, start=_d0, end=_d1, progress=False, threads=False)
                    if _ext is not None and len(_ext) > 0:
                        _ser = _ext["Close"]
                        if isinstance(_ser, pd.DataFrame):
                            _ser = _ser.iloc[:, 0]
                        _rows = _yf_close_rows_from_series(_ser)
                if _rows:
                    break
            except Exception as _e:
                _last_err = _e
                time.sleep(0.35 * (_attempt + 1))
        if not _rows:
            _chart_yf_failures.append(
                (ticker, _d0, _d1, repr(_last_err) if _last_err else "keine Zeilen")
            )
            return win
        _w = pd.concat([win[["Date", "close"]].copy(), pd.DataFrame(_rows)], ignore_index=True)
        _w["Date"] = pd.to_datetime(_w["Date"]).dt.normalize()
        _w = _w.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        return _w[(_w["Date"] >= win_lo) & (_w["Date"] <= win_hi)].copy()

    def _make_chart(ticker, sig_date_str):
        sub = df_s[df_s["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if "close" not in sub.columns or len(sub) < 5:
            return None
        sig_ts = pd.Timestamp(sig_date_str).normalize()
        win_lo = sig_ts - pd.DateOffset(months=1)
        win_hi = sig_ts + pd.DateOffset(months=1)
        win = sub[(sub["Date"] >= win_lo) & (sub["Date"] <= win_hi)].copy()
        if len(win) < 5:
            return None
        win = _extend_chart_close_yfinance(ticker, win, win_lo, win_hi)
        sig_row = win[pd.to_datetime(win["Date"]).dt.normalize() == sig_ts]
        if not sig_row.empty:
            ref = float(sig_row["close"].iloc[0])
        else:
            j = int(pd.to_datetime(sub["Date"]).searchsorted(sig_ts))
            j = min(max(j, 0), len(sub) - 1)
            ref = float(sub.iloc[j]["close"])
        if not np.isfinite(ref) or ref <= 0:
            return None
        try:
            close = win["close"].astype(float)
            dt = pd.to_datetime(win["Date"])
            pct = (close / ref - 1.0) * 100.0
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.plot(dt, close, color="#42a5f5", lw=1.5, label="Close")
            ax.fill_between(dt, close, alpha=0.10, color="#42a5f5")
            ax2 = ax.twinx()
            ax2.plot(dt, pct, color="#ce93d8", lw=1.2, label="% vs. Datenstandstag")
            ax2.axhline(0.0, color="#66bb6a", lw=1.0, ls="-", alpha=0.9, zorder=4)
            ax.axvline(sig_ts, color="#66bb6a", lw=2, ls="--", zorder=5, label="Datenstand (0 %)")
            ax.set_xlim(win_lo, win_hi)
            ymin, ymax = float(close.min()), float(close.max())
            if ymax > ymin:
                pad = (ymax - ymin) * 0.02
                ax.set_ylim(ymin - pad, ymax + pad)
            else:
                span = max(abs(ymax), 1e-6) * 0.02
                ax.set_ylim(ymin - span, ymax + span)
            pmin, pmax = float(np.nanmin(pct)), float(np.nanmax(pct))
            ppad = max((pmax - pmin) * 0.06, 0.8)
            ax2.set_ylim(pmin - ppad, pmax + ppad)
            today_ts = pd.Timestamp(datetime.today().date()).normalize()
            if win_lo <= today_ts <= win_hi:
                ax.axvline(today_ts, color="#ffa726", lw=1.5, ls=":", zorder=5, label="Heute")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(
                h1 + h2,
                l1 + l2,
                fontsize=7,
                loc="upper left",
                facecolor="#1a1a2e",
                edgecolor="#2d2d4e",
                labelcolor="#e0e0e0",
            )
            _ds = pd.Timestamp(sig_date_str).strftime("%d.%m.%Y")
            ax.set_title(
                f"{ticker} — {COMPANY_NAMES.get(ticker, ticker)}\n"
                f"Datenstand Modell: bis einschl. {_ds} (grüner Strich); rechts nur Kurs (Anzeige)",
                fontsize=8,
                color="#81d4fa",
            )
            ax.set_ylabel("Close", color="#90a4ae", fontsize=8)
            ax2.set_ylabel("% vs. Datenstand (Close)", color="#e1bee7", fontsize=8)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7)
            ax.tick_params(colors="#90a4ae", labelsize=7)
            ax2.tick_params(colors="#ce93d8", labelsize=7)
            ax.grid(True, alpha=0.18)
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#1a1a2e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#2d2d4e")
            ax2.spines["right"].set_edgecolor("#4a3f55")
            plt.tight_layout(pad=0.8)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=95, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            plt.close("all")
            return None

    print("Generating charts (yfinance + matplotlib, bis zu 600 OOS-Signale) …", flush=True)
    _chart_yf_failures.clear()
    chart_cache = {}
    for s in website_signals[:600]:
        key = (s["ticker"], s["date"])
        if key not in chart_cache:
            b64 = _make_chart(s["ticker"], s["date"])
            if b64:
                chart_cache[key] = b64
    print(f"  {len(chart_cache)} charts generated.")
    if _chart_yf_failures:
        _bad = sorted({t for t, _d0, _d1, _e in _chart_yf_failures})
        print(
            f"  Warnung: Kursnachzug (yfinance): {len(_chart_yf_failures)} fehlgeschlagene Abrufe "
            f"({len(_bad)} eindeutige Ticker) — Charts enden am letzten df_features-Tag. "
            f"Ticker: {_bad[:25]}"
            f'{" …" if len(_bad) > 25 else ""}',
            flush=True,
        )

    _yf_failed_tickers = {t for t, _d0, _d1, _e in _chart_yf_failures}

    pr_b64 = ""
    if not getattr(c, "SCORING_ONLY", False):
        try:
            y_threshold = c.df_threshold["target"].values.astype(np.int8)
            y_prob_threshold = c.df_threshold["prob"].values
            y_final = c.df_final["target"].values.astype(np.int8)
            y_prob_final = c.df_final["prob"].values
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            fig.patch.set_facecolor("#1a1a2e")
            for ax, y_t, y_p, lbl, col in [
                (axes[0], y_threshold, y_prob_threshold, "THRESHOLD", "#42a5f5"),
                (axes[1], y_final, y_prob_final, "FINAL", "#66bb6a"),
            ]:
                pc, rc, _ = precision_recall_curve(y_t, y_p)
                ap = average_precision_score(y_t, y_p)
                ax.plot(rc, pc, color=col, lw=2, label=f"AP={ap:.3f}")
                ax.axhline(y=0.60, color="#ef5350", lw=1, ls="--", label="60 %")
                ax.set_title(f"PR — {lbl}", color="#81d4fa", fontsize=9)
                ax.set_xlabel("Recall", color="#90a4ae", fontsize=8)
                ax.set_ylabel("Precision", color="#90a4ae", fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.2)
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors="#90a4ae", labelsize=7)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2d2d4e")
            plt.tight_layout(pad=1.2)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=95, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            buf.seek(0)
            pr_b64 = base64.b64encode(buf.read()).decode()
        except Exception as exc:
            print(f"PR plot failed: {exc}")
    else:
        print("[SCORING_ONLY] PR-Kurven übersprungen (keine Trainings-Labels in dieser Session).")

    def _signal_card(s):
        key = (s["ticker"], s["date"])
        b64 = chart_cache.get(key, "")
        bar = int(s["prob"] * 100)
        _yf_note = (
            '<p class="yf-hint">Kursnachzug (yfinance) fehlgeschlagen — Chart endet am letzten Tag der '
            "Feature-Matrix; beim nächsten Lauf kann es wieder klappen.</p>"
            if b64 and s["ticker"] in _yf_failed_tickers
            else ""
        )
        chart_html = (
            f'<img src="data:image/png;base64,{b64}" alt="{s["ticker"]}" loading="lazy">' if b64 else ""
        )
        chart_note = (
            f'<p class="sig-chart-note" style="font-size:0.72em;color:#546e7a;margin-top:6px;line-height:1.35">'
            f'Merkmale und Kursbasis für dieses Signal: bis einschließlich <strong>{s["date"]}</strong> '
            f"(nicht der Zeitpunkt der Berechnung); rechts vom grünen Strich nur nachgelagerter Kurs "
            f"(Anzeige) — kein Look-ahead fürs Modell.</p>"
            if b64
            else ""
        )
        _gics_bits = [x for x in (s.get("gics_sector"), s.get("gics_industry")) if x]
        _gics_line = " · ".join(str(x) for x in _gics_bits) if _gics_bits else ""
        _gics_std = (s.get("classification_standard") or "").strip()
        _gics_html = ""
        if _gics_line or _gics_std:
            if _gics_line and _gics_std:
                _body = f"{_html_std.escape(_gics_std)} — {_html_std.escape(_gics_line)}"
            elif _gics_line:
                _body = _html_std.escape(_gics_line)
            else:
                _body = _html_std.escape(_gics_std)
            _gics_html = (
                f'<p class="sig-gics" title="Näherung aus Yahoo Finance; keine offiziellen MSCI-GICS-Codes im Free-Tier">'
                f"{_body}</p>"
            )
        _gics_sec = (s.get("gics_sector") or "").strip()
        _internal_cluster = str(s.get("sector") or "").strip()
        if _gics_sec:
            _badge_label = _gics_sec
            _badge_title = (
                f"Modell-Cluster (News/Features): {_internal_cluster}"
                if _internal_cluster
                else ""
            )
        else:
            _badge_label = _internal_cluster.replace("_", " ").title() if _internal_cluster else "—"
            _badge_title = "Kein Yahoo-GICS — nur internes Modell-Label"
        _badge_title_esc = _html_std.escape(_badge_title) if _badge_title else ""
        _badge_title_attr = f' title="{_badge_title_esc}"' if _badge_title_esc else ""
        return f"""
      <div class="sig-card">
        <div class="sig-head">
          <span class="sig-ticker">{s['ticker']}</span>
          <span class="sig-company">{s['company']}</span>
          <span class="sig-sector"{_badge_title_attr}>{_html_std.escape(_badge_label)}</span>
          <span class="sig-date" title="Kurs- und Merkmalsdaten bis einschließlich diesem Tag — nicht der Laufzeitpunkt der Berechnung"><span class="sig-date-pre">Daten bis</span> {s['date']}</span>
          <div class="score-bar-bg"><div class="score-bar" style="width:{bar}%">{s['prob']:.3f}</div></div>
        </div>
        {_gics_html}
        {_yf_note}
        {chart_html}
        {chart_note}
      </div>"""

    recent_html = (
        "".join(_signal_card(s) for s in recent_signals)
        or '<p class="empty">Keine Signale in den letzten 30 Tagen.</p>'
    )
    hist_html = (
        "".join(_signal_card(s) for s in website_signals[:600])
        or '<p class="empty">Keine OOS-Signale.</p>'
    )

    pr_section = (
        f'<div class="section"><h2>Model Quality &#8212; Precision-Recall</h2>'
        f'<img src="data:image/png;base64,{pr_b64}" alt="PR" style="max-width:100%;border-radius:6px"></div>'
        if pr_b64
        else ""
    )
    recent_badge_cls = " zero" if not recent_signals else ""

    _prompt_path = Path("docs") / "website_analysis_prompt.txt"
    if _prompt_path.is_file():
        _ANALYSIS_PROMPT_DE = _prompt_path.read_text(encoding="utf-8")
    else:
        _ANALYSIS_PROMPT_DE = "[docs/website_analysis_prompt.txt fehlt]"
    try:
        from lib.website_rally_prompt import load_rally_prompt_injection

        _rally_llm_block = load_rally_prompt_injection(Path.cwd())
        if _rally_llm_block.strip():
            _ANALYSIS_PROMPT_DE = _rally_llm_block.rstrip() + "\n\n---\n\n" + _ANALYSIS_PROMPT_DE
    except Exception:
        pass
    _ANALYSIS_PROMPT_ESC = _html_std.escape(_ANALYSIS_PROMPT_DE)

    try:
        _root_llm = Path.cwd()
        if str(_root_llm) not in sys.path:
            sys.path.insert(0, str(_root_llm))
        from config.load_env import load_project_env as _load_env_llm

        _load_env_llm(_root_llm)
    except Exception:
        pass
    _analysis_llm_section = ""
    _script_gemini = Path("scripts") / "run_website_analysis_gemini.py"
    _gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if _gemini_key and _script_gemini.is_file():
        try:
            print("Gemini: KI-Analyse (CSV-Upload + Google Search) …", flush=True)
            _r = subprocess.run(
                [sys.executable, str(_script_gemini)],
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if _r.returncode != 0:
                print("LLM-Analyse Fehler:", (_r.stderr or _r.stdout or "")[:2500], flush=True)
            _hf = Path("docs") / "analysis_llm_last.html"
            if _hf.is_file():
                _analysis_llm_section = _hf.read_text(encoding="utf-8")
            else:
                print(
                    "Hinweis: docs/analysis_llm_last.html fehlt nach Gemini-Lauf. Ausgabe:",
                    ((_r.stdout or "") + (_r.stderr or ""))[:3000],
                    flush=True,
                )
        except Exception as _e_llm:
            print("LLM-Analyse:", _e_llm, flush=True)
    else:
        if not _gemini_key:
            print("Hinweis: GEMINI_API_KEY setzen — keine automatische KI-Analyse.", flush=True)

    _matrix_last_date = str(pd.to_datetime(df_s["Date"]).max().date())
    _thr_cal = getattr(c, "threshold_calibration_end_date", None)
    if _thr_cal is None and getattr(c, "df_threshold", None) is not None:
        try:
            _thr_cal = str(pd.to_datetime(c.df_threshold["Date"]).max().date())
        except Exception:
            _thr_cal = None
    if isinstance(_thr_cal, str) and len(_thr_cal) >= 10:
        _thr_cal = _thr_cal[:10]
    _thr_chip = (
        f'<div class="chip" title="Letzter Handelstag der THRESHOLD-Stichprobe bei der Kalibrierung von best_threshold">'
        f"Threshold-Kalibrierung: Stichprobe bis <strong>{_thr_cal}</strong></div>"
        if _thr_cal
        else '<div class="chip" title="Artefakt mit Zelle 18 neu speichern (nach Training) — dann steht das Datum der THRESHOLD-Stichprobe im Bundle.">'
        "Threshold-Kalibrierung: Stichprobe bis <strong>—</strong> (nicht im Artefakt)</div>"
    )
    _target_section = cfg.html_target_definition_section(c)

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <meta name="theme-color" content="#16213e">
  <link rel="manifest" href="manifest.json">
  <title>Stock Signals</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f1a;color:#e0e0e0;min-height:100vh}}
    header{{background:#16213e;padding:12px 18px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.6)}}
    header h1{{font-size:1.1em;color:#81d4fa}}
    header .ts{{font-size:.72em;color:#546e7a}}
    .page-wrap{{display:flex;align-items:flex-start;gap:18px;max-width:1320px;margin:0 auto;padding:12px 16px}}
    .main-col{{flex:1;min-width:0;max-width:900px}}
    .llm-sidebar{{width:min(340px,100%);flex-shrink:0;position:sticky;top:52px;align-self:flex-start;max-height:calc(100vh - 64px);overflow-y:auto;background:#141428;border:1px solid #2d2d4e;border-radius:10px;padding:10px 12px}}
    .llm-details summary{{cursor:pointer;color:#81d4fa;font-size:.92em;font-weight:600;padding:6px 4px;user-select:none;list-style:none}}
    .llm-details summary::-webkit-details-marker{{display:none}}
    .llm-details summary::before{{content:'▸ ';color:#64b5f6}}
    .llm-details[open] summary::before{{content:'▾ '}}
    .llm-sum-hint{{font-weight:400;font-size:.78em;color:#78909c;margin-left:6px}}
    .llm-slot{{margin-top:8px;padding-top:8px;border-top:1px solid #2a2a44}}
    .llm-sidebar .analysis-llm h2{{display:none}}
    .llm-sidebar .section.analysis-llm{{background:transparent;padding:0;margin:0;border:none}}
    .llm-sidebar .analysis-llm-body{{max-width:none;font-size:.88em}}
    .yf-hint{{font-size:.68em;color:#78909c;margin:0 0 6px;line-height:1.35}}
    .section{{background:#1a1a2e;border-radius:10px;padding:16px;margin-bottom:14px}}
    .section h2{{font-size:.95em;color:#81d4fa;margin-bottom:12px;border-bottom:1px solid #2d2d4e;padding-bottom:7px;display:flex;align-items:center;gap:8px}}
    .badge{{background:#4caf50;color:#fff;border-radius:10px;padding:1px 8px;font-size:.72em}}
    .badge.zero{{background:#607d8b}}
    .sig-card{{background:#0d1117;border-radius:8px;padding:12px;margin-bottom:10px}}
    .sig-head{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}}
    .sig-ticker{{font-weight:700;color:#81d4fa;font-size:.95em;min-width:70px}}
    .sig-company{{color:#90a4ae;font-size:.82em;flex:1;min-width:100px}}
    .sig-sector{{background:#1e2a3a;color:#64b5f6;font-size:.72em;padding:2px 7px;border-radius:10px;white-space:nowrap;align-self:center}}
    .sig-gics{{font-size:.74em;color:#90a4ae;line-height:1.45;margin:6px 0 0;padding:0 2px;max-width:100%}}
    .sig-date{{color:#546e7a;font-size:.78em;white-space:nowrap}}
    .sig-date-pre{{font-size:.68em;color:#78909c;margin-right:5px}}
    .score-bar-bg{{background:#1a1a2e;border-radius:4px;overflow:hidden;min-width:70px;max-width:110px}}
    .score-bar{{background:#4caf50;padding:2px 6px;color:#fff;font-size:.75em;white-space:nowrap;border-radius:4px}}
    img{{max-width:100%;border-radius:5px;display:block}}
    .model-chips{{display:flex;flex-wrap:wrap;gap:8px}}
    .chip{{background:#0d1117;border-radius:6px;padding:5px 10px;font-size:.8em;color:#90a4ae}}
    .chip strong{{color:#e0e0e0}}
    .target-def-body p{{font-size:.84em;color:#b0bec5;line-height:1.58;margin-bottom:10px}}
    .target-def-body p:last-child{{margin-bottom:0}}
    .empty{{color:#546e7a;font-size:.85em;padding:10px 0}}
    details summary{{cursor:pointer;color:#81d4fa;font-size:.9em;padding:4px 0;user-select:none;list-style:none}}
    details summary::before{{content:'&#9654; '}}
    details[open] summary::before{{content:'&#9660; '}}
    details[open] summary{{margin-bottom:10px}}
    .prompt-lead{{font-size:.85em;color:#90a4ae;margin-bottom:10px;line-height:1.4}}
    .prompt-pre{{font-size:.78em;line-height:1.45;color:#b0bec5;white-space:pre-wrap;word-break:break-word;background:#0d1117;border-radius:8px;padding:12px;margin-top:8px;border:1px solid #2d2d4e}}
    .analysis-llm-body{{font-size:.92em;line-height:1.62;color:#eceff1;max-width:52em}}
    .analysis-llm-body.prose-analysis p{{margin:0 0 12px}}
    .analysis-llm-body h3{{font-size:1.05em;font-weight:600;color:#81d4fa;margin:18px 0 8px;border-bottom:1px solid #2d3f50;padding-bottom:4px}}
    .analysis-llm-body h4{{font-size:.98em;font-weight:600;color:#b3e5fc;margin:14px 0 6px}}
    .analysis-llm-body strong{{color:#fff;font-weight:600}}
    .analysis-llm-body .analysis-ul,.analysis-llm-body .analysis-ol{{margin:6px 0 14px 1.1em;padding:0}}
    .analysis-llm-body li{{margin:5px 0;padding-left:2px}}
    .analysis-llm-body .analysis-code{{font-family:ui-monospace,Consolas,monospace;font-size:.88em;background:#1a1a2e;color:#b0bec5;padding:2px 7px;border-radius:4px;border:1px solid #2d2d4e}}
    .analysis-llm-body .analysis-bq{{margin:10px 0;padding:10px 14px;border-left:3px solid #4fc3f7;background:#12121f;border-radius:0 6px 6px 0;color:#b0bec5}}
    .analysis-llm-body .analysis-hr{{border:none;border-top:1px solid #37474f;margin:16px 0}}
    .analysis-llm-missing h2{{font-size:.95em;color:#ffb74d;margin-bottom:8px}}
    @media(max-width:960px){{.page-wrap{{flex-direction:column}}.llm-sidebar{{position:relative;top:0;max-height:none;width:100%;order:-1}}}}
    @media(max-width:600px){{header h1{{font-size:.95em}}}}
  </style>
</head>
<body>
<header>
  <h1>&#128200; Stock Signals</h1>
  <span class="ts" title="Zeitpunkt dieses Export-Laufs (nicht der Kursdatenstand)">Lauf: {today_str}</span>
</header>
<div class="page-wrap">
<main class="main-col">

  <div class="section">
    <h2>Letzte 30 Tage <span class="badge{recent_badge_cls}">{len(recent_signals)}</span></h2>
    {recent_html}
  </div>

  <div class="section">
    <h2>Model Info</h2>
    <div class="model-chips">
      <div class="chip">Threshold <strong>{best_threshold:.3f}</strong></div>
      <div class="chip">Tickers <strong>{df_s['ticker'].nunique()}</strong></div>
      <div class="chip" title="Nur Zeilen im FINAL-Kalender (nicht BASE/META/THRESHOLD); bei Pipeline-Regressor zusätzlich ohne dessen Train-Teil + Filter">Signale Website (OOS) <strong>{len(website_signals)}</strong></div>
      {_thr_chip}
      <div class="chip" title="Letzter Handelstag der Merkmals-/Kursmatrix in diesem Scoring-Lauf (Volllauf über df_features)">Scoring &amp; Features bis <strong>{_matrix_last_date}</strong></div>
      <div class="chip">Run <strong>{today_str}</strong></div>
    </div>
  </div>

  {_target_section}

  <div class="section">
    <details>
      <summary>KI-Analyse-Prompt</summary>
      <p class="prompt-lead">Vorgabe für ergänzende Einordnung der Meta-Hits (aktuellster Signaltag in den Daten; Schluss: welche Aktie — falls eine — am ehesten infrage kommt). Keine Anlageberatung.</p>
      <pre class="prompt-pre">{_ANALYSIS_PROMPT_ESC}</pre>
    </details>
  </div>

  {pr_section}

  <div class="section">
    <details>
      <summary>OOS-Signale (nicht in Classifier-Training) &mdash; {len(website_signals)} gesamt (max. 600 angezeigt)</summary>
      {hist_html}
    </details>
  </div>

</main>

<aside class="llm-sidebar" aria-label="KI-Analyse">
  <details class="llm-details" open>
    <summary>KI-Analyse <span class="llm-sum-hint">(ein-/ausklappen)</span></summary>
    <div class="llm-slot">
      __ANALYSIS_LLM_SECTION__
    </div>
  </details>
</aside>

</div>
</body>
</html>"""

    _anal_fallback = (
        '<div class="section analysis-llm-missing"><h2>KI-Antwort</h2>'
        '<p class="empty"><strong>Lokal:</strong> <code>.env</code> mit <code>GEMINI_API_KEY</code> (Projektroot). '
        "Dann <code>python scripts/run_website_analysis_gemini.py</code> — es muss "
        "<code>docs/analysis_llm_last.html</code> erscheinen. Anschließend <strong>diese Zelle (17)</strong> erneut ausführen.</p>"
        '<p class="empty"><strong>Daten:</strong> Vorher <code>data/master_daily_update.csv</code> erzeugen (holdout/build_holdout_signals_master, z. B. nach Holdout-Export).</p>'
        '<p class="empty"><strong>GitHub Pages:</strong> Ohne Commit von <code>docs/analysis_llm_last.html</code> und neuem '
        "<code>docs/index.html</code> bleibt die Analyse auf der Website leer — API-Keys liegen nicht im Repo.</p>"
        "</div>"
    )
    html = html.replace(
        "__ANALYSIS_LLM_SECTION__",
        _analysis_llm_section.strip() if _analysis_llm_section.strip() else _anal_fallback,
    )

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    (docs_dir / "signals.json").write_text(
        _json.dumps(
            {
                "generated": today_str,
                "threshold": float(best_threshold),
                "signals": website_signals,
                "signals_holdout_final": signals_holdout_final,
                "signals_all_timeline_including_in_sample_count": len(all_hist_signals),
                "note": "signals = signals_holdout_final: nur FINAL-Kalender-OOS für Classifier (TRAIN/THRESHOLD ausgeschlossen). Keine In-Sample-Signale für die Website.",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\ndocs/index.html   {len(html):,} bytes")
    print(
        f"docs/signals.json {len(website_signals)} OOS signals public "
        f"({len(all_hist_signals)} mit Schwelle über alle Zeiten nur intern gezählt)"
    )
    print("\nOpen docs/index.html in a browser to preview.")
    _dl = getattr(c, "DATA_LOAD_REPORT", None)
    if isinstance(_dl, dict):
        _missing = list(_dl.get("missing_tickers", []) or [])
        print("\n[Scoring-Ende] Ticker ohne ladbare Kursdaten:", flush=True)
        if not _missing:
            print("  keine", flush=True)
        else:
            print(f"  {len(_missing)} von {_dl.get('requested_count', '?')} Ticker:", flush=True)
            _reasons = _dl.get("failure_reasons", {}) or {}
            for t in _missing[:100]:
                print(f"  - {t}: {_reasons.get(t, 'unbekannt')}", flush=True)
            if len(_missing) > 100:
                print(f"  ... und {len(_missing) - 100} weitere", flush=True)
    try:
        from lib.signal_extra_filters import get_last_enrich_diagnostics as _get_enrich_diag

        _diag = _get_enrich_diag()
    except Exception:
        _diag = {}
    if isinstance(_diag, dict) and _diag.get("metrics"):
        print("\n[Scoring-Ende] Kennzahlen nicht berechnet (Top 15):", flush=True)
        for m in list(_diag.get("metrics", []))[:15]:
            if m.get("status") == "missing_column":
                print(f"  - {m.get('metric')}: Spalte fehlt | {m.get('reason')}", flush=True)
            else:
                print(
                    f"  - {m.get('metric')}: {m.get('missing_count')}/{_diag.get('row_count', 0)} "
                    f"({m.get('missing_pct', 0.0):.1f}%) | {m.get('reason')}",
                    flush=True,
                )
        _ss = _diag.get("short_metrics_source_counts") or {}
        if _ss:
            print(
                "  short_metrics_source: "
                + ", ".join(f"{k}={v}" for k, v in _ss.items()),
                flush=True,
            )

    _msg = (
        f"Daily signal update {today_str} "
        f"({len(website_signals)} OOS signals, "
        f"{len(recent_signals)} in last 30 days)"
    )
    _git_docs = [
        "docs/index.html",
        "docs/signals.json",
        "docs/website_analysis_prompt.txt",
        "docs/analysis_llm_last.html",
        "docs/analysis_llm_last.txt",
    ]
    _git_to_add = [p for p in _git_docs if Path(p).is_file()]
    try:
        if not _git_to_add:
            print("\nGit: keine der erwarteten docs-Dateien gefunden — Commit/Push übersprungen.")
        else:
            subprocess.run(["git", "add", "--"] + _git_to_add, check=True)
            result = subprocess.run(["git", "diff", "--cached", "--quiet"])
            if result.returncode != 0:
                subprocess.run(["git", "commit", "-m", _msg], check=True)
                subprocess.run(["git", "push"], check=True)
                print(f'\nGit: committed and pushed — "{_msg}"')
            else:
                print("\nGit: no changes to commit (docs already up to date).")
    except Exception as _e:
        print(f"\nGit push failed: {_e}")
