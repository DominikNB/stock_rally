"""Phase 17: Daily Scoring, Charts, HTML/JSON, optional Gemini & Git."""
from __future__ import annotations

import base64
import html as _html_std
import io
import json as _json
import os
import re
import shutil
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
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, precision_recall_curve

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.features import merge_news_shard_from_best_params

warnings.filterwarnings("ignore")

# Defaults wenn weder Laufzeit-``c`` noch Modul-``cfg`` das Attribut haben.
# Hinweis: ``getattr(obj, name, default)`` wertet ``default`` in Python **immer** aus —
# ``getattr(c, "X", cfg.X)`` scheitert also, sobald ``cfg.X`` fehlt, selbst wenn ``c.X`` existiert.
_DOCS_INDEX_HTML_MAX_BYTES_DEFAULT = 98 * 1024 * 1024
_DOCS_WEBSITE_MAX_CHART_SIGNALS_CAP_DEFAULT = 1200
_DOCS_WEBSITE_HTML_SHRINK_STEP_DEFAULT = 35


def _cfg_param_int(c_mod: Any, name: str, default: int) -> int:
    if hasattr(c_mod, name):
        return int(getattr(c_mod, name))
    if hasattr(cfg, name):
        return int(getattr(cfg, name))
    return int(default)


def _website_chart_storage_external(c_mod: Any) -> bool:
    mode = str(getattr(c_mod, "WEBSITE_CHART_STORAGE", "external")).strip().lower()
    return mode in ("external", "files", "file")


def _chart_file_slug(ticker: str, sig_date_str: str) -> str:
    safe_t = re.sub(r"[^\w.\-]+", "_", str(ticker).strip())[:80]
    safe_d = str(sig_date_str).strip()[:10]
    return f"{safe_t}_{safe_d}"


def _persist_chart_files(
    fig: Any,
    *,
    docs_dir: Path,
    slug: str,
    dpi: int,
    thumb_width_px: int,
    facecolor: str,
) -> tuple[str, str] | None:
    """Speichert Vollbild-PNG und Thumbnail; gibt relative Pfade ab docs/ zurück."""
    full_dir = docs_dir / "charts" / "full"
    thumb_dir = docs_dir / "charts" / "thumb"
    full_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"{slug}.png"
    try:
        fig.savefig(full_path, format="png", dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    except Exception:
        return None
    thumb_rel = f"charts/thumb/{slug}.png"
    full_rel = f"charts/full/{slug}.png"
    thumb_path = thumb_dir / f"{slug}.png"
    try:
        from PIL import Image

        with Image.open(full_path) as im:
            w, h = im.size
            tw = min(int(thumb_width_px), w)
            th = max(1, int(round(h * tw / max(w, 1))))
            im.resize((tw, th), Image.Resampling.LANCZOS).save(
                thumb_path, format="PNG", optimize=True
            )
    except Exception:
        shutil.copy2(full_path, thumb_path)
    return thumb_rel, full_rel


def _prune_stale_chart_files(docs_dir: Path, keep_slugs: set[str]) -> int:
    removed = 0
    for sub in ("full", "thumb"):
        d = docs_dir / "charts" / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            if p.stem not in keep_slugs:
                p.unlink(missing_ok=True)
                removed += 1
    return removed


def _sort_website_signals_newest_first(sigs: list[dict[str, Any]]) -> None:
    """Nach Signaldatum absteigend; bei gleichem Kalendertag höhere prob zuerst."""

    def _key(s: dict[str, Any]) -> tuple[int, float]:
        ts = int(pd.Timestamp(s["date"]).value)
        prob = float(s.get("prob", 0.0) or 0.0)
        return (ts, prob)

    sigs.sort(key=_key, reverse=True)


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
        except ImportError as _e_imp:
            print(
                f"[Gemini] google-genai nicht verfügbar ({_e_imp}) — Smoke-Test übersprungen, Website-Export läuft weiter.",
                flush=True,
            )
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
    _sent_pad = float(getattr(c, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    _miss_fc = [x for x in c.FEAT_COLS if x not in df_s.columns]
    if _miss_fc:
        _head = ", ".join(_miss_fc[:5])
        _more = f" (+{len(_miss_fc) - 5} weitere)" if len(_miss_fc) > 5 else ""
        print(
            f"[Phase17] {len(_miss_fc)} FEAT_COLS-Spalte(n) fehlen nach News-Merge — "
            f"mit Sentinel {_sent_pad:g} auffüllen (z. B. {_head}{_more}).",
            flush=True,
        )
        for _c in _miss_fc:
            df_s[_c] = np.float32(_sent_pad)
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
    _cal = getattr(c, "meta_proba_calibrator", None)
    if isinstance(_cal, dict):
        _m = str(_cal.get("method", "")).strip().lower()
        _mdl = _cal.get("model")
        try:
            p = np.clip(np.asarray(probs_all, dtype=np.float64), 1e-6, 1.0 - 1e-6)
            if _m == "sigmoid" and _mdl is not None:
                lg = np.log(p / (1.0 - p)).reshape(-1, 1)
                probs_all = _mdl.predict_proba(lg)[:, 1]
            elif _m == "isotonic" and _mdl is not None:
                probs_all = np.asarray(_mdl.predict(p), dtype=np.float64)
        except Exception as _e_cal:
            print(f"Warnung: Meta-Proba-Kalibrierung im Scoring fehlgeschlagen ({_e_cal!r}).", flush=True)
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
        "signal_max_vol_stress_z": getattr(c, "SIGNAL_MAX_VOL_STRESS_Z", None),
        "signal_min_blue_sky_volume_z": getattr(c, "SIGNAL_MIN_BLUE_SKY_VOLUME_Z", None),
        "mult_final_threshold_1": getattr(c, "MULT_FINAL_THRESHOLD_1", 1.0),
        "mult_final_threshold_2": getattr(c, "MULT_FINAL_THRESHOLD_2", 1.0),
        "mult_final_threshold_3": getattr(c, "MULT_FINAL_THRESHOLD_3", 1.0),
        "dyn_vvix_trigger": getattr(c, "DYN_VVIX_TRIGGER", 8.2),
        "dyn_rsi_trigger": getattr(c, "DYN_RSI_TRIGGER", 75.0),
        "dyn_bb_pband_trigger": getattr(c, "DYN_BB_PBAND_TRIGGER", 1.02),
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
    _sort_website_signals_newest_first(signals_holdout_final)
    _sort_website_signals_newest_first(all_hist_signals)
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
            from lib.vix_regime_ampel import ampel_fields_from_vix

            signals_holdout_final = []
            for _, _r in _exported_ho.iterrows():
                _vix_raw = _r.get("regime_vix_level")
                _amp = ampel_fields_from_vix(
                    None
                    if _vix_raw is None or (isinstance(_vix_raw, float) and _vix_raw != _vix_raw)
                    else float(_vix_raw)
                )
                signals_holdout_final.append(
                    {
                        "ticker": str(_r["ticker"]),
                        "company": str(_r.get("company", _r["ticker"])),
                        "sector": str(_r.get("sector", "—")),
                        **{k: str(_r.get(k, "")) for k in _cls_keys},
                        "date": str(_r["Date"])[:10],
                        "prob": float(_r["prob"]),
                        **_amp,
                    }
                )
            _sort_website_signals_newest_first(signals_holdout_final)
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
    _sort_website_signals_newest_first(website_signals)
    print(
        f"Website/OOS-Export: {len(website_signals)} Signale "
        f"(intern für Diagnose: {len(all_hist_signals)} über alle Zeiten mit Schwelle+Filter).",
        flush=True,
    )

    recent_cutoff = (pd.Timestamp(c.END_DATE) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    recent_signals = [s for s in website_signals if s["date"] >= recent_cutoff]
    print(f"{len(recent_signals)} OOS-Signale in den letzten 30 Tagen (Website).")
    latest_signal_date = website_signals[0]["date"] if website_signals else None
    latest_day_signals = [s for s in website_signals if s["date"] == latest_signal_date] if latest_signal_date else []

    def _minmax_norm(vals: list[float], *, invert: bool = False) -> list[float]:
        if not vals:
            return []
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).any():
            return [0.0 for _ in vals]
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi - lo <= 1e-12:
            base = np.full_like(arr, 0.5, dtype=np.float64)
        else:
            base = (arr - lo) / (hi - lo)
        if invert:
            base = 1.0 - base
        return [float(x) for x in base]

    ranking_rows: list[dict[str, Any]] = []
    if latest_day_signals:
        rsi_w = int(getattr(c, "rsi_w", c.SEED_PARAMS.get("rsi_window", 14)))
        rsi_col = f"rsi_{rsi_w}"
        rsi_col_alt = f"rsi_{rsi_w}d"
        rsi_cap = getattr(c, "SIGNAL_MAX_RSI", None)
        for sig in latest_day_signals:
            t = str(sig["ticker"])
            d = pd.Timestamp(sig["date"]).normalize()
            sub = df_s[df_s["ticker"].astype(str).str.strip() == t].copy()
            if sub.empty:
                continue
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.normalize()
            sub = sub.sort_values("Date").reset_index(drop=True)
            row = sub[sub["Date"] == d]
            if row.empty:
                continue
            ridx = int(row.index[0])
            close_hist = pd.to_numeric(sub.loc[max(0, ridx - 5) : ridx, "close"], errors="coerce").to_numpy(dtype=np.float64)
            if close_hist.size >= 3:
                rets = close_hist[1:] / np.where(close_hist[:-1] != 0.0, close_hist[:-1], np.nan) - 1.0
                stability_std = float(np.nanstd(rets))
            else:
                stability_std = np.nan
            if rsi_col in row.columns:
                rsi_val = float(pd.to_numeric(row[rsi_col], errors="coerce").iloc[0])
            elif rsi_col_alt in row.columns:
                # Backward-compatible fallback in case older datasets used "rsi_<w>d".
                rsi_val = float(pd.to_numeric(row[rsi_col_alt], errors="coerce").iloc[0])
            else:
                rsi_val = np.nan
            if rsi_cap is None or not np.isfinite(rsi_val):
                safety_buffer = 0.0
            else:
                safety_buffer = float(rsi_cap) - float(rsi_val)
            vol_z = (
                float(pd.to_numeric(row["volume_zscore"], errors="coerce").iloc[0])
                if "volume_zscore" in row.columns
                else 0.0
            )
            ranking_rows.append(
                {
                    "ticker": t,
                    "company": sig.get("company", t),
                    "date": sig["date"],
                    "prob": float(sig.get("prob", 0.0)),
                    "stability_std": stability_std if np.isfinite(stability_std) else 1.0,
                    "safety_buffer": float(safety_buffer),
                    "volume_z": float(vol_z) if np.isfinite(vol_z) else 0.0,
                }
            )
    if ranking_rows:
        st_n = _minmax_norm([r["stability_std"] for r in ranking_rows], invert=True)  # niedrige Std ist besser
        sf_n = _minmax_norm([r["safety_buffer"] for r in ranking_rows], invert=False)
        vz_n = _minmax_norm([r["volume_z"] for r in ranking_rows], invert=False)
        for i, r in enumerate(ranking_rows):
            r["rank_score"] = 0.50 * st_n[i] + 0.30 * sf_n[i] + 0.20 * vz_n[i]
        ranking_rows.sort(key=lambda x: x["rank_score"], reverse=True)
        for i, r in enumerate(ranking_rows, start=1):
            r["rank"] = i

    if ranking_rows:
        _rows_html = []
        for r in ranking_rows:
            _rows_html.append(
                "<tr>"
                f"<td>{int(r['rank'])}</td>"
                f"<td>{_html_std.escape(str(r['ticker']))}</td>"
                f"<td>{_html_std.escape(str(r['company']))}</td>"
                f"<td>{float(r['prob']):.3f}</td>"
                f"<td>{float(r['rank_score']):.3f}</td>"
                f"<td>{float(r['stability_std']):.4f}</td>"
                f"<td>{float(r['safety_buffer']):.2f}</td>"
                f"<td>{float(r['volume_z']):.2f}</td>"
                "</tr>"
            )
        ranking_html = (
            f'<div class="section"><h2>Ranking aktuellster Signale '
            f'<span class="badge">{len(ranking_rows)}</span></h2>'
            f'<p class="prompt-lead">Datum: <strong>{_html_std.escape(str(latest_signal_date))}</strong> '
            f'| Score = 0.50·Stabilität + 0.30·RSI-Puffer + 0.20·Volumen-Z.</p>'
            '<div style="overflow-x:auto">'
            '<table style="width:100%;border-collapse:collapse;font-size:.82em">'
            '<thead><tr>'
            '<th style="text-align:left;padding:6px;border-bottom:1px solid #2d2d4e">#</th>'
            '<th style="text-align:left;padding:6px;border-bottom:1px solid #2d2d4e">Ticker</th>'
            '<th style="text-align:left;padding:6px;border-bottom:1px solid #2d2d4e">Unternehmen</th>'
            '<th style="text-align:right;padding:6px;border-bottom:1px solid #2d2d4e">Prob</th>'
            '<th style="text-align:right;padding:6px;border-bottom:1px solid #2d2d4e">Score</th>'
            '<th style="text-align:right;padding:6px;border-bottom:1px solid #2d2d4e">Std(5d)</th>'
            '<th style="text-align:right;padding:6px;border-bottom:1px solid #2d2d4e">RSI-Puffer</th>'
            '<th style="text-align:right;padding:6px;border-bottom:1px solid #2d2d4e">Vol-Z</th>'
            '</tr></thead><tbody>'
            + "".join(_rows_html)
            + "</tbody></table></div></div>"
        )
    else:
        ranking_html = (
            '<div class="section"><h2>Ranking aktuellster Signale</h2>'
            '<p class="empty">Kein Ranking verfügbar (keine oder nur unvollständige aktuelle OOS-Signale).</p></div>'
        )

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

    pr_section = (
        f'<div class="section"><h2>Model Quality &#8212; Precision-Recall</h2>'
        f'<img src="data:image/png;base64,{pr_b64}" alt="PR" style="max-width:100%;border-radius:6px"></div>'
        if pr_b64
        else ""
    )

    _chart_yf_failures: list[tuple[Any, ...]] = []

    def _yf_ohlcv_rows_from_df(df_hist):
        rows = []
        if df_hist is None or len(df_hist) == 0:
            return rows
        _candidates = {
            "open": ("Open", "open"),
            "high": ("High", "high"),
            "low": ("Low", "low"),
            "close": ("Close", "close"),
            "volume": ("Volume", "volume"),
        }
        for _ti, _row in df_hist.iterrows():
            _d = pd.Timestamp(pd.Timestamp(_ti).date())
            _rec = {"Date": _d}
            for _dst, _srcs in _candidates.items():
                _val = np.nan
                for _src in _srcs:
                    if _src in df_hist.columns:
                        _val = float(pd.to_numeric(_row.get(_src), errors="coerce"))
                        break
                _rec[_dst] = _val
            rows.append(_rec)
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
                if _hist is not None and len(_hist):
                    _rows = _yf_ohlcv_rows_from_df(_hist)
                if not _rows:
                    _ext = yf.download(ticker, start=_d0, end=_d1, progress=False, threads=False)
                    if _ext is not None and len(_ext) > 0:
                        if isinstance(_ext.columns, pd.MultiIndex):
                            _ext.columns = [str(c[0]) for c in _ext.columns]
                        _rows = _yf_ohlcv_rows_from_df(_ext)
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
        _base_cols = ["Date", "open", "high", "low", "close", "volume", "volume_zscore"]
        _left = win[[c for c in _base_cols if c in win.columns]].copy()
        _w = pd.concat([_left, pd.DataFrame(_rows)], ignore_index=True)
        _w["Date"] = pd.to_datetime(_w["Date"]).dt.normalize()
        _w = _w.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
        return _w[(_w["Date"] >= win_lo) & (_w["Date"] <= win_hi)].copy()

    _df_raw_all = getattr(c, "df_raw", None)
    _ohlcv_cache: dict[str, pd.DataFrame] = {}

    def _get_base_ohlcv_for_ticker(ticker: str) -> pd.DataFrame:
        if ticker in _ohlcv_cache:
            return _ohlcv_cache[ticker]
        base = pd.DataFrame()
        if isinstance(_df_raw_all, pd.DataFrame) and len(_df_raw_all) > 0:
            cand = _df_raw_all[_df_raw_all["ticker"].astype(str).str.strip() == str(ticker)].copy()
            if len(cand) > 0:
                keep = [c for c in ("Date", "open", "high", "low", "close", "volume") if c in cand.columns]
                base = cand[keep].copy()
        if len(base) == 0:
            # Fallback: aus Feature-Matrix (falls df_raw in diesem Lauf nicht am cfg hängt).
            cand = df_s[df_s["ticker"].astype(str).str.strip() == str(ticker)].copy()
            keep = [c for c in ("Date", "open", "high", "low", "close", "volume", "volume_zscore") if c in cand.columns]
            base = cand[keep].copy() if keep else pd.DataFrame()
        if len(base) > 0 and "Date" in base.columns:
            base["Date"] = pd.to_datetime(base["Date"]).dt.normalize()
            base = base.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        _ohlcv_cache[ticker] = base
        return base

    def _make_chart(ticker, sig_date_str):
        def _norm_ts(x) -> pd.Timestamp:
            t = pd.Timestamp(x)
            try:
                if t.tz is not None:
                    t = t.tz_convert(None)
            except Exception:
                try:
                    t = t.tz_localize(None)
                except Exception:
                    pass
            return t.normalize()

        sub = _get_base_ohlcv_for_ticker(str(ticker))
        if "close" not in sub.columns or len(sub) < 5:
            return None
        sig_ts = pd.Timestamp(sig_date_str).normalize()
        _chart_layout = str(getattr(c, "WEBSITE_CHART_LAYOUT", "compact")).strip().lower()
        _chart_days = int(getattr(c, "WEBSITE_CHART_DAYS_EACH_SIDE", 21 if _chart_layout == "compact" else 30))
        _chart_dpi = _cfg_param_int(c, "WEBSITE_CHART_DPI", 72 if _chart_layout == "compact" else 95)
        win_lo = sig_ts - pd.Timedelta(days=_chart_days)
        win_hi = sig_ts + pd.Timedelta(days=_chart_days)
        # Warmup-Historie für stabilere Indikatoren (RSI/BB/EMA) im sichtbaren Fenster.
        calc_lo = win_lo - pd.Timedelta(days=90)
        win = sub[(sub["Date"] >= calc_lo) & (sub["Date"] <= win_hi)].copy()
        if len(win) < 5:
            return None
        win = _extend_chart_close_yfinance(ticker, win, calc_lo, win_hi)
        if "close" not in win.columns:
            return None
        win["close"] = pd.to_numeric(win["close"], errors="coerce")
        # Robust fallback: bei fehlendem OHLC (z. B. wenn nur feature-close vorhanden ist)
        # Candles aus close rekonstruieren, damit Charts nicht komplett ausfallen.
        if "open" not in win.columns:
            win["open"] = np.nan
        if "high" not in win.columns:
            win["high"] = np.nan
        if "low" not in win.columns:
            win["low"] = np.nan
        if "volume" not in win.columns:
            win["volume"] = 0.0
        for _col in ("open", "high", "low", "volume"):
            win[_col] = pd.to_numeric(win[_col], errors="coerce")
        prev_close = win["close"].shift(1)
        win["open"] = win["open"].fillna(prev_close).fillna(win["close"])
        _oc_max = np.maximum(win["open"].to_numpy(dtype=np.float64), win["close"].to_numpy(dtype=np.float64))
        _oc_min = np.minimum(win["open"].to_numpy(dtype=np.float64), win["close"].to_numpy(dtype=np.float64))
        win["high"] = win["high"].fillna(pd.Series(_oc_max, index=win.index))
        win["low"] = win["low"].fillna(pd.Series(_oc_min, index=win.index))
        win["volume"] = win["volume"].fillna(0.0)
        win = win.dropna(subset=["close", "open", "high", "low"]).copy()
        if len(win) < 5:
            return None
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
            dt = pd.to_datetime(win["Date"])
            x = mdates.date2num(pd.DatetimeIndex(dt).to_pydatetime())
            close = win["close"].astype(float)
            open_ = win["open"].astype(float)
            high = win["high"].astype(float)
            low = win["low"].astype(float)
            volume = pd.to_numeric(win["volume"], errors="coerce").fillna(0.0).astype(float)

            ema20 = close.ewm(span=20, adjust=False, min_periods=1).mean()
            bb_mid = close.rolling(20, min_periods=5).mean()
            bb_std = close.rolling(20, min_periods=5).std()
            bb_up = bb_mid + 2.0 * bb_std
            bb_lo = bb_mid - 2.0 * bb_std

            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi14 = 100.0 - (100.0 / (1.0 + rs))

            if "volume_zscore" in win.columns:
                vol_z = pd.to_numeric(win["volume_zscore"], errors="coerce")
            else:
                v_mean = volume.rolling(20, min_periods=5).mean()
                v_std = volume.rolling(20, min_periods=5).std()
                vol_z = (volume - v_mean) / (v_std + 1e-10)
            vol_z = vol_z.fillna(0.0)

            bb_width = (bb_up - bb_lo) / (bb_mid.abs() + 1e-10)

            ax_bw = None
            ax_vol = None
            ax_vol_z = None
            if _chart_layout == "full":
                fig, (ax_price, ax_bw, ax_rsi, ax_vol) = plt.subplots(
                    4,
                    1,
                    figsize=(9.4, 7.3),
                    sharex=True,
                    gridspec_kw={"height_ratios": [3.3, 1.0, 1.4, 1.6]},
                )
            else:
                fig, (ax_price, ax_rsi) = plt.subplots(
                    2,
                    1,
                    figsize=(6.6, 3.35),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2.6, 1.0]},
                )

            candle_w = 0.6
            up_color = "#66bb6a"
            dn_color = "#ef5350"
            for i in range(len(x)):
                o, h, l, c_ = float(open_.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
                col = up_color if c_ >= o else dn_color
                ax_price.vlines(x[i], l, h, color=col, linewidth=0.9, alpha=0.95, zorder=2)
                body_y = min(o, c_)
                body_h = max(abs(c_ - o), 1e-6)
                ax_price.add_patch(
                    Rectangle(
                        (x[i] - candle_w / 2.0, body_y),
                        candle_w,
                        body_h,
                        facecolor=col,
                        edgecolor=col,
                        alpha=0.95,
                        linewidth=0.7,
                        zorder=3,
                    )
                )
            # Zusatz zur Candle-Ansicht: durchgehende Kurslinie wie im alten Plot-Layout.
            ax_price.plot(dt, close, color="#00e676", lw=1.1, alpha=0.9, label="Close", zorder=4)
            ax_price.plot(dt, ema20, color="#42a5f5", lw=1.2, label="EMA 20", zorder=4)
            ax_price.plot(dt, bb_up, color="#ab47bc", lw=1.0, ls="--", label="BB Upper (20,2)", zorder=3)
            ax_price.plot(dt, bb_mid, color="#8d6e63", lw=0.9, ls="-.", label="BB Mid (20)", zorder=3)
            ax_price.plot(dt, bb_lo, color="#ab47bc", lw=1.0, ls="--", label="BB Lower (20,2)", zorder=3)
            ax_price.fill_between(dt, bb_lo.to_numpy(), bb_up.to_numpy(), color="#ab47bc", alpha=0.08, zorder=1)

            if ax_bw is not None:
                ax_bw.plot(dt, bb_width, color="#ba68c8", lw=1.2, label="BB Width (20,2)")
                ax_bw.axhline(float(np.nanmedian(bb_width)), color="#8e24aa", lw=0.9, ls="--", alpha=0.7)

            ax_rsi.plot(dt, rsi14, color="#ffb74d", lw=1.2, label="RSI 14")
            for _lvl, _col, _ls in ((70, "#ef5350", "--"), (30, "#66bb6a", "--"), (75, "#ff7043", ":"), (25, "#26a69a", ":")):
                ax_rsi.axhline(_lvl, color=_col, lw=0.9, ls=_ls, alpha=0.95)
            ax_rsi.set_ylim(0, 100)

            if ax_vol is not None:
                bar_colors = np.where(close.to_numpy() >= open_.to_numpy(), up_color, dn_color)
                ax_vol.bar(dt, volume.to_numpy(), width=0.8, color=bar_colors, alpha=0.72, label="Volume")
                ax_vol_z = ax_vol.twinx()
                ax_vol_z.plot(dt, vol_z.to_numpy(), color="#80cbc4", lw=1.2, label="Volume Z-Score")
                ax_vol_z.axhline(0.0, color="#607d8b", lw=0.9, ls="--", alpha=0.8)

            ax_price.axvline(sig_ts, color="#66bb6a", lw=1.8, ls="--", zorder=5, label="Datenstand")
            if ax_bw is not None:
                ax_bw.axvline(sig_ts, color="#66bb6a", lw=1.3, ls="--", zorder=5)
            ax_rsi.axvline(sig_ts, color="#66bb6a", lw=1.4, ls="--", zorder=5)
            if ax_vol is not None:
                ax_vol.axvline(sig_ts, color="#66bb6a", lw=1.4, ls="--", zorder=5)
            today_ts = _norm_ts(datetime.today().date())
            _last_dt = _norm_ts(dt.max())
            _view_hi = min(_norm_ts(win_hi), _last_dt)
            _view_hi = max(_view_hi, sig_ts + pd.Timedelta(days=2))
            if win_lo <= today_ts <= _view_hi:
                ax_price.axvline(today_ts, color="#ffa726", lw=1.4, ls=":", zorder=5, label="Heute")
                if ax_bw is not None:
                    ax_bw.axvline(today_ts, color="#ffa726", lw=1.1, ls=":", zorder=5)
                ax_rsi.axvline(today_ts, color="#ffa726", lw=1.1, ls=":", zorder=5)
                if ax_vol is not None:
                    ax_vol.axvline(today_ts, color="#ffa726", lw=1.1, ls=":", zorder=5)

            # Nach dem Signal nur so viel Platz wie echte verfügbare Kurstage.
            ax_price.set_xlim(win_lo, _view_hi)
            h1, l1 = ax_price.get_legend_handles_labels()
            ax_price.legend(
                h1,
                l1,
                fontsize=7,
                loc="upper left",
                facecolor="#1a1a2e",
                edgecolor="#2d2d4e",
                labelcolor="#e0e0e0",
            )
            _ds = pd.Timestamp(sig_date_str).strftime("%d.%m.%Y")
            _co = COMPANY_NAMES.get(ticker, ticker)
            if _chart_layout == "compact":
                ax_price.set_title(
                    f"{ticker} — {_co}  ·  Daten bis {_ds}",
                    fontsize=7,
                    color="#81d4fa",
                )
            else:
                ax_price.set_title(
                    f"{ticker} — {_co}\n"
                    f"Datenstand Modell: bis einschl. {_ds} (grüner Strich); rechts nur Kurs (Anzeige)",
                    fontsize=8,
                    color="#81d4fa",
                )
            ax_price.set_ylabel("Preis", color="#90a4ae", fontsize=7 if _chart_layout == "compact" else 8)
            if ax_bw is not None:
                ax_bw.set_ylabel("BB Width", color="#ba68c8", fontsize=8)
            ax_rsi.set_ylabel("RSI", color="#90a4ae", fontsize=7 if _chart_layout == "compact" else 8)
            if ax_vol is not None:
                ax_vol.set_ylabel("Volume", color="#90a4ae", fontsize=8)
                ax_vol_z.set_ylabel("Vol Z", color="#80cbc4", fontsize=8)
            if ax_bw is not None:
                ax_bw.legend(fontsize=7, loc="upper left", frameon=False, labelcolor="#ce93d8")
            ax_rsi.legend(fontsize=6 if _chart_layout == "compact" else 7, loc="upper left", frameon=False, labelcolor="#ffcc80")
            if ax_vol is not None and ax_vol_z is not None:
                h3, l3 = ax_vol.get_legend_handles_labels()
                h4, l4 = ax_vol_z.get_legend_handles_labels()
                ax_vol.legend(h3 + h4, l3 + l4, fontsize=7, loc="upper left", frameon=False, labelcolor="#e0e0e0")

            _xaxis_ax = ax_vol if ax_vol is not None else ax_rsi
            _xaxis_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            _xaxis_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
            plt.setp(_xaxis_ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=6 if _chart_layout == "compact" else 7)
            _chart_axes = [ax_price, ax_rsi]
            if ax_bw is not None:
                _chart_axes.insert(1, ax_bw)
            if ax_vol is not None:
                _chart_axes.append(ax_vol)
            for _ax in _chart_axes:
                _ax.tick_params(colors="#90a4ae", labelsize=6 if _chart_layout == "compact" else 7)
                _ax.grid(True, alpha=0.18)
                _ax.set_facecolor("#0d1117")
                for sp in _ax.spines.values():
                    sp.set_edgecolor("#2d2d4e")
            if ax_vol_z is not None:
                ax_vol_z.tick_params(colors="#80cbc4", labelsize=7)
                ax_vol_z.spines["right"].set_edgecolor("#2d2d4e")
            fig.patch.set_facecolor("#1a1a2e")
            plt.tight_layout(pad=0.8)
            _fc = fig.get_facecolor()
            if _website_chart_storage_external(c):
                _full_dpi = _cfg_param_int(c, "WEBSITE_CHART_FULL_DPI", max(_chart_dpi, 96))
                _thumb_w = _cfg_param_int(c, "WEBSITE_CHART_THUMB_WIDTH_PX", 520)
                paths = _persist_chart_files(
                    fig,
                    docs_dir=docs_dir,
                    slug=_chart_file_slug(ticker, sig_date_str),
                    dpi=_full_dpi,
                    thumb_width_px=_thumb_w,
                    facecolor=_fc,
                )
                plt.close()
                return paths
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=_chart_dpi, bbox_inches="tight", facecolor=_fc)
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception as _e_chart:
            print(
                f"Chart failed for {ticker} @ {sig_date_str}: {type(_e_chart).__name__}: {_e_chart}",
                flush=True,
            )
            plt.close("all")
            return None

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
    _matrix_last_date = str(pd.to_datetime(df_s["Date"]).max().date())
    if _gemini_key and _script_gemini.is_file():
        try:
            print("Gemini: KI-Analyse (CSV-Upload + Google Search) …", flush=True)
            _env_llm = os.environ.copy()
            _env_llm["ANALYSIS_EXPECT_SIGNAL_DATE"] = _matrix_last_date
            _r = subprocess.run(
                [sys.executable, str(_script_gemini)],
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=600,
                env=_env_llm,
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

    _html_cap = _cfg_param_int(c, "DOCS_INDEX_HTML_MAX_BYTES", _DOCS_INDEX_HTML_MAX_BYTES_DEFAULT)
    _k_start = min(
        len(website_signals),
        _cfg_param_int(c, "DOCS_WEBSITE_MAX_CHART_SIGNALS_CAP", _DOCS_WEBSITE_MAX_CHART_SIGNALS_CAP_DEFAULT),
    )
    _k_step = max(1, _cfg_param_int(c, "DOCS_WEBSITE_HTML_SHRINK_STEP", _DOCS_WEBSITE_HTML_SHRINK_STEP_DEFAULT))

    from lib.vix_regime_ampel import (
        vix_ampel_css_block,
        vix_ampel_html_span,
        vix_ampel_panel_html,
    )

    _ampel_css_esc = vix_ampel_css_block().replace("{", "{{").replace("}", "}}")
    _vix_ampel_panel = vix_ampel_panel_html()

    _charts_external = _website_chart_storage_external(c)

    def _signal_card_site(s: dict, _cch: dict, *, _recent_cutoff: str) -> str:
        key = (s["ticker"], s["date"])
        ch = _cch.get(key)
        has_chart = bool(ch)
        bar = int(s["prob"] * 100)
        _ampel_html = vix_ampel_html_span(s)
        _yf_note = (
            '<p class="yf-hint">Kursnachzug (yfinance) fehlgeschlagen — Chart endet am letzten Tag der '
            "Feature-Matrix; beim nächsten Lauf kann es wieder klappen.</p>"
            if has_chart and s["ticker"] in _yf_failed_tickers
            else ""
        )
        if isinstance(ch, tuple) and len(ch) == 2:
            thumb_rel, full_rel = ch
            _alt = _html_std.escape(f'{s["ticker"]} {s["date"]}')
            chart_html = (
                f'<a class="sig-chart-link" href="{_html_std.escape(full_rel)}" target="_blank" '
                f'rel="noopener noreferrer">'
                f'<img class="sig-chart-thumb" src="{_html_std.escape(thumb_rel)}" alt="{_alt}" loading="lazy">'
                f"</a>"
                '<p class="sig-chart-tap">Tippen für Vollbild (im Browser vergrößerbar)</p>'
            )
        elif isinstance(ch, str) and ch:
            chart_html = (
                f'<img src="data:image/png;base64,{ch}" alt="{_html_std.escape(s["ticker"])}" loading="lazy">'
            )
        else:
            chart_html = ""
        chart_note = (
            f'<p class="sig-chart-note" style="font-size:0.72em;color:#546e7a;margin-top:6px;line-height:1.35">'
            f'Merkmale und Kursbasis für dieses Signal: bis einschließlich <strong>{s["date"]}</strong> '
            f"(nicht der Zeitpunkt der Berechnung); rechts vom grünen Strich nur nachgelagerter Kurs "
            f"(Anzeige) — kein Look-ahead fürs Modell.</p>"
            if has_chart
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
        _is_recent = s["date"] >= _recent_cutoff
        _recent_cls = " sig-card--recent" if _is_recent else ""
        _recent_tag = (
            '<span class="sig-recent-tag">≤30 Tage</span>' if _is_recent else ""
        )
        return f"""
      <div class="sig-card{_recent_cls}">
        <div class="sig-head">
          <span class="sig-ticker">{s['ticker']}{_recent_tag}</span>
          <span class="sig-company">{s['company']}</span>
          <span class="sig-sector"{_badge_title_attr}>{_html_std.escape(_badge_label)}</span>
          <span class="sig-date" title="Kurs- und Merkmalsdaten bis einschließlich diesem Tag — nicht der Laufzeitpunkt der Berechnung"><span class="sig-date-pre">Daten bis</span> {s['date']}</span>
          {_ampel_html}
          <div class="score-bar-bg"><div class="score-bar" style="width:{bar}%">{s['prob']:.3f}</div></div>
        </div>
        {_gics_html}
        {_yf_note}
        {chart_html}
        {chart_note}
      </div>"""

    html_chart_cap_k = 0
    html = ""
    _chart_k = _k_start
    while _chart_k >= 0:
        _site_chart_layout = str(getattr(c, "WEBSITE_CHART_LAYOUT", "compact")).strip().lower()
        _storage_label = "external" if _charts_external else "inline"
        print(
            f"Generating charts (storage={_storage_label}, layout={_site_chart_layout}, "
            f"dpi={_cfg_param_int(c, 'WEBSITE_CHART_DPI', 72)}, "
            f"bis zu {_chart_k} neueste OOS-Signale"
            + (
                ") …"
                if _charts_external
                else f"; HTML-Ziel ≤ {_html_cap // 1024 // 1024} MiB) …"
            ),
            flush=True,
        )
        _chart_yf_failures.clear()
        chart_cache: dict = {}
        for s in website_signals[:_chart_k]:
            key = (s["ticker"], s["date"])
            if key not in chart_cache:
                ch_out = _make_chart(s["ticker"], s["date"])
                if ch_out:
                    chart_cache[key] = ch_out
        if _charts_external and chart_cache:
            _keep_slugs = {_chart_file_slug(t, d) for (t, d) in chart_cache}
            _rm = _prune_stale_chart_files(docs_dir, _keep_slugs)
            if _rm:
                print(f"  {len(chart_cache)} charts; {_rm} veraltete PNG(s) entfernt.", flush=True)
            else:
                print(f"  {len(chart_cache)} charts in docs/charts/.", flush=True)
        else:
            print(f"  {len(chart_cache)} charts generated.", flush=True)
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
        chart_signals = website_signals[:_chart_k]
        recent_slice = [s for s in chart_signals if s["date"] >= recent_cutoff]
        n_recent_site = len(recent_slice)

        def _cards_grid(signals: list) -> str:
            if not signals:
                return ""
            return '<div class="signals-grid">' + "".join(
                _signal_card_site(s, chart_cache, _recent_cutoff=recent_cutoff) for s in signals
            ) + "</div>"

        all_signals_html = (
            _cards_grid(chart_signals)
            or '<p class="empty">Keine OOS-Signale mit Chart.</p>'
        )
        recent_badge_cls = " zero" if not recent_slice else ""
        _chart_chip_title = (
            "Externe PNG unter docs/charts/ — index.html bleibt unter 100 MiB"
            if _charts_external
            else "Eingebettete Base64-Charts; K wird bei Bedarf reduziert (GitHub 100-MiB-Limit)"
        )
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
        .page-wrap{{display:flex;align-items:flex-start;gap:18px;max-width:1480px;margin:0 auto;padding:12px 16px}}
        .main-col{{flex:1;min-width:0;max-width:1100px}}
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
        .signals-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 12px}}
        .sig-card{{background:#0d1117;border-radius:8px;padding:10px;margin-bottom:0}}
        .sig-head{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}}
        .sig-ticker{{font-weight:700;color:#81d4fa;font-size:.95em;min-width:70px}}
        .sig-company{{color:#90a4ae;font-size:.82em;flex:1;min-width:100px}}
        .sig-sector{{background:#1e2a3a;color:#64b5f6;font-size:.72em;padding:2px 7px;border-radius:10px;white-space:nowrap;align-self:center}}
        .sig-gics{{font-size:.74em;color:#90a4ae;line-height:1.45;margin:6px 0 0;padding:0 2px;max-width:100%}}
        .sig-date{{color:#546e7a;font-size:.78em;white-space:nowrap}}
        .sig-date-pre{{font-size:.68em;color:#78909c;margin-right:5px}}
        {_ampel_css_esc}
        .score-bar-bg{{background:#1a1a2e;border-radius:4px;overflow:hidden;min-width:70px;max-width:110px}}
        .score-bar{{background:#4caf50;padding:2px 6px;color:#fff;font-size:.75em;white-space:nowrap;border-radius:4px}}
        img{{max-width:100%;max-height:300px;width:100%;height:auto;object-fit:contain;border-radius:5px;display:block}}
        a.sig-chart-link{{display:block;text-decoration:none}}
        img.sig-chart-thumb{{cursor:pointer}}
        .sig-chart-tap{{font-size:.68em;color:#78909c;margin:4px 0 0;line-height:1.3}}
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
        @media(max-width:960px){{.page-wrap{{flex-direction:column}}.llm-sidebar{{position:relative;top:0;max-height:none;width:100%;order:-1}}.signals-grid{{grid-template-columns:1fr}}}}
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

      {ranking_html}

      <div class="section vix-regime-section">
        <h2>VIX-Regime (Ampel)</h2>
        {_vix_ampel_panel}
      </div>

      <div class="section signals-all-section">
        <h2>Alle OOS-Signale mit Chart <span class="badge">{len(chart_cache)}</span></h2>
        <p class="section-lead">
          {len(chart_cache)} von {len(website_signals)} FINAL-OOS-Signalen (neueste zuerst).
          <span class="sig-recent-tag">≤30 Tage</span> = {n_recent_site} Signale in den letzten 30 Kalendertagen
          (grüner Kartenrand). Tippen auf Chart = Vollbild.
        </p>
        {all_signals_html}
      </div>

      <div class="section">
        <h2>Model Info</h2>
        <div class="model-chips">
          <div class="chip">Threshold <strong>{best_threshold:.3f}</strong></div>
          <div class="chip">Tickers <strong>{df_s['ticker'].nunique()}</strong></div>
          <div class="chip" title="{_html_std.escape(_chart_chip_title)}">Charts <strong>{len(chart_cache)}</strong> / {len(website_signals)} OOS</div>
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


        html = html.replace(
            "__ANALYSIS_LLM_SECTION__",
            _analysis_llm_section.strip() if _analysis_llm_section.strip() else _anal_fallback,
        )
        _nb = len(html.encode("utf-8"))
        if _nb <= _html_cap or _chart_k == 0:
            html_chart_cap_k = _chart_k
            if _chart_k < _k_start and not _charts_external:
                print(
                    f"Website: index.html = {_nb:,} B (~{_nb / (1024 * 1024):.1f} MiB) mit {_chart_k} Chart-Signalen "
                    f"(Start {_k_start}; Ziel ≤ {_html_cap // 1024 // 1024} MiB).",
                    flush=True,
                )
            break
        if _charts_external:
            print(
                f"Website: index.html = {_nb:,} B (~{_nb / (1024 * 1024):.1f} MiB) überschreitet "
                f"{_html_cap // 1024 // 1024} MiB trotz externer Charts — PR-Plot/KI-Block prüfen.",
                flush=True,
            )
            html_chart_cap_k = _chart_k
            break
        _chart_k -= _k_step
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    _html_bytes = len(html.encode("utf-8"))
    _html_cap_i = _cfg_param_int(c, "DOCS_INDEX_HTML_MAX_BYTES", _DOCS_INDEX_HTML_MAX_BYTES_DEFAULT)
    (docs_dir / "signals.json").write_text(
        _json.dumps(
            {
                "generated": today_str,
                "threshold": float(best_threshold),
                "signals": website_signals,
                "signals_holdout_final": signals_holdout_final,
                "signals_all_timeline_including_in_sample_count": len(all_hist_signals),
                "signals_html_chart_slots": int(html_chart_cap_k),
                "signals_with_chart_files": int(len(chart_cache)),
                "chart_storage": "external" if _charts_external else "inline",
                "index_html_bytes_utf8": int(_html_bytes),
                "index_html_max_bytes_config": int(_html_cap_i),
                "note": (
                    "signals = signals_holdout_final: nur FINAL-Kalender-OOS. "
                    + (
                        "Charts als PNG unter docs/charts/ (Thumb in HTML, Vollbild per Link); "
                        "index.html unter DOCS_INDEX_HTML_MAX_BYTES."
                        if _charts_external
                        else "index.html enthält die neuesten signals_html_chart_slots OOS-Signale mit eingebettetem Base64-Chart (Größenlimit)."
                    )
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\ndocs/index.html   {_html_bytes:,} bytes (~{_html_bytes / (1024 * 1024):.2f} MiB)")
    if _charts_external:
        _charts_root = docs_dir / "charts"
        _chart_bytes = sum(
            p.stat().st_size for sub in ("full", "thumb") for p in (_charts_root / sub).glob("*.png")
            if (_charts_root / sub).is_dir()
        )
        print(
            f"docs/charts/      {_chart_bytes:,} bytes (~{_chart_bytes / (1024 * 1024):.1f} MiB), "
            f"{len(chart_cache)} Signale mit Chart",
            flush=True,
        )
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
        f"{len(recent_signals)} in last 30 days, "
        f"{html_chart_cap_k} chart slots in HTML, {_html_bytes // 1024} KiB index)"
    )
    _git_docs = [
        "docs/index.html",
        "docs/signals.json",
        "docs/website_analysis_prompt.txt",
        "docs/analysis_llm_last.html",
        "docs/analysis_llm_last.txt",
    ]
    if _charts_external and (docs_dir / "charts").is_dir():
        _git_docs.append("docs/charts")
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
