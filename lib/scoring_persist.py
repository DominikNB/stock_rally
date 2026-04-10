"""
Persist / load Meta-Scoring bundle (base models, meta_clf, threshold, FEAT_COLS, filters).
Used by stock_rally_v10.ipynb (``save_scoring_artifacts`` aus Cell 2; automatischer Aufruf nach Phase 5 in Cell 14).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, MutableMapping

import joblib
import numpy as np


def _max_date_iso_from_threshold_df(df: Any) -> str | None:
    """Letzter Kalendertag der THRESHOLD-Stichprobe (für Kalibrierung von ``best_threshold``)."""
    if df is None or not hasattr(df, "columns"):
        return None
    cols = getattr(df, "columns", None)
    if cols is None or "Date" not in cols:
        return None
    try:
        import pandas as pd

        return str(pd.to_datetime(df["Date"]).max().date())
    except Exception:
        return None


def _news_sql_manifest_from_g(g: MutableMapping[str, Any]) -> dict[str, Any]:
    """Alles, was BigQuery/GDELT-News für reproduzierbare Features braucht (ohne Exploration)."""
    sec = g.get("SECTOR_BQ_THEME_WHERE")
    if isinstance(sec, dict):
        sector_where = {str(k): str(v) for k, v in sec.items()}
    else:
        sector_where = {}
    macro = g.get("MACRO_BQ_THEME_WHERE")
    return {
        "NEWS_SOURCE": g.get("NEWS_SOURCE"),
        "MACRO_BQ_THEME_WHERE": None if macro is None else str(macro),
        "SECTOR_BQ_THEME_WHERE": sector_where,
        "GDELT_BQ_EVENTS_TABLE": g.get("GDELT_BQ_EVENTS_TABLE"),
        "BQ_USE_GKG_TABLE": g.get("BQ_USE_GKG_TABLE"),
        "BQ_USE_PARTITION_FILTER": g.get("BQ_USE_PARTITION_FILTER"),
        "BQ_SINGLE_SCAN": g.get("BQ_SINGLE_SCAN"),
        "BQ_THEMES_COLUMN": g.get("BQ_THEMES_COLUMN"),
        "BQ_MACRO_GEO_COUNTRIES": g.get("BQ_MACRO_GEO_COUNTRIES"),
        "BQ_SECTOR_USE_GEO_FILTER": g.get("BQ_SECTOR_USE_GEO_FILTER"),
        "BQ_SECTOR_GEO_COUNTRIES": list(g["BQ_SECTOR_GEO_COUNTRIES"])
        if isinstance(g.get("BQ_SECTOR_GEO_COUNTRIES"), (list, tuple))
        else g.get("BQ_SECTOR_GEO_COUNTRIES"),
        "NEWS_BQ_START_DATE": g.get("NEWS_BQ_START_DATE"),
        "NEWS_BQ_END_DATE": g.get("NEWS_BQ_END_DATE"),
        "NEWS_EXTRA_HISTORY_YEARS_BEFORE": g.get("NEWS_EXTRA_HISTORY_YEARS_BEFORE"),
        "NEWS_EXTRA_HISTORY_YEARS_AFTER": g.get("NEWS_EXTRA_HISTORY_YEARS_AFTER"),
        "USE_NEWS_SENTIMENT": g.get("USE_NEWS_SENTIMENT"),
        "GKG_AUTO_EXPLORE_THEMES": g.get("GKG_AUTO_EXPLORE_THEMES"),
        "GKG_THEME_SELECTION_PATH": str(g.get("GKG_THEME_SELECTION_PATH") or ""),
        "GKG_THEME_SQL_TRIPLES": [list(t) for t in (g.get("GKG_THEME_SQL_TRIPLES") or [])],
    }


def _read_gkg_theme_audit(g: MutableMapping[str, Any]) -> dict[str, Any] | None:
    p = g.get("GKG_THEME_SELECTION_PATH")
    if not p:
        return None
    path = Path(p)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _apply_news_sql_manifest(g: MutableMapping[str, Any], manifest: dict[str, Any] | None) -> None:
    if not manifest:
        return
    for key, val in manifest.items():
        if key == "SECTOR_BQ_THEME_WHERE" and isinstance(val, dict):
            g["SECTOR_BQ_THEME_WHERE"] = {str(k): str(v) for k, v in val.items()}
        elif key == "GKG_THEME_SQL_TRIPLES" and val is not None:
            g["GKG_THEME_SQL_TRIPLES"] = [tuple(x) for x in val]
        else:
            g[key] = val
    print(
        "News-SQL-Manifest aus Artefakt geladen — GDELT/BigQuery-Filter entsprechen dem Trainingsstand.",
        flush=True,
    )


def save_scoring_artifacts(g: MutableMapping[str, Any], path: Path | None = None) -> Path:
    """Build bundle from notebook namespace ``g`` and joblib-dump to ``path``."""
    path = Path(path or g.get("SCORING_ARTIFACT_PATH") or Path("models") / "scoring_artifacts.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    _thr_end = g.get("threshold_calibration_end_date")
    if _thr_end is None:
        _thr_end = _max_date_iso_from_threshold_df(g.get("df_threshold"))
    if isinstance(_thr_end, str) and len(_thr_end) >= 10:
        g["threshold_calibration_end_date"] = _thr_end[:10]
    else:
        g["threshold_calibration_end_date"] = _thr_end
    bundle = {
        "base_models": g["base_models"],
        "meta_clf": g["meta_clf"],
        "best_threshold": float(g["best_threshold"]),
        "FEAT_COLS": list(g["FEAT_COLS"]),
        "topk_idx": np.asarray(g["topk_idx"]),
        "topk_names": list(g["topk_names"]),
        "best_params": dict(g["best_params"]),
        "tickers_for_run": list(g.get("_tickers_for_run") or []),
        "CONSECUTIVE_DAYS": int(g["CONSECUTIVE_DAYS"]),
        "SIGNAL_COOLDOWN_DAYS": int(g["SIGNAL_COOLDOWN_DAYS"]),
        "rsi_w": int(g["rsi_w"]),
        "bb_w": int(g["bb_w"]),
        "sma_w": int(g["sma_w"]),
        "signal_skip_near_peak": bool(g.get("SIGNAL_SKIP_NEAR_PEAK", True)),
        "peak_lookback_days": int(g.get("PEAK_LOOKBACK_DAYS", 20)),
        "peak_min_dist_from_high_pct": float(g.get("PEAK_MIN_DIST_FROM_HIGH_PCT", 0.012)),
        "signal_max_rsi": None
        if g.get("SIGNAL_MAX_RSI") is None
        else float(g["SIGNAL_MAX_RSI"]),
        "threshold_calibration_end_date": g.get("threshold_calibration_end_date"),
        "news_sql_manifest": _news_sql_manifest_from_g(g),
        "gkg_theme_selection_audit": _read_gkg_theme_audit(g),
    }
    joblib.dump(bundle, path)
    print(f"Gespeichert: {path}  (threshold={bundle['best_threshold']:.4f})")
    return path


def load_scoring_artifacts(g: MutableMapping[str, Any], path: Path | None = None) -> Path:
    """Load joblib bundle and write into notebook namespace ``g``; (re)defines build_meta_features."""
    import xgboost as xgb

    path = Path(path or g.get("SCORING_ARTIFACT_PATH") or Path("models") / "scoring_artifacts.joblib")
    if not path.is_file():
        raise FileNotFoundError(
            f"Artefakt fehlt: {path} — zuerst vollständig trainieren (SCORING_ONLY=False)."
        )
    b = joblib.load(path)
    _apply_news_sql_manifest(g, b.get("news_sql_manifest"))
    if b.get("gkg_theme_selection_audit") is not None:
        g["_gkg_theme_selection_audit"] = b["gkg_theme_selection_audit"]
    g["base_models"] = b["base_models"]
    g["meta_clf"] = b["meta_clf"]
    g["best_threshold"] = float(b["best_threshold"])
    g["FEAT_COLS"] = list(b["FEAT_COLS"])
    g["topk_idx"] = np.asarray(b["topk_idx"])
    g["topk_names"] = list(b["topk_names"])
    if b.get("best_params") is not None:
        g["best_params"] = dict(b["best_params"])
    g["CONSECUTIVE_DAYS"] = int(b.get("CONSECUTIVE_DAYS", 2))
    g["SIGNAL_COOLDOWN_DAYS"] = int(b.get("SIGNAL_COOLDOWN_DAYS", 4))
    g["rsi_w"] = int(b["rsi_w"])
    g["bb_w"] = int(b["bb_w"])
    g["sma_w"] = int(b["sma_w"])
    g["SIGNAL_SKIP_NEAR_PEAK"] = bool(b.get("signal_skip_near_peak", True))
    g["PEAK_LOOKBACK_DAYS"] = int(b.get("peak_lookback_days", 20))
    g["PEAK_MIN_DIST_FROM_HIGH_PCT"] = float(b.get("peak_min_dist_from_high_pct", 0.012))
    _sr = b.get("signal_max_rsi")
    g["SIGNAL_MAX_RSI"] = None if _sr is None else float(_sr)
    if b.get("tickers_for_run"):
        g["_tickers_for_run"] = list(b["tickers_for_run"])
    _tce = b.get("threshold_calibration_end_date")
    g["threshold_calibration_end_date"] = _tce[:10] if isinstance(_tce, str) and len(_tce) >= 10 else _tce

    base_models = g["base_models"]
    topk_idx = g["topk_idx"]

    def _predict_base_logged(model_tuple, X, dataset_label=""):
        name, model, mtype = model_tuple
        t0 = time.time()
        n = len(X)
        print(f"  [{name}] scoring {n:,} Zeilen ({dataset_label})...", end="", flush=True)
        if mtype == "xgb":
            result = 1.0 / (1.0 + np.exp(-model.predict(xgb.DMatrix(X))))
        elif mtype == "lgb":
            result = 1.0 / (1.0 + np.exp(-model.predict(X)))
        else:
            result = model.predict_proba(X)[:, 1]
        print(f" {time.time() - t0:.1f}s", flush=True)
        return result

    def build_meta_features(X_feat, dataset_label=""):
        if dataset_label:
            print(f"\n--- {dataset_label}: {len(X_feat):,} Samples ---")
        base_preds = np.column_stack(
            [_predict_base_logged(m, X_feat, dataset_label) for m in base_models]
        )
        topk_feats = X_feat[:, topk_idx]
        result = np.hstack([base_preds, topk_feats]).astype(np.float32)
        if dataset_label:
            print(f"  Meta-Matrix Shape: {result.shape}")
        return result

    g["_predict_base_logged"] = _predict_base_logged
    g["build_meta_features"] = build_meta_features
    g["_SCORING_ARTIFACTS_LOADED"] = True
    print(f"Geladen: {path}  threshold={g['best_threshold']:.4f}")
    return path
