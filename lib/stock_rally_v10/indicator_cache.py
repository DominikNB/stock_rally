"""
Parquet-Cache für ``add_technical_indicators`` (voller Fenster-Modus).

Wenn ``cfg.INDICATOR_CACHE_ENABLED`` und die Eingabe-Signatur (Ticker-Universum,
Datumsbereich, Fenster-Raster, ``indicators.py``-mtime, Cache-Version) mit einem
gespeicherten Artefakt übereinstimmt, wird die Tabelle aus Platte geladen —
sonst wird neu gerechnet und der Cache atomar geschrieben.

Hinweis: Jede Änderung an Kursdaten (neue Tage), Tickerliste oder Fensterlisten
invalidiert den Cache automatisch (neuer Signatur-Hash).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _window_args_for_signature(meta_only: bool) -> dict[str, Any]:
    from lib.stock_rally_v10 import indicators as ind

    raw = ind._meta_only_window_args() if meta_only else ind._full_window_args()
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "breakout_volume_trigger_options":
            out[k] = [float(x) for x in v]
        else:
            out[k] = [int(x) for x in v]
    return out


def build_indicator_cache_signature(
    df: pd.DataFrame,
    *,
    meta_only: bool,
    cfg_mod: Any,
) -> dict[str, Any]:
    """Deterministische Signatur für Cache-Hit/Miss."""
    from lib.stock_rally_v10 import indicators as ind_mod

    tickers = sorted(df["ticker"].astype(str).unique().tolist())
    d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    ind_path = Path(ind_mod.__file__).resolve()
    ind_mtime = float(ind_path.stat().st_mtime)
    return {
        "cache_version": int(getattr(cfg_mod, "INDICATOR_CACHE_VERSION", 1)),
        "meta_only": bool(meta_only),
        "tickers_hash": hashlib.md5("\n".join(tickers).encode("utf-8")).hexdigest(),
        "n_tickers": len(tickers),
        "n_rows": int(len(df)),
        "date_min": str(d.min().date()),
        "date_max": str(d.max().date()),
        "window_args": _window_args_for_signature(meta_only),
        "indicators_py_mtime": round(ind_mtime, 6),
    }


def _cache_key_hex(sig: dict[str, Any]) -> str:
    return hashlib.md5(
        json.dumps(sig, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cache_paths(sig: dict[str, Any], cfg_mod: Any) -> tuple[Path, Path]:
    base = Path(
        getattr(cfg_mod, "INDICATOR_CACHE_DIR", os.path.join(os.getcwd(), "data", "indicator_cache"))
    )
    key = _cache_key_hex(sig)
    return base / f"ind_{key}.parquet", base / f"ind_{key}.sig.json"


def try_load_indicator_cache(
    df: pd.DataFrame,
    *,
    meta_only: bool,
    cfg_mod: Any,
) -> pd.DataFrame | None:
    """Liefert gecachte Indikator-Matrix oder ``None``."""
    if not bool(getattr(cfg_mod, "INDICATOR_CACHE_ENABLED", False)):
        return None
    sig = build_indicator_cache_signature(df, meta_only=meta_only, cfg_mod=cfg_mod)
    pq_path, js_path = _cache_paths(sig, cfg_mod)
    if not pq_path.is_file() or not js_path.is_file():
        return None
    try:
        disk_sig = json.loads(js_path.read_text(encoding="utf-8")).get("signature")
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    if disk_sig != sig:
        return None
    try:
        cached = pd.read_parquet(pq_path)
    except (OSError, ValueError):
        return None
    if len(cached) != int(sig["n_rows"]):
        print(
            f"[IndicatorCache] Treffer verworfen: rows disk={len(cached)} != sig={sig['n_rows']}.",
            flush=True,
        )
        return None
    keys = ["Date", "ticker"]
    # Nur Spalten, die ``df`` (aktuell nach ``create_target``) noch nicht hat —
    # typisch alle neuen Indikator-Spalten. Merge auf Keys: Reihenfolge von ``df`` bleibt,
    # Ziel/Preis-Spalten kommen immer aus dem aktuellen Lauf.
    extra = [c for c in cached.columns if c not in df.columns]
    if not extra:
        print("[IndicatorCache] Treffer ohne Zusatzspalten — verworfen.", flush=True)
        return None
    try:
        right = cached[keys + extra].copy()
        right["Date"] = pd.to_datetime(right["Date"], errors="coerce").dt.normalize()
        right["ticker"] = right["ticker"].astype(str)
        left = df.copy()
        left["Date"] = pd.to_datetime(left["Date"], errors="coerce").dt.normalize()
        left["ticker"] = left["ticker"].astype(str)
        out = left.merge(right, on=keys, how="left", validate="one_to_one")
    except (ValueError, KeyError) as exc:
        print(f"[IndicatorCache] Merge nach Treffer fehlgeschlagen ({exc}) — verworfen.", flush=True)
        return None
    print(
        f"[IndicatorCache] Treffer — {pq_path.name} "
        f"(+{len(extra)} Indikator-Spalten an df angehängt, shape={out.shape}).",
        flush=True,
    )
    return out


def save_indicator_cache(
    df_with_indicators: pd.DataFrame,
    df_with_target: pd.DataFrame,
    *,
    meta_only: bool,
    cfg_mod: Any,
) -> None:
    """Schreibt Parquet + Signatur-JSON atomar."""
    if not bool(getattr(cfg_mod, "INDICATOR_CACHE_ENABLED", False)):
        return
    sig = build_indicator_cache_signature(df_with_target, meta_only=meta_only, cfg_mod=cfg_mod)
    pq_path, js_path = _cache_paths(sig, cfg_mod)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = pq_path.with_suffix(pq_path.suffix + ".tmp")
    tmp_js = js_path.with_suffix(js_path.suffix + ".tmp")
    try:
        df_with_indicators.to_parquet(tmp_pq, index=False, compression="zstd")
    except (OSError, ValueError) as exc:
        print(f"[IndicatorCache] Parquet-Schreiben fehlgeschlagen ({exc}).", flush=True)
        if tmp_pq.is_file():
            tmp_pq.unlink(missing_ok=True)
        return
    payload = {
        "signature": sig,
        "shape": [int(df_with_indicators.shape[0]), int(df_with_indicators.shape[1])],
        "columns_sample": list(df_with_indicators.columns[:40]),
    }
    try:
        tmp_js.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"[IndicatorCache] JSON-Schreiben fehlgeschlagen ({exc}).", flush=True)
        tmp_pq.unlink(missing_ok=True)
        return
    try:
        os.replace(tmp_pq, pq_path)
        os.replace(tmp_js, js_path)
    except OSError as exc:
        print(f"[IndicatorCache] Atomisches Ersetzen fehlgeschlagen ({exc}).", flush=True)
        return
    print(
        f"[IndicatorCache] gespeichert: {pq_path} "
        f"(shape={df_with_indicators.shape}, ~{pq_path.stat().st_size / 1e6:.1f} MB).",
        flush=True,
    )


__all__ = [
    "build_indicator_cache_signature",
    "try_load_indicator_cache",
    "save_indicator_cache",
]
