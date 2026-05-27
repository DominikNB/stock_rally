"""
Parquet-Cache für ``augment_df_macro_regime_and_vol`` (Kurzfrist-Regime + ``mr_*``).

Treffer bei gleichem Ticker-/Datums-Universum, Zeilenanzahl, ``momentum_20d``-Digest,
``has_regime_vix_level`` und ``MACRO_AUGMENT_CACHE_VERSION``. **Keine** Python-Modul-Mtimes
in der Signatur — sonst würde jede kleine Code-Speicherung den Cache unbrauchbar machen.
Nach Änderungen an der Augment-Logik: ``MACRO_AUGMENT_CACHE_VERSION`` in ``config`` erhöhen.

Externe Makro-Zeitreihen (FRED/Yahoo) können sich für denselben Kalender still ändern;
bei Verdacht auf veraltete Fills ebenfalls Version erhöhen.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _column_float_digest(df: pd.DataFrame, col: str) -> str:
    """Deterministischer Fingerabdruck einer numerischen Spalte (Volldaten, float64)."""
    if col not in df.columns:
        return "missing"
    s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    h = hashlib.md5()
    h.update(np.ascontiguousarray(s).tobytes())
    h.update(str(int(len(s))).encode("utf-8"))
    return h.hexdigest()


def build_macro_augment_cache_signature(df: pd.DataFrame, cfg_mod: Any) -> dict[str, Any]:
    tickers = sorted(df["ticker"].astype(str).unique().tolist())
    d = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return {
        "cache_version": int(getattr(cfg_mod, "MACRO_AUGMENT_CACHE_VERSION", 1)),
        "tickers_hash": hashlib.md5("\n".join(tickers).encode("utf-8")).hexdigest(),
        "n_tickers": len(tickers),
        "n_rows": int(len(df)),
        "date_min": str(d.min().date()),
        "date_max": str(d.max().date()),
        "has_regime_vix_level": bool("regime_vix_level" in df.columns),
        "momentum_20d_digest": _column_float_digest(df, "momentum_20d"),
    }


def _cache_key_hex(sig: dict[str, Any]) -> str:
    return hashlib.md5(
        json.dumps(sig, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cache_paths(sig: dict[str, Any], cfg_mod: Any) -> tuple[Path, Path]:
    base = Path(
        getattr(
            cfg_mod,
            "MACRO_AUGMENT_CACHE_DIR",
            os.path.join(os.getcwd(), "data", "macro_augment_cache"),
        )
    )
    key = _cache_key_hex(sig)
    return base / f"macro_aug_{key}.parquet", base / f"macro_aug_{key}.sig.json"


def _sig_mismatch_keys(need: dict[str, Any], disk: dict[str, Any] | None) -> str:
    if not isinstance(disk, dict):
        return "signature fehlt/ungültig"
    diff = [k for k in sorted(need) if need.get(k) != disk.get(k)]
    if diff:
        return ", ".join(diff[:12]) + (" …" if len(diff) > 12 else "")
    extra = [k for k in sorted(disk) if k not in need]
    if extra:
        return "veraltete Felder auf Platte: " + ", ".join(extra[:8]) + (" …" if len(extra) > 8 else "")
    return "unbekannt"


def try_load_macro_augment_cache(df: pd.DataFrame, *, cfg_mod: Any) -> pd.DataFrame | None:
    if not bool(getattr(cfg_mod, "MACRO_AUGMENT_CACHE_ENABLED", False)):
        return None
    sig = build_macro_augment_cache_signature(df, cfg_mod)
    pq_path, js_path = _cache_paths(sig, cfg_mod)
    if not pq_path.is_file() or not js_path.is_file():
        print(
            f"[MacroAugmentCache] Miss — keine Datei für diese Signatur ({pq_path.name}).",
            flush=True,
        )
        return None
    try:
        disk_sig = json.loads(js_path.read_text(encoding="utf-8")).get("signature")
    except (json.JSONDecodeError, OSError, KeyError):
        print(f"[MacroAugmentCache] Miss — Signatur-JSON unleserlich: {js_path.name}.", flush=True)
        return None
    if disk_sig != sig:
        print(
            f"[MacroAugmentCache] Miss — Signatur abweichend ({_sig_mismatch_keys(sig, disk_sig)}). "
            "Nach Änderungen an der Augment-Logik ``MACRO_AUGMENT_CACHE_VERSION`` erhöhen.",
            flush=True,
        )
        return None
    try:
        cached = pd.read_parquet(pq_path)
    except (OSError, ValueError):
        return None
    if len(cached) != int(sig["n_rows"]):
        print(
            f"[MacroAugmentCache] Treffer verworfen: rows disk={len(cached)} != sig={sig['n_rows']}.",
            flush=True,
        )
        return None
    keys = ["Date", "ticker"]
    extra = [c for c in cached.columns if c not in df.columns]
    if not extra:
        print("[MacroAugmentCache] Treffer ohne Zusatzspalten — verworfen.", flush=True)
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
        print(f"[MacroAugmentCache] Merge nach Treffer fehlgeschlagen ({exc}) — verworfen.", flush=True)
        return None
    print(
        f"[MacroAugmentCache] Treffer — {pq_path.name} "
        f"(+{len(extra)} Spalten, shape={out.shape}).",
        flush=True,
    )
    return out


def save_macro_augment_cache(
    df_after: pd.DataFrame,
    cols_before: set[str],
    *,
    cfg_mod: Any,
) -> None:
    if not bool(getattr(cfg_mod, "MACRO_AUGMENT_CACHE_ENABLED", False)):
        return
    extra = [c for c in df_after.columns if c not in cols_before]
    if not extra:
        return
    in_cols = [c for c in df_after.columns if c not in extra]
    sig = build_macro_augment_cache_signature(df_after[in_cols], cfg_mod)

    pq_path, js_path = _cache_paths(sig, cfg_mod)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = pq_path.with_suffix(pq_path.suffix + ".tmp")
    tmp_js = js_path.with_suffix(js_path.suffix + ".tmp")
    keys = ["Date", "ticker"]
    try:
        df_after[keys + extra].copy().to_parquet(tmp_pq, index=False, compression="zstd")
    except (OSError, ValueError) as exc:
        print(f"[MacroAugmentCache] Parquet-Schreiben fehlgeschlagen ({exc}).", flush=True)
        if tmp_pq.is_file():
            tmp_pq.unlink(missing_ok=True)
        return
    payload = {
        "signature": sig,
        "shape": [int(df_after.shape[0]), int(len(extra))],
        "extra_columns": extra,
    }
    try:
        tmp_js.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"[MacroAugmentCache] JSON-Schreiben fehlgeschlagen ({exc}).", flush=True)
        tmp_pq.unlink(missing_ok=True)
        return
    try:
        os.replace(tmp_pq, pq_path)
        os.replace(tmp_js, js_path)
    except OSError as exc:
        print(f"[MacroAugmentCache] Atomisches Ersetzen fehlgeschlagen ({exc}).", flush=True)
        return
    print(
        f"[MacroAugmentCache] gespeichert: {pq_path.name} "
        f"(+{len(extra)} Spalten, ~{pq_path.stat().st_size / 1e6:.1f} MB).",
        flush=True,
    )


__all__ = [
    "build_macro_augment_cache_signature",
    "try_load_macro_augment_cache",
    "save_macro_augment_cache",
]
