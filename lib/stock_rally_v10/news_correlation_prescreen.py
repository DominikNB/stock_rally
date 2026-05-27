"""
News-Korrelations-Pre-Screen (vor Optuna Phase 1).

Idee:
* Heute optimiert Optuna pro Trial **ein** News-Tripel ``(mom_w, vol_ma, tone_roll)`` —
  alle anderen Tripel werden ignoriert. Trials sehen daher nur eine kleine
  Auswahl der ~360 News-Spalten, die im Shard-Storage liegen.
* Hier laden wir **alle** Tripel-Shards in eine breite Tabelle und identifizieren
  hochkorrelierte Cluster (|ρ| ≥ ``NEWS_CORRELATION_PRESCREEN_THRESHOLD``).
* Pro Cluster überlebt nur **ein** Vertreter — derjenige mit höchster
  ``mean(|SHAP|)`` aus einem Walk-Forward-Fold-XGBoost. Fallback bei SHAP-Fail:
  höchste Varianz.
* Resultat: Survivor-Liste — eine **Tripel-übergreifende Mischung** aus News-Spalten,
  die Optuna in Phase 1 dann als feste Erweiterung sieht (Tripel-Hyperparameter
  werden in dem Pfad gar nicht mehr verwendet).

Sample-Strategie (siehe Diskussion mit User):
* News-Spalten sind **Date-aligned**, daher random-sample über alle Zeilen führt
  zu Pseudo-Replikation. Wir samplen stratifiziert nach (Date, sector) — einmal
  pro Kombination. Das gibt typisch 12 sectors × ~1247 Tage ≈ 15 000 unabhängige
  Beobachtungen für ``news_sec_*`` und implizit ~1247 für ``news_macro_*``.
* Sentinel-NaN (``-1e8``) wird vor der Korrelation zu echtem ``NaN`` gemappt, dann
  ``corr(min_periods=cfg.NEWS_CORRELATION_PRESCREEN_MIN_PERIODS)``.

Persistenz:
* Artefakt: ``NEWS_CORRELATION_PRESCREEN_DIR / NEWS_CORRELATION_PRESCREEN_ARTIFACT_NAME``.
* Cache greift wenn Signatur passt (Zeilenanzahl, Spalten-Hash, Settings, Datums-Bereich).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import Counter
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.stock_rally_v10.helpers import _strip_tz, make_focal_objective


# ── Familien-Klassifikation ──────────────────────────────────────────────────
def _news_family(col: str) -> str | None:
    """Liefert Familien-Label oder ``None`` falls keine News-Spalte."""
    s = str(col)
    if not s.startswith("news_"):
        return None
    if s.startswith("news_macro_"):
        return "macro"
    if s.startswith("news_sec_") and "_anchor_" in s:
        return "anchor"
    if s.startswith("news_sec_"):
        return "sec"
    if s.startswith("news_anchor_"):
        return "anchor"
    if s.startswith("news_cross_"):
        return "cross"
    return "other"


def _print_stage(num: int, title: str) -> None:
    print(f"\n--- Stage {num} — {title} ---", flush=True)


# ── Shard-Loader: alle Tripel zu einer breiten Tabelle ────────────────────────
def _all_shard_paths(cfg_mod: Any) -> list[tuple[str, str]]:
    """Liefert Liste von (tag, path) für alle Tripel im NEWS_SHARD_MANIFEST.

    Falls Manifest nicht gesetzt: scannt ``FEATURE_SHARD_DIR`` nach ``news_tag_*.parquet``.
    """
    manifest = getattr(cfg_mod, "NEWS_SHARD_MANIFEST", None) or {}
    if manifest:
        return [(str(k), str(v)) for k, v in manifest.items() if os.path.isfile(str(v))]
    shard_dir = getattr(cfg_mod, "FEATURE_SHARD_DIR", None) or ""
    if not shard_dir or not os.path.isdir(shard_dir):
        return []
    out: list[tuple[str, str]] = []
    for fn in sorted(os.listdir(shard_dir)):
        if not fn.startswith("news_tag_"):
            continue
        if not (fn.endswith(".parquet") or fn.endswith(".pkl")):
            continue
        tag = fn[len("news_tag_") :].rsplit(".", 1)[0]
        out.append((tag, os.path.join(shard_dir, fn)))
    return out


def _load_shard(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _load_shard_narrow_for_prescreen(
    path: str, *, keys: tuple[str, str] = ("Date", "ticker")
) -> pd.DataFrame:
    """Shard nur mit Schlüsseln + ``news_*`` — vermeidet riesige Parquet-RAM-Spitzen."""
    k0, k1 = keys
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            names = list(pq.read_schema(path).names)
        except Exception:
            names = None
        if names:
            cols: list[str] = []
            for c in (k0, k1):
                if c in names and c not in cols:
                    cols.append(c)
            for c in names:
                if str(c).startswith("news_") and c not in cols:
                    cols.append(c)
            return pd.read_parquet(path, columns=cols)
    df = _load_shard(path)
    use = [c for c in df.columns if c in (k0, k1) or str(c).startswith("news_")]
    return df[use]


def _news_prescreen_key_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Gleiche Schlüssel-Normalisierung wie ``merge_news_survivors_into_df`` / Platten-Merge."""
    k = df[keys].copy()
    k["Date"] = pd.to_datetime(k["Date"], errors="coerce").dt.normalize()
    k["ticker"] = k["ticker"].astype(str).str.strip()
    return k


def _build_news_wide_table(df_train: pd.DataFrame, cfg_mod: Any) -> pd.DataFrame:
    """Mergt ``df_train[['Date','ticker','sector','target']]`` mit allen Shards.

    Rückgabe: flache Tabelle (``Date``, ``ticker``, …) + alle ``news_*``-Spalten aus allen Tripeln.

    **RAM:** (1) Shards werden nur über **Schlüssel-Spalten** auf ``df_train``-Keys
    gefiltert (kein ``merge`` auf 991 News-Spalten — vermeidet 1+ GiB Kopien).
    (2) Vor dem Shard-Loop wird ``df_train`` auf ``NEWS_CORRELATION_PRESCREEN_WIDE_MAX_ROWS``
    **zufällig** gestutzt — sonst 264k × ~10k Spalten ≈ 10+ GiB. Prescreen bleibt
    diagnostisch; Optuna lädt Survivor-Spalten später pro Trial schmal.
    (3) Parquet: ``news_*`` werden in Batches von
    ``NEWS_CORRELATION_PRESCREEN_SHARD_READ_BATCH`` gelesen (nie alle News-Spalten
    in einem ``read_parquet`` — sonst PyArrow→pandas mit tausenden Spalten / object).
    """
    shards = _all_shard_paths(cfg_mod)
    if not shards:
        raise FileNotFoundError(
            "[News-CorrPrescreen] keine News-Shards gefunden — "
            "weder NEWS_SHARD_MANIFEST noch FEATURE_SHARD_DIR liefert Treffer."
        )
    print(f"[News-CorrPrescreen] {len(shards)} Tripel-Shards gefunden.", flush=True)

    keys = ["Date", "ticker"]
    extra = [c for c in ("sector", "target") if c in df_train.columns]
    base = df_train[keys + extra].copy()
    # Schlüssel wie ``merge_news_shard_into_df``: Date normalize + tz→naiv, Ticker
    # als String (kein ``_ticker_symbol_str`` — muss 1:1 zu Shard-Export passen).
    base["Date"] = _strip_tz(pd.to_datetime(base["Date"], errors="coerce"))
    base["ticker"] = base["ticker"].astype(str)
    if bool((base["ticker"].eq("") | base["ticker"].str.lower().eq("nan")).any()):
        base = base.loc[base["ticker"].ne("") & ~base["ticker"].str.lower().eq("nan"), :].reset_index(
            drop=True
        )
    n_cap = int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_WIDE_MAX_ROWS", 20000))
    if len(base) > n_cap:
        base = base.sample(
            n=n_cap,
            random_state=int(getattr(cfg_mod, "RANDOM_STATE", 42)),
        ).reset_index(drop=True)
        print(
            f"  [News-CorrPrescreen] Wide-Build: zufällige Zeilen-Stichprobe "
            f"n={n_cap:,} (RAM-Deckel; voller BASE-Split hat {len(df_train):,} Zeilen).",
            flush=True,
        )
    base = base.reset_index(drop=True)
    key_df = _news_prescreen_key_frame(base, keys).drop_duplicates()
    # Zeilenweise Ausrichtung wie ``merge_news_survivors_into_df``: MultiIndex aus
    # ``base``-Keys + ``reindex`` auf de-dupliziertes Shard — vermeidet ``merge``-Key-Drift.
    k_align = _news_prescreen_key_frame(base, keys)
    target_idx = pd.MultiIndex.from_frame(k_align)
    news_parts: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    batch_sz = max(1, int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_SHARD_READ_BATCH", 40)))
    for i, (tag, path) in enumerate(shards, 1):
        # Parquet: **nicht** alle ``news_*`` in einem ``read_parquet`` — sonst PyArrow→pandas
        # mit tausenden Spalten / object-Dtypes → RAM-Fehler. Keys einmal, News in Batches.
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            all_names = list(pq.read_schema(path).names)
            miss = [rk for rk in keys if rk not in all_names]
            if miss:
                print(
                    f"  [News-CorrPrescreen] Shard {i} ({tag}): Spalte(n) {miss!r} fehlen — überspringe.",
                    flush=True,
                )
                if i % 5 == 0 or i == len(shards):
                    print(
                        f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                        f"(news-Spalten bisher = {len(news_parts):,})",
                        flush=True,
                    )
                continue
            sk = pd.read_parquet(path, columns=keys)
            sk["Date"] = pd.to_datetime(sk["Date"], errors="coerce").dt.normalize()
            sk["ticker"] = sk["ticker"].astype(str).str.strip()
            n_pre = len(sk)
            sk["_i"] = np.arange(n_pre, dtype=np.int64)
            mg = sk.merge(key_df, on=keys, how="inner")
            if mg.empty:
                del sk, mg
                if i % 5 == 0 or i == len(shards):
                    print(
                        f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                        f"(news-Spalten bisher = {len(news_parts):,})",
                        flush=True,
                    )
                continue
            idx_arr = mg["_i"].to_numpy()
            del sk, mg

            news_cols_all = [c for c in all_names if str(c).startswith("news_")]
            if not news_cols_all:
                del idx_arr
                if i % 5 == 0 or i == len(shards):
                    print(
                        f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                        f"(news-Spalten bisher = {len(news_parts):,})",
                        flush=True,
                    )
                continue

            row_filter_printed = False
            for j in range(0, len(news_cols_all), batch_sz):
                batch = news_cols_all[j : j + batch_sz]
                df_b = pd.read_parquet(path, columns=keys + batch)
                df_b["Date"] = pd.to_datetime(df_b["Date"], errors="coerce").dt.normalize()
                df_b["ticker"] = df_b["ticker"].astype(str).str.strip()
                df_b = df_b.take(idx_arr, axis=0)
                if df_b.duplicated(subset=keys).any():
                    df_b = df_b.drop_duplicates(subset=keys, keep="last")
                n_post = len(df_b)
                if n_post == 0:
                    del df_b
                    print(
                        f"  [News-CorrPrescreen] Shard {i} ({tag}): 0 Zeilen nach Key-Filter "
                        f"(pre={n_pre:,}) — überspringe Rest-Batches.",
                        flush=True,
                    )
                    break
                if i == 1 and not row_filter_printed:
                    row_filter_printed = True
                    print(
                        f"  [News-CorrPrescreen] Zeilen-Filter: Shard-Zeilen {n_pre:,} → "
                        f"{n_post:,} (nur Keys aus df_train).",
                        flush=True,
                    )
                s_idx = df_b.set_index(keys)
                if s_idx.index.has_duplicates:
                    s_idx = s_idx[~s_idx.index.duplicated(keep="last")]
                if i == 1 and j == 0:
                    _best_c, _best_hit = news_cols_all[0], 0
                    for _c in news_cols_all[: min(80, len(news_cols_all))]:
                        if _c not in s_idx.columns:
                            continue
                        _ser = s_idx[_c].reindex(target_idx)
                        _hit = int(pd.to_numeric(_ser, errors="coerce").notna().sum())
                        if _hit > _best_hit:
                            _best_hit, _best_c = _hit, _c
                    print(
                        f"  [News-CorrPrescreen] Reindex-Alias (Stichprobe bis 80 Spalten): {_best_c!r} "
                        f"nicht-NaN={_best_hit:,} / {len(target_idx):,}",
                        flush=True,
                    )
                for c in batch:
                    if c not in s_idx.columns:
                        continue
                    aligned_c = s_idx[c].reindex(target_idx)
                    v = pd.to_numeric(aligned_c, errors="coerce").to_numpy(dtype=np.float32, copy=True)
                    news_parts[c] = v
                    del aligned_c
                del df_b, s_idx
            del idx_arr
            gc.collect()
            if i % 5 == 0 or i == len(shards):
                print(
                    f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                    f"(news-Spalten bisher = {len(news_parts):,})",
                    flush=True,
                )
            continue

        shard = _load_shard_narrow_for_prescreen(path)
        shard["Date"] = pd.to_datetime(shard["Date"], errors="coerce").dt.normalize()
        shard["ticker"] = shard["ticker"].astype(str).str.strip()
        # Nur Zeilen im Trainings-Schlüssel: **nur** keys mergen (keine News-Spalten).
        n_pre = len(shard)
        sk = shard[keys].copy()
        sk["_i"] = np.arange(len(sk), dtype=np.int64)
        mg = sk.merge(key_df, on=keys, how="inner")
        if mg.empty:
            del sk, shard
            if i % 5 == 0 or i == len(shards):
                print(
                    f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                    f"(news-Spalten bisher = {len(news_parts):,})",
                    flush=True,
                )
            continue
        idx_arr = mg["_i"].to_numpy()
        # kein reset_index nach take — sonst tiefe Kopie (~news_cols × n_rows float32).
        shard = shard.take(idx_arr, axis=0)
        del sk, mg
        if shard.duplicated(subset=keys).any():
            shard = shard.drop_duplicates(subset=keys, keep="last")
        n_post = len(shard)
        if n_post == 0:
            print(
                f"  [News-CorrPrescreen] Shard {i} ({tag}): 0 Zeilen nach Key-Filter "
                f"(pre={n_pre:,}) — überspringe.",
                flush=True,
            )
            continue
        if i == 1:
            print(
                f"  [News-CorrPrescreen] Zeilen-Filter: Shard-Zeilen {n_pre:,} → "
                f"{n_post:,} (nur Keys aus df_train).",
                flush=True,
            )
        news_cols = [c for c in shard.columns if str(c).startswith("news_")]
        if not news_cols:
            continue
        shard_dedup = shard
        s_idx = shard_dedup.set_index(keys)
        if s_idx.index.has_duplicates:
            s_idx = s_idx[~s_idx.index.duplicated(keep="last")]
        # Kein breites ``reindex`` auf alle News-Spalten — spaltenweise (RAM).
        if i == 1 and news_cols:
            _best_c, _best_hit = news_cols[0], 0
            for _c in news_cols[: min(80, len(news_cols))]:
                if _c not in s_idx.columns:
                    continue
                _ser = s_idx[_c].reindex(target_idx)
                _hit = int(pd.to_numeric(_ser, errors="coerce").notna().sum())
                if _hit > _best_hit:
                    _best_hit, _best_c = _hit, _c
            print(
                f"  [News-CorrPrescreen] Reindex-Alias (Stichprobe bis 80 Spalten): {_best_c!r} "
                f"nicht-NaN={_best_hit:,} / {len(target_idx):,}",
                flush=True,
            )
        for c in news_cols:
            if c not in s_idx.columns:
                continue
            aligned_c = s_idx[c].reindex(target_idx)
            v = pd.to_numeric(aligned_c, errors="coerce").to_numpy(dtype=np.float32, copy=True)
            news_parts[c] = v
            del aligned_c
        del shard, shard_dedup, s_idx
        gc.collect()
        if i % 5 == 0 or i == len(shards):
            print(
                f"  [News-CorrPrescreen] {i}/{len(shards)} Shards eingelesen "
                f"(news-Spalten bisher = {len(news_parts):,})",
                flush=True,
            )
    left = base.reset_index(drop=True)
    # Ein DataFrame aus Spalten-Dicts — kein axis=1-concat (Index-/Block-Grenzen).
    col_blocks: dict[str, np.ndarray] = {
        str(c): left[c].to_numpy(copy=True) for c in left.columns
    }
    col_blocks.update(news_parts)
    out = pd.DataFrame(col_blocks, copy=False)
    if news_parts:
        _keys_sorted = sorted(news_parts.keys())
        # Alphabetisch zuerst oft ``news_cross_*`` oder leere Makro-Spalten — Sektor-Ton ist aussagekräftiger.
        _probe_m = next(
            (
                k
                for k in _keys_sorted
                if k.startswith("news_sec_") and k.endswith("_tone") and "_anchor_" not in k
            ),
            _keys_sorted[0],
        )
        _a_m = np.asarray(out[_probe_m], dtype=np.float64)
        _nf_m = int(np.isfinite(_a_m).sum())
        _probe_s = next((k for k in _keys_sorted if "news_sec_" in k and k.endswith("_tone")), None)
        _nf_s = -1
        if _probe_s is not None:
            _a_s = np.asarray(out[_probe_s], dtype=np.float64)
            _nf_s = int(np.isfinite(_a_s).sum())
        print(
            f"  [News-CorrPrescreen] Wide-Sanity: Ton-Beispiel {_probe_m!r} n_finite={_nf_m:,}; "
            f"zweites Sektor-Ton {_probe_s!r} n_finite={_nf_s:,} / {len(out):,}",
            flush=True,
        )
    print(
        f"[News-CorrPrescreen] Wide-Tabelle fertig: "
        f"shape={out.shape}, t={time.perf_counter() - t0:.1f}s.",
        flush=True,
    )
    return out


# ── Stage 0: Konstanz / all-NaN ────────────────────────────────────────────────
def _stage0_constancy(
    df_wide: pd.DataFrame, news_cols: list[str], *, sentinel: float
) -> tuple[list[str], dict[str, str]]:
    """Wie ``feature_prescreen._stage0_constancy_filter``: endliche Werte minus exakter Sentinel."""
    keep: list[str] = []
    drop: dict[str, str] = {}
    sentinel_eq = float(sentinel)
    for c in news_cols:
        col = df_wide[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        s = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        ok = np.isfinite(s)
        if int(ok.sum()) == 0:
            drop[c] = "all_nan"
            continue
        v = s[ok]
        non_sent = v[v != sentinel_eq]
        if non_sent.size == 0:
            drop[c] = "all_sentinel"
            continue
        spread = float(non_sent.max() - non_sent.min())
        std = float(np.std(non_sent, ddof=0))
        if spread <= 1e-12 and std <= 1e-12:
            drop[c] = "constant"
            continue
        keep.append(c)
    return keep, drop


# ── Stage 2: Sample-Strategie ─────────────────────────────────────────────────
def _stratified_sample(
    df_wide: pd.DataFrame, news_cols: list[str], *, sample_rows: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Stratifiziertes Sample über (Date, sector) — falls ``sector`` fehlt, fällt
    auf Date-stratifiziert zurück.
    """
    keys = ["Date"] + (["sector"] if "sector" in df_wide.columns else [])
    # Eine Zeile pro (Date, sector) — News ist dort konstant über alle Ticker.
    sample = df_wide.drop_duplicates(subset=keys, keep="first")
    n = len(sample)
    if n > sample_rows:
        idx = rng.choice(n, size=sample_rows, replace=False)
        sample = sample.iloc[idx]
    print(
        f"[News-CorrPrescreen] Sample für corr/SHAP: rows={len(sample):,} "
        f"(stratifiziert über {'+'.join(keys)}, news_cols={len(news_cols)}).",
        flush=True,
    )
    return sample.reset_index(drop=True)


# ── Stage 3: Korrelationsmatrix mit Sentinel→NaN ──────────────────────────────
def _correlation_matrix(
    sample: pd.DataFrame, news_cols: list[str], *, sentinel: float, min_periods: int
) -> pd.DataFrame:
    X = sample[news_cols].astype(np.float32).copy()
    # Sentinel → NaN, damit corr() es ignoriert.
    X = X.where(X > (sentinel + 1.0), np.nan)
    print(
        f"[News-CorrPrescreen] berechne Pearson-Korrelation auf {X.shape[0]:,} "
        f"× {X.shape[1]} (min_periods={min_periods}) — kann 5–20 s dauern …",
        flush=True,
    )
    t0 = time.perf_counter()
    corr = X.corr(method="pearson", min_periods=int(min_periods))
    print(
        f"[News-CorrPrescreen] Korrelationsmatrix fertig "
        f"(t={time.perf_counter() - t0:.1f}s).",
        flush=True,
    )
    return corr


# ── Stage 4: Hierarchical Clustering ──────────────────────────────────────────
def _cluster_columns(corr: pd.DataFrame, *, threshold: float) -> dict[int, list[str]]:
    """Single-linkage Clustering. distance = 1 - |ρ|, cutoff = 1 - threshold.

    Spalten ohne valide ρ-Werte (alle NaN) bilden eigene Singleton-Cluster.
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    cols = list(corr.columns)
    n = len(cols)
    if n <= 1:
        return {i: [c] for i, c in enumerate(cols)}

    abs_rho = corr.abs().to_numpy(dtype=np.float64, copy=True)
    # NaN → 0.0  (also unkorreliert/unbekannt → wird **nicht** mit anderem geclustert)
    np.fill_diagonal(abs_rho, 1.0)
    abs_rho = np.nan_to_num(abs_rho, nan=0.0, posinf=1.0, neginf=0.0)
    abs_rho = np.clip(abs_rho, 0.0, 1.0)
    distance = 1.0 - abs_rho
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0  # Symmetrisierung gegen FP-Drift
    np.fill_diagonal(distance, 0.0)
    try:
        cond = squareform(distance, checks=False)
        Z = hierarchy.linkage(cond, method="single")
        labels = hierarchy.fcluster(Z, t=1.0 - float(threshold), criterion="distance")
    except (ValueError, TypeError) as exc:
        print(
            f"[News-CorrPrescreen] Cluster-Fehler ({exc}); jede Spalte wird Singleton.",
            flush=True,
        )
        return {i: [c] for i, c in enumerate(cols)}

    clusters: dict[int, list[str]] = {}
    for col, lab in zip(cols, labels):
        clusters.setdefault(int(lab), []).append(col)
    return clusters


# ── Stage 5: SHAP-Importance (1 Walk-Forward Fold) ────────────────────────────
def _wf_last_fold_indices(n_dates: int, n_folds: int, *, min_train_frac: float = 0.40) -> tuple[int, int]:
    min_train = int(n_dates * float(min_train_frac))
    fold_size = max(1, (n_dates - min_train) // max(1, int(n_folds)))
    last = int(n_folds) - 1
    train_end = min_train + last * fold_size
    val_end = min(min_train + (last + 1) * fold_size, n_dates)
    return int(train_end), int(val_end)


def _compute_shap_importance(
    df_wide: pd.DataFrame, news_cols: list[str], *, cfg_mod: Any, sentinel: float
) -> dict[str, float]:
    """Trainiert ein leichtes XGBoost auf einem Walk-Forward-Fold, berechnet
    ``mean(|SHAP|)`` pro News-Spalte. Fallback {} bei Fehler."""
    if "target" not in df_wide.columns or "Date" not in df_wide.columns:
        return {}
    try:
        import xgboost as xgb
    except ImportError:
        return {}

    seed_params = dict(getattr(cfg_mod, "SEED_PARAMS", {}))
    n_folds = int(getattr(cfg_mod, "N_WF_SPLITS", 5))
    sample_rows = int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_SAMPLE_ROWS", 30000))

    dates_sorted = np.sort(df_wide["Date"].drop_duplicates().to_numpy())
    n_dates = len(dates_sorted)
    if n_dates < 50:
        return {}
    train_end_d, val_end_d = _wf_last_fold_indices(n_dates, n_folds)
    if train_end_d <= 0 or val_end_d <= train_end_d:
        return {}
    train_dates = set(dates_sorted[:train_end_d].tolist())
    val_dates = set(dates_sorted[train_end_d:val_end_d].tolist())

    df_tr = df_wide[df_wide["Date"].isin(train_dates)]
    df_va = df_wide[df_wide["Date"].isin(val_dates)]
    if len(df_tr) < 200 or len(df_va) < 50:
        return {}

    y_tr = df_tr["target"].astype(np.float32).to_numpy()
    y_va = df_va["target"].astype(np.float32).to_numpy()
    X_tr = df_tr[news_cols].astype(np.float32).to_numpy()
    X_va = df_va[news_cols].astype(np.float32).to_numpy()
    # Sentinel bleibt drin — XGBoost behandelt es wie alle anderen Werte; das ist OK
    # für Importance-Schätzung (Stage 0 hat all-Sentinel-Spalten ja schon entfernt).

    rng = np.random.default_rng(int(getattr(cfg_mod, "RANDOM_STATE", 42)))
    if X_tr.shape[0] > sample_rows:
        idx = rng.choice(X_tr.shape[0], size=sample_rows, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
    if X_va.shape[0] > sample_rows:
        idx = rng.choice(X_va.shape[0], size=sample_rows, replace=False)
        X_va = X_va[idx]
        y_va = y_va[idx]

    focal_gamma = float(seed_params.get("focal_gamma", 1.5))
    focal_alpha = float(seed_params.get("focal_alpha", 0.25))
    obj = make_focal_objective(focal_gamma, focal_alpha)

    params = dict(
        max_depth=int(seed_params.get("max_depth", 6)),
        learning_rate=float(seed_params.get("learning_rate", 0.05)),
        subsample=float(seed_params.get("subsample", 0.8)),
        colsample_bytree=float(seed_params.get("colsample_bytree", 0.6)),
        min_child_weight=float(seed_params.get("min_child_weight", 10)),
        reg_alpha=float(seed_params.get("reg_alpha", 1e-3)),
        reg_lambda=float(seed_params.get("reg_lambda", 1e-3)),
        tree_method="hist",
        random_state=int(getattr(cfg_mod, "RANDOM_STATE", 42)),
        n_jobs=int(getattr(cfg_mod, "N_WORKERS", 4)),
        verbosity=0,
    )
    print(
        f"[News-CorrPrescreen] SHAP-Importance: trainiere XGBoost auf "
        f"{X_tr.shape[0]:,} × {X_tr.shape[1]} (val={X_va.shape[0]:,}) …",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        booster = xgb.XGBRegressor(
            n_estimators=int(seed_params.get("n_estimators", 200)),
            objective=obj,
            **params,
        )
        booster.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        # TreeSHAP über das Validierungs-Sample (kleinere RAM-Spitze).
        booster_native = booster.get_booster()
        dval = xgb.DMatrix(X_va, feature_names=news_cols)
        shap_vals = booster_native.predict(dval, pred_contribs=True)
        # Letzte Spalte = Bias → abschneiden.
        if shap_vals.ndim == 2 and shap_vals.shape[1] == len(news_cols) + 1:
            shap_vals = shap_vals[:, : len(news_cols)]
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"[News-CorrPrescreen] SHAP-Fehler ({exc}); Fallback Varianz.", flush=True)
        return {}
    print(
        f"[News-CorrPrescreen] SHAP-Importance fertig "
        f"(t={time.perf_counter() - t0:.1f}s).",
        flush=True,
    )
    return {col: float(v) for col, v in zip(news_cols, mean_abs)}


# ── Stage 6: Survivor pro Cluster ─────────────────────────────────────────────
def _pick_survivors(
    clusters: dict[int, list[str]],
    *,
    importance: dict[str, float],
    sample: pd.DataFrame,
) -> tuple[list[str], list[dict[str, Any]]]:
    survivors: list[str] = []
    cluster_log: list[dict[str, Any]] = []
    for lab, members in clusters.items():
        if len(members) == 1:
            survivors.append(members[0])
            cluster_log.append({
                "cluster": int(lab), "size": 1,
                "winner": members[0], "criterion": "singleton",
                "members": members,
            })
            continue
        if importance:
            scored = [(c, float(importance.get(c, 0.0))) for c in members]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            winner = scored[0][0]
            crit = "shap"
        else:
            # Fallback: höchste Varianz auf dem Sample.
            scored_var = [(c, float(np.nanvar(sample[c].to_numpy(dtype=np.float32)))) for c in members]
            scored_var.sort(key=lambda kv: kv[1], reverse=True)
            winner = scored_var[0][0]
            crit = "variance"
        survivors.append(winner)
        cluster_log.append({
            "cluster": int(lab), "size": len(members),
            "winner": winner, "criterion": crit,
            "members": members,
        })
    return survivors, cluster_log


# ── Cache-Signatur & Persistenz ────────────────────────────────────────────────
def _input_signature(
    df_wide: pd.DataFrame,
    news_cols: list[str],
    cfg_mod: Any,
    *,
    n_rows_full_train: int,
) -> dict[str, Any]:
    payload = {
        "n_rows_wide_sample": int(len(df_wide)),
        "n_rows_full_train": int(n_rows_full_train),
        "n_news_cols": int(len(news_cols)),
        "news_cols_hash": hashlib.md5(
            "\n".join(sorted(news_cols)).encode("utf-8")
        ).hexdigest(),
        "threshold": float(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_THRESHOLD", 0.95)),
        "sample_rows": int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_SAMPLE_ROWS", 30000)),
        "wide_max_rows": int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_WIDE_MAX_ROWS", 35000)),
        "min_periods": int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_MIN_PERIODS", 200)),
        "n_wf_splits": int(getattr(cfg_mod, "N_WF_SPLITS", 5)),
        "random_state": int(getattr(cfg_mod, "RANDOM_STATE", 42)),
    }
    if "Date" in df_wide.columns and len(df_wide) > 0:
        d = pd.to_datetime(df_wide["Date"], errors="coerce")
        payload["date_min"] = str(d.min().date())
        payload["date_max"] = str(d.max().date())
    return payload


def _try_load_cached(path: Path, sig: dict[str, Any], cfg_mod: Any) -> dict | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[News-CorrPrescreen] Cache nicht lesbar ({exc}) — wird neu gebaut.", flush=True)
        return None
    if cached.get("input_signature") != sig:
        if bool(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_REUSE_SAME_CALENDAR_DAY", True)):
            built_on = cached.get("built_on")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if built_on == today:
                print(
                    f"[News-CorrPrescreen] Cache vom heutigen Tag — Signatur weicht ab, aber "
                    f"REUSE_SAME_CALENDAR_DAY=True. Übernehme: {path}",
                    flush=True,
                )
                return cached
        print("[News-CorrPrescreen] Cache-Signatur weicht ab — wird neu gebaut.", flush=True)
        return None
    print(f"[News-CorrPrescreen] Cache übernommen: {path}", flush=True)
    return cached


# ── Hauptfunktion ─────────────────────────────────────────────────────────────
def run_news_correlation_prescreen(
    df_train: pd.DataFrame, *, cfg_mod: Any
) -> dict | None:
    """Führt den News-Korrelations-Pre-Screen aus.

    Return: Artefakt mit Survivor-Liste, oder ``None`` wenn übersprungen.
    """
    if not bool(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_ENABLED", False)):
        return None
    if not bool(getattr(cfg_mod, "USE_NEWS_SENTIMENT", False)):
        print("[News-CorrPrescreen] USE_NEWS_SENTIMENT=False — überspringe.", flush=True)
        return None
    if not bool(getattr(cfg_mod, "_FEATURE_NEWS_SHARDS_ACTIVE", False)):
        print("[News-CorrPrescreen] _FEATURE_NEWS_SHARDS_ACTIVE=False — überspringe.", flush=True)
        return None
    if df_train is None or len(df_train) == 0:
        print("[News-CorrPrescreen] df_train leer — überspringe.", flush=True)
        return None

    print("=" * 72, flush=True)
    print("News-Korrelations-Pre-Screen (vor Optuna Phase 1)", flush=True)
    print("=" * 72, flush=True)
    threshold = float(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_THRESHOLD", 0.95))
    sample_rows = int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_SAMPLE_ROWS", 30000))
    min_periods = int(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_MIN_PERIODS", 200))
    sentinel = float(getattr(cfg_mod, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    print(
        f"  Settings: threshold={threshold:.2f}, sample_rows={sample_rows:,}, "
        f"min_periods={min_periods}, sentinel={sentinel:.0e}",
        flush=True,
    )

    _print_stage(0, "Shards laden + breite Tabelle")
    t0 = time.perf_counter()
    try:
        df_wide = _build_news_wide_table(df_train, cfg_mod)
    except FileNotFoundError as exc:
        print(f"[News-CorrPrescreen] {exc}", flush=True)
        return None
    news_cols_full = [c for c in df_wide.columns if str(c).startswith("news_")]
    print(
        f"  Wide-Tabelle: rows={len(df_wide):,}, news_cols={len(news_cols_full)}, "
        f"t={time.perf_counter() - t0:.1f}s.",
        flush=True,
    )

    _print_stage(1, "Konstanz / all-NaN-Filter")
    keep, dropped_const = _stage0_constancy(df_wide, news_cols_full, sentinel=sentinel)
    print(
        f"  Konstanz: kept={len(keep)}/{len(news_cols_full)}, dropped={len(dropped_const)}.",
        flush=True,
    )
    if not keep:
        _ctr = Counter(dropped_const.values())
        print(
            f"  [News-CorrPrescreen] Stage-1 verworfen (Top-Gründe): "
            f"{_ctr.most_common(6)}",
            flush=True,
        )
        print("[News-CorrPrescreen] keine News-Spalten überleben Stage 1 — Abbruch.", flush=True)
        return None

    art_dir = Path(getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_DIR", "data"))
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / str(
        getattr(cfg_mod, "NEWS_CORRELATION_PRESCREEN_ARTIFACT_NAME", "news_correlation_prescreen_v1.json")
    )
    sig = _input_signature(df_wide, keep, cfg_mod, n_rows_full_train=int(len(df_train)))
    cached = _try_load_cached(art_path, sig, cfg_mod)
    if cached is not None:
        return cached

    _print_stage(2, "Stratifiziertes Sample (Date × Sector)")
    rng = np.random.default_rng(int(getattr(cfg_mod, "RANDOM_STATE", 42)))
    sample = _stratified_sample(df_wide, keep, sample_rows=sample_rows, rng=rng)

    _print_stage(3, "Pearson-Korrelation (Sentinel→NaN)")
    corr = _correlation_matrix(sample, keep, sentinel=sentinel, min_periods=min_periods)

    _print_stage(4, f"Hierarchical Clustering (|ρ| ≥ {threshold:.2f})")
    clusters = _cluster_columns(corr, threshold=threshold)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(
        f"  Cluster: total={len(clusters)}, größtes={sizes[0] if sizes else 0}, "
        f"singletons={sum(1 for s in sizes if s == 1)}.",
        flush=True,
    )

    _print_stage(5, "SHAP-Importance (1 WF-Fold, focal XGBoost)")
    importance = _compute_shap_importance(df_wide, keep, cfg_mod=cfg_mod, sentinel=sentinel)
    if importance:
        top5 = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("  Top-5 mean(|SHAP|):", flush=True)
        for c, v in top5:
            print(f"     {c}: {v:.4f}", flush=True)
    else:
        print("  SHAP nicht verfügbar — Fallback auf Varianz pro Cluster.", flush=True)

    _print_stage(6, "Survivor pro Cluster wählen")
    survivors, cluster_log = _pick_survivors(clusters, importance=importance, sample=sample)
    print(
        f"  Survivors: {len(survivors)} von {len(keep)} News-Spalten "
        f"(Reduktion {100.0 * (1 - len(survivors)/max(1,len(keep))):.0f}%).",
        flush=True,
    )

    # Familien-Aufschlüsselung (für Diagnose)
    fam_count_in: dict[str, int] = {}
    fam_count_out: dict[str, int] = {}
    for c in keep:
        fam = _news_family(c) or "unknown"
        fam_count_in[fam] = fam_count_in.get(fam, 0) + 1
    for c in survivors:
        fam = _news_family(c) or "unknown"
        fam_count_out[fam] = fam_count_out.get(fam, 0) + 1
    print("  Familien-Aufschlüsselung (in → out):", flush=True)
    for fam in sorted(set(fam_count_in) | set(fam_count_out)):
        print(f"     {fam}: {fam_count_in.get(fam,0)} → {fam_count_out.get(fam,0)}", flush=True)

    artifact: dict[str, Any] = {
        "version": 1,
        "built_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_signature": sig,
        "survivors": sorted(survivors),
        "n_news_in": len(keep),
        "n_news_out": len(survivors),
        "stage1_constancy_dropped": dropped_const,
        "family_summary_in": fam_count_in,
        "family_summary_out": fam_count_out,
        "clusters": cluster_log,
        "importance_top_50": [
            {"feature": k, "mean_abs_shap": float(v)}
            for k, v in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:50]
        ] if importance else [],
        "diagnostics": {
            "elapsed_total_s": round(time.perf_counter() - t0, 2),
            "n_clusters": len(clusters),
            "largest_cluster_size": sizes[0] if sizes else 0,
            "n_singletons": sum(1 for s in sizes if s == 1),
            "shap_used": bool(importance),
        },
    }
    try:
        art_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"[News-CorrPrescreen] Artefakt gespeichert: {art_path}", flush=True)
    except OSError as exc:
        print(f"[News-CorrPrescreen] Artefakt-Persist fehlgeschlagen ({exc}).", flush=True)

    print("=" * 72, flush=True)
    return artifact


__all__ = ["run_news_correlation_prescreen"]
