"""
Phase 11: Statistisches Feature-Pre-Pruning (nur BASE-Kalender ``df_train``).

Sparsity (Sentinel-Anteil) + Mutual Information vs. ``target``; Whitelist für ``mr_*``,
``regime_*`` und ID-Spalten. Mit ``STATISTICAL_PRE_PRUNE_PER_TAG_NEWS``: je Manifest-Tag
eigene News-Survivors + gemeinsame Nicht-News-Liste → ``cfg.FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG``.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from lib.stock_rally_v10.extended_base_features import append_macro_regime_vol_numeric_cols
from lib.stock_rally_v10.features import (
    clear_news_shard_frame_cache,
    merge_news_shard_from_best_params,
    merge_news_shard_into_df,
)
from lib.stock_rally_v10.news_tag_sync import news_tag_from_params


def _log_phase11(msg: str) -> None:
    print(f"[Phase 11] {msg}", flush=True)


def _available_news_shard_tags(cfg_mod: Any) -> set[str]:
    manifest = getattr(cfg_mod, "NEWS_SHARD_MANIFEST", None) or {}
    if isinstance(manifest, dict) and manifest:
        return {str(k) for k in manifest.keys()}
    shard_dir = getattr(cfg_mod, "FEATURE_SHARD_DIR", None) or ""
    if not shard_dir or not Path(shard_dir).is_dir():
        return set()
    tags: set[str] = set()
    for fn in Path(shard_dir).glob("news_tag_*.parquet"):
        tags.add(fn.stem.replace("news_tag_", "", 1))
    return tags


def resolve_news_params_for_phase11(
    cfg_mod: Any,
    seed_params: dict[str, Any],
) -> dict[str, Any]:
    """
    News-Tripel für Phase 11: SEED_PARAMS, aber nur Tags mit existierendem Shard.

    Fallback-Reihenfolge: exakter SEED-Tag → gleiches mom/vol mit anderem tone_roll
    aus ``NEWS_TONE_ROLL_WINDOWS`` → ``STATISTICAL_PRE_PRUNE_NEWS_TAG`` → ``5_10_3``.
    """
    sp = dict(seed_params or {})
    available = _available_news_shard_tags(cfg_mod)
    mom = int(sp.get("news_mom_w", 5))
    vol = int(sp.get("news_vol_ma", 10))
    tone = int(sp.get("news_tone_roll", 3))
    tag_fn = getattr(cfg_mod, "news_feat_tag", None)
    if not callable(tag_fn):
        raise AttributeError("cfg_mod.news_feat_tag fehlt")

    def _bp(m: int, v: int, t: int) -> dict[str, Any]:
        return {
            "news_mom_w": m,
            "news_vol_ma": v,
            "news_tone_roll": t,
            "news_extra_zscore_w": sp.get("news_extra_zscore_w"),
            "news_extra_tone_accel": sp.get("news_extra_tone_accel"),
            "news_extra_macro_sec_diff": sp.get("news_extra_macro_sec_diff"),
            "news_add_sign_confirmation": sp.get("news_add_sign_confirmation"),
        }

    if not available:
        _log_phase11(
            "Kein NEWS_SHARD_MANIFEST — News-Merge in Phase 11 übersprungen "
            "(nur technische/mr_-Spalten für MI)."
        )
        return _bp(mom, vol, tone)

    seed_tag = str(tag_fn(mom, vol, tone))
    if seed_tag in available:
        return _bp(mom, vol, tone)

    tone_grid = tuple(getattr(cfg_mod, "NEWS_TONE_ROLL_WINDOWS", (3, 5, 10)))
    for t in tone_grid:
        if str(tag_fn(mom, vol, int(t))) in available:
            _log_phase11(
                f"News-Shard: SEED-Tag {seed_tag!r} fehlt — Fallback {mom}_{vol}_{t} "
                f"(tone_roll {tone} → {t})."
            )
            return _bp(mom, vol, int(t))

    for m in getattr(cfg_mod, "NEWS_MOM_WINDOWS", (3, 5, 7)):
        for v in getattr(cfg_mod, "NEWS_VOL_MA_WINDOWS", (10, 20)):
            for t in tone_grid:
                if str(tag_fn(int(m), int(v), int(t))) in available:
                    _log_phase11(
                        f"News-Shard: Fallback auf Optuna-Grid-Tag {m}_{v}_{t} "
                        f"(SEED {seed_tag!r} nicht vorhanden)."
                    )
                    return _bp(int(m), int(v), int(t))

    ref = str(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_NEWS_TAG", "5_10_3"))
    if ref in available:
        parts = ref.split("_")
        if len(parts) == 3:
            _log_phase11(f"News-Shard: Fallback STATISTICAL_PRE_PRUNE_NEWS_TAG={ref!r}.")
            return _bp(int(parts[0]), int(parts[1]), int(parts[2]))

    raise FileNotFoundError(
        f"Phase 11: kein News-Shard für SEED-Tag {seed_tag!r} und kein Fallback in "
        f"Manifest ({len(available)} Tags). assemble_features [News → Platte] ausführen."
    )


def _feat_bucket_counts(names: list[str]) -> dict[str, int]:
    buckets = {"news": 0, "mr_": 0, "regime_": 0, "other": 0}
    for c in names:
        s = str(c)
        if s.startswith("news_"):
            buckets["news"] += 1
        elif s.startswith("mr_"):
            buckets["mr_"] += 1
        elif s.startswith("regime_"):
            buckets["regime_"] += 1
        else:
            buckets["other"] += 1
    return buckets


def _is_whitelisted(
    col: str,
    *,
    prefixes: tuple[str, ...],
    exact: frozenset[str],
) -> bool:
    c = str(col)
    if c in exact:
        return True
    return any(c.startswith(p) for p in prefixes)


def _sparsity_keep_columns(
    df: pd.DataFrame,
    pool: list[str],
    *,
    sentinel: float,
    max_sentinel_frac: float,
) -> list[bool]:
    """Spaltenweise Sparsity (kein N×M-``isclose``-Block — vermeidet ~1 GiB RAM pro Tag)."""
    max_frac = float(max_sentinel_frac)
    keep: list[bool] = []
    for col in pool:
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if s.size == 0:
            keep.append(False)
            continue
        bad = np.isclose(s, sentinel, rtol=0.0, atol=0.0) | ~np.isfinite(s)
        keep.append(float(bad.mean()) < max_frac)
    return keep


def _resolve_mi_sample_rows(cfg_mod: Any, n_rows: int, n_cols: int) -> int | None:
    """
    Zeilen für MI: ``STATISTICAL_PRE_PRUNE_MI_MAX_ROWS`` wenn gesetzt, sonst Auto-Cap
    wenn ``n_rows × n_cols`` den Speicher-Budget überschreitet (18 Tags × ~550 News).
    """
    explicit = getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_MAX_ROWS", None)
    if explicit is not None and int(explicit) > 0:
        return int(explicit) if n_rows > int(explicit) else None
    budget_mb = float(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_MATRIX_BUDGET_MB", 180.0))
    budget_bytes = int(budget_mb * 1024 * 1024)
    need = n_rows * max(1, n_cols) * 8
    if need <= budget_bytes:
        return None
    cap = max(8_000, budget_bytes // (max(1, n_cols) * 8))
    return min(n_rows, int(cap))


def execute_statistical_filter(
    df_train: pd.DataFrame,
    full_features: list[str],
    *,
    cfg_mod: Any,
) -> list[str]:
    """
    Filtert ``full_features`` anhand von Sparsity + MI (nur ``df_train``).

    Whitelist-Spalten werden immer behalten; ``min_keep`` erzwingt eine Mindestgröße.
    """
    t0 = time.perf_counter()
    sentinel = float(getattr(cfg_mod, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    max_sent_frac = float(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MAX_SENTINEL_FRAC", 0.98))
    mi_top_k = int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_TOP_K", 150))
    mi_min = float(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_MIN_SCORE", 0.0))
    min_keep = int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MIN_KEEP", 80))
    prefixes = tuple(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_WHITELIST_PREFIXES", ("mr_", "regime_")))
    exact = frozenset(
        getattr(
            cfg_mod,
            "STATISTICAL_PRE_PRUNE_WHITELIST_EXACT",
            ("month", "sector_id", "gics_sector_id", "gics_industry_id"),
        )
    )
    rs = int(getattr(cfg_mod, "RANDOM_STATE", 42))
    log_top_mi = int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_LOG_TOP_MI", 15))
    log_drop_sample = int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_LOG_DROP_SAMPLE", 12))

    _log_phase11(
        f"Config: max_sentinel_frac={max_sent_frac}, mi_top_k={mi_top_k}, "
        f"mi_min={mi_min}, min_keep={min_keep}, whitelist_prefixes={prefixes}"
    )

    if "target" not in df_train.columns:
        raise KeyError("feature_pre_pruning: 'target' fehlt in df_train.")

    candidates = [str(c) for c in full_features]
    present = [c for c in candidates if c in df_train.columns]
    missing = [c for c in candidates if c not in df_train.columns]
    if missing:
        _log_phase11(
            f"{len(missing)} Kandidaten-Spalten fehlen in df_train — "
            f"Sentinel-Fill (z. B. {missing[:3]})."
        )
        for c in missing:
            df_train[c] = sentinel
        present = candidates

    whitelist = [c for c in present if _is_whitelisted(c, prefixes=prefixes, exact=exact)]
    pool = [c for c in present if c not in set(whitelist)]

    keep_sparse = _sparsity_keep_columns(
        df_train, pool, sentinel=sentinel, max_sentinel_frac=max_sent_frac
    )
    dropped_sparse = [pool[i] for i in range(len(pool)) if not keep_sparse[i]]
    pool_kept = [pool[i] for i in range(len(pool)) if keep_sparse[i]]
    _log_phase11(
        f"Sparsity: pool={len(pool)} → {len(pool_kept)} "
        f"(gedroppt={len(dropped_sparse)}, Whitelist={len(whitelist)})."
    )
    if dropped_sparse and log_drop_sample > 0:
        sample = dropped_sparse[:log_drop_sample]
        suffix = f" … +{len(dropped_sparse) - len(sample)}" if len(dropped_sparse) > len(sample) else ""
        _log_phase11(f"Sparsity-Drop-Beispiele: {sample}{suffix}")

    y = df_train["target"].to_numpy(dtype=np.int8, copy=False)
    n_pos = int((y == 1).sum())
    _log_phase11(
        f"BASE-Zeilen={len(df_train):,}, target pos={n_pos:,} "
        f"({100.0 * n_pos / max(1, len(y)):.2f} %)."
    )
    mi_survivors: list[str] = []
    if pool_kept:
        n_all = len(df_train)
        mi_sample_rows = _resolve_mi_sample_rows(cfg_mod, n_all, len(pool_kept))
        if mi_sample_rows is not None and n_all > mi_sample_rows:
            rng = np.random.RandomState(rs)
            idx = rng.choice(n_all, size=int(mi_sample_rows), replace=False)
            df_mi = df_train.iloc[idx]
            y_mi = y[idx]
            _log_phase11(
                f"MI: Stichprobe {mi_sample_rows:,} / {n_all:,} Zeilen × {len(pool_kept)} Spalten "
                f"(Speicher-Cap; Budget={getattr(cfg_mod, 'STATISTICAL_PRE_PRUNE_MI_MATRIX_BUDGET_MB', 180)} MB) …"
            )
        else:
            df_mi = df_train
            y_mi = y
            _log_phase11(
                f"MI: starte mutual_info_classif auf {n_all:,} Zeilen × {len(pool_kept)} Spalten "
                f"(sklearn ohne Fortschritts-Log) …"
            )
        t_mi = time.perf_counter()
        X_mi = df_mi[pool_kept].to_numpy(dtype=np.float32, copy=True)
        X_mi[np.isclose(X_mi, np.float32(sentinel), rtol=0.0, atol=0.0)] = np.nan
        X_mi[~np.isfinite(X_mi)] = np.nan
        try:
            mi_scores = mutual_info_classif(
                X_mi,
                y_mi,
                random_state=rs,
                discrete_features=False,
            )
        except Exception as exc:
            _log_phase11(f"MI fehlgeschlagen ({exc}) — behalte Sparsity-Pool ({len(pool_kept)} Spalten).")
            mi_survivors = list(pool_kept)
            mi_series = None
        else:
            mi_series = pd.Series(mi_scores, index=pool_kept).sort_values(ascending=False)
            above = mi_series[mi_series >= mi_min]
            mi_survivors = list(above.head(mi_top_k).index)
            if len(mi_survivors) < min(len(pool_kept), mi_top_k):
                mi_survivors = list(mi_series.head(mi_top_k).index)
            _log_phase11(
                f"MI: fertig in {time.perf_counter() - t_mi:.1f}s — {len(pool_kept)} Spalten → "
                f"Top-{len(mi_survivors)} (mi_min={mi_min}, max_score={float(mi_series.max()):.6f})."
            )
            if log_top_mi > 0:
                top = mi_series.head(log_top_mi)
                _log_phase11(
                    "Top-MI: "
                    + ", ".join(f"{n}={v:.5f}" for n, v in top.items())
                )

    surviving = list(dict.fromkeys(whitelist + mi_survivors))
    n_before_keep = len(surviving)
    if len(surviving) < min_keep:
        ranked_extra = list(pool_kept)
        if pool_kept and "mi_series" in locals():
            ranked_extra = list(mi_series.index)
        for c in ranked_extra + [x for x in present if x not in set(ranked_extra)]:
            if c in surviving:
                continue
            surviving.append(c)
            if len(surviving) >= min_keep:
                break
        _log_phase11(
            f"min_keep={min_keep}: Survivors von {n_before_keep} auf {len(surviving)} aufgefüllt."
        )

    b_in = _feat_bucket_counts(present)
    b_out = _feat_bucket_counts(surviving)
    _log_phase11(
        f"Buckets vorher  news={b_in['news']} mr_={b_in['mr_']} regime_={b_in['regime_']} "
        f"other={b_in['other']}"
    )
    _log_phase11(
        f"Buckets nachher news={b_out['news']} mr_={b_out['mr_']} regime_={b_out['regime_']} "
        f"other={b_out['other']}"
    )
    _log_phase11(
        f"Ergebnis: {len(present)} Kandidaten → {len(surviving)} Survivors "
        f"(−{len(present) - len(surviving)}, {100.0 * (1 - len(surviving) / max(1, len(present))):.1f} % reduziert, "
        f"t={time.perf_counter() - t0:.1f}s)."
    )
    return surviving


def news_params_from_tag(tag: str, seed_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """``mom_vol_tone`` → News-Hyperparameter-Dict (wie ``SEED_PARAMS``-Subset)."""
    parts = str(tag).strip().split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"news_params_from_tag: ungültiger Tag {tag!r} (erwartet z. B. '5_10_3').")
    sp = dict(seed_params or {})
    m, v, t = int(parts[0]), int(parts[1]), int(parts[2])
    return {
        "news_mom_w": m,
        "news_vol_ma": v,
        "news_tone_roll": t,
        "news_extra_zscore_w": sp.get("news_extra_zscore_w"),
        "news_extra_tone_accel": sp.get("news_extra_tone_accel"),
        "news_extra_macro_sec_diff": sp.get("news_extra_macro_sec_diff"),
        "news_add_sign_confirmation": sp.get("news_add_sign_confirmation"),
    }


def _non_news_reference_cols(
    cfg_mod: Any,
    df_train: pd.DataFrame,
    *,
    news_params: dict[str, Any] | None = None,
) -> list[str]:
    """Technik + Makro/Regime ohne ``news_*`` (Namen hängen nicht vom News-Shard ab)."""
    full = build_reference_feat_cols(cfg_mod, df_train, news_params=news_params)
    return [str(c) for c in full if not str(c).startswith("news_")]


def resolve_statistical_survivors_for_tag(
    cfg_mod: Any,
    trial_news_tag: str | None,
) -> set[str] | None:
    """
    Survivor-Menge für Optuna/Phase 12: pro Tag aus ``FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG``,
    sonst flache ``FEAT_COLS_STATISTICAL_SURVIVORS`` / Artefakt (Legacy).
    """
    tag = str(trial_news_tag).strip() if trial_news_tag else ""
    by_tag = getattr(cfg_mod, "FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG", None)
    if not by_tag:
        art = getattr(cfg_mod, "_STATISTICAL_PRE_PRUNE_ARTIFACT", None)
        if isinstance(art, dict):
            raw_bt = art.get("survivors_by_tag")
            if isinstance(raw_bt, dict):
                by_tag = raw_bt
    if isinstance(by_tag, dict) and tag and tag in by_tag:
        raw = by_tag[tag]
        if raw:
            return {str(x) for x in raw}
    if isinstance(by_tag, dict) and tag and tag not in by_tag:
        print(
            f"[Optuna] Phase-11: kein Eintrag survivors_by_tag[{tag!r}] — "
            f"Fallback auf Referenz-Survivors.",
            flush=True,
        )
    surv_raw = getattr(cfg_mod, "FEAT_COLS_STATISTICAL_SURVIVORS", None)
    if not surv_raw:
        art = getattr(cfg_mod, "_STATISTICAL_PRE_PRUNE_ARTIFACT", None)
        if isinstance(art, dict):
            surv_raw = art.get("survivors")
    if surv_raw:
        return {str(x) for x in surv_raw}
    return None


def _news_feature_names_for_params(cfg_mod: Any, bp: dict[str, Any]) -> list[str]:
    """``news_*``-Namen für ein Fenster-Tripel (ohne Makro-Append / df-Mutation)."""
    sp = dict(getattr(cfg_mod, "SEED_PARAMS", None) or {})
    cols = cfg_mod.build_feature_cols(
        int(sp.get("rsi_window", 21)),
        int(sp.get("bb_window", 20)),
        int(sp.get("sma_window", 50)),
        int(bp["news_mom_w"]),
        int(bp["news_vol_ma"]),
        int(bp["news_tone_roll"]),
        bp.get("news_extra_zscore_w", sp.get("news_extra_zscore_w")),
        bp.get("news_extra_tone_accel", sp.get("news_extra_tone_accel")),
        bp.get("news_extra_macro_sec_diff", sp.get("news_extra_macro_sec_diff")),
        btc_momentum_z_window=sp.get("btc_momentum_z_window"),
        market_breadth_z_window=sp.get("market_breadth_z_window"),
        rel_momentum_window=sp.get("rel_momentum_window"),
        adr_window=sp.get("adr_window"),
        breakout_lookback_window=sp.get("breakout_lookback_window"),
        vcp_window=sp.get("vcp_window"),
        btc_corr_window=sp.get("btc_corr_window"),
        yz_vol_window=sp.get("yz_vol_window"),
        downside_vol_window=sp.get("downside_vol_window"),
        ret_moment_window=sp.get("ret_moment_window"),
        amihud_window=sp.get("amihud_window"),
        vcp_lower_low_window=sp.get("vcp_lower_low_window"),
        breakout_volume_trigger_z=sp.get("breakout_volume_trigger_z"),
        news_add_sign_confirmation=bp.get(
            "news_add_sign_confirmation", sp.get("news_add_sign_confirmation")
        ),
    )
    return [str(c) for c in cols if str(c).startswith("news_")]


def _run_phase11_per_tag_news(
    cfg_mod: Any,
    df_train: pd.DataFrame,
    *,
    sp: dict[str, Any],
    tags: list[str],
) -> tuple[dict[str, list[str]], list[str], str]:
    """Variante A: Nicht-News einmal filtern; News pro Manifest-Tag."""
    bp_ref = resolve_news_params_for_phase11(cfg_mod, sp)
    ref_tag = news_tag_from_params(bp_ref, cfg_mod)
    df_base = df_train.copy()
    non_news_names = _non_news_reference_cols(cfg_mod, df_base, news_params=bp_ref)
    _log_phase11(
        f"Gemeinsame Nicht-News-Kandidaten: {len(non_news_names)} (ohne Shard-Merge)."
    )
    shared = execute_statistical_filter(df_base, non_news_names, cfg_mod=cfg_mod)
    b_shared = _feat_bucket_counts(shared)
    _log_phase11(
        f"Gemeinsame Nicht-News-Survivors: {len(shared)} "
        f"(news={b_shared['news']} mr_={b_shared['mr_']} regime_={b_shared['regime_']} "
        f"other={b_shared['other']})."
    )

    use_news = bool(getattr(cfg_mod, "USE_NEWS_SENTIMENT", False))
    survivors_by_tag: dict[str, list[str]] = {}
    core_cols = [c for c in ("Date", "ticker", "target", "sector") if c in df_train.columns]
    clear_news_shard_frame_cache()
    for i, tag in enumerate(tags):
        _log_phase11(f"News-Tag {i + 1}/{len(tags)}: {tag} …")
        bp = news_params_from_tag(tag, sp)
        news_names = _news_feature_names_for_params(cfg_mod, bp)
        df_work = df_train[core_cols].copy()
        if use_news:
            sc = bp.get("news_add_sign_confirmation")
            df_work = merge_news_shard_into_df(
                df_work,
                tag,
                wanted_news_cols=news_names,
                add_sign_confirmation=bool(sc) if sc is not None else None,
            )
        news_surv = execute_statistical_filter(df_work, news_names, cfg_mod=cfg_mod)
        combined = list(dict.fromkeys(shared + news_surv))
        survivors_by_tag[tag] = combined
        b = _feat_bucket_counts(combined)
        _log_phase11(
            f"Tag {tag}: news-Kandidaten={len(news_names)} news_survivors={b['news']} "
            f"gesamt={len(combined)} (mr_={b['mr_']} other={b['other']})."
        )
        del df_work
        clear_news_shard_frame_cache()
    return survivors_by_tag, shared, ref_tag


def build_reference_feat_cols(
    cfg_mod: Any,
    df_train: pd.DataFrame,
    *,
    news_params: dict[str, Any] | None = None,
) -> list[str]:
    """Referenz-``FEAT_COLS`` aus ``SEED_PARAMS`` + Makro-Spalten (vor Optuna)."""
    sp = dict(getattr(cfg_mod, "SEED_PARAMS", None) or {})
    np_ = dict(news_params or {})
    cols = cfg_mod.build_feature_cols(
        int(sp.get("rsi_window", 21)),
        int(sp.get("bb_window", 20)),
        int(sp.get("sma_window", 50)),
        int(np_.get("news_mom_w", sp.get("news_mom_w", 5))),
        int(np_.get("news_vol_ma", sp.get("news_vol_ma", 10))),
        int(np_.get("news_tone_roll", sp.get("news_tone_roll", 3))),
        np_.get("news_extra_zscore_w", sp.get("news_extra_zscore_w")),
        np_.get("news_extra_tone_accel", sp.get("news_extra_tone_accel")),
        np_.get("news_extra_macro_sec_diff", sp.get("news_extra_macro_sec_diff")),
        btc_momentum_z_window=sp.get("btc_momentum_z_window"),
        market_breadth_z_window=sp.get("market_breadth_z_window"),
        rel_momentum_window=sp.get("rel_momentum_window"),
        adr_window=sp.get("adr_window"),
        breakout_lookback_window=sp.get("breakout_lookback_window"),
        vcp_window=sp.get("vcp_window"),
        btc_corr_window=sp.get("btc_corr_window"),
        yz_vol_window=sp.get("yz_vol_window"),
        downside_vol_window=sp.get("downside_vol_window"),
        ret_moment_window=sp.get("ret_moment_window"),
        amihud_window=sp.get("amihud_window"),
        vcp_lower_low_window=sp.get("vcp_lower_low_window"),
        breakout_volume_trigger_z=sp.get("breakout_volume_trigger_z"),
        news_add_sign_confirmation=np_.get(
            "news_add_sign_confirmation", sp.get("news_add_sign_confirmation")
        ),
    )
    return append_macro_regime_vol_numeric_cols(cols, df_train)


def try_load_statistical_pre_prune_artifact(cfg_mod: Any) -> dict[str, Any] | None:
    """Lädt gespeichertes Phase-11-Artefakt auf ``cfg_mod`` (kein erneutes 18×-MI)."""
    if not bool(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_REUSE_ARTIFACT", True)):
        return None
    art_dir = Path(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_DIR", "data"))
    art_name = str(
        getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_ARTIFACT_NAME", "statistical_pre_prune_v1.json")
    )
    art_path = art_dir / art_name
    if not art_path.is_file():
        return None
    try:
        payload = json.loads(art_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log_phase11(f"Artefakt lesen fehlgeschlagen ({art_path}): {exc}")
        return None
    if not isinstance(payload, dict):
        return None
    survivors = payload.get("survivors")
    by_tag = payload.get("survivors_by_tag")
    if not survivors or not isinstance(survivors, list):
        return None
    if not isinstance(by_tag, dict) or not by_tag:
        return None
    expected_tags = sorted(_available_news_shard_tags(cfg_mod))
    if expected_tags and set(by_tag.keys()) != set(expected_tags):
        _log_phase11(
            f"Artefakt veraltet: Tags {set(by_tag.keys()) ^ set(expected_tags)} — Rebuild."
        )
        return None
    _cfg_keys = {
        "max_sentinel_frac": "STATISTICAL_PRE_PRUNE_MAX_SENTINEL_FRAC",
        "mi_top_k": "STATISTICAL_PRE_PRUNE_MI_TOP_K",
    }
    for art_key, cfg_key in _cfg_keys.items():
        if art_key in payload and hasattr(cfg_mod, cfg_key):
            if float(payload[art_key]) != float(getattr(cfg_mod, cfg_key)):
                _log_phase11(f"Artefakt Config-Drift ({art_key}) — Rebuild.")
                return None
    cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG = {
        str(k): [str(x) for x in v] for k, v in by_tag.items()
    }
    cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS = [str(x) for x in survivors]
    artifact = {k: v for k, v in payload.items() if k not in ("survivors", "survivors_by_tag", "shared_non_news_survivors")}
    cfg_mod._STATISTICAL_PRE_PRUNE_ARTIFACT = artifact
    _log_phase11(
        f"Artefakt wiederverwendet: {art_path} "
        f"({len(by_tag)} Tags, Referenz {artifact.get('news_tag_reference')!r}, "
        f"|survivors|={len(survivors)})."
    )
    return artifact


def run_statistical_pre_pruning(cfg_mod: Any) -> dict[str, Any]:
    """
    Phase 11: nur ``df_train`` (BASE). Setzt Survivors + Artefakt auf ``cfg_mod``.
    """
    if bool(getattr(cfg_mod, "SCORING_ONLY", False)):
        print("[Phase 11] übersprungen (SCORING_ONLY).", flush=True)
        return {}
    if not bool(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_ENABLED", False)):
        return {}

    df_train = getattr(cfg_mod, "df_train", None)
    if df_train is None or len(df_train) == 0:
        raise ValueError("Phase 11: df_train leer — zuerst run_data_download_and_split.")

    print("\n" + "=" * 60, flush=True)
    print("Phase 11: Statistisches Feature Pre-Pruning (nur BASE)", flush=True)
    print("=" * 60, flush=True)

    cached = try_load_statistical_pre_prune_artifact(cfg_mod)
    if cached:
        return cached

    sp = dict(getattr(cfg_mod, "SEED_PARAMS", None) or {})
    df_work = df_train.copy()
    bp_news = resolve_news_params_for_phase11(cfg_mod, sp)
    if getattr(cfg_mod, "USE_NEWS_SENTIMENT", False) and _available_news_shard_tags(cfg_mod):
        df_work = merge_news_shard_from_best_params(df_work, bp_news)

    full_features = build_reference_feat_cols(cfg_mod, df_work)
    cfg_mod.FEAT_COLS_FULL_REFERENCE = list(full_features)
    tag_preview = news_tag_from_params(bp_news, cfg_mod)
    d0 = pd.to_datetime(df_work["Date"]).min()
    d1 = pd.to_datetime(df_work["Date"]).max()
    _log_phase11(
        f"Referenz-FEAT_COLS={len(full_features)} (News-Tag SEED={tag_preview}), "
        f"Kalender BASE {d0.date()} … {d1.date()}."
    )
    b_ref = _feat_bucket_counts(full_features)
    _log_phase11(
        f"Referenz-Buckets: news={b_ref['news']} mr_={b_ref['mr_']} "
        f"regime_={b_ref['regime_']} other={b_ref['other']}"
    )

    per_tag = bool(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_PER_TAG_NEWS", True))
    available_tags = sorted(_available_news_shard_tags(cfg_mod))
    use_per_tag = (
        per_tag
        and bool(getattr(cfg_mod, "USE_NEWS_SENTIMENT", False))
        and len(available_tags) > 0
    )

    if use_per_tag:
        _log_phase11(
            f"Variante A: Pre-Prune pro News-Tag ({len(available_tags)} Shards im Manifest)."
        )
        survivors_by_tag, shared_non_news, ref_tag = _run_phase11_per_tag_news(
            cfg_mod,
            df_train,
            sp=sp,
            tags=available_tags,
        )
        cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG = survivors_by_tag
        surviving = list(survivors_by_tag.get(ref_tag) or next(iter(survivors_by_tag.values())))
        cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS = list(surviving)
        tag = ref_tag
        n_news_union = len(
            {
                c
                for cols in survivors_by_tag.values()
                for c in cols
                if str(c).startswith("news_")
            }
        )
        artifact = {
            "phase": 11,
            "schema_version": 2,
            "per_tag_news": True,
            "n_tags": len(survivors_by_tag),
            "n_shared_non_news": len(shared_non_news),
            "n_news_survivors_union": n_news_union,
            "n_before": len(full_features),
            "n_after_reference_tag": len(surviving),
            "news_tag_reference": tag,
            "max_sentinel_frac": float(
                getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MAX_SENTINEL_FRAC", 0.98)
            ),
            "mi_top_k": int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_TOP_K", 150)),
            "bucket_counts_after_reference": _feat_bucket_counts(surviving),
            "survivors_by_tag_counts": {k: len(v) for k, v in survivors_by_tag.items()},
        }
        payload = {
            **artifact,
            "shared_non_news_survivors": list(shared_non_news),
            "survivors_by_tag": survivors_by_tag,
            "survivors": surviving,
        }
    else:
        surviving = execute_statistical_filter(df_work, full_features, cfg_mod=cfg_mod)
        cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS = list(surviving)
        cfg_mod.FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG = None
        tag = news_tag_from_params(bp_news, cfg_mod)
        artifact = {
            "phase": 11,
            "schema_version": 1,
            "per_tag_news": False,
            "n_before": len(full_features),
            "n_after": len(surviving),
            "news_tag_reference": tag,
            "max_sentinel_frac": float(
                getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MAX_SENTINEL_FRAC", 0.98)
            ),
            "mi_top_k": int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MI_TOP_K", 150)),
            "n_dropped_sparsity": len(full_features) - len(surviving),
            "bucket_counts_after": _feat_bucket_counts(surviving),
        }
        payload = {**artifact, "survivors": surviving}

    cfg_mod._STATISTICAL_PRE_PRUNE_ARTIFACT = artifact

    art_dir = Path(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_DIR", "data"))
    art_dir.mkdir(parents=True, exist_ok=True)
    art_name = str(
        getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_ARTIFACT_NAME", "statistical_pre_prune_v1.json")
    )
    art_path = art_dir / art_name
    art_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if use_per_tag:
        _log_phase11(
            f"Survivors: BY_TAG ({len(survivors_by_tag)} Tags), "
            f"Referenz {tag!r} → cfg.FEAT_COLS_STATISTICAL_SURVIVORS ({len(surviving)} Spalten)."
        )
    else:
        _log_phase11(
            f"Survivors gesetzt: cfg.FEAT_COLS_STATISTICAL_SURVIVORS ({len(surviving)} Spalten)."
        )
    _log_phase11(f"Artefakt: {art_path}")
    return artifact


__all__ = [
    "execute_statistical_filter",
    "run_statistical_pre_pruning",
    "build_reference_feat_cols",
    "news_params_from_tag",
    "resolve_statistical_survivors_for_tag",
    "try_load_statistical_pre_prune_artifact",
]
