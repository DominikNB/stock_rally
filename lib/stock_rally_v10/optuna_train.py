"""stock_rally_v10 — Optuna / Base-XGB (Pipeline-Modul)."""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import ta
import xgboost as xgb
from sklearn.metrics import average_precision_score

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.features import merge_news_shard_into_df, merge_news_survivors_into_df
from lib.stock_rally_v10.helpers import make_focal_objective
from lib.stock_rally_v10.target import rebuild_target_for_train


# Gates: cfg.OPT_MIN_PRECISION_BASE; Phase 5 nutzt cfg.OPT_MIN_PRECISION (= THRESHOLD-Ziel)
_OPT_MIN_PRECISION = cfg.OPT_MIN_PRECISION_BASE
# Maximum consecutive FP signals per ticker before a trial is penalised
_OPT_MAX_CONSEC_FP = 4


def _base_optuna_checkpoint_path(cfg_mod: Any) -> Path:
    p = getattr(cfg_mod, "BASE_OPTUNA_CHECKPOINT_PATH", None)
    if p is None:
        return Path("models") / "base_optuna_checkpoint.joblib"
    return Path(p)


def try_load_base_optuna_checkpoint(cfg_mod: Any) -> bool:
    """Lädt ``base_optuna_best_params`` von Platte, wenn im Namespace noch keiner steht.

    Ermöglicht Resume/Seed für den nächsten Base-Optuna-Lauf ohne vollständiges
    ``scoring_artifacts.joblib`` (Meta läuft erst später).
    """
    if getattr(cfg_mod, "base_optuna_best_params", None):
        return False
    if not bool(getattr(cfg_mod, "BASE_OPTUNA_CHECKPOINT_LOAD", True)):
        return False
    path = _base_optuna_checkpoint_path(cfg_mod)
    if not path.is_file():
        return False
    try:
        blob = joblib.load(path)
    except Exception as exc:
        print(f"[Optuna] Base-Checkpoint lesen fehlgeschlagen ({path}): {exc}", flush=True)
        return False
    if not isinstance(blob, dict):
        return False
    bp = blob.get("base_optuna_best_params")
    if not isinstance(bp, dict) or not bp:
        return False
    cfg_mod.base_optuna_best_params = dict(bp)
    bv = blob.get("base_optuna_best_value")
    if bv is not None:
        try:
            cfg_mod.base_optuna_best_value = float(bv)
        except (TypeError, ValueError):
            cfg_mod.base_optuna_best_value = None
    print(
        f"[Optuna] Base-Checkpoint geladen: {path.resolve()} "
        f"(best_value={getattr(cfg_mod, 'base_optuna_best_value', None)!r}, |params|={len(bp)}).",
        flush=True,
    )
    return True


def save_base_optuna_checkpoint(cfg_mod: Any) -> None:
    """Schreibt Base-Optuna-Bestwerte direkt nach Phase 1 (atomar)."""
    if not bool(getattr(cfg_mod, "BASE_OPTUNA_CHECKPOINT_SAVE", True)):
        return
    bp = getattr(cfg_mod, "base_optuna_best_params", None)
    if not isinstance(bp, dict) or not bp:
        return
    path = _base_optuna_checkpoint_path(cfg_mod)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "base_optuna_best_params": dict(bp),
        "base_optuna_best_value": getattr(cfg_mod, "base_optuna_best_value", None),
    }
    try:
        joblib.dump(payload, tmp)
        tmp.replace(path)
    except Exception as exc:
        print(f"[Optuna] Base-Checkpoint schreiben fehlgeschlagen ({path}): {exc}", flush=True)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return
    print(f"[Optuna] Base-Checkpoint gespeichert: {path.resolve()}", flush=True)
# Walk-Forward-Fold: zu wenig Zeilen/Positive zum sinnvollen nested Threshold + Val-Test
# (schlägt sichtbar vs. 0,03-0,2 Kalibrier-Lücke; Optuna-Pruner sieht niedrigen report-Schritt)
_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA = -2.0


def _auto_scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost-Klassengewicht aus Neg/Pos-Verhältnis (>=1.0)."""
    yy = np.asarray(y, dtype=np.int8)
    n_pos = int((yy == 1).sum())
    n_neg = int((yy == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(max(1.0, n_neg / n_pos))


_NEWS_SUBSET_TAG_RE = re.compile(
    r"^news_(?:macro|sec|cross)_(\d+)_(\d+)_(\d+)(?:_|$)",
    re.IGNORECASE,
)


def _feature_subset_smoothness_primary(col: str) -> int:
    """Skalar: größer ≈ längere / schwerere Glättung — für Katalog-Reihenfolge (kurz → lang)."""
    s = str(col)
    m = _NEWS_SUBSET_TAG_RE.match(s)
    if m:
        mom_w, vol_ma, tone_roll = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Tripel-Fenster: Vol-MA und Tone-Roll dominieren die „Länge“; Mom ist zweites Gewicht.
        return vol_ma * 100 + tone_roll * 10 + mom_w
    toks = [int(x) for x in re.findall(r"\d+", s)]
    if not toks:
        return 0
    # Mehrfenster-Spalten (z. B. vcp_…_10_60d): typisch max. Fenster = relevanteste Glättungslänge.
    return int(max(toks))


def _feature_subset_family_order(col: str) -> int:
    """Fein-Sortierung: ähnliche Familien blockweise, News/Makro eher ans Ende (bei gleichem Score)."""
    s = str(col)
    if s.startswith("news_"):
        return 5
    if s.startswith("mr_") or s.startswith("regime_"):
        return 4
    if s.startswith("rsi") or s.startswith("bb_") or s.startswith("sma") or "sma_" in s:
        return 0
    if s.startswith("corr_") or "momentum" in s or "breadth" in s:
        return 1
    if "vol" in s or "yz_" in s or "amihud" in s or "drawdown" in s:
        return 2
    if "vcp" in s or "breakout" in s or "dist_to_prior" in s or "blue_sky" in s:
        return 3
    return 2


def _feature_subset_catalog_columns(df: pd.DataFrame) -> list[str]:
    """Numerische/bool-Spalten; Reihenfolge: steigend nach Glättungs-/Fenster-Heuristik (kurz → lang).

    So haben Indizes weit hinten im Katalog im Mittel längere Rollfenster (und News-Tripel mit
    größeren Parametern liegen tendenziell später), was für den Subset-Pool semantische Nähe
    benachbarter Listplätze verbessert.
    """
    ex = getattr(cfg, "OPTUNA_FEATURE_SUBSET_EXCLUDE_COLS", frozenset())
    raw: list[str] = []
    for c in df.columns:
        if c in ex:
            continue
        if str(c).startswith("_"):
            continue
        col = df[c]
        if isinstance(col, pd.DataFrame):
            continue
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            raw.append(str(c))
    raw = sorted(set(raw))
    raw.sort(
        key=lambda name: (
            _feature_subset_smoothness_primary(name),
            _feature_subset_family_order(name),
            name,
        )
    )
    return raw


def _allocate_k_across_b_bins(k: int, b: int) -> list[int]:
    """Verteilt k auf b nicht-negative Ganzzahlen mit Summe k."""
    if b <= 0:
        return [max(0, k)]
    b = int(b)
    k = int(k)
    base = [k // b] * b
    for i in range(k % b):
        base[i] += 1
    return base


def _rotate_int_list(lst: list[int], r: int) -> list[int]:
    if not lst:
        return lst
    r %= len(lst)
    return lst[r:] + lst[:r]


def _catalog_index_bins(n_cat: int, num_bins: int) -> list[list[int]]:
    """Katalog-Indizes 0..n-1 in Eimer nach Position (≈ Glättungsstufe, Katalog ist kurz→lang sortiert)."""
    if n_cat <= 0 or num_bins <= 0:
        return []
    b_eff = min(int(num_bins), n_cat)
    bins: list[list[int]] = [[] for _ in range(b_eff)]
    for i in range(n_cat):
        b = min(b_eff - 1, (i * b_eff) // n_cat) if n_cat > 1 else 0
        bins[b].append(i)
    return bins


def _one_stratified_subset_indices(
    bins: list[list[int]],
    quotas: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    """Zieht aus jedem Bin bis zur Quote; füllt auf k mit zufälligem Rest aus dem Gesamtrest auf."""
    work = [list(x) for x in bins]
    chosen: list[int] = []
    seen_ch: set[int] = set()
    for b, q in enumerate(quotas):
        if b >= len(work) or q <= 0:
            continue
        need = min(int(q), len(work[b]))
        if need <= 0:
            continue
        pick = rng.choice(work[b], size=need, replace=False).tolist()
        for x in pick:
            if x not in seen_ch:
                chosen.append(x)
                seen_ch.add(x)
    rem = int(k) - len(chosen)
    if rem > 0:
        pool_ix = [i for bi, lst in enumerate(work) for i in lst if i not in seen_ch]
        rng.shuffle(pool_ix)
        for i in pool_ix:
            if rem <= 0:
                break
            chosen.append(i)
            seen_ch.add(i)
            rem -= 1
    return chosen[: int(k)]


def _build_feature_subset_pool(
    n_cat: int,
    k: int,
    pool_size: int,
    *,
    rng: np.random.Generator,
    max_attempts: int,
) -> list[tuple[int, ...]]:
    """K-Teilmengen (sortierte Index-Tupel), paarweise verschieden.

    Mischung aus:
    * **Stratifiziert:** K über ``OPTUNA_FEATURE_SUBSET_POOL_NUM_BINS`` Katalog-Eimer verteilen,
      Quoten-Rotation pro Listplatz → unterschiedliche Gewichte kurz vs. lang (Optuna kann Profile unterscheiden).
    * **Zufällig:** Anteil ``OPTUNA_FEATURE_SUBSET_POOL_RANDOM_FRACTION`` rein zufällige K-Sets (Exploration).
    """
    if n_cat <= 0 or k <= 0 or pool_size <= 0:
        return []
    kk = min(int(k), n_cat)
    num_bins_cfg = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_NUM_BINS", 5))
    b_eff = max(2, min(num_bins_cfg, kk, n_cat))
    frac_rand = float(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_RANDOM_FRACTION", 0.25))
    frac_rand = float(np.clip(frac_rand, 0.0, 1.0))
    bins = _catalog_index_bins(n_cat, b_eff)
    base_quotas = _allocate_k_across_b_bins(kk, b_eff)

    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    attempts = 0
    slot = 0
    while len(out) < int(pool_size) and attempts < int(max_attempts):
        attempts += 1
        if rng.random() < frac_rand:
            ix = rng.choice(n_cat, size=kk, replace=False)
            chosen = ix.tolist()
        else:
            quotas = _rotate_int_list(list(base_quotas), slot)
            slot += 1
            chosen = _one_stratified_subset_indices(bins, quotas, kk, rng)
        t = tuple(sorted(int(x) for x in chosen))
        if len(t) < kk or len(set(t)) < kk:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _expand_df_for_feature_subset_catalog(df_train: pd.DataFrame, seed_params: dict) -> pd.DataFrame:
    """Einmal News an df hängen (Survivors bevorzugt, sonst SEED_PARAMS-Tripel), damit der Katalog News enthält."""
    out = df_train.copy()
    if not (bool(getattr(cfg, "USE_NEWS_SENTIMENT", False)) and getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False)):
        return out
    art = getattr(cfg, "_NEWS_CORRELATION_PRESCREEN_ARTIFACT", None) or {}
    surv = art.get("survivors") if isinstance(art, dict) else None
    if surv:
        return merge_news_survivors_into_df(out, list(surv))
    tag = cfg.news_feat_tag(
        int(seed_params["news_mom_w"]),
        int(seed_params["news_vol_ma"]),
        int(seed_params["news_tone_roll"]),
    )
    return merge_news_shard_into_df(out, tag, wanted_news_cols=None)


def intersect_feat_cols_with_prescreen_kept(
    feat_cols: list[str],
    *,
    rsi_window: int | float,
    bb_window: int | float,
    breakout_lookback_window: int | float,
    cfg_mod: Any,
) -> list[str]:
    """Reduziert ``feat_cols`` auf Pre-Screen-``kept_features`` (ohne extra Optuna-Dimension).

    - Immer: RSI-/BB-/Blue-Sky-Pflichtspalten (wie im Subset-Pfad).
    - Alle ``news_*``: unverändert (Tripel/Survivor hängen nicht 1:1 an SHAP-Liste).
    - Sonst: nur Spalten aus ``_FEATURE_PRESCREEN_ARTIFACT['kept_features']``.
    """
    if bool(getattr(cfg_mod, "OPTUNA_FEATURE_SUBSET_POOL_ENABLED", False)):
        return feat_cols
    if not bool(getattr(cfg_mod, "OPTUNA_INTERSECT_FEAT_COLS_WITH_PRESCREEN_KEPT", False)):
        return feat_cols
    art = getattr(cfg_mod, "_FEATURE_PRESCREEN_ARTIFACT", None)
    if not isinstance(art, dict):
        return feat_cols
    kept_raw = art.get("kept_features")
    if not kept_raw:
        return feat_cols
    kept = set(str(x) for x in kept_raw)
    mandatory = {
        f"rsi_{int(rsi_window)}d",
        f"bb_pband_{int(bb_window)}",
        f"blue_sky_breakout_{int(breakout_lookback_window)}d",
    }
    out: list[str] = []
    seen: set[str] = set()
    n_in = len(feat_cols)
    for c in feat_cols:
        sc = str(c)
        if sc in seen:
            continue
        if sc in mandatory:
            out.append(c)
            seen.add(sc)
        elif sc.startswith("news_"):
            out.append(c)
            seen.add(sc)
        elif sc in kept:
            out.append(c)
            seen.add(sc)
    min_keep = int(getattr(cfg_mod, "OPTUNA_PRESCREEN_FEAT_INTERSECT_MIN_COLS", 12))
    if len(out) < min_keep:
        print(
            f"[Optuna] intersect prescreen kept: nur {len(out)} Spalten (<{min_keep}) — "
            f"ungefilterte feat_cols ({n_in}) behalten.",
            flush=True,
        )
        return feat_cols
    if len(out) < n_in:
        print(
            f"[Optuna] feat_cols ∩ Pre-Screen kept: {n_in} → {len(out)} Spalten "
            f"(|kept_features|={len(kept)}).",
            flush=True,
        )
    return out


def intersect_feat_cols_with_statistical_prune(
    feat_cols: list[str],
    *,
    cfg_mod: Any,
    trial_news_tag: str | None = None,
) -> list[str]:
    """Schnittmenge mit Phase-11-Survivors.

    Mit ``survivors_by_tag`` (Variante A): ``trial_news_tag`` wählt die passende Liste,
    sodass jedes News-Fenster in Optuna **gefilterte** News behält (nicht 0, nicht ~500 Roh).

    ``STATISTICAL_PRE_PRUNE_STRICT_NEWS_IN_OPTUNA=True``: ``news_*`` nur aus dieser Liste.
    """
    if not bool(getattr(cfg_mod, "OPTUNA_INTERSECT_FEAT_COLS_WITH_STATISTICAL_PRUNE", True)):
        return feat_cols
    from lib.stock_rally_v10.training_phases.feature_pre_pruning import (
        resolve_statistical_survivors_for_tag,
    )

    survivors = resolve_statistical_survivors_for_tag(cfg_mod, trial_news_tag)
    if not survivors:
        print(
            "[Optuna] intersect statistical prune: keine Survivors auf cfg — "
            "Phase 11 gelaufen?",
            flush=True,
        )
        return feat_cols
    strict_news = bool(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_STRICT_NEWS_IN_OPTUNA", True))
    prefixes = tuple(
        getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_WHITELIST_PREFIXES", ("mr_", "regime_"))
    )
    exact = frozenset(
        getattr(
            cfg_mod,
            "STATISTICAL_PRE_PRUNE_WHITELIST_EXACT",
            ("month", "sector_id", "gics_sector_id", "gics_industry_id"),
        )
    )
    out: list[str] = []
    seen: set[str] = set()
    n_in = len(feat_cols)
    n_news_in = sum(1 for c in feat_cols if str(c).startswith("news_"))
    for c in feat_cols:
        sc = str(c)
        if sc in seen:
            continue
        if sc.startswith("news_"):
            if strict_news:
                if sc in survivors:
                    out.append(sc)
                    seen.add(sc)
            else:
                out.append(sc)
                seen.add(sc)
        elif sc in survivors or any(sc.startswith(p) for p in prefixes) or sc in exact:
            out.append(sc)
            seen.add(sc)
    min_keep = int(getattr(cfg_mod, "STATISTICAL_PRE_PRUNE_MIN_KEEP", 80))
    if len(out) < min(12, min_keep):
        print(
            f"[Optuna] intersect statistical prune: nur {len(out)} Spalten — "
            f"ungefilterte feat_cols ({n_in}) behalten.",
            flush=True,
        )
        return feat_cols
    if len(out) < n_in:
        n_news_out = sum(1 for c in out if c.startswith("news_"))
        print(
            f"[Optuna] feat_cols ∩ Phase-11 survivors: {n_in} → {len(out)} Spalten "
            f"(|survivors|={len(survivors)}, news {n_news_in}→{n_news_out}, "
            f"strict_news={strict_news}, trial_tag={trial_news_tag!r}).",
            flush=True,
        )
    return out


def _rsi_from_close_1d(close_arr, window):
    """RSI für Anti-Peak-Filter aus Schlusskursen (ta) — unabhängig von FEAT_COLS."""
    if close_arr is None or window is None:
        return None
    w = int(window)
    if w < 2 or len(close_arr) < w + 1:
        return None
    s = pd.Series(np.asarray(close_arr, dtype=np.float64))
    r = ta.momentum.rsi(s, window=w)
    return np.asarray(r.values, dtype=np.float64)


def _peak_rsi_mask_1d(close, rsi, skip_peak, N, min_dist, max_rsi):
    """True = Tag besteht Anti-Peak-/RSI-Check (gleiche Logik wie apply_signal_filters)."""
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    if skip_peak:
        ser = pd.Series(close)
        rh = ser.rolling(int(N), min_periods=min(5, int(N))).max()
        dist_hi = (rh - ser) / (rh + 1e-12)
        not_at_peak = (dist_hi.values >= float(min_dist))
    else:
        not_at_peak = np.ones(n, dtype=bool)
    if max_rsi is None or rsi is None:
        rsi_ok = np.ones(n, dtype=bool)
    else:
        rsi = np.asarray(rsi, dtype=np.float64)
        rsi_ok = np.isfinite(rsi) & (rsi <= float(max_rsi))
    return not_at_peak & rsi_ok


def _vol_stress_mask_1d(close, vol_stress, max_vol_stress_z):
    """True = Signal besteht Vol-Stress-Hardfilter."""
    if max_vol_stress_z is None:
        return np.ones(len(close), dtype=bool)
    close = np.asarray(close, dtype=np.float64)
    vs = np.asarray(vol_stress, dtype=np.float64)
    n = len(close)
    if len(vs) != n:
        return np.ones(n, dtype=bool)
    ret1 = np.full(n, np.nan, dtype=np.float64)
    if n >= 2:
        prev = close[:-1]
        curr = close[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret1[1:] = np.where(np.isfinite(prev) & (prev > 0.0), curr / prev - 1.0, np.nan)
    s = pd.Series(vs)
    mu = s.rolling(20, min_periods=10).mean()
    sd = s.rolling(20, min_periods=10).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    bad = np.isfinite(ret1) & (ret1 > 0.0) & np.isfinite(z.values) & (z.values > float(max_vol_stress_z))
    return ~bad


def _blue_sky_weak_volume_mask_1d(blue_sky_breakout, volume_zscore, min_volume_z):
    """True = Signal besteht Blue-Sky-Volumen-Check."""
    if min_volume_z is None:
        return np.ones(len(blue_sky_breakout), dtype=bool)
    b = np.asarray(blue_sky_breakout, dtype=np.float64)
    vz = np.asarray(volume_zscore, dtype=np.float64)
    n = len(b)
    if len(vz) != n:
        return np.ones(n, dtype=bool)
    prev_vz = np.full(n, np.nan, dtype=np.float64)
    if n >= 2:
        prev_vz[1:] = vz[:-1]
    bad = (b >= 0.5) & np.isfinite(prev_vz) & (prev_vz < float(min_volume_z))
    return ~bad


def _dynamic_threshold_mask_1d(
    probs,
    base_threshold,
    vvix_ratio=None,
    rsi_arr=None,
    bb_pband_arr=None,
    vvix_trigger=None,
    rsi_trigger=None,
    bb_pband_trigger=None,
    mult1=1.0,
    mult2=1.0,
    mult3=1.0,
):
    """True = prob liegt über dynamisch erhöhtem Threshold."""
    p = np.asarray(probs, dtype=np.float64)
    n = len(p)
    thr = np.full(n, float(base_threshold), dtype=np.float64)
    if vvix_ratio is not None and vvix_trigger is not None:
        v = np.asarray(vvix_ratio, dtype=np.float64)
        if len(v) == n:
            thr = np.where(np.isfinite(v) & (v > float(vvix_trigger)), thr * float(mult1), thr)
    if rsi_arr is not None and rsi_trigger is not None:
        r = np.asarray(rsi_arr, dtype=np.float64)
        if len(r) == n:
            thr = np.where(np.isfinite(r) & (r > float(rsi_trigger)), thr * float(mult2), thr)
    if bb_pband_arr is not None and bb_pband_trigger is not None:
        b = np.asarray(bb_pband_arr, dtype=np.float64)
        if len(b) == n:
            thr = np.where(np.isfinite(b) & (b > float(bb_pband_trigger)), thr * float(mult3), thr)
    return p >= thr


def _apply_filters_cv(probs_arr, dates_arr, tickers_arr, targets_arr,
                      threshold, consecutive_days, signal_cooldown_days,
                      close_arr=None, rsi_window=None,
                      signal_skip_near_peak=True,
                      peak_lookback_days=20,
                      peak_min_dist_from_high_pct=0.012,
                      signal_max_rsi=78.0,
                      vol_stress_arr=None,
                      signal_max_vol_stress_z=None,
                      blue_sky_breakout_arr=None,
                      volume_zscore_arr=None,
                      signal_min_blue_sky_volume_z=None,
                      vvix_ratio_arr=None,
                      rsi_arr=None,
                      bb_pband_arr=None,
                      dyn_vvix_trigger=None,
                      dyn_rsi_trigger=None,
                      dyn_bb_pband_trigger=None,
                      dyn_mult_1=1.0,
                      dyn_mult_2=1.0,
                      dyn_mult_3=1.0,
                      return_details=False):
    """
    Apply consecutive + cooldown filter per ticker on a fold's val set.
    Anti-Peak/RSI: RSI aus close via rsi_window (ta); rsi_window=None → cfg.__dict__['rsi_w'].
    Returns (n_tp, n_signals, max_consec_fp).
    """
    n_tp = 0
    n_signals = 0
    max_consec_fp = 0
    n_raw_signals = 0
    signal_days: set = set()
    df_v = pd.DataFrame({
        'ticker': tickers_arr,
        'Date':   dates_arr,
        'prob':   probs_arr,
        'target': targets_arr,
    })
    if close_arr is not None:
        df_v['close'] = close_arr
    if vol_stress_arr is not None:
        df_v['vol_stress'] = vol_stress_arr
    if blue_sky_breakout_arr is not None:
        df_v['blue_sky_breakout'] = blue_sky_breakout_arr
    if volume_zscore_arr is not None:
        df_v['volume_zscore'] = volume_zscore_arr
    if vvix_ratio_arr is not None:
        df_v['vvix_ratio'] = vvix_ratio_arr
    if rsi_arr is not None:
        df_v['rsi_dyn'] = rsi_arr
    if bb_pband_arr is not None:
        df_v['bb_pband_dyn'] = bb_pband_arr

    _rw = rsi_window if rsi_window is not None else cfg.__dict__.get('rsi_w')

    for ticker, sub in df_v.groupby('ticker'):
        sub = sub.sort_values('Date').reset_index(drop=True)
        raw_mask = _dynamic_threshold_mask_1d(
            sub['prob'].values,
            threshold,
            vvix_ratio=sub['vvix_ratio'].values if 'vvix_ratio' in sub.columns else None,
            rsi_arr=sub['rsi_dyn'].values if 'rsi_dyn' in sub.columns else None,
            bb_pband_arr=sub['bb_pband_dyn'].values if 'bb_pband_dyn' in sub.columns else None,
            vvix_trigger=dyn_vvix_trigger,
            rsi_trigger=dyn_rsi_trigger,
            bb_pband_trigger=dyn_bb_pband_trigger,
            mult1=dyn_mult_1,
            mult2=dyn_mult_2,
            mult3=dyn_mult_3,
        )
        raw = raw_mask.astype(np.int8)
        n_raw_signals += int(raw.sum())
        n   = len(raw)

        # Consecutive filter: at least consecutive_days of 3 must be positive
        consec = np.zeros(n, dtype=np.int8)
        for i in range(2, n):
            if raw[i-2] + raw[i-1] + raw[i] >= consecutive_days:
                consec[i] = 1

        # Cooldown filter
        final = np.zeros(n, dtype=np.int8)
        last_sig = -999
        for i in range(n):
            if consec[i] == 1 and (i - last_sig) >= signal_cooldown_days:
                final[i] = 1
                last_sig = i

        if close_arr is not None and 'close' in sub.columns:
            rsi_sub = _rsi_from_close_1d(sub['close'].values, _rw)
            mask_ok = _peak_rsi_mask_1d(
                sub['close'].values, rsi_sub,
                signal_skip_near_peak,
                peak_lookback_days,
                peak_min_dist_from_high_pct,
                signal_max_rsi,
            )
            for i in range(n):
                if final[i] == 1 and not mask_ok[i]:
                    final[i] = 0
            if signal_max_vol_stress_z is not None and 'vol_stress' in sub.columns:
                stress_ok = _vol_stress_mask_1d(
                    sub['close'].values,
                    sub['vol_stress'].values,
                    signal_max_vol_stress_z,
                )
                for i in range(n):
                    if final[i] == 1 and not stress_ok[i]:
                        final[i] = 0
            if (
                signal_min_blue_sky_volume_z is not None
                and 'blue_sky_breakout' in sub.columns
                and 'volume_zscore' in sub.columns
            ):
                blue_ok = _blue_sky_weak_volume_mask_1d(
                    sub['blue_sky_breakout'].values,
                    sub['volume_zscore'].values,
                    signal_min_blue_sky_volume_z,
                )
                for i in range(n):
                    if final[i] == 1 and not blue_ok[i]:
                        final[i] = 0

        sig_mask_idx = (final == 1)
        sig_targets = sub.loc[sig_mask_idx, 'target'].values
        n_tp      += int(sig_targets.sum())
        n_signals += int(sig_targets.size)

        # Tagesabdeckung: alle Tage mit >= 1 finalem Signal (über alle Ticker zusammen).
        # Wird vom tp_precision-Pfad gebraucht, um avg_signals_per_day korrekt zu loggen.
        if int(sig_mask_idx.sum()) > 0:
            sig_days_t = pd.to_datetime(
                sub.loc[sig_mask_idx, 'Date'], errors='coerce'
            ).dt.normalize().dropna()
            signal_days.update(sig_days_t.tolist())

        # Max consecutive FP run for this ticker
        run = 0
        for is_tp in sig_targets:
            if is_tp == 0:
                run += 1
                if run > max_consec_fp:
                    max_consec_fp = run
            else:
                run = 0

    if return_details:
        details = {
            "n_raw_signals": int(n_raw_signals),
            "n_final_signals": int(n_signals),
            "n_filtered_out": int(max(0, n_raw_signals - n_signals)),
            "n_signal_days": int(len(signal_days)),
        }
        return n_tp, n_signals, max_consec_fp, details
    return n_tp, n_signals, max_consec_fp


def _score_tp_precision_fold(
    probs_arr,
    dates_arr,
    tickers_arr,
    targets_arr,
    threshold,
    consecutive_days,
    signal_cooldown_days,
    *,
    close_arr=None,
    rsi_window=None,
    signal_skip_near_peak=True,
    peak_lookback_days=20,
    peak_min_dist_from_high_pct=0.012,
    signal_max_rsi=78.0,
    vol_stress_arr=None,
    signal_max_vol_stress_z=None,
    blue_sky_breakout_arr=None,
    volume_zscore_arr=None,
    signal_min_blue_sky_volume_z=None,
    vvix_ratio_arr=None,
    rsi_arr=None,
    bb_pband_arr=None,
    dyn_vvix_trigger=None,
    dyn_rsi_trigger=None,
    dyn_bb_pband_trigger=None,
    dyn_mult_1=1.0,
    dyn_mult_2=1.0,
    dyn_mult_3=1.0,
):
    """Einen Walk-Forward-Fold bewerten (Base-Optuna).

    Rückgabe ``(score, n_tp, n_signals, max_consec_fp)``. Zuerst wie bisher: max-FP-Strafe,
    ohne Signale weicher Kalibrierungs-Score. Mit Signalen: **Precision-Gate**
    ``n_tp / n_signals >= OPT_MIN_PRECISION_BASE``; bei erfülltem Gate ist die Belohnung
    **Recall in Prozentpunkten**: ``score = 100 · n_tp / n_pos`` mit ``n_pos`` = Anzahl Zeilen
    mit ``target == 1`` im Fold (wie viele positive Targets im Bewertungsausschnitt, wie viele
    davon an einem Signaltag mit Treffer abgedeckt). Sonst Penalty ``precision - 1``.
    """
    n_tp, n_sig, max_cfp = _apply_filters_cv(
        probs_arr,
        dates_arr,
        tickers_arr,
        targets_arr,
        float(threshold),
        int(consecutive_days),
        int(signal_cooldown_days),
        close_arr=close_arr,
        rsi_window=rsi_window,
        signal_skip_near_peak=signal_skip_near_peak,
        peak_lookback_days=peak_lookback_days,
        peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
        signal_max_rsi=signal_max_rsi,
        vol_stress_arr=vol_stress_arr,
        signal_max_vol_stress_z=signal_max_vol_stress_z,
        blue_sky_breakout_arr=blue_sky_breakout_arr,
        volume_zscore_arr=volume_zscore_arr,
        signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
        vvix_ratio_arr=vvix_ratio_arr,
        rsi_arr=rsi_arr,
        bb_pband_arr=bb_pband_arr,
        dyn_vvix_trigger=dyn_vvix_trigger,
        dyn_rsi_trigger=dyn_rsi_trigger,
        dyn_bb_pband_trigger=dyn_bb_pband_trigger,
        dyn_mult_1=dyn_mult_1,
        dyn_mult_2=dyn_mult_2,
        dyn_mult_3=dyn_mult_3,
    )
    yy = np.asarray(targets_arr, dtype=np.int8)
    n_pos = int(np.sum(yy == 1))
    if max_cfp > _OPT_MAX_CONSEC_FP:
        return float(-2.0 - (max_cfp - _OPT_MAX_CONSEC_FP) * 0.1), int(n_tp), int(n_sig), int(max_cfp)
    if n_sig == 0:
        p = np.clip(np.asarray(probs_arr, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        yy = np.asarray(targets_arr, dtype=np.int8)
        pos_m = yy == 1
        neg_m = yy == 0
        if pos_m.any() and neg_m.any():
            score = float(np.mean(p[pos_m]) - np.mean(p[neg_m]))
        elif pos_m.any():
            score = float(np.mean(p[pos_m]) - 0.5)
        elif neg_m.any():
            score = float(0.5 - np.mean(p[neg_m]))
        else:
            score = 0.0
        return score, int(n_tp), int(n_sig), int(max_cfp)
    precision = float(n_tp) / float(n_sig)
    if precision >= _OPT_MIN_PRECISION:
        if n_pos <= 0:
            return 0.0, int(n_tp), int(n_sig), int(max_cfp)
        recall_pct = 100.0 * float(n_tp) / float(n_pos)
        return float(recall_pct), int(n_tp), int(n_sig), int(max_cfp)
    return float(precision - 1.0), int(n_tp), int(n_sig), int(max_cfp)


def _pick_threshold_nested_base(
    seed_threshold,
    *,
    probs_cal,
    dates_cal,
    tickers_cal,
    y_cal,
    consecutive_days,
    signal_cooldown_days,
    close_cal=None,
    rsi_window=None,
    signal_skip_near_peak=True,
    peak_lookback_days=20,
    peak_min_dist_from_high_pct=0.012,
    signal_max_rsi=78.0,
    vol_stress_cal=None,
    signal_max_vol_stress_z=None,
    blue_sky_cal=None,
    volume_z_cal=None,
    signal_min_blue_sky_volume_z=None,
    vvix_ratio_cal=None,
    rsi_cal=None,
    bb_cal=None,
    dyn_vvix_trigger=None,
    dyn_rsi_trigger=None,
    dyn_bb_pband_trigger=None,
    dyn_mult_1=1.0,
    dyn_mult_2=1.0,
    dyn_mult_3=1.0,
):
    thr_grid = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.05, 0.95, 19, dtype=np.float64),
                    np.array([float(seed_threshold)], dtype=np.float64),
                ]
            ),
            0.001,
            0.999,
        )
    )
    best_thr = float(seed_threshold)
    best_score = -np.inf
    for thr in thr_grid:
        fold_score, _, _, _ = _score_tp_precision_fold(
            probs_cal,
            dates_cal,
            tickers_cal,
            y_cal,
            float(thr),
            consecutive_days,
            signal_cooldown_days,
            close_arr=close_cal,
            rsi_window=rsi_window,
            signal_skip_near_peak=signal_skip_near_peak,
            peak_lookback_days=peak_lookback_days,
            peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
            signal_max_rsi=signal_max_rsi,
            vol_stress_arr=vol_stress_cal,
            signal_max_vol_stress_z=signal_max_vol_stress_z,
            blue_sky_breakout_arr=blue_sky_cal,
            volume_zscore_arr=volume_z_cal,
            signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
            vvix_ratio_arr=vvix_ratio_cal,
            rsi_arr=rsi_cal,
            bb_pband_arr=bb_cal,
            dyn_vvix_trigger=dyn_vvix_trigger,
            dyn_rsi_trigger=dyn_rsi_trigger,
            dyn_bb_pband_trigger=dyn_bb_pband_trigger,
            dyn_mult_1=dyn_mult_1,
            dyn_mult_2=dyn_mult_2,
            dyn_mult_3=dyn_mult_3,
        )
        if (fold_score > best_score) or (
            np.isclose(fold_score, best_score)
            and abs(float(thr) - float(seed_threshold)) < abs(best_thr - float(seed_threshold))
        ):
            best_score = float(fold_score)
            best_thr = float(thr)
    return float(best_thr), float(best_score)


def optimize_xgb(df_train, n_trials=None, seed_params=cfg.SEED_PARAMS):
    """
    Hyperparameter optimisation via Optuna with Walk-Forward temporal CV.

    Jointly optimises (depending on cfg.OPT_MODEL_HYPERPARAMS):
      wenn cfg.opt_optimize_y_targets(): Rally-/Label-Parameter (return_window, …); sonst feste Band-Regel
      immer (je nach cfg): consecutive/cooldown, ggf. Feature-Fenster inkl. News
      optional: cfg.OPTUNA_FEATURE_SUBSET_POOL_ENABLED — feat_cols aus vorgefertigtem Pool
      von K-Teilmengen (``feature_subset_id``) statt ``build_feature_cols``
      optional: cfg.OPTUNA_INTERSECT_FEAT_COLS_WITH_PRESCREEN_KEPT — ohne Subset-Pool:
      ``feat_cols`` nach jedem ``build_feature_cols`` auf Pre-Screen ``kept_features`` schneiden
      (kein zusätzlicher Optuna-Dimension; Voraussetzung: Pre-Screen-Artefakt).
      cfg.OPT_MODEL_HYPERPARAMS=True:  auch XGBoost-Hyperparameter (Option A)
      cfg.OPT_MODEL_HYPERPARAMS=False: Modell-HPs aus cfg.SEED_PARAMS (Option B)

    Nicht final festgelegt hier (wird in späteren Phasen überschrieben):
      - ``base_eval_threshold`` als Seed; pro Fold wird die productive Schwelle nested
        auf inner-cal gewählt und erst dann auf outer-val bewertet.
      - Folds mit zu wenig Val-/Cal-Positiven (u. a. <2) u. a.: Straf-Score
        ``_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA``, ``trial.report`` + user_attrs, kein stilles Weglassen.
      - Anti-Peak/RSI: fest aus ``seed_params`` (Phase 4 Meta-Optuna optimiert diese Werte).
      - Nach dem Study: ``cfg.base_optuna_best_*`` und sofort ``save_base_optuna_checkpoint`` (Datei
        ``cfg.BASE_OPTUNA_CHECKPOINT_PATH``); vor dem Study: ``try_load_base_optuna_checkpoint`` wenn
        noch kein ``base_optuna_best_params`` im Namespace liegt.

    Objective (compound, precision-first, recall-reward):
      - Hard gate: per-ticker max consecutive FP run <= _OPT_MAX_CONSEC_FP
      - Soft gate: filter-Precision TP/Signals >= cfg.OPT_MIN_PRECISION_BASE (pro Fold)
      - Reward:    100·(n_TP / n_Pos) = Recall in Prozentpunkten, sobald Gate erfüllt
        (n_Pos = Zeilen mit target==1 im Fold); sonst Penalty (precision−1)
      - Pro abgeschlossenem Trial: u. a. ``trial.user_attrs["tp_n"]`` = Summe ``n_TP`` über alle
        bewerteten WF-Val-Folds (gleich ``base_wf_n_tp_sum``).

    Returns best_params dict.
    """
    if n_trials is None:
        n_trials = int(cfg.N_OPTUNA_TRIALS)
    else:
        n_trials = int(n_trials)
    if n_trials <= 0:
        if try_load_base_optuna_checkpoint(cfg):
            best = dict(cfg.base_optuna_best_params)
            seed_params = dict(getattr(cfg, "SEED_PARAMS", None) or {})
            for k, v in seed_params.items():
                if k not in best:
                    best[k] = v
            if not cfg.opt_optimize_y_targets():
                _, _wh, _rt, _, _ld, _ed, _ = cfg.fixed_y_rule_params()
                best["return_window"] = _wh
                best["rally_threshold"] = _rt
                best["lead_days"] = _ld
                best["entry_days"] = _ed
                best["min_rally_tail_days"] = 5
            cfg.base_optuna_best_params = dict(best)
            print(
                "[Optuna] Phase 1 übersprungen (N_OPTUNA_TRIALS<=0) — "
                f"Base-Checkpoint übernommen (|params|={len(best)}).",
                flush=True,
            )
            return best
        raise ValueError(
            "N_OPTUNA_TRIALS<=0, aber kein gültiger Base-Checkpoint "
            f"({ _base_optuna_checkpoint_path(cfg).resolve()!s })."
        )
    wf = getattr(cfg, "OPTUNA_WF_SPLITS", None)
    wf = int(wf) if wf is not None else cfg.N_WF_SPLITS
    if wf < 1:
        wf = cfg.N_WF_SPLITS
    _subset = bool(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_ENABLED", False))
    catalog: list[str] = []
    pool: list[tuple[int, ...]] = []
    df_source = df_train
    if _subset:
        print(
            "Optuna Phase 1: OPTUNA_FEATURE_SUBSET_POOL_ENABLED=True — "
            "feat_cols = K Spalten aus vorgefertigtem Pool (+ Pflicht-Spalten für Filter).",
            flush=True,
        )
        print(
            "  → Subset-Katalog: merge News in df_train (einmalig; bei vielen Zeilen oft 2–15+ min, "
            "ohne Zwischen-Logs aus features.py) …",
            flush=True,
        )
        _df_cat = _expand_df_for_feature_subset_catalog(df_train, seed_params)
        print(
            f"  → Subset-Katalog-DataFrame fertig: {len(_df_cat):,} Zeilen × {len(_df_cat.columns)} Spalten.",
            flush=True,
        )
        catalog = _feature_subset_catalog_columns(_df_cat)
        _k_sub = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_K", 50))
        _ps_sub = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_SIZE", 10000))
        _mx_sub = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_MAX_ATTEMPTS", 500000))
        _sd_sub = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_SEED", 42))
        _nb_sub = int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_NUM_BINS", 5))
        _fr_sub = float(np.clip(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_RANDOM_FRACTION", 0.25), 0.0, 1.0))
        _rng_sub = np.random.default_rng(_sd_sub + int(getattr(cfg, "RANDOM_STATE", 42)))
        pool = _build_feature_subset_pool(
            len(catalog), _k_sub, _ps_sub, rng=_rng_sub, max_attempts=_mx_sub
        )
        _min_pool = min(100, _ps_sub)
        if len(pool) < _min_pool:
            raise ValueError(
                f"[Optuna] Feature-Subset-Pool zu klein: |pool|={len(pool)} < {_min_pool}. "
                f"|Katalog|={len(catalog)}, K={_k_sub}. K oder Pool-Größe anpassen, "
                f"oder OPTUNA_FEATURE_SUBSET_POOL_MAX_ATTEMPTS erhöhen."
            )
        print(
            f"  → Subset-Pool: K={_k_sub}, |Katalog|={len(catalog):,}, |Pool|={len(pool):,} "
            f"(Katalog: kurz→lang nach Glättungs-/Fenster-Heuristik; Pool: "
            f"{100.0 * (1.0 - _fr_sub):.0f}% stratifiziert über {_nb_sub} Katalog-Eimer mit rotierenden Quoten, "
            f"{100.0 * _fr_sub:.0f}% rein zufällig — je Eintrag K verschiedene Indizes).",
            flush=True,
        )
        df_source = _df_cat

    _ps_ix = bool(getattr(cfg, "OPTUNA_INTERSECT_FEAT_COLS_WITH_PRESCREEN_KEPT", False))
    _art_pf = getattr(cfg, "_FEATURE_PRESCREEN_ARTIFACT", None)
    if _ps_ix and not _subset:
        if isinstance(_art_pf, dict) and _art_pf.get("kept_features"):
            print(
                "Optuna Phase 1: OPTUNA_INTERSECT_FEAT_COLS_WITH_PRESCREEN_KEPT=True — "
                "feat_cols = build_feature_cols ∩ Pre-Screen kept_features (+ Pflicht + news_*).",
                flush=True,
            )
        else:
            print(
                "[Optuna] OPTUNA_INTERSECT_FEAT_COLS_WITH_PRESCREEN_KEPT=True, aber kein "
                "kept_features im Artefakt — Schnitt pro Trial deaktiviert.",
                flush=True,
            )

    all_dates = np.sort(df_source["Date"].unique())
    n_dates = len(all_dates)
    min_train = int(n_dates * 0.40)
    fold_size = max(1, (n_dates - min_train) // wf)
    _opt_y = cfg.opt_optimize_y_targets()
    if _opt_y:
        _trial_hint = (
            f"pro Trial: Rally-/Label-Parameter variabel → rebuild_target, dann {wf}× WF-XGB"
        )
    else:
        _trial_hint = (
            f"pro Trial: OPT_OPTIMIZE_Y_TARGETS=False (feste Target-Band-Regel); "
            f"rebuild_target wendet dieselbe Regel an, dann {wf}× WF-XGB"
        )
    _ntr = getattr(cfg, "_tickers_for_run", None)
    _ntr_n = len(_ntr) if _ntr is not None else None
    _nu_df = int(df_source["ticker"].nunique())
    print(
        f'Optuna Phase 1: TRAIN {len(df_source):,} Zeilen, {_nu_df} Ticker, '
        f'{n_dates} Kalendertage → {n_trials} Trials × {wf} WF-Folds '
        f"({_trial_hint}; oft dominiert das die Laufzeit).",
        flush=True,
    )
    if _ntr_n is not None:
        print(
            f"  → Abgleich Universum: cfg._tickers_for_run = {_ntr_n} Ticker "
            f"(wenn << ALL_TICKERS, sollten Zeilen hier ~ proportional kleiner sein).",
            flush=True,
        )
    if not _opt_y:
        print(cfg.describe_target_rule_text(), flush=True)
    else:
        print(
            "  → Labels: Positive-Rate aus create_target (Pipeline) ist nur Baseline; "
            "hier wird pro Trial mit rebuild_target_for_train neu gelabelt.",
            flush=True,
        )

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df_base = df_source.copy()
    df_base["_date_idx"] = df_base["Date"].map(date_to_idx)

    def objective(trial):
        _t_trial = time.perf_counter()
        # ── Rally-/Label-Params (nur wenn opt_optimize_y_targets() True) ──────────
        if _opt_y:
            return_window   = trial.suggest_int(  'return_window',   3,    12)
            rally_threshold = trial.suggest_float('rally_threshold', 0.06, 0.15)
            lead_days            = trial.suggest_int('lead_days',            1, 3)
            entry_days           = trial.suggest_int('entry_days',           1, 3)
            min_rally_tail_days  = trial.suggest_int('min_rally_tail_days',  3, 5)
        else:
            _, return_window, rally_threshold, _, lead_days, entry_days, _ = (
                cfg.fixed_y_rule_params()
            )
            min_rally_tail_days = 5
        # ── Post-processing params ─────────────────────────────────────────
        consecutive_days     = trial.suggest_int('consecutive_days',     1, 3)
        signal_cooldown_days = trial.suggest_int('signal_cooldown_days', 1, 10)
        # Schwelle: Seed wird pro Fold auf inner-cal nested kalibriert (wie Meta-Optuna).
        base_eval_threshold = trial.suggest_float(
            'base_eval_threshold',
            0.05,
            0.95,
        )
        signal_skip_near_peak = seed_params.get('signal_skip_near_peak', True)
        peak_lookback_days = int(seed_params.get('peak_lookback_days', 20))
        peak_min_dist_from_high_pct = float(seed_params.get('peak_min_dist_from_high_pct', 0.012))
        _sr = seed_params.get('signal_max_rsi', 78.0)
        signal_max_rsi = float(_sr) if _sr is not None else None
        _svs = seed_params.get('signal_max_vol_stress_z', getattr(cfg, 'SIGNAL_MAX_VOL_STRESS_Z', None))
        signal_max_vol_stress_z = float(_svs) if _svs is not None else None
        _bsv = seed_params.get(
            'signal_min_blue_sky_volume_z',
            getattr(cfg, 'SIGNAL_MIN_BLUE_SKY_VOLUME_Z', None),
        )
        signal_min_blue_sky_volume_z = float(_bsv) if _bsv is not None else None
        dyn_mult_1 = float(seed_params.get('mult_final_threshold_1', getattr(cfg, 'MULT_FINAL_THRESHOLD_1', 1.0)))
        dyn_mult_2 = float(seed_params.get('mult_final_threshold_2', getattr(cfg, 'MULT_FINAL_THRESHOLD_2', 1.0)))
        dyn_mult_3 = float(seed_params.get('mult_final_threshold_3', getattr(cfg, 'MULT_FINAL_THRESHOLD_3', 1.0)))
        dyn_vvix_trigger = float(
            seed_params.get('dyn_vvix_trigger', getattr(cfg, 'DYN_VVIX_TRIGGER', 8.2))
        )
        dyn_rsi_trigger = float(seed_params.get('dyn_rsi_trigger', getattr(cfg, 'DYN_RSI_TRIGGER', 75.0)))
        dyn_bb_pband_trigger = float(
            seed_params.get('dyn_bb_pband_trigger', getattr(cfg, 'DYN_BB_PBAND_TRIGGER', 1.02))
        )

        # Labels: nur bei variabler Y-Regel jedes Mal neu. Feste Band-Regel
        # (OPT_OPTIMIZE_Y_TARGETS=False): Targets sind identisch zu df_train —
        # rebuild wäre nur ein teures groupby('ticker') × N_trials ohne Nutzen.
        if _opt_y:
            df_trial = rebuild_target_for_train(
                df_base, lead_days, entry_days,
                return_window=return_window, rally_threshold=rally_threshold,
                min_rally_tail_days=min_rally_tail_days,
            )
        else:
            # Eigene Kopie: merge_news_* schreibt news_*-Spalten in-place an df_trial.
            df_trial = df_base.copy()

        # ── Model params ───────────────────────────────────────────────────
        # ── Feature-window params: always optimised (affect all base models equally) ──
        # News: Tag-Tripel + news_extra_* (Z-Score-Fenster, Accel, macro−sec) wenn cfg.USE_NEWS_SENTIMENT
        # cfg.effective_window_grid(name) liefert das vom Pre-Screen ggf. eingeschränkte
        # Grid (Original-Reihenfolge bleibt erhalten); ohne Pre-Screen-Artefakt = Original.
        rsi_w = trial.suggest_categorical('rsi_window', cfg.effective_window_grid('RSI_WINDOWS'))
        bb_w  = trial.suggest_categorical('bb_window',  cfg.effective_window_grid('BB_WINDOWS'))
        sma_w = trial.suggest_categorical('sma_window', cfg.effective_window_grid('SMA_WINDOWS'))
        btc_momentum_z_window = trial.suggest_categorical(
            'btc_momentum_z_window', cfg.effective_window_grid('BTC_MOMENTUM_Z_WINDOWS')
        )
        market_breadth_z_window = trial.suggest_categorical(
            'market_breadth_z_window', cfg.effective_window_grid('MARKET_BREADTH_Z_WINDOWS')
        )
        rel_momentum_window = trial.suggest_categorical(
            'rel_momentum_window', cfg.effective_window_grid('REL_MOMENTUM_WINDOWS')
        )
        adr_window = trial.suggest_categorical('adr_window', cfg.effective_window_grid('ADR_WINDOWS'))
        breakout_lookback_window = trial.suggest_categorical(
            'breakout_lookback_window', cfg.effective_window_grid('BREAKOUT_LOOKBACK_WINDOWS')
        )
        vcp_window = trial.suggest_categorical('vcp_window', cfg.effective_window_grid('VCP_WINDOWS'))
        btc_corr_window = trial.suggest_categorical(
            'btc_corr_window', cfg.effective_window_grid('BTC_CORR_WINDOWS')
        )
        # ── Risk / Liquidity / VCP-Lower-Low / Breakout-Volumen — pro Trial ein Pick ──
        yz_vol_window = trial.suggest_categorical(
            'yz_vol_window', cfg.effective_window_grid('YANG_ZHANG_WINDOWS')
        )
        downside_vol_window = trial.suggest_categorical(
            'downside_vol_window', cfg.effective_window_grid('DOWNSIDE_VOL_WINDOWS')
        )
        ret_moment_window = trial.suggest_categorical(
            'ret_moment_window', cfg.effective_window_grid('RET_MOMENT_WINDOWS')
        )
        amihud_window = trial.suggest_categorical(
            'amihud_window', cfg.effective_window_grid('AMIHUD_WINDOWS')
        )
        vcp_lower_low_window = trial.suggest_categorical(
            'vcp_lower_low_window', cfg.effective_window_grid('VCP_LOWER_LOW_WINDOWS')
        )
        breakout_volume_trigger_z = trial.suggest_categorical(
            'breakout_volume_trigger_z',
            cfg.effective_window_grid('BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS'),
        )
        if cfg.USE_NEWS_SENTIMENT:
            news_mom_w = trial.suggest_categorical('news_mom_w', cfg.NEWS_MOM_WINDOWS)
            news_vol_ma = trial.suggest_categorical('news_vol_ma', cfg.NEWS_VOL_MA_WINDOWS)
            news_tone_roll = trial.suggest_categorical('news_tone_roll', cfg.NEWS_TONE_ROLL_WINDOWS)
            news_extra_zscore_w = trial.suggest_categorical(
                "news_extra_zscore_w", cfg.NEWS_EXTRA_ZSCORE_WINDOWS
            )
            news_extra_tone_accel = trial.suggest_categorical('news_extra_tone_accel', cfg.NEWS_EXTRA_TONE_ACCEL_OPTIONS)
            news_extra_macro_sec_diff = trial.suggest_categorical(
                'news_extra_macro_sec_diff', cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS
            )
            news_add_sign_confirmation = trial.suggest_categorical(
                'news_add_sign_confirmation', cfg.NEWS_ADD_SIGN_CONFIRMATION_OPTIONS
            )
        else:
            news_mom_w = seed_params.get('news_mom_w', cfg.NEWS_MOM_WINDOWS[len(cfg.NEWS_MOM_WINDOWS) // 2])
            news_vol_ma = seed_params.get('news_vol_ma', cfg.NEWS_VOL_MA_WINDOWS[len(cfg.NEWS_VOL_MA_WINDOWS) // 2])
            news_tone_roll = seed_params.get('news_tone_roll', cfg.NEWS_TONE_ROLL_WINDOWS[0])
            news_extra_zscore_w = None
            news_extra_tone_accel = None
            news_extra_macro_sec_diff = None
            news_add_sign_confirmation = None

        # ── Model params: optimised (Option A) or fixed from cfg.SEED_PARAMS (Option B) ─
        if cfg.OPT_MODEL_HYPERPARAMS:
            grow_policy = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
            max_leaves  = trial.suggest_int('max_leaves', 31, 1024)
            if grow_policy == 'depthwise':
                max_leaves = 0
            params = dict(
                grow_policy      = grow_policy,
                max_leaves       = max_leaves,
                max_bin          = trial.suggest_categorical('max_bin', [64, 128, 256]),
                max_depth        = trial.suggest_int('max_depth', 3, 12),
                min_child_weight = trial.suggest_int('min_child_weight', 5, 100),
                gamma            = trial.suggest_float('gamma', 0.0, 10.0),
                reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                n_estimators     = trial.suggest_int('n_estimators', 100, 600),
                subsample        = trial.suggest_float('subsample', 0.5, 0.9),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 0.7),
            )
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
            focal_alpha = trial.suggest_float('focal_alpha', 0.05, 0.5)
        else:
            # Fixed hyperparameters from cfg.SEED_PARAMS — all base models treated equally
            params = {k: seed_params[k] for k in (
                'grow_policy', 'max_leaves', 'max_bin', 'max_depth',
                'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda',
                'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree',
            )}
            focal_gamma = seed_params['focal_gamma']
            focal_alpha = seed_params['focal_alpha']

        if _subset and pool and catalog:
            _sid = trial.suggest_categorical("feature_subset_id", list(range(len(pool))))
            _sid = int(_sid)
            _ix_tuple = pool[_sid]
            feat_cols = [catalog[i] for i in _ix_tuple]
            for _mc in (
                f"rsi_{int(rsi_w)}d",
                f"bb_pband_{int(bb_w)}",
                f"blue_sky_breakout_{int(breakout_lookback_window)}d",
            ):
                if _mc in df_trial.columns and _mc not in feat_cols:
                    feat_cols.append(_mc)
        else:
            feat_cols = cfg.build_feature_cols(
                rsi_w, bb_w, sma_w,
                news_mom_w, news_vol_ma, news_tone_roll,
                news_extra_zscore_w, news_extra_tone_accel, news_extra_macro_sec_diff,
                btc_momentum_z_window=int(btc_momentum_z_window),
                market_breadth_z_window=int(market_breadth_z_window),
                rel_momentum_window=int(rel_momentum_window),
                adr_window=int(adr_window),
                breakout_lookback_window=int(breakout_lookback_window),
                vcp_window=int(vcp_window),
                btc_corr_window=int(btc_corr_window),
                yz_vol_window=int(yz_vol_window),
                downside_vol_window=int(downside_vol_window),
                ret_moment_window=int(ret_moment_window),
                amihud_window=int(amihud_window),
                vcp_lower_low_window=int(vcp_lower_low_window),
                breakout_volume_trigger_z=float(breakout_volume_trigger_z),
                news_add_sign_confirmation=(
                    bool(news_add_sign_confirmation)
                    if news_add_sign_confirmation is not None
                    else None
                ),
            )
            if cfg.USE_NEWS_SENTIMENT and getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False):
                # News-Korrelations-Pre-Screen: ersetzt News-Spalten in feat_cols durch
                # die Tripel-übergreifenden Survivors (Mischung statt eines Tripels).
                _ncp_art = getattr(cfg, "_NEWS_CORRELATION_PRESCREEN_ARTIFACT", None)
                if (
                    bool(getattr(cfg, "NEWS_CORRELATION_PRESCREEN_ENABLED", False))
                    and _ncp_art is not None
                    and _ncp_art.get("survivors")
                ):
                    _survivors = list(_ncp_art["survivors"])
                    # feat_cols um News-Survivors erweitern und Tripel-spezifische News-Spalten ersetzen.
                    _non_news_feats = [c for c in feat_cols if not str(c).startswith("news_")]
                    feat_cols = _non_news_feats + _survivors
                    df_trial = merge_news_survivors_into_df(df_trial, _survivors)
                else:
                    _tag = cfg.news_feat_tag(news_mom_w, news_vol_ma, news_tone_roll)
                    _need_news = [c for c in feat_cols if str(c).startswith("news_")]
                    df_trial = merge_news_shard_into_df(
                        df_trial,
                        _tag,
                        wanted_news_cols=_need_news,
                        add_sign_confirmation=(
                            bool(news_add_sign_confirmation)
                            if news_add_sign_confirmation is not None
                            else None
                        ),
                    )
        feat_cols = intersect_feat_cols_with_prescreen_kept(
            feat_cols,
            rsi_window=rsi_w,
            bb_window=bb_w,
            breakout_lookback_window=breakout_lookback_window,
            cfg_mod=cfg,
        )
        _trial_news_tag = (
            cfg.news_feat_tag(news_mom_w, news_vol_ma, news_tone_roll)
            if cfg.USE_NEWS_SENTIMENT
            else None
        )
        feat_cols = intersect_feat_cols_with_statistical_prune(
            feat_cols,
            cfg_mod=cfg,
            trial_news_tag=_trial_news_tag,
        )
        _n_news_fc = sum(1 for c in feat_cols if str(c).startswith("news_"))
        if trial.number == 0 or trial.number % 10 == 0:
            print(
                f"[Optuna Trial {trial.number}] {len(feat_cols)} feat_cols "
                f"(news={_n_news_fc}), wf_folds={wf}, "
                f"train_rows={len(df_trial):,} …",
                flush=True,
            )
        # Defensive Sicherung: ``build_feature_cols`` kann Spalten emittieren, deren Erzeugung
        # an dynamischen Bedingungen hängt (z. B. News-Anchor-Sign-Bestätigung, neue Indikatoren
        # ohne aktiviertes Flag). Wenn so etwas durchschlägt, lieber den einzelnen Trial sauber
        # prunen als den ganzen Lauf abzuschießen — Bug wird im Log sichtbar protokolliert.
        _missing_feats = [c for c in feat_cols if c not in df_trial.columns]
        if _missing_feats:
            print(
                f"[Trial {trial.number}] WARN feat_cols enthält {len(_missing_feats)} "
                f"in df fehlende Spalte(n): {_missing_feats[:6]}"
                + (" …" if len(_missing_feats) > 6 else "")
                + " — Trial wird übersprungen (Pruned).",
                flush=True,
            )
            trial.set_user_attr("missing_feat_cols", ";".join(_missing_feats[:30])[:1000])
            raise optuna.exceptions.TrialPruned()
        focal_obj = make_focal_objective(focal_gamma, focal_alpha)

        fold_scores: list[float] = []
        wf_fold_n_tp: list[int] = []
        wf_fold_n_sig: list[int] = []
        wf_fold_n_pos: list[int] = []
        trial_nested_thresholds: list[float] = []
        insufficient_fold_tags: list[str] = []
        date_norm = pd.to_datetime(df_trial['Date'], errors='coerce').dt.normalize().values

        def _register_insufficient_wf_fold(reason: str) -> None:
            fold_scores.append(float(_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA))
            insufficient_fold_tags.append(f"{fold_i}:{reason}")
            trial.set_user_attr("n_base_insufficient_wf_folds", len(insufficient_fold_tags))
            trial.set_user_attr(
                "base_insufficient_wf_folds",
                ";".join(insufficient_fold_tags[-20:])[:1000],
            )
            m = float(np.mean(fold_scores)) if fold_scores else float(_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA)
            trial.report(m, int(fold_i))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        for fold_i in range(wf):
            train_end = min_train + fold_i * fold_size
            val_end   = min_train + (fold_i + 1) * fold_size
            if val_end > n_dates:
                break

            train_mask = df_trial['_date_idx'] < train_end
            val_mask   = (df_trial['_date_idx'] >= train_end) & \
                         (df_trial['_date_idx'] < val_end)

            X_val = df_trial.loc[val_mask,   feat_cols].to_numpy(dtype=np.float32, copy=True)
            y_val = df_trial.loc[val_mask,   'target'].values.astype(np.int8)
            _nan_sentinel = np.float32(getattr(cfg, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
            np.nan_to_num(X_val, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)

            if X_val.shape[0] < 10:
                _register_insufficient_wf_fold("val_rows_lt10")
                continue
            if y_val.sum() < 2:
                _register_insufficient_wf_fold("y_val_pos_lt2")
                continue

            tr_date_vals = date_norm[train_mask]
            tr_date_vals = tr_date_vals[pd.notna(tr_date_vals)]
            tr_unique_dates = np.sort(np.unique(tr_date_vals))
            if len(tr_unique_dates) < 20:
                _register_insufficient_wf_fold("train_unique_dates_lt20")
                continue
            split_pos = max(1, int(len(tr_unique_dates) * 0.80))
            split_pos = min(split_pos, len(tr_unique_dates) - 1)
            cal_start_date = tr_unique_dates[split_pos]
            inner_train_mask = train_mask & (date_norm < cal_start_date)
            inner_cal_mask = train_mask & (date_norm >= cal_start_date)
            if int(inner_train_mask.sum()) < 50 or int(inner_cal_mask.sum()) < 20:
                _register_insufficient_wf_fold("inner_train_or_cal_too_shallow")
                continue

            X_inner_train = df_trial.loc[inner_train_mask, feat_cols].to_numpy(dtype=np.float32, copy=True)
            y_inner_train = df_trial.loc[inner_train_mask, 'target'].values.astype(np.int8)
            X_inner_cal = df_trial.loc[inner_cal_mask, feat_cols].to_numpy(dtype=np.float32, copy=True)
            y_inner_cal = df_trial.loc[inner_cal_mask, 'target'].values.astype(np.int8)
            if y_inner_cal.sum() < 2:
                _register_insufficient_wf_fold("y_cal_pos_lt2")
                continue
            np.nan_to_num(X_inner_train, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)
            np.nan_to_num(X_inner_cal, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)

            # Inner 90/10 split for early stopping (on inner-train only)
            rng_es  = np.random.RandomState(cfg.RANDOM_STATE + fold_i)
            perm    = rng_es.permutation(len(X_inner_train))
            n_fit   = int(len(perm) * 0.9)
            n_fit = max(1, min(n_fit, len(perm) - 1))
            fit_idx, es_idx = perm[:n_fit], perm[n_fit:]

            dtrain = xgb.DMatrix(X_inner_train[fit_idx], label=y_inner_train[fit_idx])
            des    = xgb.DMatrix(X_inner_train[es_idx],  label=y_inner_train[es_idx])
            dcal   = xgb.DMatrix(X_inner_cal, label=y_inner_cal)
            dval   = xgb.DMatrix(X_val, label=y_val)

            # n_estimators = sklearn-Name; xgb.train nutzt num_boost_round (XGBoost 2.x warnt sonst)
            n_trees = int(params['n_estimators'])
            xgb_params = {k: v for k, v in params.items() if k != 'n_estimators'}
            xgb_params.update({'tree_method': 'hist', 'seed': cfg.RANDOM_STATE,
                 'disable_default_eval_metric': 1})
            if bool(getattr(cfg, "BASE_USE_SCALE_POS_WEIGHT", True)):
                xgb_params["scale_pos_weight"] = _auto_scale_pos_weight(y_inner_train[fit_idx])
            bst = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=n_trees,
                obj=focal_obj,
                evals=[(des, 'es')],
                custom_metric=lambda p, d: ('logloss',
                    float(np.mean(
                        -d.get_label() * np.log(np.clip(1/(1+np.exp(-p)), 1e-7, 1-1e-7))
                        -(1-d.get_label()) * np.log(np.clip(1/(1+np.exp(p)), 1e-7, 1-1e-7))
                    ))),
                early_stopping_rounds=cfg.EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )

            raw_preds_cal = bst.predict(dcal)
            probs_cal = 1.0 / (1.0 + np.exp(-raw_preds_cal))
            raw_preds_val = bst.predict(dval)
            probs_val = 1.0 / (1.0 + np.exp(-raw_preds_val))

            dates_cal = df_trial.loc[inner_cal_mask, 'Date'].values
            tickers_cal = df_trial.loc[inner_cal_mask, 'ticker'].values
            close_cal = df_trial.loc[inner_cal_mask, 'close'].values
            vol_stress_cal = (
                df_trial.loc[inner_cal_mask, 'vol_stress'].values
                if 'vol_stress' in df_trial.columns
                else None
            )
            blue_col = f"blue_sky_breakout_{int(breakout_lookback_window)}d"
            blue_sky_cal = (
                df_trial.loc[inner_cal_mask, blue_col].values
                if blue_col in df_trial.columns
                else None
            )
            volume_z_cal = (
                df_trial.loc[inner_cal_mask, 'volume_zscore'].values
                if 'volume_zscore' in df_trial.columns
                else None
            )
            vvix_ratio_cal = (
                df_trial.loc[inner_cal_mask, 'mr_vvix_div_vix'].values
                if 'mr_vvix_div_vix' in df_trial.columns
                else None
            )
            dates_val = df_trial.loc[val_mask, 'Date'].values
            tickers_val = df_trial.loc[val_mask, 'ticker'].values
            close_val = df_trial.loc[val_mask, 'close'].values
            vol_stress_val = (
                df_trial.loc[val_mask, 'vol_stress'].values
                if 'vol_stress' in df_trial.columns
                else None
            )
            blue_sky_val = (
                df_trial.loc[val_mask, blue_col].values
                if blue_col in df_trial.columns
                else None
            )
            volume_z_val = (
                df_trial.loc[val_mask, 'volume_zscore'].values
                if 'volume_zscore' in df_trial.columns
                else None
            )
            vvix_ratio_val = (
                df_trial.loc[val_mask, 'mr_vvix_div_vix'].values
                if 'mr_vvix_div_vix' in df_trial.columns
                else None
            )
            rsi_dyn_col = f"rsi_{int(rsi_w)}d"
            rsi_dyn_cal = (
                df_trial.loc[inner_cal_mask, rsi_dyn_col].values
                if rsi_dyn_col in df_trial.columns
                else None
            )
            rsi_dyn_val = (
                df_trial.loc[val_mask, rsi_dyn_col].values
                if rsi_dyn_col in df_trial.columns
                else None
            )
            bb_dyn_col = f"bb_pband_{int(bb_w)}"
            bb_dyn_cal = (
                df_trial.loc[inner_cal_mask, bb_dyn_col].values
                if bb_dyn_col in df_trial.columns
                else None
            )
            bb_dyn_val = (
                df_trial.loc[val_mask, bb_dyn_col].values
                if bb_dyn_col in df_trial.columns
                else None
            )

            nested_thr, _nested_thr_score = _pick_threshold_nested_base(
                base_eval_threshold,
                probs_cal=probs_cal,
                dates_cal=dates_cal,
                tickers_cal=tickers_cal,
                y_cal=y_inner_cal,
                consecutive_days=consecutive_days,
                signal_cooldown_days=signal_cooldown_days,
                close_cal=close_cal,
                rsi_window=rsi_w,
                signal_skip_near_peak=signal_skip_near_peak,
                peak_lookback_days=peak_lookback_days,
                peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                signal_max_rsi=signal_max_rsi,
                vol_stress_cal=vol_stress_cal,
                signal_max_vol_stress_z=signal_max_vol_stress_z,
                blue_sky_cal=blue_sky_cal,
                volume_z_cal=volume_z_cal,
                signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                vvix_ratio_cal=vvix_ratio_cal,
                rsi_cal=rsi_dyn_cal,
                bb_cal=bb_dyn_cal,
                dyn_vvix_trigger=dyn_vvix_trigger,
                dyn_rsi_trigger=dyn_rsi_trigger,
                dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                dyn_mult_1=dyn_mult_1,
                dyn_mult_2=dyn_mult_2,
                dyn_mult_3=dyn_mult_3,
            )
            trial_nested_thresholds.append(float(nested_thr))

            fold_score, n_tp, n_sig, max_cfp = _score_tp_precision_fold(
                probs_val, dates_val, tickers_val, y_val,
                nested_thr, consecutive_days, signal_cooldown_days,
                close_arr=close_val,
                rsi_window=rsi_w,
                signal_skip_near_peak=signal_skip_near_peak,
                peak_lookback_days=peak_lookback_days,
                peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                signal_max_rsi=signal_max_rsi,
                vol_stress_arr=vol_stress_val,
                signal_max_vol_stress_z=signal_max_vol_stress_z,
                blue_sky_breakout_arr=blue_sky_val,
                volume_zscore_arr=volume_z_val,
                signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                vvix_ratio_arr=vvix_ratio_val,
                rsi_arr=rsi_dyn_val,
                bb_pband_arr=bb_dyn_val,
                dyn_vvix_trigger=dyn_vvix_trigger,
                dyn_rsi_trigger=dyn_rsi_trigger,
                dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                dyn_mult_1=dyn_mult_1,
                dyn_mult_2=dyn_mult_2,
                dyn_mult_3=dyn_mult_3,
            )
            fold_scores.append(float(fold_score))
            wf_fold_n_tp.append(int(n_tp))
            wf_fold_n_sig.append(int(n_sig))
            wf_fold_n_pos.append(int(y_val.sum()))

            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if fold_scores:
            trial.set_user_attr("n_base_wf_folds_scored", int(len(fold_scores)))
        if wf_fold_n_tp:
            _ntp = int(sum(wf_fold_n_tp))
            _nsg = int(sum(wf_fold_n_sig))
            _nps = int(sum(wf_fold_n_pos))
            trial.set_user_attr("base_wf_n_tp_sum", _ntp)
            trial.set_user_attr("tp_n", int(_ntp))
            trial.set_user_attr("base_wf_n_signals_sum", _nsg)
            trial.set_user_attr("base_wf_n_pos_sum", _nps)
            trial.set_user_attr("base_wf_tp_slash_sig", f"{_ntp}/{_nsg}")
            trial.set_user_attr("base_wf_tp_slash_pos", f"{_ntp}/{_nps}")
            if _nsg > 0:
                trial.set_user_attr("base_wf_precision_pooled", float(_ntp) / float(_nsg))
            if _nps > 0:
                trial.set_user_attr("base_wf_recall_pooled", float(_ntp) / float(_nps))
        if trial_nested_thresholds:
            trial.set_user_attr("nested_thr_mean", float(np.mean(trial_nested_thresholds)))
            trial.set_user_attr("nested_thr_min", float(np.min(trial_nested_thresholds)))
            trial.set_user_attr("nested_thr_max", float(np.max(trial_nested_thresholds)))
        if _subset and pool:
            trial.set_user_attr("feature_subset_columns", list(feat_cols))
        _score = float(np.mean(fold_scores)) if fold_scores else -1.0
        print(
            f"[Optuna Trial {trial.number}] fertig in {time.perf_counter() - _t_trial:.1f}s — "
            f"score={_score:.4f}, feat_cols={len(feat_cols)}, folds={len(fold_scores)}",
            flush=True,
        )
        return _score

    _chk_path = _base_optuna_checkpoint_path(cfg)
    _chk_exists = _chk_path.is_file()
    _chk_load = bool(getattr(cfg, "BASE_OPTUNA_CHECKPOINT_LOAD", True))
    _bp_before_load = getattr(cfg, "base_optuna_best_params", None)
    _had_params_before_load = isinstance(_bp_before_load, dict) and bool(_bp_before_load)
    _loaded_ckpt = try_load_base_optuna_checkpoint(cfg)
    _bp_after = getattr(cfg, "base_optuna_best_params", None)
    _has_opt_params = isinstance(_bp_after, dict) and bool(_bp_after)
    print(
        "[Optuna] Phase 1 — vor Optimierung: Base-Checkpoint "
        f"Pfad={_chk_path.resolve()!s}, Datei vorhanden={_chk_exists}, "
        f"BASE_OPTUNA_CHECKPOINT_LOAD={_chk_load}.",
        flush=True,
    )
    if _has_opt_params:
        if _had_params_before_load:
            _seed_quelle = "bereits im Namespace (z.B. scoring_artifacts oder gleiche Session)"
        elif _loaded_ckpt:
            _seed_quelle = "aus Base-Checkpoint-Datei geladen"
        else:
            _seed_quelle = "im Namespace (Quelle unklar)"
        print(
            "[Optuna] Phase 1 — alte optimierte Params: vorhanden; "
            f"Quelle: {_seed_quelle}. Erster Trial (enqueue_trial): diese Params als Seed.",
            flush=True,
        )
    else:
        _hint = ""
        if _chk_exists and _chk_load and not _loaded_ckpt and not _had_params_before_load:
            _hint = " Checkpoint-Datei existiert, enthielt aber keine nutzbaren Params (oder Lesen fehlgeschlagen — siehe Meldungen oben)."
        elif _chk_exists and not _chk_load:
            _hint = " Checkpoint-Datei existiert, Laden ist deaktiviert (BASE_OPTUNA_CHECKPOINT_LOAD=False)."
        print(
            "[Optuna] Phase 1 — alte optimierte Params: keine. "
            f"Erster Trial (enqueue_trial): nutzt cfg.SEED_PARAMS als Seed.{_hint}",
            flush=True,
        )

    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3)
    study   = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    _base_seed_src = getattr(cfg, "base_optuna_best_params", None)
    if isinstance(_base_seed_src, dict) and _base_seed_src:
        _seed_enq = dict(_base_seed_src)
        print(
            f"Optuna Phase 1: Seed-Trial aus gespeichertem base_optuna_best_params "
            f"({len(_seed_enq)} Parameter).",
            flush=True,
        )
    else:
        _seed_enq = dict(seed_params)
    _seed_enq.setdefault("base_eval_threshold", float(seed_params.get("threshold", 0.5)))
    if not cfg.opt_optimize_y_targets():
        for _k in ('return_window', 'rally_threshold', 'lead_days', 'entry_days', 'min_rally_tail_days'):
            _seed_enq.pop(_k, None)
    if _subset and pool:
        _seed_enq["feature_subset_id"] = 0
    else:
        _seed_enq.pop("feature_subset_id", None)
    _cat_choices = {
        "rsi_window": cfg.effective_window_grid('RSI_WINDOWS'),
        "bb_window": cfg.effective_window_grid('BB_WINDOWS'),
        "sma_window": cfg.effective_window_grid('SMA_WINDOWS'),
        "btc_momentum_z_window": cfg.effective_window_grid('BTC_MOMENTUM_Z_WINDOWS'),
        "market_breadth_z_window": cfg.effective_window_grid('MARKET_BREADTH_Z_WINDOWS'),
        "rel_momentum_window": cfg.effective_window_grid('REL_MOMENTUM_WINDOWS'),
        "adr_window": cfg.effective_window_grid('ADR_WINDOWS'),
        "breakout_lookback_window": cfg.effective_window_grid('BREAKOUT_LOOKBACK_WINDOWS'),
        "vcp_window": cfg.effective_window_grid('VCP_WINDOWS'),
        "btc_corr_window": cfg.effective_window_grid('BTC_CORR_WINDOWS'),
        "yz_vol_window": cfg.effective_window_grid('YANG_ZHANG_WINDOWS'),
        "downside_vol_window": cfg.effective_window_grid('DOWNSIDE_VOL_WINDOWS'),
        "ret_moment_window": cfg.effective_window_grid('RET_MOMENT_WINDOWS'),
        "amihud_window": cfg.effective_window_grid('AMIHUD_WINDOWS'),
        "vcp_lower_low_window": cfg.effective_window_grid('VCP_LOWER_LOW_WINDOWS'),
        "breakout_volume_trigger_z": cfg.effective_window_grid('BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS'),
        "news_mom_w": list(cfg.NEWS_MOM_WINDOWS),
        "news_vol_ma": list(cfg.NEWS_VOL_MA_WINDOWS),
        "news_tone_roll": list(cfg.NEWS_TONE_ROLL_WINDOWS),
        "news_extra_zscore_w": list(cfg.NEWS_EXTRA_ZSCORE_WINDOWS),
        "news_extra_tone_accel": list(cfg.NEWS_EXTRA_TONE_ACCEL_OPTIONS),
        "news_extra_macro_sec_diff": list(cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS),
        "news_add_sign_confirmation": list(cfg.NEWS_ADD_SIGN_CONFIRMATION_OPTIONS),
    }
    for _k, _choices in _cat_choices.items():
        if _k not in _seed_enq or not _choices:
            continue
        if _seed_enq[_k] not in _choices:
            _old = _seed_enq[_k]
            _seed_enq[_k] = _choices[0]
            print(
                f"Seed-Parameter angepasst: {_k}={_old!r} nicht im aktuellen Grid {_choices} "
                f"-> nutze {_seed_enq[_k]!r}.",
                flush=True,
            )
    _hist_bv = getattr(cfg, "base_optuna_best_value", None)
    if _hist_bv is not None and _has_opt_params:
        print(
            "[Optuna] Phase 1 — Referenz: gespeicherter base_optuna_best_value="
            f"{float(_hist_bv):.4f} (Checkpoint/Artefakt). Optuna startet eine neue Study; "
            "der Seed-Trial wird neu bewertet — die Zielfunktion in tqdm kann davon abweichen "
            "(Walk-Forward, Pruner, andere Features/News, Grid-Snap oben; Skala: mean Recall/Fold "
            "100·TP/Pos nach Precision-Gate, nicht mehr n_TP oder 100·Precision).",
            flush=True,
        )
    study.enqueue_trial(_seed_enq)
    # Nur tqdm-Fortschritt (eine Zeile); Optuna-INFO würde jeden Trial doppelt loggen
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _spb = getattr(cfg, "OPTUNA_SHOW_PROGRESS_BAR", None)
    if _spb is None:
        _spb = True
    study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(_spb))

    best = dict(study.best_params)
    if _subset and pool and study.best_trial is not None:
        _fsc_bt = study.best_trial.user_attrs.get("feature_subset_columns")
        if _fsc_bt:
            best["feature_subset_columns"] = list(_fsc_bt)
    # Ensure all model hyperparameters are present — if cfg.OPT_MODEL_HYPERPARAMS=False
    # they were never suggested by Optuna, so fill them from seed_params.
    for k, v in seed_params.items():
        if k not in best:
            best[k] = v

    if not cfg.opt_optimize_y_targets():
        _, _wh, _rt, _, _ld, _ed, _ = cfg.fixed_y_rule_params()
        best['return_window'] = _wh
        best['rally_threshold'] = _rt
        best['lead_days'] = _ld
        best['entry_days'] = _ed
        best['min_rally_tail_days'] = 5

    mode = 'Option A (model HPs optimised)' if cfg.OPT_MODEL_HYPERPARAMS \
           else 'Option B (model HPs fixed from cfg.SEED_PARAMS)'
    print(f'\nBest trial score={study.best_value:.4f}  '
          f'(= mean Recall-Score/Fold: 100·TP/Pos bei Filter-Prec>={_OPT_MIN_PRECISION:.0%}, '
          f'max consec FP <= {_OPT_MAX_CONSEC_FP})  [{mode}]')
    if study.best_trial is not None and study.best_trial.user_attrs.get("tp_n") is not None:
        print(
            f"  Best trial tp_n (Summe TP über WF-Val-Folds) = {study.best_trial.user_attrs['tp_n']}",
            flush=True,
        )
    print("Optuna Phase 1 — finale Bestwerte (alle Parameter):", flush=True)
    for _k in sorted(best.keys()):
        print(f"  {_k} = {best[_k]!r}", flush=True)
    cfg.base_optuna_best_params = dict(best)
    cfg.base_optuna_best_value = float(study.best_value)
    save_base_optuna_checkpoint(cfg)
    return best
