"""
Feature pre-screening (vor Optuna Phase 1).

Conservative, family-aware Auswahl-Strategie:

* **Stage 0 — Konstanz/Sentinel-Filter** (deterministisch).
* **Stage 1 — Importance-Ensemble** über ``cfg.N_WF_SPLITS`` Walk-Forward Folds
  (gleicher Expanding-Window-Mechanismus wie ``optuna_train.optimize_xgb``).
  Pro Fold: focal-loss XGBoost mit ``cfg.SEED_PARAMS`` → TreeSHAP auf dem
  Validations-Fold; ``mean(|SHAP|)`` pro Feature.
* **Stage 2 — Boruta-Schatten**: pro Fold zusätzlich ``cfg.FEATURE_PRESCREEN_N_SHADOWS``
  zufällig permutierte Kopien existierender Features. Quantil
  (``cfg.FEATURE_PRESCREEN_NOISE_QUANTILE``) ihrer aggregierten Importance
  liefert den daten-getriebenen Rausch-Floor.
* **Stage 3 — Familien-bewusste Auswahl**:

  - Kategorische/ID-Spalten und News (``news_*``) immer behalten.
  - ``regime_*``/``mr_*``: Top-K (``MACRO_TOPK``) immer behalten.
  - Pro Familie (``rsi``, ``bb``, ``yz_vol``, …) Top-K (``FAMILY_TOPK``) immer behalten.
  - Sonst: behalten falls ``imp >= floor`` **oder** globaler Rang ≤ ``GLOBAL_TOPK``.

* **Stage 4 — Sanity-Probe**: vergleicht AUC und PR-AUC eines kleinen XGB auf
  ``feat_all`` vs. ``feat_kept`` im letzten (jüngsten) Fold; bricht ab und liefert
  das volle Set zurück, wenn die Toleranzen verletzt werden.
* **Stage 5 — Anwendung**: ``allowed_windows`` pro ``cfg``-Liste wird abgeleitet
  und im Artefakt persistiert. ``optuna_train`` liest via ``cfg.effective_window_grid``.

News-Spalten werden bewusst **nicht** auf Fenster-Ebene gefiltert (User-Wunsch:
"News auslassen") — das News-Tag-Tripel-Optimum bleibt Sache von Optuna selbst.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from lib.stock_rally_v10.helpers import make_focal_objective


# Feature-Familien: (regex, family_key, cfg_list_name).
# ``family_key`` gruppiert für Top-K-pro-Familie; ``cfg_list_name`` mappt zur
# Optuna-Grid-Konstanten in ``config`` (für ``allowed_windows``).
# ``None`` als ``cfg_list_name`` = Familie ohne eigenen cfg-Grid (z. B. bb_x_rsi:
# Kreuz aus zwei anderen Familien — wird dann nur via deren Grids beeinflusst).
_FAMILY_PATTERNS: list[tuple[re.Pattern, str, str | None]] = [
    (re.compile(r"^rsi_(\d+)$"), "rsi", "RSI_WINDOWS"),
    (re.compile(r"^rsi_delta_3d_(\d+)$"), "rsi", "RSI_WINDOWS"),
    (re.compile(r"^rsi_weekly_(\d+)$"), "rsi", "RSI_WINDOWS"),
    (re.compile(r"^sector_avg_rsi_(\d+)$"), "rsi", "RSI_WINDOWS"),
    (re.compile(r"^bb_pband_(\d+)$"), "bb", "BB_WINDOWS"),
    (re.compile(r"^bb_squeeze_factor_(\d+)$"), "bb", "BB_WINDOWS"),
    (re.compile(r"^bb_delta_3d_(\d+)$"), "bb", "BB_WINDOWS"),
    (re.compile(r"^bb_slope_5d_(\d+)$"), "bb", "BB_WINDOWS"),
    (re.compile(r"^bb_x_rsi_(\d+)_(\d+)$"), "bb_x_rsi", None),
    (re.compile(r"^sma_cross_20_(\d+)$"), "sma", "SMA_WINDOWS"),
    (re.compile(r"^market_breadth_(\d+)$"), "sma", "SMA_WINDOWS"),
    (
        re.compile(r"^market_breadth_z_(\d+)_w(\d+)$"),
        "market_breadth_z",
        "MARKET_BREADTH_Z_WINDOWS",
    ),
    (re.compile(r"^btc_momentum_z_w(\d+)$"), "btc_momentum_z", "BTC_MOMENTUM_Z_WINDOWS"),
    (re.compile(r"^corr_stock_btc_(\d+)d$"), "btc_corr", "BTC_CORR_WINDOWS"),
    (re.compile(r"^rel_momentum_(\d+)d$"), "rel_momentum", "REL_MOMENTUM_WINDOWS"),
    (re.compile(r"^adr_pct_(\d+)d$"), "adr", "ADR_WINDOWS"),
    (re.compile(r"^vcp_tightness_(\d+)d$"), "vcp", "VCP_WINDOWS"),
    (re.compile(r"^vcp_tightness_hl_(\d+)d$"), "vcp", "VCP_WINDOWS"),
    (
        re.compile(r"^vcp_at_period_low_frac_(\d+)_(\d+)d$"),
        "vcp_lower_low",
        "VCP_LOWER_LOW_WINDOWS",
    ),
    (
        re.compile(r"^vcp_tightness_slope_(\d+)_(\d+)d$"),
        "vcp_lower_low",
        "VCP_LOWER_LOW_WINDOWS",
    ),
    (re.compile(r"^blue_sky_breakout_(\d+)d$"), "breakout_lookback", "BREAKOUT_LOOKBACK_WINDOWS"),
    (re.compile(r"^dist_to_prior_hi_pct_(\d+)d$"), "breakout_lookback", "BREAKOUT_LOOKBACK_WINDOWS"),
    (re.compile(r"^volume_at_resistance_(\d+)d$"), "breakout_lookback", "BREAKOUT_LOOKBACK_WINDOWS"),
    (
        re.compile(r"^breakout_volume_confirmed_(\d+)_z(\d+(?:p\d+)?)d$"),
        "breakout_volume_trigger",
        "BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS",
    ),
    (re.compile(r"^yz_vol_(\d+)d$"), "yz_vol", "YANG_ZHANG_WINDOWS"),
    (re.compile(r"^downside_vol_(\d+)d$"), "downside_vol", "DOWNSIDE_VOL_WINDOWS"),
    (re.compile(r"^downside_vol_ratio_(\d+)d$"), "downside_vol", "DOWNSIDE_VOL_WINDOWS"),
    (re.compile(r"^ret_skew_(\d+)d$"), "ret_moment", "RET_MOMENT_WINDOWS"),
    (re.compile(r"^ret_kurt_(\d+)d$"), "ret_moment", "RET_MOMENT_WINDOWS"),
    (re.compile(r"^amihud_illiquidity_(\d+)d$"), "amihud", "AMIHUD_WINDOWS"),
]

# Spalten, die wir IMMER behalten (Kategorische / IDs).
_ALWAYS_KEEP_NAMES: tuple[str, ...] = (
    "month",
    "sector_id",
    "gics_sector_id",
    "gics_industry_id",
)

# Singletons (kein Familien-Grid) — werden nicht über cfg-Listen geschrumpft,
# bleiben aber Kandidaten für globalen Top-K-/Floor-Schutz.
_SINGLETON_NAMES: tuple[str, ...] = (
    "macd_diff",
    "vol_stress",
    "drawdown",
    "adx",
    "vol_ratio",
    "adx_delta_3d",
    "momentum_accel",
    "close_vs_sma200",
    "sma200_delta_5d",
    "drawdown_252d",
    "volume_zscore",
    "btc_momentum",
    "dollar_volume_zscore",
    "volume_force_1d",
)


def _classify_family(col: str) -> tuple[str | None, str | None, float | None]:
    """``(family_key, cfg_list_name, window)`` oder ``(None, None, None)``."""
    for pat, fam, cfg_list in _FAMILY_PATTERNS:
        m = pat.match(col)
        if not m:
            continue
        if fam == "bb_x_rsi":
            return fam, cfg_list, None
        if fam == "breakout_volume_trigger":
            tok = m.group(2)
            return fam, cfg_list, float(tok.replace("p", "."))
        win = int(m.group(m.lastindex))
        return fam, cfg_list, float(win)
    return None, None, None


def _is_categorical_keep(col: str) -> bool:
    return col in _ALWAYS_KEEP_NAMES


def _is_news_keep(col: str) -> bool:
    return col.startswith("news_")


def _is_macro_regime(col: str) -> bool:
    return col.startswith("mr_") or col.startswith("regime_")


def _collect_candidate_features(df: pd.DataFrame, cfg_mod: Any) -> list[str]:
    """Kandidaten-Spalten direkt aus den Spalten in ``df`` (Schnitt mit bekannten Familien
    + Singletons + Kategorischen + Macro/Regime).

    News-Features werden ausgelassen (User-Wunsch). Wir ENUMERIEREN das Optuna-Grid
    bewusst nicht (``all_model_tech_col_names_for_assemble_dropna`` ist O(10^8) Iter.):
    es genügt, was ``add_technical_indicators`` real berechnet hat.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for c in df.columns:
        if c in seen:
            continue
        if _is_news_keep(c):
            continue
        if c in {"target", "Date", "ticker", "open", "high", "low", "close", "volume"}:
            continue
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            continue
        # Nur bekannte Spalten — Familien-Match, Singletons, Kategorische, Macro/Regime.
        fam, _cfg_list, _win = _classify_family(c)
        if (
            fam is not None
            or c in _SINGLETON_NAMES
            or c in _ALWAYS_KEEP_NAMES
            or _is_macro_regime(c)
        ):
            candidates.append(c)
            seen.add(c)
    return candidates


def _stage0_constancy_filter(
    df: pd.DataFrame, feat_cols: list[str], *, sentinel: float
) -> tuple[list[str], dict[str, str]]:
    """Entfernt fast-konstante / nur-Sentinel-/nur-NaN-Spalten."""
    sentinel_eq = float(sentinel)
    kept: list[str] = []
    drop_reason: dict[str, str] = {}
    for c in feat_cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        ok = np.isfinite(s)
        n_ok = int(ok.sum())
        if n_ok == 0:
            drop_reason[c] = "all_nan"
            continue
        v = s[ok]
        # Sentinel maskieren, dann auf reale Werte prüfen
        non_sent = v[v != sentinel_eq]
        if non_sent.size == 0:
            drop_reason[c] = "all_sentinel"
            continue
        spread = float(non_sent.max() - non_sent.min())
        std = float(np.std(non_sent, ddof=0))
        if spread <= 1e-12 and std <= 1e-12:
            drop_reason[c] = "constant"
            continue
        kept.append(c)
    return kept, drop_reason


def _wf_fold_indices(
    n_dates: int, n_folds: int, *, min_train_frac: float = 0.40
) -> list[tuple[int, int]]:
    """Expanding-window Folds: gleiche Mechanik wie ``optuna_train.optimize_xgb``."""
    min_train = int(n_dates * float(min_train_frac))
    fold_size = max(1, (n_dates - min_train) // max(1, int(n_folds)))
    folds: list[tuple[int, int]] = []
    for i in range(int(n_folds)):
        train_end = min_train + i * fold_size
        val_end = min_train + (i + 1) * fold_size
        if val_end > n_dates:
            break
        folds.append((train_end, val_end))
    return folds


def _make_shadow_block(
    X: np.ndarray,
    feat_cols: list[str],
    *,
    n_shadows: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[str]]:
    """Liefert (X_shadow, shadow_names) — ``n_shadows`` permutierte Kopien
    zufällig gewählter Real-Spalten (mit Wiederholung wenn n_feat < n_shadows)."""
    n_feat = X.shape[1]
    if n_feat == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32), []
    src_idx = rng.randint(0, n_feat, size=int(n_shadows))
    shadow = np.empty((X.shape[0], int(n_shadows)), dtype=np.float32)
    names: list[str] = []
    for j, s in enumerate(src_idx):
        col = X[:, int(s)].copy()
        rng.shuffle(col)
        shadow[:, j] = col
        names.append(f"__shadow_{j:03d}__{feat_cols[int(s)]}")
    return shadow, names


def _train_fold_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed_params: dict,
    cfg_mod: Any,
    seed: int,
) -> xgb.Booster:
    """Trainiert ein einzelnes XGB pro Fold mit cfg.SEED_PARAMS + focal loss."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(max(1.0, n_neg / n_pos)) if n_pos > 0 else 1.0
    params = {
        k: seed_params[k]
        for k in (
            "grow_policy",
            "max_leaves",
            "max_bin",
            "max_depth",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        )
        if k in seed_params
    }
    if params.get("grow_policy") == "depthwise":
        params["max_leaves"] = 0
    n_trees = int(seed_params.get("n_estimators", 300))
    params.update(
        tree_method="hist",
        seed=int(seed),
        scale_pos_weight=spw,
        disable_default_eval_metric=1,
    )
    focal_obj = make_focal_objective(
        float(seed_params.get("focal_gamma", 1.5)),
        float(seed_params.get("focal_alpha", 0.25)),
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_trees,
        obj=focal_obj,
        evals=[(dval, "val")],
        custom_metric=lambda p, d: (
            "logloss",
            float(
                np.mean(
                    -d.get_label() * np.log(np.clip(1 / (1 + np.exp(-p)), 1e-7, 1 - 1e-7))
                    - (1 - d.get_label()) * np.log(np.clip(1 / (1 + np.exp(p)), 1e-7, 1 - 1e-7))
                )
            ),
        ),
        early_stopping_rounds=int(getattr(cfg_mod, "EARLY_STOPPING_ROUNDS", 30)),
        verbose_eval=False,
    )
    return bst


def _shap_mean_abs(bst: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """``mean(|SHAP|)`` pro Feature via ``predict(pred_contribs=True)``.

    Liefert Array der Länge ``X.shape[1]`` (bias-Spalte verworfen).
    """
    contribs = bst.predict(xgb.DMatrix(X), pred_contribs=True)
    if contribs.ndim != 2 or contribs.shape[1] < X.shape[1]:
        return np.zeros(X.shape[1], dtype=np.float64)
    feat_contribs = contribs[:, : X.shape[1]]
    return np.mean(np.abs(feat_contribs), axis=0).astype(np.float64)


def _walkforward_importance(
    df: pd.DataFrame,
    feat_cols: list[str],
    *,
    cfg_mod: Any,
    n_folds: int,
    n_shadows: int,
    sample_per_fold: int,
    rng_seed: int,
    verbose: bool = True,
) -> tuple[dict[str, float], np.ndarray, dict]:
    """Median(|SHAP|) pro Real-Feature über alle Folds + alle Shadow-Importances.

    Rückgabe: (real_importance_dict, shadow_importance_array, diag).
    """
    if "target" not in df.columns:
        raise KeyError("feature_prescreen: 'target' fehlt in df_train.")
    all_dates = np.sort(pd.to_datetime(df["Date"]).unique())
    n_dates = int(len(all_dates))
    folds = _wf_fold_indices(n_dates, n_folds)
    if not folds:
        raise ValueError(
            f"feature_prescreen: keine WF-Folds bildbar (n_dates={n_dates}, n_folds={n_folds})."
        )
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    date_idx = pd.to_datetime(df["Date"]).map(date_to_idx).to_numpy(dtype=np.int64)
    sentinel = float(getattr(cfg_mod, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    seed_params = dict(getattr(cfg_mod, "SEED_PARAMS", {}))

    real_imps: list[np.ndarray] = []
    shadow_imps: list[np.ndarray] = []
    fold_diag: list[dict] = []
    rng = np.random.RandomState(int(rng_seed))

    X_full = df[feat_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(X_full, nan=sentinel, posinf=sentinel, neginf=sentinel, copy=False)
    y_full = df["target"].to_numpy(dtype=np.int8)

    if verbose:
        print(f"  -> Folds: {len(folds)}  (Spalten: {len(feat_cols)} real + {n_shadows} shadow/fold)", flush=True)

    for fi, (train_end, val_end) in enumerate(folds):
        t0 = time.perf_counter()
        tr_mask = date_idx < train_end
        va_mask = (date_idx >= train_end) & (date_idx < val_end)
        n_tr, n_va = int(tr_mask.sum()), int(va_mask.sum())
        if n_tr < 200 or n_va < 50 or int(y_full[tr_mask].sum()) < 5 or int(y_full[va_mask].sum()) < 2:
            fold_diag.append({"fold": fi, "skipped": True, "n_tr": n_tr, "n_va": n_va})
            if verbose:
                print(
                    f"     Fold {fi+1}/{len(folds)}: SKIP (n_tr={n_tr}, n_va={n_va}, "
                    f"pos_tr={int(y_full[tr_mask].sum())}, pos_va={int(y_full[va_mask].sum())})",
                    flush=True,
                )
            continue

        X_tr_real = X_full[tr_mask]
        y_tr = y_full[tr_mask]
        X_va_real = X_full[va_mask]
        y_va = y_full[va_mask]

        # Shadow-Block pro Fold: Zeilen-permutierte Kopien zufällig gewählter Spalten.
        X_tr_shadow, shadow_names = _make_shadow_block(
            X_tr_real, feat_cols, n_shadows=n_shadows, rng=rng
        )
        X_va_shadow, _ = _make_shadow_block(
            X_va_real, feat_cols, n_shadows=n_shadows, rng=rng
        )

        X_tr = np.concatenate([X_tr_real, X_tr_shadow], axis=1) if n_shadows > 0 else X_tr_real
        X_va = np.concatenate([X_va_real, X_va_shadow], axis=1) if n_shadows > 0 else X_va_real

        bst = _train_fold_xgb(
            X_tr,
            y_tr,
            X_va,
            y_va,
            seed_params=seed_params,
            cfg_mod=cfg_mod,
            seed=int(getattr(cfg_mod, "RANDOM_STATE", 42)) + fi,
        )

        # SHAP auf Validations-Sample (subsampled für Speed).
        if X_va.shape[0] > sample_per_fold:
            sub_idx = rng.choice(X_va.shape[0], size=int(sample_per_fold), replace=False)
            X_shap = X_va[sub_idx]
        else:
            X_shap = X_va
        mean_abs = _shap_mean_abs(bst, X_shap)
        n_real = len(feat_cols)
        real_part = mean_abs[:n_real]
        shadow_part = mean_abs[n_real : n_real + len(shadow_names)] if shadow_names else np.array([])

        real_imps.append(real_part)
        if shadow_part.size > 0:
            shadow_imps.append(shadow_part)

        elapsed_fold = time.perf_counter() - t0
        best_iter = int(getattr(bst, "best_iteration", 0))
        fold_diag.append(
            {
                "fold": fi,
                "skipped": False,
                "n_tr": n_tr,
                "n_va": n_va,
                "n_shap_sample": int(X_shap.shape[0]),
                "elapsed_s": round(elapsed_fold, 2),
                "best_iteration": best_iter,
            }
        )
        if verbose:
            real_max = float(real_part.max()) if real_part.size else 0.0
            shadow_max = float(shadow_part.max()) if shadow_part.size else 0.0
            print(
                f"     Fold {fi+1}/{len(folds)}: train={n_tr:>7,}  val={n_va:>6,}  "
                f"pos_tr={int(y_tr.sum()):>5,}  pos_va={int(y_va.sum()):>4,}  "
                f"best_iter={best_iter:>4}  shap_n={int(X_shap.shape[0]):>5,}  "
                f"max(|SHAP|) real={real_max:.3e} shadow={shadow_max:.3e}  "
                f"({elapsed_fold:5.1f}s)",
                flush=True,
            )

    if not real_imps:
        raise ValueError("feature_prescreen: alle Folds übersprungen — zu dünne Daten?")

    real_stack = np.stack(real_imps, axis=0)
    real_median = np.median(real_stack, axis=0)
    real_dict = {feat_cols[i]: float(real_median[i]) for i in range(len(feat_cols))}
    shadow_arr = np.concatenate(shadow_imps) if shadow_imps else np.array([])
    diag = {"folds": fold_diag, "n_folds_used": int(real_stack.shape[0])}
    return real_dict, shadow_arr.astype(np.float64), diag


def _select_features(
    importance: dict[str, float],
    *,
    floor: float,
    family_topk: int,
    global_topk: int,
    macro_topk: int,
) -> tuple[set[str], set[str], dict[str, str]]:
    """Family-aware Auswahl. Rückgabe: (kept, dropped, drop_reasons)."""
    cols = list(importance.keys())
    sorted_global = sorted(cols, key=lambda c: importance.get(c, 0.0), reverse=True)
    global_rank = {c: i + 1 for i, c in enumerate(sorted_global)}

    families: dict[str, list[str]] = {}
    for c in cols:
        if _is_categorical_keep(c) or _is_news_keep(c):
            continue
        if _is_macro_regime(c):
            families.setdefault("__macro_regime__", []).append(c)
            continue
        fam, _cfg_list, _win = _classify_family(c)
        if fam is not None:
            families.setdefault(fam, []).append(c)
    family_rank: dict[str, int] = {}
    family_of: dict[str, str] = {}
    for fam, members in families.items():
        ordered = sorted(members, key=lambda c: importance.get(c, 0.0), reverse=True)
        for r, c in enumerate(ordered, start=1):
            family_rank[c] = r
            family_of[c] = fam

    macro_protect: set[str] = set()
    if "__macro_regime__" in families:
        ordered = sorted(
            families["__macro_regime__"], key=lambda c: importance.get(c, 0.0), reverse=True
        )
        macro_protect.update(ordered[: max(0, int(macro_topk))])

    kept: set[str] = set()
    dropped: set[str] = set()
    reasons: dict[str, str] = {}
    for c in cols:
        imp = float(importance.get(c, 0.0))
        if _is_categorical_keep(c):
            kept.add(c)
            reasons[c] = "always_keep_categorical"
            continue
        if _is_news_keep(c):
            kept.add(c)
            reasons[c] = "always_keep_news"
            continue
        if c in macro_protect:
            kept.add(c)
            reasons[c] = "macro_regime_topk"
            continue
        fam = family_of.get(c)
        f_rank = family_rank.get(c, 9999)
        g_rank = global_rank.get(c, 9999)
        if fam is not None and f_rank <= int(family_topk):
            kept.add(c)
            reasons[c] = f"family_topk_{fam}_rank{f_rank}"
            continue
        if g_rank <= int(global_topk):
            kept.add(c)
            reasons[c] = f"global_topk_rank{g_rank}"
            continue
        if imp >= float(floor):
            kept.add(c)
            reasons[c] = "above_noise_floor"
            continue
        dropped.add(c)
        reasons[c] = (
            f"below_floor (imp={imp:.3e} < floor={floor:.3e}, "
            f"fam_rank={f_rank}, global_rank={g_rank})"
        )
    return kept, dropped, reasons


def _compute_allowed_windows(
    kept: set[str], cfg_mod: Any
) -> tuple[dict[str, list], dict[str, dict[str, list]]]:
    """Pro cfg-Liste: Subset der Original-Reihenfolge, der durch ``kept`` belegt ist.

    Rückgabe: (allowed_windows, family_summary).
    family_summary: ``{cfg_list_name: {"kept_windows": [...], "all_windows": [...]}}``.
    """
    by_list: dict[str, set[float]] = {}
    for c in kept:
        _fam, cfg_list, win = _classify_family(c)
        if cfg_list is None or win is None:
            continue
        by_list.setdefault(cfg_list, set()).add(win)

    allowed: dict[str, list] = {}
    summary: dict[str, dict[str, list]] = {}
    for cfg_list, wins in by_list.items():
        original = list(getattr(cfg_mod, cfg_list, []))
        if not original:
            continue
        is_int_grid = all(isinstance(x, (int, np.integer)) for x in original)

        def _norm(x: float):
            return int(round(x)) if is_int_grid else float(x)

        wins_norm = {_norm(w) for w in wins}
        kept_in_order = [w for w in original if _norm(w) in wins_norm]
        if kept_in_order and len(kept_in_order) < len(original):
            allowed[cfg_list] = kept_in_order
        summary[cfg_list] = {
            "kept_windows": [str(x) for x in kept_in_order],
            "all_windows": [str(x) for x in original],
        }
    return allowed, summary


def _sanity_probe(
    df: pd.DataFrame,
    *,
    feat_all: list[str],
    feat_kept: list[str],
    cfg_mod: Any,
) -> dict[str, float]:
    """AUC + PR-AUC auf dem letzten WF-Fold (jüngstes Regime). Verwendet sample-effiziente
    Hyperparameter aus SEED_PARAMS."""
    all_dates = np.sort(pd.to_datetime(df["Date"]).unique())
    n_dates = int(len(all_dates))
    folds = _wf_fold_indices(n_dates, int(getattr(cfg_mod, "N_WF_SPLITS", 5)))
    if not folds:
        return {"auc_all": np.nan, "auc_kept": np.nan, "prauc_all": np.nan, "prauc_kept": np.nan}
    train_end, val_end = folds[-1]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    date_idx = pd.to_datetime(df["Date"]).map(date_to_idx).to_numpy(dtype=np.int64)
    tr_mask = date_idx < train_end
    va_mask = (date_idx >= train_end) & (date_idx < val_end)
    sentinel = float(getattr(cfg_mod, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    y_tr = df.loc[tr_mask, "target"].to_numpy(dtype=np.int8)
    y_va = df.loc[va_mask, "target"].to_numpy(dtype=np.int8)
    seed_params = dict(getattr(cfg_mod, "SEED_PARAMS", {}))
    seed = int(getattr(cfg_mod, "RANDOM_STATE", 42)) + 999

    def _eval(cols: list[str]) -> tuple[float, float]:
        X_tr = df.loc[tr_mask, cols].to_numpy(dtype=np.float32, copy=True)
        X_va = df.loc[va_mask, cols].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(X_tr, nan=sentinel, posinf=sentinel, neginf=sentinel, copy=False)
        np.nan_to_num(X_va, nan=sentinel, posinf=sentinel, neginf=sentinel, copy=False)
        bst = _train_fold_xgb(X_tr, y_tr, X_va, y_va, seed_params=seed_params, cfg_mod=cfg_mod, seed=seed)
        raw = bst.predict(xgb.DMatrix(X_va))
        prob = 1.0 / (1.0 + np.exp(-raw))
        if y_va.sum() == 0 or y_va.sum() == len(y_va):
            return float("nan"), float("nan")
        return float(roc_auc_score(y_va, prob)), float(average_precision_score(y_va, prob))

    auc_all, prauc_all = _eval(feat_all)
    auc_kept, prauc_kept = _eval(feat_kept)
    return {
        "auc_all": auc_all,
        "auc_kept": auc_kept,
        "prauc_all": prauc_all,
        "prauc_kept": prauc_kept,
        "n_train": int(tr_mask.sum()),
        "n_val": int(va_mask.sum()),
    }


def _input_signature(
    df: pd.DataFrame, feat_cols: list[str], cfg_mod: Any
) -> dict[str, Any]:
    seed_params = dict(getattr(cfg_mod, "SEED_PARAMS", {}))
    sp_keys = sorted(
        [
            "rsi_window",
            "bb_window",
            "sma_window",
            "max_depth",
            "max_leaves",
            "max_bin",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "focal_gamma",
            "focal_alpha",
        ]
    )
    sp_subset = {k: seed_params.get(k) for k in sp_keys}
    payload = {
        "n_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
        "feat_cols_hash": hashlib.md5("\n".join(sorted(feat_cols)).encode("utf-8")).hexdigest(),
        "seed_params_hash": hashlib.md5(
            json.dumps(sp_subset, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "random_state": int(getattr(cfg_mod, "RANDOM_STATE", 42)),
        "n_wf_splits": int(getattr(cfg_mod, "N_WF_SPLITS", 5)),
        "n_shadows": int(getattr(cfg_mod, "FEATURE_PRESCREEN_N_SHADOWS", 20)),
        "noise_quantile": float(getattr(cfg_mod, "FEATURE_PRESCREEN_NOISE_QUANTILE", 0.95)),
        "family_topk": int(getattr(cfg_mod, "FEATURE_PRESCREEN_FAMILY_TOPK", 2)),
        "global_topk": int(getattr(cfg_mod, "FEATURE_PRESCREEN_GLOBAL_TOPK", 30)),
        "macro_topk": int(getattr(cfg_mod, "FEATURE_PRESCREEN_MACRO_TOPK", 5)),
    }
    if "Date" in df.columns and len(df) > 0:
        d = pd.to_datetime(df["Date"], errors="coerce")
        payload["date_min"] = str(d.min().date())
        payload["date_max"] = str(d.max().date())
    return payload


def _try_load_cached(path: Path, sig: dict[str, Any], cfg_mod: Any) -> dict | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[Pre-Screen] Cache nicht lesbar ({exc}) — wird neu gebaut.", flush=True)
        return None
    if cached.get("input_signature") != sig:
        if bool(getattr(cfg_mod, "FEATURE_PRESCREEN_REUSE_SAME_CALENDAR_DAY", True)):
            built_on = cached.get("built_on")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if built_on == today:
                print(
                    f"[Pre-Screen] Cache-Signatur weicht ab, aber built_on=heute ({built_on}) — "
                    "übernehme bestehendes Artefakt (REUSE_SAME_CALENDAR_DAY=True).",
                    flush=True,
                )
                return cached
        print("[Pre-Screen] Cache-Signatur weicht ab — wird neu gebaut.", flush=True)
        return None
    print(f"[Pre-Screen] Cache übernommen: {path}", flush=True)
    return cached


def _save_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_stage(num: int, title: str) -> None:
    print(f"\n--- Stage {num} — {title} ---", flush=True)


def _log_constancy_drops(
    feat_cols_full: list[str], feat_cols: list[str], drop_const: dict[str, str]
) -> None:
    print(
        f"  Kandidaten total: {len(feat_cols_full)}  -> behalten: {len(feat_cols)}  "
        f"(verworfen: {len(drop_const)})",
        flush=True,
    )
    if not drop_const:
        return
    by_reason: dict[str, list[str]] = {}
    for c, r in drop_const.items():
        by_reason.setdefault(r, []).append(c)
    for reason, cols in sorted(by_reason.items()):
        head = ", ".join(sorted(cols)[:8])
        more = f"  (+{len(cols)-8} weitere)" if len(cols) > 8 else ""
        print(f"     [{reason}] {len(cols)}: {head}{more}", flush=True)


def _log_importance_leaderboard(
    importance: dict[str, float], shadow_arr: np.ndarray, top_n: int = 15
) -> None:
    """Top-N Real-Features + Shadow-Verteilungsstatistik."""
    if not importance:
        return
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    n = min(int(top_n), len(items))
    print(
        f"  Top-{n} Real-Features (Median |SHAP| über alle Folds):",
        flush=True,
    )
    width = max(len(c) for c, _ in items[:n])
    for i, (col, val) in enumerate(items[:n], start=1):
        print(f"     {i:>2}. {col:<{width}}  {val:.4e}", flush=True)
    if shadow_arr.size > 0:
        q05 = float(np.quantile(shadow_arr, 0.05))
        q50 = float(np.quantile(shadow_arr, 0.50))
        q95 = float(np.quantile(shadow_arr, 0.95))
        smin = float(shadow_arr.min())
        smax = float(shadow_arr.max())
        print(
            f"  Shadow-Verteilung (n={shadow_arr.size}): "
            f"min={smin:.3e}  q05={q05:.3e}  q50={q50:.3e}  q95={q95:.3e}  max={smax:.3e}",
            flush=True,
        )


def _log_drop_summary(
    importance: dict[str, float],
    dropped: set[str],
    reasons: dict[str, str],
    floor: float,
    *,
    list_limit: int = 40,
) -> None:
    """Histogramm der Drop-Gründe + Liste der verworfenen Features."""
    if not dropped:
        print("  Verworfene Features: 0  (keine droppen über alle Stage-3-Regeln).", flush=True)
        return
    # Reason-Histogramm: kürze den dynamischen Suffix („imp=…“) auf den Kategorienamen.
    cat_counts: dict[str, int] = {}
    for c in dropped:
        r = reasons.get(c, "?")
        if r.startswith("below_floor"):
            cat = "below_floor"
        elif r.startswith("sanity_revert"):
            cat = "sanity_revert"
        else:
            cat = r
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(
        f"  Verworfene Features: {len(dropped)}   Floor (Q-Schatten)={floor:.4e}",
        flush=True,
    )
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f"     [{cat}] {n}", flush=True)

    dropped_sorted = sorted(dropped, key=lambda c: importance.get(c, 0.0), reverse=True)
    print("  Liste verworfener Features (sortiert nach |SHAP| absteigend):", flush=True)
    width = max((len(c) for c in dropped_sorted[:list_limit]), default=10)
    for c in dropped_sorted[:list_limit]:
        imp = importance.get(c, 0.0)
        print(f"     - {c:<{width}}  imp={imp:.3e}   {reasons.get(c, '?')}", flush=True)
    if len(dropped_sorted) > list_limit:
        print(f"     ... +{len(dropped_sorted) - list_limit} weitere (siehe Artefakt-JSON)", flush=True)


def _log_family_summary(
    importance: dict[str, float],
    kept: set[str],
    cfg_mod: Any,
) -> None:
    """Pro Familie: Fenster nach Importance sortiert, OK/X je nach kept-Status."""
    fam_groups: dict[str, list[str]] = {}
    for col in importance:
        fam, cfg_list, _w = _classify_family(col)
        if fam is None:
            continue
        # Nur Familien, die ein cfg-Grid steuern (bb_x_rsi etc. ausgeschlossen).
        if cfg_list is None:
            continue
        fam_groups.setdefault(fam, []).append(col)
    if not fam_groups:
        return
    print("  Familien-Uebersicht ([OK] = behalten, [X ] = verworfen):", flush=True)
    for fam in sorted(fam_groups):
        members = sorted(fam_groups[fam], key=lambda c: importance.get(c, 0.0), reverse=True)
        flags = []
        for c in members:
            mark = "[OK]" if c in kept else "[X ]"
            flags.append(f"{mark} {c} ({importance.get(c, 0.0):.2e})")
        print(f"     {fam}:  " + "  |  ".join(flags), flush=True)


def _log_sanity_decision(
    sanity: dict, sanity_ok: bool, auc_tol: float, prauc_tol: float
) -> None:
    auc_d = (sanity["auc_all"] - sanity["auc_kept"]) if (
        np.isfinite(sanity["auc_all"]) and np.isfinite(sanity["auc_kept"])
    ) else float("nan")
    pra_d = (sanity["prauc_all"] - sanity["prauc_kept"]) if (
        np.isfinite(sanity["prauc_all"]) and np.isfinite(sanity["prauc_kept"])
    ) else float("nan")
    decision = "PASS — Auswahl wird angewandt" if sanity_ok else "FAIL — Revert auf Vollset"
    print(
        f"  Letzter Fold (n_train={sanity.get('n_train', '?')}, n_val={sanity.get('n_val', '?')}):",
        flush=True,
    )
    print(
        f"     AUC    all={sanity['auc_all']:.4f}  kept={sanity['auc_kept']:.4f}  "
        f"delta={auc_d:+.4f}  (tol={auc_tol:.3f})",
        flush=True,
    )
    print(
        f"     PR-AUC all={sanity['prauc_all']:.4f}  kept={sanity['prauc_kept']:.4f}  "
        f"delta={pra_d:+.4f}  (tol={prauc_tol:.3f})",
        flush=True,
    )
    print(f"     -> {decision}", flush=True)


def _log_grid_shrinkage(allowed_windows: dict, cfg_mod: Any) -> None:
    if not allowed_windows:
        print("    (keine Optuna-Grid-Schrumpfung — alle Original-Fenster bleiben.)", flush=True)
        return
    print("  Geschrumpfte Optuna-Grids (gegenüber config.py):", flush=True)
    for k in sorted(allowed_windows):
        full = list(getattr(cfg_mod, k, []))
        eff = list(allowed_windows[k])
        removed = [str(x) for x in full if x not in eff]
        rm_str = f"  (-{','.join(removed)})" if removed else ""
        print(f"     {k}: {full}  ->  {eff}{rm_str}", flush=True)


def run_feature_prescreen(df: pd.DataFrame, *, cfg_mod: Any) -> dict | None:
    """Führt das Pre-Screening aus und liefert das Artefakt (oder ``None`` bei Skip).

    Persistiert in ``FEATURE_PRESCREEN_DIR / FEATURE_PRESCREEN_ARTIFACT_NAME``.
    Setzt **nicht** ``cfg._FEATURE_PRESCREEN_ARTIFACT`` — das macht der Aufrufer.
    """
    if not bool(getattr(cfg_mod, "FEATURE_PRESCREEN_ENABLED", False)):
        return None
    if df is None or len(df) == 0:
        print("[Pre-Screen] df leer — überspringe.", flush=True)
        return None
    print("=" * 72, flush=True)
    print("Feature Pre-Screen (vor Optuna Phase 1)", flush=True)
    print("=" * 72, flush=True)
    n_folds = int(getattr(cfg_mod, "N_WF_SPLITS", 5))
    n_shadows = int(getattr(cfg_mod, "FEATURE_PRESCREEN_N_SHADOWS", 20))
    sample_per_fold = int(getattr(cfg_mod, "FEATURE_PRESCREEN_SHAP_SAMPLE_PER_FOLD", 20000))
    noise_q = float(getattr(cfg_mod, "FEATURE_PRESCREEN_NOISE_QUANTILE", 0.95))
    family_topk = int(getattr(cfg_mod, "FEATURE_PRESCREEN_FAMILY_TOPK", 2))
    global_topk = int(getattr(cfg_mod, "FEATURE_PRESCREEN_GLOBAL_TOPK", 30))
    macro_topk = int(getattr(cfg_mod, "FEATURE_PRESCREEN_MACRO_TOPK", 5))
    auc_tol = float(getattr(cfg_mod, "FEATURE_PRESCREEN_AUC_TOL", 0.005))
    prauc_tol = float(getattr(cfg_mod, "FEATURE_PRESCREEN_PRAUC_TOL", 0.005))
    print(
        f"  Settings: WF-Folds={n_folds}, Shadows/Fold={n_shadows}, "
        f"Noise-Quantil={noise_q:.2f}, family_topk={family_topk}, "
        f"global_topk={global_topk}, macro_topk={macro_topk}, "
        f"AUC-Tol={auc_tol:.3f}, PR-AUC-Tol={prauc_tol:.3f}",
        flush=True,
    )
    print(
        f"  Daten: rows={len(df):,}  Ticker={int(df['ticker'].nunique()) if 'ticker' in df.columns else '?'}  "
        f"Datumsbereich="
        + (
            f"{pd.to_datetime(df['Date']).min().date()} … {pd.to_datetime(df['Date']).max().date()}"
            if "Date" in df.columns
            else "?"
        ),
        flush=True,
    )

    _print_stage(0, "Konstanz/Sentinel-Filter")
    feat_cols_full = _collect_candidate_features(df, cfg_mod)
    sentinel = float(getattr(cfg_mod, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    feat_cols, drop_const = _stage0_constancy_filter(df, feat_cols_full, sentinel=sentinel)
    _log_constancy_drops(feat_cols_full, feat_cols, drop_const)
    if not feat_cols:
        print("[Pre-Screen] keine Kandidaten nach Konstanzfilter — überspringe.", flush=True)
        return None

    art_dir = Path(getattr(cfg_mod, "FEATURE_PRESCREEN_DIR", "data"))
    art_path = art_dir / str(getattr(cfg_mod, "FEATURE_PRESCREEN_ARTIFACT_NAME", "feature_prescreen_v1.json"))
    sig = _input_signature(df, feat_cols, cfg_mod)
    cached = _try_load_cached(art_path, sig, cfg_mod)
    if cached is not None:
        return cached

    _print_stage(1, "Importance-Ensemble (Walk-Forward + TreeSHAP)")
    t_start = time.perf_counter()
    try:
        importance, shadow_arr, diag = _walkforward_importance(
            df,
            feat_cols,
            cfg_mod=cfg_mod,
            n_folds=n_folds,
            n_shadows=n_shadows,
            sample_per_fold=sample_per_fold,
            rng_seed=int(getattr(cfg_mod, "RANDOM_STATE", 42)) + 7,
            verbose=True,
        )
    except (ValueError, KeyError) as exc:
        print(f"[Pre-Screen] Importance-Pass fehlgeschlagen ({exc}) — überspringe.", flush=True)
        return None
    elapsed_imp = time.perf_counter() - t_start
    print(
        f"  Importance-Pass abgeschlossen in {elapsed_imp:.1f}s "
        f"(gültige Folds: {diag['n_folds_used']}/{n_folds}).",
        flush=True,
    )
    _log_importance_leaderboard(importance, shadow_arr, top_n=15)

    _print_stage(2, "Boruta-Schatten Rauschfloor")
    floor = float(np.quantile(shadow_arr, noise_q)) if shadow_arr.size > 0 else 0.0
    n_above_floor = int(sum(1 for v in importance.values() if v >= floor))
    print(
        f"  Floor (Q{noise_q:.0%} der Schatten-Importance) = {floor:.4e}",
        flush=True,
    )
    print(
        f"  Real-Features oberhalb Floor: {n_above_floor}/{len(importance)} "
        f"({100.0 * n_above_floor / max(1, len(importance)):.1f}%)",
        flush=True,
    )

    _print_stage(3, "Familien-bewusste Auswahl")
    kept, dropped, reasons = _select_features(
        importance,
        floor=floor,
        family_topk=family_topk,
        global_topk=global_topk,
        macro_topk=macro_topk,
    )
    print(
        f"  Auswahl-Ergebnis: {len(kept)} behalten / {len(dropped)} verworfen "
        f"(von {len(importance)} bewerteten Features)",
        flush=True,
    )
    _log_drop_summary(importance, dropped, reasons, floor)
    _log_family_summary(importance, kept, cfg_mod)

    _print_stage(4, "Sanity-Probe auf jüngstem Fold")
    feat_kept_list = sorted(kept)
    feat_all_list = list(importance.keys())
    sanity = _sanity_probe(df, feat_all=feat_all_list, feat_kept=feat_kept_list, cfg_mod=cfg_mod)
    sanity_ok = True
    if np.isfinite(sanity["auc_all"]) and np.isfinite(sanity["auc_kept"]):
        if (sanity["auc_all"] - sanity["auc_kept"]) > auc_tol:
            sanity_ok = False
    if np.isfinite(sanity["prauc_all"]) and np.isfinite(sanity["prauc_kept"]):
        if (sanity["prauc_all"] - sanity["prauc_kept"]) > prauc_tol:
            sanity_ok = False
    _log_sanity_decision(sanity, sanity_ok, auc_tol, prauc_tol)

    sanity_revert = not sanity_ok
    if sanity_revert:
        kept = set(feat_all_list)
        dropped = set()
        for c in feat_all_list:
            reasons[c] = "sanity_revert: " + reasons.get(c, "?")

    _print_stage(5, "Anwendung auf Optuna-Search-Space")
    if sanity_revert:
        allowed_windows: dict[str, list] = {}
        _, family_summary = _compute_allowed_windows(kept, cfg_mod)
    else:
        allowed_windows, family_summary = _compute_allowed_windows(kept, cfg_mod)
    _log_grid_shrinkage(allowed_windows, cfg_mod)

    importance_sorted = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    artifact = {
        "version": 1,
        "built_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_signature": sig,
        "kept_features": sorted(kept),
        "dropped_features": sorted(dropped),
        "allowed_windows": {k: list(v) for k, v in allowed_windows.items()},
        "family_summary": family_summary,
        "noise_floor": floor,
        "metrics": sanity,
        "importance_top_50": [{"feature": k, "mean_abs_shap": v} for k, v in importance_sorted[:50]],
        "drop_reasons": {k: reasons.get(k, "?") for k in sorted(dropped)},
        "stage0_constancy_dropped": drop_const,
        "diagnostics": {
            "n_features_in": len(feat_cols_full),
            "n_features_after_constancy": len(feat_cols),
            "n_features_kept": len(kept),
            "n_features_dropped": len(dropped),
            "elapsed_importance_s": round(elapsed_imp, 2),
            "fold_diagnostics": diag,
            "sanity_passed": bool(sanity_ok),
            "n_above_floor": n_above_floor,
        },
    }
    try:
        _save_artifact(art_path, artifact)
        print(f"  Artefakt geschrieben: {art_path}", flush=True)
    except OSError as exc:
        print(f"[Pre-Screen] Artefakt-Speichern fehlgeschlagen ({exc}).", flush=True)

    print("=" * 72, flush=True)
    print(
        f"Pre-Screen Zusammenfassung: "
        f"in={len(feat_cols_full)}  konstant_drop={len(drop_const)}  "
        f"shap_evaluiert={len(importance)}  kept={len(kept)}  dropped={len(dropped)}  "
        f"grids_geschrumpft={len(allowed_windows)}  sanity={'PASS' if sanity_ok else 'REVERT'}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return artifact


__all__ = ["run_feature_prescreen"]
