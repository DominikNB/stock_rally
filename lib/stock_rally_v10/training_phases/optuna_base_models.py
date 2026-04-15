"""Phase 12: Optuna, Rebuild, Base-Modelle, SHAP."""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib.stock_rally_v10.features import merge_news_shard_from_best_params
from lib.stock_rally_v10.extended_base_features import append_macro_regime_vol_numeric_cols


def _resolve_meta_shap_top_k(c: Any, mean_shap: np.ndarray) -> tuple[int, float]:
    """
    Anzahl Roh-Features für Meta-Stacking: festes K oder kleinstes K mit
    cumsum(mean|SHAP| der Top-K) >= META_SHAP_CUM_FRAC * sum(mean|SHAP|).
    Rückgabe: (K, erreichte kumulierte Masse / Gesamtsumme nach Clip).
    """
    n_feat = int(len(mean_shap))
    k_min = max(1, int(getattr(c, "META_SHAP_TOP_K_MIN", 1)))
    k_cap = getattr(c, "META_SHAP_TOP_K_MAX", None)
    k_max = n_feat if k_cap is None else max(k_min, min(int(k_cap), n_feat))

    total = float(np.sum(mean_shap))
    cum_frac = getattr(c, "META_SHAP_CUM_FRAC", None)
    if cum_frac is not None:
        f = float(cum_frac)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"META_SHAP_CUM_FRAC muss in (0, 1] liegen, nicht {f!r}")
        if total <= 0.0:
            k = max(k_min, min(int(getattr(c, "META_SHAP_TOP_K", 10)), k_max))
        else:
            order = np.argsort(mean_shap)[::-1]
            vcum = np.cumsum(mean_shap[order])
            thr = f * total
            k = int(np.searchsorted(vcum, thr, side="left")) + 1
            k = max(k_min, min(k, k_max))
    else:
        k = max(k_min, min(int(getattr(c, "META_SHAP_TOP_K", 10)), k_max))

    order = np.argsort(mean_shap)[::-1][:k]
    mass = float(np.sum(mean_shap[order])) / total if total > 0 else 0.0
    return k, mass


def _log_feature_health(dataset_label: str, df: pd.DataFrame, feat_cols: list[str], top_n: int = 15) -> None:
    """Diagnose NaN/Inf in Feature-Spalten (Phase12, gleiche Struktur wie Phase17)."""
    if df is None or len(df) == 0:
        print(f"[Feature-Diagnose] {dataset_label}: leer.", flush=True)
        return
    if not feat_cols:
        print(f"[Feature-Diagnose] {dataset_label}: keine FEAT_COLS.", flush=True)
        return
    feat_df = df[feat_cols]
    bad_cols: list[tuple[str, int, int, int]] = []
    bad_rows_mask = np.zeros(len(feat_df), dtype=bool)
    nan_cells = 0
    inf_cells = 0
    for col in feat_cols:
        s = pd.to_numeric(feat_df[col], errors="coerce")
        arr = s.to_numpy(dtype=np.float64, copy=False)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        n_bad = n_nan + n_inf
        nan_cells += n_nan
        inf_cells += n_inf
        if n_bad:
            bad_cols.append((col, n_bad, n_nan, n_inf))
            bad_rows_mask |= np.isnan(arr) | np.isinf(arr)
    bad_rows = int(bad_rows_mask.sum())
    if not bad_cols:
        print(
            f"Phase12 [{dataset_label}]: FEAT_COLS ok (keine NaN/Inf in {len(feat_cols)} Spalten, {len(df):,} Zeilen).",
            flush=True,
        )
        return
    bad_cols.sort(key=lambda x: x[1], reverse=True)
    print(
        f"Phase12 [{dataset_label}]: Problematische FEAT_COLS erkannt "
        f"(NaN={nan_cells}, Inf={inf_cells}, Zeilen mit >=1 Problemwert: {bad_rows}/{len(df):,}).",
        flush=True,
    )
    n_rows = max(1, len(df))
    print("  Top-Spalten nach Problemwerten (count = NaN+Inf):", flush=True)
    for col, n_bad, n_nan, n_inf in bad_cols[:top_n]:
        pct_bad = 100.0 * float(n_bad) / float(n_rows)
        print(
            f"    - {col}: {n_bad} ({pct_bad:.1f}%) (NaN={n_nan}, Inf={n_inf})",
            flush=True,
        )
    if len(bad_cols) > top_n:
        print(f"    ... +{len(bad_cols) - top_n} weitere Spalten", flush=True)


def run_phase_optuna_base_models(cfg_mod: Any) -> None:
    if getattr(cfg_mod, "SCORING_ONLY", False):
        print("[SCORING_ONLY] Training-Zelle übersprungen.")
        return
    _run_phase12(cfg_mod)


def _run_phase12(c: Any) -> None:
    c._SCORING_ARTIFACT_SAVED_THIS_SESSION = False

    try:
        import mlflow

        mlflow.autolog(disable=True)
    except Exception:
        pass

    print("=" * 60)
    print("Phase 1: Optuna base-model HP optimisation")
    print("=" * 60)
    best_params = c.optimize_xgb(c.df_train)

    opt_y = getattr(c, "OPT_OPTIMIZE_Y_TARGETS", True)
    sp = c.SEED_PARAMS
    if opt_y:
        RETURN_WINDOW = best_params["return_window"]
        RALLY_THRESHOLD = best_params["rally_threshold"]
        LEAD_DAYS = best_params["lead_days"]
        ENTRY_DAYS = best_params["entry_days"]
        MIN_RALLY_TAIL_DAYS = best_params.get("min_rally_tail_days", sp["min_rally_tail_days"])
    else:
        _, RETURN_WINDOW, RALLY_THRESHOLD, _, LEAD_DAYS, ENTRY_DAYS, _ = c.fixed_y_rule_params()
        MIN_RALLY_TAIL_DAYS = 5

    CONSECUTIVE_DAYS = best_params["consecutive_days"]
    SIGNAL_COOLDOWN_DAYS = best_params["signal_cooldown_days"]
    BEST_THRESHOLD = best_params.get("threshold", sp["threshold"])
    SIGNAL_SKIP_NEAR_PEAK = sp["signal_skip_near_peak"]
    PEAK_LOOKBACK_DAYS = int(sp["peak_lookback_days"])
    PEAK_MIN_DIST_FROM_HIGH_PCT = float(sp["peak_min_dist_from_high_pct"])
    _v_rsi_se = sp.get("signal_max_rsi", getattr(c, "SIGNAL_MAX_RSI", None))
    SIGNAL_MAX_RSI = float(_v_rsi_se) if _v_rsi_se is not None else None

    if opt_y:
        print(
            f"\nOptimised rally def:      return_window={RETURN_WINDOW}, "
            f"rally_threshold={RALLY_THRESHOLD:.2%}"
        )
        print(
            f"Optimised target params:  lead_days={LEAD_DAYS}, entry_days={ENTRY_DAYS}, "
            f"min_rally_tail_days={MIN_RALLY_TAIL_DAYS}"
        )
    else:
        print(
            f"\nFixed rally band (no y-opt): windows {c.FIXED_Y_WINDOW_MIN}-{c.FIXED_Y_WINDOW_MAX}d, "
            f"threshold={RALLY_THRESHOLD:.2%}; target rule by segment vs {c.FIXED_Y_SEGMENT_SPLIT}d"
        )
        print(
            f"Fixed target (nur Info):  lead_days={LEAD_DAYS}, entry_days={ENTRY_DAYS} "
            f"(Rebuild nutzt _create_target_one_ticker_fixed_bands)"
        )
    print(
        f"Optimised filter params:  consecutive_days={CONSECUTIVE_DAYS}, "
        f"signal_cooldown_days={SIGNAL_COOLDOWN_DAYS}"
    )
    print(f"Seed threshold (bis Phase 5): {BEST_THRESHOLD:.3f}")
    _rsi_s = f"{SIGNAL_MAX_RSI:.1f}" if SIGNAL_MAX_RSI is not None else "off"
    print(
        f"Anti-peak (SEED_PARAMS bis Meta): skip={SIGNAL_SKIP_NEAR_PEAK}, lookback={PEAK_LOOKBACK_DAYS}d, "
        f"minDist={PEAK_MIN_DIST_FROM_HIGH_PCT:.4f}, maxRSI={_rsi_s}"
    )

    _rebuild_kw = dict(
        return_window=RETURN_WINDOW,
        rally_threshold=RALLY_THRESHOLD,
        min_rally_tail_days=MIN_RALLY_TAIL_DAYS,
    )
    df_train = c.rebuild_target_for_train(c.df_train, LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_test = c.rebuild_target_for_train(c.df_test, LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_threshold = c.rebuild_target_for_train(c.df_threshold, LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_final = c.rebuild_target_for_train(c.df_final, LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    if c.USE_NEWS_SENTIMENT:
        df_train = merge_news_shard_from_best_params(df_train, best_params)
        df_test = merge_news_shard_from_best_params(df_test, best_params)
        df_threshold = merge_news_shard_from_best_params(df_threshold, best_params)
        df_final = merge_news_shard_from_best_params(df_final, best_params)
    for _name, _df in [
        ("df_train", df_train),
        ("df_test", df_test),
        ("df_threshold", df_threshold),
        ("df_final", df_final),
    ]:
        print(f"{_name:15s} target rebuilt  (positive rate: {_df['target'].mean():.1%})")

    rsi_w = best_params["rsi_window"]
    bb_w = best_params["bb_window"]
    sma_w = best_params["sma_window"]
    _btc_z = int(best_params.get("btc_momentum_z_window", sp.get("btc_momentum_z_window", 60)))
    _brd_z = int(best_params.get("market_breadth_z_window", sp.get("market_breadth_z_window", 60)))
    _rel_m = int(best_params.get("rel_momentum_window", sp.get("rel_momentum_window", 20)))
    if c.USE_NEWS_SENTIMENT:
        FEAT_COLS = c.build_feature_cols(
            rsi_w,
            bb_w,
            sma_w,
            best_params["news_mom_w"],
            best_params["news_vol_ma"],
            best_params["news_tone_roll"],
            best_params.get("news_extra_zscore_w", sp.get("news_extra_zscore_w")),
            best_params.get("news_extra_tone_accel", sp.get("news_extra_tone_accel")),
            best_params.get("news_extra_macro_sec_diff", sp.get("news_extra_macro_sec_diff")),
            btc_momentum_z_window=_btc_z,
            market_breadth_z_window=_brd_z,
            rel_momentum_window=_rel_m,
        )
    else:
        FEAT_COLS = c.build_feature_cols(
            rsi_w,
            bb_w,
            sma_w,
            btc_momentum_z_window=_btc_z,
            market_breadth_z_window=_brd_z,
            rel_momentum_window=_rel_m,
        )
    FEAT_COLS = append_macro_regime_vol_numeric_cols(FEAT_COLS, df_train)
    print(
        f"\nUsing features: RSI={rsi_w}, BB={bb_w}, SMA={sma_w}, "
        f"BTCz={_btc_z}, BreadthZ={_brd_z}, relMom={_rel_m}d  ({len(FEAT_COLS)} features)"
    )

    focal_gamma = best_params["focal_gamma"]
    focal_alpha = best_params["focal_alpha"]
    focal_obj = c.make_focal_objective(focal_gamma, focal_alpha)
    focal_obj_lgb = c.make_focal_objective_lgb(focal_gamma, focal_alpha)

    xgb_base_params = {
        k: v
        for k, v in best_params.items()
        if k not in (
            "rsi_window",
            "bb_window",
            "sma_window",
            "news_mom_w",
            "news_vol_ma",
            "news_tone_roll",
            "news_extra_zscore_w",
            "news_extra_tone_accel",
            "news_extra_macro_sec_diff",
            "btc_momentum_z_window",
            "market_breadth_z_window",
            "rel_momentum_window",
            "focal_gamma",
            "focal_alpha",
            "return_window",
            "rally_threshold",
            "lead_days",
            "entry_days",
            "min_rally_tail_days",
            "consecutive_days",
            "signal_cooldown_days",
            "threshold",
            "signal_skip_near_peak",
            "peak_lookback_days",
            "peak_min_dist_from_high_pct",
            "signal_max_rsi",
        )
    }
    xgb_base_params["tree_method"] = "hist"

    lgb_params = dict(
        max_depth=best_params["max_depth"],
        num_leaves=min(best_params.get("max_leaves", 127), 255),
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        n_estimators=best_params["n_estimators"],
        verbose=-1,
    )

    X_train_all = df_train[FEAT_COLS].values.astype(np.float32)
    y_train_all = df_train["target"].values.astype(np.int8)

    print("\n" + "=" * 60)
    print("Feature-Diagnose vor Base-Training")
    print("=" * 60)
    _log_feature_health("BASE TRAIN", df_train, FEAT_COLS)
    _log_feature_health("META TRAIN (for stacking inputs)", df_test, FEAT_COLS)
    _log_feature_health("THRESHOLD", df_threshold, FEAT_COLS)
    _log_feature_health("FINAL", df_final, FEAT_COLS)

    print("\n" + "=" * 60)
    print("Phase 2: Training 10 base models")
    print("=" * 60)
    base_models = []

    def _bootstrap_split(seed_i, X, y):
        rng = np.random.RandomState(seed_i)
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        oob_idx = np.setdiff1d(np.arange(len(X)), boot_idx)
        if len(oob_idx) < 10:
            oob_idx = boot_idx[int(len(boot_idx) * 0.9) :]
        return X[boot_idx], y[boot_idx], X[oob_idx], y[oob_idx]

    rs = c.RANDOM_STATE
    esr = c.EARLY_STOPPING_ROUNDS

    for m_idx, seed_i in enumerate([rs, rs + 1, rs + 2, rs + 3]):
        name = f"XGB-{m_idx+1}"
        print(f"  Training {name}...")
        X_fit, y_fit, X_es, y_es = _bootstrap_split(seed_i, X_train_all, y_train_all)
        dtrain_m = xgb.DMatrix(X_fit, label=y_fit)
        des_m = xgb.DMatrix(X_es, label=y_es)
        p = {**xgb_base_params, "seed": seed_i, "disable_default_eval_metric": 1}
        bst = xgb.train(
            p,
            dtrain_m,
            num_boost_round=xgb_base_params["n_estimators"],
            obj=focal_obj,
            evals=[(des_m, "es")],
            custom_metric=lambda pr, d: (
                "logloss",
                float(
                    np.mean(
                        -d.get_label() * np.log(np.clip(1 / (1 + np.exp(-pr)), 1e-7, 1 - 1e-7))
                        - (1 - d.get_label()) * np.log(np.clip(1 / (1 + np.exp(pr)), 1e-7, 1 - 1e-7))
                    )
                ),
            ),
            early_stopping_rounds=esr,
            verbose_eval=False,
        )
        base_models.append((name, bst, "xgb"))
        print(f"  {name} done — best iteration: {bst.best_iteration}")

    for m_idx, seed_i in enumerate([rs + 10, rs + 11, rs + 12]):
        name = f"LGB-{m_idx+1}"
        print(f"  Training {name}...")
        X_fit, y_fit, X_es, y_es = _bootstrap_split(seed_i, X_train_all, y_train_all)
        dtrain_lgb = lgb.Dataset(X_fit, label=y_fit)
        des_lgb = lgb.Dataset(X_es, label=y_es, reference=dtrain_lgb)
        callbacks = [lgb.early_stopping(esr, verbose=False), lgb.log_evaluation(-1)]
        p_lgb = {**lgb_params, "seed": seed_i, "verbosity": -1}
        p_lgb["objective"] = focal_obj_lgb
        p_lgb["metric"] = "binary_logloss"
        n_est = p_lgb.pop("n_estimators", 300)
        bst_lgb = lgb.train(
            p_lgb,
            dtrain_lgb,
            num_boost_round=n_est,
            valid_sets=[des_lgb],
            callbacks=callbacks,
        )
        base_models.append((name, bst_lgb, "lgb"))
        print(f"  {name} done — best iteration: {bst_lgb.best_iteration}")

    print("  Training RF...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=best_params["max_depth"],
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=rs,
        n_jobs=-1,
    )
    rf.fit(X_train_all, y_train_all)
    base_models.append(("RF", rf, "rf"))
    print("  RF done.")

    print("  Training ET...")
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=best_params["max_depth"],
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=rs + 20,
        n_jobs=-1,
    )
    et.fit(X_train_all, y_train_all)
    base_models.append(("ET", et, "et"))
    print("  ET done.")

    print("  Training LR...")
    # Macro-Regime/mr_* können NaN haben (Yahoo, Rollen); Bäume tolerieren das, LogisticRegression nicht.
    lr_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=rs,
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lr_pipe.fit(X_train_all, y_train_all)
    base_models.append(("LR", lr_pipe, "lr"))
    print("  LR done.")

    print(f"\nBase models ready: {[m[0] for m in base_models]}")

    print("\n" + "=" * 60)
    print("Phase 3: SHAP feature selection")
    print("=" * 60)

    X_test_feat = df_test[FEAT_COLS].values.astype(np.float32)
    X_final_feat = df_final[FEAT_COLS].values.astype(np.float32)

    rename_map = c.build_rename_map(
        rsi_w,
        bb_w,
        sma_w,
        best_params.get("news_mom_w"),
        best_params.get("news_vol_ma"),
        best_params.get("news_tone_roll"),
        best_params.get("news_extra_zscore_w"),
        best_params.get("news_extra_tone_accel"),
        best_params.get("news_extra_macro_sec_diff"),
        btc_momentum_z_window=_btc_z,
        market_breadth_z_window=_brd_z,
        rel_momentum_window=_rel_m,
    )
    feat_display = [rename_map.get(col, col) for col in FEAT_COLS]

    _shap_hp = {
        k: best_params[k]
        for k in (
            "max_depth",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "max_leaves",
        )
        if k in best_params
    }
    print("SHAP-Modell: XGB-1; verwendete optimierte XGB-Hyperparameter:", _shap_hp)

    rng_shap = np.random.RandomState(rs)
    shap_idx = rng_shap.choice(len(X_test_feat), size=min(2000, len(X_test_feat)), replace=False)
    xgb_model_1 = base_models[0][1]
    explainer = shap.TreeExplainer(xgb_model_1)
    shap_vals = explainer.shap_values(xgb.DMatrix(X_test_feat[shap_idx]))
    mean_shap = np.abs(shap_vals).mean(axis=0)

    shap_df = (
        pd.DataFrame({"feature": feat_display, "mean_abs_shap": mean_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print("\nMittlere |SHAP| (alle Features, absteigend):")
    print(shap_df.to_string(index=False))
    _shap_zeroish = (shap_df["mean_abs_shap"] <= 1e-12).sum()
    if _shap_zeroish:
        print(
            f"Hinweis: {_shap_zeroish} Feature(s) mit ~0 mittlerer |SHAP| — beim XGB-1 typisch für Spalten, "
            "auf die kein Split fällt (Redundanz, `colsample_bytree`, seltene Nutzung) oder die in der "
            "SHAP-Stichprobe (fast) konstant sind.",
            flush=True,
        )

    _shap_top = min(25, len(shap_df))
    fig, ax = plt.subplots(figsize=(8, max(6, _shap_top * 0.28)))
    top = shap_df.head(_shap_top).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("Mittlere |SHAP|")
    ax.set_title("SHAP: Base-XGB-1 (Optuna-Parameter), Test-Stichprobe")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(
        shap_vals,
        X_test_feat[shap_idx],
        feature_names=feat_display,
        max_display=min(20, len(FEAT_COLS)),
        show=True,
    )

    topk, _mass_frac = _resolve_meta_shap_top_k(c, mean_shap)
    topk_idx = np.argsort(mean_shap)[::-1][:topk]
    topk_names = [FEAT_COLS[i] for i in topk_idx]
    _cf = getattr(c, "META_SHAP_CUM_FRAC", None)
    if _cf is not None:
        print(
            f"\nMeta SHAP: K={topk} (Ziel kumul. Masse ≥ {_cf:.0%} der Summe mean|SHAP|; "
            f"erreicht {_mass_frac:.1%}) — Features: {[rename_map.get(n, n) for n in topk_names]}",
            flush=True,
        )
    else:
        print(
            f"\nTop {topk} SHAP features (Meta-Stacking, META_SHAP_TOP_K): "
            f"{[rename_map.get(n, n) for n in topk_names]}",
            flush=True,
        )
    print(
        "\nPhase 12 abgeschlossen (Base + SHAP). Weiter mit Meta-Learner / Threshold (Phase 13).",
        flush=True,
    )

    c.RETURN_WINDOW = RETURN_WINDOW
    c.RALLY_THRESHOLD = RALLY_THRESHOLD
    c.LEAD_DAYS = LEAD_DAYS
    c.ENTRY_DAYS = ENTRY_DAYS
    c.MIN_RALLY_TAIL_DAYS = MIN_RALLY_TAIL_DAYS
    c.CONSECUTIVE_DAYS = CONSECUTIVE_DAYS
    c.SIGNAL_COOLDOWN_DAYS = SIGNAL_COOLDOWN_DAYS
    c.BEST_THRESHOLD = BEST_THRESHOLD
    c.SIGNAL_SKIP_NEAR_PEAK = SIGNAL_SKIP_NEAR_PEAK
    c.PEAK_LOOKBACK_DAYS = PEAK_LOOKBACK_DAYS
    c.PEAK_MIN_DIST_FROM_HIGH_PCT = PEAK_MIN_DIST_FROM_HIGH_PCT
    c.SIGNAL_MAX_RSI = SIGNAL_MAX_RSI
    c.best_params = best_params
    c.df_train = df_train
    c.df_test = df_test
    c.df_threshold = df_threshold
    c.df_final = df_final
    c.rsi_w = rsi_w
    c.bb_w = bb_w
    c.sma_w = sma_w
    c.FEAT_COLS = FEAT_COLS
    c.base_models = base_models
    c.rename_map = rename_map
    c.topk_idx = topk_idx
    c.topk_names = topk_names
