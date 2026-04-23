"""Phase 13: Meta-Features, Meta-Optuna, Schwellenfindung, Artefakt-Speichern."""
from __future__ import annotations

import json
import time
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb

from lib.stock_rally_v10.features import merge_news_shard_from_best_params
from lib.stock_rally_v10.optuna_train import (
    _blue_sky_weak_volume_mask_1d,
    _dynamic_threshold_mask_1d,
    _vol_stress_mask_1d,
)


def run_phase_meta_learner_and_threshold(cfg_mod: Any) -> None:
    if getattr(cfg_mod, "SCORING_ONLY", False):
        print("[SCORING_ONLY] Training-Zelle übersprungen.")
        return
    _run_phase13(cfg_mod)


def _run_phase13(c: Any) -> None:
    base_models = c.base_models
    topk_idx = c.topk_idx
    topk_names = c.topk_names
    # Im RETRAIN_META_ONLY-Pfad kann Phase 12 (wo rename_map gesetzt wird) übersprungen sein.
    # Dann für Darstellungszwecke auf leeres Mapping fallen.
    rename_map = getattr(c, "rename_map", {}) or {}
    FEAT_COLS = c.FEAT_COLS
    df_test = c.df_test
    df_final = c.df_final
    df_threshold = c.df_threshold
    rsi_w = c.rsi_w
    rs = c.RANDOM_STATE

    print("=" * 60)
    print("Phase 3b: Meta-Feature-Matrix aufbauen")
    print("=" * 60)

    def _log_array_health(dataset_label: str, arr: np.ndarray, feat_cols: list[str], top_n: int = 15) -> None:
        bad_mask = np.isnan(arr) | np.isinf(arr)
        bad_cells = int(bad_mask.sum())
        if bad_cells == 0:
            print(f"[Feature-Diagnose] {dataset_label}: ok (keine NaN/Inf) — shape={arr.shape}", flush=True)
            return
        bad_rows = int(bad_mask.any(axis=1).sum())
        print(
            f"[Feature-Diagnose] {dataset_label}: NaN/Inf-Zellen={bad_cells}, "
            f"Zeilen mit >=1 Problemwert={bad_rows}/{arr.shape[0]} — shape={arr.shape}",
            flush=True,
        )
        n_rows = max(1, arr.shape[0])
        bad_cols = []
        for i, col in enumerate(feat_cols):
            col_mask = bad_mask[:, i]
            n_bad = int(col_mask.sum())
            if n_bad == 0:
                continue
            col_arr = arr[:, i]
            n_nan = int(np.isnan(col_arr).sum())
            n_inf = int(np.isinf(col_arr).sum())
            bad_cols.append((col, n_bad, n_nan, n_inf))
        bad_cols.sort(key=lambda x: x[1], reverse=True)
        print("  Top-Spalten nach Problemwerten (count = NaN+Inf):", flush=True)
        for col, n_bad, n_nan, n_inf in bad_cols[:top_n]:
            pct_bad = 100.0 * float(n_bad) / float(n_rows)
            print(
                f"    - {col}: {n_bad} ({pct_bad:.1f}%) (NaN={n_nan}, Inf={n_inf})",
                flush=True,
            )
        if len(bad_cols) > top_n:
            print(f"    ... +{len(bad_cols) - top_n} weitere Spalten", flush=True)

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
        print(f" {time.time()-t0:.1f}s", flush=True)
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

    meta_feature_names = [m[0] + "_prob" for m in base_models] + [
        rename_map.get(n, n) for n in topk_names
    ]
    print(f"Meta-Feature-Namen: {meta_feature_names}")

    if c.USE_NEWS_SENTIMENT:
        df_test = merge_news_shard_from_best_params(df_test, c.best_params)
        df_final = merge_news_shard_from_best_params(df_final, c.best_params)
        df_threshold = merge_news_shard_from_best_params(df_threshold, c.best_params)

    _nan_sentinel = np.float32(getattr(c, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
    X_test_feat = df_test[FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
    y_test = df_test["target"].values.astype(np.int8)
    X_final_feat = df_final[FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
    y_final = df_final["target"].values.astype(np.int8)
    X_threshold_feat = df_threshold[FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
    y_threshold = df_threshold["target"].values.astype(np.int8)
    np.nan_to_num(X_test_feat, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)
    np.nan_to_num(X_final_feat, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)
    np.nan_to_num(X_threshold_feat, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)

    print("\n" + "=" * 60)
    print("Feature-Diagnose vor Meta-Training")
    print("=" * 60)
    _log_array_health("META Input-Features", X_test_feat, FEAT_COLS)
    _log_array_health("THRESHOLD Input-Features", X_threshold_feat, FEAT_COLS)
    _log_array_health("FINAL Input-Features", X_final_feat, FEAT_COLS)

    t_total = time.time()
    X_meta_test = build_meta_features(X_test_feat, "META (zeitlich nach Base+Purge)")
    X_meta_final = build_meta_features(X_final_feat, "FINAL")
    X_meta_threshold = build_meta_features(X_threshold_feat, "THRESHOLD")
    print(f"\nAlle Meta-Matrizen fertig in {time.time()-t_total:.0f}s")
    print(f"  EARLY_TRAIN: {X_meta_test.shape}")
    print(f"  THRESHOLD:   {X_meta_threshold.shape}")
    print(f"  FINAL:       {X_meta_final.shape}")

    print("\n" + "=" * 60)
    print("Phase 4: Meta-Learner Optuna")
    print("=" * 60)

    _OPT_MIN_PRECISION = c.OPT_MIN_PRECISION_META
    _apply_filters_cv = c._apply_filters_cv
    _OPT_MAX_CONSEC_FP = c._OPT_MAX_CONSEC_FP
    _meta_obj_mode = str(getattr(c, "META_OBJECTIVE_MODE", "tp_precision")).strip().lower()
    _hold_horizons = tuple(
        sorted({int(h) for h in (getattr(c, "META_SIGNAL_RETURN_HORIZONS", (1, 2, 3, 4, 5)) or ()) if int(h) > 0})
    )
    _meta_min_signals = int(getattr(c, "META_OBJECTIVE_MIN_SIGNALS_PER_FOLD", 1) or 1)
    if _meta_obj_mode not in {"tp_precision", "signal_mean_return"}:
        print(
            f"WARNUNG: Unbekanntes META_OBJECTIVE_MODE={_meta_obj_mode!r} -> fallback 'tp_precision'.",
            flush=True,
        )
        _meta_obj_mode = "tp_precision"
    if _meta_obj_mode == "signal_mean_return" and not _hold_horizons:
        print(
            "WARNUNG: META_SIGNAL_RETURN_HORIZONS leer -> fallback (1,2,3,4,5).",
            flush=True,
        )
        _hold_horizons = (1, 2, 3, 4, 5)
    print(
        f"Meta-Objective-Modus: {_meta_obj_mode}"
        + (
            f" | horizons={_hold_horizons} | min_signals_per_fold={_meta_min_signals}"
            if _meta_obj_mode == "signal_mean_return"
            else ""
        ),
        flush=True,
    )

    N_META_FOLDS = 3
    all_dates_test = np.sort(df_test["Date"].unique())
    n_meta_dates = len(all_dates_test)
    meta_min_train = int(n_meta_dates * 0.40)
    meta_fold_size = (n_meta_dates - meta_min_train) // N_META_FOLDS
    date_to_idx_test = {d: i for i, d in enumerate(all_dates_test)}
    df_test_idx = df_test["Date"].map(date_to_idx_test).values

    CONSECUTIVE_DAYS = c.CONSECUTIVE_DAYS
    SIGNAL_COOLDOWN_DAYS = c.SIGNAL_COOLDOWN_DAYS
    _blue_breakout_w = int(
        getattr(c, "breakout_lookback_window", 0)
        or getattr(c, "best_params", {}).get("breakout_lookback_window", 0)
        or c.SEED_PARAMS.get("breakout_lookback_window", 20)
    )
    _blue_col = f"blue_sky_breakout_{_blue_breakout_w}d"

    def _signal_forward_return_scores_cv(
        probs_arr,
        dates_arr,
        tickers_arr,
        threshold,
        consecutive_days,
        signal_cooldown_days,
        close_arr,
        rsi_window,
        signal_skip_near_peak,
        peak_lookback_days,
        peak_min_dist_from_high_pct,
        signal_max_rsi,
        signal_max_vol_stress_z,
        signal_min_blue_sky_volume_z,
        dyn_vvix_trigger,
        dyn_rsi_trigger,
        dyn_bb_pband_trigger,
        dyn_mult_1,
        dyn_mult_2,
        dyn_mult_3,
        hold_horizons,
        vol_stress_arr=None,
        blue_sky_breakout_arr=None,
        volume_zscore_arr=None,
        vvix_ratio_arr=None,
        rsi_arr=None,
        bb_pband_arr=None,
    ):
        """
        Wie _apply_filters_cv, aber liefert Rendite-Score je gefiltertem Signal:
        pro Signal Mittel über hold_horizons, dann über alle Signale mitteln.
        """
        df_v = pd.DataFrame(
            {
                "ticker": tickers_arr,
                "Date": dates_arr,
                "prob": probs_arr,
                "close": close_arr,
            }
        )
        if vol_stress_arr is not None:
            df_v["vol_stress"] = vol_stress_arr
        if blue_sky_breakout_arr is not None:
            df_v["blue_sky_breakout"] = blue_sky_breakout_arr
        if volume_zscore_arr is not None:
            df_v["volume_zscore"] = volume_zscore_arr
        if vvix_ratio_arr is not None:
            df_v["vvix_ratio"] = vvix_ratio_arr
        if rsi_arr is not None:
            df_v["rsi_dyn"] = rsi_arr
        if bb_pband_arr is not None:
            df_v["bb_pband_dyn"] = bb_pband_arr
        signal_scores: list[float] = []
        n_signals = 0
        n_raw_signals = 0
        for _, sub in df_v.groupby("ticker"):
            sub = sub.sort_values("Date").reset_index(drop=True)
            raw = _dynamic_threshold_mask_1d(
                sub["prob"].to_numpy(dtype=np.float64, copy=False),
                float(threshold),
                vvix_ratio=sub["vvix_ratio"].to_numpy(dtype=np.float64, copy=False) if "vvix_ratio" in sub.columns else None,
                rsi_arr=sub["rsi_dyn"].to_numpy(dtype=np.float64, copy=False) if "rsi_dyn" in sub.columns else None,
                bb_pband_arr=sub["bb_pband_dyn"].to_numpy(dtype=np.float64, copy=False) if "bb_pband_dyn" in sub.columns else None,
                vvix_trigger=dyn_vvix_trigger,
                rsi_trigger=dyn_rsi_trigger,
                bb_pband_trigger=dyn_bb_pband_trigger,
                mult1=dyn_mult_1,
                mult2=dyn_mult_2,
                mult3=dyn_mult_3,
            ).astype(np.int8)
            n_raw_signals += int(raw.sum())
            n = len(raw)
            if n == 0:
                continue
            consec = np.zeros(n, dtype=np.int8)
            for i in range(2, n):
                if raw[i - 2] + raw[i - 1] + raw[i] >= consecutive_days:
                    consec[i] = 1
            final = np.zeros(n, dtype=np.int8)
            last_sig = -999
            for i in range(n):
                if consec[i] == 1 and (i - last_sig) >= signal_cooldown_days:
                    final[i] = 1
                    last_sig = i
            close_sub = pd.to_numeric(sub["close"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            rsi_sub = c._rsi_from_close_1d(close_sub, rsi_window)
            mask_ok = c._peak_rsi_mask_1d(
                close_sub,
                rsi_sub,
                signal_skip_near_peak,
                peak_lookback_days,
                peak_min_dist_from_high_pct,
                signal_max_rsi,
            )
            for i in range(n):
                if final[i] == 1 and not bool(mask_ok[i]):
                    final[i] = 0
            if signal_max_vol_stress_z is not None and "vol_stress" in sub.columns:
                stress_ok = _vol_stress_mask_1d(
                    close_sub,
                    sub["vol_stress"].to_numpy(dtype=np.float64, copy=False),
                    signal_max_vol_stress_z,
                )
                for i in range(n):
                    if final[i] == 1 and not bool(stress_ok[i]):
                        final[i] = 0
            if (
                signal_min_blue_sky_volume_z is not None
                and "blue_sky_breakout" in sub.columns
                and "volume_zscore" in sub.columns
            ):
                blue_ok = _blue_sky_weak_volume_mask_1d(
                    sub["blue_sky_breakout"].to_numpy(dtype=np.float64, copy=False),
                    sub["volume_zscore"].to_numpy(dtype=np.float64, copy=False),
                    signal_min_blue_sky_volume_z,
                )
                for i in range(n):
                    if final[i] == 1 and not bool(blue_ok[i]):
                        final[i] = 0
            idxs = np.where(final == 1)[0]
            for i in idxs:
                p0 = float(close_sub[i])
                if not np.isfinite(p0) or p0 <= 0.0:
                    continue
                # Nur Signale mit vollständig verfügbarem Zukunftshorizont bewerten/zählen.
                if any(int(i + h) >= n for h in hold_horizons):
                    continue
                rlist: list[float] = []
                for h in hold_horizons:
                    j = int(i + h)
                    if j >= n:
                        continue
                    pj = float(close_sub[j])
                    if np.isfinite(pj) and pj > 0.0:
                        rlist.append(pj / p0 - 1.0)
                if rlist:
                    n_signals += 1
                    signal_scores.append(float(np.mean(rlist)))
        return signal_scores, n_signals, n_raw_signals

    def meta_objective(trial):
        signal_skip_near_peak = trial.suggest_categorical("signal_skip_near_peak", [True, False])
        peak_lookback_days = trial.suggest_int("peak_lookback_days", 10, 40)
        peak_min_dist_from_high_pct = trial.suggest_float("peak_min_dist_from_high_pct", 0.004, 0.035)
        dyn_rsi_trigger = trial.suggest_float("dyn_rsi_trigger", 70.0, 75.0)
        # RSI-Logik entkoppeln: dyn_rsi_trigger = weicher Bereich (70-75),
        # signal_max_rsi = harter Kill-Switch (>=80) und immer oberhalb dyn_rsi_trigger.
        signal_max_rsi = trial.suggest_float(
            "signal_max_rsi",
            max(80.0, float(dyn_rsi_trigger) + 1.0),
            90.0,
        )
        signal_max_vol_stress_z = trial.suggest_float("signal_max_vol_stress_z", 1.5, 3.5)
        meta_eval_threshold = trial.suggest_float("meta_eval_threshold", 0.05, 0.95)
        mult_final_threshold_1 = trial.suggest_float("mult_final_threshold_1", 1.0, 1.5)
        mult_final_threshold_2 = trial.suggest_float("mult_final_threshold_2", 1.0, 1.5)
        mult_final_threshold_3 = trial.suggest_float("mult_final_threshold_3", 1.0, 1.5)
        dyn_vvix_trigger = trial.suggest_float("dyn_vvix_trigger", 6.0, 10.0)
        dyn_bb_pband_trigger = trial.suggest_float("dyn_bb_pband_trigger", 0.98, 1.10)
        signal_min_blue_sky_volume_z = trial.suggest_float("signal_min_blue_sky_volume_z", 0.0, 1.5)

        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_int("min_child_weight", 10, 200),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            subsample=trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=20,
            seed=rs,
        )

        _dates_test = df_test["Date"].values
        _tickers_test = df_test["ticker"].values
        _close_test = df_test["close"].values
        _vol_stress_test = df_test["vol_stress"].values if "vol_stress" in df_test.columns else None
        _blue_test = df_test[_blue_col].values if _blue_col in df_test.columns else None
        _volume_z_test = df_test["volume_zscore"].values if "volume_zscore" in df_test.columns else None
        _vvix_ratio_test = df_test["mr_vvix_div_vix"].values if "mr_vvix_div_vix" in df_test.columns else None
        _rsi_col = f"rsi_{int(rsi_w)}d"
        _bb_col = f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}"
        _rsi_test = df_test[_rsi_col].values if _rsi_col in df_test.columns else None
        _bb_test = df_test[_bb_col].values if _bb_col in df_test.columns else None

        fold_scores = []
        trial_raw_signals = 0
        trial_final_signals = 0
        for fold_i in range(N_META_FOLDS):
            train_end = meta_min_train + fold_i * meta_fold_size
            val_end = meta_min_train + (fold_i + 1) * meta_fold_size
            if val_end > n_meta_dates:
                break

            tr_mask = df_test_idx < train_end
            val_mask = (df_test_idx >= train_end) & (df_test_idx < val_end)

            X_tr = X_meta_test[tr_mask]
            y_tr = y_test[tr_mask]
            X_val = X_meta_test[val_mask]
            y_val = y_test[val_mask]

            if X_tr.shape[0] < 20 or y_val.sum() < 2:
                continue

            rng_m = np.random.RandomState(rs)
            perm = rng_m.permutation(len(X_tr))
            n_fit = int(len(perm) * 0.85)

            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_tr[perm[:n_fit]],
                y_tr[perm[:n_fit]],
                eval_set=[(X_tr[perm[n_fit:]], y_tr[perm[n_fit:]])],
                verbose=False,
            )
            probs = clf.predict_proba(X_val)[:, 1]

            if _meta_obj_mode == "signal_mean_return":
                sig_scores, n_sig, n_raw_sig = _signal_forward_return_scores_cv(
                    probs,
                    _dates_test[val_mask],
                    _tickers_test[val_mask],
                    meta_eval_threshold,
                    CONSECUTIVE_DAYS,
                    SIGNAL_COOLDOWN_DAYS,
                    _close_test[val_mask],
                    rsi_w,
                    signal_skip_near_peak,
                    peak_lookback_days,
                    peak_min_dist_from_high_pct,
                    signal_max_rsi,
                    signal_max_vol_stress_z,
                    signal_min_blue_sky_volume_z,
                    dyn_vvix_trigger,
                    dyn_rsi_trigger,
                    dyn_bb_pband_trigger,
                    mult_final_threshold_1,
                    mult_final_threshold_2,
                    mult_final_threshold_3,
                    _hold_horizons,
                    vol_stress_arr=None if _vol_stress_test is None else _vol_stress_test[val_mask],
                    blue_sky_breakout_arr=None if _blue_test is None else _blue_test[val_mask],
                    volume_zscore_arr=None if _volume_z_test is None else _volume_z_test[val_mask],
                    vvix_ratio_arr=None if _vvix_ratio_test is None else _vvix_ratio_test[val_mask],
                    rsi_arr=None if _rsi_test is None else _rsi_test[val_mask],
                    bb_pband_arr=None if _bb_test is None else _bb_test[val_mask],
                )
                trial_raw_signals += int(n_raw_sig)
                trial_final_signals += int(n_sig)
                if n_sig == 0 or not sig_scores:
                    fs = -1.0
                else:
                    fs = float(np.mean(sig_scores))
                    if n_sig < _meta_min_signals:
                        fs -= 0.05 * float(_meta_min_signals - n_sig)
            else:
                n_tp, n_sig, max_cfp, _det = _apply_filters_cv(
                    probs,
                    _dates_test[val_mask],
                    _tickers_test[val_mask],
                    y_val,
                    meta_eval_threshold,
                    CONSECUTIVE_DAYS,
                    SIGNAL_COOLDOWN_DAYS,
                    close_arr=_close_test[val_mask],
                    rsi_window=rsi_w,
                    signal_skip_near_peak=signal_skip_near_peak,
                    peak_lookback_days=peak_lookback_days,
                    peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                    signal_max_rsi=signal_max_rsi,
                    vol_stress_arr=None if _vol_stress_test is None else _vol_stress_test[val_mask],
                    signal_max_vol_stress_z=signal_max_vol_stress_z,
                    blue_sky_breakout_arr=None if _blue_test is None else _blue_test[val_mask],
                    volume_zscore_arr=None if _volume_z_test is None else _volume_z_test[val_mask],
                    signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                    vvix_ratio_arr=None if _vvix_ratio_test is None else _vvix_ratio_test[val_mask],
                    rsi_arr=None if _rsi_test is None else _rsi_test[val_mask],
                    bb_pband_arr=None if _bb_test is None else _bb_test[val_mask],
                    dyn_vvix_trigger=dyn_vvix_trigger,
                    dyn_rsi_trigger=dyn_rsi_trigger,
                    dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                    dyn_mult_1=mult_final_threshold_1,
                    dyn_mult_2=mult_final_threshold_2,
                    dyn_mult_3=mult_final_threshold_3,
                    return_details=True,
                )
                trial_raw_signals += int(_det.get("n_raw_signals", 0))
                trial_final_signals += int(_det.get("n_final_signals", n_sig))
                if max_cfp > _OPT_MAX_CONSEC_FP:
                    fs = -2.0 - (max_cfp - _OPT_MAX_CONSEC_FP) * 0.1
                elif n_sig == 0:
                    p = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
                    pos_m = y_val == 1
                    neg_m = y_val == 0
                    if pos_m.any() and neg_m.any():
                        fs = float(np.mean(p[pos_m]) - np.mean(p[neg_m]))
                    elif pos_m.any():
                        fs = float(np.mean(p[pos_m]) - 0.5)
                    elif neg_m.any():
                        fs = float(0.5 - np.mean(p[neg_m]))
                    else:
                        fs = 0.0
                elif (n_tp / n_sig) >= _OPT_MIN_PRECISION:
                    fs = float(n_tp)
                else:
                    fs = (n_tp / n_sig) - 1.0
            fold_scores.append(fs)
            trial.report(fs, fold_i)
            trial.set_user_attr("n_raw_signals", int(trial_raw_signals))
            trial.set_user_attr("n_final_signals", int(trial_final_signals))
            trial.set_user_attr("n_filtered_out", int(max(0, trial_raw_signals - trial_final_signals)))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("n_raw_signals", int(trial_raw_signals))
        trial.set_user_attr("n_final_signals", int(trial_final_signals))
        trial.set_user_attr("n_filtered_out", int(max(0, trial_raw_signals - trial_final_signals)))
        return np.mean(fold_scores) if fold_scores else -1.0

    def _meta_trial_log_callback(study, frozen_trial):
        _raw = frozen_trial.user_attrs.get("n_raw_signals")
        _final = frozen_trial.user_attrs.get("n_final_signals")
        _filt = frozen_trial.user_attrs.get("n_filtered_out")
        _state = str(getattr(frozen_trial.state, "name", frozen_trial.state))
        _val = frozen_trial.value
        _val_s = "nan" if _val is None else f"{float(_val):.6f}"
        if _raw is None or _final is None or _filt is None:
            print(
                f"[Meta-Optuna Trial {frozen_trial.number:03d}] state={_state} value={_val_s} "
                "(Signalzähler nicht verfügbar)",
                flush=True,
            )
            return
        print(
            f"[Meta-Optuna Trial {frozen_trial.number:03d}] state={_state} value={_val_s} "
            f"signals_raw={int(_raw)} filtered_out={int(_filt)} signals_final={int(_final)}",
            flush=True,
        )

    meta_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=42)
    meta_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    meta_study = optuna.create_study(direction="maximize", sampler=meta_sampler, pruner=meta_pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    meta_study.optimize(
        meta_objective,
        n_trials=c.N_META_TRIALS,
        show_progress_bar=True,
        callbacks=[_meta_trial_log_callback],
    )
    print("Meta-Optuna — finale Bestwerte (alle Trial-Parameter):", flush=True)
    for _k in sorted(meta_study.best_params.keys()):
        print(f"  {_k} = {meta_study.best_params[_k]!r}", flush=True)

    _META_NONMODEL = {
        "signal_skip_near_peak",
        "peak_lookback_days",
        "peak_min_dist_from_high_pct",
        "signal_max_rsi",
        "signal_max_vol_stress_z",
        "mult_final_threshold_1",
        "mult_final_threshold_2",
        "mult_final_threshold_3",
        "dyn_vvix_trigger",
        "dyn_rsi_trigger",
        "dyn_bb_pband_trigger",
        "signal_min_blue_sky_volume_z",
        "meta_eval_threshold",
    }
    meta_best = {k: v for k, v in meta_study.best_params.items() if k not in _META_NONMODEL}
    meta_best.update(
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=20,
        seed=rs,
    )
    SIGNAL_SKIP_NEAR_PEAK = meta_study.best_params.get("signal_skip_near_peak", c.SIGNAL_SKIP_NEAR_PEAK)
    PEAK_LOOKBACK_DAYS = int(meta_study.best_params.get("peak_lookback_days", c.PEAK_LOOKBACK_DAYS))
    PEAK_MIN_DIST_FROM_HIGH_PCT = float(
        meta_study.best_params.get("peak_min_dist_from_high_pct", c.PEAK_MIN_DIST_FROM_HIGH_PCT)
    )
    _mv = meta_study.best_params.get("signal_max_rsi", c.SIGNAL_MAX_RSI)
    SIGNAL_MAX_RSI = float(_mv) if _mv is not None else None
    _msz = meta_study.best_params.get("signal_max_vol_stress_z", getattr(c, "SIGNAL_MAX_VOL_STRESS_Z", 2.0))
    SIGNAL_MAX_VOL_STRESS_Z = float(_msz) if _msz is not None else None
    MULT_FINAL_THRESHOLD_1 = float(meta_study.best_params.get("mult_final_threshold_1", getattr(c, "MULT_FINAL_THRESHOLD_1", 1.0)))
    MULT_FINAL_THRESHOLD_2 = float(meta_study.best_params.get("mult_final_threshold_2", getattr(c, "MULT_FINAL_THRESHOLD_2", 1.0)))
    MULT_FINAL_THRESHOLD_3 = float(meta_study.best_params.get("mult_final_threshold_3", getattr(c, "MULT_FINAL_THRESHOLD_3", 1.0)))
    DYN_VVIX_TRIGGER = float(meta_study.best_params.get("dyn_vvix_trigger", getattr(c, "DYN_VVIX_TRIGGER", 8.2)))
    DYN_RSI_TRIGGER = float(meta_study.best_params.get("dyn_rsi_trigger", getattr(c, "DYN_RSI_TRIGGER", 75.0)))
    DYN_BB_PBAND_TRIGGER = float(meta_study.best_params.get("dyn_bb_pband_trigger", getattr(c, "DYN_BB_PBAND_TRIGGER", 1.02)))
    SIGNAL_MIN_BLUE_SKY_VOLUME_Z = float(
        meta_study.best_params.get(
            "signal_min_blue_sky_volume_z",
            getattr(c, "SIGNAL_MIN_BLUE_SKY_VOLUME_Z", 0.5),
        )
    )
    _mrsi_s = f"{SIGNAL_MAX_RSI:.1f}" if SIGNAL_MAX_RSI is not None else "off"
    _mvs_s = f"{SIGNAL_MAX_VOL_STRESS_Z:.2f}" if SIGNAL_MAX_VOL_STRESS_Z is not None else "off"
    _mbs_s = f"{SIGNAL_MIN_BLUE_SKY_VOLUME_Z:.2f}"
    print(
        f"Meta anti-peak: skip={SIGNAL_SKIP_NEAR_PEAK}, lookback={PEAK_LOOKBACK_DAYS}d, "
        f"minDist={PEAK_MIN_DIST_FROM_HIGH_PCT:.4f}, maxRSI={_mrsi_s}, maxVolStressZ={_mvs_s}, "
        f"blueSkyPrevVolZ>={_mbs_s} (col={_blue_col})"
    )
    print(
        f"Dynamic threshold: mult=({MULT_FINAL_THRESHOLD_1:.3f},{MULT_FINAL_THRESHOLD_2:.3f},{MULT_FINAL_THRESHOLD_3:.3f}) "
        f"triggers(vvix>{DYN_VVIX_TRIGGER:.2f}, rsi>{DYN_RSI_TRIGGER:.1f}, bb>{DYN_BB_PBAND_TRIGGER:.3f})"
    )

    if _meta_obj_mode == "signal_mean_return":
        print(
            f"\nMeta-Learner best score={meta_study.best_value:.4f}  "
            f"(= mean signal return per fold, Signal-Score über horizons={_hold_horizons})"
        )
    else:
        print(
            f"\nMeta-Learner best score={meta_study.best_value:.4f}  "
            f"(= mean TP/fold bei Filter-Prec>={_OPT_MIN_PRECISION:.0%}, "
            f"max consec FP <= {_OPT_MAX_CONSEC_FP})"
        )
    _met = meta_study.best_params.get("meta_eval_threshold")
    if _met is not None:
        print(
            f"  (CV nutzt meta_eval_threshold={_met:.3f} — nicht Phase-1 BEST_THRESHOLD; "
            "Phase 5 setzt den produktiven Schwellenwert auf THRESHOLD.)"
        )

    rng_final_meta = np.random.RandomState(rs)
    perm_final = rng_final_meta.permutation(len(X_meta_test))
    n_fit_final = int(len(perm_final) * 0.9)
    meta_clf = xgb.XGBClassifier(**meta_best)
    meta_clf.fit(
        X_meta_test[perm_final[:n_fit_final]],
        y_test[perm_final[:n_fit_final]],
        eval_set=[(X_meta_test[perm_final[n_fit_final:]], y_test[perm_final[n_fit_final:]])],
        verbose=False,
    )
    print("Finales Meta-Modell trainiert.")

    print("\n" + "=" * 60)
    print("Meta-Learner SHAP: Welche Features sind dem Meta-Classifier wichtig?")
    print("=" * 60)

    meta_explainer = shap.TreeExplainer(meta_clf)
    meta_shap_vals = meta_explainer.shap_values(X_meta_test)
    meta_mean_shap = np.abs(meta_shap_vals).mean(axis=0)

    print("\nMeta-Feature Wichtigkeit (absteigend):")
    sorted_meta = sorted(zip(meta_feature_names, meta_mean_shap), key=lambda x: -x[1])
    for fname, imp in sorted_meta:
        bar = "\u2588" * max(1, int(imp / meta_mean_shap.max() * 25))
        print(f"  {fname:30s}  {imp:.4f}  {bar}")
    try:
        out_path = Path("data") / "meta_feature_shap_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _rows = [
            {
                "rank": int(i + 1),
                "feature": str(fname),
                "mean_abs_shap": float(imp),
            }
            for i, (fname, imp) in enumerate(sorted_meta)
        ]
        payload = {
            "phase": "meta_training_phase13",
            "feature_count": int(len(meta_feature_names)),
            "features": [str(x) for x in meta_feature_names],
            "shap_mean_abs_sorted": _rows,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Meta-SHAP-Export geschrieben: {out_path}", flush=True)
    except Exception as _e_meta_export:
        print(f"Warnung: Meta-SHAP-Export fehlgeschlagen ({_e_meta_export})", flush=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(meta_feature_names) * 0.5)))
    names_s = [p[0] for p in sorted_meta[::-1]]
    vals_s = [p[1] for p in sorted_meta[::-1]]
    ax.barh(names_s, vals_s, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Meta-Learner Feature Importance (SHAP)")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Phase 5: Produktiver Threshold direkt aus Meta-Optuna")
    print("=" * 60)

    print(f"Scoring THRESHOLD-Set ({len(y_threshold):,} Zeilen)...", flush=True)
    y_prob_threshold = meta_clf.predict_proba(X_meta_threshold)[:, 1]
    print(f"Scoring FINAL-Set ({len(y_final):,} Zeilen)...", flush=True)
    y_prob_final = meta_clf.predict_proba(X_meta_final)[:, 1]

    print(f"THRESHOLD-Set: Positive Rate = {y_threshold.mean():.1%}")
    best_threshold = float(meta_study.best_params.get("meta_eval_threshold", 0.5))
    f1_thresh = best_threshold  # Downstream-Kompatibilität (Plot-Linie).
    print(
        f"Gewählter produktiver Threshold (aus Meta-Optuna/CV): {best_threshold:.3f}",
        flush=True,
    )

    _pred_thr_raw = (y_prob_threshold >= best_threshold).astype(int)
    _raw_sig = int(_pred_thr_raw.sum())
    _raw_tp = int(((_pred_thr_raw == 1) & (y_threshold == 1)).sum())
    _raw_prec = (_raw_tp / _raw_sig) if _raw_sig > 0 else 0.0
    print(
        f"  THRESHOLD raw@{best_threshold:.3f}: signals={_raw_sig}, TP={_raw_tp}, precision={_raw_prec:.2%}",
        flush=True,
    )

    n_tp_f, n_sig_f, max_cfp_f = _apply_filters_cv(
        y_prob_threshold,
        df_threshold["Date"].values,
        df_threshold["ticker"].values,
        y_threshold,
        best_threshold,
        CONSECUTIVE_DAYS,
        SIGNAL_COOLDOWN_DAYS,
        close_arr=df_threshold["close"].values,
        rsi_window=rsi_w,
        signal_skip_near_peak=SIGNAL_SKIP_NEAR_PEAK,
        peak_lookback_days=PEAK_LOOKBACK_DAYS,
        peak_min_dist_from_high_pct=PEAK_MIN_DIST_FROM_HIGH_PCT,
        signal_max_rsi=SIGNAL_MAX_RSI,
        vol_stress_arr=df_threshold["vol_stress"].values if "vol_stress" in df_threshold.columns else None,
        signal_max_vol_stress_z=SIGNAL_MAX_VOL_STRESS_Z,
        blue_sky_breakout_arr=df_threshold[_blue_col].values if _blue_col in df_threshold.columns else None,
        volume_zscore_arr=df_threshold["volume_zscore"].values if "volume_zscore" in df_threshold.columns else None,
        signal_min_blue_sky_volume_z=(
            SIGNAL_MIN_BLUE_SKY_VOLUME_Z
        ),
        vvix_ratio_arr=df_threshold["mr_vvix_div_vix"].values if "mr_vvix_div_vix" in df_threshold.columns else None,
        rsi_arr=df_threshold[f"rsi_{int(rsi_w)}d"].values if f"rsi_{int(rsi_w)}d" in df_threshold.columns else None,
        bb_pband_arr=df_threshold[f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}"].values if f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}" in df_threshold.columns else None,
        dyn_vvix_trigger=DYN_VVIX_TRIGGER,
        dyn_rsi_trigger=DYN_RSI_TRIGGER,
        dyn_bb_pband_trigger=DYN_BB_PBAND_TRIGGER,
        dyn_mult_1=MULT_FINAL_THRESHOLD_1,
        dyn_mult_2=MULT_FINAL_THRESHOLD_2,
        dyn_mult_3=MULT_FINAL_THRESHOLD_3,
    )
    _f_prec = (n_tp_f / n_sig_f) if n_sig_f > 0 else 0.0
    print(
        f"  THRESHOLD filtered@{best_threshold:.3f}: signals={n_sig_f}, TP={n_tp_f}, "
        f"precision={_f_prec:.2%}, max_consec_fp={max_cfp_f}",
        flush=True,
    )

    df_threshold["prob"] = y_prob_threshold
    df_final["prob"] = y_prob_final
    df_test["prob"] = meta_clf.predict_proba(X_meta_test)[:, 1]
    print("\nPhase 5 complete.")

    c.SIGNAL_SKIP_NEAR_PEAK = SIGNAL_SKIP_NEAR_PEAK
    c.PEAK_LOOKBACK_DAYS = PEAK_LOOKBACK_DAYS
    c.PEAK_MIN_DIST_FROM_HIGH_PCT = PEAK_MIN_DIST_FROM_HIGH_PCT
    c.SIGNAL_MAX_RSI = SIGNAL_MAX_RSI
    c.SIGNAL_MAX_VOL_STRESS_Z = SIGNAL_MAX_VOL_STRESS_Z
    c.MULT_FINAL_THRESHOLD_1 = MULT_FINAL_THRESHOLD_1
    c.MULT_FINAL_THRESHOLD_2 = MULT_FINAL_THRESHOLD_2
    c.MULT_FINAL_THRESHOLD_3 = MULT_FINAL_THRESHOLD_3
    c.DYN_VVIX_TRIGGER = DYN_VVIX_TRIGGER
    c.DYN_RSI_TRIGGER = DYN_RSI_TRIGGER
    c.DYN_BB_PBAND_TRIGGER = DYN_BB_PBAND_TRIGGER
    c.SIGNAL_MIN_BLUE_SKY_VOLUME_Z = SIGNAL_MIN_BLUE_SKY_VOLUME_Z
    c.build_meta_features = build_meta_features
    c.meta_clf = meta_clf
    c.best_threshold = best_threshold
    c.f1_thresh = f1_thresh
    c.df_test = df_test
    c.df_threshold = df_threshold
    c.df_final = df_final

    c.save_scoring_artifacts()
    print(
        "\n[Artefakt] Automatisch gespeichert (models/scoring_artifacts.joblib). "
        "Zelle 18 nur n\u00f6tig, wenn du ohne diese Zelle erneut speichern willst.",
        flush=True,
    )
