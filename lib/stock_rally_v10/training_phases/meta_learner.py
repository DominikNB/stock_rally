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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from lib.stock_rally_v10.features import merge_news_shard_from_best_params
from lib.stock_rally_v10.optuna_train import (
    _FOLD_PENALTY_INSUFFICIENT_LABELED_DATA,
    _blue_sky_weak_volume_mask_1d,
    _dynamic_threshold_mask_1d,
    _vol_stress_mask_1d,
)


def _auto_scale_pos_weight(y: np.ndarray) -> float:
    """XGBoost-Klassengewicht aus Neg/Pos-Verhältnis (>=1.0)."""
    yy = np.asarray(y, dtype=np.int8)
    n_pos = int((yy == 1).sum())
    n_neg = int((yy == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(max(1.0, n_neg / n_pos))


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

    def _calibrate_meta_probs(raw_probs: np.ndarray, y_true: np.ndarray):
        method = str(getattr(c, "META_PROBA_CALIBRATION_METHOD", "none")).strip().lower()
        if method in {"", "none", "off", "false"}:
            return None, "none"
        p = np.asarray(raw_probs, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.int8)
        if p.size == 0 or y.size == 0 or p.size != y.size:
            print("[Meta-Proba-Cal] Skip: ungültige Kalibrierungsdaten.", flush=True)
            return None, "none"
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos < 5 or n_neg < 5:
            print(
                f"[Meta-Proba-Cal] Skip: zu wenige Klassenbeispiele (pos={n_pos}, neg={n_neg}).",
                flush=True,
            )
            return None, "none"
        p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
        if method in {"sigmoid", "platt", "platt_sigmoid"}:
            logits = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            lr.fit(logits, y)

            def _apply(v: np.ndarray) -> np.ndarray:
                vv = np.asarray(v, dtype=np.float64)
                vv = np.clip(vv, 1e-6, 1.0 - 1e-6)
                ll = np.log(vv / (1.0 - vv)).reshape(-1, 1)
                return lr.predict_proba(ll)[:, 1].astype(np.float64)

            print(
                f"[Meta-Proba-Cal] Aktiv: sigmoid/Platt (fit auf THRESHOLD, n={len(y):,}).",
                flush=True,
            )
            return {"method": "sigmoid", "model": lr}, _apply
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_clip, y.astype(np.float64))

            def _apply(v: np.ndarray) -> np.ndarray:
                vv = np.asarray(v, dtype=np.float64)
                vv = np.clip(vv, 1e-6, 1.0 - 1e-6)
                return np.asarray(iso.predict(vv), dtype=np.float64)

            print(
                f"[Meta-Proba-Cal] Aktiv: isotonic (fit auf THRESHOLD, n={len(y):,}).",
                flush=True,
            )
            return {"method": "isotonic", "model": iso}, _apply
        print(
            f"[Meta-Proba-Cal] Unbekannte Methode {method!r} -> ohne Kalibrierung.",
            flush=True,
        )
        return None, "none"

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
    _meta_min_signals_per_day = float(
        getattr(c, "META_OBJECTIVE_MIN_SIGNALS_PER_DAY_PER_FOLD", 1.0) or 1.0
    )
    _meta_winrate_tie = float(getattr(c, "META_OBJECTIVE_WINRATE_RETURN_TIEBREAKER", 0.1) or 0.0)
    if _meta_obj_mode not in {"tp_precision", "signal_mean_return", "signal_win_rate"}:
        print(
            f"WARNUNG: Unbekanntes META_OBJECTIVE_MODE={_meta_obj_mode!r} -> fallback 'tp_precision'.",
            flush=True,
        )
        _meta_obj_mode = "tp_precision"
    if _meta_obj_mode in {"signal_mean_return", "signal_win_rate"} and not _hold_horizons:
        print(
            "WARNUNG: META_SIGNAL_RETURN_HORIZONS leer -> fallback (1,2,3,4,5).",
            flush=True,
        )
        _hold_horizons = (1, 2, 3, 4, 5)
    print(
        f"Meta-Objective-Modus: {_meta_obj_mode}"
        + (
            f" | horizons={_hold_horizons} | min_signals_per_fold={_meta_min_signals} "
            f"| min_signals_per_day_per_fold={_meta_min_signals_per_day:.3f} "
            f"| winrate_return_tiebreaker={_meta_winrate_tie:.3f}"
            if _meta_obj_mode in {"signal_mean_return", "signal_win_rate"}
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
        open_arr,
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
                "open": open_arr,
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
        signal_days_any: set[pd.Timestamp] = set()
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
                # Coverage-Zähler: Tag gilt als "abgedeckt", sobald irgendein finales Signal existiert.
                signal_days_any.add(pd.Timestamp(sub["Date"].iloc[i]).normalize())
                # Einstiegskurs: Open des Folgetags (Tag 1 relativ zum Signaltag).
                entry_idx = int(i + 1)
                if entry_idx >= n:
                    continue
                entry_open = float(pd.to_numeric(sub["open"].iloc[entry_idx], errors="coerce"))
                if not np.isfinite(entry_open) or entry_open <= 0.0:
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
                        # h=1: close(Tag1)/open(Tag1), h=2: close(Tag2)/open(Tag1), ...
                        rlist.append(pj / entry_open - 1.0)
                if rlist:
                    n_signals += 1
                    signal_scores.append(float(np.mean(rlist)))
        return signal_scores, n_signals, n_raw_signals, len(signal_days_any)

    def _score_tp_precision_from_arrays(
        probs_arr,
        dates_arr,
        tickers_arr,
        y_arr,
        threshold,
        *,
        close_arr,
        vol_stress_arr,
        blue_arr,
        volume_z_arr,
        vvix_ratio_arr,
        rsi_arr,
        bb_arr,
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
        mult_final_threshold_1,
        mult_final_threshold_2,
        mult_final_threshold_3,
    ):
        n_tp, n_sig, max_cfp, det = _apply_filters_cv(
            probs_arr,
            dates_arr,
            tickers_arr,
            y_arr,
            threshold,
            CONSECUTIVE_DAYS,
            SIGNAL_COOLDOWN_DAYS,
            close_arr=close_arr,
            rsi_window=rsi_window,
            signal_skip_near_peak=signal_skip_near_peak,
            peak_lookback_days=peak_lookback_days,
            peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
            signal_max_rsi=signal_max_rsi,
            vol_stress_arr=vol_stress_arr,
            signal_max_vol_stress_z=signal_max_vol_stress_z,
            blue_sky_breakout_arr=blue_arr,
            volume_zscore_arr=volume_z_arr,
            signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
            vvix_ratio_arr=vvix_ratio_arr,
            rsi_arr=rsi_arr,
            bb_pband_arr=bb_arr,
            dyn_vvix_trigger=dyn_vvix_trigger,
            dyn_rsi_trigger=dyn_rsi_trigger,
            dyn_bb_pband_trigger=dyn_bb_pband_trigger,
            dyn_mult_1=mult_final_threshold_1,
            dyn_mult_2=mult_final_threshold_2,
            dyn_mult_3=mult_final_threshold_3,
            return_details=True,
        )
        # Kontinuierliche Variante des bisherigen Vier-Pfade-Scores.
        # Ziel: monoton steigend in Precision UND in n_tp ohne harten Sprung an
        # ``_OPT_MIN_PRECISION``. Optuna findet so früher tragfähige Regionen.
        if max_cfp > _OPT_MAX_CONSEC_FP:
            # Hart-Constraint bleibt — Drawdown durch lange FP-Serien ist nicht weichzukochen.
            fs = -2.0 - (max_cfp - _OPT_MAX_CONSEC_FP) * 0.1
        elif n_sig == 0:
            # Kein Signal: separierender Prob-Score als Gradient für den Sampler,
            # damit Trials mit besserer Trennschärfe schon vor dem ersten Signal
            # belohnt werden. Begrenzt auf [-1, 1] (kleiner als n_sig>0-Pfad).
            p = np.clip(np.asarray(probs_arr, dtype=np.float64), 1e-7, 1.0 - 1e-7)
            pos_m = np.asarray(y_arr) == 1
            neg_m = np.asarray(y_arr) == 0
            if pos_m.any() and neg_m.any():
                fs = float(np.mean(p[pos_m]) - np.mean(p[neg_m]))
            elif pos_m.any():
                fs = float(np.mean(p[pos_m]) - 0.5)
            elif neg_m.any():
                fs = float(0.5 - np.mean(p[neg_m]))
            else:
                fs = 0.0
        else:
            precision = float(n_tp) / float(n_sig)
            # Sigmoid-Gate, zentriert auf _OPT_MIN_PRECISION; Slope so gewählt, dass bei
            # ±0.05 das Gate bei ~0.27 bzw. ~0.73 liegt → glatter Übergang ohne Sprung.
            slope = 20.0
            gate = 1.0 / (1.0 + np.exp(-slope * (precision - _OPT_MIN_PRECISION)))
            # Über der Schwelle: Belohnung mit n_tp (skaliert), darunter ein glatter
            # negativer Score in [-1, ~0). Konvexe Mischung über das Gate.
            score_above = float(n_tp) * gate
            score_below = (precision - 1.0) * (1.0 - gate)
            fs = float(score_above + score_below)
        return (
            fs,
            int(det.get("n_raw_signals", 0)),
            int(det.get("n_final_signals", n_sig)),
            int(det.get("n_signal_days", 0)),
        )

    def _score_return_mode_from_arrays(
        probs_arr,
        dates_arr,
        tickers_arr,
        open_arr,
        close_arr,
        threshold,
        *,
        vol_stress_arr,
        blue_arr,
        volume_z_arr,
        vvix_ratio_arr,
        rsi_arr,
        bb_arr,
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
        mult_final_threshold_1,
        mult_final_threshold_2,
        mult_final_threshold_3,
    ):
        n_days = int(pd.Series(dates_arr).nunique())
        sig_scores, n_sig, n_raw_sig, n_sig_days = _signal_forward_return_scores_cv(
            probs_arr,
            dates_arr,
            tickers_arr,
            threshold,
            CONSECUTIVE_DAYS,
            SIGNAL_COOLDOWN_DAYS,
            open_arr,
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
            mult_final_threshold_1,
            mult_final_threshold_2,
            mult_final_threshold_3,
            _hold_horizons,
            vol_stress_arr=vol_stress_arr,
            blue_sky_breakout_arr=blue_arr,
            volume_zscore_arr=volume_z_arr,
            vvix_ratio_arr=vvix_ratio_arr,
            rsi_arr=rsi_arr,
            bb_pband_arr=bb_arr,
        )
        day_coverage = (float(n_sig_days) / float(n_days)) if n_days > 0 else 0.0
        if n_days <= 0:
            fs = -3.0
        else:
            density_gap = max(0.0, _meta_min_signals_per_day - day_coverage)
            if day_coverage < _meta_min_signals_per_day:
                fs = -float(density_gap)
            else:
                if n_sig == 0 or not sig_scores:
                    base_return_score = -1.0
                    win_rate = 0.0
                else:
                    base_return_score = float(np.mean(sig_scores))
                    win_rate = float(np.mean(np.asarray(sig_scores, dtype=np.float64) > 0.0))
                if _meta_obj_mode == "signal_win_rate":
                    fs = 1.0 + win_rate + (_meta_winrate_tie * base_return_score)
                else:
                    fs = 1.0 + base_return_score
                if n_sig < _meta_min_signals:
                    fs -= 0.05 * float(_meta_min_signals - n_sig)
        return float(fs), int(n_raw_sig), int(n_sig), float(day_coverage)

    def _pick_threshold_nested(
        seed_threshold,
        *,
        probs_cal,
        dates_cal,
        tickers_cal,
        y_cal,
        open_cal,
        close_cal,
        vol_stress_cal,
        blue_cal,
        volume_z_cal,
        vvix_ratio_cal,
        rsi_cal,
        bb_cal,
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
        mult_final_threshold_1,
        mult_final_threshold_2,
        mult_final_threshold_3,
    ):
        # Threshold wird im inneren Kalibrierteil gewählt, anschließend auf outer-val bewertet.
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
            if _meta_obj_mode in {"signal_mean_return", "signal_win_rate"}:
                fs, _, _, _ = _score_return_mode_from_arrays(
                    probs_cal,
                    dates_cal,
                    tickers_cal,
                    open_cal,
                    close_cal,
                    float(thr),
                    vol_stress_arr=vol_stress_cal,
                    blue_arr=blue_cal,
                    volume_z_arr=volume_z_cal,
                    vvix_ratio_arr=vvix_ratio_cal,
                    rsi_arr=rsi_cal,
                    bb_arr=bb_cal,
                    rsi_window=rsi_window,
                    signal_skip_near_peak=signal_skip_near_peak,
                    peak_lookback_days=peak_lookback_days,
                    peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                    signal_max_rsi=signal_max_rsi,
                    signal_max_vol_stress_z=signal_max_vol_stress_z,
                    signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                    dyn_vvix_trigger=dyn_vvix_trigger,
                    dyn_rsi_trigger=dyn_rsi_trigger,
                    dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                    mult_final_threshold_1=mult_final_threshold_1,
                    mult_final_threshold_2=mult_final_threshold_2,
                    mult_final_threshold_3=mult_final_threshold_3,
                )
            else:
                fs, _, _, _ = _score_tp_precision_from_arrays(
                    probs_cal,
                    dates_cal,
                    tickers_cal,
                    y_cal,
                    float(thr),
                    close_arr=close_cal,
                    vol_stress_arr=vol_stress_cal,
                    blue_arr=blue_cal,
                    volume_z_arr=volume_z_cal,
                    vvix_ratio_arr=vvix_ratio_cal,
                    rsi_arr=rsi_cal,
                    bb_arr=bb_cal,
                    rsi_window=rsi_window,
                    signal_skip_near_peak=signal_skip_near_peak,
                    peak_lookback_days=peak_lookback_days,
                    peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                    signal_max_rsi=signal_max_rsi,
                    signal_max_vol_stress_z=signal_max_vol_stress_z,
                    signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                    dyn_vvix_trigger=dyn_vvix_trigger,
                    dyn_rsi_trigger=dyn_rsi_trigger,
                    dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                    mult_final_threshold_1=mult_final_threshold_1,
                    mult_final_threshold_2=mult_final_threshold_2,
                    mult_final_threshold_3=mult_final_threshold_3,
                )
            if (fs > best_score) or (np.isclose(fs, best_score) and abs(float(thr) - float(seed_threshold)) < abs(best_thr - float(seed_threshold))):
                best_score = float(fs)
                best_thr = float(thr)
        return best_thr, float(best_score)

    def meta_objective(trial):
        use_meta_spw = bool(getattr(c, "META_USE_SCALE_POS_WEIGHT", True))
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
        _open_test = df_test["open"].values if "open" in df_test.columns else None
        _close_test = df_test["close"].values
        _vol_stress_test = df_test["vol_stress"].values if "vol_stress" in df_test.columns else None
        _blue_test = df_test[_blue_col].values if _blue_col in df_test.columns else None
        _volume_z_test = df_test["volume_zscore"].values if "volume_zscore" in df_test.columns else None
        _vvix_ratio_test = df_test["mr_vvix_div_vix"].values if "mr_vvix_div_vix" in df_test.columns else None
        _rsi_col = f"rsi_{int(rsi_w)}d"
        _bb_col = f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}"
        _rsi_test = df_test[_rsi_col].values if _rsi_col in df_test.columns else None
        _bb_test = df_test[_bb_col].values if _bb_col in df_test.columns else None

        fold_scores: list[float] = []
        trial_raw_signals = 0
        trial_final_signals = 0
        trial_sig_per_day_sum = 0.0
        trial_sig_per_day_n = 0
        trial_nested_thresholds: list[float] = []
        trial_spw_values: list[float] = []
        insufficient_meta_fold_tags: list[str] = []
        _date_norm_meta = pd.to_datetime(df_test["Date"], errors="coerce").dt.normalize().values

        def _register_insufficient_meta_wf_fold(reason: str) -> None:
            fold_scores.append(float(_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA))
            insufficient_meta_fold_tags.append(f"{fold_i}:{reason}")
            trial.set_user_attr("n_meta_insufficient_wf_folds", len(insufficient_meta_fold_tags))
            trial.set_user_attr(
                "meta_insufficient_wf_folds",
                ";".join(insufficient_meta_fold_tags[-20:])[:1000],
            )
            trial.report(float(_FOLD_PENALTY_INSUFFICIENT_LABELED_DATA), int(fold_i))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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

            if X_tr.shape[0] < 20:
                _register_insufficient_meta_wf_fold("x_tr_lt20")
                continue
            if y_val.sum() < 2:
                _register_insufficient_meta_wf_fold("y_val_pos_lt2")
                continue

            tr_idx = np.where(tr_mask)[0]
            if len(tr_idx) < 40:
                _register_insufficient_meta_wf_fold("train_idx_lt40")
                continue
            tr_dates = pd.to_datetime(df_test["Date"].iloc[tr_idx], errors="coerce").dt.normalize()
            tr_unique_dates = np.sort(tr_dates.dropna().unique())
            if len(tr_unique_dates) < 20:
                _register_insufficient_meta_wf_fold("train_unique_dates_lt20")
                continue
            split_pos = max(1, int(len(tr_unique_dates) * 0.80))
            split_pos = min(split_pos, len(tr_unique_dates) - 1)
            cal_start_date = tr_unique_dates[split_pos]
            inner_train_mask = tr_mask & (_date_norm_meta < cal_start_date)
            inner_cal_mask = tr_mask & (_date_norm_meta >= cal_start_date)
            if int(inner_train_mask.sum()) < 20 or int(inner_cal_mask.sum()) < 10:
                _register_insufficient_meta_wf_fold("inner_train_or_cal_too_shallow")
                continue

            X_inner_train = X_meta_test[inner_train_mask]
            y_inner_train = y_test[inner_train_mask]
            X_inner_cal = X_meta_test[inner_cal_mask]
            y_inner_cal = y_test[inner_cal_mask]
            if y_inner_cal.sum() < 2:
                _register_insufficient_meta_wf_fold("y_cal_pos_lt2")
                continue

            rng_m = np.random.RandomState(rs)
            perm = rng_m.permutation(len(X_inner_train))
            n_fit = int(len(perm) * 0.85)
            n_fit = max(1, min(n_fit, len(perm) - 1))
            fit_idx = perm[:n_fit]
            es_idx = perm[n_fit:]

            params_fold = dict(params)
            if use_meta_spw:
                params_fold["scale_pos_weight"] = _auto_scale_pos_weight(y_inner_train[fit_idx])
                trial_spw_values.append(float(params_fold["scale_pos_weight"]))

            clf = xgb.XGBClassifier(**params_fold)
            clf.fit(
                X_inner_train[fit_idx],
                y_inner_train[fit_idx],
                eval_set=[(X_inner_train[es_idx], y_inner_train[es_idx])],
                verbose=False,
            )
            probs_cal = clf.predict_proba(X_inner_cal)[:, 1]
            probs = clf.predict_proba(X_val)[:, 1]

            nested_thr, _nested_thr_score = _pick_threshold_nested(
                meta_eval_threshold,
                probs_cal=probs_cal,
                dates_cal=_dates_test[inner_cal_mask],
                tickers_cal=_tickers_test[inner_cal_mask],
                y_cal=y_inner_cal,
                open_cal=(_open_test[inner_cal_mask] if _open_test is not None else _close_test[inner_cal_mask]),
                close_cal=_close_test[inner_cal_mask],
                vol_stress_cal=(None if _vol_stress_test is None else _vol_stress_test[inner_cal_mask]),
                blue_cal=(None if _blue_test is None else _blue_test[inner_cal_mask]),
                volume_z_cal=(None if _volume_z_test is None else _volume_z_test[inner_cal_mask]),
                vvix_ratio_cal=(None if _vvix_ratio_test is None else _vvix_ratio_test[inner_cal_mask]),
                rsi_cal=(None if _rsi_test is None else _rsi_test[inner_cal_mask]),
                bb_cal=(None if _bb_test is None else _bb_test[inner_cal_mask]),
                rsi_window=rsi_w,
                signal_skip_near_peak=signal_skip_near_peak,
                peak_lookback_days=peak_lookback_days,
                peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                signal_max_rsi=signal_max_rsi,
                signal_max_vol_stress_z=signal_max_vol_stress_z,
                signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                dyn_vvix_trigger=dyn_vvix_trigger,
                dyn_rsi_trigger=dyn_rsi_trigger,
                dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                mult_final_threshold_1=mult_final_threshold_1,
                mult_final_threshold_2=mult_final_threshold_2,
                mult_final_threshold_3=mult_final_threshold_3,
            )
            trial_nested_thresholds.append(float(nested_thr))

            if _meta_obj_mode in {"signal_mean_return", "signal_win_rate"}:
                val_dates = _dates_test[val_mask]
                n_val_days = int(pd.Series(val_dates).nunique())
                sig_scores, n_sig, n_raw_sig, n_sig_days = _signal_forward_return_scores_cv(
                    probs,
                    val_dates,
                    _tickers_test[val_mask],
                    nested_thr,
                    CONSECUTIVE_DAYS,
                    SIGNAL_COOLDOWN_DAYS,
                    _open_test[val_mask] if _open_test is not None else _close_test[val_mask],
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
                # Tagesabdeckung: Anteil Validierungstage mit >= 1 gefiltertem Signal über alle Ticker.
                # Für die Coverage zählen finale Signale (auch ohne vollständigen Return-Horizont),
                # damit das Coverage-Ziel durch die Filterlogik erreichbar bleibt.
                day_coverage = (float(n_sig_days) / float(n_val_days)) if n_val_days > 0 else 0.0
                trial_sig_per_day_sum += float(day_coverage)
                trial_sig_per_day_n += 1
                # Stufenweise Optimierung mit Regime-Wechsel:
                # 1) Unterhalb Ziel-Signaldichte: stark negativer Dichte-Score dominiert.
                # 2) Ab Zielerfüllung: Umschalten auf Return-Optimierung.
                if n_val_days <= 0:
                    fs = -3.0
                else:
                    density_gap = max(0.0, _meta_min_signals_per_day - day_coverage)
                    if day_coverage < _meta_min_signals_per_day:
                        # Phase A (Coverage nicht erreicht): nur Coverage-Lücke optimieren.
                        # Wertebereich [-1, 0): je kleiner die Lücke, desto näher an 0.
                        fs = -float(density_gap)
                    else:
                        # Phase B (Coverage erreicht): auf Return umschalten.
                        # +1.0 Offset trennt die Regime klar:
                        # jeder Trial in Phase B schlägt jeden Trial in Phase A.
                        if n_sig == 0 or not sig_scores:
                            base_return_score = -1.0
                            win_rate = 0.0
                        else:
                            base_return_score = float(np.mean(sig_scores))
                            win_rate = float(np.mean(np.asarray(sig_scores, dtype=np.float64) > 0.0))
                        if _meta_obj_mode == "signal_win_rate":
                            # Kernziel: Anteil profitabler Signale maximieren.
                            # Optionaler Return-Anteil dient nur als feiner Tiebreaker.
                            fs = 1.0 + win_rate + (_meta_winrate_tie * base_return_score)
                        else:
                            fs = 1.0 + base_return_score
                        if n_sig < _meta_min_signals:
                            fs -= 0.05 * float(_meta_min_signals - n_sig)
            else:
                fs, n_raw_sig, n_sig, n_sig_days = _score_tp_precision_from_arrays(
                    probs,
                    _dates_test[val_mask],
                    _tickers_test[val_mask],
                    y_val,
                    nested_thr,
                    close_arr=_close_test[val_mask],
                    vol_stress_arr=None if _vol_stress_test is None else _vol_stress_test[val_mask],
                    blue_arr=None if _blue_test is None else _blue_test[val_mask],
                    volume_z_arr=None if _volume_z_test is None else _volume_z_test[val_mask],
                    vvix_ratio_arr=None if _vvix_ratio_test is None else _vvix_ratio_test[val_mask],
                    rsi_arr=None if _rsi_test is None else _rsi_test[val_mask],
                    bb_arr=None if _bb_test is None else _bb_test[val_mask],
                    rsi_window=rsi_w,
                    signal_skip_near_peak=signal_skip_near_peak,
                    peak_lookback_days=peak_lookback_days,
                    peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                    signal_max_rsi=signal_max_rsi,
                    signal_max_vol_stress_z=signal_max_vol_stress_z,
                    signal_min_blue_sky_volume_z=signal_min_blue_sky_volume_z,
                    dyn_vvix_trigger=dyn_vvix_trigger,
                    dyn_rsi_trigger=dyn_rsi_trigger,
                    dyn_bb_pband_trigger=dyn_bb_pband_trigger,
                    mult_final_threshold_1=mult_final_threshold_1,
                    mult_final_threshold_2=mult_final_threshold_2,
                    mult_final_threshold_3=mult_final_threshold_3,
                )
                trial_raw_signals += int(n_raw_sig)
                trial_final_signals += int(n_sig)
                # Tagesabdeckung auch im tp_precision-Pfad mitführen, damit
                # ``avg_signals_per_day`` im Log nicht mehr immer 0.000 ausweist.
                _val_dates_tp = _dates_test[val_mask]
                _n_val_days_tp = int(pd.Series(_val_dates_tp).nunique())
                if _n_val_days_tp > 0:
                    trial_sig_per_day_sum += float(n_sig_days) / float(_n_val_days_tp)
                    trial_sig_per_day_n += 1
            fold_scores.append(fs)
            trial.report(fs, fold_i)
            trial.set_user_attr("n_raw_signals", int(trial_raw_signals))
            trial.set_user_attr("n_final_signals", int(trial_final_signals))
            trial.set_user_attr("n_filtered_out", int(max(0, trial_raw_signals - trial_final_signals)))
            trial.set_user_attr(
                "avg_signals_per_day",
                float(trial_sig_per_day_sum / trial_sig_per_day_n) if trial_sig_per_day_n > 0 else 0.0,
            )
            if trial_nested_thresholds:
                trial.set_user_attr("nested_thr_mean", float(np.mean(trial_nested_thresholds)))
                trial.set_user_attr("nested_thr_min", float(np.min(trial_nested_thresholds)))
                trial.set_user_attr("nested_thr_max", float(np.max(trial_nested_thresholds)))
            if trial_spw_values:
                trial.set_user_attr("spw_mean", float(np.mean(trial_spw_values)))
                trial.set_user_attr("spw_min", float(np.min(trial_spw_values)))
                trial.set_user_attr("spw_max", float(np.max(trial_spw_values)))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("n_raw_signals", int(trial_raw_signals))
        trial.set_user_attr("n_final_signals", int(trial_final_signals))
        trial.set_user_attr("n_filtered_out", int(max(0, trial_raw_signals - trial_final_signals)))
        trial.set_user_attr(
            "avg_signals_per_day",
            float(trial_sig_per_day_sum / trial_sig_per_day_n) if trial_sig_per_day_n > 0 else 0.0,
        )
        if fold_scores:
            trial.set_user_attr("n_meta_wf_folds_scored", int(len(fold_scores)))
        if trial_nested_thresholds:
            trial.set_user_attr("nested_thr_mean", float(np.mean(trial_nested_thresholds)))
            trial.set_user_attr("nested_thr_min", float(np.min(trial_nested_thresholds)))
            trial.set_user_attr("nested_thr_max", float(np.max(trial_nested_thresholds)))
        if trial_spw_values:
            trial.set_user_attr("spw_mean", float(np.mean(trial_spw_values)))
            trial.set_user_attr("spw_min", float(np.min(trial_spw_values)))
            trial.set_user_attr("spw_max", float(np.max(trial_spw_values)))
        return np.mean(fold_scores) if fold_scores else -1.0

    def _meta_trial_log_callback(study, frozen_trial):
        _raw = frozen_trial.user_attrs.get("n_raw_signals")
        _final = frozen_trial.user_attrs.get("n_final_signals")
        _filt = frozen_trial.user_attrs.get("n_filtered_out")
        _spd = frozen_trial.user_attrs.get("avg_signals_per_day")
        _nth_m = frozen_trial.user_attrs.get("nested_thr_mean")
        _nth_lo = frozen_trial.user_attrs.get("nested_thr_min")
        _nth_hi = frozen_trial.user_attrs.get("nested_thr_max")
        _spw_m = frozen_trial.user_attrs.get("spw_mean")
        _spw_lo = frozen_trial.user_attrs.get("spw_min")
        _spw_hi = frozen_trial.user_attrs.get("spw_max")
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
            f"signals_raw={int(_raw)} filtered_out={int(_filt)} signals_final={int(_final)} "
            f"avg_signals_per_day={float(_spd) if _spd is not None else 0.0:.3f} "
            f"nested_thr_mean={float(_nth_m) if _nth_m is not None else float('nan'):.3f} "
            f"nested_thr_min={float(_nth_lo) if _nth_lo is not None else float('nan'):.3f} "
            f"nested_thr_max={float(_nth_hi) if _nth_hi is not None else float('nan'):.3f} "
            f"spw_mean={float(_spw_m) if _spw_m is not None else float('nan'):.3f} "
            f"spw_min={float(_spw_lo) if _spw_lo is not None else float('nan'):.3f} "
            f"spw_max={float(_spw_hi) if _spw_hi is not None else float('nan'):.3f}",
            flush=True,
        )

    meta_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=42)
    meta_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    meta_study = optuna.create_study(direction="maximize", sampler=meta_sampler, pruner=meta_pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _meta_seed = getattr(c, "meta_optuna_best_params", None)
    if isinstance(_meta_seed, dict) and _meta_seed:
        _allowed = {
            "signal_skip_near_peak",
            "peak_lookback_days",
            "peak_min_dist_from_high_pct",
            "dyn_rsi_trigger",
            "signal_max_rsi",
            "signal_max_vol_stress_z",
            "meta_eval_threshold",
            "mult_final_threshold_1",
            "mult_final_threshold_2",
            "mult_final_threshold_3",
            "dyn_vvix_trigger",
            "dyn_bb_pband_trigger",
            "signal_min_blue_sky_volume_z",
            "max_depth",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
        }
        _seed_trial = {k: _meta_seed[k] for k in _allowed if k in _meta_seed}
        if _seed_trial:
            meta_study.enqueue_trial(_seed_trial)
            print(
                f"Meta-Optuna: Seed-Trial aus gespeichertem meta_optuna_best_params enqueued "
                f"({len(_seed_trial)} Parameter).",
                flush=True,
            )
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
    if bool(getattr(c, "META_USE_SCALE_POS_WEIGHT", True)):
        meta_best["scale_pos_weight"] = _auto_scale_pos_weight(y_test)
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
    if bool(getattr(c, "META_USE_SCALE_POS_WEIGHT", True)):
        print(
            f"Meta class-weighting: META_USE_SCALE_POS_WEIGHT=True, "
            f"final scale_pos_weight={float(meta_best.get('scale_pos_weight', 1.0)):.4f}",
            flush=True,
        )

    if _meta_obj_mode == "signal_mean_return":
        print(
            f"\nMeta-Learner best score={meta_study.best_value:.4f}  "
            f"(= mean signal return per fold, Signal-Score über horizons={_hold_horizons})"
        )
    elif _meta_obj_mode == "signal_win_rate":
        print(
            f"\nMeta-Learner best score={meta_study.best_value:.4f}  "
            f"(= signal win-rate per fold + {_meta_winrate_tie:.3f}*mean_return, "
            f"horizons={_hold_horizons})"
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
            f"  (CV nutzt meta_eval_threshold={_met:.3f} als Seed; "
            "pro Fold wird nested auf inner-calibriertem Split optimiert, "
            "und Phase 5 wählt den produktiven Schwellenwert auf THRESHOLD.)"
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
    y_prob_threshold_raw = meta_clf.predict_proba(X_meta_threshold)[:, 1]
    print(f"Scoring FINAL-Set ({len(y_final):,} Zeilen)...", flush=True)
    y_prob_final_raw = meta_clf.predict_proba(X_meta_final)[:, 1]
    y_prob_test_raw = meta_clf.predict_proba(X_meta_test)[:, 1]

    _cal_obj, _cal_apply = _calibrate_meta_probs(y_prob_threshold_raw, y_threshold)
    if callable(_cal_apply):
        y_prob_threshold = _cal_apply(y_prob_threshold_raw)
        y_prob_final = _cal_apply(y_prob_final_raw)
        y_prob_test = _cal_apply(y_prob_test_raw)
    else:
        y_prob_threshold = y_prob_threshold_raw
        y_prob_final = y_prob_final_raw
        y_prob_test = y_prob_test_raw

    print(f"THRESHOLD-Set: Positive Rate = {y_threshold.mean():.1%}")
    # Produktiver Threshold: separat auf THRESHOLD-Set gewählt (nested-konform, ohne Val-Leakage).
    _seed_thr = float(meta_study.best_params.get("meta_eval_threshold", 0.5))
    best_threshold, _thr_fit_score = _pick_threshold_nested(
        _seed_thr,
        probs_cal=y_prob_threshold,
        dates_cal=df_threshold["Date"].values,
        tickers_cal=df_threshold["ticker"].values,
        y_cal=y_threshold,
        open_cal=(df_threshold["open"].values if "open" in df_threshold.columns else df_threshold["close"].values),
        close_cal=df_threshold["close"].values,
        vol_stress_cal=(df_threshold["vol_stress"].values if "vol_stress" in df_threshold.columns else None),
        blue_cal=(df_threshold[_blue_col].values if _blue_col in df_threshold.columns else None),
        volume_z_cal=(df_threshold["volume_zscore"].values if "volume_zscore" in df_threshold.columns else None),
        vvix_ratio_cal=(df_threshold["mr_vvix_div_vix"].values if "mr_vvix_div_vix" in df_threshold.columns else None),
        rsi_cal=(df_threshold[f"rsi_{int(rsi_w)}d"].values if f"rsi_{int(rsi_w)}d" in df_threshold.columns else None),
        bb_cal=(df_threshold[f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}"].values if f"bb_pband_{int(getattr(c, 'bb_w', c.SEED_PARAMS.get('bb_window', 20)))}" in df_threshold.columns else None),
        rsi_window=rsi_w,
        signal_skip_near_peak=SIGNAL_SKIP_NEAR_PEAK,
        peak_lookback_days=PEAK_LOOKBACK_DAYS,
        peak_min_dist_from_high_pct=PEAK_MIN_DIST_FROM_HIGH_PCT,
        signal_max_rsi=SIGNAL_MAX_RSI,
        signal_max_vol_stress_z=SIGNAL_MAX_VOL_STRESS_Z,
        signal_min_blue_sky_volume_z=SIGNAL_MIN_BLUE_SKY_VOLUME_Z,
        dyn_vvix_trigger=DYN_VVIX_TRIGGER,
        dyn_rsi_trigger=DYN_RSI_TRIGGER,
        dyn_bb_pband_trigger=DYN_BB_PBAND_TRIGGER,
        mult_final_threshold_1=MULT_FINAL_THRESHOLD_1,
        mult_final_threshold_2=MULT_FINAL_THRESHOLD_2,
        mult_final_threshold_3=MULT_FINAL_THRESHOLD_3,
    )
    f1_thresh = best_threshold  # Downstream-Kompatibilität (Plot-Linie).
    print(
        f"Gewählter produktiver Threshold (nested; auf THRESHOLD-Set gewählt, seed={_seed_thr:.3f}): {best_threshold:.3f}",
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
    df_test["prob"] = y_prob_test
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
    c.meta_proba_calibrator = _cal_obj
    c.best_threshold = best_threshold
    c.f1_thresh = f1_thresh
    c.meta_optuna_best_params = dict(meta_study.best_params)
    c.meta_optuna_best_value = float(meta_study.best_value)
    c.meta_optuna_best_user_attrs = dict(meta_study.best_trial.user_attrs)
    c.df_test = df_test
    c.df_threshold = df_threshold
    c.df_final = df_final

    c.save_scoring_artifacts()
    print(
        "\n[Artefakt] Automatisch gespeichert (models/scoring_artifacts.joblib). "
        "Zelle 18 nur n\u00f6tig, wenn du ohne diese Zelle erneut speichern willst.",
        flush=True,
    )
