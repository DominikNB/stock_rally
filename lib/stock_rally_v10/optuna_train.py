"""stock_rally_v10 — Optuna / Base-XGB (Pipeline-Modul)."""
from __future__ import annotations

import time
import warnings

import numpy as np
import optuna
import pandas as pd
import ta
import xgboost as xgb
from sklearn.metrics import average_precision_score

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.features import merge_news_shard_into_df
from lib.stock_rally_v10.helpers import make_focal_objective
from lib.stock_rally_v10.target import rebuild_target_for_train


# Gates: cfg.OPT_MIN_PRECISION_BASE; Phase 5 nutzt cfg.OPT_MIN_PRECISION (= THRESHOLD-Ziel)
_OPT_MIN_PRECISION = cfg.OPT_MIN_PRECISION_BASE
# Maximum consecutive FP signals per ticker before a trial is penalised
_OPT_MAX_CONSEC_FP = 3


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

        sig_targets = sub.loc[final == 1, 'target'].values
        n_tp      += int(sig_targets.sum())
        n_signals += int(sig_targets.size)

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
        }
        return n_tp, n_signals, max_consec_fp, details
    return n_tp, n_signals, max_consec_fp


def optimize_xgb(df_train, n_trials=None, seed_params=cfg.SEED_PARAMS):
    """
    Hyperparameter optimisation via Optuna with Walk-Forward temporal CV.

    Jointly optimises (depending on cfg.OPT_MODEL_HYPERPARAMS):
      wenn cfg.opt_optimize_y_targets(): Rally-/Label-Parameter (return_window, …); sonst feste Band-Regel
      immer (je nach cfg): consecutive/cooldown, ggf. Feature-Fenster inkl. News
      cfg.OPT_MODEL_HYPERPARAMS=True:  auch XGBoost-Hyperparameter (Option A)
      cfg.OPT_MODEL_HYPERPARAMS=False: Modell-HPs aus cfg.SEED_PARAMS (Option B)

    Nicht optimiert hier (wird in späteren Phasen überschrieben):
      - Schwelle ``threshold`` für CV: fest aus ``seed_params`` (Phase 5 setzt ``best_threshold``).
      - Anti-Peak/RSI: fest aus ``seed_params`` (Phase 4 Meta-Optuna optimiert diese Werte).

    Objective (compound, precision-first):
      - Hard gate: per-ticker max consecutive FP run <= _OPT_MAX_CONSEC_FP
      - Soft gate: filter-Precision TP/Signals >= cfg.OPT_MIN_PRECISION_BASE (pro Fold)
      - Reward:    n_TP wenn Gate erfüllt; sonst weicher Penalty (TP/Signals) - 1

    Returns best_params dict.
    """
    if n_trials is None:
        n_trials = int(cfg.N_OPTUNA_TRIALS)
    else:
        n_trials = int(n_trials)
    wf = getattr(cfg, "OPTUNA_WF_SPLITS", None)
    wf = int(wf) if wf is not None else cfg.N_WF_SPLITS
    if wf < 1:
        wf = cfg.N_WF_SPLITS
    all_dates = np.sort(df_train['Date'].unique())
    n_dates   = len(all_dates)
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
    _nu_df = int(df_train["ticker"].nunique())
    print(
        f'Optuna Phase 1: TRAIN {len(df_train):,} Zeilen, {_nu_df} Ticker, '
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
    df_base = df_train.copy()
    df_base['_date_idx'] = df_base['Date'].map(date_to_idx)

    def objective(trial):
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
        # Schwelle + Anti-Peak: fest aus seed_params (Phase 5 / Meta überschreiben später)
        threshold = float(seed_params['threshold'])
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

        # Rebuild targets for this trial's label params
        df_trial = rebuild_target_for_train(
            df_base, lead_days, entry_days,
            return_window=return_window, rally_threshold=rally_threshold,
            min_rally_tail_days=min_rally_tail_days,
        )
        # Kritisch für RAM: Trial darf df_base/andere Trials nie in-place aufblähen.
        # Sonst akkumulieren news_*-Spalten über Trials und X_tr wird riesig.
        df_trial = df_trial.copy()

        # ── Model params ───────────────────────────────────────────────────
        # ── Feature-window params: always optimised (affect all base models equally) ──
        # News: Tag-Tripel + news_extra_* (Z-Score-Fenster, Accel, macro−sec) wenn cfg.USE_NEWS_SENTIMENT
        rsi_w = trial.suggest_categorical('rsi_window', cfg.RSI_WINDOWS)
        bb_w  = trial.suggest_categorical('bb_window',  cfg.BB_WINDOWS)
        sma_w = trial.suggest_categorical('sma_window', cfg.SMA_WINDOWS)
        btc_momentum_z_window = trial.suggest_categorical(
            'btc_momentum_z_window', cfg.BTC_MOMENTUM_Z_WINDOWS
        )
        market_breadth_z_window = trial.suggest_categorical(
            'market_breadth_z_window', cfg.MARKET_BREADTH_Z_WINDOWS
        )
        rel_momentum_window = trial.suggest_categorical(
            'rel_momentum_window', cfg.REL_MOMENTUM_WINDOWS
        )
        adr_window = trial.suggest_categorical('adr_window', cfg.ADR_WINDOWS)
        breakout_lookback_window = trial.suggest_categorical(
            'breakout_lookback_window', cfg.BREAKOUT_LOOKBACK_WINDOWS
        )
        vcp_window = trial.suggest_categorical('vcp_window', cfg.VCP_WINDOWS)
        btc_corr_window = trial.suggest_categorical('btc_corr_window', cfg.BTC_CORR_WINDOWS)
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
        else:
            news_mom_w = seed_params.get('news_mom_w', cfg.NEWS_MOM_WINDOWS[len(cfg.NEWS_MOM_WINDOWS) // 2])
            news_vol_ma = seed_params.get('news_vol_ma', cfg.NEWS_VOL_MA_WINDOWS[len(cfg.NEWS_VOL_MA_WINDOWS) // 2])
            news_tone_roll = seed_params.get('news_tone_roll', cfg.NEWS_TONE_ROLL_WINDOWS[0])
            news_extra_zscore_w = None
            news_extra_tone_accel = None
            news_extra_macro_sec_diff = None

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
        )
        if cfg.USE_NEWS_SENTIMENT and getattr(cfg, "_FEATURE_NEWS_SHARDS_ACTIVE", False):
            _tag = cfg.news_feat_tag(news_mom_w, news_vol_ma, news_tone_roll)
            _need_news = [c for c in feat_cols if str(c).startswith("news_")]
            df_trial = merge_news_shard_into_df(df_trial, _tag, wanted_news_cols=_need_news)
        focal_obj = make_focal_objective(focal_gamma, focal_alpha)

        fold_scores = []
        for fold_i in range(wf):
            train_end = min_train + fold_i * fold_size
            val_end   = min_train + (fold_i + 1) * fold_size
            if val_end > n_dates:
                break

            train_mask = df_trial['_date_idx'] < train_end
            val_mask   = (df_trial['_date_idx'] >= train_end) & \
                         (df_trial['_date_idx'] < val_end)

            X_tr  = df_trial.loc[train_mask, feat_cols].to_numpy(dtype=np.float32, copy=True)
            y_tr  = df_trial.loc[train_mask, 'target'].values.astype(np.int8)
            X_val = df_trial.loc[val_mask,   feat_cols].to_numpy(dtype=np.float32, copy=True)
            y_val = df_trial.loc[val_mask,   'target'].values.astype(np.int8)
            _nan_sentinel = np.float32(getattr(cfg, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))
            np.nan_to_num(X_tr, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)
            np.nan_to_num(X_val, nan=_nan_sentinel, posinf=_nan_sentinel, neginf=_nan_sentinel, copy=False)

            if X_tr.shape[0] < 50 or X_val.shape[0] < 10:
                continue
            if y_val.sum() < 2:
                continue

            # Inner 90/10 split for early stopping
            rng_es  = np.random.RandomState(cfg.RANDOM_STATE + fold_i)
            perm    = rng_es.permutation(len(X_tr))
            n_fit   = int(len(perm) * 0.9)
            fit_idx, es_idx = perm[:n_fit], perm[n_fit:]

            dtrain = xgb.DMatrix(X_tr[fit_idx], label=y_tr[fit_idx])
            des    = xgb.DMatrix(X_tr[es_idx],  label=y_tr[es_idx])
            dval   = xgb.DMatrix(X_val, label=y_val)

            # n_estimators = sklearn-Name; xgb.train nutzt num_boost_round (XGBoost 2.x warnt sonst)
            n_trees = int(params['n_estimators'])
            xgb_params = {k: v for k, v in params.items() if k != 'n_estimators'}
            xgb_params.update({'tree_method': 'hist', 'seed': cfg.RANDOM_STATE,
                 'disable_default_eval_metric': 1})
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

            raw_preds = bst.predict(dval)
            probs_val = 1.0 / (1.0 + np.exp(-raw_preds))

            dates_val   = df_trial.loc[val_mask, 'Date'].values
            tickers_val = df_trial.loc[val_mask, 'ticker'].values
            close_val = df_trial.loc[val_mask, 'close'].values
            vol_stress_val = (
                df_trial.loc[val_mask, 'vol_stress'].values
                if 'vol_stress' in df_trial.columns
                else None
            )
            blue_col = f"blue_sky_breakout_{int(breakout_lookback_window)}d"
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
            rsi_dyn_val = (
                df_trial.loc[val_mask, rsi_dyn_col].values
                if rsi_dyn_col in df_trial.columns
                else None
            )
            bb_dyn_col = f"bb_pband_{int(bb_w)}"
            bb_dyn_val = (
                df_trial.loc[val_mask, bb_dyn_col].values
                if bb_dyn_col in df_trial.columns
                else None
            )

            n_tp, n_sig, max_cfp = _apply_filters_cv(
                probs_val, dates_val, tickers_val, y_val,
                threshold, consecutive_days, signal_cooldown_days,
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

            # ── Compound objective ─────────────────────────────────────────
            if max_cfp > _OPT_MAX_CONSEC_FP:
                # Ticker producing consecutive FPs — hardest penalise
                fold_scores.append(-2.0 - (max_cfp - _OPT_MAX_CONSEC_FP) * 0.1)
            elif n_sig == 0:
                # Kein Signal nach Konsekutiv-/Cooldown-Filter: Proxy aus Roh-p
                # (mittlere Trennung pos/neg) — sonst konstanter Score, wenig Trial-Signal für den Sampler.
                p = np.clip(probs_val.astype(np.float64), 1e-7, 1.0 - 1e-7)
                pos_m = y_val == 1
                neg_m = y_val == 0
                if pos_m.any() and neg_m.any():
                    fold_scores.append(float(np.mean(p[pos_m]) - np.mean(p[neg_m])))
                elif pos_m.any():
                    fold_scores.append(float(np.mean(p[pos_m]) - 0.5))
                elif neg_m.any():
                    fold_scores.append(float(0.5 - np.mean(p[neg_m])))
                else:
                    fold_scores.append(0.0)
            elif (n_tp / n_sig) >= _OPT_MIN_PRECISION:
                fold_scores.append(float(n_tp))  # Precision-Gate: mehr Treffer bevorzugen
            else:
                fold_scores.append((n_tp / n_sig) - 1.0)  # unter Precision-Floor

            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_scores) if fold_scores else -1.0

    sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
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
    if not cfg.opt_optimize_y_targets():
        for _k in ('return_window', 'rally_threshold', 'lead_days', 'entry_days', 'min_rally_tail_days'):
            _seed_enq.pop(_k, None)
    _cat_choices = {
        "rsi_window": list(cfg.RSI_WINDOWS),
        "bb_window": list(cfg.BB_WINDOWS),
        "sma_window": list(cfg.SMA_WINDOWS),
        "btc_momentum_z_window": list(cfg.BTC_MOMENTUM_Z_WINDOWS),
        "market_breadth_z_window": list(cfg.MARKET_BREADTH_Z_WINDOWS),
        "rel_momentum_window": list(cfg.REL_MOMENTUM_WINDOWS),
        "adr_window": list(cfg.ADR_WINDOWS),
        "breakout_lookback_window": list(cfg.BREAKOUT_LOOKBACK_WINDOWS),
        "vcp_window": list(cfg.VCP_WINDOWS),
        "btc_corr_window": list(cfg.BTC_CORR_WINDOWS),
        "news_mom_w": list(cfg.NEWS_MOM_WINDOWS),
        "news_vol_ma": list(cfg.NEWS_VOL_MA_WINDOWS),
        "news_tone_roll": list(cfg.NEWS_TONE_ROLL_WINDOWS),
        "news_extra_zscore_w": list(cfg.NEWS_EXTRA_ZSCORE_WINDOWS),
        "news_extra_tone_accel": list(cfg.NEWS_EXTRA_TONE_ACCEL_OPTIONS),
        "news_extra_macro_sec_diff": list(cfg.NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS),
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
    study.enqueue_trial(_seed_enq)
    # Nur tqdm-Fortschritt (eine Zeile); Optuna-INFO würde jeden Trial doppelt loggen
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _spb = getattr(cfg, "OPTUNA_SHOW_PROGRESS_BAR", None)
    if _spb is None:
        _spb = True
    study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(_spb))

    best = study.best_params
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
          f'(= mean TP/fold bei Filter-Prec>={_OPT_MIN_PRECISION:.0%}, '
          f'max consec FP <= {_OPT_MAX_CONSEC_FP})  [{mode}]')
    print("Optuna Phase 1 — finale Bestwerte (alle Parameter):", flush=True)
    for _k in sorted(best.keys()):
        print(f"  {_k} = {best[_k]!r}", flush=True)
    cfg.base_optuna_best_params = dict(best)
    cfg.base_optuna_best_value = float(study.best_value)
    return best
