if globals().get('SCORING_ONLY', False):
    print('[SCORING_ONLY] Training-Zelle übersprungen.')
else:
    # Cell 14 — 8c: Meta-Feature-Aufbau + Meta-Learner + Meta-SHAP + Threshold
    # Hinweis: Tracebacks stehen ggf. am Ende der Zell-Ausgabe (nach langem Optuna-Log) — nach unten scrollen.
    # Voraussetzung: Cell 13 (Phase 1–2) + Cell 13a (SHAP → topk_names/topk_idx) ausgeführt

    import time

    # ── Phase 3b: Meta-Feature-Matrix aufbauen ────────────────────────────────────
    print('=' * 60) 
    print('Phase 3b: Meta-Feature-Matrix aufbauen')
    print('=' * 60)

    def _predict_base_logged(model_tuple, X, dataset_label=''):
        """Wie _predict_base, aber mit Fortschritts-Logs."""
        name, model, mtype = model_tuple
        t0 = time.time()
        n = len(X)
        print(f'  [{name}] scoring {n:,} Zeilen ({dataset_label})...', end='', flush=True)
        if mtype == 'xgb':
            result = 1.0 / (1.0 + np.exp(-model.predict(xgb.DMatrix(X))))
        elif mtype == 'lgb':
            result = 1.0 / (1.0 + np.exp(-model.predict(X)))
        else:  # rf, et, lr
            result = model.predict_proba(X)[:, 1]
        print(f' {time.time()-t0:.1f}s', flush=True)
        return result

    def build_meta_features(X_feat, dataset_label=''):
        """Stapelt Base-Model-Predictions + Top-K-SHAP-Roh-Features zur Meta-Feature-Matrix."""
        if dataset_label:
            print(f'\n--- {dataset_label}: {len(X_feat):,} Samples ---')
        base_preds = np.column_stack([
            _predict_base_logged(m, X_feat, dataset_label) for m in base_models
        ])
        topk_feats = X_feat[:, topk_idx]
        result = np.hstack([base_preds, topk_feats]).astype(np.float32)
        if dataset_label:
            print(f'  Meta-Matrix Shape: {result.shape}')
        return result

    # Feature-Namen für das Meta-Modell (Basis-Predictions + Top-K-SHAP-Roh-Features)
    meta_feature_names = [m[0] + '_prob' for m in base_models] + \
                         [rename_map.get(n, n) for n in topk_names]
    print(f'Meta-Feature-Namen: {meta_feature_names}')

    # Rohdaten für alle Partitionen vorbereiten
    X_test_feat      = df_test[FEAT_COLS].values.astype(np.float32)
    y_test           = df_test['target'].values.astype(np.int8)
    X_final_feat     = df_final[FEAT_COLS].values.astype(np.float32)
    y_final          = df_final['target'].values.astype(np.int8)
    X_threshold_feat = df_threshold[FEAT_COLS].values.astype(np.float32)
    y_threshold      = df_threshold['target'].values.astype(np.int8)

    t_total = time.time()
    X_meta_test      = build_meta_features(X_test_feat,      'TRAIN_META')
    X_meta_final     = build_meta_features(X_final_feat,     'FINAL')
    X_meta_threshold = build_meta_features(X_threshold_feat, 'THRESHOLD')
    print(f'\nAlle Meta-Matrizen fertig in {time.time()-t_total:.0f}s')
    print(f'  TRAIN_META:  {X_meta_test.shape}')
    print(f'  THRESHOLD:   {X_meta_threshold.shape}')
    print(f'  FINAL:       {X_meta_final.shape}')

    # ── Phase 4: Meta-Learner Optuna ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Phase 4: Meta-Learner Optuna')
    print('=' * 60)

    _OPT_MIN_PRECISION = OPT_MIN_PRECISION_META

    N_META_FOLDS = 3
    all_dates_test = np.sort(df_test['Date'].unique())
    n_meta_dates   = len(all_dates_test)
    meta_min_train = int(n_meta_dates * 0.40)
    meta_fold_size = (n_meta_dates - meta_min_train) // N_META_FOLDS
    date_to_idx_test = {d: i for i, d in enumerate(all_dates_test)}
    df_test_idx  = df_test['Date'].map(date_to_idx_test).values

    def meta_objective(trial):
        """
        Compound objective — wie Phase 1 (Cell 9): Filter-Precision >= OPT_MIN_PRECISION_META pro Fold.
          - Hard gate: max consecutive FP run per ticker <= _OPT_MAX_CONSEC_FP
          - Soft gate: TP/Signals (gefiltert) >= Gate
          - Reward:    n_TP wenn Gate erfüllt; sonst weicher Penalty (TP/Signals) - 1
        Konsekutiv / Cooldown wie Phase 1; Anti-Peak/RSI wie Cell 9.
        WICHTIG: Nicht BEST_THRESHOLD (Phase-1, Base-Roh-p) — Meta-predict_proba hat andere Skala.
        """
        signal_skip_near_peak = trial.suggest_categorical('signal_skip_near_peak', [True, False])
        peak_lookback_days = trial.suggest_int('peak_lookback_days', 10, 40)
        peak_min_dist_from_high_pct = trial.suggest_float('peak_min_dist_from_high_pct', 0.004, 0.035)
        signal_max_rsi = trial.suggest_float('signal_max_rsi', 68.0, 88.0)
        meta_eval_threshold = trial.suggest_float('meta_eval_threshold', 0.05, 0.95)

        params = dict(
            max_depth        = trial.suggest_int('max_depth', 2, 6),
            min_child_weight = trial.suggest_int('min_child_weight', 10, 200),
            gamma            = trial.suggest_float('gamma', 0.0, 5.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            n_estimators     = trial.suggest_int('n_estimators', 50, 500),
            subsample        = trial.suggest_float('subsample', 0.5, 0.9),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1.0),
            tree_method           = 'hist',
            eval_metric           = 'aucpr',
            early_stopping_rounds = 20,
            seed                  = RANDOM_STATE,
        )

        _dates_test   = df_test['Date'].values
        _tickers_test = df_test['ticker'].values
        _close_test   = df_test['close'].values

        fold_scores = []
        for fold_i in range(N_META_FOLDS):
            train_end = meta_min_train + fold_i * meta_fold_size
            val_end   = meta_min_train + (fold_i + 1) * meta_fold_size
            if val_end > n_meta_dates:
                break

            tr_mask  = df_test_idx < train_end
            val_mask = (df_test_idx >= train_end) & (df_test_idx < val_end)

            X_tr  = X_meta_test[tr_mask]
            y_tr  = y_test[tr_mask]
            X_val = X_meta_test[val_mask]
            y_val = y_test[val_mask]

            if X_tr.shape[0] < 20 or y_val.sum() < 2:
                continue

            rng_m = np.random.RandomState(RANDOM_STATE)
            perm  = rng_m.permutation(len(X_tr))
            n_fit = int(len(perm) * 0.85)

            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_tr[perm[:n_fit]], y_tr[perm[:n_fit]],
                eval_set=[(X_tr[perm[n_fit:]], y_tr[perm[n_fit:]])],
                verbose=False,
            )
            probs = clf.predict_proba(X_val)[:, 1]

            n_tp, n_sig, max_cfp = _apply_filters_cv(
                probs,
                _dates_test[val_mask],
                _tickers_test[val_mask],
                y_val,
                meta_eval_threshold, CONSECUTIVE_DAYS, SIGNAL_COOLDOWN_DAYS,
                close_arr=_close_test[val_mask],
                rsi_window=rsi_w,
                signal_skip_near_peak=signal_skip_near_peak,
                peak_lookback_days=peak_lookback_days,
                peak_min_dist_from_high_pct=peak_min_dist_from_high_pct,
                signal_max_rsi=signal_max_rsi,
            )

            # ── Compound objective (same as Phase 1) ──────────────────────────
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
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_scores) if fold_scores else -1.0

    meta_sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=42)
    meta_pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    meta_study   = optuna.create_study(direction='maximize', sampler=meta_sampler, pruner=meta_pruner)
    # Nur tqdm-Fortschritt (eine Zeile); Optuna-INFO würde jeden Trial doppelt loggen
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    meta_study.optimize(meta_objective, n_trials=N_META_TRIALS, show_progress_bar=True)

    _META_NONMODEL = {
        'signal_skip_near_peak', 'peak_lookback_days',
        'peak_min_dist_from_high_pct', 'signal_max_rsi',
        'meta_eval_threshold',
    }
    meta_best = {k: v for k, v in meta_study.best_params.items() if k not in _META_NONMODEL}
    meta_best.update(
        tree_method='hist',
        eval_metric='aucpr',
        early_stopping_rounds=20,
        seed=RANDOM_STATE,
    )
    SIGNAL_SKIP_NEAR_PEAK = meta_study.best_params.get('signal_skip_near_peak', SIGNAL_SKIP_NEAR_PEAK)
    PEAK_LOOKBACK_DAYS = int(meta_study.best_params.get('peak_lookback_days', PEAK_LOOKBACK_DAYS))
    PEAK_MIN_DIST_FROM_HIGH_PCT = float(meta_study.best_params.get('peak_min_dist_from_high_pct', PEAK_MIN_DIST_FROM_HIGH_PCT))
    _mv = meta_study.best_params.get('signal_max_rsi', SIGNAL_MAX_RSI)
    SIGNAL_MAX_RSI = float(_mv) if _mv is not None else None
    _mrsi_s = f'{SIGNAL_MAX_RSI:.1f}' if SIGNAL_MAX_RSI is not None else 'off'
    print(f'Meta anti-peak: skip={SIGNAL_SKIP_NEAR_PEAK}, lookback={PEAK_LOOKBACK_DAYS}d, '
          f'minDist={PEAK_MIN_DIST_FROM_HIGH_PCT:.4f}, maxRSI={_mrsi_s}')

    print(f'\nMeta-Learner best score={meta_study.best_value:.4f}  '
          f'(= mean TP/fold bei Filter-Prec>={_OPT_MIN_PRECISION:.0%}, '
          f'max consec FP <= {_OPT_MAX_CONSEC_FP})')
    _met = meta_study.best_params.get('meta_eval_threshold')
    if _met is not None:
        print(f'  (CV nutzt meta_eval_threshold={_met:.3f} — nicht Phase-1 BEST_THRESHOLD; '
              'Phase 5 setzt den produktiven Schwellenwert auf THRESHOLD.)')

    # Finales Meta-Modell auf allen TRAIN_META-Daten trainieren
    rng_final_meta = np.random.RandomState(RANDOM_STATE)
    perm_final     = rng_final_meta.permutation(len(X_meta_test))
    n_fit_final    = int(len(perm_final) * 0.9)
    meta_clf = xgb.XGBClassifier(**meta_best)
    meta_clf.fit(
        X_meta_test[perm_final[:n_fit_final]],  y_test[perm_final[:n_fit_final]],
        eval_set=[(X_meta_test[perm_final[n_fit_final:]], y_test[perm_final[n_fit_final:]])],
        verbose=False,
    )
    print('Finales Meta-Modell trainiert.')

    # ── Meta-Learner SHAP ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Meta-Learner SHAP: Welche Features sind dem Meta-Classifier wichtig?')
    print('=' * 60)

    meta_explainer  = shap.TreeExplainer(meta_clf)
    meta_shap_vals  = meta_explainer.shap_values(X_meta_test)
    meta_mean_shap  = np.abs(meta_shap_vals).mean(axis=0)

    print('\nMeta-Feature Wichtigkeit (absteigend):')
    sorted_meta = sorted(zip(meta_feature_names, meta_mean_shap), key=lambda x: -x[1])
    for fname, imp in sorted_meta:
        bar = '█' * max(1, int(imp / meta_mean_shap.max() * 25))
        print(f'  {fname:30s}  {imp:.4f}  {bar}')

    fig, ax = plt.subplots(figsize=(10, max(4, len(meta_feature_names) * 0.5)))
    names_s = [p[0] for p in sorted_meta[::-1]]
    vals_s  = [p[1] for p in sorted_meta[::-1]]
    ax.barh(names_s, vals_s, color='steelblue')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Meta-Learner Feature Importance (SHAP)')
    plt.tight_layout()
    plt.show()

    # ── Phase 5: Threshold-Optimierung auf THRESHOLD-Set ──────────────────────
    print('\n' + '=' * 60)
    print('Phase 5: Threshold-Optimierung (saubere Partition)')
    print('=' * 60)

    print(f'Scoring THRESHOLD-Set ({len(y_threshold):,} Zeilen)...', flush=True)
    y_prob_threshold = meta_clf.predict_proba(X_meta_threshold)[:, 1]
    print(f'Scoring FINAL-Set ({len(y_final):,} Zeilen)...', flush=True)
    y_prob_final = meta_clf.predict_proba(X_meta_final)[:, 1]

    print(f'THRESHOLD-Set: Positive Rate = {y_threshold.mean():.1%}')

    def find_precision_threshold(y_true, y_prob, target_prec=None, min_signals=None):
        """Niedrigstes thr mit sklearn-Precision >= target_prec (roh: jede Zeile = Vorhersage)."""
        if target_prec is None:
            target_prec = OPT_MIN_PRECISION
        if min_signals is None:
            min_signals = int(globals().get("PHASE5_MIN_SIGNALS", 5))
        for thr in np.arange(0.01, 0.991, 0.005):
            preds = (y_prob >= thr).astype(int)
            if preds.sum() < min_signals:
                continue
            if precision_score(y_true, preds, zero_division=0) >= target_prec:
                return float(thr)
        return None

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_threshold, y_prob_threshold)
    f1_arr    = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10)
    f1_thresh = float(thr_arr[np.argmax(f1_arr[:-1])])
    prec_thresh = find_precision_threshold(y_threshold, y_prob_threshold)
    _min_s = int(globals().get("PHASE5_MIN_SIGNALS", 5))

    if prec_thresh is not None:
        best_threshold = prec_thresh
        _p_ok = precision_score(
            y_threshold, (y_prob_threshold >= best_threshold).astype(int), zero_division=0
        )
        print(
            f'Precision-Ziel OPT_MIN_PRECISION={OPT_MIN_PRECISION:.0%} erreicht: '
            f'threshold={best_threshold:.3f}, Roh-Precision={_p_ok:.2%} (THRESHOLD-Set) ✓'
        )
    else:
        best_prec, best_thresh_fb = 0.0, 0.50
        for thr in np.arange(0.01, 0.991, 0.005):
            preds = (y_prob_threshold >= thr).astype(int)
            if preds.sum() < _min_s:
                continue
            p = precision_score(y_threshold, preds, zero_division=0)
            if p > best_prec:
                best_prec, best_thresh_fb = p, float(thr)
        best_threshold = float(best_thresh_fb)
        print(
            f'OPT_MIN_PRECISION={OPT_MIN_PRECISION:.0%} mit ≥{_min_s} Roh-Signalen nicht erreichbar. '
            f'Fallback: threshold={best_threshold:.3f} (beste Roh-Precision={best_prec:.2%}). '
            f'OPT_MIN_PRECISION senken oder PHASE5_MIN_SIGNALS reduzieren.'
        )

    print(f'F1-optimaler Threshold:  {f1_thresh:.3f}')
    print(f'Gewählter Threshold:     {best_threshold:.3f}')

    df_threshold['prob'] = y_prob_threshold
    df_final['prob']     = y_prob_final
    df_test['prob']      = meta_clf.predict_proba(X_meta_test)[:, 1]
    print('\nPhase 5 complete.')
    save_scoring_artifacts()
    print(
        '\n[Artefakt] Automatisch gespeichert (models/scoring_artifacts.joblib). '
        'Zelle 18 nur nötig, wenn du ohne diese Zelle erneut speichern willst.',
        flush=True,
    )
