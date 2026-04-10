if globals().get('SCORING_ONLY', False):
    print('[SCORING_ONLY] Training-Zelle übersprungen.')
else:
    globals()['_SCORING_ARTIFACT_SAVED_THIS_SESSION'] = False  # bis Cell 14 (Phase 5) kein aktuelles Bundle
    # Cell 13 — 8b: Optuna + Base-Modelle (Phases 1–2); SHAP → eigene Zelle 13a
    # Phase 1: Optuna base; Phase 2: Base models. Anschließend Cell 13a (SHAP), dann Cell 14 (Meta …)
    # Phase 4: Meta-Learner Optuna, Phase 5: Calibration + Threshold

    try:
        import mlflow
        mlflow.autolog(disable=True)
    except Exception:
        pass

    # ── Phase 1: Optuna HP optimisation on TRAIN ────────────────────────────────
    print('=' * 60)
    print('Phase 1: Optuna base-model HP optimisation')
    print('=' * 60)
    best_params = optimize_xgb(df_train)

    # ── Extract & apply all optimised label/filter params ────────────────────────
    if globals().get('OPT_OPTIMIZE_Y_TARGETS', True):
        RETURN_WINDOW        = best_params['return_window']
        RALLY_THRESHOLD      = best_params['rally_threshold']
        LEAD_DAYS            = best_params['lead_days']
        ENTRY_DAYS           = best_params['entry_days']
        MIN_RALLY_TAIL_DAYS  = best_params.get('min_rally_tail_days', SEED_PARAMS['min_rally_tail_days'])
    else:
        RETURN_WINDOW        = FIXED_Y_WINDOW_MAX
        RALLY_THRESHOLD      = FIXED_Y_RALLY_THRESHOLD
        LEAD_DAYS            = 3
        ENTRY_DAYS           = 2
        MIN_RALLY_TAIL_DAYS  = 5
    CONSECUTIVE_DAYS     = best_params['consecutive_days']
    SIGNAL_COOLDOWN_DAYS = best_params['signal_cooldown_days']
    # Schwelle: Phase 1 optimiert sie nicht mehr; Seed bis Phase 5 (find_precision_threshold)
    BEST_THRESHOLD       = best_params.get('threshold', SEED_PARAMS['threshold'])
    SIGNAL_SKIP_NEAR_PEAK = SEED_PARAMS['signal_skip_near_peak']
    PEAK_LOOKBACK_DAYS = int(SEED_PARAMS['peak_lookback_days'])
    PEAK_MIN_DIST_FROM_HIGH_PCT = float(SEED_PARAMS['peak_min_dist_from_high_pct'])
    _v_rsi_se = SEED_PARAMS.get('signal_max_rsi', SIGNAL_MAX_RSI)
    SIGNAL_MAX_RSI = float(_v_rsi_se) if _v_rsi_se is not None else None

    if globals().get('OPT_OPTIMIZE_Y_TARGETS', True):
        print(f'\nOptimised rally def:      return_window={RETURN_WINDOW}, '
              f'rally_threshold={RALLY_THRESHOLD:.2%}')
        print(f'Optimised target params:  lead_days={LEAD_DAYS}, entry_days={ENTRY_DAYS}, '
              f'min_rally_tail_days={MIN_RALLY_TAIL_DAYS}')
    else:
        print(f'\nFixed rally band (no y-opt): windows {FIXED_Y_WINDOW_MIN}-{FIXED_Y_WINDOW_MAX}d, '
              f'threshold={RALLY_THRESHOLD:.2%}; target rule by segment vs {FIXED_Y_SEGMENT_SPLIT}d')
        print(f'Fixed target (nur Info):  lead_days={LEAD_DAYS}, entry_days={ENTRY_DAYS} '
              f'(Rebuild nutzt _create_target_one_ticker_fixed_bands)')
    print(f'Optimised filter params:  consecutive_days={CONSECUTIVE_DAYS}, '
          f'signal_cooldown_days={SIGNAL_COOLDOWN_DAYS}')
    print(f'Seed threshold (bis Phase 5): {BEST_THRESHOLD:.3f}')
    _rsi_s = f'{SIGNAL_MAX_RSI:.1f}' if SIGNAL_MAX_RSI is not None else 'off'
    print(f'Anti-peak (SEED_PARAMS bis Meta): skip={SIGNAL_SKIP_NEAR_PEAK}, lookback={PEAK_LOOKBACK_DAYS}d, '
          f'minDist={PEAK_MIN_DIST_FROM_HIGH_PCT:.4f}, maxRSI={_rsi_s}')

    # Rebuild targets for ALL splits so base models, meta-learner, calibration
    # and threshold evaluation all use the same optimised label definition.
    _rebuild_kw = dict(
        return_window=RETURN_WINDOW,
        rally_threshold=RALLY_THRESHOLD,
        min_rally_tail_days=MIN_RALLY_TAIL_DAYS,
    )
    df_train     = rebuild_target_for_train(df_train,     LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_test      = rebuild_target_for_train(df_test,      LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_threshold = rebuild_target_for_train(df_threshold, LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    df_final     = rebuild_target_for_train(df_final,     LEAD_DAYS, ENTRY_DAYS, **_rebuild_kw)
    for _name, _df in [('df_train', df_train), ('df_test', df_test),
                       ('df_threshold', df_threshold), ('df_final', df_final)]:
        print(f'{_name:15s} target rebuilt  (positive rate: {_df["target"].mean():.1%})')

    rsi_w = best_params['rsi_window']
    bb_w  = best_params['bb_window']
    sma_w = best_params['sma_window']
    if USE_NEWS_SENTIMENT:
        FEAT_COLS = build_feature_cols(
            rsi_w, bb_w, sma_w,
            best_params['news_mom_w'], best_params['news_vol_ma'], best_params['news_tone_roll'],
            best_params.get('news_extra_zscore_w', SEED_PARAMS.get('news_extra_zscore_w')),
            best_params.get('news_extra_tone_accel', SEED_PARAMS.get('news_extra_tone_accel')),
            best_params.get('news_extra_macro_sec_diff', SEED_PARAMS.get('news_extra_macro_sec_diff')),
        )
    else:
        FEAT_COLS = build_feature_cols(rsi_w, bb_w, sma_w)
    print(f'\nUsing features: RSI={rsi_w}, BB={bb_w}, SMA={sma_w}  ({len(FEAT_COLS)} features)')

    focal_gamma = best_params['focal_gamma']
    focal_alpha = best_params['focal_alpha']
    focal_obj   = make_focal_objective(focal_gamma, focal_alpha)
    focal_obj_lgb = make_focal_objective_lgb(focal_gamma, focal_alpha)

    xgb_base_params = {
        k: v for k, v in best_params.items()
        if k not in (
            'rsi_window', 'bb_window', 'sma_window',
            'news_mom_w', 'news_vol_ma', 'news_tone_roll',
            'news_extra_zscore_w', 'news_extra_tone_accel', 'news_extra_macro_sec_diff',
            'focal_gamma', 'focal_alpha',
            'return_window', 'rally_threshold', 'lead_days', 'entry_days', 'min_rally_tail_days',
            'consecutive_days', 'signal_cooldown_days', 'threshold',
            'signal_skip_near_peak', 'peak_lookback_days',
            'peak_min_dist_from_high_pct', 'signal_max_rsi',
        )
    }
    xgb_base_params['tree_method'] = 'hist'

    lgb_params = dict(
        max_depth        = best_params['max_depth'],
        num_leaves       = min(best_params.get('max_leaves', 127), 255),
        learning_rate    = best_params['learning_rate'],
        subsample        = best_params['subsample'],
        colsample_bytree = best_params['colsample_bytree'],
        min_child_weight = best_params['min_child_weight'],
        reg_alpha        = best_params['reg_alpha'],
        reg_lambda       = best_params['reg_lambda'],
        n_estimators     = best_params['n_estimators'],
        verbose          = -1,
    )

    X_train_all = df_train[FEAT_COLS].values.astype(np.float32)
    y_train_all = df_train['target'].values.astype(np.int8)

    # ── Phase 2: Train 6 base models ────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Phase 2: Training 10 base models')
    print('=' * 60)
    base_models = []  # list of (name, model_or_booster, model_type)

    def _bootstrap_split(seed_i, X, y):
        """Bootstrap sample + Out-of-Bag early-stopping set.
        fit  = bootstrap sample (with replacement, same size as X)
        es   = OOB indices — never seen during training, truly independent.
        Fallback to last 10% of boot_idx if OOB set is too small (rare).
        """
        rng      = np.random.RandomState(seed_i)
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        oob_idx  = np.setdiff1d(np.arange(len(X)), boot_idx)
        if len(oob_idx) < 10:
            oob_idx = boot_idx[int(len(boot_idx) * 0.9):]
        return X[boot_idx], y[boot_idx], X[oob_idx], y[oob_idx]

    # Models 1–4: XGBoost (4 bootstrap variants)
    for m_idx, seed_i in enumerate([RANDOM_STATE, RANDOM_STATE + 1,
                                     RANDOM_STATE + 2, RANDOM_STATE + 3]):
        name = f'XGB-{m_idx+1}'
        print(f'  Training {name}...')
        X_fit, y_fit, X_es, y_es = _bootstrap_split(seed_i, X_train_all, y_train_all)
        dtrain_m = xgb.DMatrix(X_fit, label=y_fit)
        des_m    = xgb.DMatrix(X_es,  label=y_es)
        p = {**xgb_base_params, 'seed': seed_i, 'disable_default_eval_metric': 1}
        bst = xgb.train(
            p, dtrain_m,
            num_boost_round=xgb_base_params['n_estimators'],
            obj=focal_obj,
            evals=[(des_m, 'es')],
            custom_metric=lambda pr, d: ('logloss',
                float(np.mean(-d.get_label() * np.log(np.clip(1/(1+np.exp(-pr)), 1e-7, 1-1e-7))
                          - (1-d.get_label()) * np.log(np.clip(1/(1+np.exp(pr)), 1e-7, 1-1e-7))))),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        base_models.append((name, bst, 'xgb'))
        print(f'  {name} done — best iteration: {bst.best_iteration}')

    # Models 5–7: LightGBM (3 bootstrap variants)
    for m_idx, seed_i in enumerate([RANDOM_STATE + 10, RANDOM_STATE + 11,
                                     RANDOM_STATE + 12]):
        name = f'LGB-{m_idx+1}'
        print(f'  Training {name}...')
        X_fit, y_fit, X_es, y_es = _bootstrap_split(seed_i, X_train_all, y_train_all)
        dtrain_lgb = lgb.Dataset(X_fit, label=y_fit)
        des_lgb    = lgb.Dataset(X_es,  label=y_es, reference=dtrain_lgb)
        callbacks  = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(-1)]
        p_lgb = {**lgb_params, 'seed': seed_i, 'verbosity': -1}
        p_lgb['objective'] = focal_obj_lgb
        p_lgb['metric'] = 'binary_logloss'
        n_est = p_lgb.pop('n_estimators', 300)
        bst_lgb = lgb.train(
            p_lgb, dtrain_lgb,
            num_boost_round=n_est,
            valid_sets=[des_lgb],
            callbacks=callbacks,
        )
        base_models.append((name, bst_lgb, 'lgb'))
        print(f'  {name} done — best iteration: {bst_lgb.best_iteration}')

    # Model 8: Random Forest
    print('  Training RF...')
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=best_params['max_depth'],
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_all, y_train_all)
    base_models.append(('RF', rf, 'rf'))
    print('  RF done.')

    # Model 9: ExtraTreesClassifier — extreme randomisation, orthogonal errors to RF/XGB
    print('  Training ET...')
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=best_params['max_depth'],
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE + 20,
        n_jobs=-1,
    )
    et.fit(X_train_all, y_train_all)
    base_models.append(('ET', et, 'et'))
    print('  ET done.')

    # Model 10: Logistic Regression — linear perspective, low variance, balanced classes
    print('  Training LR...')
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=0.1,                   # strong L2 regularisation — avoids overfitting on noisy labels
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
            n_jobs=-1,
        )),
    ])
    lr_pipe.fit(X_train_all, y_train_all)
    base_models.append(('LR', lr_pipe, 'lr'))
    print('  LR done.')

    print(f'\nBase models ready: {[m[0] for m in base_models]}')

    # ── Predict helper ───────────────────────────────────────────────────────────
    def _predict_base(model_tuple, X):
        name, model, mtype = model_tuple
        if mtype == 'xgb':
            raw = model.predict(xgb.DMatrix(X))
            return 1.0 / (1.0 + np.exp(-raw))
        elif mtype == 'lgb':
            raw = model.predict(X)
            return 1.0 / (1.0 + np.exp(-raw))
        elif mtype in ('rf', 'et'):
            return model.predict_proba(X)[:, 1]
        elif mtype == 'lr':
            return model.predict_proba(X)[:, 1]

    # ── Phase 3: SHAP feature selection + meta feature matrix ────────────────────
    print('\n' + '=' * 60)
    print('Phase 3: SHAP feature selection')
    print('=' * 60)

    X_test_feat  = df_test[FEAT_COLS].values.astype(np.float32)
    y_test       = df_test['target'].values.astype(np.int8)
    X_final_feat = df_final[FEAT_COLS].values.astype(np.float32)
    y_final      = df_final['target'].values.astype(np.int8)

    rename_map = build_rename_map(
        rsi_w, bb_w, sma_w,
        best_params.get('news_mom_w'), best_params.get('news_vol_ma'), best_params.get('news_tone_roll'),
        best_params.get('news_extra_zscore_w'), best_params.get('news_extra_tone_accel'),
        best_params.get('news_extra_macro_sec_diff'),
    )
    feat_display = [rename_map.get(c, c) for c in FEAT_COLS]

    # SHAP on XGB-1 (optimierte Optuna-Hyperparameter aus Phase 1), Stichprobe aus TEST
    _shap_hp = {k: best_params[k] for k in (
        'max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree',
        'min_child_weight', 'reg_alpha', 'reg_lambda', 'gamma', 'max_leaves',
    ) if k in best_params}
    print('SHAP-Modell: XGB-1; verwendete optimierte XGB-Hyperparameter:', _shap_hp)

    rng_shap    = np.random.RandomState(RANDOM_STATE)
    shap_idx    = rng_shap.choice(len(X_test_feat), size=min(2000, len(X_test_feat)), replace=False)
    xgb_model_1 = base_models[0][1]  # first XGBoost booster (trainiert mit xgb_base_params / best_params)
    explainer   = shap.TreeExplainer(xgb_model_1)
    shap_vals   = explainer.shap_values(xgb.DMatrix(X_test_feat[shap_idx]))
    mean_shap   = np.abs(shap_vals).mean(axis=0)

    # ── SHAP-Analyse: globale Bedeutung aller Features (mittlere |SHAP|) ───────────
    import matplotlib.pyplot as plt

    shap_df = (
        pd.DataFrame({'feature': feat_display, 'mean_abs_shap': mean_shap})
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    print('\nMittlere |SHAP| (alle Features, absteigend):')
    print(shap_df.to_string(index=False))

    _shap_top = min(25, len(shap_df))
    fig, ax = plt.subplots(figsize=(8, max(6, _shap_top * 0.28)))
    top = shap_df.head(_shap_top).iloc[::-1]
    ax.barh(top['feature'], top['mean_abs_shap'], color='steelblue')
    ax.set_xlabel('Mittlere |SHAP|')
    ax.set_title('SHAP: Base-XGB-1 (Optuna-Parameter), Test-Stichprobe')
    plt.tight_layout()
    plt.show()

    shap.summary_plot(
        shap_vals, X_test_feat[shap_idx], feature_names=feat_display,
        max_display=min(20, len(FEAT_COLS)), show=True,
    )

    topk_idx   = np.argsort(mean_shap)[::-1][:META_SHAP_TOP_K]
    topk_names = [FEAT_COLS[i] for i in topk_idx]
    print(f'\nTop {META_SHAP_TOP_K} SHAP features (Meta-Stacking): {[rename_map.get(n, n) for n in topk_names]}')
    print('\nCell 13 abgeschlossen. Bitte Cell 14 ausführen → Meta-Features + Meta-Learner + Threshold.')
