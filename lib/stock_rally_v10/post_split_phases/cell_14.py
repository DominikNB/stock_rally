if globals().get('SCORING_ONLY', False):
    print('[SCORING_ONLY] Training-Zelle übersprungen.')
else:
    # Cell 13b — Markt-Regime nur für Evaluation (kein Training)
    # Benchmark-Kurs: Trend (Close vs. SMA200) und Volatilität (RV20 vs. RV60), alles nur aus Vergangenheit.
    # Optional: VIX vs. rollierendem Median (Vergangenheit). Danach: AP/Precision/Recall pro Jahr & Regime.

    REGIME_BENCHMARK = '^GSPC'   # z. B. '^GSPC', '^STOXX', '^GDAXI', 'SPY'
    REGIME_VIX = '^VIX'          # None = VIX weglassen
    REGIME_VIX_MIN_PERIODS = 60


    def _yf_close_series(ticker):
        """Einzelner Close als Series (DatetimeIndex)."""
        raw = yf.download(
            ticker, start=START_DATE, end=END_DATE,
            auto_adjust=True, threads=False, progress=False,
        )
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw['Close'].columns:
                c = raw['Close'][ticker]
            else:
                c = raw['Close'].iloc[:, 0]
        else:
            c = raw['Close']
        c = c.dropna()
        if hasattr(c.index, 'tz') and c.index.tz is not None:
            c.index = c.index.tz_localize(None)
        return c


    def _regime_table_from_close(close: pd.Series, name='bench'):
        """Pro Handelstag des Benchmarks: Trend- und Vol-Label (kausal)."""
        s = close.sort_index().astype(float)
        log_ret = np.log(s / s.shift(1))
        rv20 = log_ret.rolling(20, min_periods=10).std()
        rv60 = log_ret.rolling(60, min_periods=20).std()
        sma200 = s.rolling(200, min_periods=50).mean()
        trend = np.where(s > sma200, 'bull', 'bear')
        vol_c = np.where(rv20 > rv60, 'high_vol', 'low_vol')
        return pd.DataFrame({
            'Date': pd.to_datetime(s.index).normalize(),
            f'{name}_trend': trend,
            f'{name}_vol': vol_c,
        })


    def _merge_regime_on_threshold(df_thr, regime_df, date_col='Date'):
        """Ordnet jeder Zeile den letzten bekannten Regime-Stand (merge_asof backward)."""
        left = df_thr.copy()
        left['_d'] = pd.to_datetime(left[date_col]).dt.normalize()
        right = regime_df.sort_values('Date').rename(columns={'Date': '_regime_date'})
        left = left.sort_values('_d')
        out = pd.merge_asof(
            left, right,
            left_on='_d', right_on='_regime_date',
            direction='backward',
        )
        return out.drop(columns=['_d', '_regime_date'], errors='ignore')


    def _report_regime_metrics(df_ev, prob_col='prob', pred_col='pred', true_col='target', group_cols=None):
        """AP aus Wahrscheinlichkeiten; Precision/Recall aus binärer Vorhersage (≥ best_threshold)."""
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        if group_cols is None:
            group_cols = []
        rows = []
        if not group_cols:
            y = df_ev[true_col].values
            pr = df_ev[prob_col].values
            pb = df_ev[pred_col].values.astype(int)
            rows.append({
                'n': len(df_ev),
                'pos': int(y.sum()),
                'AP': average_precision_score(y, pr) if y.sum() > 0 else float('nan'),
                'Prec': precision_score(y, pb, zero_division=0),
                'Rec': recall_score(y, pb, zero_division=0),
            })
            return pd.DataFrame(rows)
        for keys, sub in df_ev.groupby(group_cols, dropna=False):
            if len(sub) < 50:
                continue
            yt = sub[true_col].values
            pr = sub[prob_col].values
            pb = sub[pred_col].values.astype(int)
            ap = average_precision_score(yt, pr) if yt.sum() > 0 else float('nan')
            row = {
                'n': len(sub),
                'pos': int(yt.sum()),
                'AP': ap,
                'Prec': precision_score(yt, pb, zero_division=0),
                'Rec': recall_score(yt, pb, zero_division=0),
            }
            keys = keys if isinstance(keys, tuple) else (keys,)
            for c, v in zip(group_cols, keys):
                row[c] = v
            rows.append(row)
        return pd.DataFrame(rows)


    # --- Ausführung (nach Phase 5: df_threshold['prob'], best_threshold gesetzt) ---
    try:
        _close_b = _yf_close_series(REGIME_BENCHMARK)
        if _close_b is None:
            print('Regime-Evaluation: Benchmark-Download leer — übersprungen.')
        else:
            _reg_b = _regime_table_from_close(_close_b, name='mkt')
            _df_r = df_threshold.copy()
            _df_r['year'] = pd.to_datetime(_df_r['Date']).dt.year
            _df_r = _merge_regime_on_threshold(_df_r, _reg_b)
            if REGIME_VIX:
                _vx = _yf_close_series(REGIME_VIX)
                if _vx is not None and len(_vx) > REGIME_VIX_MIN_PERIODS:
                    _v = _vx.sort_index().astype(float)
                    _med = _v.rolling(252, min_periods=REGIME_VIX_MIN_PERIODS).median().shift(1)
                    _vb = np.where(
                        (_v.values > _med.values) & np.isfinite(_med.values),
                        'vix_high',
                        'vix_low',
                    )
                    _vix_df = pd.DataFrame({
                        'Date': pd.to_datetime(_v.index).normalize(),
                        'vix_bucket': _vb,
                    }).sort_values('Date')
                    _df_r = _merge_regime_on_threshold(_df_r, _vix_df)
            thr = float(best_threshold)
            _df_r['pred'] = (_df_r['prob'].values >= thr).astype(np.int8)
            print('\n' + '=' * 60)
            print('Regime-Evaluation (nur Bericht, THRESHOLD-Set)')
            print(f'Benchmark: {REGIME_BENCHMARK}  |  thr={thr:.3f}  |  AP aus prob; Prec/Rec aus pred ≥ thr')
            print('=' * 60)
            _by_y = _report_regime_metrics(_df_r, 'prob', 'pred', 'target', ['year'])
            if len(_by_y):
                print('\nPro Kalenderjahr:')
                print(_by_y.sort_values('year').to_string(index=False))
            _cols = [c for c in ['mkt_trend', 'mkt_vol'] if c in _df_r.columns]
            if _cols:
                print('\nPro Markt-Regime (Benchmark), Trend × Vol (alle Kombinationen):')
                print(_report_regime_metrics(_df_r, 'prob', 'pred', 'target', _cols).to_string(index=False))
            if 'vix_bucket' in _df_r.columns:
                print('\nPro VIX-Bucket (vs. 252d-Median, kausal shift(1)):')
                print(_report_regime_metrics(_df_r, 'prob', 'pred', 'target', ['vix_bucket']).to_string(index=False))
            print('\nHinweis: pred = (prob >= best_threshold); Konsekutiv-/Cooldown-Filter sind hier nicht enthalten.')
    except NameError as _e:
        print(f'Regime-Evaluation übersprungen (fehlt: {_e}). Zuerst Cell 13 (Phase 5) ausführen.')
    except Exception as _e:
        print(f'Regime-Evaluation Fehler: {_e}')
