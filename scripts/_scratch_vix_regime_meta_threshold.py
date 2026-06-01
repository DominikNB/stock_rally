"""VIX-Regime-Effekt auf META / THRESHOLD / FINAL verifizieren.

Reproduziert die produktiven Meta-Signale (Base->Meta->Kalibrierung->Filter)
exakt wie die Pipeline auf ALLEN drei Zeitfenstern und bucketet die
Forward-Returns je Signal nach ``regime_vix_level``.

Ziel: Pruefen, ob der auf FINAL beobachtete VIX-Regime-Effekt auch auf den
legitimen Tuning-Datensaetzen (META, THRESHOLD) haelt. Nur wenn er dort
reproduziert, ist ein VIX-Filter methodisch gerechtfertigt (kein FINAL-Overfit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg

ART_PATH = ROOT / "models" / "scoring_artifacts.joblib"


def _predict_base(model_tuple, X):
    import xgboost as xgb

    name, model, mtype = model_tuple
    if mtype == "xgb":
        return 1.0 / (1.0 + np.exp(-model.predict(xgb.DMatrix(X))))
    if mtype == "lgb":
        return 1.0 / (1.0 + np.exp(-model.predict(X)))
    return model.predict_proba(X)[:, 1]


def main() -> None:
    art = joblib.load(ART_PATH)
    tickers = list(art["tickers_for_run"])

    # ── Universum/Modus so setzen, dass der Kalender-Split exakt dem
    #    trainierten Lauf entspricht (245 Ticker, gleiche Zeitfenster). ──
    cfg.RETRAIN_META_ONLY = True
    cfg.SCORING_ONLY = False
    cfg.ALL_TICKERS = tickers
    cfg.UNIVERSE_FRACTION = 1.0

    from lib.stock_rally_v10.data_and_split import run_data_download_and_split
    from lib.stock_rally_v10.features import merge_news_shard_from_best_params
    from lib.stock_rally_v10.optuna_train import (
        _blue_sky_weak_volume_mask_1d,
        _dynamic_threshold_mask_1d,
        _peak_rsi_mask_1d,
        _rsi_from_close_1d,
        _vol_stress_mask_1d,
    )

    print("=" * 70)
    print("Daten laden + Split (RETRAIN_META_ONLY, Caches werden genutzt) ...")
    print("=" * 70)
    run_data_download_and_split()

    FEAT_COLS = list(art["FEAT_COLS"])
    base_models = art["base_models"]
    topk_idx = np.asarray(art["topk_idx"])
    meta_clf = art["meta_clf"]
    cal = art.get("meta_proba_calibrator")
    best_threshold = float(art["best_threshold"])
    best_params = art.get("best_params", {})

    # ── Filter-Parameter aus dem Artefakt ──
    CONSEC = int(art["CONSECUTIVE_DAYS"])
    COOLDOWN = int(art["SIGNAL_COOLDOWN_DAYS"])
    rsi_w = int(art["rsi_w"])
    bb_w = int(art["bb_w"])
    skip_peak = bool(art["signal_skip_near_peak"])
    peak_lb = int(art["peak_lookback_days"])
    peak_dist = float(art["peak_min_dist_from_high_pct"])
    max_rsi = float(art["signal_max_rsi"])
    max_vol_stress = art["signal_max_vol_stress_z"]
    min_blue_z = art["signal_min_blue_sky_volume_z"]
    tv = float(art["dyn_vvix_trigger"])
    tr = float(art["dyn_rsi_trigger"])
    tb = float(art["dyn_bb_pband_trigger"])
    m1 = float(art["mult_final_threshold_1"])
    m2 = float(art["mult_final_threshold_2"])
    m3 = float(art["mult_final_threshold_3"])

    horizons = tuple(
        sorted({int(h) for h in getattr(cfg, "META_SIGNAL_RETURN_HORIZONS", (1, 2, 3, 4, 5)) if int(h) > 0})
    )
    hor_agg = str(getattr(cfg, "META_OBJECTIVE_HORIZON_AGGREGATION", "trimmed_mean")).strip().lower()
    hor_trim = float(getattr(cfg, "META_OBJECTIVE_HORIZON_TRIM_FRAC", 0.20) or 0.0)

    blue_w = int(
        best_params.get("breakout_lookback_window", 0)
        or cfg.SEED_PARAMS.get("breakout_lookback_window", 20)
    )
    blue_col = f"blue_sky_breakout_{blue_w}d"
    rsi_col = f"rsi_{rsi_w}d"
    bb_col = f"bb_pband_{bb_w}"

    def _agg_horizon(rlist):
        if not rlist:
            return 0.0
        arr = np.asarray(rlist, dtype=np.float64)
        if hor_agg == "mean":
            return float(np.mean(arr))
        if hor_agg == "median":
            return float(np.median(arr))
        n = arr.size
        k = int(np.floor(hor_trim * n))
        if k <= 0:
            return float(np.mean(arr))
        if 2 * k >= n:
            return float(np.median(arr))
        srt = np.sort(arr)
        return float(np.mean(srt[k:n - k]))

    nan_sent = np.float32(getattr(cfg, "FEATURE_NUMERIC_NAN_SENTINEL", -1e8))

    def build_meta(X):
        base_preds = np.column_stack([_predict_base(m, X) for m in base_models])
        return np.hstack([base_preds, X[:, topk_idx]]).astype(np.float32)

    def apply_cal(p):
        if not cal or not isinstance(cal, dict):
            return p
        method = cal.get("method")
        model = cal.get("model")
        if model is None:
            return p
        vv = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        if method == "sigmoid":
            ll = np.log(vv / (1.0 - vv)).reshape(-1, 1)
            return model.predict_proba(ll)[:, 1].astype(np.float64)
        if method == "isotonic":
            return np.asarray(model.predict(vv), dtype=np.float64)
        return p

    def score_signals(df_in):
        """Reproduziert Meta-Signale + per-Signal Forward-Return + VIX je Signal."""
        df = df_in.copy()
        if getattr(cfg, "USE_NEWS_SENTIMENT", False):
            df = merge_news_shard_from_best_params(df, best_params)
        X = df[FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(X, nan=nan_sent, posinf=nan_sent, neginf=nan_sent, copy=False)
        Xm = build_meta(X)
        prob = apply_cal(meta_clf.predict_proba(Xm)[:, 1])
        df = df.reset_index(drop=True)
        df["prob"] = prob

        records = []
        for _, sub in df.groupby("ticker"):
            sub = sub.sort_values("Date").reset_index(drop=True)
            n = len(sub)
            if n == 0:
                continue
            raw = _dynamic_threshold_mask_1d(
                sub["prob"].to_numpy(dtype=np.float64),
                best_threshold,
                vvix_ratio=sub["mr_vvix_div_vix"].to_numpy(dtype=np.float64) if "mr_vvix_div_vix" in sub.columns else None,
                rsi_arr=sub[rsi_col].to_numpy(dtype=np.float64) if rsi_col in sub.columns else None,
                bb_pband_arr=sub[bb_col].to_numpy(dtype=np.float64) if bb_col in sub.columns else None,
                vvix_trigger=tv, rsi_trigger=tr, bb_pband_trigger=tb,
                mult1=m1, mult2=m2, mult3=m3,
            ).astype(np.int8)

            consec = np.zeros(n, dtype=np.int8)
            for i in range(2, n):
                if raw[i - 2] + raw[i - 1] + raw[i] >= CONSEC:
                    consec[i] = 1
            final = np.zeros(n, dtype=np.int8)
            last_sig = -999
            for i in range(n):
                if consec[i] == 1 and (i - last_sig) >= COOLDOWN:
                    final[i] = 1
                    last_sig = i

            close_sub = pd.to_numeric(sub["close"], errors="coerce").to_numpy(dtype=np.float64)
            rsi_sub = _rsi_from_close_1d(close_sub, rsi_w)
            mask_ok = _peak_rsi_mask_1d(close_sub, rsi_sub, skip_peak, peak_lb, peak_dist, max_rsi)
            for i in range(n):
                if final[i] == 1 and not bool(mask_ok[i]):
                    final[i] = 0
            if max_vol_stress is not None and "vol_stress" in sub.columns:
                stress_ok = _vol_stress_mask_1d(
                    close_sub, sub["vol_stress"].to_numpy(dtype=np.float64), max_vol_stress
                )
                for i in range(n):
                    if final[i] == 1 and not bool(stress_ok[i]):
                        final[i] = 0
            if min_blue_z is not None and blue_col in sub.columns and "volume_zscore" in sub.columns:
                blue_ok = _blue_sky_weak_volume_mask_1d(
                    sub[blue_col].to_numpy(dtype=np.float64),
                    sub["volume_zscore"].to_numpy(dtype=np.float64),
                    min_blue_z,
                )
                for i in range(n):
                    if final[i] == 1 and not bool(blue_ok[i]):
                        final[i] = 0

            open_sub = pd.to_numeric(sub["open"], errors="coerce").to_numpy(dtype=np.float64)

            def _col(name):
                return (
                    pd.to_numeric(sub[name], errors="coerce").to_numpy(dtype=np.float64)
                    if name in sub.columns else np.full(n, np.nan)
                )

            vix_sub = _col("regime_vix_level")
            # dist_from_20d_high_pct direkt aus dem Kurs (wie _merge_ticker_date_features):
            # (close - rollmax20(close)) / rollmax20(close). Prob-unabhaengig, historisch
            # exakt rekonstruierbar — anders als die Earnings-Naehe.
            _cs = pd.Series(close_sub)
            _rm20 = _cs.rolling(20, min_periods=5).max().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                dist_high_sub = np.where(_rm20 > 0, (close_sub - _rm20) / _rm20, np.nan)
            tkr = str(sub["ticker"].iloc[0])
            for i in np.where(final == 1)[0]:
                entry_idx = i + 1
                if entry_idx >= n:
                    continue
                entry_open = open_sub[entry_idx]
                if not np.isfinite(entry_open) or entry_open <= 0.0:
                    continue
                if any((i + h) >= n for h in horizons):
                    continue
                rlist = []
                for h in horizons:
                    pj = close_sub[i + h]
                    if np.isfinite(pj) and pj > 0.0:
                        rlist.append(pj / entry_open - 1.0)
                if not rlist:
                    continue
                records.append({
                    "ticker": tkr,
                    "Date": pd.Timestamp(sub["Date"].iloc[i]).normalize(),
                    "ret": _agg_horizon(rlist),
                    "vix": vix_sub[i],
                    "dist_high": dist_high_sub[i],
                })
        return pd.DataFrame(records)

    print("\nFilter-/Return-Setup:")
    print(f"  best_threshold={best_threshold:.3f} | consec={CONSEC} cooldown={COOLDOWN}")
    print(f"  horizons={horizons} | horizon_agg={hor_agg}(trim={hor_trim})")
    print(f"  cols: rsi={rsi_col} bb={bb_col} blue={blue_col}")

    datasets = {
        "META": cfg.df_train_meta,
        "THRESHOLD": cfg.df_threshold,
        "FINAL": cfg.df_final,
    }
    vix_bins = [0, 15, 20, 25, 100]
    vix_labels = ["VIX<15", "15-20", "20-25", "VIX>25"]

    all_sig = {}
    for name, dfds in datasets.items():
        d0 = pd.to_datetime(dfds["Date"]).min().date()
        d1 = pd.to_datetime(dfds["Date"]).max().date()
        sig = score_signals(dfds)
        sig["dataset"] = name
        all_sig[name] = sig
        print("\n" + "=" * 70)
        print(f"{name}  ({d0} .. {d1})  Zeilen={len(dfds):,}  Signale={len(sig):,}")
        print("=" * 70)
        if sig.empty:
            print("  (keine Signale)")
            continue
        r = sig["ret"]
        print(f"  GESAMT: mean={100*r.mean():+.3f}%  median={100*r.median():+.3f}%  "
              f"win={100*(r > 0).mean():.1f}%  n={len(sig)}")
        sig["vix_bucket"] = pd.cut(sig["vix"], bins=vix_bins, labels=vix_labels, right=False)
        print("  Nach VIX-Regime:")
        for lab in vix_labels:
            g = sig[sig["vix_bucket"] == lab]
            if len(g) == 0:
                print(f"    {lab:>8}: n=  0")
                continue
            rr = g["ret"]
            print(f"    {lab:>8}: n={len(g):4d}  mean={100*rr.mean():+.3f}%  "
                  f"median={100*rr.median():+.3f}%  win={100*(rr > 0).mean():.1f}%")

    # ── Pooled META+THRESHOLD vs FINAL: haelt der Effekt out-of-FINAL? ──
    print("\n" + "=" * 70)
    print("POOLED  META + THRESHOLD  (legitimer Tuning-Datensatz, FINAL-unabhaengig)")
    print("=" * 70)
    pooled = pd.concat([all_sig["META"], all_sig["THRESHOLD"]], ignore_index=True)
    if not pooled.empty:
        pooled["vix_bucket"] = pd.cut(pooled["vix"], bins=vix_bins, labels=vix_labels, right=False)
        r = pooled["ret"]
        print(f"  GESAMT: mean={100*r.mean():+.3f}%  median={100*r.median():+.3f}%  "
              f"win={100*(r > 0).mean():.1f}%  n={len(pooled)}")
        print("  Nach VIX-Regime:")
        for lab in vix_labels:
            g = pooled[pooled["vix_bucket"] == lab]
            if len(g) == 0:
                print(f"    {lab:>8}: n=  0")
                continue
            rr = g["ret"]
            print(f"    {lab:>8}: n={len(g):4d}  mean={100*rr.mean():+.3f}%  "
                  f"median={100*rr.median():+.3f}%  win={100*(rr > 0).mean():.1f}%")

        # Einfacher Schwellen-Test: VIX>=20 vs <20 (gepoolt, FINAL-unabhaengig)
        hi = pooled[pooled["vix"] >= 20]["ret"]
        lo = pooled[pooled["vix"] < 20]["ret"]
        print("\n  Schwelle VIX>=20 (gepoolt META+THRESHOLD):")
        if len(hi):
            print(f"    VIX>=20: n={len(hi):4d}  mean={100*hi.mean():+.3f}%  win={100*(hi > 0).mean():.1f}%")
        if len(lo):
            print(f"    VIX< 20: n={len(lo):4d}  mean={100*lo.mean():+.3f}%  win={100*(lo > 0).mean():.1f}%")

    # ──────────────────────────────────────────────────────────────────
    #  STRUKTURELLE, PROB-UNABHAENGIGE FILTER  (Earnings-Naehe / dist_high)
    #  Validierung auf POOLED META+THRESHOLD vs. FINAL als Bestaetigung.
    # ──────────────────────────────────────────────────────────────────
    def _bucket_report(df, col, bins, labels, title):
        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)
        for src, label in [(pooled, "META+THR"), (all_sig["FINAL"], "FINAL")]:
            d = src.dropna(subset=[col, "ret"]).copy()
            print(f"  [{label}] n={len(d)}")
            if d.empty:
                continue
            d["_b"] = pd.cut(d[col], bins=bins, labels=labels, right=False)
            for lab in labels:
                g = d[d["_b"] == lab]
                if len(g) == 0:
                    print(f"    {lab:>14}: n=  0")
                    continue
                rr = g["ret"]
                print(f"    {lab:>14}: n={len(g):4d}  mean={100*rr.mean():+.3f}%  "
                      f"median={100*rr.median():+.3f}%  win={100*(rr > 0).mean():.1f}%")

    # Earnings: kein point-in-time Kalender verfuegbar -> nicht historisch testbar.
    print("\n" + "=" * 70)
    print("EARNINGS-NAEHE: uebersprungen (kein historischer Earnings-Kalender; "
          "yfinance liefert nur den aktuell naechsten Termin).")
    print("=" * 70)

    # dist_from_20d_high_pct (negativ = unter dem Hoch)
    _bucket_report(
        pooled,
        "dist_high",
        bins=[-1.0, -0.10, -0.05, -0.02, 0.0, 1.0],
        labels=["<-10%", "-10..-5%", "-5..-2%", "-2..0%", ">=0% (Hoch)"],
        title="ABSTAND ZUM 20-TAGE-HOCH (dist_from_20d_high_pct)",
    )

    # Records sichern, damit weitere Auswertungen ohne Re-Load moeglich sind.
    out_csv = ROOT / "data" / "_scratch_meta_thr_final_signals.csv"
    pd.concat(list(all_sig.values()), ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nSignal-Records gespeichert: {out_csv}")


if __name__ == "__main__":
    main()
