"""
Rot-Regime: Gesamtscore aus Chips + Badge testen (IS/OOS).

  python scripts/_scratch_validate_red_composite.py

Methoden:
  - additiv (Chips grün + Badge-Treffer)
  - gewichtet (nur OOS-bestätigte Einzelfaktoren)
  - Logistic Regression (META+THR fit, Median-Split hi/lo)
  - Random Forest (max_depth=2, min_samples_leaf=30)

Ziel: ret_mean_5 — hi vs lo Bucket wie _scratch_validate_red_quality_tiers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.red_signal_quality import calibrate_gld_ret5_median_red_ref
from lib.vix_red_context_chips import _chip_thresholds
from scripts._scratch_validate_red_quality_tiers import (
    MIN_DELTA,
    MIN_N,
    VIX_RED,
    _assign_dataset,
)

MASTER = ROOT / "data" / "master_complete.csv"
OUT = ROOT / "data" / "_scratch_red_composite_validation.json"


def _feature_frame(red: pd.DataFrame) -> pd.DataFrame:
    thr = _chip_thresholds()
    gld_ref = calibrate_gld_ret5_median_red_ref()
    o = red.copy()
    z = pd.to_numeric(o["regime_vix_z_20d"], errors="coerce")
    vr = pd.to_numeric(o["vix3m_vix_ratio"], errors="coerce")
    h = pd.to_numeric(o["sector_hhi_same_day"], errors="coerce")
    gld = pd.to_numeric(o["gld_ret_5d"], errors="coerce")
    tier = o["liquidity_tier"].astype(str).str.strip().str.lower()

    o["f_vix_z_good"] = (z < 0).astype(float)
    o["f_vix_term_good"] = (vr < thr["vix3m_vix_max"]).astype(float)
    o["f_crowd_good"] = (h < thr["sector_hhi_max"]).astype(float)
    o["f_liq_ok"] = (tier == "ok").astype(float)
    o["f_gld_low"] = (gld < gld_ref).astype(float)
    o["chip_good"] = o[["f_vix_z_good", "f_vix_term_good", "f_crowd_good"]].sum(axis=1)
    o["badge_hits"] = o[["f_liq_ok", "f_gld_low"]].sum(axis=1)
    o["composite_add"] = o["chip_good"] + o["badge_hits"]
    # nur OOS-validierte Badge-Faktoren + Chips als Kontext (explizit getrennt testbar)
    o["composite_validated_only"] = o["f_liq_ok"] + o["f_gld_low"]
    o["composite_ctx_plus_valid"] = o["chip_good"] + o["f_liq_ok"] + o["f_gld_low"]
    return o


def _delta(hi: pd.DataFrame, lo: pd.DataFrame) -> tuple[float, int, int]:
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan"), len(hi), len(lo)
    return float(hi["ret"].mean() - lo["ret"].mean()), len(hi), len(lo)


def _eval_split(hi: pd.DataFrame, lo: pd.DataFrame) -> tuple[float, int, int, bool]:
    d, nh, nl = _delta(hi, lo)
    ok = np.isfinite(d) and abs(d) >= MIN_DELTA
    return d, nh, nl, ok


def _mask_median(s: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = s[col].median()
    return s[s[col] >= med], s[s[col] < med]


def _ml_scores(
    tune: pd.DataFrame,
    fin: pd.DataFrame,
    feat_cols: list[str],
    model_name: str,
) -> tuple[pd.Series, pd.Series]:
    tune = tune.dropna(subset=feat_cols + ["ret"])
    fin = fin.dropna(subset=feat_cols + ["ret"])
    if len(tune) < 80:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    y_med = tune["ret"].median()
    y_tune = (tune["ret"] >= y_med).astype(int)
    X_tune = tune[feat_cols].values
    if model_name == "logit":
        clf = Pipeline(
            [
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced")),
            ]
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            min_samples_leaf=30,
            random_state=42,
            class_weight="balanced",
        )
    clf.fit(X_tune, y_tune)
    p_tune = pd.Series(clf.predict_proba(X_tune)[:, 1], index=tune.index)
    p_fin = pd.Series(clf.predict_proba(fin[feat_cols].values)[:, 1], index=fin.index)
    return p_tune, p_fin


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    red = _feature_frame(red)
    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]
    print(f"Rot: tune={len(tune)} final={len(fin)}")

    results: list[dict] = []

    def record(rid: str, label: str, dt: float, df: float, nth: int, ntl: int, nfh: int, nfl: int):
        oos = (
            np.isfinite(dt)
            and np.isfinite(df)
            and dt * df > 0
            and abs(dt) >= MIN_DELTA
            and abs(df) >= MIN_DELTA
        )
        impl = bool(oos and dt > 0 and df > 0)
        results.append(
            {
                "id": rid,
                "label": label,
                "d_tune_pp": round(100 * dt, 3) if np.isfinite(dt) else None,
                "d_fin_pp": round(100 * df, 3) if np.isfinite(df) else None,
                "n_tune_hi": int(nth),
                "n_tune_lo": int(ntl),
                "n_fin_hi": int(nfh),
                "n_fin_lo": int(nfl),
                "oos_ok": bool(oos),
                "implement": bool(impl),
            }
        )
        mark = "IMPLEMENT" if impl else ("oos+" if oos else "fail")
        print(
            f"  [{mark}] {label}: tune {100*dt:+.2f}pp (n={nth}/{ntl}) | "
            f"fin {100*df:+.2f}pp (n={nfh}/{nfl})"
            if np.isfinite(dt) and np.isfinite(df)
            else f"  [fail] {label}: insufficient n"
        )

    print("\n--- Additive / Schwellen ---")
    for col, label in [
        ("composite_add", "chip_good + badge_hits (0-5), Median-Split"),
        ("composite_validated_only", "nur liquidity_ok + gld_low (0-2), Median-Split"),
        ("composite_ctx_plus_valid", "chip_good + liquidity + gld, Median-Split"),
        ("chip_good", "nur Chip-Score 0-3, Median-Split"),
        ("badge_hits", "nur Badge 0-2, Median-Split"),
    ]:
        for sub_name, sub in [("META+THR", tune), ("FINAL", fin)]:
            pass
        hi_t, lo_t = _mask_median(tune, col)
        hi_f, lo_f = _mask_median(fin, col)
        dt, nth, ntl, _ = _eval_split(hi_t, lo_t)
        df, nfh, nfl, _ = _eval_split(hi_f, lo_f)
        record(col, label, dt, df, nth, ntl, nfh, nfl)

    # chip >= 2 vs < 2 (research hint)
    hi_t, lo_t = tune[tune["chip_good"] >= 2], tune[tune["chip_good"] < 2]
    hi_f, lo_f = fin[fin["chip_good"] >= 2], fin[fin["chip_good"] < 2]
    dt, nth, ntl, _ = _eval_split(hi_t, lo_t)
    df, nfh, nfl, _ = _eval_split(hi_f, lo_f)
    record("chip_ge2", "chip_good >= 2 vs < 2", dt, df, nth, ntl, nfh, nfl)

    # badge both when known
    def _both_badge(s: pd.DataFrame):
        known = s["f_liq_ok"].notna() & s["f_gld_low"].notna()
        s2 = s[known]
        hi = s2[(s2["f_liq_ok"] == 1) & (s2["f_gld_low"] == 1)]
        lo = s2[~((s2["f_liq_ok"] == 1) & (s2["f_gld_low"] == 1))]
        return hi, lo

    hi_t, lo_t = _both_badge(tune)
    hi_f, lo_f = _both_badge(fin)
    dt, nth, ntl, _ = _eval_split(hi_t, lo_t)
    df, nfh, nfl, _ = _eval_split(hi_f, lo_f)
    record("badge_2of2", "liquidity_ok AND gld_low", dt, df, nth, ntl, nfh, nfl)

    print("\n--- ML (fit META+THR, Median prob hi/lo) ---")
    feat5 = ["f_vix_z_good", "f_vix_term_good", "f_crowd_good", "f_liq_ok", "f_gld_low"]
    feat3 = ["f_vix_z_good", "f_vix_term_good", "f_crowd_good"]
    feat2 = ["f_liq_ok", "f_gld_low"]

    for model_name in ("logit", "rf"):
        for feats, fid in [
            (feat5, f"{model_name}_5feat"),
            (feat3, f"{model_name}_chips"),
            (feat2, f"{model_name}_badge"),
        ]:
            p_tune, p_fin = _ml_scores(tune, fin, feats, model_name)
            if p_tune.empty:
                record(fid, f"{model_name} {len(feats)} feat", float("nan"), float("nan"), 0, 0, 0, 0)
                continue
            tune2 = tune.loc[p_tune.index].copy()
            tune2["_p"] = p_tune
            fin2 = fin.loc[p_fin.index].copy()
            fin2["_p"] = p_fin
            hi_t, lo_t = _mask_median(tune2, "_p")
            hi_f, lo_f = _mask_median(fin2, "_p")
            dt, nth, ntl, _ = _eval_split(hi_t, lo_t)
            df, nfh, nfl, _ = _eval_split(hi_f, lo_f)
            record(fid, f"{model_name.upper()} {len(feats)} features, p>=median", dt, df, nth, ntl, nfh, nfl)

    passed = [r for r in results if r["implement"]]
    best_oos = sorted(
        [r for r in results if r["oos_ok"]],
        key=lambda r: (r.get("d_fin_pp") or -999),
        reverse=True,
    )

    payload = {
        "passed_ids": [r["id"] for r in passed],
        "best_oos_positive_fin": best_oos[:5],
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")
    print(f"OOS-implementierbar: {[r['id'] for r in passed]}")


if __name__ == "__main__":
    main()
