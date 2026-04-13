"""
Platt-Kalibrierung der gespeicherten ``prob`` auf dem Holdout (``data/master_complete.csv``)
und Prüfung, ob die Score-Höhe trägt.

``master_complete`` enthält **alle** Zeilen aus ``holdout_signals.csv`` (inkl. unterhalb der
Schwelle), sofern beim Build nicht anders gefiltert. Standard hier: nur **Signale**
``prob >= threshold_used`` — wie in der Handelslogik.

- Zeitlicher Split: ältere Zeilen = Kalibrierung, jüngere = Test (kein Shuffle).
- Platt: Logistische Regression auf logit(prob).
- Metriken: AUC, Brier, Kalibrierungstabelle, Rendite nach Score-Dezil (Test).

Ausführung vom Projektroot:
  python -m holdout.platt_calibrate_holdout # nur prob >= threshold_used
  python -m holdout.platt_calibrate_holdout --all-rows # gesamte CSV (Vergleich)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "master_complete.csv"
CAL_FRACTION = 0.5 # ältere Hälfte kalibrieren, jüngere evaluieren
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _platt_fit_predict(
    prob_cal: np.ndarray,
    y_cal: np.ndarray,
    prob_test: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    """Kalibrierung auf Cal, angewandt auf Test."""
    X_cal = _logit(prob_cal).reshape(-1, 1)
    X_test = _logit(prob_test).reshape(-1, 1)
    # Schwache Regularisierung: bei sehr wenigen Positiven numerisch stabiler
    lr = LogisticRegression(
        C=1e6,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_cal, y_cal)
    p_test = lr.predict_proba(X_test)[:, 1]
    return p_test, lr


def _calibration_table(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mittlere vorhergesagte Wahrscheinlichkeit vs. Häufigkeit y=1 pro Bin."""
    df = pd.DataFrame({"p": prob, "y": y})
    df = df[np.isfinite(df["p"]) & np.isfinite(df["y"])]
    if len(df) < n_bins * 5:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    out = g.agg(mean_pred=("p", "mean"), freq=("y", "mean"), n=("y", "count"))
    return out.reset_index()


def _decile_returns(prob: np.ndarray, ret: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    m = np.isfinite(prob) & np.isfinite(ret)
    df = pd.DataFrame({"p": prob[m], "r": ret[m]})
    if len(df) < n_bins * 5:
        return pd.DataFrame()
    df["dec"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("dec", observed=True)
    return g.agg(mean_prob=("p", "mean"), mean_ret=("r", "mean"), n=("r", "count")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Platt + Score-Diagnostik (Holdout master_complete)")
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="Kein Filter auf threshold_used (auch Scores unter der Schwelle)",
    )
    args = ap.parse_args()

    if not CSV_PATH.is_file():
        print(f"Fehlt {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    if "fwd_error" not in df.columns:
        print("Spalte fwd_error erwartet.", file=sys.stderr)
        sys.exit(1)
    df = df[df["fwd_error"].isna()].copy()
    n_all = len(df)
    if not args.all_rows:
        if "threshold_used" not in df.columns or "prob" not in df.columns:
            print("Spalten prob und threshold_used erwartet.", file=sys.stderr)
            sys.exit(1)
        thr = pd.to_numeric(df["threshold_used"], errors="coerce")
        pr = pd.to_numeric(df["prob"], errors="coerce")
        ok = pr >= thr
        n_drop = int((~ok).sum())
        df = df.loc[ok].copy()
        print(
            f"Filter Signale: prob >= threshold_used  (entfernt {n_drop} von {n_all} Zeilen unterhalb Schwelle)"
        )
    else:
        print("Modus --all-rows: keine Schwellen-Filterung.")
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    cut = int(n * CAL_FRACTION)
    cal = df.iloc[:cut]
    tst = df.iloc[cut:]
    print(f"Holdout-Zeilen (ohne fwd_error): n={n}")
    print(f"Kalibrierung: Datum {cal['Date'].min().date()} … {cal['Date'].max().date()} (n={len(cal)})")
    print(f"Test:         Datum {tst['Date'].min().date()} … {tst['Date'].max().date()} (n={len(tst)})")

    prob_c = cal["prob"].to_numpy()
    prob_t = tst["prob"].to_numpy()

    # --- Label 1: Trainingsziel (wie im Modell) ---
    yc = cal["train_target"].to_numpy(dtype=np.float64)
    yt = tst["train_target"].to_numpy(dtype=np.float64)
    print("\n=== Label: train_target (Modell-Ziel) ===")
    print(f"Cal: positiv {yc.sum():.0f} / {len(yc)} ({yc.mean()*100:.2f}%)")
    print(f"Tst: positiv {yt.sum():.0f} / {len(yt)} ({yt.mean()*100:.2f}%)")

    if yc.sum() < 5 or (1 - yc).sum() < 5:
        print("Zu wenig Klassen in Cal für stabiles Platt — nur deskriptive Kennzahlen.")
    else:
        p_platt_t, lr = _platt_fit_predict(prob_c, yc.astype(int), prob_t)
        auc_raw = roc_auc_score(yt, prob_t) if len(np.unique(yt)) > 1 else float("nan")
        auc_pl = roc_auc_score(yt, p_platt_t) if len(np.unique(yt)) > 1 else float("nan")
        b_raw = brier_score_loss(yt, prob_t)
        b_pl = brier_score_loss(yt, p_platt_t)
        print(f"AUC (raw prob vs train_target): {auc_raw:.4f}")
        if auc_raw < 0.5:
            print(
                f"  Hinweis: AUC < 0.5 => höheres prob tendiert zu weniger target=1 (oder Shift). "
                f"1-AUC={1-auc_raw:.4f}"
            )
        print(f"AUC (Platt vs train_target): {auc_pl:.4f}")
        if lr.coef_.ravel()[0] < 0:
            print(
                "  Hinweis: Platt-Koeffizient < 0 => kalibrierte Wahrscheinlichkeit fällt mit steigendem "
                "Roh-Score; AUC kann sich gegenüber raw umkehren (Ranking-Inversion)."
            )
        print(f"Brier raw:       {b_raw:.5f}")
        print(f"Brier Platt:     {b_pl:.5f}")
        print("Kalibrierung Test — roh (10 Quantile):")
        print(_calibration_table(prob_t, yt, 10).to_string(index=False))
        print("Kalibrierung Test — nach Platt:")
        print(_calibration_table(p_platt_t, yt, 10).to_string(index=False))
        print("Platt-Koeffizienten (auf logit prob): coef=", lr.coef_.ravel(), "intercept=", lr.intercept_)

    # --- Label 2: Forward-Rendite > 0 (ökonomisch) ---
    if "ret_mean_5" in tst.columns:
        rc = pd.to_numeric(cal["ret_mean_5"], errors="coerce")
        rt = pd.to_numeric(tst["ret_mean_5"], errors="coerce")
        yc2 = (rc > 0).astype(int).to_numpy()
        yt2 = (rt > 0).astype(int).to_numpy()
        mc = np.isfinite(rc)
        mt = np.isfinite(rt)
        print("\n=== Label: ret_mean_5 > 0 (nur finite Zeilen) ===")
        print(f"Cal positiv-Rate: {yc2[mc].mean()*100:.2f}% (n={mc.sum()})")
        print(f"Tst positiv-Rate: {yt2[mt].mean()*100:.2f}% (n={mt.sum()})")
        if mt.sum() > 50 and len(np.unique(yt2[mt])) > 1:
            auc_e = roc_auc_score(yt2[mt], prob_t[mt])
            print(f"AUC prob vs (ret_mean_5>0) [Test]: {auc_e:.4f}")
        print("\nTest: mittlere ret_mean_5 nach Roh-Score-Dezil (höheres Dezil = höherer prob):")
        tab = _decile_returns(prob_t, rt.to_numpy(), 10)
        print(tab.to_string(index=False))
        if len(tab) >= 3:
            cors = tab["mean_prob"].corr(tab["mean_ret"])
            print(f"Korrelation (Dezil) mean_prob vs mean_ret: {cors:.4f}")

    print("\nFertig.")


if __name__ == "__main__":
    main()
