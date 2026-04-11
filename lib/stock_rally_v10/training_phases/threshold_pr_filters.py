"""Phase 15: PR-Kurven, Threshold-Sweep, ``apply_signal_filters`` auf dem Config-Modul."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from lib.stock_rally_v10.holdout_plot import (
    _rows_for_signal_calendar_day,
    apply_signal_filters as apply_signal_filters_from_cfg,
)


def run_phase_threshold_pr_and_filters(cfg_mod: Any) -> None:
    # Immer setzen — Phase 17 braucht das auch bei SCORING_ONLY (Phase 15 sonst früh return).
    cfg_mod.apply_signal_filters = apply_signal_filters_from_cfg
    if getattr(cfg_mod, "_rows_for_signal_calendar_day", None) is None:
        cfg_mod._rows_for_signal_calendar_day = _rows_for_signal_calendar_day

    if getattr(cfg_mod, "SCORING_ONLY", False):
        print("[SCORING_ONLY] Training-Zelle übersprungen.")
        return

    df_threshold = cfg_mod.df_threshold
    df_final = cfg_mod.df_final
    best_threshold = cfg_mod.best_threshold
    f1_thresh = cfg_mod.f1_thresh
    threshold_tickers = cfg_mod.threshold_tickers
    final_tickers = cfg_mod.final_tickers
    OPT_MIN_PRECISION = cfg_mod.OPT_MIN_PRECISION

    y_prob_threshold = df_threshold["prob"].values
    y_threshold = df_threshold["target"].values.astype(np.int8)
    y_prob_final = df_final["prob"].values
    y_final = df_final["target"].values.astype(np.int8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    prec_thr, rec_thr, _ = precision_recall_curve(y_threshold, y_prob_threshold)
    ap_thr = average_precision_score(y_threshold, y_prob_threshold)
    ax1.plot(rec_thr, prec_thr, color="steelblue", label=f"THRESHOLD (AP={ap_thr:.3f})")

    prec_fin, rec_fin, _ = precision_recall_curve(y_final, y_prob_final)
    ap_fin = average_precision_score(y_final, y_prob_final)
    ax1.plot(rec_fin, prec_fin, color="tomato", linestyle="--", label=f"FINAL (AP={ap_fin:.3f})")
    ax1.axhline(
        OPT_MIN_PRECISION,
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label=f"{OPT_MIN_PRECISION:.0%} precision target",
    )
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    thr_sweep = np.arange(0.01, 1.0, 0.01)
    prec_sw = []
    rec_sw = []
    f1_sw = []
    for t in thr_sweep:
        p_ = (y_prob_threshold >= t).astype(int)
        prec_sw.append(precision_score(y_threshold, p_, zero_division=0))
        rec_sw.append(recall_score(y_threshold, p_, zero_division=0))
        f1_sw.append(f1_score(y_threshold, p_, zero_division=0))

    ax2.plot(thr_sweep, prec_sw, color="tomato", label="Precision")
    ax2.plot(thr_sweep, rec_sw, color="steelblue", label="Recall")
    ax2.plot(thr_sweep, f1_sw, color="green", linestyle="--", label="F1")
    ax2.axvline(best_threshold, color="black", linewidth=1.2, label=f"Best thr={best_threshold:.2f}")
    ax2.axvline(f1_thresh, color="purple", linestyle=":", linewidth=1.0, label=f"F1-opt={f1_thresh:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("THRESHOLD-Set — Threshold vs Metrics")
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()

    print("\n── THRESHOLD-Set (raw) Sweep ──")
    rows_test = []
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (y_prob_threshold >= t).astype(int)
        n_sig = preds.sum()
        tp = int((preds * y_threshold).sum())
        fp = int(preds.sum() - tp)
        prec = precision_score(y_threshold, preds, zero_division=0)
        rec = recall_score(y_threshold, preds, zero_division=0)
        f1 = f1_score(y_threshold, preds, zero_division=0)
        rows_test.append(
            {
                "Thr": f"{t:.2f}",
                "Prec": f"{prec:.2%}",
                "Rec": f"{rec:.2%}",
                "F1": f"{f1:.3f}",
                "Signals": n_sig,
                "TP": tp,
                "FP": fp,
            }
        )
    print(pd.DataFrame(rows_test).to_string(index=False))

    apply_signal_filters = cfg_mod.apply_signal_filters

    print("\n── FINAL (filtered) Threshold Sweep ──")
    rows_final = []
    for t in np.arange(0.05, 0.96, 0.05):
        sig_dates_by_ticker = {}
        for ticker in final_tickers:
            sub = df_final[df_final["ticker"] == ticker]
            sig_dates_by_ticker[ticker] = apply_signal_filters(sub, t)

        n_sig = 0
        tp = 0
        fp = 0
        for ticker, sig_dates in sig_dates_by_ticker.items():
            sub = df_final[df_final["ticker"] == ticker]
            for d in sig_dates:
                row = _rows_for_signal_calendar_day(sub, d)
                if row.empty:
                    continue
                n_sig += 1
                if row["target"].values[0] == 1:
                    tp += 1
                else:
                    fp += 1
        if n_sig == 0:
            continue
        prec = tp / n_sig
        check = "\u2713" if prec >= OPT_MIN_PRECISION else ""
        rows_final.append(
            {
                "Thr": f"{t:.2f}",
                "Prec": f"{prec:.2%}",
                "Signals": n_sig,
                "TP": tp,
                "FP": fp,
                f"≥{OPT_MIN_PRECISION:.0%}": check,
            }
        )
    print(pd.DataFrame(rows_final).to_string(index=False))

    print("\n── Result Summary ──")
    print(
        f"  Hinweis: Die Sweep-Tabellen oben zeigen Precision für *jedes* t (was-wäre-wenn). "
        f"Die Zeilen unten beziehen sich nur auf best_threshold={best_threshold:.3f} aus Phase 5 "
        f"(niedrigstes t mit roher Precision ≥ OPT_MIN_PRECISION={OPT_MIN_PRECISION:.0%}, sonst Fallback). "
        f"THRESHOLD (raw) = ohne Konsekutiv-/Cooldown-Filter; FINAL (filtered) = mit Filter."
    )
    for label, y_true, y_prob_arr, tickers_list, df_part in [
        ("THRESHOLD (raw)", y_threshold, y_prob_threshold, threshold_tickers, df_threshold),
        ("FINAL (filtered)", y_final, y_prob_final, final_tickers, df_final),
    ]:
        if label.startswith("THRESHOLD"):
            preds = (y_prob_arr >= best_threshold).astype(int)
            n_sig = preds.sum()
            tp = int((preds * y_true).sum())
            fp = n_sig - tp
            prec = precision_score(y_true, preds, zero_division=0)
        else:
            n_sig = tp = fp = 0
            for t in tickers_list:
                sub = df_part[df_part["ticker"] == t]
                for d in apply_signal_filters(sub, best_threshold):
                    row = _rows_for_signal_calendar_day(sub, d)
                    if row.empty:
                        continue
                    n_sig += 1
                    if row["target"].values[0] == 1:
                        tp += 1
                    else:
                        fp += 1
            prec = tp / n_sig if n_sig > 0 else 0.0

        status = "PASS \u2713" if prec >= OPT_MIN_PRECISION else "MISS \u2717"
        print(
            f"  {label:20s} | Signals={n_sig:4d} | TP={tp:4d} | FP={fp:4d} | "
            f"Precision={prec:.1%} | {status}  (gate: ≥{OPT_MIN_PRECISION:.0%})"
        )
