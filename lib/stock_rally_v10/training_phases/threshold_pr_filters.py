"""Phase 15: PR-Kurven und kompakte Filter-Diagnose am produktiven Threshold."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
)

from lib.stock_rally_v10.equity_classification import _ticker_symbol_str
from lib.stock_rally_v10.holdout_plot import (
    _rows_for_signal_calendar_day,
    apply_signal_filters as apply_signal_filters_from_cfg,
    diagnose_signal_filter_stages,
)


def _unique_tickers_from_df(df: pd.DataFrame) -> list[str]:
    """Ticker aus dem DataFrame (normalisiert); Plausibilitäts-Filter mit Roh-Fallback."""
    if df.empty or "ticker" not in df.columns:
        return []
    raw_vals = df["ticker"].dropna().unique()
    norm_set = {_ticker_symbol_str(x) for x in raw_vals}
    norm_set.discard("")
    plausible = sorted(x for x in norm_set if x.lower() != "nan" and x not in ("0.0", "0"))
    if plausible:
        return plausible
    fallback = sorted(norm_set)
    if fallback:
        print(
            "[FINAL] KRITISCH: Nach Plausibilitätsfilter keine Ticker — nutze Roh-Unique "
            f"(n={len(fallback)}, z. B. {fallback[:8]}). Prüfen, ob `ticker` durch Merge/Join beschädigt wurde.",
            flush=True,
        )
        return fallback
    return []


def _effective_final_tickers(df_final: pd.DataFrame, cfg_final_tickers: Any) -> list[str]:
    """
    Quelle der Wahrheit für FINAL-Operationen ist ``df_final``.
    ``cfg.final_tickers`` kann leer oder veraltet sein (z. B. nach manuellen Edits / Artefakt).
    """
    from_df = _unique_tickers_from_df(df_final)
    if cfg_final_tickers is None:
        cfg_list: list[str] = []
    elif isinstance(cfg_final_tickers, (str, bytes)):
        cfg_list = [str(cfg_final_tickers)]
    else:
        try:
            cfg_list = [str(x).strip() for x in list(cfg_final_tickers)]
        except TypeError:
            cfg_list = [str(cfg_final_tickers).strip()]
    cfg_list = [x for x in cfg_list if x and x.lower() != "nan" and x not in ("0.0", "0")]
    df_set = set(from_df)
    if not cfg_list:
        if from_df:
            print(
                f"  [FINAL] Hinweis: cfg.final_tickers leer — nutze {len(from_df)} Ticker aus df_final.",
                flush=True,
            )
        return from_df
    inter = [t for t in cfg_list if t in df_set]
    if not inter and from_df:
        miss = sorted(set(cfg_list) - df_set)[:12]
        print(
            f"  [FINAL] Warnung: kein Overlap cfg.final_tickers ({len(cfg_list)}) mit df_final-Tickern "
            f"({len(from_df)}). Nutze df_final. Fehlende (Beispiel): {miss}",
            flush=True,
        )
        return from_df
    if len(inter) < len(cfg_list):
        print(
            f"  [FINAL] Hinweis: {len(cfg_list) - len(inter)} cfg-Ticker fehlen in df_final — "
            f"arbeite mit {len(inter)} Treffern.",
            flush=True,
        )
    result = inter or from_df
    if not result and len(df_final) > 0:
        print(
            "  [FINAL] KRITISCH: weder cfg-Overlap noch df-Tickerliste — Phase 15/16 können nicht sinnvoll laufen.",
            flush=True,
        )
    return result


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
    final_tickers_cfg = cfg_mod.final_tickers
    final_tickers = _effective_final_tickers(df_final, final_tickers_cfg)
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

    bins = np.linspace(0.0, 1.0, 41)
    ax2.hist(
        y_prob_threshold,
        bins=bins,
        alpha=0.45,
        color="steelblue",
        label=f"THRESHOLD probs (n={len(y_prob_threshold):,})",
        density=True,
    )
    ax2.hist(
        y_prob_final,
        bins=bins,
        alpha=0.35,
        color="tomato",
        label=f"FINAL probs (n={len(y_prob_final):,})",
        density=True,
    )
    ax2.axvline(best_threshold, color="black", linewidth=1.2, label=f"Best thr={best_threshold:.2f}")
    ax2.axvline(f1_thresh, color="purple", linestyle=":", linewidth=1.0, label=f"Ref thr={f1_thresh:.2f}")
    ax2.set_xlabel("Model probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Probability Distribution @ produktivem Threshold")
    ax2.legend()
    ax2.set_xlim([0, 1])
    plt.tight_layout()
    plt.show()

    apply_signal_filters = cfg_mod.apply_signal_filters

    print("\n── Result Summary ──")
    print(
        f"  Produktiv bewertet wird nur FILTERED bei best_threshold={best_threshold:.3f} "
        "aus Meta-Optuna (inkl. Konsekutiv/Cooldown/Anti-Peak/RSI)."
    )
    n_sig = tp = fp = 0
    for t in final_tickers:
        sub = df_final[df_final["ticker"].astype(str).str.strip() == str(t)]
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
        f"  FINAL (filtered)     | Signals={n_sig:4d} | TP={tp:4d} | FP={fp:4d} | "
        f"Precision={prec:.1%} | {status}  (gate: ≥{OPT_MIN_PRECISION:.0%})"
    )

    # ── Diagnose: wo verschwinden FINAL-Signale? (aggregiert über alle final_tickers)
    _dt_thr = pd.to_datetime(df_threshold["Date"])
    _dt_fin = pd.to_datetime(df_final["Date"])
    print(
        f"\n── DEBUG FINAL vs THRESHOLD (best_threshold={best_threshold:.3f}) ──\n"
        f"  df_threshold: {len(df_threshold):,} Zeilen, "
        f"prob≥thr: {int((df_threshold['prob'].values >= best_threshold).sum()):,}  "
        f"({_dt_thr.min().date()} … {_dt_thr.max().date()})\n"
        f"  df_final:     {len(df_final):,} Zeilen, "
        f"prob≥thr: {int((df_final['prob'].values >= best_threshold).sum()):,}  "
        f"({_dt_fin.min().date()} … {_dt_fin.max().date()})"
    )
    _rw = getattr(cfg_mod, "rsi_w", None)
    if _rw is None:
        _rw = int(cfg_mod.SEED_PARAMS.get("rsi_window", 14))
    else:
        _rw = int(_rw)
    _agg = {
        "n_rows": 0,
        "n_raw_prob_ge_thr": 0,
        "n_consec_slots": 0,
        "n_after_cooldown_pre_peak": 0,
        "n_killed_by_peak_or_rsi": 0,
        "n_killed_by_vol_stress": 0,
        "n_killed_by_blue_sky_weak_volume": 0,
        "n_final_signals": 0,
    }
    _tickers_no_close = 0
    try:
        _n_cfg_ft = len(final_tickers_cfg)  # type: ignore[arg-type]
    except TypeError:
        _n_cfg_ft = 0 if final_tickers_cfg is None else 1
    print(
        f"  Diagnose-Schleife über {len(final_tickers)} Ticker "
        f"(cfg.final_tickers: {_n_cfg_ft} Einträge vor Normalisierung).",
        flush=True,
    )
    for t in final_tickers:
        sub = df_final[df_final["ticker"].astype(str).str.strip() == str(t)]
        if sub.empty:
            continue
        if "close" not in sub.columns:
            _tickers_no_close += 1
        st = diagnose_signal_filter_stages(
            sub,
            float(best_threshold),
            int(cfg_mod.CONSECUTIVE_DAYS),
            int(cfg_mod.SIGNAL_COOLDOWN_DAYS),
            rsi_window=_rw,
            signal_skip_near_peak=bool(cfg_mod.SIGNAL_SKIP_NEAR_PEAK),
            peak_lookback_days=int(cfg_mod.PEAK_LOOKBACK_DAYS),
            peak_min_dist_from_high_pct=float(cfg_mod.PEAK_MIN_DIST_FROM_HIGH_PCT),
            signal_max_rsi=getattr(cfg_mod, "SIGNAL_MAX_RSI", None),
            signal_max_vol_stress_z=getattr(cfg_mod, "SIGNAL_MAX_VOL_STRESS_Z", None),
            signal_min_blue_sky_volume_z=getattr(cfg_mod, "SIGNAL_MIN_BLUE_SKY_VOLUME_Z", None),
            mult_final_threshold_1=getattr(cfg_mod, "MULT_FINAL_THRESHOLD_1", 1.0),
            mult_final_threshold_2=getattr(cfg_mod, "MULT_FINAL_THRESHOLD_2", 1.0),
            mult_final_threshold_3=getattr(cfg_mod, "MULT_FINAL_THRESHOLD_3", 1.0),
            dyn_vvix_trigger=getattr(cfg_mod, "DYN_VVIX_TRIGGER", 8.2),
            dyn_rsi_trigger=getattr(cfg_mod, "DYN_RSI_TRIGGER", 75.0),
            dyn_bb_pband_trigger=getattr(cfg_mod, "DYN_BB_PBAND_TRIGGER", 1.02),
        )
        for k in _agg:
            _agg[k] += int(st[k])
    print(
        "  Aggregat FINAL (Summe über Ticker): "
        f"Zeilen={_agg['n_rows']:,} | "
        f"raw prob≥thr={_agg['n_raw_prob_ge_thr']:,} | "
        f"Konsekutiv-Slots={_agg['n_consec_slots']:,} | "
        f"nach Cooldown (pre Anti-Peak)={_agg['n_after_cooldown_pre_peak']:,} | "
        f"durch Anti-Peak/RSI verworfen={_agg['n_killed_by_peak_or_rsi']:,} | "
        f"durch Vol-Stress verworfen={_agg['n_killed_by_vol_stress']:,} | "
        f"durch BlueSky+LowVol verworfen={_agg['n_killed_by_blue_sky_weak_volume']:,} | "
        f"End-Signale={_agg['n_final_signals']:,}"
    )
    if _tickers_no_close:
        print(f"  Warnung: {_tickers_no_close} Ticker ohne 'close' in df_final (Anti-Peak ausgeschaltet).")
    if _agg["n_raw_prob_ge_thr"] == 0:
        print(
            "  → Im FINAL-Zeitraum keine einzige Zeile mit prob≥threshold — "
            "nicht die Filter, sondern Modell/Schwelle vs. diese Periode prüfen."
        )
    elif _agg["n_final_signals"] == 0:
        if _agg["n_after_cooldown_pre_peak"] == 0:
            print(
                "  → Alle Kandidaten sterben bei Konsekutiv/Cooldown "
                f"(CONSECUTIVE_DAYS={cfg_mod.CONSECUTIVE_DAYS}, "
                f"SIGNAL_COOLDOWN_DAYS={cfg_mod.SIGNAL_COOLDOWN_DAYS})."
            )
        elif _agg["n_killed_by_peak_or_rsi"] >= _agg["n_after_cooldown_pre_peak"] > 0:
            print(
                "  → Nach Cooldown vorhandene Slots werden fast alle durch Anti-Peak/RSI entfernt. "
                "Zum Test: SIGNAL_SKIP_NEAR_PEAK=False oder PEAK_MIN_DIST_FROM_HIGH_PCT senken."
            )
