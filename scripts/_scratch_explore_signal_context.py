"""
Breite Exploration: Welche Kontext-Dimensionen trennen Signale — speziell innerhalb VIX-rot?

Nutzt:
  - data/_scratch_meta_thr_final_signals.csv (META/THRESHOLD/FINAL, ret, vix, dist_high)
  - data/master_complete.csv (angereicherte Spalten, Schwerpunkt FINAL)
  - lib TICKER_TO_SECTOR fuer Sektor-Label auf allen Signalen

Kein Filter-Vorschlag ohne META+THR vs. FINAL-Richtungscheck.

  python scripts/_scratch_explore_signal_context.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg

SIGNALS_CSV = ROOT / "data" / "_scratch_meta_thr_final_signals.csv"
MASTER_CSV = ROOT / "data" / "master_complete.csv"
VIX_RED = 20.0
MIN_N = 25
GOOD_RET = 0.02


def _stats(r: pd.Series) -> str:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 5:
        return f"n={len(r)}"
    return (
        f"n={len(r):4d} mean={100*r.mean():+.2f}% win={100*(r>0).mean():.0f}% "
        f"good2={100*(r>GOOD_RET).mean():.0f}%"
    )


def _oos_delta(tune: pd.DataFrame, fin: pd.DataFrame, col: str, hi_if_ge_median: bool = True) -> tuple[bool, float, float]:
    """Split by median of col within each subset."""
    def _d(sub: pd.DataFrame) -> float:
        if sub.empty or col not in sub.columns:
            return float("nan")
        s = sub.dropna(subset=[col, "ret"])
        if len(s) < MIN_N * 2:
            return float("nan")
        med = float(s[col].median())
        hi = s[s[col] >= med] if hi_if_ge_median else s[s[col] < med]
        lo = s[s[col] < med] if hi_if_ge_median else s[s[col] >= med]
        if len(hi) < MIN_N or len(lo) < MIN_N:
            return float("nan")
        return float(hi["ret"].mean() - lo["ret"].mean())

    dt, df = _d(tune), _d(fin)
    ok = np.isfinite(dt) and np.isfinite(df) and dt * df > 0 and abs(dt) >= 0.001
    return bool(ok), dt, df


def _report_categorical(df: pd.DataFrame, col: str, title: str) -> None:
    print(f"\n--- {title} ---")
    for label, sub in [("META+THR", df[df["dataset"].isin(["META", "THRESHOLD"])]), ("FINAL", df[df["dataset"] == "FINAL"])]:
        d = sub.dropna(subset=[col, "ret"])
        if d.empty:
            continue
        print(f"  [{label}]")
        for cat, g in d.groupby(col, observed=True):
            if len(g) < 10:
                continue
            print(f"    {str(cat)[:28]:28} {_stats(g['ret'])}")


def _scan_numeric(df: pd.DataFrame, col: str, tune: pd.DataFrame, fin: pd.DataFrame, results: list) -> None:
    sub = df.dropna(subset=[col, "ret"])
    if len(sub) < MIN_N * 2:
        return
    med = float(sub[col].median())
    ok, dt, dfin = _oos_delta(tune, fin, col)
    results.append(
        {
            "feature": col,
            "rule": f">= median ({med:.4g})",
            "oos_ok": ok,
            "d_tune": dt,
            "d_fin": dfin,
            "n": len(sub),
        }
    )


def main() -> None:
    sig = pd.read_csv(SIGNALS_CSV)
    sig["Date"] = pd.to_datetime(sig["Date"]).dt.normalize()
    sig["ret"] = pd.to_numeric(sig["ret"], errors="coerce")
    sig["vix"] = pd.to_numeric(sig["vix"], errors="coerce")
    sig["sector"] = sig["ticker"].astype(str).map(cfg.TICKER_TO_SECTOR).fillna("unknown")
    sig["is_eu"] = sig["ticker"].astype(str).str.contains(
        r"\.(?:DE|AS|PA|MI|L|SW|ST|HE|F)$", regex=True
    )
    sig["is_red"] = sig["vix"] < VIX_RED
    sig["cal_month"] = sig["Date"].dt.month

    if MASTER_CSV.is_file():
        mc = pd.read_csv(MASTER_CSV)
        mc["Date"] = pd.to_datetime(mc["Date"]).dt.normalize()
        extra = [c for c in mc.columns if c not in sig.columns and c not in ("ticker", "Date")]
        sig = sig.merge(mc[["ticker", "Date"] + extra], on=["ticker", "Date"], how="left")

    tune = sig[sig["dataset"].isin(["META", "THRESHOLD"])]
    fin = sig[sig["dataset"] == "FINAL"]
    red = sig[sig["is_red"]]
    red_tune = tune[tune["is_red"]]
    red_fin = fin[fin["is_red"]]

    print("=" * 72)
    print("A) SEKTOR — Performance rot vs. nicht-rot (META+THR)")
    print("=" * 72)
    for sec, g in tune.groupby("sector"):
        if len(g) < 40:
            continue
        r = g[g["is_red"]]["ret"]
        nr = g[~g["is_red"]]["ret"]
        if len(r) < 15 or len(nr) < 15:
            continue
        print(
            f"  {sec:22} rot {_stats(r)}  |  nicht-rot {_stats(nr)}  "
            f"| delta_mean={100*(r.mean()-nr.mean()):+.2f}pp"
        )

    print("\n" + "=" * 72)
    print("B) SEKTOR — nur innerhalb rot (FINAL)")
    print("=" * 72)
    _report_categorical(red_fin, "sector", "Sektor (FINAL, VIX<20)")

    print("\n" + "=" * 72)
    print("C) BEREITS ENRICHED (master_complete) — innerhalb rot, FINAL")
    print("=" * 72)
    mc_cols = [
        "prob", "meta_prob_margin", "signals_same_day", "signals_same_sector_same_day",
        "sector_share_same_day", "cluster_mean_corr_60d", "momentum_20d", "momentum_60d",
        "ret_vs_spy_5d", "ret_vs_sector_5d", "ret_vs_spy_20d", "ret_vs_sector_20d",
        "alpha_mkt_5d", "alpha_sec_5d", "beta_mkt_60d", "volatility_20d", "bb_width_20",
        "trend_efficiency_20d", "volume_zscore_20d", "open_gap_pct", "dist_from_20d_high_pct",
        "regime_vix_z_20d", "regime_spy_realvol_5d_ann", "regime_tnx_ret_5d",
        "adv_pctile_same_day", "prob_z_within_sector", "pct_rank_prob_same_day",
    ]
    results: list[dict] = []
    for col in mc_cols:
        if col not in red_fin.columns:
            continue
        _scan_numeric(red_fin, col, red_tune, red_fin, results)

    _report_categorical(red_fin, "liquidity_tier", "Liquiditaet (FINAL rot)")

    print("\n" + "=" * 72)
    print("D) SCAN — numeric splits (FINAL rot), OOS-Richtung vs. META+THR rot")
    print("=" * 72)
    results.sort(key=lambda x: (not x["oos_ok"], -abs(float(x.get("d_tune") or 0))))
    for row in results[:20]:
        mark = "OK" if row["oos_ok"] else "--"
        print(
            f"  [{mark}] {row['feature']:28} {row['rule'][:22]:22} "
            f"d_thr={100*float(row['d_tune'] or 0):+.2f}pp d_fin={100*float(row['d_fin'] or 0):+.2f}pp"
        )

    print("\n" + "=" * 72)
    print("E) STRUKTUR (alle Datensaetze, rot)")
    print("=" * 72)
    _report_categorical(red, "is_eu", "EU-Ticker (.DE etc.)")
    _report_categorical(red, "cal_month", "Kalendermonat")

    print("\n" + "=" * 72)
    print("F) IDEEN NOCH NICHT IM CSV (manuell / Feature-Shard noetig)")
    print("=" * 72)
    ideas = [
        "news_sec_*_tone / news_macro_* (Sektor- vs. Makro-News am Signaltag)",
        "Sektor-Rally-Breadth: Anteil Sektor-Ticker ueber MA50",
        "Peer-Crowding: viele Sektor-Signale gleicher Richtung am Tag",
        "Faktor-Residual: Return vs. Fama-French / Sektor-ETF multi-factor",
        "Options: Put/Call, IV-Rank (nicht im Repo)",
        "Insider / Buybacks (nicht im Repo)",
        "Makro-Kalender: CPI/FOMC-Fenster (nicht im Repo)",
        "Cross-Asset: Gold, Oil, EUR/USD 5d vor Signal",
        "Regime x Sektor Interaktion (z.B. rot + Technology + pos. sector_ret_3d)",
    ]
    for i, t in enumerate(ideas, 1):
        print(f"  {i}. {t}")


if __name__ == "__main__":
    main()
