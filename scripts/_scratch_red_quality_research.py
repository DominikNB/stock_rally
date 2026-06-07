"""Deep research: quality within VIX-red signals (IS META+THR vs OOS FINAL)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

MIN_N = 25
MASTER = ROOT / "data" / "master_complete.csv"


def _assign_dataset(mc: pd.DataFrame) -> pd.DataFrame:
    raw = yf.download(
        "SPY",
        start=mc["Date"].min(),
        end=(mc["Date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        progress=False,
    )
    cal = np.sort(pd.to_datetime(raw.index).tz_localize(None).normalize().unique())
    _, meta, thr, final = _split_calendar_four_way(
        cal,
        float(cfg.TIME_SPLIT_FRAC_BASE),
        float(cfg.TIME_SPLIT_FRAC_META),
        float(cfg.TIME_SPLIT_FRAC_THRESHOLD),
        int(cfg.TIME_PURGE_TRADING_DAYS),
    )
    mt = {pd.Timestamp(d) for d in meta} | {pd.Timestamp(d) for d in thr}
    fin = {pd.Timestamp(d) for d in final}

    def _lab(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d).normalize()
        if d in fin:
            return "FINAL"
        if d in mt:
            return "META+THR"
        return "OTHER"

    mc = mc.copy()
    mc["dataset"] = mc["Date"].map(_lab)
    return mc


def _delta_hi_lo(hi: pd.DataFrame, lo: pd.DataFrame) -> float:
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan")
    return float(hi["ret"].mean() - lo["ret"].mean())


def _stats(sub: pd.DataFrame, label: str) -> None:
    r = sub["ret"]
    print(
        f"  {label:14} n={len(sub):4d}  mean={100 * r.mean():+.2f}%  "
        f"win={( r > 0).mean():.1%}  good2={(r > 0.02).mean():.1%}"
    )


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < 20) & mc["ret"].notna()].copy()
    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]

    print("=" * 72)
    print("VIX-ROT: Signalqualität (ret_mean_5)")
    print(f"  gesamt={len(red)}  META+THR={len(tune)}  FINAL={len(fin)}  OTHER={len(red[red['dataset']=='OTHER'])}")
    _stats(red, "ALL rot")
    _stats(tune, "META+THR")
    _stats(fin, "FINAL")

    tests: list[tuple[str, callable]] = []

    def add(name: str, fn):
        tests.append((name, fn))

    add("vix_z < 0", lambda s: (s[s["regime_vix_z_20d"] < 0], s[s["regime_vix_z_20d"] >= 0]))
    add("vix3m/vix < 1.16", lambda s: (s[s["vix3m_vix_ratio"] < 1.16], s[s["vix3m_vix_ratio"] >= 1.16]))
    add("hhi < 0.35", lambda s: (s[s["sector_hhi_same_day"] < 0.35], s[s["sector_hhi_same_day"] >= 0.35]))
    add("dist_20d_high < -10%", lambda s: (s[s["dist_from_20d_high_pct"] < -0.10], s[s["dist_from_20d_high_pct"] >= -0.10]))
    add("dist_20d_high < -5%", lambda s: (s[s["dist_from_20d_high_pct"] < -0.05], s[s["dist_from_20d_high_pct"] >= -0.05]))
    add("ret_vs_spy_5d > 0", lambda s: (s[s["ret_vs_spy_5d"] > 0], s[s["ret_vs_spy_5d"] <= 0]))
    add("ret_vs_sector_5d > 0", lambda s: (s[s["ret_vs_sector_5d"] > 0], s[s["ret_vs_sector_5d"] <= 0]))
    add("alpha_mkt_5d > 0", lambda s: (s[s["alpha_mkt_5d"] > 0], s[s["alpha_mkt_5d"] <= 0]))
    add("alpha_sec_5d > 0", lambda s: (s[s["alpha_sec_5d"] > 0], s[s["alpha_sec_5d"] <= 0]))
    add("cluster_corr < median", lambda s: (
        s[s["cluster_mean_corr_60d"] < s["cluster_mean_corr_60d"].median()],
        s[s["cluster_mean_corr_60d"] >= s["cluster_mean_corr_60d"].median()],
    ))
    add("volume_z >= median", lambda s: (
        s[s["volume_zscore_20d"] >= s["volume_zscore_20d"].median()],
        s[s["volume_zscore_20d"] < s["volume_zscore_20d"].median()],
    ))
    add("news_diff > 0", lambda s: (s[s["news_sec_minus_macro_tone"] > 0], s[s["news_sec_minus_macro_tone"] <= 0]))
    add("news_diff >= median", lambda s: (
        s[s["news_sec_minus_macro_tone"] >= s["news_sec_minus_macro_tone"].median()],
        s[s["news_sec_minus_macro_tone"] < s["news_sec_minus_macro_tone"].median()],
    ))
    add("prob >= 0.80", lambda s: (s[s["prob"] >= 0.80], s[s["prob"] < 0.80]))
    add("momentum_20d >= median", lambda s: (
        s[s["momentum_20d"] >= s["momentum_20d"].median()],
        s[s["momentum_20d"] < s["momentum_20d"].median()],
    ))
    add("open_gap > 2%", lambda s: (s[s["open_gap_pct"] > 0.02], s[s["open_gap_pct"] <= 0.02]))
    add("liquidity_tier=ok", lambda s: (
        s[s["liquidity_tier"].astype(str).str.lower() == "ok"],
        s[s["liquidity_tier"].astype(str).str.lower() != "ok"],
    ))
    add("spy_realvol < median", lambda s: (
        s[s["regime_spy_realvol_5d_ann"] < s["regime_spy_realvol_5d_ann"].median()],
        s[s["regime_spy_realvol_5d_ann"] >= s["regime_spy_realvol_5d_ann"].median()],
    ))
    add("sector_ret_1d > 0", lambda s: (s[s["sector_ret_1d"] > 0], s[s["sector_ret_1d"] <= 0]))
    add("market_ret_1d > 0", lambda s: (s[s["market_ret_1d"] > 0], s[s["market_ret_1d"] <= 0]))

    print("\n" + "=" * 72)
    print(f"{'Hypothese':<28} {'d_tune':>8} {'d_fin':>8} {'OOS':>4}")
    print("-" * 72)
    rows = []
    for name, fn in tests:
        row = {"name": name}
        for lab, sub in [("tune", tune), ("fin", fin)]:
            s = sub.dropna(subset=["ret"])
            try:
                hi, lo = fn(s)
                hi = hi.dropna(subset=["ret"])
                lo = lo.dropna(subset=["ret"])
                row[lab] = _delta_hi_lo(hi, lo)
            except Exception:
                row[lab] = float("nan")
        dt, df = row.get("tune", np.nan), row.get("fin", np.nan)
        ok = np.isfinite(dt) and np.isfinite(df) and dt * df > 0 and abs(dt) >= 0.001
        rows.append((ok, dt, df, name))
        ds = f"{100 * dt:+.2f}pp" if np.isfinite(dt) else "   n/a"
        dfs = f"{100 * df:+.2f}pp" if np.isfinite(df) else "   n/a"
        print(f"{name:<28} {ds:>8} {dfs:>8} {'ja' if ok else 'nein':>4}")

    print("\n" + "=" * 72)
    print("Chip-Score (0–3 grüne aktuelle Chips)")
    for lab, sub in [("META+THR", tune), ("FINAL", fin), ("ALL", red)]:
        s = sub.copy()
        z = pd.to_numeric(s["regime_vix_z_20d"], errors="coerce")
        vr = pd.to_numeric(s["vix3m_vix_ratio"], errors="coerce")
        h = pd.to_numeric(s["sector_hhi_same_day"], errors="coerce")
        s["chip_score"] = (
            (z < 0).astype(int) + (vr < 1.16).astype(int) + (h < 0.35).astype(int)
        )
        print(f"  [{lab}]")
        for sc in range(4):
            g = s[s["chip_score"] == sc]
            if len(g) >= 15:
                _stats(g, f"score={sc}")

    print("\n" + "=" * 72)
    print("Sektor × rot (META+THR, mind. 40 Signale)")
    tune2 = tune.copy()
    tune2["sector"] = tune2["sector"].astype(str)
    for sec, g in tune2.groupby("sector"):
        if len(g) < 40:
            continue
        _stats(g, sec[:14])

    print("\n" + "=" * 72)
    print("Kombination: vix_z<0 UND rs_spy_5d>0 AND hhi<0.35")
    for lab, sub in [("META+THR", tune), ("FINAL", fin)]:
        s = sub.dropna(
            subset=["regime_vix_z_20d", "ret_vs_spy_5d", "sector_hhi_same_day", "ret"]
        )
        good = s[(s["regime_vix_z_20d"] < 0) & (s["ret_vs_spy_5d"] > 0) & (s["sector_hhi_same_day"] < 0.35)]
        rest = s[~s.index.isin(good.index)]
        if len(good) >= 10 and len(rest) >= 10:
            d = _delta_hi_lo(good, rest)
            print(f"  {lab}: good n={len(good)} mean={100*good.ret.mean():+.2f}% | rest n={len(rest)} mean={100*rest.ret.mean():+.2f}% | delta={100*d:+.2f}pp")


if __name__ == "__main__":
    main()
