"""Diagnose: ATR-Labels vs. realisierte OOS-Renditen."""
from __future__ import annotations

import pandas as pd

mc = pd.read_csv("data/master_complete.csv").dropna(subset=["ret_mean_5"]).copy()
mc["ret_mean_5"] = mc["ret_mean_5"].astype(float)
atr = pd.to_numeric(mc["avg_hl_range_pct_14d"], errors="coerce")
k = 1.5
rt_sig = k * atr
mc["_rt_sig"] = rt_sig
mc["_bucket"] = pd.cut(
    rt_sig, [0, 0.02, 0.03, 0.045, 0.06, 0.15], labels=["<=2%", "2-3%", "3-4.5%", "4.5-6%", ">6%"]
)

print("=== OOS ret_mean_5 nach simulierter Label-Schwelle (k=1.5 * ATR%) ===")
for b, g in mc.groupby("_bucket", observed=True):
    r = g["ret_mean_5"]
    print(f"  {b}: n={len(g):3d}  win={100*(r>0).mean():5.1f}%  mean={100*r.mean():+6.2f}%")

vol = pd.to_numeric(mc["volatility_20d"], errors="coerce")
mc["vq"] = pd.qcut(vol.rank(method="first"), 4, labels=["Q1 ruhig", "Q2", "Q3", "Q4 volatil"])
print("\n=== Nach Volatilitaets-Quartil (volatility_20d) ===")
for b, g in mc.groupby("vq", observed=True):
    r = g["ret_mean_5"]
    ap = pd.to_numeric(g["avg_hl_range_pct_14d"], errors="coerce").mean()
    print(f"  {b}: n={len(g):3d}  win={100*(r>0).mean():5.1f}%  mean={100*r.mean():+6.2f}%  atr~{100*ap:.2f}%")

print("\n=== Positive-Rate bei verschiedenen k (Simulation auf OOS-Signal-ATR) ===")
for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
    thr = k * atr
    # Anteil Signale deren Label-Schwelle unter alter 4.5% Haette
    print(f"  k={k:.1f}: median Schwelle={100*thr.median():.2f}%  share thr<3%={(thr<0.03).mean()*100:.0f}%  share thr<4.5%={(thr<0.045).mean()*100:.0f}%")
