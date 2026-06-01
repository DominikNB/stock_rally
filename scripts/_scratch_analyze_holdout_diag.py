"""Scratch: parse FINAL holdout signal_target_diag from terminal export."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TEXT = Path(
    r"C:\Users\HP\.cursor\projects\c-Python-projects-stock-rally\uploads"
    r"\_c-Python-projects-stock-rally_3-L1-L854-0.txt"
).read_text(encoding="utf-8")
lines = TEXT.splitlines()

rows1 = []
for ln in lines:
    if ln.startswith("Signale mit target=0"):
        break
    m = re.match(r"^\s*(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$", ln)
    if m:
        rows1.append(
            dict(
                ticker=m.group(1),
                n=int(m.group(2)),
                on_target=int(m.group(3)),
                on_target_pct=float(m.group(4)),
            )
        )

g = pd.DataFrame(rows1)
late_start = (
    lines.index(
        "Signale mit target=0, rally=1 (spät in grüner Phase) — Tage seit Rally-Start:"
    )
    + 2
)
rows2 = []
for ln in lines[late_start:]:
    m = re.match(r"^\s*(\S+)\s+(\d{4}-\d{2}-\d{2})\s+([\d.]+)\s*$", ln)
    if m:
        rows2.append(dict(ticker=m.group(1), date=m.group(2), days=float(m.group(3))))
late = pd.DataFrame(rows2)

MIN_N = 5
gr = g[g["n"] >= MIN_N].copy()
worst = gr.nsmallest(10, "on_target_pct")
best = gr.nlargest(10, "on_target_pct")

lt = late.groupby("ticker").agg(
    n_late=("days", "size"),
    mean_days=("days", "mean"),
    median_days=("days", "median"),
    max_days=("days", "max"),
).reset_index()

total_n = int(g["n"].sum())
total_on = int(g["on_target"].sum())
print("=== GESAMT ===")
print(f"Signale gesamt: {total_n}, on_target: {total_on} ({100 * total_on / total_n:.1f}%)")
print(
    f"Late rally=1 target=0: {len(late)} ({100 * len(late) / total_n:.1f}% aller Signale)"
)
print(
    "days_from_rally_start: "
    f"median={late['days'].median():.1f}, mean={late['days'].mean():.1f}, "
    f"p90={late['days'].quantile(0.9):.1f}, max={late['days'].max():.0f}"
)
for label, mask in [
    ("<=5 Tage", late["days"] <= 5),
    ("6-15 Tage", (late["days"] >= 6) & (late["days"] <= 15)),
    (">15 Tage", late["days"] > 15),
    (">30 Tage", late["days"] > 30),
]:
    print(f"  {label}: {int(mask.sum())} ({100 * mask.mean():.1f}%)")

print(f"\n=== SCHLECHTESTE 10 (n>={MIN_N}) ===")
print(worst.to_string(index=False))
print(f"\n=== BESTE 10 (n>={MIN_N}) ===")
print(best.to_string(index=False))

print("\n=== LATE-PROFIL SCHLECHTESTE 10 ===")
for _, r in worst.iterrows():
    sub = late[late.ticker == r.ticker]
    if len(sub):
        med = float(sub.days.median())
        mx = float(sub.days.max())
        print(
            f"{r.ticker}: on_target_pct={r.on_target_pct}%, n={int(r.n)}, "
            f"late={len(sub)}, late_med={med:.0f}, late_max={mx:.0f}"
        )
    else:
        print(f"{r.ticker}: on_target_pct={r.on_target_pct}%, n={int(r.n)}, late=0")

print("\n=== EXTREM SPAET (>30d), mind. 2 Signale ===")
ext = late[late.days > 30].groupby("ticker").size().sort_values(ascending=False)
for t, c in ext.items():
    if c >= 2:
        hit = g.loc[g.ticker == t, "on_target_pct"]
        ot = float(hit.iloc[0]) if len(hit) else float("nan")
        print(f"  {t}: {int(c)}x >30d, on_target_pct={ot}%")

merged = g.merge(lt, on="ticker", how="left")
merged["n_late"] = merged["n_late"].fillna(0).astype(int)
merged["late_share"] = merged["n_late"] / merged["n"].replace(0, np.nan)
subm = merged[merged.n >= MIN_N]
print("\n=== KORRELATION (Ticker mit n>=5) ===")
print(f"on_target_pct vs late_share: r={subm['on_target_pct'].corr(subm['late_share']):.3f}")
pos = subm[subm.n_late > 0]
print(
    "on_target_pct vs mean_days (nur Ticker mit late>0): "
    f"r={pos['on_target_pct'].corr(pos['mean_days']):.3f}"
)
