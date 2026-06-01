"""What-if: Wirkung verschiedener Zusatzfilter auf das FINAL-OOS (master_complete.csv).

KEINE Pipeline-Aenderung — reine Simulation auf den bereits erzeugten Signalen.
Zeigt pro Regel: verbleibende Signale, Win-Rate, Mean/Median, und wie viele Tage
noch abgedeckt sind (Coverage), damit wir nicht alles wegfiltern.
"""
import numpy as np
import pandas as pd

mc = pd.read_csv("data/master_complete.csv")
mc = mc.dropna(subset=["ret_mean_5"]).copy()
mc["ret_mean_5"] = mc["ret_mean_5"].astype(float)
mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce")
for c in ["regime_vix_level", "dist_from_20d_high_pct", "volume_zscore_20d",
          "prob", "trend_efficiency_20d", "momentum_60d"]:
    mc[c] = pd.to_numeric(mc[c], errors="coerce")

n0 = len(mc)
days0 = mc["Date"].dt.normalize().nunique()


def rep(label, sub):
    if len(sub) == 0:
        print(f"{label:<42} n=   0  (alles weggefiltert)")
        return
    r = sub["ret_mean_5"].astype(float)
    win = 100 * (r > 0).mean()
    mean = 100 * r.mean()
    med = 100 * r.median()
    days = sub["Date"].dt.normalize().nunique()
    crash = 100 * (r < -0.03).mean()
    keep = 100 * len(sub) / n0
    # Gesamt-"Ertrag" als Summe (Proxy fuer eingesammelten Edge)
    tot = r.sum() * 100
    print(f"{label:<42} n={len(sub):4d} ({keep:4.0f}%)  win={win:5.1f}%  "
          f"mean={mean:+.2f}%  med={med:+.2f}%  crash={crash:4.1f}%  "
          f"sumRet={tot:+6.0f}pp  tage={days}")


print(f"{'REGEL':<42} {'Signale':>8}        Win      Mean    Median   Crash<-3%  SummeRet  Tage")
print("-" * 110)
rep("BASIS (keine Zusatzfilter)", mc)
print()

# 1. VIX-Filter (verschiedene Schwellen)
print("--- Einzelfilter: VIX-Mindestlevel (nur Signale wenn VIX >= X) ---")
for thr in [16, 18, 20, 22]:
    rep(f"VIX >= {thr}", mc[mc["regime_vix_level"] >= thr])
print()

# 2. Earnings
print("--- Einzelfilter: Earnings-Fenster ---")
if "earnings_in_3_15_bday_window" in mc.columns:
    rep("kein Earnings in 3-15 Bday", mc[mc["earnings_in_3_15_bday_window"] != True])
print()

# 3. dist_from_20d_high Sperrzone -10..-5%
print("--- Einzelfilter: dist_20d_high Sperrzone (-10..-5%) raus ---")
mask_trap = (mc["dist_from_20d_high_pct"] >= -0.10) & (mc["dist_from_20d_high_pct"] <= -0.05)
rep("ohne -10..-5% Zone", mc[~mask_trap])
print()

# 4. Volume-Spike raus
print("--- Einzelfilter: hoher volume_zscore raus ---")
rep("volume_zscore_20d <= 1.0", mc[mc["volume_zscore_20d"] <= 1.0])
print()

# 5. Schwache Sektoren raus
print("--- Einzelfilter: schwache Sektoren raus ---")
weak = ["communication_services", "real_estate", "utilities"]
rep("ohne comm/realestate/utilities", mc[~mc["sector"].isin(weak)])
print()

# 6. Kombi-Varianten
print("=== KOMBINATIONEN ===")
base_ok = pd.Series(True, index=mc.index)

# Kombi A: VIX>=18 + kein Earnings
mA = (mc["regime_vix_level"] >= 18) & (mc["earnings_in_3_15_bday_window"] != True)
rep("A) VIX>=18 & kein Earnings", mc[mA])

# Kombi B: VIX>=18 + kein Earnings + ohne Trap-Zone
mB = mA & ~mask_trap
rep("B) A + ohne -10..-5% Zone", mc[mB])

# Kombi C: B + volume<=1.0
mC = mB & (mc["volume_zscore_20d"] <= 1.0)
rep("C) B + volume_z<=1.0", mc[mC])

# Kombi D: C + ohne schwache Sektoren
mD = mC & ~mc["sector"].isin(weak)
rep("D) C + ohne schwache Sektoren", mc[mD])

# Kombi E: nur VIX>=20 (aggressiv)
rep("E) nur VIX>=20", mc[mc["regime_vix_level"] >= 20])

# Kombi F: VIX>=16 (mild) + kein Earnings + ohne Trap
mF = (mc["regime_vix_level"] >= 16) & (mc["earnings_in_3_15_bday_window"] != True) & ~mask_trap
rep("F) VIX>=16 & kein Earn & ohne Trap", mc[mF])

print()
print("Hinweis: 'sumRet' = Summe aller Signal-Returns in pp = grober Proxy fuer den")
print("insgesamt eingesammelten Edge (mehr Signale x Edge). Tradeoff Qualitaet vs Menge.")
