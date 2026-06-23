"""What-if: neue Cross-Section-/Kontext-Filter auf OOS (master_complete.csv)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MC = ROOT / "data/master_complete.csv"
PRE = ROOT / "data/model_snapshots/pre_meta_optimization_20260614_210323/master_complete.csv"
TARGET_WIN = 57.6  # alter OOS-Stand


def _macro_false(s: pd.Series) -> pd.Series:
    return ~s.map(
        lambda v: str(v).strip().lower() in {"true", "1", "1.0", "yes"} or v is True or v == 1
    )


def load() -> pd.DataFrame:
    mc = pd.read_csv(MC)
    mc = mc.dropna(subset=["ret_mean_5"]).copy()
    mc["ret_mean_5"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce")
    for c in [
        "prob",
        "meta_prob_margin",
        "regime_vix_level",
        "ret_vs_spy_5d",
        "ret_vs_spy_20d",
        "ret_vs_sector_5d",
        "prob_zscore_same_day",
        "pct_rank_prob_same_day",
        "quality_red",
        "macro_event_within_2bd",
    ]:
        if c in mc.columns:
            mc[c] = pd.to_numeric(mc[c], errors="coerce")
    return mc


def rep(label: str, sub: pd.DataFrame, n0: int) -> dict:
    if sub.empty:
        print(f"{label:<48} n=    0  — alles weg")
        return {"label": label, "n": 0}
    r = sub["ret_mean_5"]
    win = 100 * (r > 0).mean()
    mean = 100 * r.mean()
    med = 100 * r.median()
    keep = 100 * len(sub) / n0
    tot = 100 * r.sum()
    hit = " ◀ ~alt" if win >= TARGET_WIN - 0.5 else ""
    print(
        f"{label:<48} n={len(sub):4d} ({keep:4.0f}%)  "
        f"win={win:5.1f}%{hit}  mean={mean:+.2f}%  med={med:+.2f}%  sum={tot:+6.0f}pp"
    )
    return {"label": label, "n": len(sub), "win": win, "mean": mean, "sum": tot}


def main() -> None:
    mc = load()
    n0 = len(mc)
    no_macro = _macro_false(mc["macro_event_within_2bd"])

    print(f"OOS-Basis: n={n0}  win={100*(mc['ret_mean_5']>0).mean():.1f}%  "
          f"mean={100*mc['ret_mean_5'].mean():+.2f}%  (Ziel Win ~{TARGET_WIN}%)")
    print()
    print(f"{'REGEL':<48} {'n':>6}  Win    Mean   (sumRet=Summe aller ret_mean_5)")
    print("-" * 95)

    results = []
    results.append(rep("BASIS (alle Signale)", mc, n0))
    print("\n--- Cross-Section / Kontext (neu) ---")
    results.append(rep("ohne Makro-Event (±2bd)", mc[no_macro], n0))
    if "quality_red" in mc.columns:
        for q in [0, 1, 2]:
            sub = mc[mc["quality_red"] <= q]
            results.append(rep(f"quality_red <= {q}", sub, n0))
    if "red_quality_tier" in mc.columns:
        for tier in ["green", "yellow"]:
            sub = mc[mc["red_quality_tier"].astype(str).str.lower() == tier]
            results.append(rep(f"red_quality_tier = {tier}", sub, n0))
        sub = mc[~mc["red_quality_tier"].astype(str).str.lower().eq("red")]
        results.append(rep("tier != red (gelb+grün)", sub, n0))

    print("\n--- Relative Stärke vs. SPY ---")
    for thr in [0.0, 0.01, 0.02, 0.03]:
        sub = mc[mc["ret_vs_spy_5d"] >= thr]
        results.append(rep(f"ret_vs_spy_5d >= {thr:.0%}", sub, n0))
    sub = mc[mc["ret_vs_spy_5d"].notna() & (mc["ret_vs_spy_5d"] >= mc["ret_vs_spy_5d"].median())]
    results.append(rep("ret_vs_spy_5d >= Median", sub, n0))

    print("\n--- Proba / Ranking ---")
    for thr in [0.92, 0.94, 0.96, 0.98]:
        sub = mc[mc["prob"] >= thr]
        results.append(rep(f"prob >= {thr}", sub, n0))
    if "meta_prob_margin" in mc.columns:
        sub = mc[mc["meta_prob_margin"] >= mc["meta_prob_margin"].quantile(0.75)]
        results.append(rep("meta_prob_margin top 25%", sub, n0))
    if "prob_zscore_same_day" in mc.columns:
        sub = mc[mc["prob_zscore_same_day"] >= 0]
        results.append(rep("prob_zscore_same_day >= 0", sub, n0))
        sub = mc[mc["prob_zscore_same_day"] >= 1.0]
        results.append(rep("prob_zscore_same_day >= 1", sub, n0))

    print("\n--- Kombinationen (praktikabel) ---")
    combos = [
        ("K1: kein Makro", no_macro),
        ("K2: kein Makro & ret_vs_spy_5d>=0", no_macro & (mc["ret_vs_spy_5d"] >= 0)),
        ("K3: kein Makro & ret_vs_spy_5d>=1%", no_macro & (mc["ret_vs_spy_5d"] >= 0.01)),
        ("K4: kein Makro & VIX>=18", no_macro & (mc["regime_vix_level"] >= 18)),
        ("K5: kein Makro & prob>=0.94", no_macro & (mc["prob"] >= 0.94)),
        ("K6: kein Makro & RSY>=0 & VIX>=16", no_macro & (mc["ret_vs_spy_5d"] >= 0) & (mc["regime_vix_level"] >= 16)),
        ("K7: tier!=red & ret_vs_spy_5d>=0", (~mc["red_quality_tier"].astype(str).str.lower().eq("red")) & (mc["ret_vs_spy_5d"] >= 0) if "red_quality_tier" in mc.columns else pd.Series(False, index=mc.index)),
        ("K8: kein Makro & quality_red<=1", no_macro & (mc["quality_red"] <= 1) if "quality_red" in mc.columns else no_macro),
        ("K9: kein Makro & RSY>=0 & prob>=0.92", no_macro & (mc["ret_vs_spy_5d"] >= 0) & (mc["prob"] >= 0.92)),
        ("K10: kein Makro & RSY>=1% & zscore>=0", no_macro & (mc["ret_vs_spy_5d"] >= 0.01) & (mc["prob_zscore_same_day"] >= 0)),
    ]
    for label, mask in combos:
        results.append(rep(label, mc[mask], n0))

    print("\n--- Beste Win-Rate bei mind. 100 Signalen ---")
    viable = [r for r in results if r.get("n", 0) >= 100 and "win" in r]
    viable.sort(key=lambda x: (-x["win"], -x["n"]))
    for r in viable[:8]:
        print(f"  {r['label']}: win={r['win']:.1f}%  n={r['n']}  mean={r['mean']:+.2f}%")

    if PRE.is_file():
        pre = pd.read_csv(PRE).dropna(subset=["ret_mean_5"])
        pre_r = pd.to_numeric(pre["ret_mean_5"], errors="coerce")
        print(f"\nReferenz PRE-Snapshot: n={len(pre_r)}  win={100*(pre_r>0).mean():.1f}%  mean={100*pre_r.mean():+.2f}%")


if __name__ == "__main__":
    main()
