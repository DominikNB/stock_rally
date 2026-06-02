"""Validierung Badge 0/1/2 (GLD + Makro-Vol-Spike)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.red_regime_summary import red_quality_fields
from lib.red_signal_quality import NEWS_MACRO_VOL_SPIKE_COL
from scripts._scratch_validate_red_quality_tiers import MIN_DELTA, MIN_N, VIX_RED, _assign_dataset

OUT = ROOT / "data" / "_scratch_red_badge_gld_spike.json"


def main() -> None:
    mc = pd.read_csv(ROOT / "data" / "master_complete.csv")
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    sh = pd.read_parquet(
        ROOT / "data" / "feature_shards_news" / "news_tag_3_20_10.parquet",
        columns=["Date", "ticker", NEWS_MACRO_VOL_SPIKE_COL],
    )
    sh["Date"] = pd.to_datetime(sh["Date"]).dt.normalize()
    red = red.merge(sh.drop_duplicates(["Date", "ticker"]), on=["Date", "ticker"], how="left")

    scores = []
    for _, r in red.iterrows():
        f = red_quality_fields(r.to_dict())
        scores.append(f.get("quality_red"))
    red["quality_red"] = scores

    results = []
    for cid, label, fn in [
        ("badge_ge1_vs_0", "Score >=1 vs 0", lambda s: (s[s.quality_red >= 1], s[s.quality_red == 0])),
        ("badge_eq2_vs_0", "Score ==2 vs 0", lambda s: (s[s.quality_red == 2], s[s.quality_red == 0])),
        ("badge_eq2_vs_le1", "Score ==2 vs <=1", lambda s: (s[s.quality_red == 2], s[s.quality_red <= 1])),
    ]:
        row = {"id": cid, "label": label}
        for ds in ["META+THR", "FINAL"]:
            sub = red[red["dataset"] == ds]
            hi, lo = fn(sub)
            hi, lo = hi.dropna(subset=["ret"]), lo.dropna(subset=["ret"])
            if len(hi) < MIN_N or len(lo) < MIN_N:
                row[f"d_{ds.lower().replace('+','_')}_pp"] = None
                row[f"n_{ds.lower().replace('+','_')}_hi"] = len(hi)
                row[f"n_{ds.lower().replace('+','_')}_lo"] = len(lo)
            else:
                d = float(hi["ret"].mean() - lo["ret"].mean())
                row[f"d_{ds.lower().replace('+','_')}_pp"] = round(100 * d, 3)
                row[f"n_{ds.lower().replace('+','_')}_hi"] = len(hi)
                row[f"n_{ds.lower().replace('+','_')}_lo"] = len(lo)
        dt = row.get("d_meta_thr_pp")
        df = row.get("d_final_pp")
        row["oos_ok"] = (
            dt is not None
            and df is not None
            and dt * df > 0
            and abs(dt) >= 100 * MIN_DELTA
            and abs(df) >= 100 * MIN_DELTA
        )
        row["implement"] = bool(row["oos_ok"] and dt > 0 and df > 0)
        results.append(row)
        print(row)

    fin = red[red["dataset"] == "FINAL"]
    dist = fin["quality_red"].value_counts().sort_index().to_dict()
    payload = {"final_badge_dist": dist, "results": results}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFINAL Verteilung: {dist}")
    print(f"Geschrieben: {OUT}")


if __name__ == "__main__":
    main()
