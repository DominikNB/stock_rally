from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holdout.oos_performance import summarize_holdout_frame
import pandas as pd

PRE = ROOT / "data/model_snapshots/pre_meta_optimization_20260614_210323/master_complete.csv"
POST = ROOT / "data/master_complete.csv"


def _print_block(label: str, path: Path) -> dict:
    r = summarize_holdout_frame(pd.read_csv(path), source=str(path))
    rm = r["forward_returns"]["ret_mean_5"]
    red = r["context_tiers"]["red_macro_event"]
    yg = r["context_tiers"]["yellow_green_no_macro"]
    print(f"=== {label} ({path.name}) ===")
    print(
        f"  Signale: {r['n_signals_total']}  |  ret_mean_5: n={rm['n']}  "
        f"Mittel={rm['mean_pct']:+.2f}%  Win={rm['win_rate_pct']:.1f}%  Median={rm['median_pct']:+.2f}%"
    )
    print(
        f"  Makro-Rot: n={red['n']}  Mittel={red['mean_pct']:+.2f}%  Win={red['win_rate_pct']:.1f}%"
    )
    print(
        f"  Ohne Makro: n={yg['n']}  Mittel={yg['mean_pct']:+.2f}%  Win={yg['win_rate_pct']:.1f}%"
    )
    print(f"  Zeitraum: {r.get('date_min')} … {r.get('date_max')}")
    if r.get("by_year"):
        print("  Nach Jahr (ret_mean_5):")
        for y, v in sorted(r["by_year"].items()):
            print(f"    {y}: n={v['n']}  Mittel={v['mean_ret_mean_5_pct']:+.2f}%  Win={v['win_rate_pct']:.1f}%")
    print()
    return r


def main() -> None:
    pre = _print_block("VORHER (Snapshot 14.06.)", PRE)
    post = _print_block("NACHHER (ATR + Cross-Section)", POST)
    pr = pre["forward_returns"]["ret_mean_5"]
    po = post["forward_returns"]["ret_mean_5"]
    print("=== DELTA (Post − Pre, Gesamt) ===")
    print(f"  Signale: {post['n_signals_total'] - pre['n_signals_total']:+d}")
    print(f"  ret_mean_5 Mittel: {po['mean_pct'] - pr['mean_pct']:+.2f} pp")
    print(f"  ret_mean_5 Win-Rate: {po['win_rate_pct'] - pr['win_rate_pct']:+.1f} pp")
    rd0 = pre["context_tiers"]["red_macro_event"]
    rd1 = post["context_tiers"]["red_macro_event"]
    print(f"  Makro-Rot Win-Rate: {rd1['win_rate_pct'] - rd0['win_rate_pct']:+.1f} pp")
    print(f"  Makro-Rot Mittel: {rd1['mean_pct'] - rd0['mean_pct']:+.2f} pp")


if __name__ == "__main__":
    main()
