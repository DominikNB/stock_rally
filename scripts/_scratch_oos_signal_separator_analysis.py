"""Deep OOS analysis: which signal-time features separate good vs bad trades."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

MASTER = ROOT / "data" / "master_complete.csv"
MIN_N = 25

# Not usable for live filtering (lookahead / label / duplicate of target)
EXCLUDE = {
    "ret_mean_5", "ret_2d", "ret_4d", "ret_6d", "ret_8d", "ret_10d",
    "train_target", "rally", "prob", "threshold_used", "meta_prob_margin",
    "ret",  # alias
}


def _assign_split(df: pd.DataFrame) -> pd.DataFrame:
    raw = yf.download(
        "SPY",
        start=df["Date"].min(),
        end=(df["Date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
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
    fin = {pd.Timestamp(d) for d in final}
    mt = {pd.Timestamp(d) for d in meta} | {pd.Timestamp(d) for d in thr}

    def _lab(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d).normalize()
        if d in fin:
            return "FINAL"
        if d in mt:
            return "META+THR"
        return "OTHER"

    out = df.copy()
    out["dataset"] = out["Date"].map(_lab)
    return out


def _summ(sub: pd.DataFrame, label: str) -> dict | None:
    if len(sub) < MIN_N:
        return None
    r = sub["ret"].astype(float)
    return {
        "label": label,
        "n": int(len(sub)),
        "mean": float(r.mean()),
        "win": float((r > 0).mean()),
        "good2": float((r > 0.02).mean()),
    }


def _print_summ(s: dict | None) -> None:
    if not s:
        return
    print(
        f"  {s['label']:28} n={s['n']:4d}  mean={100*s['mean']:+.2f}%  "
        f"win={100*s['win']:.0f}%  >2%={100*s['good2']:.0f}%"
    )


def quintile_lift(df: pd.DataFrame, col: str, q: int = 5) -> dict | None:
    sub = df.dropna(subset=[col, "ret"])
    if len(sub) < 5 * MIN_N:
        return None
    try:
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[col], q=q, duplicates="drop")
    except ValueError:
        return None
    g = sub.groupby("q", observed=True)["ret"].agg(["count", "mean", lambda x: (x > 0).mean()])
    g.columns = ["n", "mean", "win"]
    if len(g) < 2:
        return None
    lo, hi = g.iloc[0], g.iloc[-1]
    rho, p = stats.spearmanr(sub[col], sub["ret"])
    return {
        "col": col,
        "rho": float(rho),
        "p": float(p),
        "lo_mean": float(lo["mean"]),
        "hi_mean": float(hi["mean"]),
        "lift": float(hi["mean"] - lo["mean"]),
        "lo_win": float(lo["win"]),
        "hi_win": float(hi["win"]),
    }


def eval_rule(df: pd.DataFrame, mask: pd.Series, label: str) -> dict | None:
    keep = df[mask.fillna(False)]
    drop = df[~mask.fillna(False)]
    sk = _summ(keep, f"KEEP {label}")
    sd = _summ(drop, f"SKIP {label}")
    if not sk or not sd:
        return None
    return {"label": label, "keep": sk, "skip": sd, "delta": sk["mean"] - sd["mean"]}


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    df = mc.dropna(subset=["ret"]).copy()
    df = _assign_split(df)

    print("=" * 78)
    print("OOS SIGNAL SEPARATOR ANALYSIS")
    print("=" * 78)
    _print_summ(_summ(df, "ALL"))
    _print_summ(_summ(df[df["dataset"] == "FINAL"], "FINAL only"))
    _print_summ(_summ(df[df["dataset"] == "META+THR"], "META+THR only"))

    # --- 1) Numeric feature screen ---
    num_feats = [
        c for c in df.columns
        if c not in EXCLUDE
        and df[c].dtype in ("float64", "float32", "int64", "int32")
        and int(df[c].notna().sum()) >= 80
    ]
    lifts = [quintile_lift(df, c) for c in num_feats]
    lifts = [x for x in lifts if x and abs(x["lift"]) >= 0.01]

    print("\n--- 1) Top numeric separators (quintile hi-lo lift >= 1pp) ---")
    for r in sorted(lifts, key=lambda x: abs(x["lift"]), reverse=True)[:18]:
        sig = "*" if r["p"] < 0.05 else ""
        print(
            f"  {r['col']:32} rho={r['rho']:+.3f} p={r['p']:.3f}{sig}  "
            f"lo={100*r['lo_mean']:+.2f}% hi={100*r['hi_mean']:+.2f}% "
            f"lift={100*r['lift']:+.2f}pp"
        )

    # --- 2) Stability FINAL only vs full sample for top features ---
    top_feats = [r["col"] for r in sorted(lifts, key=lambda x: abs(x["lift"]), reverse=True)[:10]]
    fin = df[df["dataset"] == "FINAL"]
    print("\n--- 2) Stability: lift on FULL vs FINAL-only (top features) ---")
    for col in top_feats:
        a, b = quintile_lift(df, col), quintile_lift(fin, col)
        if not a:
            continue
        bl = f"{100*b['lift']:+.2f}" if b else "n/a"
        print(f"  {col:32} full_lift={100*a['lift']:+.2f}pp  final_lift={bl}pp")

    # --- 3) Categorical ---
    print("\n--- 3) Categorical features ---")
    for col in [
        "vix_regime_ampel", "red_quality_tier", "liquidity_tier", "sector",
        "quality_red", "quality_gld", "earnings_in_3_15_bday_window",
        "macro_event_within_2bd", "signals_same_day",
    ]:
        if col not in df.columns:
            continue
        if df[col].dtype in ("float64", "int64"):
            # treat as binary
            for val in sorted(df[col].dropna().unique()):
                m = df[col] == val
                s = _summ(df[m], f"{col}={val}")
                if s:
                    _print_summ(s)
            continue
        g = (
            df.groupby(col)["ret"]
            .agg(n="count", mean="mean", win=lambda x: (x > 0).mean())
            .reset_index()
        )
        g = g[g["n"] >= MIN_N].sort_values("mean", ascending=False)
        if g.empty:
            continue
        print(f"\n  [{col}]")
        for _, row in g.iterrows():
            print(
                f"    {str(row[col]):22} n={int(row['n']):3d} "
                f"mean={100*row['mean']:+.2f}% win={100*row['win']:.0f}%"
            )

    # --- 4) Actionable rule candidates ---
    print("\n--- 4) Actionable filter rules (FULL sample) ---")
    rules: list[tuple[str, pd.Series]] = []

    if "regime_vix_level" in df.columns:
        rules.append(("VIX>=20", df["regime_vix_level"] >= 20))
        rules.append(("VIX>=25", df["regime_vix_level"] >= 25))
    if "vix_regime_ampel" in df.columns:
        rules.append(("ampel!=red", df["vix_regime_ampel"] != "red"))
        rules.append(("ampel=green", df["vix_regime_ampel"] == "green"))
    if "bdays_to_next_earnings" in df.columns:
        rules.append(("earnings>10bd", df["bdays_to_next_earnings"] > 10))
        rules.append(("earnings>20bd", df["bdays_to_next_earnings"] > 20))
    if "news_macro_7_10_10_tone" in df.columns:
        med = df["news_macro_7_10_10_tone"].median()
        rules.append(("macro_tone<=median", df["news_macro_7_10_10_tone"] <= med))
    if "dist_ma200_pct" in df.columns:
        rules.append(("above_MA200", df["dist_ma200_pct"] > 0))
        rules.append(("dist_MA200 top40%", df["dist_ma200_pct"] >= df["dist_ma200_pct"].quantile(0.6)))
    if "momentum_20d" in df.columns:
        rules.append(("mom20 top40%", df["momentum_20d"] >= df["momentum_20d"].quantile(0.6)))
    if "volatility_20d" in df.columns:
        rules.append(("vol20 top40%", df["volatility_20d"] >= df["volatility_20d"].quantile(0.6)))
    if "signals_same_day" in df.columns:
        rules.append(("cluster<=2 same day", df["signals_same_day"] <= 2))
        rules.append(("cluster<=4 same day", df["signals_same_day"] <= 4))
    if "sector_ret_1d" in df.columns:
        rules.append(("sector_ret_1d>0", df["sector_ret_1d"] > 0))
    if "vix3m_vix_ratio" in df.columns:
        rules.append(("vvix_ratio<=1.16", df["vix3m_vix_ratio"] <= 1.16))
    if "quality_red" in df.columns:
        rules.append(("quality_red=0", df["quality_red"] == 0))
    if "alpha_mkt_5d" in df.columns:
        rules.append(("alpha_mkt_5d bot50%", df["alpha_mkt_5d"] <= df["alpha_mkt_5d"].quantile(0.5)))

    scored = []
    for label, mask in rules:
        ev = eval_rule(df, mask, label)
        if ev:
            scored.append(ev)
    for ev in sorted(scored, key=lambda x: x["delta"], reverse=True):
        print(f"\n  Rule: {ev['label']}  (delta keep-skip mean = {100*ev['delta']:+.2f}pp)")
        _print_summ(ev["keep"])
        _print_summ(ev["skip"])

    # --- 5) Same rules on FINAL only (honest OOS) ---
    print("\n--- 5) Same rules on FINAL only (true OOS) ---")
    scored_fin = []
    for label, mask in rules:
        ev = eval_rule(fin, mask, label)
        if ev:
            scored_fin.append(ev)
    for ev in sorted(scored_fin, key=lambda x: x["delta"], reverse=True)[:12]:
        print(f"\n  Rule: {ev['label']}  delta={100*ev['delta']:+.2f}pp")
        _print_summ(ev["keep"])
        _print_summ(ev["skip"])

    # --- 6) Composite rules ---
    print("\n--- 6) Composite filters ---")
    composites: list[tuple[str, pd.Series]] = []
    if "regime_vix_level" in df.columns and "bdays_to_next_earnings" in df.columns:
        composites.append((
            "VIX>=20 & earnings>10bd",
            (df["regime_vix_level"] >= 20) & (df["bdays_to_next_earnings"] > 10),
        ))
    if "regime_vix_level" in df.columns and "dist_ma200_pct" in df.columns:
        composites.append((
            "VIX>=20 & above_MA200",
            (df["regime_vix_level"] >= 20) & (df["dist_ma200_pct"] > 0),
        ))
    if "vix_regime_ampel" in df.columns and "bdays_to_next_earnings" in df.columns:
        composites.append((
            "ampel!=red & earnings>10bd",
            (df["vix_regime_ampel"] != "red") & (df["bdays_to_next_earnings"] > 10),
        ))
    if "regime_vix_level" in df.columns and "signals_same_day" in df.columns:
        composites.append((
            "VIX>=20 & cluster<=2",
            (df["regime_vix_level"] >= 20) & (df["signals_same_day"] <= 2),
        ))
    if "dist_ma200_pct" in df.columns and "momentum_20d" in df.columns:
        q_mom = df["momentum_20d"].quantile(0.6)
        composites.append((
            "above_MA200 & mom20 top40%",
            (df["dist_ma200_pct"] > 0) & (df["momentum_20d"] >= q_mom),
        ))

    for label, mask in composites:
        for subset, tag in [(df, "FULL"), (fin, "FINAL")]:
            ev = eval_rule(subset, mask, f"{label} [{tag}]")
            if ev:
                print(
                    f"  {ev['label']:40} delta={100*ev['delta']:+.2f}pp  "
                    f"keep_n={ev['keep']['n']} keep_mean={100*ev['keep']['mean']:+.2f}%"
                )

    # --- 7) Year stability for best rules ---
    print("\n--- 7) Year stability (best single rule: VIX>=20 if exists) ---")
    if "regime_vix_level" in df.columns:
        m = df["regime_vix_level"] >= 20
        df["year"] = df["Date"].dt.year
        for yr, g in df.groupby("year"):
            ev = eval_rule(g, m, "VIX>=20")
            if ev:
                print(
                    f"  {int(yr)}: keep n={ev['keep']['n']:3d} mean={100*ev['keep']['mean']:+.2f}% | "
                    f"skip n={ev['skip']['n']:3d} mean={100*ev['skip']['mean']:+.2f}% | "
                    f"delta={100*ev['delta']:+.2f}pp"
                )

    # --- 8) Lookback alpha paradox ---
    print("\n--- 8) Lookback alpha (contemporaneous, NOT forward) ---")
    for col in ["alpha_mkt_5d", "alpha_sec_5d", "ret_vs_spy_5d", "ret_vs_sector_5d"]:
        if col not in df.columns:
            continue
        r = quintile_lift(df, col)
        if r:
            print(
                f"  {col}: rho={r['rho']:+.3f} lift={100*r['lift']:+.2f}pp "
                f"(lo={100*r['lo_mean']:+.2f}% hi={100*r['hi_mean']:+.2f}%)"
            )


if __name__ == "__main__":
    main()
