"""
OOS-Validierung aller Kontext-Hypothesen auf angereicherten Scratch-Signalen.

Voraussetzung:
  python scripts/_scratch_enrich_all_signals.py

  python scripts/_scratch_validate_all_context.py

Regel: Hypothese nur [OK], wenn Richtung auf META+THR und FINAL gleich (rot-Subset wo sinnvoll).
Ausgeschlossen: Earnings-*, short_float*, short_days_to_cover*
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENRICHED = ROOT / "data" / "_scratch_signals_enriched.parquet"
VIX_RED = 20.0
MIN_N = 25
GOOD_RET = 0.02
SKIP_PREFIXES = ("earnings_", "next_earnings", "bdays_to_next")
SKIP_EXACT = {"short_float_pct", "short_days_to_cover", "short_float_confidence"}


@dataclass
class Hypothesis:
    name: str
    col: str
    rule: str  # ge_median | lt_median | eq_true | eq_false | lt_zero | lt_pct
    scope: str  # all | red
    category: str
    thr: float | None = None  # fuer lt_pct


def _stats(r: pd.Series) -> str:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 5:
        return f"n={len(r)}"
    return (
        f"n={len(r):4d} mean={100*r.mean():+.2f}% win={100*(r>0).mean():.0f}% "
        f"good2={100*(r>GOOD_RET).mean():.0f}%"
    )


def _delta(sub: pd.DataFrame, col: str, ge: bool = True) -> float:
    s = sub.dropna(subset=[col, "ret"])
    if len(s) < MIN_N * 2:
        return float("nan")
    med = float(s[col].median())
    hi = s[s[col] >= med] if ge else s[s[col] < med]
    lo = s[s[col] < med] if ge else s[s[col] >= med]
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan")
    return float(hi["ret"].mean() - lo["ret"].mean())


def _delta_bool(sub: pd.DataFrame, col: str, want_true: bool = True) -> float:
    s = sub.dropna(subset=[col, "ret"])
    if len(s) < MIN_N * 2:
        return float("nan")
    hi = s[s[col].astype(bool) == want_true]
    lo = s[s[col].astype(bool) != want_true]
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan")
    return float(hi["ret"].mean() - lo["ret"].mean())


def _eval_hyp(
    df: pd.DataFrame, h: Hypothesis, tune: pd.DataFrame, fin: pd.DataFrame
) -> dict:
    scope_tune = tune[tune["is_red"]] if h.scope == "red" else tune
    scope_fin = fin[fin["is_red"]] if h.scope == "red" else fin

    if h.rule == "ge_median":
        dt, dfin = _delta(scope_tune, h.col, True), _delta(scope_fin, h.col, True)
    elif h.rule == "lt_median":
        dt, dfin = _delta(scope_tune, h.col, False), _delta(scope_fin, h.col, False)
    elif h.rule == "eq_true":
        dt, dfin = _delta_bool(scope_tune, h.col, True), _delta_bool(scope_fin, h.col, True)
    elif h.rule == "eq_false":
        dt, dfin = _delta_bool(scope_tune, h.col, False), _delta_bool(scope_fin, h.col, False)
    elif h.rule == "lt_zero":
        def _dlt(sub: pd.DataFrame) -> float:
            s = sub.dropna(subset=[h.col, "ret"])
            if len(s) < MIN_N * 2:
                return float("nan")
            hi, lo = s[pd.to_numeric(s[h.col], errors="coerce") < 0], s[pd.to_numeric(s[h.col], errors="coerce") >= 0]
            if len(hi) < MIN_N or len(lo) < MIN_N:
                return float("nan")
            return float(hi["ret"].mean() - lo["ret"].mean())

        dt, dfin = _dlt(scope_tune), _dlt(scope_fin)
    elif h.rule == "lt_pct" and h.thr is not None:
        def _dlt2(sub: pd.DataFrame) -> float:
            s = sub.dropna(subset=[h.col, "ret"])
            if len(s) < MIN_N * 2:
                return float("nan")
            x = pd.to_numeric(s[h.col], errors="coerce")
            hi, lo = s[x < h.thr], s[x >= h.thr]
            if len(hi) < MIN_N or len(lo) < MIN_N:
                return float("nan")
            return float(hi["ret"].mean() - lo["ret"].mean())

        dt, dfin = _dlt2(scope_tune), _dlt2(scope_fin)
    else:
        return {"hypothesis": h.name, "skip": True}

    ok = (
        np.isfinite(dt)
        and np.isfinite(dfin)
        and dt * dfin > 0
        and abs(dt) >= 0.001
        and abs(dfin) >= 0.001
    )
    return {
        "category": h.category,
        "hypothesis": h.name,
        "col": h.col,
        "scope": h.scope,
        "d_tune_pp": 100 * dt,
        "d_fin_pp": 100 * dfin,
        "oos_ok": bool(ok),
    }


def _build_numeric_hyps(cols: list[str], scope: str, category: str) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for c in cols:
        out.append(Hypothesis(f"{c} >= median", c, "ge_median", scope, category))
    return out


def main() -> None:
    if not ENRICHED.is_file():
        print(f"Fehlt {ENRICHED} — zuerst: python scripts/_scratch_enrich_all_signals.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(ENRICHED)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    if "is_red" not in df.columns:
        df["is_red"] = df["vix"] < VIX_RED

    tune = df[df["dataset"].isin(["META", "THRESHOLD"])]
    fin = df[df["dataset"] == "FINAL"]

    skip = {c for c in df.columns if any(c.startswith(p) for p in SKIP_PREFIXES)} | SKIP_EXACT

    enrich_num = [
        c
        for c in df.columns
        if c not in skip
        and c not in ("ticker", "Date", "dataset", "vix_bucket", "company", "ret", "is_red", "is_eu")
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= MIN_N * 4
    ]

    hyps: list[Hypothesis] = []

    # Bekannte stabile + Interaktionen
    hyps += [
        Hypothesis("VIX-Z < 0 (unter 20d-Mittel)", "regime_vix_z_20d", "lt_zero", "red", "regime"),
        Hypothesis("dist_high < -10%", "dist_high", "lt_pct", "red", "price", thr=-0.10),
        Hypothesis("dist_from_20d_high < -10%", "dist_from_20d_high_pct", "lt_pct", "red", "price", thr=-0.10),
        Hypothesis("news_sec > macro (diff>=median)", "news_sec_minus_macro_tone", "ge_median", "red", "news"),
        Hypothesis("VIX-Z >= median (regime_vix_z_20d)", "regime_vix_z_20d", "ge_median", "red", "regime"),
        Hypothesis("dist_from_20d_high >= median", "dist_from_20d_high_pct", "ge_median", "red", "price"),
        Hypothesis("rot + Technology (eq_true)", "rot_tech", "eq_true", "all", "interaction"),
        Hypothesis("rot + ret_vs_sector_20d < 0", "rot_lag_sector_20d", "eq_true", "all", "interaction"),
        Hypothesis("macro_event_within_2bd", "macro_event_within_2bd", "eq_false", "red", "macro_cal"),
        Hypothesis("macro_event_within_2bd", "macro_event_within_2bd", "eq_false", "all", "macro_cal"),
    ]

    news_cols = [c for c in df.columns if "news_" in c and c not in skip and df[c].notna().sum() >= MIN_N * 4]
    hyps += _build_numeric_hyps(
        [c for c in news_cols if "tone" in c or "diff" in c or "minus" in c][:8],
        "red",
        "news",
    )
    hyps += _build_numeric_hyps(
        [c for c in news_cols if "tone" in c or "diff" in c or "minus" in c][:8],
        "all",
        "news",
    )

    cross = [c for c in ("gld_ret_5d", "oil_ret_5d", "dxy_ret_5d", "eurusd_ret_5d", "vix3m_vix_ratio") if c in df.columns]
    hyps += _build_numeric_hyps(cross, "red", "cross_asset")
    hyps += _build_numeric_hyps(cross, "all", "cross_asset")

    sector_rel = [
        c
        for c in (
            "ret_vs_spy_5d",
            "ret_vs_sector_5d",
            "ret_vs_spy_20d",
            "ret_vs_sector_20d",
            "alpha_mkt_5d",
            "alpha_sec_5d",
            "beta_mkt_60d",
            "beta_sec_60d",
            "sector_breadth_ma50",
            "sector_ret_1d",
            "sector_ret_3d",
            "market_ret_3d",
        )
        if c in enrich_num
    ]
    hyps += _build_numeric_hyps(sector_rel, "red", "sector_relative")
    hyps += _build_numeric_hyps(sector_rel, "all", "sector_relative")

    crowd = [
        c
        for c in enrich_num
        if any(
            x in c
            for x in (
                "signals_same",
                "sector_share",
                "sector_hhi",
                "cluster_mean_corr",
                "pct_rank_prob",
                "prob_z",
            )
        )
    ]
    hyps += _build_numeric_hyps(crowd, "red", "crowding")
    hyps += _build_numeric_hyps(crowd, "all", "crowding")

    regime = [c for c in enrich_num if c.startswith("regime_")]
    hyps += _build_numeric_hyps(regime, "red", "regime")
    hyps += _build_numeric_hyps(regime, "all", "regime")

    tech = [
        c
        for c in enrich_num
        if any(
            x in c
            for x in (
                "momentum_",
                "volatility_",
                "bb_width",
                "trend_efficiency",
                "volume_zscore",
                "open_gap",
                "dist_from",
                "dist_ma200",
                "amihud",
            )
        )
    ]
    hyps += _build_numeric_hyps(tech[:20], "red", "technical")
    hyps += _build_numeric_hyps(tech[:20], "all", "technical")

    liq = [c for c in enrich_num if "adv" in c or c == "liquidity_tier"]
    # liquidity_tier is categorical — skip in numeric

    inst = [c for c in ("inst_own_pct",) if c in enrich_num]
    hyps += _build_numeric_hyps(inst, "red", "ownership_proxy")

    hyps.append(Hypothesis("EU ticker", "is_eu", "eq_false", "red", "structure"))
    hyps.append(Hypothesis("EU ticker", "is_eu", "eq_false", "all", "structure"))

    # Deduplicate
    seen: set[tuple] = set()
    uniq: list[Hypothesis] = []
    for h in hyps:
        k = (h.name, h.col, h.rule, h.scope)
        if k in seen:
            continue
        seen.add(k)
        if h.col not in df.columns:
            continue
        uniq.append(h)

    results = [_eval_hyp(df, h, tune, fin) for h in uniq]
    res_df = pd.DataFrame([r for r in results if not r.get("skip")])
    out_csv = ROOT / "data" / "_scratch_context_validation_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("=" * 72)
    print("OOS-STABIL (META+THR und FINAL gleiche Richtung, |delta|>=0.1pp)")
    print("=" * 72)
    ok_df = res_df[res_df["oos_ok"]].sort_values("d_tune_pp", key=abs, ascending=False)
    if ok_df.empty:
        print("  (keine)")
    else:
        for _, r in ok_df.iterrows():
            print(
                f"  [{r['category']:16}] {r['scope']:3} {r['hypothesis'][:50]:50} "
                f"d_thr={r['d_tune_pp']:+.2f}pp d_fin={r['d_fin_pp']:+.2f}pp"
            )

    print("\n" + "=" * 72)
    print("TOP FINAL-ONLY (Richtung widerspricht META+THR) — nicht fuer Chips")
    print("=" * 72)
    bad = res_df[
        np.isfinite(res_df["d_tune_pp"])
        & np.isfinite(res_df["d_fin_pp"])
        & (res_df["d_tune_pp"] * res_df["d_fin_pp"] < 0)
    ].copy()
    bad["conflict"] = bad["d_fin_pp"].abs()
    for _, r in bad.sort_values("conflict", ascending=False).head(15).iterrows():
        print(
            f"  [{r['category']:16}] {r['scope']:3} {r['hypothesis'][:45]:45} "
            f"d_thr={r['d_tune_pp']:+.2f}pp d_fin={r['d_fin_pp']:+.2f}pp"
        )

    print("\n" + "=" * 72)
    print("ROT-BASELINE")
    print("=" * 72)
    for label, sub in [("META+THR rot", tune[tune["is_red"]]), ("FINAL rot", fin[fin["is_red"]])]:
        print(f"  {label}: {_stats(sub['ret'])}")

    print("\n" + "=" * 72)
    print("CHIP-KANDIDATEN (nur rot, OOS-stabil, fuer Website-Kontext)")
    print("=" * 72)
    chip_hints = [
        ("VIX-Z < 0", "VIX unter 20-Tage-Mittel"),
        ("vix3m_vix_ratio", "hoher VIX-Term (3M/VIX)"),
        ("sector_hhi_same_day", "Sektor-Crowding am Tag"),
        ("ret_vs_sector_5d", "relative Sektor-Staerke 5d"),
        ("news_macro_3_20_5_tone", "Makro-News-Ton"),
        ("gld_ret_5d", "Gold 5d-Rendite"),
    ]
    for needle, label in chip_hints:
        sub = ok_df[ok_df["hypothesis"].str.contains(needle, case=False, na=False)]
        sub_red = sub[sub["scope"] == "red"]
        if sub_red.empty:
            sub_red = sub[sub["scope"] == "all"]
        if sub_red.empty:
            print(f"  — {label}: nicht OOS-stabil")
            continue
        r = sub_red.iloc[0]
        direction = "hoch besser" if r["d_tune_pp"] > 0 else "niedrig besser"
        print(f"  + {label}: {direction} (d_thr={r['d_tune_pp']:+.2f}pp, d_fin={r['d_fin_pp']:+.2f}pp)")

    print(f"\nErgebnis-CSV: {out_csv}  ({len(res_df)} Hypothesen)")


if __name__ == "__main__":
    main()
