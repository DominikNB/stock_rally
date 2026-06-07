"""
Einfaches Rot-Qualitäts-Badge (0/1 oder 0/1/2) — IS/OOS-Test.

  python scripts/_scratch_validate_red_badge_simple.py

Regel wie _scratch_validate_red_quality_tiers.py:
  META+THR + FINAL gleiche Richtung, |delta| >= 0.1pp, MIN_N je Bucket.

Liquidität nur als Feature wenn Fill-Rate hoch genug; primaer GLD + Chips.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.red_signal_quality import calibrate_gld_ret5_median_red_ref
from lib.vix_red_context_chips import _chip_thresholds
from scripts._scratch_validate_red_quality_tiers import (
    MIN_DELTA,
    MIN_N,
    VIX_RED,
    _assign_dataset,
)

MASTER = ROOT / "data" / "master_complete.csv"
OUT = ROOT / "data" / "_scratch_red_badge_simple_validation.json"


@dataclass
class Result:
    id: str
    kind: str  # binary | ternary
    label: str
    n_tune_hi: int
    n_tune_lo: int
    d_tune_pp: float | None
    n_fin_hi: int
    n_fin_lo: int
    d_fin_pp: float | None
    oos_ok: bool
    implement: bool
    tune_bucket_counts: dict | None = None
    fin_bucket_counts: dict | None = None


def _delta(hi: pd.DataFrame, lo: pd.DataFrame) -> tuple[float, int, int]:
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan"), len(hi), len(lo)
    return float(hi["ret"].mean() - lo["ret"].mean()), len(hi), len(lo)


def _eval_binary(tune: pd.DataFrame, fin: pd.DataFrame, col: str, hi_val: int = 1) -> Result:
    def _run(sub: pd.DataFrame):
        s = sub.dropna(subset=["ret", col])
        hi = s[s[col] == hi_val]
        lo = s[s[col] != hi_val]
        return _delta(hi, lo)

    dt, nth, ntl = _run(tune)
    df, nfh, nfl = _run(fin)
    oos = (
        np.isfinite(dt)
        and np.isfinite(df)
        and dt * df > 0
        and abs(dt) >= MIN_DELTA
        and abs(df) >= MIN_DELTA
    )
    impl = bool(oos and dt > 0 and df > 0)
    return Result(
        id=f"bin_{col}",
        kind="binary",
        label=f"{col}==1 vs 0",
        n_tune_hi=nth,
        n_tune_lo=ntl,
        d_tune_pp=round(100 * dt, 3) if np.isfinite(dt) else None,
        n_fin_hi=nfh,
        n_fin_lo=nfl,
        d_fin_pp=round(100 * df, 3) if np.isfinite(df) else None,
        oos_ok=bool(oos),
        implement=impl,
    )


def _eval_ternary(tune: pd.DataFrame, fin: pd.DataFrame, col: str) -> Result:
    """hi=2 vs lo=0 (mittel=1 ausgeschlossen oder als lo gewertet je Test)."""

    def _run_hi2_vs_0(sub: pd.DataFrame):
        s = sub.dropna(subset=["ret", col])
        hi = s[s[col] == 2]
        lo = s[s[col] == 0]
        return _delta(hi, lo)

    def _run_hi_ge1_vs_0(sub: pd.DataFrame):
        s = sub.dropna(subset=["ret", col])
        hi = s[s[col] >= 1]
        lo = s[s[col] == 0]
        return _delta(hi, lo)

    dt, nth, ntl = _run_hi2_vs_0(tune)
    df, nfh, nfl = _run_hi2_vs_0(fin)
    oos = (
        np.isfinite(dt)
        and np.isfinite(df)
        and dt * df > 0
        and abs(dt) >= MIN_DELTA
        and abs(df) >= MIN_DELTA
    )
    impl = bool(oos and dt > 0 and df > 0)
    tc = tune[col].value_counts().to_dict() if col in tune.columns else {}
    fc = fin[col].value_counts().to_dict() if col in fin.columns else {}
    return Result(
        id=f"ter_{col}_2vs0",
        kind="ternary",
        label=f"{col}: 2 vs 0",
        n_tune_hi=nth,
        n_tune_lo=ntl,
        d_tune_pp=round(100 * dt, 3) if np.isfinite(dt) else None,
        n_fin_hi=nfh,
        n_fin_lo=nfl,
        d_fin_pp=round(100 * df, 3) if np.isfinite(df) else None,
        oos_ok=bool(oos),
        implement=impl,
        tune_bucket_counts={str(k): int(v) for k, v in tc.items()},
        fin_bucket_counts={str(k): int(v) for k, v in fc.items()},
    )


def _features(red: pd.DataFrame) -> pd.DataFrame:
    thr = _chip_thresholds()
    gld_ref = calibrate_gld_ret5_median_red_ref()
    o = red.copy()
    z = pd.to_numeric(o["regime_vix_z_20d"], errors="coerce")
    vr = pd.to_numeric(o["vix3m_vix_ratio"], errors="coerce")
    h = pd.to_numeric(o["sector_hhi_same_day"], errors="coerce")
    gld = pd.to_numeric(o["gld_ret_5d"], errors="coerce")
    tier = o["liquidity_tier"].astype(str).str.strip().str.lower()

    o["f_gld"] = (gld < gld_ref).astype(float)
    o.loc[gld.isna(), "f_gld"] = np.nan
    o["f_vix_z"] = (z < 0).astype(float)
    o.loc[z.isna(), "f_vix_z"] = np.nan
    o["f_vix_term"] = (vr < thr["vix3m_vix_max"]).astype(float)
    o.loc[vr.isna(), "f_vix_term"] = np.nan
    o["f_crowd"] = (h < thr["sector_hhi_max"]).astype(float)
    o.loc[h.isna(), "f_crowd"] = np.nan
    o["f_liq"] = (tier == "ok").astype(float)
    o.loc[tier.isin(("", "unknown", "nan")) | o["liquidity_tier"].isna(), "f_liq"] = np.nan

    o["chip_good"] = o[["f_vix_z", "f_vix_term", "f_crowd"]].sum(axis=1, min_count=1)
    o["chip_good_int"] = o["chip_good"].round().astype("Int64")

    # --- Badge-Kandidaten ---
    # Binaer 0/1
    o["badge_gld_only"] = o["f_gld"].round().astype("Int64")
    o["badge_chip_ge2"] = (o["chip_good"] >= 2).astype("Int64")
    o["badge_chip_ge1"] = (o["chip_good"] >= 1).astype("Int64")
    o["badge_vix_z"] = o["f_vix_z"].round().astype("Int64")
    o["badge_gld_and_vixz"] = ((o["f_gld"] == 1) & (o["f_vix_z"] == 1)).astype("Int64")
    o.loc[o["f_gld"].isna() | o["f_vix_z"].isna(), "badge_gld_and_vixz"] = pd.NA

    # Ternaer 0/1/2: GLD + Chip-Punkte (max 4 -> cap 2)
    o["score_gld_plus_chips"] = o["f_gld"].fillna(0) + o["chip_good"].fillna(0)
    o.loc[o["f_gld"].isna() & o["chip_good"].isna(), "score_gld_plus_chips"] = np.nan
    o["badge_gld_chips_012"] = o["score_gld_plus_chips"].clip(0, 2).round().astype("Int64")

    # Ternaer: nur Chips 0/1/2
    o["badge_chips_012"] = o["chip_good_int"].clip(0, 2)

    # Ternaer: GLD (0/1) + 1 wenn >=2 chips gruen -> 0,1,2
    def _gld_chips_v2(row: pd.Series) -> float:
        g = row.get("f_gld")
        c = row.get("chip_good")
        if pd.isna(g) and pd.isna(c):
            return np.nan
        g = 0 if pd.isna(g) else int(g)
        c = 0 if pd.isna(c) else int(c)
        bonus = 1 if c >= 2 else 0
        return float(min(2, g + bonus))

    o["badge_gld_plus_chip2"] = o.apply(_gld_chips_v2, axis=1)
    o["badge_gld_plus_chip2"] = o["badge_gld_plus_chip2"].round().astype("Int64")

    # Mit Liquiditaet (nur wenn explizit getestet)
    o["score_no_liq"] = o["f_gld"].fillna(0) + o["chip_good"].fillna(0)
    o.loc[o["f_gld"].isna() & o["chip_good"].isna(), "score_no_liq"] = np.nan
    o["badge_no_liq_01"] = (o["score_no_liq"] >= 1).astype("Int64")
    o.loc[o["score_no_liq"].isna(), "badge_no_liq_01"] = pd.NA

    o["score_with_liq"] = o["score_no_liq"] + o["f_liq"].fillna(0)
    o["badge_with_liq_01"] = (o["score_with_liq"] >= 2).astype("Int64")
    o.loc[o["score_no_liq"].isna(), "badge_with_liq_01"] = pd.NA

    return o


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    red = _features(red)
    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]

    liq_rate = red["f_liq"].notna().mean()
    print(f"Rot: tune={len(tune)} final={len(fin)}")
    print(f"Liquiditaet bekannt: {100*liq_rate:.1f}% (Master)")

    results: list[Result] = []

    binary_cols = [
        ("badge_gld_only", "nur GLD niedrig (0/1)"),
        ("badge_chip_ge2", "Chips gruen >=2 (0/1)"),
        ("badge_chip_ge1", "Chips gruen >=1 (0/1)"),
        ("badge_vix_z", "VIX-Z < 0 (0/1)"),
        ("badge_gld_and_vixz", "GLD niedrig UND VIX-Z gruen (0/1)"),
        ("badge_no_liq_01", "GLD oder >=1 Chip gruen (0/1)"),
        ("badge_with_liq_01", "GLD+Chips+Liq Score>=2 (0/1)"),
    ]
    for col, desc in binary_cols:
        r = _eval_binary(tune, fin, col)
        r.label = desc
        r.id = col
        results.append(r)
        mark = "OK" if r.implement else ("oos" if r.oos_ok else "fail")
        print(
            f"  [{mark}] {desc}: tune {r.d_tune_pp}pp (n={r.n_tune_hi}/{r.n_tune_lo}) | "
            f"fin {r.d_fin_pp}pp (n={r.n_fin_hi}/{r.n_fin_lo})"
        )

    ternary_cols = [
        ("badge_chips_012", "Chip-Score 0/1/2"),
        ("badge_gld_chips_012", "GLD + Chips capped 0/1/2"),
        ("badge_gld_plus_chip2", "GLD + Bonus wenn 2+ Chips (0/1/2)"),
    ]
    print("\n--- Ternaer (Bucket 2 vs 0) ---")
    for col, desc in ternary_cols:
        r = _eval_ternary(tune, fin, col)
        r.label = desc
        r.id = col
        results.append(r)
        mark = "OK" if r.implement else ("oos" if r.oos_ok else "fail")
        print(
            f"  [{mark}] {desc}: tune {r.d_tune_pp}pp | fin {r.d_fin_pp}pp | "
            f"tune buckets {r.tune_bucket_counts} fin {r.fin_bucket_counts}"
        )

    # Ternaer: hi>=1 vs 0 fuer bestes binaeres Muster
    print("\n--- Ternaer hi>=1 vs 0 ---")
    for col, desc in ternary_cols:
        def _run(sub, c=col):
            s = sub.dropna(subset=["ret", c])
            return _delta(s[s[c] >= 1], s[s[c] == 0])

        dt, nth, ntl = _run(tune)
        df, nfh, nfl = _run(fin)
        oos = np.isfinite(dt) and np.isfinite(df) and dt * df > 0 and abs(dt) >= MIN_DELTA and abs(df) >= MIN_DELTA
        impl = bool(oos and dt > 0 and df > 0)
        print(
            f"  [{'OK' if impl else 'fail'}] {desc} >=1 vs 0: "
            f"tune {100*dt:+.2f}pp fin {100*df:+.2f}pp"
            if np.isfinite(dt) and np.isfinite(df)
            else f"  [fail] {desc} >=1 vs 0: insufficient n"
        )

    passed = [r for r in results if r.implement]
    payload = {
        "liquidity_fill_rate_master": round(liq_rate, 4),
        "passed_ids": [r.id for r in passed],
        "results": [asdict(r) for r in results],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")
    print(f"OOS-implementierbar: {[r.id for r in passed]}")


if __name__ == "__main__":
    main()
