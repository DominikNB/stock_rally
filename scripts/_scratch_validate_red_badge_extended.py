"""Ergaenzende Badge-Tests (Median-Splits, raw deltas bei kleinem n)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.red_signal_quality import calibrate_gld_ret5_median_red_ref
from lib.vix_red_context_chips import _chip_thresholds
from scripts._scratch_validate_red_quality_tiers import MIN_N, VIX_RED, _assign_dataset

MIN_DELTA = 0.001


def report(name: str, tune, fin, fn) -> None:
    ht, lt = fn(tune)
    hf, lf = fn(fin)

    def d(hi, lo):
        if not len(hi) or not len(lo):
            return None, len(hi), len(lo), float("nan")
        raw = 100 * (hi["ret"].mean() - lo["ret"].mean())
        if len(hi) < MIN_N or len(lo) < MIN_N:
            return None, len(hi), len(lo), raw
        return raw, len(hi), len(lo), None

    dt, nth, ntl, rt = d(ht, lt)
    df, nfh, nfl, rf = d(hf, lf)
    oos = (
        dt is not None
        and df is not None
        and dt * df > 0
        and abs(dt) >= 100 * MIN_DELTA
        and abs(df) >= 100 * MIN_DELTA
    )
    ts = f"{dt:+.2f}pp" if dt is not None else (f"{rt:+.2f}pp (n<{MIN_N})" if rt == rt else "n/a")
    fs = f"{df:+.2f}pp" if df is not None else (f"{rf:+.2f}pp (n<{MIN_N})" if rf == rf else "n/a")
    mark = "OOS-OK" if oos else "fail"
    print(f"[{mark}] {name}")
    print(f"       tune {ts}  n={nth}/{ntl}  |  fin {fs}  n={nfh}/{nfl}")


def main() -> None:
    mc = pd.read_csv(ROOT / "data" / "master_complete.csv")
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    thr = _chip_thresholds()
    gref = calibrate_gld_ret5_median_red_ref()
    z = pd.to_numeric(red["regime_vix_z_20d"], errors="coerce")
    vr = pd.to_numeric(red["vix3m_vix_ratio"], errors="coerce")
    h = pd.to_numeric(red["sector_hhi_same_day"], errors="coerce")
    gld = pd.to_numeric(red["gld_ret_5d"], errors="coerce")
    red["f_gld"] = (gld < gref).astype(int)
    red.loc[gld.isna(), "f_gld"] = np.nan
    red["f_z"] = (z < 0).astype(int)
    red.loc[z.isna(), "f_z"] = np.nan
    red["f_term"] = (vr < thr["vix3m_vix_max"]).astype(int)
    red.loc[vr.isna(), "f_term"] = np.nan
    red["f_crowd"] = (h < thr["sector_hhi_max"]).astype(int)
    red.loc[h.isna(), "f_crowd"] = np.nan
    red["chips"] = red[["f_z", "f_term", "f_crowd"]].sum(axis=1, min_count=1)
    red["badge012"] = red["f_gld"].fillna(0) + red["chips"].fillna(0)
    red.loc[red["f_gld"].isna() & red["chips"].isna(), "badge012"] = np.nan
    red["badge012"] = red["badge012"].clip(0, 2)
    red["b_gld_or_c2"] = ((red["f_gld"] == 1) | (red["chips"] >= 2)).astype(int)

    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]
    print(f"FINAL: f_gld=1: {(fin.f_gld==1).sum()}/{fin.f_gld.notna().sum()}  badge012: {fin.badge012.value_counts().to_dict()}\n")

    report("0/1 GLD global ref (1=low)", tune, fin, lambda s: (s[s.f_gld == 1], s[s.f_gld == 0]))
    report(
        "0/1 GLD within-set median (orig. Validierung)",
        tune,
        fin,
        lambda s: (s[s["gld_ret_5d"] < s["gld_ret_5d"].median()], s[s["gld_ret_5d"] >= s["gld_ret_5d"].median()]),
    )
    report("0/1/2 GLD+Chips cap2: score=2 vs 0", tune, fin, lambda s: (s[s.badge012 == 2], s[s.badge012 == 0]))
    report("0/1/2 GLD+Chips cap2: score>=1 vs 0", tune, fin, lambda s: (s[s.badge012 >= 1], s[s.badge012 == 0]))
    report("0/1 GLD OR chips>=2", tune, fin, lambda s: (s[s.b_gld_or_c2 == 1], s[s.b_gld_or_c2 == 0]))
    report("0/1 chips>=2 vs <2", tune, fin, lambda s: (s[s.chips >= 2], s[s.chips < 2]))
    report(
        "0/1 median-split badge012",
        tune,
        fin,
        lambda s: (s[s.badge012 >= s.badge012.median()], s[s.badge012 < s.badge012.median()]),
    )


if __name__ == "__main__":
    main()
