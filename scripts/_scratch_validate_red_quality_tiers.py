"""
IS/OOS-Validierung aller Rot-Qualitäts-Kandidaten (Tier 1–2).

  python scripts/_scratch_validate_red_quality_tiers.py

Schreibt: data/_scratch_red_quality_validation.json
Regel OOS-OK: META+THR und FINAL gleiche Richtung, |Δ| >= 0.001, MIN_N je Bucket.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

MASTER = ROOT / "data" / "master_complete.csv"
OUT = ROOT / "data" / "_scratch_red_quality_validation.json"
MIN_N = 20
MIN_DELTA = 0.001
VIX_RED = 20.0


@dataclass
class Result:
    id: str
    tier: int
    label: str
    n_tune_hi: int
    n_tune_lo: int
    d_tune_pp: float
    n_fin_hi: int
    n_fin_lo: int
    d_fin_pp: float
    oos_ok: bool
    implement: bool


def _assign_dataset(mc: pd.DataFrame) -> pd.DataFrame:
    raw = yf.download(
        "SPY",
        start=mc["Date"].min(),
        end=(mc["Date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
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
    mt = {pd.Timestamp(d) for d in meta} | {pd.Timestamp(d) for d in thr}
    fin = {pd.Timestamp(d) for d in final}

    def _lab(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d).normalize()
        if d in fin:
            return "FINAL"
        if d in mt:
            return "META+THR"
        return "OTHER"

    mc = mc.copy()
    mc["dataset"] = mc["Date"].map(_lab)
    return mc


def _macro_event_flag(dates: pd.Series) -> pd.Series:
    from scripts._scratch_enrich_all_signals import _bdays_to_event, _load_macro_event_days

    ev = _load_macro_event_days()
    return _bdays_to_event(dates, ev, window=2)


def _cross_asset_5d(dates: pd.Series) -> pd.DataFrame:
    syms = {"gld_ret_5d": "GLD", "oil_ret_5d": "CL=F", "dxy_ret_5d": "DX-Y.NYB"}
    start = (dates.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(list(syms.values()), start=start, end=end, progress=False)
    out: dict[str, dict] = {}
    for col, sym in syms.items():
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"][sym]
        else:
            s = raw["Close"]
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index).normalize()
        ret5 = s.pct_change(5)
        out[col] = {d.strftime("%Y-%m-%d"): float(ret5.loc[d]) for d in ret5.index if pd.notna(ret5.loc[d])}
    rows = []
    for d in pd.to_datetime(dates).dt.normalize().unique():
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        rows.append({"Date": pd.Timestamp(d), **{c: out.get(c, {}).get(key, np.nan) for c in syms}})
    return pd.DataFrame(rows)


def _vix_ret_cols(dates: pd.Series) -> pd.DataFrame:
    start = (dates.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"].squeeze()
    else:
        s = raw["Close"]
    s = pd.to_numeric(s, errors="coerce")
    r1 = s.pct_change(1)
    r5 = s.pct_change(5)
    m1 = {d.strftime("%Y-%m-%d"): float(r1.loc[d]) for d in r1.index if pd.notna(r1.loc[d])}
    m5 = {d.strftime("%Y-%m-%d"): float(r5.loc[d]) for d in r5.index if pd.notna(r5.loc[d])}
    rows = []
    for d in pd.to_datetime(dates).dt.normalize().unique():
        key = pd.Timestamp(d).strftime("%Y-%m-%d")
        rows.append(
            {
                "Date": pd.Timestamp(d),
                "regime_vix_ret_1d": m1.get(key, np.nan),
                "regime_vix_ret_5d": m5.get(key, np.nan),
            }
        )
    return pd.DataFrame(rows)


def _delta(hi: pd.DataFrame, lo: pd.DataFrame) -> tuple[float, int, int]:
    if len(hi) < MIN_N or len(lo) < MIN_N:
        return float("nan"), len(hi), len(lo)
    return float(hi["ret"].mean() - lo["ret"].mean()), len(hi), len(lo)


def _eval(
    tune: pd.DataFrame,
    fin: pd.DataFrame,
    cid: str,
    tier: int,
    label: str,
    mask_fn,
) -> Result:
    def _run(sub: pd.DataFrame):
        s = sub.dropna(subset=["ret"])
        try:
            hi, lo = mask_fn(s)
            hi, lo = hi.dropna(subset=["ret"]), lo.dropna(subset=["ret"])
            return _delta(hi, lo)
        except Exception:
            return float("nan"), 0, 0

    dt, nth, ntl = _run(tune)
    df, nfh, nfl = _run(fin)
    oos = (
        np.isfinite(dt)
        and np.isfinite(df)
        and dt * df > 0
        and abs(dt) >= MIN_DELTA
        and abs(df) >= MIN_DELTA
    )
    # implement only if OOS and positive delta (good bucket better)
    impl = bool(oos and dt > 0 and df > 0)
    return Result(
        id=cid,
        tier=tier,
        label=label,
        n_tune_hi=nth,
        n_tune_lo=ntl,
        d_tune_pp=round(100 * dt, 3) if np.isfinite(dt) else None,
        n_fin_hi=nfh,
        n_fin_lo=nfl,
        d_fin_pp=round(100 * df, 3) if np.isfinite(df) else None,
        oos_ok=bool(oos),
        implement=impl,
    )


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"])
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = _assign_dataset(mc)
    red = mc[(mc["vix"] < VIX_RED) & mc["ret"].notna()].copy()
    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]

    print(f"Rot: tune={len(tune)} final={len(fin)}")

    # Zusatzspalten
    red["macro_event_within_2bd"] = _macro_event_flag(red["Date"])
    xdf = _cross_asset_5d(red["Date"])
    red = red.merge(xdf, on="Date", how="left")
    vdf = _vix_ret_cols(red["Date"])
    red = red.merge(vdf, on="Date", how="left")
    tune = red[red["dataset"] == "META+THR"]
    fin = red[red["dataset"] == "FINAL"]

    tests: list[tuple[str, int, str, callable]] = []

    def T(cid, tier, label, fn):
        tests.append((cid, tier, label, fn))

    T("alpha_sec_pos", 1, "alpha_sec_5d > 0", lambda s: (s[s["alpha_sec_5d"] > 0], s[s["alpha_sec_5d"] <= 0]))
    T("alpha_mkt_pos", 1, "alpha_mkt_5d > 0", lambda s: (s[s["alpha_mkt_5d"] > 0], s[s["alpha_mkt_5d"] <= 0]))
    T("rs_spy_pos", 1, "ret_vs_spy_5d > 0", lambda s: (s[s["ret_vs_spy_5d"] > 0], s[s["ret_vs_spy_5d"] <= 0]))
    T("rs_sec_pos", 1, "ret_vs_sector_5d > 0", lambda s: (s[s["ret_vs_sector_5d"] > 0], s[s["ret_vs_sector_5d"] <= 0]))
    T(
        "idio_proxy",
        1,
        "RS spy+sec>0 & cluster_corr<med",
        lambda s: (
            s[
                (s["ret_vs_spy_5d"] > 0)
                & (s["ret_vs_sector_5d"] > 0)
                & (s["cluster_mean_corr_60d"] < s["cluster_mean_corr_60d"].median())
            ],
            s[
                ~(
                    (s["ret_vs_spy_5d"] > 0)
                    & (s["ret_vs_sector_5d"] > 0)
                    & (s["cluster_mean_corr_60d"] < s["cluster_mean_corr_60d"].median())
                )
            ],
        ),
    )
    T(
        "liquidity_ok",
        1,
        "liquidity_tier=ok",
        lambda s: (
            s[s["liquidity_tier"].astype(str).str.lower() == "ok"],
            s[s["liquidity_tier"].astype(str).str.lower() != "ok"],
        ),
    )
    T("vix_z_lt0", 1, "regime_vix_z_20d < 0", lambda s: (s[s["regime_vix_z_20d"] < 0], s[s["regime_vix_z_20d"] >= 0]))
    T("vix3m_lt116", 1, "vix3m/vix < 1.16", lambda s: (s[s["vix3m_vix_ratio"] < 1.16], s[s["vix3m_vix_ratio"] >= 1.16]))
    T("hhi_lt035", 1, "sector_hhi < 0.35", lambda s: (s[s["sector_hhi_same_day"] < 0.35], s[s["sector_hhi_same_day"] >= 0.35]))
    T(
        "cluster_corr_low",
        1,
        "cluster_corr < median",
        lambda s: (
            s[s["cluster_mean_corr_60d"] < s["cluster_mean_corr_60d"].median()],
            s[s["cluster_mean_corr_60d"] >= s["cluster_mean_corr_60d"].median()],
        ),
    )
    T(
        "earnings_clear",
        1,
        "earnings_beyond_swing_15b",
        lambda s: (
            s[s["earnings_beyond_swing_15b"].astype(str).str.lower().isin(("true", "1", "yes"))],
            s[~s["earnings_beyond_swing_15b"].astype(str).str.lower().isin(("true", "1", "yes"))],
        ),
    )
    T(
        "no_macro_event_2bd",
        2,
        "macro_event_within_2bd=False",
        lambda s: (s[~s["macro_event_within_2bd"].astype(bool)], s[s["macro_event_within_2bd"].astype(bool)]),
    )
    T(
        "gld_ret5_low",
        2,
        "gld_ret_5d < median",
        lambda s: (
            s[s["gld_ret_5d"] < s["gld_ret_5d"].median()],
            s[s["gld_ret_5d"] >= s["gld_ret_5d"].median()],
        ),
    )
    T(
        "vix_ret5_nonneg",
        1,
        "regime_vix_ret_5d >= 0",
        lambda s: (s[s["regime_vix_ret_5d"] >= 0], s[s["regime_vix_ret_5d"] < 0]),
    )
    T(
        "not_crypto",
        2,
        "sector != crypto",
        lambda s: (s[s["sector"].astype(str) != "crypto"], s[s["sector"].astype(str) == "crypto"]),
    )
    T(
        "combo_ctx_idio",
        1,
        "vix_z<0 & rs_spy>0 & hhi<0.35",
        lambda s: (
            s[(s["regime_vix_z_20d"] < 0) & (s["ret_vs_spy_5d"] > 0) & (s["sector_hhi_same_day"] < 0.35)],
            s[~((s["regime_vix_z_20d"] < 0) & (s["ret_vs_spy_5d"] > 0) & (s["sector_hhi_same_day"] < 0.35))],
        ),
    )

    results: list[Result] = []
    for cid, tier, label, fn in tests:
        r = _eval(tune, fin, cid, tier, label, fn)
        results.append(r)

    passed = [r for r in results if r.implement]
    print("\n=== OOS-bestätigt (implementieren) ===")
    for r in passed:
        print(f"  [{r.tier}] {r.label}: d_tune={r.d_tune_pp}pp d_fin={r.d_fin_pp}pp")

    print("\n=== OOS Richtung OK, aber Δ negativ (nicht implementieren) ===")
    for r in results:
        if r.oos_ok and not r.implement:
            print(f"  [{r.tier}] {r.label}: d_tune={r.d_tune_pp}pp d_fin={r.d_fin_pp}pp")

    payload = {
        "n_tune": len(tune),
        "n_final": len(fin),
        "passed_ids": [r.id for r in passed],
        "results": [asdict(r) for r in results],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {OUT}")


if __name__ == "__main__":
    main()
