"""
News-Chip-Schwellen: >0 vs. Median-Varianten — IS (META+THR) vs. OOS (FINAL).

  python scripts/_scratch_news_chip_threshold_test.py

Rot-Subset (VIX < 20), Ziel: ret_mean_5. Kein Chip-Code geändert — nur Auswertung.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

MASTER = ROOT / "data" / "master_complete.csv"
VIX_RED = 20.0
MIN_N = 25
GOOD_RET = 0.02


@dataclass
class RuleResult:
    name: str
    split: str
    n_green: int
    n_warn: int
    green_rate: float
    ret_green: float
    ret_warn: float
    win_green: float
    win_warn: float
    delta: float


def _trading_calendar(start: str, end: str) -> np.ndarray:
    raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("SPY download leer")
    idx = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return np.sort(idx.unique())


def _assign_dataset(dates: pd.Series, cal: np.ndarray) -> pd.Series:
    fb = float(cfg.TIME_SPLIT_FRAC_BASE)
    fm = float(cfg.TIME_SPLIT_FRAC_META)
    ft = float(cfg.TIME_SPLIT_FRAC_THRESHOLD)
    p = int(cfg.TIME_PURGE_TRADING_DAYS)
    _, meta, thr, final = _split_calendar_four_way(cal, fb, fm, ft, p)
    meta_thr = set(pd.Timestamp(d) for d in meta) | set(pd.Timestamp(d) for d in thr)
    fin = set(pd.Timestamp(d) for d in final)

    def _lab(d: pd.Timestamp) -> str:
        d = pd.Timestamp(d).normalize()
        if d in fin:
            return "FINAL"
        if d in meta_thr:
            return "META+THR"
        return "OTHER"

    return dates.map(_lab)


def _stats(sub: pd.DataFrame, green: pd.Series) -> RuleResult | None:
    s = sub.dropna(subset=["ret", "diff"])
    if s.empty:
        return None
    g = green.reindex(s.index).fillna(False).astype(bool)
    ng, nw = int(g.sum()), int((~g).sum())
    if ng < MIN_N or nw < MIN_N:
        return None
    rg = s.loc[g, "ret"]
    rw = s.loc[~g, "ret"]
    return RuleResult(
        name="",
        split="",
        n_green=ng,
        n_warn=nw,
        green_rate=ng / (ng + nw),
        ret_green=float(rg.mean()),
        ret_warn=float(rw.mean()),
        win_green=float((rg > 0).mean()),
        win_warn=float((rw > 0).mean()),
        delta=float(rg.mean() - rw.mean()),
    )


def _eval_rule(
    sub: pd.DataFrame,
    green_fn,
    name: str,
    split_label: str,
) -> RuleResult | None:
    green = green_fn(sub)
    r = _stats(sub, green)
    if r is None:
        return None
    r.name = name
    r.split = split_label
    return r


def _print_block(results: list[RuleResult]) -> None:
    print(f"{'Regel':<32} {'Split':<10} {'grün%':>6} {'Δret':>8} {'grün ret':>9} {'warn ret':>9} {'OK?':>4}")
    print("-" * 88)
    by_name: dict[str, dict[str, RuleResult]] = {}
    for r in results:
        by_name.setdefault(r.name, {})[r.split] = r
    for name, splits in by_name.items():
        tune = splits.get("META+THR")
        fin = splits.get("FINAL")
        ok = ""
        if tune and fin and np.isfinite(tune.delta) and np.isfinite(fin.delta):
            ok = "ja" if tune.delta * fin.delta > 0 and abs(tune.delta) >= 0.001 else "nein"
        for lab in ("META+THR", "FINAL"):
            r = splits.get(lab)
            if not r:
                print(f"{name:<32} {lab:<10} {'—':>6} {'—':>8} {'—':>9} {'—':>9}")
                continue
            print(
                f"{name:<32} {lab:<10} {100*r.green_rate:5.1f}% "
                f"{100*r.delta:+.2f}pp {100*r.ret_green:+.2f}% {100*r.ret_warn:+.2f}% "
                f"{ok if lab == 'FINAL' else '':>4}"
            )


def main() -> None:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"]).dt.normalize()
    mc["ret"] = pd.to_numeric(mc["ret_mean_5"], errors="coerce")
    mc["diff"] = pd.to_numeric(mc["news_sec_minus_macro_tone"], errors="coerce")
    mc["vix"] = pd.to_numeric(mc["regime_vix_level"], errors="coerce")
    mc = mc.dropna(subset=["diff", "ret", "vix"])
    mc["is_red"] = mc["vix"] < VIX_RED

    dmin = mc["Date"].min().strftime("%Y-%m-%d")
    dmax = (mc["Date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    cal = _trading_calendar(dmin, dmax)
    mc["dataset"] = _assign_dataset(mc["Date"], cal)

    tune = mc[mc["dataset"] == "META+THR"].copy()
    fin = mc[mc["dataset"] == "FINAL"].copy()
    tune_red = tune[tune["is_red"]].copy()
    fin_red = fin[fin["is_red"]].copy()

    print("=" * 88)
    print("News-Chip-Schwellen — rot (VIX<20), ret_mean_5")
    print(
        f"Kalender: {pd.Timestamp(cal[0]).date()} … {pd.Timestamp(cal[-1]).date()}  |  "
        f"META+THR rot: {len(tune_red)}  FINAL rot: {len(fin_red)}"
    )
    print("=" * 88)

    # Kalibrier-Schwellen auf META+THR (rot), angewandt ohne Lookahead
    thr_global = float(tune_red["diff"].median())
    print(f"Globaler Median (META+THR rot): {thr_global:.4f}")

    results: list[RuleResult] = []

    def rule_gt0(sub: pd.DataFrame) -> pd.Series:
        return sub["diff"] > 0

    def rule_global_median(sub: pd.DataFrame) -> pd.Series:
        return sub["diff"] >= thr_global

    def rule_split_median(sub: pd.DataFrame) -> pd.Series:
        med = float(sub["diff"].median())
        return sub["diff"] >= med

    def rule_per_day_median(sub: pd.DataFrame) -> pd.Series:
        day_med = sub.groupby("Date")["diff"].transform("median")
        return sub["diff"] >= day_med

    def rule_rolling_60d(sub: pd.DataFrame) -> pd.Series:
        """Median der letzten 60 Kalendertage (nur Vergangenheit, rot-Signale)."""
        sub = sub.sort_values("Date")
        daily = (
            sub.groupby("Date")["diff"]
            .median()
            .reset_index()
            .sort_values("Date")
        )
        daily["roll_med"] = daily["diff"].rolling(window=60, min_periods=10).median().shift(1)
        merged = sub.merge(daily[["Date", "roll_med"]], on="Date", how="left")
        # Fallback: global IS median wenn noch keine Historie
        merged["roll_med"] = merged["roll_med"].fillna(thr_global)
        return merged["diff"] >= merged["roll_med"]

    rules = [
        ("> 0 (aktuell)", rule_gt0),
        (f">= global IS-Median ({thr_global:.3f})", rule_global_median),
        (">= Split-Median (50/50)", rule_split_median),
        (">= Median am Signaltag", rule_per_day_median),
        (">= Rolling-60d-Median (lag)", rule_rolling_60d),
    ]

    for name, fn in rules:
        for lab, sub in [("META+THR", tune_red), ("FINAL", fin_red)]:
            r = _eval_rule(sub, fn, name, lab)
            if r:
                results.append(r)

    _print_block(results)

    # Zusatz: grün-Anteil auf Website-OOS (signals.json FINAL keys in master)
    print("\n" + "=" * 88)
    print("Grün-Anteil (nur zur Einordnung des Chip-UI)")
    for name, fn in rules:
        rates = []
        for lab, sub in [("META+THR", tune_red), ("FINAL", fin_red)]:
            g = fn(sub)
            rates.append(f"{lab} {100*g.mean():.0f}%")
        print(f"  {name:<40} {' | '.join(rates)}")


if __name__ == "__main__":
    main()
