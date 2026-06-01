"""Post-pipeline FINAL OOS return analysis vs Meta-Optuna."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import _split_calendar_four_way

ART = ROOT / "models" / "scoring_artifacts.joblib"
MASTER = ROOT / "data" / "master_complete.csv"
SIGNALS_JSON = ROOT / "docs" / "signals.json"


def final_date_set() -> set[pd.Timestamp]:
    dates = pd.bdate_range(cfg.START_DATE, cfg.TRAIN_END_DATE)
    _, _, _, final_dates = _split_calendar_four_way(
        dates.values,
        float(cfg.TIME_SPLIT_FRAC_BASE),
        float(cfg.TIME_SPLIT_FRAC_META),
        float(cfg.TIME_SPLIT_FRAC_THRESHOLD),
        int(cfg.TIME_PURGE_TRADING_DAYS),
    )
    return {pd.Timestamp(d).normalize() for d in final_dates}


def report_block(label: str, df: pd.DataFrame, meta_ret: float) -> None:
    sub = df.dropna(subset=["ret_mean_5"]).copy()
    if sub.empty:
        print(f"\n{label}: keine Return-Daten")
        return
    r = sub["ret_mean_5"].astype(float)
    r4 = sub["ret_4d"].dropna().astype(float) if "ret_4d" in sub.columns else pd.Series(dtype=float)
    thr_u = sorted(sub["threshold_used"].dropna().unique()) if "threshold_used" in sub.columns else []

    print(f"\n=== {label} ===")
    print(f"  Signale (mit ret_mean_5): {len(sub):,}")
    if thr_u:
        print(f"  threshold_used: {', '.join(f'{t:.3f}' for t in thr_u)}")
    print(f"  Mean ret_mean_5:   {100*r.mean():.3f}%")
    print(f"  Median:            {100*r.median():.3f}%")
    print(f"  Win-rate (>0):     {100*(r > 0).mean():.1f}%")
    print(f"  Std:               {100*r.std():.3f}%")
    print(f"  p10 / p90:         {100*r.quantile(0.1):.2f}% / {100*r.quantile(0.9):.2f}%")
    if len(r4):
        print(f"  ret_4d:            mean={100*r4.mean():.3f}%, win={100*(r4 > 0).mean():.1f}%")
    print(f"  Delta vs Meta:     {100*(r.mean() - meta_ret):+.3f} pp")

    if "train_target" in sub.columns:
        tp = (sub["train_target"] == 1).mean() * 100
        print(f"  train_target=1:    {100*tp:.1f}%  (Legacy-Label im Holdout-Export, nicht rally_plus_entry)")
    if "rally" in sub.columns:
        rp = (sub["rally"] == 1).mean() * 100
        print(f"  rally=1:           {100*rp:.1f}%")

    sub["year"] = pd.to_datetime(sub["Date"]).dt.year
    print("  Nach Jahr:")
    for yr, g in sub.groupby("year"):
        rr = g["ret_mean_5"].astype(float)
        print(f"    {int(yr)}: n={len(g):4d}, mean={100*rr.mean():+.3f}%, win={100*(rr > 0).mean():.1f}%")


def main() -> None:
    art = joblib.load(ART)
    meta_best = float(art.get("meta_optuna_best_value", float("nan")))
    meta_ret = meta_best - 1.0
    thr_art = float(art.get("best_threshold", float("nan")))

    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce").dt.normalize()
    mc = mc.dropna(subset=["Date"])

    final_set = final_date_set()
    mc_final = mc[mc["Date"].isin(final_set)].copy()

    payload = json.loads(SIGNALS_JSON.read_text(encoding="utf-8")) if SIGNALS_JSON.exists() else {}
    json_thr = payload.get("threshold")
    json_ho = payload.get("signals_holdout_final") or []
    ho_keys = {(str(s["ticker"]), str(s["date"])[:10]) for s in json_ho}
    mc_final["key"] = list(zip(mc_final["ticker"].astype(str), mc_final["Date"].dt.strftime("%Y-%m-%d")))
    mc_json = mc_final[mc_final["key"].isin(ho_keys)] if ho_keys else mc_final

    d0 = min(final_set).date()
    d1 = max(final_set).date()

    print("=" * 60)
    print("FINAL OOS — Auswertung nach Pipeline-Ende")
    print("=" * 60)
    print(f"FINAL-Kalender: {d0} .. {d1} ({len(final_set)} Handelstage)")
    print(f"Artefakt best_threshold: {thr_art:.3f}")
    print(f"signals.json threshold:  {json_thr}")
    print(f"signals_holdout_final:   {len(json_ho):,} Signale")
    print(f"master_complete gesamt:  {len(mc):,} Zeilen")
    print(f"master_complete FINAL:   {len(mc_final):,} Zeilen")
    print(f"Schnittmenge JSON/FINAL: {len(mc_json):,} Zeilen")

    print("\n--- Meta-Optuna (META-CV, signal_mean_return) ---")
    print(f"  meta_optuna_best_value: {meta_best:.6f}")
    print(f"  approx mean return:     {100*meta_ret:.3f}%")

    report_block("FINAL (alle master_complete im FINAL-Fenster)", mc_final, meta_ret)
    report_block("FINAL (signals.json holdout_final)", mc_json, meta_ret)

    # Signale ohne Return (Horizont nicht vollstaendig)
    no_ret = mc_final[mc_final["ret_mean_5"].isna()]
    if len(no_ret):
        print(f"\n  Ohne ret_mean_5 (Horizont unvollstaendig): {len(no_ret):,} Signale")


if __name__ == "__main__":
    main()
