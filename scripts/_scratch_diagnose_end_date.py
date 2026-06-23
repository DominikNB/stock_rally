"""Diagnose END_DATE, Feature-Kalender, FINAL-Fenster und Signalzählung."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.pipeline_runner import bind_step_functions
from lib.stock_rally_v10.data_and_split import run_data_download_and_split


def main() -> None:
    bind_step_functions()
    print("=== Config ===")
    print("END_DATE:", cfg.END_DATE)
    print("TRAIN_END_DATE:", cfg.TRAIN_END_DATE)
    print("SCORING_ONLY:", cfg.SCORING_ONLY)
    print("SCORING_ARTIFACT_PATH:", cfg.SCORING_ARTIFACT_PATH)

    run_data_download_and_split()

    df = cfg.df_features
    dc = pd.to_datetime(df["Date"])
    print("\n=== df_features ===")
    print("rows:", len(df), "tickers:", df["ticker"].nunique())
    print("Date min:", dc.min().date(), "max:", dc.max().date())

    df_final = cfg.df_final
    dcf = pd.to_datetime(df_final["Date"])
    print("\n=== df_final (OOS-Fenster) ===")
    print("rows:", len(df_final))
    print("Date min:", dcf.min().date(), "max:", dcf.max().date())
    print("unique days:", dcf.dt.normalize().nunique())

    # Grobe Signal-Schätzung: prob >= threshold an FINAL-Tagen
    thr = float(getattr(cfg, "best_threshold", 0.8) or 0.8)
    if "prob" not in df.columns:
        print("\n(prob noch nicht berechnet — nur Kalender-Diagnose)")
        return

    keys_final = set(
        zip(
            df_final["ticker"].astype(str).str.strip(),
            dcf.dt.strftime("%Y-%m-%d"),
        )
    )
    sub = df[df["prob"] >= thr].copy()
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
    sub["ticker"] = sub["ticker"].astype(str).str.strip()
    in_final = sub[sub.apply(lambda r: (r["ticker"], r["Date"]) in keys_final, axis=1)]
    print(f"\n=== Grobe Treffer prob>={thr:.3f} in FINAL-Kalender ===")
    print("Treffer gesamt (ohne Cooldown/Filter):", len(in_final))
    if len(in_final):
        c = in_final["Date"].value_counts().sort_index()
        print("Top 8 neueste Signaltage:")
        for d, n in c.sort_index(ascending=False).head(8).items():
            print(f"  {d}: {n}")

    sj = ROOT / "docs" / "signals.json"
    if sj.is_file():
        import json

        data = json.loads(sj.read_text(encoding="utf-8"))
        sigs = data.get("signals") or []
        dates = pd.Series([s.get("date", "")[:10] for s in sigs])
        print(f"\n=== docs/signals.json (aktuell) ===")
        print("generated:", data.get("generated"), "count:", len(sigs))
        print("max date:", dates.max() if len(dates) else "—")


if __name__ == "__main__":
    main()
