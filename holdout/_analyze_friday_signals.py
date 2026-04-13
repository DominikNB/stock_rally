"""One-off: Friday vs Mon–Thu signals in master_complete.csv (forward rets from next open)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "master_complete.csv"


def main() -> None:
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["Date"])
    if "entry_date" in df.columns:
        df["entry_parsed"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["dow"] = df["Date"].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _dow_name(i: float) -> str:
        if pd.isna(i):
            return "NA"
        ii = int(i)
        return dow_names[ii] if 0 <= ii <= 6 else "NA"

    df["dow_name"] = df["dow"].map(_dow_name)

    horizons = ["ret_2d", "ret_4d", "ret_6d", "ret_8d", "ret_10d", "ret_mean_5"]
    cols = [c for c in horizons if c in df.columns]
    print("rows", len(df), "ret cols:", cols)
    print("signal weekday counts:")
    print(df["dow_name"].value_counts().sort_index().to_string())

    # Erfolgreiche Forward-Rechnung: ``fwd_error`` ist leer (NaN), siehe build_holdout_signals_master.
    valid = df["fwd_error"].isna() if "fwd_error" in df.columns else pd.Series(True, index=df.index)
    fri = (df["dow"] == 4) & valid
    nonfri = df["dow"].isin([0, 1, 2, 3]) & valid

    def summarize(sub: pd.DataFrame, label: str) -> None:
        print(f"\n=== {label} (n={len(sub)}) ===")
        for c in cols:
            x = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(x) == 0:
                print(f"{c:12s} no data")
                continue
            print(
                f"{c:12s} mean={x.mean() * 100:+.3f}%  median={x.median() * 100:+.3f}%  "
                f"win%={(x > 0).mean() * 100:.1f}%  n={len(x)}"
            )

    summarize(df[fri], "Friday signal -> entry next session OPEN (usually Monday)")
    summarize(df[nonfri], "Mon-Thu signal -> entry next session OPEN")

    if "entry_parsed" in df.columns:
        ff = df[fri & df["entry_parsed"].notna()].copy()
        ff["gap_days"] = (ff["entry_parsed"] - ff["Date"]).dt.days
        print("\nFriday signals: calendar days signal Date -> entry_date:")
        print(ff["gap_days"].value_counts().sort_index().head(12).to_string())

    # Difference of means (Friday - other), same horizons
    print("\n=== Friday minus Mon-Thu (mean return, percentage points) ===")
    for c in cols:
        a = pd.to_numeric(df.loc[fri, c], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[nonfri, c], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            print(f"{c}: insufficient n")
            continue
        diff = (a.mean() - b.mean()) * 100
        print(f"{c:12s} diff_mean={diff:+.3f} pp  (Fri n={len(a)}, other n={len(b)})")


if __name__ == "__main__":
    main()
