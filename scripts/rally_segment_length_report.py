"""Report length distribution of contiguous rally==1 segments (current FIXED_Y config)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import create_target


def seg_lengths(rally: np.ndarray) -> list[int]:
    r = np.asarray(rally, dtype=np.int8)
    out: list[int] = []
    i, n = 0, len(r)
    while i < n:
        if r[i] != 1:
            i += 1
            continue
        j = i
        while j < n and r[j] == 1:
            j += 1
        out.append(j - i)
        i = j
    return out


def main() -> None:
    print(
        "FIXED_Y: w",
        cfg.FIXED_Y_WINDOW_MIN,
        "-",
        cfg.FIXED_Y_WINDOW_MAX,
        "rt=",
        cfg.FIXED_Y_RALLY_THRESHOLD,
        flush=True,
    )
    df = load_stock_data(tickers=None)
    print("Zeilen", len(df), "Ticker (unique)", df["ticker"].nunique(), flush=True)
    df = create_target(df, quiet=True)
    lens_list: list[int] = []
    per_t: dict[str, tuple[int, int, int]] = {}
    for t, g in df.groupby("ticker"):
        L = seg_lengths(g["rally"].values)
        per_t[t] = (len(L), max(L) if L else 0, int(sum(L)) if L else 0)
        lens_list.extend(L)
    lens = np.array(lens_list, dtype=np.int64)
    n = len(lens)
    print()
    print("=== Zusammenhängende rally-Segmente (rally==1) über alle Ticker ===")
    print("Anzahl Segmente:", n)
    print("min / max Länge (Handelstage):", int(lens.min()), "/", int(lens.max()))
    print("Mittel / Median:", float(lens.mean()), "/", float(np.median(lens)))
    for p in (10, 25, 50, 75, 90, 95, 99):
        print("  p%02d:" % p, int(np.percentile(lens, p)))
    print()
    print("Anteil (%) / Anzahl — Längen-Buckets:")
    buckets: list[tuple[str, np.ndarray]] = [
        ("5-9 (bis ein max-w-Fenster w/o Merge)", (lens >= 5) & (lens <= 9)),
        ("10-19", (lens >= 10) & (lens <= 19)),
        ("20-29", (lens >= 20) & (lens <= 29)),
        ("30-49", (lens >= 30) & (lens <= 49)),
        ("50-99", (lens >= 50) & (lens <= 99)),
        ("100+", lens >= 100),
    ]
    for name, m in buckets:
        c = int(m.sum())
        if c or name.startswith("5-9"):
            print(f"  {name:38s} {100.0 * c / n:5.1f}%  N={c}")
    print()
    for thr in (30, 50, 75, 100):
        c = int((lens >= thr).sum())
        print(f"  L>={thr}:", c, f"({100.0 * c / n:.1f}%)")
    print()
    print("Höchste max. Segmentlänge pro Ticker (Top 15):")
    for t, (nseg, mx, s) in sorted(per_t.items(), key=lambda x: -x[1][1])[:15]:
        print(f"  {t}: max {mx}d, {nseg} Segmente, Tage in rally gesamt={s}")


if __name__ == "__main__":
    main()
