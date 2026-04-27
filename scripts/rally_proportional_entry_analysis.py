"""
Rechnet vorgeschlagene Einstiegs-Tage k(L) = max(1, round(alpha * L)) pro gemergtes
rally-Segment; liefert Verteilungen und Summen (aktueller FIXED_Y / create_target).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import create_target


def rally_segments(rally: np.ndarray) -> list[tuple[int, int, int]]:
    """Pro Ticker: Liste (start, end, L) für jedes maximale rally==1-Intervall."""
    r = np.asarray(rally, dtype=np.int8)
    out: list[tuple[int, int, int]] = []
    i, n = 0, len(r)
    while i < n:
        if r[i] != 1:
            i += 1
            continue
        j = i
        while j < n and r[j] == 1:
            j += 1
        end = j - 1
        L = end - i + 1
        out.append((i, end, L))
        i = j
    return out


def k_prop(L: int, alpha: float) -> int:
    return max(1, int(round(alpha * float(L))))


def main() -> None:
    alpha = 0.3
    print(
        "FIXED_Y / create_target (OPT_OPTIMIZE_Y_TARGETS=%s)"
        % getattr(cfg, "OPT_OPTIMIZE_Y_TARGETS", "?"),
        flush=True,
    )
    print(
        "Regel: k(L) = max(1, round(%.2f * L)) — z. B. L=10 -> %d, L=20 -> %d"
        % (alpha, k_prop(10, alpha), k_prop(20, alpha)),
        flush=True,
    )
    print(flush=True)

    df = load_stock_data(tickers=None)
    df = create_target(df, quiet=True)

    rows: list[dict] = []
    for t, g in df.groupby("ticker"):
        for start, _end, L in rally_segments(g["rally"].values):
            k = k_prop(L, alpha)
            rows.append(
                {
                    "ticker": t,
                    "L": L,
                    "k": k,
                    "k_over_L": k / L,
                }
            )
    s = pd.DataFrame(rows)
    nseg = len(s)
    if nseg == 0:
        print("Keine Segmente.")
        return

    print("=== Segmente (gemergt, rally==1) ===")
    print("Anzahl:", nseg)
    print("L: min / max / mean / median:", s["L"].min(), s["L"].max(), s["L"].mean(), s["L"].median())
    for p in (10, 25, 50, 75, 90, 95, 99):
        print("  L p%02d:" % p, float(np.percentile(s["L"], p)))
    print()
    print("k = max(1, round(%.1f*L)):" % alpha)
    print("k: min / max / mean / median:", s["k"].min(), s["k"].max(), s["k"].mean(), s["k"].median())
    for p in (10, 25, 50, 75, 90, 95, 99):
        print("  k p%02d:" % p, float(np.percentile(s["k"], p)))
    print()
    print("k/L (Anteil des Segments als Einstiegs-„Front“):")
    print("  mean / median:", float(s["k_over_L"].mean()), float(s["k_over_L"].median()))
    print("  p90 / p95:", float(np.percentile(s["k_over_L"], 90)), float(np.percentile(s["k_over_L"], 95)))
    print()
    print("Wenn pro Segment die ersten k grünen Tage positiv wären: Summe k =", int(s["k"].sum()))
    print("  (Vergleich: Summe L = Tage in rally gesamt =", int(s["L"].sum()), ")")
    print("  Anteil Tage, die in „Einstiegs-Front“ lägen: %.2f%%" % (100.0 * s["k"].sum() / s["L"].sum()))
    print()

    print("Verteilung k (Häufigkeit):")
    vc = s["k"].value_counts().sort_index()
    for kk, c in vc.items():
        print("  k=%2d: %5d  (%5.1f%%)" % (kk, c, 100.0 * c / nseg))
    print()

    print("L-Bucket -> mittleres L, mittleres k, mittleres k/L:")
    s["Lbin2"] = np.where(
        s["L"] <= 9,
        "5-9",
        np.where(
            s["L"] <= 14,
            "10-14",
            np.where(
                s["L"] <= 19,
                "15-19",
                np.where(
                    s["L"] <= 24,
                    "20-24",
                    np.where(s["L"] <= 29, "25-29", "30+"),
                ),
            ),
        ),
    )
    for lab in ["5-9", "10-14", "15-19", "20-24", "25-29", "30+"]:
        sub = s[s["Lbin2"] == lab]
        if len(sub) == 0:
            continue
        print(
            "  %5s: n=%5d  mean L=%.1f  mean k=%.2f  mean k/L=%.3f"
            % (lab, len(sub), sub["L"].mean(), sub["k"].mean(), (sub["k"] / sub["L"]).mean()),
        )
    print()

    for a in (0.25, 0.3, 0.35):
        ks = s["L"].apply(lambda L: k_prop(int(L), a))
        print("alpha=%.2f: mean(k)=%.2f, sum(k)=%d, mean(k/L)=%.3f" % (a, ks.mean(), int(ks.sum()), (ks / s["L"]).mean()))


if __name__ == "__main__":
    main()
