"""
Kerndichteschätzung (KDE) der Forward-Renditen aus data/master_complete.csv.

python -m holdout.plot_holdout_returns_kde
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "master_complete.csv"
OUT_PATH = ROOT / "figures" / "holdout_returns_kde.png"
HORIZONS = (2, 4, 6, 8, 10)


def _kde_line(x: np.ndarray, color: str, label: str, ax) -> None:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return
    kde = gaussian_kde(x)
    lo, hi = np.percentile(x, [0.5, 99.5])
    pad = (hi - lo) * 0.15 + 1e-6
    grid = np.linspace(lo - pad, hi + pad, 400)
    ax.plot(grid, kde(grid), color=color, label=label, lw=1.8)


def main() -> None:
    if not CSV_PATH.is_file():
        print(f"Fehlt: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    ret_cols = [f"ret_{h}d" for h in HORIZONS]
    for c in ret_cols + ["ret_mean_5"]:
        if c not in df.columns:
            print(f"Spalte fehlt: {c}", file=sys.stderr)
            sys.exit(1)

    df = df.dropna(subset=ret_cols + ["ret_mean_5"], how="any")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(HORIZONS)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (h, col) in enumerate(zip(HORIZONS, ret_cols)):
        _kde_line(df[col].to_numpy(), colors[i], f"H={h} Tage", ax)
    _kde_line(df["ret_mean_5"].to_numpy(), "#c0392b", "Mittel (5 Horizonte)", ax)

    ax.axvline(0.0, color="0.5", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Rendite (Kauf Open → Verkauf Close)")
    ax.set_ylabel("Kerndichte")
    ax.set_title("Verteilung der Forward-Renditen (Holdout, vollständige Zeilen)")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
