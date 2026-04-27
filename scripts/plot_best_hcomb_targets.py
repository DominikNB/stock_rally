"""
Target-Plots für die Gitterzeile mit höchstem ``homogeneity_score`` (H_comb) aus
``data/target_param_grid_homogeneity.csv`` — gleicher Stil wie ``plot_fixed_y_targets.py``,
Titelzeile mit Kennzahlen aus der CSV.

  .venv/Scripts/python.exe scripts/plot_best_hcomb_targets.py
  .venv/Scripts/python.exe scripts/plot_best_hcomb_targets.py --require-min-pos 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import create_target, override_fixed_y_config_for_grid

_PLOT_DPI = 300
_FIGSIZE = (16, 6.2)

_DEFAULT_TICKERS = [
    "ALB",
    "NVDA",
    "MSFT",
    "SOL-USD",
    "ADA-USD",
    "R3NK.DE",
    "SAP.DE",
    "JPM",
]


def _snapshot_all_fixed_y() -> dict[str, object]:
    import lib.stock_rally_v10.config as cmod

    return {k: getattr(cmod, k) for k in dir(cmod) if k.startswith("FIXED_Y_")}


def _row_to_overrides(row: pd.Series) -> dict[str, object]:
    o: dict[str, object] = {
        "OPT_OPTIMIZE_Y_TARGETS": False,
        "FIXED_Y_LABEL_MODE": "entry_direct",
        "FIXED_Y_RALLY_THRESHOLD": float(row["FIXED_Y_RALLY_THRESHOLD"]),
        "FIXED_Y_WINDOW_MIN": int(row["FIXED_Y_WINDOW_MIN"]),
        "FIXED_Y_WINDOW_MAX": int(row["FIXED_Y_WINDOW_MAX"]),
        "FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY": bool(
            row["FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY"]
        ),
    }
    if "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION" in row.index and pd.notna(
        row.get("FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION")
    ):
        o["FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION"] = float(
            row["FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION"]
        )
    elif "FIXED_Y_REQUIRE_NO_DIP_BELOW_ENTRY_UNTIL_THRESHOLD" in row.index:
        o["FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION"] = (
            0.0 if bool(row["FIXED_Y_REQUIRE_NO_DIP_BELOW_ENTRY_UNTIL_THRESHOLD"]) else 1.0
        )
    else:
        o["FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION"] = float(
            getattr(cfg, "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION", 0.01)
        )
    return o


def _plot_one(
    sub: pd.DataFrame,
    ticker: str,
    out_png: Path,
    *,
    title_line2: str,
    dpi: int = _PLOT_DPI,
) -> None:
    sub = sub.sort_values("Date").copy()
    d = pd.to_datetime(sub["Date"], errors="coerce")
    close = pd.to_numeric(sub["close"], errors="coerce")
    rally = pd.to_numeric(sub["rally"], errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(sub["target"], errors="coerce").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(d, close, color="#42a5f5", linewidth=1.4, label="close")
    in_seg = False
    seg_start = None
    for i in range(len(sub)):
        is_r = rally.iloc[i] == 1
        if is_r and not in_seg:
            in_seg = True
            seg_start = d.iloc[i]
        if in_seg and (not is_r or i == len(sub) - 1):
            seg_end = d.iloc[i] if is_r and i == len(sub) - 1 else d.iloc[i - 1]
            ax.axvspan(seg_start, seg_end, color="#66bb6a", alpha=0.12)
            in_seg = False
            seg_start = None

    t_mask = target == 1
    ax.scatter(
        d[t_mask],
        close[t_mask],
        color="#ef5350",
        s=16,
        marker="o",
        alpha=0.42,
        linewidths=0.0,
        label="target=1 (t0, entry_direct)",
        zorder=5,
    )
    n_target = int(t_mask.sum())
    n_rally = int((rally == 1).sum())
    ax.set_title(
        f"{ticker} — rally={n_rally}d, target={n_target}d\n{title_line2}",
        fontsize=11,
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Target-Plots für beste H_comb-Zeile aus der Homogenitäts-CSV"
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "data" / "target_param_grid_homogeneity.csv",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=_ROOT / "figures" / "target_debug_best_hcomb",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=_PLOT_DPI,
        help=f"PNG-Auflösung (Default {_PLOT_DPI})",
    )
    ap.add_argument("--tickers", type=str, default="", help="Komma-Liste, sonst Preset")
    ap.add_argument(
        "--require-min-pos",
        type=float,
        default=None,
        help="Optional: nur Zeilen mit pos_rate >= diesem Wert (z. B. 0.15)",
    )
    ap.add_argument(
        "--csv-idx",
        type=int,
        default=None,
        help="Gitterzeile per Spalte idx wählen (statt max. homogeneity_score)",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"CSV fehlt: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if "homogeneity_score" not in df.columns:
        print("Spalte homogeneity_score fehlt.", file=sys.stderr)
        return 1

    if args.csv_idx is not None:
        sel = df[df["idx"] == int(args.csv_idx)]
        if sel.empty:
            print(f"Keine Zeile mit idx={int(args.csv_idx)}.", file=sys.stderr)
            return 1
        best = sel.iloc[0]
    else:
        work = df.copy()
        if args.require_min_pos is not None:
            mp = float(args.require_min_pos)
            work = work[work["pos_rate"] >= mp - 1e-9]
            if work.empty:
                print(
                    f"Keine Zeile mit pos_rate>={mp}. Ohne --require-min-pos erneut versuchen.",
                    file=sys.stderr,
                )
                return 1

        best = work.sort_values("homogeneity_score", ascending=False).iloc[0]
    idx = int(best.get("idx", -1))
    ovr = {**_snapshot_all_fixed_y(), **_row_to_overrides(best)}

    h = float(best["homogeneity_score"])
    h_in = float(best["inner_homogeneity"]) if "inner_homogeneity" in best else float("nan")
    uq = float(best["uniqueness"]) if "uniqueness" in best and pd.notna(best["uniqueness"]) else float("nan")
    pr = float(best["pos_rate"])
    lbl = str(best.get("row_label", ""))

    n_pos_v = int(best["n_pos"]) if "n_pos" in best.index and pd.notna(best.get("n_pos")) else -1
    print(
        f"Gewählte Zeile (idx={idx}): H_comb={h:.4f}  H_in={h_in:.4f}  uniqueness={uq:.4f}  "
        f"pos_rate={pr:.1%}  n_pos={n_pos_v}",
        flush=True,
    )
    print(f"  {lbl}", flush=True)

    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list(_DEFAULT_TICKERS)

    with override_fixed_y_config_for_grid(**ovr):
        df_raw = load_stock_data(
            tickers=tickers, start=cfg.START_DATE, end=cfg.END_DATE
        )
        dft = create_target(df_raw)

    npos_s = f"n_pos={int(best['n_pos'])}  " if "n_pos" in best.index and pd.notna(best.get("n_pos")) else ""
    title_line2 = f"H_comb={h:.4f}  {npos_s}pos={pr:.1%}  {lbl}"

    outdir: Path = args.outdir
    made = 0
    for t in tickers:
        sub = dft[dft["ticker"] == t]
        if sub.empty:
            print(f"skip {t}: keine Zeilen", flush=True)
            continue
        safe = t.replace("/", "_")
        out_png = outdir / f"target_plot_{safe}.png"
        _plot_one(sub, t, out_png, title_line2=title_line2, dpi=int(args.dpi))
        print(f"wrote {out_png}", flush=True)
        made += 1

    with open(outdir / "best_row_meta.txt", "w", encoding="utf-8") as f:
        f.write(
            f"idx={idx}\nhomogeneity_score={h}\ninner_homogeneity={h_in}\nuniqueness={uq}\n"
        )
        if "n_pos" in best.index:
            f.write(f"n_pos={int(best['n_pos'])}\n")
        f.write(f"pos_rate={pr}\nrow_label={lbl}\n")
        f.write(f"outdir={outdir}\n")
    print(f"done: {made} plot(s) in {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
