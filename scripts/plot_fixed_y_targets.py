from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data import load_stock_data
from lib.stock_rally_v10.target import create_target, override_fixed_y_config_for_grid

# Hohe Auflösung für Zoom im Bildbetrachter (~4800 px Breite bei figsize 16×300 dpi)
_PLOT_DPI = 300
_FIGSIZE = (16, 6.2)


def _default_tickers() -> list[str]:
    return [
        "ALB",
        "NVDA",
        "MSFT",
        "SOL-USD",
        "ADA-USD",
        "R3NK.DE",
        "SAP.DE",
        "JPM",
    ]


def _plot_one_ticker(
    sub: pd.DataFrame, ticker: str, out_png: Path, *, dpi: int = _PLOT_DPI
) -> None:
    sub = sub.sort_values("Date").copy()
    d = pd.to_datetime(sub["Date"], errors="coerce")
    close = pd.to_numeric(sub["close"], errors="coerce")
    rally = pd.to_numeric(sub["rally"], errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(sub["target"], errors="coerce").fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(d, close, color="#42a5f5", linewidth=1.4, label="close")

    # Rally-Phasen als grüne Hinterlegung.
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
        label="target=1 (entry day t0)",
        zorder=5,
    )

    n_target = int(t_mask.sum())
    n_rally = int((rally == 1).sum())
    ax.set_title(f"{ticker} — rally(days)={n_rally}, target(days)={n_target}", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot fixed-Y targets/rally for selected tickers")
    p.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers; default uses a diverse preset",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="PNG-Ausgabeordner (Default: figures/target_debug bzw. bei --no-dip-below-entry figures/target_debug_no_dip)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=_PLOT_DPI,
        help=f"PNG-Auflösung (Default {_PLOT_DPI})",
    )
    p.add_argument(
        "--no-dip-below-entry",
        action="store_true",
        help="FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION=0.0 (streng, kein Close unter Entry-Open; sonst wie cfg); "
        "Default-outdir figures/target_debug_no_dip",
    )
    args = p.parse_args()

    if args.outdir is not None and str(args.outdir).strip() != "":
        outdir_str = str(args.outdir).strip()
    elif args.no_dip_below_entry:
        outdir_str = "figures/target_debug_no_dip"
    else:
        outdir_str = "figures/target_debug"

    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _default_tickers()

    print(f"[TargetPlot] Tickers={tickers}", flush=True)
    if args.no_dip_below_entry:
        print("[TargetPlot] max_dip_below_entry=0.0 (Close nicht unter Entry-Open bis Ziel)", flush=True)

    df_raw = load_stock_data(tickers=tickers, start=cfg.START_DATE, end=cfg.END_DATE)
    if args.no_dip_below_entry:
        with override_fixed_y_config_for_grid(
            FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION=0.0,
        ):
            print(
                f"[TargetPlot] FIXED_Y_LABEL_MODE={cfg.fixed_y_label_mode()} | "
                f"rule={cfg.describe_target_rule_fixed_bands().splitlines()[0]}",
                flush=True,
            )
            df_t = create_target(df_raw)
    else:
        print(
            f"[TargetPlot] FIXED_Y_LABEL_MODE={cfg.fixed_y_label_mode()} | "
            f"rule={cfg.describe_target_rule_fixed_bands().splitlines()[0]}",
            flush=True,
        )
        df_t = create_target(df_raw)
    outdir = Path(outdir_str)

    made = 0
    for ticker in tickers:
        sub = df_t[df_t["ticker"] == ticker].copy()
        if sub.empty:
            print(f"[TargetPlot] skip {ticker}: no rows", flush=True)
            continue
        out_png = outdir / f"target_plot_{ticker.replace('/', '_')}.png"
        _plot_one_ticker(sub, ticker, out_png, dpi=int(args.dpi))
        print(f"[TargetPlot] wrote {out_png}", flush=True)
        made += 1

    print(f"[TargetPlot] done: {made} plot(s) in {outdir}", flush=True)


if __name__ == "__main__":
    main()
