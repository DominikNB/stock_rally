#!/usr/bin/env python3
"""
10 Plots zur **aktuellen Target-Definition** (``target`` / ``rally``) wie in
``target.rebuild_target_for_train`` / ``fixed_y_rule_params`` bzw. Optuna-Y-Parametern.

Quelle:

- ``cfg.df_train`` (nach ``run_data_download_and_split``), oder
- ``--parquet=…``, oder
- ``--synthetic`` (sofort: Demo-Kursreihen + gleiche Target-Logik wie in cfg).

    .venv\\Scripts\\python scripts/plot_target_definition_tour.py
    .venv\\Scripts\\python scripts/plot_target_definition_tour.py --synthetic

Ausgabe: ``reports/target_definition/target_tour.pdf``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.target import rebuild_target_for_train

OUT_DIR = _ROOT / "reports" / "target_definition"


def _prepare_df_train(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Minimalspalten + sortiert; ``rebuild_target_for_train`` schreibt ``target``/``rally``."""
    cols = ["Date", "ticker", "close"]
    if "open" in df_raw.columns:
        cols.append("open")
    out = df_raw[cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "ticker", "close"])
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return out


def _rebuild_current(df: pd.DataFrame) -> pd.DataFrame:
    if cfg.opt_optimize_y_targets():
        return rebuild_target_for_train(
            df,
            int(cfg.LEAD_DAYS),
            int(cfg.ENTRY_DAYS),
            return_window=int(cfg.RETURN_WINDOW),
            rally_threshold=float(cfg.RALLY_THRESHOLD),
            min_rally_tail_days=int(getattr(cfg, "MIN_RALLY_TAIL_DAYS", 5)),
        )
    return rebuild_target_for_train(
        df,
        int(getattr(cfg, "FIXED_Y_LEAD_DAYS", 2)),
        int(getattr(cfg, "FIXED_Y_ENTRY_DAYS", 1)),
        return_window=None,
        rally_threshold=None,
        min_rally_tail_days=None,
    )


def _summary_lines() -> list[str]:
    lines = [
        "Aktuelle Target-Konfiguration (Auszug)",
        "",
        f"OPT_OPTIMIZE_Y_TARGETS = {cfg.opt_optimize_y_targets()}",
    ]
    if cfg.opt_optimize_y_targets():
        lines += [
            f"RETURN_WINDOW = {getattr(cfg, 'RETURN_WINDOW', None)}",
            f"RALLY_THRESHOLD = {float(getattr(cfg, 'RALLY_THRESHOLD', 0)):.4f}",
            f"LEAD_DAYS = {getattr(cfg, 'LEAD_DAYS', None)}",
            f"ENTRY_DAYS = {getattr(cfg, 'ENTRY_DAYS', None)}",
            f"MIN_RALLY_TAIL_DAYS = {getattr(cfg, 'MIN_RALLY_TAIL_DAYS', None)}",
        ]
    else:
        w_lo, w_hi, rt, sp, ld, ed, tex = cfg.fixed_y_rule_params()
        lines += [
            f"FIXED_Y Fenster w ∈ [{w_lo}, {w_hi}], RALLY_THRESHOLD = {rt:.4f}",
            f"Segment-Split = {sp}d, lead={ld}, entry={ed}, tail_excl={tex}",
            f"label_mode = {cfg.fixed_y_label_mode()}",
            f"strict_daily_up = {cfg.fixed_y_require_strict_daily_up_in_rally()}",
            f"max_dip_below_entry = {cfg.fixed_y_max_dip_below_entry_fraction():.4f}",
            f"long_segment: {cfg.fixed_y_long_segment_label_params()}",
        ]
    return lines


def _build_synthetic_price_panel(*, n_days: int = 720, n_tickers: int = 6) -> pd.DataFrame:
    """Kunst-Kursreihen (gleiche cfg-Target-Regel), damit das Skript ohne Download läuft."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-06-01", periods=n_days)
    parts: list[pd.DataFrame] = []
    for k in range(n_tickers):
        sym = f"SYN_{k}"
        c = 80.0 + rng.standard_normal(n_days).cumsum() * 0.35
        c = np.maximum(c, 1.0)
        o = np.roll(c, 1)
        o[0] = float(c[0])
        # Ein klarer Aufwärtssprint pro Ticker (unterschiedlicher Start), damit Rally/Target nicht leer bleiben
        mid = 140 + k * 55
        span = min(35, n_days - mid - 15)
        if span > 12 and mid + span < n_days:
            base = float(np.mean(c[mid - 5 : mid]))
            ramp = np.linspace(0.0, 0.12, span)
            c[mid : mid + span] = base * (1.0 + ramp)
            o[mid : mid + span] = np.maximum(o[mid : mid + span], c[mid : mid + span] * 0.998)
        parts.append(pd.DataFrame({"Date": dates, "ticker": sym, "open": o, "close": c}))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Target-Definition als PDF-Plots")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Demo-Panel ohne df_train (sofort lauffähig).",
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Optional: Parquet mit Spalten Date,ticker,close[,open] statt cfg.df_train",
    )
    args = ap.parse_args()
    mode_note = ""

    if args.synthetic:
        df_raw = _build_synthetic_price_panel()
        mode_note = "Datenquelle: --synthetic (Demo-Kurse, gleiche cfg-Target-Regeln)"
    elif args.parquet is not None:
        df_raw = pd.read_parquet(args.parquet)
        mode_note = f"Datenquelle: --parquet={args.parquet}"
    else:
        df_raw = getattr(cfg, "df_train", None)
        if df_raw is None or len(df_raw) == 0:
            print(
                "Kein df_train: bitte zuerst run_data_download_and_split() ausführen, "
                "--parquet=… nutzen, oder --synthetic für Demo-PDF.",
                file=sys.stderr,
            )
            sys.exit(1)
        mode_note = "Datenquelle: cfg.df_train"

    df = _prepare_df_train(df_raw)
    df = _rebuild_current(df)
    if "rally" not in df.columns:
        df["rally"] = np.int8(0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "target_tour.pdf"

    pos = df["target"].to_numpy(dtype=np.float64)
    pos_rate = float(pos.mean())

    with PdfPages(pdf_path) as pdf:
        # 1 — Gesamt-Verteilung target
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["target=0", "target=1"], [(1 - pos_rate), pos_rate], color=["#4472c4", "#ed7d31"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Anteil")
        ax.set_title(f"Global: positives Target = {pos_rate:.2%}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2 — rally vs target (Mittelwerte)
        fig, ax = plt.subplots(figsize=(6, 5))
        r = df["rally"].to_numpy(dtype=np.float64)
        ax.bar(
            ["rally|t=0", "rally|t=1", "target|r=0", "target|r=1"],
            [
                float((r[pos == 0] == 1).mean()) if (pos == 0).any() else 0.0,
                float((r[pos == 1] == 1).mean()) if (pos == 1).any() else 0.0,
                float((pos[r == 0] == 1).mean()) if (r == 0).any() else 0.0,
                float((pos[r == 1] == 1).mean()) if (r == 1).any() else 0.0,
            ],
            color=["#5b9bd5", "#f4b183", "#9e9e9e", "#70ad47"],
        )
        ax.set_ylabel("Bedingter Mittelwert (Anteil =1)")
        ax.set_title("Kopplung rally ↔ target")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3 — Positivrate nach Kalender-Monat
        fig, ax = plt.subplots(figsize=(9, 4))
        df["_ym"] = df["Date"].dt.to_period("M").astype(str)
        g = df.groupby("_ym", sort=True)["target"].mean()
        ax.plot(range(len(g)), g.values, "o-", color="#2e75b6")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(list(g.index), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mittel target")
        ax.set_title("Positivrate pro Monat (alle Ticker)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4 — Positivrate pro Ticker (Top 25 nach Volumen Zeilen)
        fig, ax = plt.subplots(figsize=(9, 6))
        tc = df.groupby("ticker").agg(n=("target", "size"), p=("target", "mean")).reset_index()
        tc = tc.sort_values("n", ascending=False).head(25)
        ax.barh(tc["ticker"].astype(str), tc["p"], color="#c55a11")
        ax.set_xlabel("Anteil target=1")
        ax.set_title("25 Ticker mit meisten Zeilen: Positivrate")
        ax.invert_yaxis()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5 — Längen aufeinanderfolgender target=1 (Run-Length), alle Ticker
        fig, ax = plt.subplots(figsize=(8, 5))
        runs: list[int] = []
        for _, sub in df.groupby("ticker"):
            t = sub["target"].to_numpy(dtype=np.int8)
            if not t.any():
                continue
            i = 0
            nloc = len(t)
            while i < nloc:
                if t[i] != 1:
                    i += 1
                    continue
                j = i
                while j < nloc and t[j] == 1:
                    j += 1
                runs.append(j - i)
                i = j
        if runs:
            ax.hist(runs, bins=min(40, max(5, int(np.sqrt(len(runs))))), color="#7030a0", edgecolor="white")
            ax.set_xlabel("Aufeinanderfolgende Handelstage mit target=1")
            ax.set_ylabel("Häufigkeit")
            ax.set_title("Run-Längen positiver Labels")
        else:
            ax.text(0.5, 0.5, "Keine positiven Runs", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 6–8 — Drei Beispiel-Ticker: Kurs + rally/target
        rng = np.random.default_rng(7)
        tickers_rank = tc["ticker"].tolist()
        if not tickers_rank:
            tickers_rank = df["ticker"].drop_duplicates().tolist()
        pick = tickers_rank[:3] if len(tickers_rank) >= 3 else tickers_rank
        if len(pick) < 3 and len(df["ticker"].unique()) >= 3:
            extra = [x for x in df["ticker"].unique() if x not in pick]
            while len(pick) < 3 and extra:
                pick.append(extra.pop(rng.integers(0, len(extra))))

        for ti in pick[:3]:
            sub = df[df["ticker"] == ti].sort_values("Date")
            if len(sub) < 30:
                continue
            # letzte ~400 Handelstage
            sub = sub.tail(400)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sub["Date"], sub["close"], color="black", lw=1, label="close")
            y0, y1 = float(sub["close"].min()), float(sub["close"].max())
            dr = sub["Date"].to_numpy()
            for name, series, color, alpha in [
                ("rally", sub["rally"], "#92d050", 0.25),
                ("target", sub["target"], "#ffc000", 0.35),
            ]:
                mask = series.to_numpy(dtype=np.int8) == 1
                if mask.any():
                    ax.fill_between(
                        dr[mask],
                        y0,
                        y1,
                        color=color,
                        alpha=alpha,
                        label=name,
                    )
            ax.set_title(f"Ticker {ti}: close + Flächen rally/target")
            ax.legend(loc="upper left")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.autofmt_xdate()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 9 — Wochentag × Monat (Mittel target)
        fig, ax = plt.subplots(figsize=(8, 5))
        df["_dow"] = df["Date"].dt.dayofweek
        df["_mon"] = df["Date"].dt.month
        pivot = df.pivot_table(index="_dow", columns="_mon", values="target", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.15, pos_rate * 2))
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"])
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_xlabel("Monat")
        ax.set_title("Mittelwert target (Heatmap DOW × Monat)")
        fig.colorbar(im, ax=ax, label="mean(target)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 10 — Konfig-Text
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        y = 0.98
        for ln in _summary_lines() + [
            "",
            mode_note,
            "",
            f"Zeilen (Plot): {len(df):,}",
            f"Ticker: {df['ticker'].nunique()}",
            "",
            f"PDF: {pdf_path}",
        ]:
            ax.text(0.04, y, ln, transform=ax.transAxes, fontsize=10, family="monospace", va="top")
            y -= 0.028
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Geschrieben: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
