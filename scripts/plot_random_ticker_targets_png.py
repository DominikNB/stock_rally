#!/usr/bin/env python3
"""
Zufällige Ticker: Close-Kurs + ``target``/``rally`` nach cfg (feste Rally-Bänder oder
``Y_LABEL_RULE=cross_sectional_top_q``).

Cross-Sectional: Ausgabe unter ``figures/cross_sectional_target_plots/H{H}_q{pct}pct/`` (eigener Baum).
Labels dafür auf einer **Kohorte** von Ticker (df_train voll oder Download mit ``--cs-cohort-size``).

    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py
    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py --n 6 --dpi 360 --tail-days 500
    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py --rally-threshold 0.08 --seed 42
    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py --targets-only --n 3 --out-dir figures/target_only_demo --dpi 300
    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py --rally-plus-head-frac 0.1 --n 3 --out-dir figures/head10pct_demo
    .venv\\Scripts\\python scripts/plot_random_ticker_targets_png.py --no-clean --out-dir figures/custom
"""
from __future__ import annotations

import argparse
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _safe_stem(sym: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(sym), flags=re.UNICODE)
    return s[:120] if len(s) > 120 else s


@contextmanager
def _plot_fixed_y_overrides(
    *,
    rally_threshold: float | None = None,
    rally_plus_head_frac: float | None = None,
) -> Iterator[None]:
    """Temporär FIXED_Y_RALLY_THRESHOLD / RALLY_THRESHOLD und optional Kopf-Anteil rally_plus."""
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.target import override_fixed_y_config_for_grid

    kw: dict[str, Any] = {}
    if rally_threshold is not None:
        kw["RALLY_THRESHOLD"] = float(rally_threshold)
        if not cfg.opt_optimize_y_targets():
            kw["FIXED_Y_RALLY_THRESHOLD"] = float(rally_threshold)
    if rally_plus_head_frac is not None:
        kw["FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION"] = float(rally_plus_head_frac)
    if not kw:
        yield
        return
    with override_fixed_y_config_for_grid(**kw):
        yield


@contextmanager
def _rally_threshold_context(rt: float | None) -> Iterator[None]:
    """Kompatibel: delegiert an ``_plot_fixed_y_overrides`` (nur Schwelle)."""
    if rt is None:
        yield
        return
    with _plot_fixed_y_overrides(rally_threshold=float(rt)):
        yield


def _pick_tickers(rng: np.random.Generator, n: int, df_train: pd.DataFrame | None) -> list[str]:
    from lib.stock_rally_v10 import config as cfg

    if df_train is not None and "ticker" in df_train.columns and df_train["ticker"].nunique() > 0:
        u = df_train["ticker"].dropna().astype(str).unique().tolist()
        rng.shuffle(u)
        return u[: min(int(n), len(u))]
    u = [str(x) for x in cfg.ALL_TICKERS]
    if len(u) < n:
        return u
    rng.shuffle(u)
    return u[: int(n)]


def _prepare_minimal(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Date", "ticker", "close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Spalte fehlt: {c}")
    out = df[need + (["open"] if "open" in df.columns else [])].copy()
    if "open" not in out.columns:
        out["open"] = out["close"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "ticker", "close"])
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return out


def _label_panel(df: pd.DataFrame) -> pd.DataFrame:
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.target import rebuild_target_for_train

    df = _prepare_minimal(df)
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


def _cs_cohort_tickers(display: list[str], rng: np.random.Generator, cohort_size: int) -> list[str]:
    """Ticker-Liste für Cross-Sectional-Labels: Anzeige-Ticker + Zufalls-Kohorte aus dem Universum."""
    from lib.stock_rally_v10 import config as cfg

    out = list(dict.fromkeys(str(t) for t in display))
    pool = [t for t in cfg.ALL_TICKERS if t not in out]
    rng.shuffle(pool)
    for t in pool:
        if len(out) >= int(cohort_size):
            break
        out.append(t)
    return out


def _download_panel(
    tickers: list[str],
    start,
    end,
    *,
    rng: np.random.Generator | None = None,
    cohort_size: int = 120,
) -> pd.DataFrame:
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.data import load_stock_data
    from lib.stock_rally_v10.target import create_target

    load_list = list(tickers)
    if not cfg.opt_optimize_y_targets() and cfg.y_label_rule() == "cross_sectional_top_q":
        if rng is None:
            rng = np.random.default_rng(0)
        load_list = _cs_cohort_tickers(tickers, rng, cohort_size)

    raw = load_stock_data(tickers=load_list, start=start, end=end)
    raw = _prepare_minimal(raw)
    return create_target(raw, quiet=True)


def _effective_rally_threshold(cfg_mod: Any, rt_arg: float | None) -> float:
    if rt_arg is not None:
        return float(rt_arg)
    if cfg_mod.opt_optimize_y_targets():
        return float(getattr(cfg_mod, "RALLY_THRESHOLD", 0.07))
    return float(getattr(cfg_mod, "FIXED_Y_RALLY_THRESHOLD", getattr(cfg_mod, "RALLY_THRESHOLD", 0.07)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=10, help="Anzahl Ticker")
    ap.add_argument("--tail-days", type=int, default=900, help="Pro Ticker: letzte N Handelstage plotten")
    ap.add_argument(
        "--rally-threshold",
        type=float,
        default=None,
        help="Mindest-Trade-Rendite für Rally (z.B. 0.08 = 8 %%). Überschreibt cfg temporär.",
    )
    ap.add_argument(
        "--rally-plus-head-frac",
        type=float,
        default=None,
        help="Temporär FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION (z. B. 0.1 = erste 10%% je Fensterkopf).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Ausgabeordner (Default: figures/ bzw. figures/target_rally_<pct>pct bei --rally-threshold).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=720,
        help="PNG-Raster (Standard hoch für Zoom); z.B. 600–1200",
    )
    ap.add_argument("--width", type=float, default=28.0, help="Figure-Breite in Zoll")
    ap.add_argument("--height", type=float, default=10.0, help="Figure-Höhe in Zoll")
    ap.add_argument(
        "--no-clean",
        action="store_true",
        help="Keine PNGs im Zielordner vorher löschen (Standard: *.png im Zielordner löschen).",
    )
    ap.add_argument(
        "--targets-only",
        action="store_true",
        help="Nur Kurs + target=1 markieren (kein rally=1, kein grünes Band).",
    )
    ap.add_argument(
        "--cs-cohort-size",
        type=int,
        default=120,
        help="Nur cross_sectional_top_q: so viele Ticker parallel laden für korrektes Ranking pro Tag.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    from lib.stock_rally_v10 import config as cfg

    if args.out_dir:
        out_dir = (_ROOT / args.out_dir).resolve()
    elif args.rally_threshold is not None:
        pct = int(round(float(args.rally_threshold) * 100))
        out_dir = (_ROOT / "figures" / f"target_rally_{pct}pct").resolve()
    elif not cfg.opt_optimize_y_targets() and cfg.y_label_rule() == "cross_sectional_top_q":
        _hq = int(getattr(cfg, "CS_TARGET_FORWARD_HORIZON", 20))
        _qq = float(getattr(cfg, "CS_TARGET_TOP_Q", 0.1))
        out_dir = (
            _ROOT
            / "figures"
            / "cross_sectional_target_plots"
            / f"H{_hq}_q{int(round(_qq * 100))}pct"
        ).resolve()
    elif (
        not cfg.opt_optimize_y_targets()
        and str(getattr(cfg, "FIXED_Y_LABEL_MODE", "")).strip().lower() == "rally_plus_entry"
    ):
        out_dir = (_ROOT / "figures" / "rally_plus_signal_targets").resolve()
    else:
        out_dir = (_ROOT / "figures").resolve()

    if bool(getattr(args, "targets_only", False)):
        if args.out_dir:
            out_dir = (_ROOT / args.out_dir).resolve()
        else:
            out_dir = (_ROOT / "figures" / "targets_only_plots").resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_clean:
        removed = 0
        for p in sorted(out_dir.glob("*.png")):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            print(f"Entfernt: {removed} PNG(s) in {out_dir}", flush=True)

    ctx = _plot_fixed_y_overrides(
        rally_threshold=args.rally_threshold,
        rally_plus_head_frac=args.rally_plus_head_frac,
    )
    label_rt = _effective_rally_threshold(cfg, args.rally_threshold)

    with ctx:
        df_train = getattr(cfg, "df_train", None)
        tickers = _pick_tickers(rng, int(args.n), df_train)

        if df_train is not None and len(df_train) > 0:
            if not cfg.opt_optimize_y_targets() and cfg.y_label_rule() == "cross_sectional_top_q":
                need = [c for c in ("Date", "ticker", "close") if c in df_train.columns]
                if len(need) < 3:
                    raise ValueError("df_train braucht Spalten Date, ticker, close für Cross-Sectional.")
                full = df_train[[c for c in need + (["open"] if "open" in df_train.columns else [])]].copy()
                labeled = _label_panel(full)
                panel = labeled[labeled["ticker"].isin(tickers)].copy()
            else:
                sub = df_train[df_train["ticker"].isin(tickers)].copy()
                if len(sub) == 0:
                    print("df_train enthält die gewählten Ticker nicht — Fallback Download.", flush=True)
                    panel = _download_panel(
                        tickers, cfg.START_DATE, cfg.END_DATE, rng=rng, cohort_size=int(args.cs_cohort_size)
                    )
                else:
                    panel = _label_panel(sub)
        else:
            print("Kein cfg.df_train — lade Ticker per yfinance …", flush=True)
            panel = _download_panel(
                tickers, cfg.START_DATE, cfg.END_DATE, rng=rng, cohort_size=int(args.cs_cohort_size)
            )

    eff_rally_plus_head_frac = (
        float(args.rally_plus_head_frac)
        if args.rally_plus_head_frac is not None
        else float(getattr(cfg, "FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION", 0.1))
    )

    if "rally" not in panel.columns:
        panel["rally"] = np.int8(0)

    tail_n = max(120, int(args.tail_days))
    written = 0
    for sym in tickers:
        sub = panel[panel["ticker"] == sym].sort_values("Date")
        if len(sub) < 20:
            print(f"Überspringe {sym}: zu wenige Zeilen ({len(sub)}).", flush=True)
            continue
        sub = sub.tail(tail_n)
        dr = sub["Date"].to_numpy()
        cl = sub["close"].to_numpy(dtype=float)
        y0, y1 = float(np.nanmin(cl)), float(np.nanmax(cl))
        pad = (y1 - y0) * 0.03 + 1e-6
        y0 -= pad
        y1 += pad
        r = sub["rally"].to_numpy(dtype=np.int8)
        t = sub["target"].to_numpy(dtype=np.int8)

        w_in = float(args.width)
        h_in = float(args.height)
        dpi = int(args.dpi)
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        lw = max(1.0, 1.4 * (dpi / 400.0))
        ax.plot(dr, cl, color="#1f1f1f", lw=lw, label="close")
        if not args.targets_only:
            if (r == 1).any():
                ax.fill_between(dr, y0, y1, where=(r == 1), color="#81c784", alpha=0.35, label="rally=1")
        if (t == 1).any():
            _ta = 0.55 if args.targets_only else 0.45
            ax.fill_between(dr, y0, y1, where=(t == 1), color="#ffb74d", alpha=_ta, label="target=1")
        pr = float(np.mean(t)) if len(t) else 0.0
        _suffix = "  [nur target]" if args.targets_only else ""
        if cfg.y_label_rule() == "cross_sectional_top_q":
            _H = int(getattr(cfg, "CS_TARGET_FORWARD_HORIZON", 20))
            _q = float(getattr(cfg, "CS_TARGET_TOP_Q", 0.1))
            _gb = str(getattr(cfg, "CS_TARGET_GROUPBY", "calendar_day"))
            title = (
                f"{sym}  —  cross-sectional Top-{_q:.0%}  H={_H}d  ({_gb})  —  "
                f"target pos. (dieser Ticker) = {pr:.2%}{_suffix}"
            )
        elif cfg.fixed_y_label_mode() == "rally_plus_entry":
            _nsp = int(getattr(cfg, "FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS", 2))
            _hf = float(eff_rally_plus_head_frac)
            title = (
                f"{sym}  —  rally_plus_entry: pro Fenster Kopf ceil(Lw·{_hf:g}) + {_nsp}d vor entry  —  "
                f"Rally-Schwelle ≥ {label_rt:.1%}  —  target pos. = {pr:.2%}{_suffix}"
            )
        else:
            title = (
                f"{sym}  —  Rally-Schwelle ≥ {label_rt:.1%}  —  target pos. (Fenster) = {pr:.2%}{_suffix}"
            )
        ax.set_title(title, fontsize=max(14, int(10 + dpi / 80)))
        ax.legend(loc="upper left", fontsize=max(10, int(8 + dpi / 100)))
        ax.tick_params(axis="both", labelsize=max(10, int(8 + dpi / 100)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("close", fontsize=max(11, int(9 + dpi / 100)))
        fig.tight_layout()
        _stem = "target_only" if args.targets_only else "target_random"
        out = out_dir / f"{_stem}_{written:02d}_{_safe_stem(sym)}.png"
        fig.savefig(out, format="png", bbox_inches="tight", dpi=dpi, pil_kwargs={"compress_level": 3})
        plt.close(fig)
        print(out.resolve(), flush=True)
        written += 1

    if written == 0:
        print("Keine PNGs geschrieben.", file=sys.stderr)
        sys.exit(1)
    px_w = int(round(float(args.width) * dpi))
    px_h = int(round(float(args.height) * dpi))
    if args.targets_only:
        tail_msg = f"nur target=1 (ohne rally); Raster grob bis {px_w}×{px_h} px, dpi={dpi}."
    elif cfg.y_label_rule() == "cross_sectional_top_q":
        tail_msg = (
            f"Cross-sectional H={int(getattr(cfg, 'CS_TARGET_FORWARD_HORIZON', 20))} "
            f"q={float(getattr(cfg, 'CS_TARGET_TOP_Q', 0.1)):.2%}; Raster grob bis {px_w}×{px_h} px, dpi={dpi}."
        )
    elif cfg.fixed_y_label_mode() == "rally_plus_entry":
        _hf = float(eff_rally_plus_head_frac)
        tail_msg = (
            f"rally_plus_entry (je Fenster Kopf ceil(Lw·{_hf:g}) + Vorlauf); Raster grob bis {px_w}×{px_h} px, dpi={dpi}."
        )
    else:
        tail_msg = (
            f"Rally-Schwelle Plot={label_rt:.2%}; Raster grob bis {px_w}×{px_h} px, dpi={dpi}."
        )
    print(f"Fertig: {written} Datei(en) in {out_dir} ({tail_msg})", flush=True)


if __name__ == "__main__":
    main()
