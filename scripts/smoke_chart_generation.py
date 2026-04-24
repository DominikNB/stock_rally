"""
Quick smoke test for website chart generation.

Runs the core plotting path (OHLC load + Candles + EMA/BB + RSI + Volume/Z)
for one signal, without running the full pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.patches import Rectangle


def _pick_signal(root: Path, ticker: str | None, date_str: str | None) -> tuple[str, str]:
    if ticker and date_str:
        return ticker.strip(), date_str.strip()[:10]
    sig_path = root / "docs" / "signals.json"
    if not sig_path.is_file():
        raise FileNotFoundError(f"Missing {sig_path}")
    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    sigs = payload.get("signals") or []
    if not sigs:
        raise RuntimeError("docs/signals.json contains no signals")
    s0 = sigs[0]
    return str(s0["ticker"]).strip(), str(s0["date"])[:10]


def _load_ohlcv(ticker: str, d0: str, d1: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(
        start=d0,
        end=d1,
        interval="1d",
        auto_adjust=True,
        actions=False,
    )
    if hist is None or len(hist) == 0:
        ext = yf.download(ticker, start=d0, end=d1, progress=False, threads=False)
        if ext is None or len(ext) == 0:
            raise RuntimeError(f"No OHLCV data for {ticker} in {d0}..{d1}")
        if isinstance(ext.columns, pd.MultiIndex):
            ext.columns = [str(c[0]) for c in ext.columns]
        hist = ext
    need = ["Open", "High", "Low", "Close", "Volume"]
    for col in need:
        if col not in hist.columns:
            raise RuntimeError(f"Missing column {col} for {ticker}")
    hist = hist.reset_index(drop=False)
    if "Date" not in hist.columns:
        # yfinance fallback naming
        if "index" in hist.columns:
            hist = hist.rename(columns={"index": "Date"})
        elif "Datetime" in hist.columns:
            hist = hist.rename(columns={"Datetime": "Date"})
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(hist["Date"]).dt.normalize(),
            "open": pd.to_numeric(hist["Open"], errors="coerce"),
            "high": pd.to_numeric(hist["High"], errors="coerce"),
            "low": pd.to_numeric(hist["Low"], errors="coerce"),
            "close": pd.to_numeric(hist["Close"], errors="coerce"),
            "volume": pd.to_numeric(hist["Volume"], errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    if len(out) < 10:
        raise RuntimeError(f"Too few OHLC rows for {ticker}: {len(out)}")
    return out.sort_values("Date").reset_index(drop=True)


def _plot_smoke(
    df: pd.DataFrame,
    ticker: str,
    sig_date: str,
    out_png: Path,
    *,
    view_lo: pd.Timestamp | None = None,
    view_hi: pd.Timestamp | None = None,
) -> None:
    def _norm_ts(x) -> pd.Timestamp:
        t = pd.Timestamp(x)
        try:
            if t.tz is not None:
                t = t.tz_convert(None)
        except Exception:
            try:
                t = t.tz_localize(None)
            except Exception:
                pass
        return t.normalize()

    dt = pd.to_datetime(df["Date"])
    x = mdates.date2num(pd.DatetimeIndex(dt).to_pydatetime())
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ema20 = close.ewm(span=20, adjust=False, min_periods=1).mean()
    bb_mid = close.rolling(20, min_periods=5).mean()
    bb_std = close.rolling(20, min_periods=5).std()
    bb_up = bb_mid + 2.0 * bb_std
    bb_lo = bb_mid - 2.0 * bb_std
    bb_width = (bb_up - bb_lo) / (bb_mid.abs() + 1e-10)

    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = -d.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    v_mean = volume.rolling(20, min_periods=5).mean()
    v_std = volume.rolling(20, min_periods=5).std()
    vol_z = ((volume - v_mean) / (v_std + 1e-10)).fillna(0.0)

    fig, (ax_price, ax_bw, ax_rsi, ax_vol) = plt.subplots(
        4,
        1,
        figsize=(9.4, 7.3),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0, 1.4, 1.6]},
    )

    candle_w = 0.6
    up_color, dn_color = "#66bb6a", "#ef5350"
    for i in range(len(x)):
        o, h, l, c = float(open_.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        col = up_color if c >= o else dn_color
        ax_price.vlines(x[i], l, h, color=col, linewidth=0.9, alpha=0.95, zorder=2)
        body_y = min(o, c)
        body_h = max(abs(c - o), 1e-6)
        ax_price.add_patch(
            Rectangle(
                (x[i] - candle_w / 2.0, body_y),
                candle_w,
                body_h,
                facecolor=col,
                edgecolor=col,
                linewidth=0.7,
                alpha=0.95,
                zorder=3,
            )
        )

    # Klassische Kurslinie zusätzlich zur Candle-Ansicht.
    ax_price.plot(dt, close, color="#00e676", lw=1.1, alpha=0.9, label="Close")
    ax_price.plot(dt, ema20, color="#42a5f5", lw=1.2, label="EMA 20")
    ax_price.plot(dt, bb_up, color="#ab47bc", lw=1.0, ls="--", label="BB Upper (20,2)")
    ax_price.plot(dt, bb_mid, color="#8d6e63", lw=0.9, ls="-.", label="BB Mid (20)")
    ax_price.plot(dt, bb_lo, color="#ab47bc", lw=1.0, ls="--", label="BB Lower (20,2)")
    ax_bw.plot(dt, bb_width, color="#ba68c8", lw=1.2, label="BB Width (20,2)")
    ax_rsi.plot(dt, rsi14, color="#ffb74d", lw=1.2, label="RSI 14")
    for lvl, col, ls in ((70, "#ef5350", "--"), (30, "#66bb6a", "--"), (75, "#ff7043", ":"), (25, "#26a69a", ":")):
        ax_rsi.axhline(lvl, color=col, lw=0.9, ls=ls, alpha=0.95)
    ax_rsi.set_ylim(0, 100)

    bar_colors = np.where(close.to_numpy() >= open_.to_numpy(), up_color, dn_color)
    ax_vol.bar(dt, volume.to_numpy(), width=0.8, color=bar_colors, alpha=0.72, label="Volume")
    ax_vz = ax_vol.twinx()
    ax_vz.plot(dt, vol_z.to_numpy(), color="#80cbc4", lw=1.2, label="Volume Z-Score")
    ax_vz.axhline(0.0, color="#607d8b", lw=0.9, ls="--", alpha=0.8)

    sig_ts = pd.Timestamp(sig_date).normalize()
    for a in (ax_price, ax_bw, ax_rsi, ax_vol):
        a.axvline(sig_ts, color="#66bb6a", lw=1.2, ls="--")

    if view_lo is not None and view_hi is not None:
        _view_lo = _norm_ts(view_lo)
        _view_hi_nominal = _norm_ts(view_hi)
        _last_dt = _norm_ts(dt.max())
        _sig_ts = _norm_ts(sig_date)
        _view_hi = min(_view_hi_nominal, _last_dt)
        _view_hi = max(_view_hi, _sig_ts + pd.Timedelta(days=2))
        ax_price.set_xlim(_view_lo, _view_hi)

    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7)
    ax_price.set_title(f"Smoke chart: {ticker} @ {sig_date}", fontsize=9)
    ax_price.legend(fontsize=7, loc="upper left")
    ax_bw.legend(fontsize=7, loc="upper left")
    ax_rsi.legend(fontsize=7, loc="upper left")
    h1, l1 = ax_vol.get_legend_handles_labels()
    h2, l2 = ax_vz.get_legend_handles_labels()
    ax_vol.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test website chart rendering")
    p.add_argument("--ticker", type=str, default=None, help="Ticker symbol")
    p.add_argument("--date", type=str, default=None, help="Signal date YYYY-MM-DD")
    p.add_argument("--out", type=str, default="docs/smoke_chart.png", help="Output PNG path")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    ticker, sig_date = _pick_signal(root, args.ticker, args.date)
    sig_ts = pd.Timestamp(sig_date).normalize()
    win_lo = sig_ts - pd.DateOffset(months=1)
    win_hi = sig_ts + pd.DateOffset(months=1)
    # Warmup-Historie für Indikatoren (RSI/BB/EMA), damit am linken Plotrand weniger NaN entsteht.
    warmup_days = 90
    d0 = (win_lo - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    d1 = (win_hi + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = _load_ohlcv(ticker, d0, d1)
    out_png = root / args.out
    _plot_smoke(df, ticker, sig_date, out_png, view_lo=win_lo, view_hi=win_hi)
    print(f"SMOKE OK: {ticker} @ {sig_date} -> {out_png}")


if __name__ == "__main__":
    main()
