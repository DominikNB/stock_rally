"""
Point-in-time Cross-Section-Features für Training/Scoring (vor Train/Test-Split).

Spiegelt die Holdout-Logik aus ``lib/signal_extra_filters.py`` (Makro-Kalender,
relative Stärke vs. SPY), ohne ``prob`` oder post-hoc-Ranks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAINING_CROSS_SECTION_COLS: tuple[str, ...] = (
    "macro_event_within_2bd",
    "ret_vs_spy_5d",
    "ret_vs_spy_20d",
)


def _macro_event_days_index() -> pd.DatetimeIndex:
    path = Path(__file__).resolve().parents[2] / "data" / "_scratch_us_macro_event_days.json"
    if path.is_file():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            days = [pd.Timestamp(d).normalize() for d in obj.get("dates", [])]
            if days:
                return pd.DatetimeIndex(days)
        except Exception:
            pass
    fomc = pd.to_datetime(
        [
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
            "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19",
            "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-12-10",
            "2026-01-28", "2026-03-18",
        ]
    ).normalize()
    return pd.DatetimeIndex(fomc.unique())


def _bdays_to_macro_event(signal_d: pd.Timestamp, event_days: pd.DatetimeIndex, window: int = 2) -> bool:
    sd = pd.Timestamp(signal_d).normalize()
    for ed in event_days:
        ed = pd.Timestamp(ed).normalize()
        if abs((ed - sd).days) <= window:
            return True
    return False


def _spy_close_series_from_df(df: pd.DataFrame) -> pd.Series | None:
    sub = df.loc[df["ticker"].astype(str) == "SPY", ["Date", "close"]].copy()
    if sub.empty:
        return None
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.normalize()
    sub = sub.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    s = pd.to_numeric(sub["close"], errors="coerce")
    s.index = sub["Date"].values
    return s.dropna()


def _download_spy_close(d_min: pd.Timestamp, d_max: pd.Timestamp) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    start_s = (d_min - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    end_s = (d_max + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("SPY", start=start_s, end=end_s, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"]["SPY"] if "SPY" in raw["Close"].columns else raw["Close"].iloc[:, 0]
        else:
            s = raw["Close"]
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.dropna()
    except Exception:
        return None


def _resolve_spy_close(df: pd.DataFrame) -> pd.Series | None:
    spy = _spy_close_series_from_df(df)
    if spy is not None and len(spy) >= 25:
        return spy
    d_norm = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if d_norm.notna().any():
        return _download_spy_close(d_norm.min(), d_norm.max())
    return None


def augment_df_cross_section_features(df: pd.DataFrame) -> pd.DataFrame:
    """Makro-Event-Flag (0/1) und relative 5d/20d-Rendite vs. SPY — nur point-in-time."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    ev_days = _macro_event_days_index()
    uniq_dates = sorted(out["Date"].dropna().unique())
    macro_map = {
        pd.Timestamp(d).normalize(): float(_bdays_to_macro_event(d, ev_days, 2))
        for d in uniq_dates
    }
    out["macro_event_within_2bd"] = out["Date"].map(macro_map).astype(float)

    out = out.sort_values(["ticker", "Date"])
    close = pd.to_numeric(out["close"], errors="coerce")
    out["_stk_ret5"] = out.groupby("ticker", sort=False)["close"].pct_change(5)
    out["_stk_ret20"] = out.groupby("ticker", sort=False)["close"].pct_change(20)

    spy = _resolve_spy_close(out)
    if spy is not None and len(spy) >= 21:
        spy = spy.sort_index()
        spy_ret5 = spy.pct_change(5)
        spy_ret20 = spy.pct_change(20)
        out["_spy_ret5"] = out["Date"].map(spy_ret5)
        out["_spy_ret20"] = out["Date"].map(spy_ret20)
        out["ret_vs_spy_5d"] = out["_stk_ret5"] - out["_spy_ret5"]
        out["ret_vs_spy_20d"] = out["_stk_ret20"] - out["_spy_ret20"]
    else:
        out["ret_vs_spy_5d"] = np.nan
        out["ret_vs_spy_20d"] = np.nan

    out.drop(
        columns=["_stk_ret5", "_stk_ret20", "_spy_ret5", "_spy_ret20"],
        inplace=True,
        errors="ignore",
    )
    return out


def append_training_cross_section_cols(
    feat_cols: list[str],
    df_train: pd.DataFrame,
) -> list[str]:
    """Hängt ``TRAINING_CROSS_SECTION_COLS`` an, wenn Spalte existiert und numerisch ist."""
    seen = set(feat_cols)
    out = list(feat_cols)
    added = 0
    for c in TRAINING_CROSS_SECTION_COLS:
        if c not in df_train.columns or c in seen:
            continue
        s = df_train[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        out.append(c)
        seen.add(c)
        added += 1
    if added:
        print(
            f"Phase 12: +{added} Cross-Section-Spalte(n) "
            f"(festes Training-Set, max. {len(TRAINING_CROSS_SECTION_COLS)}).",
            flush=True,
        )
    return out
