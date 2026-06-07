"""CNN Fear & Greed Index (0–100) — Tagesmerge auf ``Date``."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _parse_cnn_fear_greed_payload(raw: dict) -> pd.Series:
    """Extrahiert Tages-Scores aus CNN graphdata JSON."""
    hist = None
    if isinstance(raw.get("fear_and_greed_historical"), dict):
        hist = raw["fear_and_greed_historical"].get("data")
    if not hist and isinstance(raw.get("fear_and_greed"), dict):
        fg = raw["fear_and_greed"]
        if "data" in fg:
            hist = fg["data"]
    if not hist:
        return pd.Series(dtype=float)
    rows: list[tuple[pd.Timestamp, float]] = []
    for pt in hist:
        if not isinstance(pt, dict):
            continue
        ts = pt.get("x")
        val = pt.get("y", pt.get("score"))
        if ts is None or val is None:
            continue
        try:
            if isinstance(ts, (int, float)):
                dt = pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(None).normalize()
            else:
                dt = pd.Timestamp(ts).normalize()
            score = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            rows.append((dt, score))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["Date", "score"]).drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date", kind="mergesort")
    idx = pd.to_datetime(df["Date"], errors="coerce")
    if hasattr(idx, "dt"):
        idx = idx.dt.normalize()
    else:
        idx = pd.Series(idx).dt.normalize()
    return pd.Series(df["score"].to_numpy(dtype=float), index=idx.to_numpy())


def fetch_fear_greed_series(
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
    *,
    cache_path: str | Path | None = None,
    api_url: str | None = None,
    cache_max_age_hours: float = 18.0,
) -> pd.Series | None:
    """
    Lädt CNN Fear & Greed (historisch + aktuell), optional JSON-Cache unter ``cache_path``.
    """
    start = pd.Timestamp(d_min).normalize() - pd.Timedelta(days=30)
    end = pd.Timestamp(d_max).normalize() + pd.Timedelta(days=5)
    url = (
        api_url
        or "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    ).strip()
    cache_p = Path(cache_path) if cache_path else None
    raw: dict | None = None

    if cache_p is not None and cache_p.is_file():
        try:
            meta = json.loads(cache_p.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("payload"), dict):
                fetched_at = float(meta.get("fetched_at_unix", 0))
                if (time.time() - fetched_at) / 3600.0 <= float(cache_max_age_hours):
                    raw = meta["payload"]
                    print("[MacroVola] mr_fear_greed: Cache-Treffer.", flush=True)
        except Exception:
            raw = None

    if raw is None:
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "stock_rally/1.0 (macro research; python urllib)",
                    "Accept": "application/json",
                },
            )
            with urlopen(req, timeout=25) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            if cache_p is not None and isinstance(raw, dict):
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                cache_p.write_text(
                    json.dumps({"fetched_at_unix": time.time(), "payload": raw}, default=str),
                    encoding="utf-8",
                )
            print("[MacroVola] mr_fear_greed: CNN API geladen.", flush=True)
        except Exception as exc:
            print(f"[MacroVola] mr_fear_greed: API fehlgeschlagen ({exc}).", flush=True)
            return None

    if not isinstance(raw, dict):
        return None
    ser = _parse_cnn_fear_greed_payload(raw)
    if ser is None or len(ser) == 0:
        print("[MacroVola] mr_fear_greed: keine parsebaren Punkte.", flush=True)
        return None
    ser = ser.sort_index()
    ser = ser[(ser.index >= start) & (ser.index <= end)]
    if len(ser) == 0:
        return None
    return ser.astype(float)


def attach_fear_greed_columns(
    out: pd.DataFrame,
    *,
    cfg_mod: Any | None = None,
) -> pd.DataFrame:
    """Fügt ``mr_fear_greed*`` per ``Date``-Merge hinzu."""
    if not bool(getattr(cfg_mod, "FEAR_GREED_ENABLED", True)):
        return out
    if "Date" not in out.columns:
        return out
    d_min = out["Date"].min()
    d_max = out["Date"].max()
    cache = getattr(cfg_mod, "FEAR_GREED_CACHE_FILE", None)
    url = getattr(cfg_mod, "FEAR_GREED_API_URL", None)
    ser = fetch_fear_greed_series(d_min, d_max, cache_path=cache, api_url=url)
    if ser is None or len(ser) == 0:
        for c in ("mr_fear_greed", "mr_fear_greed_ret5d", "mr_fear_greed_z_20d"):
            out[c] = np.nan
        return out
    vdf = pd.DataFrame(
        {
            "Date": pd.to_datetime(ser.index).normalize(),
            "mr_fear_greed": ser.to_numpy(dtype=float),
        }
    )
    out = out.merge(vdf, on="Date", how="left")
    ud = (
        out[["Date", "mr_fear_greed"]]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date", kind="mergesort")
    )
    fg = pd.to_numeric(ud["mr_fear_greed"], errors="coerce").ffill(limit=3)
    mu = fg.rolling(20, min_periods=10).mean()
    sd = fg.rolling(20, min_periods=10).std(ddof=0).replace(0, np.nan)
    ud = ud.assign(
        mr_fear_greed=fg.values,
        mr_fear_greed_ret5d=fg / fg.shift(5).replace(0, np.nan) - 1.0,
        mr_fear_greed_z_20d=(fg - mu) / sd,
    )
    out = out.drop(columns=["mr_fear_greed", "mr_fear_greed_ret5d", "mr_fear_greed_z_20d"], errors="ignore")
    out = out.merge(
        ud[["Date", "mr_fear_greed", "mr_fear_greed_ret5d", "mr_fear_greed_z_20d"]],
        on="Date",
        how="left",
    )
    return out
