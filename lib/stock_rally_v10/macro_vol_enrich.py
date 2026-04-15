"""
``mr_*`` nur für Base-Classifier-Training: Yahoo-Vola-Indizes + Ratios, die sich aus
``regime_*`` (nach ``add_short_horizon_macro_regime_columns``) und ``momentum_20d`` (Indikatoren) ergeben.

Keine Abhängigkeit von ``prob``, Alpha, Betas, RS-Spalten, ``volatility_20d_ann``, etc.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

# Reihenfolge für stabile FEAT_COLS (Regime kommt vom Macro-Merge, ``mr_*`` von unten).
TRAINING_MACRO_VOL_COLS: tuple[str, ...] = (
    "regime_vix_level",
    "regime_vix_z_20d",
    "regime_spy_realvol_5d_ann",
    "regime_tnx_ret_5d",
    "mr_vvix_level",
    "mr_vxn_level",
    "mr_rvx_level",
    "mr_vxv_level",
    "mr_vix_level_ret5d",
    "mr_vix_level_ret1d",
    "mr_vxn_div_vix",
    "mr_rvx_div_vix",
    "mr_vix3m_div_vix",
    "mr_vxv_div_vix",
    "mr_spyrv_points_div_vix",
    "mr_vvix_div_vix",
    "mr_vvix_level_ret5d",
    "mr_vvix_level_ret1d",
    "mr_vvix_vix_ret1d_spread",
    "mr_vvix_vix_ret5d_spread",
    "mr_momentum20_div_spyrv",
)


def _mr_safe_div(
    num: pd.Series | np.ndarray,
    den: pd.Series | np.ndarray,
    *,
    floor: float = 8.0,
    cap: float = 100.0,
) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce").clip(lower=floor, upper=cap)
    d = d.replace(0, np.nan)
    return n / d


def _yahoo_index_close_series(
    symbol: str,
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    start = (pd.Timestamp(d_min).normalize() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(d_max).normalize() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
    return c.sort_index().astype(float)


def _fred_index_close_series(
    series_id: str,
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
) -> pd.Series | None:
    """Lädt Tageswerte aus FRED (primär via fredapi+Key, sonst CSV-Fallback)."""
    start_ts = pd.Timestamp(d_min).normalize() - pd.Timedelta(days=45)
    end_ts = pd.Timestamp(d_max).normalize() + pd.Timedelta(days=5)
    start = start_ts.strftime("%Y-%m-%d")
    end = end_ts.strftime("%Y-%m-%d")
    ser = _fred_index_close_series_via_api(series_id, start_ts, end_ts)
    if ser is not None and len(ser) > 0:
        return ser
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    try:
        df = pd.read_csv(url)
    except Exception:
        return None
    if df is None or len(df) == 0 or "DATE" not in df.columns:
        return None
    val_col = "VALUE" if "VALUE" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    if val_col is None:
        return None
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["DATE"], errors="coerce").dt.normalize(),
            "val": pd.to_numeric(df[val_col], errors="coerce"),
        }
    ).dropna(subset=["Date", "val"])
    if len(out) == 0:
        return None
    s = pd.Series(out["val"].to_numpy(dtype=float), index=out["Date"].to_numpy())
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


@lru_cache(maxsize=1)
def _get_fred_client():
    # Projekt-.env laden (falls verfügbar), damit FRED_API_KEY ohne Shell-Export greift.
    try:
        from config.load_env import load_project_env

        load_project_env()
    except Exception:
        pass
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        return None
    try:
        from fredapi import Fred

        return Fred(api_key=key)
    except Exception:
        return None


def _fred_index_close_series_via_api(
    series_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.Series | None:
    client = _get_fred_client()
    if client is None:
        return None
    try:
        ser = client.get_series(
            str(series_id),
            observation_start=start_ts.strftime("%Y-%m-%d"),
            observation_end=end_ts.strftime("%Y-%m-%d"),
        )
    except Exception:
        return None
    if ser is None or len(ser) == 0:
        return None
    ser = pd.to_numeric(ser, errors="coerce")
    idx = pd.to_datetime(ser.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    ser.index = idx.normalize()
    ser = ser.dropna()
    if len(ser) == 0:
        return None
    return ser.sort_index().astype(float)


def _yahoo_index_close_series_first_available(
    symbols: list[str] | tuple[str, ...],
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
    *,
    log_label: str | None = None,
) -> pd.Series | None:
    """
    Versucht mehrere Yahoo-Symbole der Reihe nach und nimmt das erste mit Daten.
    Nützlich bei umbenannten/alternativen Tickersymbolen (z. B. VIX3M vs VXV).
    """
    for s in symbols:
        ser = _yahoo_index_close_series(str(s), d_min, d_max)
        if ser is not None and len(ser) > 0:
            if log_label:
                print(f"[MacroVola] {log_label}: nutze Yahoo-Symbol {s}", flush=True)
            return ser
    if log_label:
        print(
            f"[MacroVola] {log_label}: keine Daten für {list(symbols)} — Spalte bleibt NaN.",
            flush=True,
        )
    return None


def _vol_index_series_with_fallback(
    *,
    label: str,
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
    fred_ids: list[str] | tuple[str, ...] = (),
    yahoo_symbols: list[str] | tuple[str, ...] = (),
) -> pd.Series | None:
    """
    Primär FRED (stabil), dann Yahoo-Fallback.
    """
    for sid in fred_ids:
        ser = _fred_index_close_series(str(sid), d_min, d_max)
        if ser is not None and len(ser) > 0:
            print(f"[MacroVola] {label}: nutze FRED-Serie {sid}", flush=True)
            return ser
    if yahoo_symbols:
        ser = _yahoo_index_close_series_first_available(yahoo_symbols, d_min, d_max, log_label=label)
        if ser is not None and len(ser) > 0:
            print(f"[MacroVola] {label}: FRED fehlte, Yahoo-Fallback aktiv.", flush=True)
            return ser
    print(
        f"[MacroVola] {label}: keine Daten in FRED {list(fred_ids)} "
        f"und Yahoo {list(yahoo_symbols)} — Spalte bleibt NaN.",
        flush=True,
    )
    return None


def _merge_yahoo_level(
    out: pd.DataFrame,
    series: pd.Series | None,
    col_name: str,
) -> pd.DataFrame:
    if series is not None and len(series) > 0:
        vdf = pd.DataFrame(
            {
                "Date": pd.to_datetime(series.index).normalize(),
                col_name: series.to_numpy(dtype=float),
            }
        )
        return out.merge(vdf, on="Date", how="left")
    out[col_name] = np.nan
    return out


def enrich_macro_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nur Spalten aus ``TRAINING_MACRO_VOL_COLS`` (ohne ``regime_*`` — die kommen vom Regime-Merge)."""
    out = df.copy()
    if "Date" not in out.columns:
        return out
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    d_min = out["Date"].min()
    d_max = out["Date"].max()

    out = _merge_yahoo_level(
        out,
        _vol_index_series_with_fallback(
            label="mr_vvix_level",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("VVIXCLS",),
            yahoo_symbols=("^VVIX",),
        ),
        "mr_vvix_level",
    )
    out = _merge_yahoo_level(
        out,
        _vol_index_series_with_fallback(
            label="mr_vxn_level",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("VXNCLS",),
            yahoo_symbols=("^VXN",),
        ),
        "mr_vxn_level",
    )
    out = _merge_yahoo_level(
        out,
        _vol_index_series_with_fallback(
            label="mr_rvx_level",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("RVXCLS",),
            yahoo_symbols=("^RVX",),
        ),
        "mr_rvx_level",
    )
    out = _merge_yahoo_level(
        out,
        _vol_index_series_with_fallback(
            label="mr_vxv_level",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("VXVCLS",),
            yahoo_symbols=("^VIX3M", "^VXV"),
        ),
        "mr_vxv_level",
    )

    if "regime_vix_level" in out.columns:
        vx = out["regime_vix_level"]
        ud = (
            out[["Date", "regime_vix_level"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date", kind="mergesort")
        )
        vxl = pd.to_numeric(ud["regime_vix_level"], errors="coerce")
        ud = ud.assign(
            mr_vix_level_ret5d=vxl / vxl.shift(5).replace(0, np.nan) - 1.0,
            mr_vix_level_ret1d=vxl / vxl.shift(1).replace(0, np.nan) - 1.0,
        )
        out = out.merge(
            ud[["Date", "mr_vix_level_ret5d", "mr_vix_level_ret1d"]],
            on="Date",
            how="left",
        )

        if "mr_vxn_level" in out.columns:
            out["mr_vxn_div_vix"] = _mr_safe_div(out["mr_vxn_level"], vx)
        if "mr_rvx_level" in out.columns:
            out["mr_rvx_div_vix"] = _mr_safe_div(out["mr_rvx_level"], vx)
        if "mr_vxv_level" in out.columns:
            out["mr_vxv_div_vix"] = _mr_safe_div(out["mr_vxv_level"], vx)
            out["mr_vix3m_div_vix"] = out["mr_vxv_div_vix"]

        if "regime_spy_realvol_5d_ann" in out.columns:
            rv_pts = pd.to_numeric(out["regime_spy_realvol_5d_ann"], errors="coerce") * 100.0
            out["mr_spyrv_points_div_vix"] = _mr_safe_div(rv_pts, vx)

    if "mr_vvix_level" in out.columns and "regime_vix_level" in out.columns:
        out["mr_vvix_div_vix"] = _mr_safe_div(out["mr_vvix_level"], out["regime_vix_level"])

    if "mr_vvix_level" in out.columns:
        ud_vv = (
            out[["Date", "mr_vvix_level"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date", kind="mergesort")
        )
        vvl = pd.to_numeric(ud_vv["mr_vvix_level"], errors="coerce")
        ud_vv = ud_vv.assign(
            mr_vvix_level_ret5d=vvl / vvl.shift(5).replace(0, np.nan) - 1.0,
            mr_vvix_level_ret1d=vvl / vvl.shift(1).replace(0, np.nan) - 1.0,
        )
        out = out.merge(
            ud_vv[["Date", "mr_vvix_level_ret5d", "mr_vvix_level_ret1d"]],
            on="Date",
            how="left",
        )

    if "mr_vvix_level_ret1d" in out.columns and "mr_vix_level_ret1d" in out.columns:
        out["mr_vvix_vix_ret1d_spread"] = pd.to_numeric(
            out["mr_vvix_level_ret1d"], errors="coerce"
        ) - pd.to_numeric(out["mr_vix_level_ret1d"], errors="coerce")
    if "mr_vvix_level_ret5d" in out.columns and "mr_vix_level_ret5d" in out.columns:
        out["mr_vvix_vix_ret5d_spread"] = pd.to_numeric(
            out["mr_vvix_level_ret5d"], errors="coerce"
        ) - pd.to_numeric(out["mr_vix_level_ret5d"], errors="coerce")

    if "momentum_20d" in out.columns and "regime_spy_realvol_5d_ann" in out.columns:
        rv = pd.to_numeric(out["regime_spy_realvol_5d_ann"], errors="coerce").clip(lower=1e-6)
        out["mr_momentum20_div_spyrv"] = pd.to_numeric(out["momentum_20d"], errors="coerce") / rv

    return out
