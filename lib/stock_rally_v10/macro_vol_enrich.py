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
    # Cross-Asset Regime — Risk-on/-off Indikatoren jenseits der Vola-Indizes.
    # HY-Spread = ICE BofA US High Yield Index OAS (FRED: BAMLH0A0HYM2); spreitet
    # bei Stress aus, kontrahiert in „risk-on"-Phasen.
    "mr_hy_spread",
    "mr_hy_spread_ret5d",
    "mr_hy_spread_ret1d",
    # DXY = US-Dollar-Index (Yahoo: ``DX-Y.NYB``/``DX=F``). Starker Dollar belastet
    # tendenziell Rohstoffe/EM/Tech-Margins; rollendes Momentum als Regime-Signal.
    "mr_dxy_level",
    "mr_dxy_mom_20d",
    "mr_dxy_mom_60d",
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


def _vix_utils_vvix_series(
    d_min: pd.Timestamp,
    d_max: pd.Timestamp,
) -> pd.Series | None:
    """
    VVIX primär direkt über vix_utils/CBOE laden.
    Unterstützt sowohl Long-Format (Spalte Symbol) als auch Wide-Format (Spalte VVIX).
    """
    try:
        import vix_utils
    except ImportError:
        return None
    try:
        raw = vix_utils.get_vix_index_histories()
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None

    df = pd.DataFrame(raw).copy()
    vvix = None

    # Long-Format: Trade Date / Symbol / Close
    if {"Symbol", "Close"}.issubset(df.columns):
        sym = df["Symbol"].astype(str).str.upper()
        vv = df.loc[sym == "VVIX"].copy()
        if len(vv) > 0:
            date_col = "Trade Date" if "Trade Date" in vv.columns else ("Date" if "Date" in vv.columns else None)
            if date_col is not None:
                vv[date_col] = pd.to_datetime(vv[date_col], errors="coerce").dt.normalize()
                vv["Close"] = pd.to_numeric(vv["Close"], errors="coerce")
                vv = vv.dropna(subset=[date_col, "Close"]).sort_values(date_col, kind="mergesort")
                if len(vv) > 0:
                    vvix = pd.Series(vv["Close"].to_numpy(dtype=float), index=vv[date_col].to_numpy())

    # Wide-Format: Date-Index und Spalte VVIX.
    if vvix is None and "VVIX" in df.columns:
        date_col = "Trade Date" if "Trade Date" in df.columns else ("Date" if "Date" in df.columns else None)
        vv = df.copy()
        if date_col is not None:
            vv[date_col] = pd.to_datetime(vv[date_col], errors="coerce").dt.normalize()
            idx = vv[date_col]
        else:
            idx = pd.to_datetime(vv.index, errors="coerce")
            try:
                idx = idx.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            idx = pd.Series(idx).dt.normalize()
        vals = pd.to_numeric(vv["VVIX"], errors="coerce")
        out = pd.DataFrame({"Date": idx, "VVIX": vals}).dropna(subset=["Date", "VVIX"]).sort_values(
            "Date", kind="mergesort"
        )
        if len(out) > 0:
            vvix = pd.Series(out["VVIX"].to_numpy(dtype=float), index=out["Date"].to_numpy())

    if vvix is None or len(vvix) == 0:
        return None

    vvix.index = pd.to_datetime(vvix.index, errors="coerce")
    try:
        vvix.index = vvix.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    vvix.index = vvix.index.normalize()
    vvix = vvix.sort_index().astype(float)

    # Etwas Puffer vor/nach dem Bedarf wie bei Yahoo/FRED.
    start = pd.Timestamp(d_min).normalize() - pd.Timedelta(days=45)
    end = pd.Timestamp(d_max).normalize() + pd.Timedelta(days=5)
    vvix = vvix[(vvix.index >= start) & (vvix.index <= end)]
    if len(vvix) == 0:
        return None
    return vvix


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


def _impute_vvix_level_series(
    out: pd.DataFrame,
    *,
    col_name: str = "mr_vvix_level",
    ffill_limit: int = 5,
    bfill_limit: int = 1,
) -> pd.DataFrame:
    """Imputiert kurze VVIX-Lücken, damit abgeleitete Returns/Spreads nicht kaskadierend ausfallen."""
    if col_name not in out.columns:
        return out
    s_raw = pd.to_numeric(out[col_name], errors="coerce")
    miss_raw = s_raw.isna()
    s_imp = s_raw.ffill(limit=max(0, int(ffill_limit)))
    if int(bfill_limit) > 0:
        s_imp = s_imp.bfill(limit=int(bfill_limit))
    out[col_name] = s_imp
    out["mr_vvix_missing_flag"] = miss_raw.astype(np.int8)
    miss_after = out[col_name].isna()
    n = max(1, len(out))
    print(
        "[MacroVola] mr_vvix_level Imputation: "
        f"raw_missing={int(miss_raw.sum())}/{n} ({100.0 * float(miss_raw.sum()) / float(n):.1f}%), "
        f"after_missing={int(miss_after.sum())}/{n} ({100.0 * float(miss_after.sum()) / float(n):.1f}%), "
        f"ffill_limit={int(ffill_limit)}, bfill_limit={int(bfill_limit)}",
        flush=True,
    )
    return out


def enrich_macro_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nur Spalten aus ``TRAINING_MACRO_VOL_COLS`` (ohne ``regime_*`` — die kommen vom Regime-Merge)."""
    out = df.copy()
    if "Date" not in out.columns:
        return out
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    d_min = out["Date"].min()
    d_max = out["Date"].max()

    vvix_series = _vix_utils_vvix_series(d_min=d_min, d_max=d_max)
    if vvix_series is not None and len(vvix_series) > 0:
        print("[MacroVola] mr_vvix_level: nutze vix_utils (CBOE direkt).", flush=True)
    else:
        vvix_series = _vol_index_series_with_fallback(
            label="mr_vvix_level",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("VVIXCLS",),
            yahoo_symbols=("^VVIX",),
        )
    out = _merge_yahoo_level(out, vvix_series, "mr_vvix_level")
    out = _impute_vvix_level_series(out, col_name="mr_vvix_level", ffill_limit=5, bfill_limit=1)
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

    # ── HY-Spread (FRED: ICE BofA US HY OAS) ───────────────────────────────
    out = _merge_yahoo_level(
        out,
        _vol_index_series_with_fallback(
            label="mr_hy_spread",
            d_min=d_min,
            d_max=d_max,
            fred_ids=("BAMLH0A0HYM2",),
            yahoo_symbols=(),  # Yahoo führt keine direkte HY-OAS-Serie.
        ),
        "mr_hy_spread",
    )
    if "mr_hy_spread" in out.columns:
        ud_hy = (
            out[["Date", "mr_hy_spread"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date", kind="mergesort")
        )
        # FRED-OAS ist ein „business day"-Stream mit gelegentlichen NaNs (Feiertage) —
        # ffill auf den Tageskalender, damit Returns nicht in NaN-Lücken laufen.
        hy = pd.to_numeric(ud_hy["mr_hy_spread"], errors="coerce").ffill(limit=5)
        ud_hy = ud_hy.assign(
            mr_hy_spread=hy.values,
            mr_hy_spread_ret5d=hy / hy.shift(5).replace(0, np.nan) - 1.0,
            mr_hy_spread_ret1d=hy / hy.shift(1).replace(0, np.nan) - 1.0,
        )
        out = out.drop(columns=["mr_hy_spread"], errors="ignore").merge(
            ud_hy[["Date", "mr_hy_spread", "mr_hy_spread_ret5d", "mr_hy_spread_ret1d"]],
            on="Date",
            how="left",
        )

    # ── DXY (US-Dollar-Index) — Yahoo, mit FRED-DTWEXBGS-Fallback ──────────
    dxy_ser = _vol_index_series_with_fallback(
        label="mr_dxy_level",
        d_min=d_min,
        d_max=d_max,
        fred_ids=("DTWEXBGS",),
        yahoo_symbols=("DX-Y.NYB", "DX=F"),
    )
    out = _merge_yahoo_level(out, dxy_ser, "mr_dxy_level")
    if "mr_dxy_level" in out.columns:
        ud_dxy = (
            out[["Date", "mr_dxy_level"]]
            .drop_duplicates(subset=["Date"])
            .sort_values("Date", kind="mergesort")
        )
        dxy = pd.to_numeric(ud_dxy["mr_dxy_level"], errors="coerce").ffill(limit=5)
        ud_dxy = ud_dxy.assign(
            mr_dxy_level=dxy.values,
            mr_dxy_mom_20d=dxy / dxy.shift(20).replace(0, np.nan) - 1.0,
            mr_dxy_mom_60d=dxy / dxy.shift(60).replace(0, np.nan) - 1.0,
        )
        out = out.drop(columns=["mr_dxy_level"], errors="ignore").merge(
            ud_dxy[["Date", "mr_dxy_level", "mr_dxy_mom_20d", "mr_dxy_mom_60d"]],
            on="Date",
            how="left",
        )

    return out
