"""stock_rally_v10 — technische Indikatoren (Pipeline-Modul)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import ta

from lib.stock_rally_v10 import config as cfg

def _linreg_slope(values):
    """Slope of OLS line through 'values' (length 5 expected)."""
    n = len(values)
    x = np.arange(n, dtype=float) - np.mean(np.arange(n, dtype=float))
    return np.dot(x, values) / np.dot(x, x)


def _compute_indicators_one_ticker(df_t):
    """
    Compute all indicator variants for one ticker.
    Returns a DataFrame with all precomputed columns.
    """
    df = df_t.sort_values('Date').copy().reset_index(drop=True)
    close  = df['close']
    volume = df['volume']

    # ── Log return for vol_stress ───────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))

    # ── Vol stress ─────────────────────────────────────────────────────────
    df['vol_stress'] = (
        log_ret.rolling(10, min_periods=5).std() /
        log_ret.rolling(60, min_periods=20).std()
    )

    # ── Drawdown 60d ───────────────────────────────────────────────────────
    rolling_max_60 = close.rolling(60, min_periods=1).max()
    df['drawdown'] = (close - rolling_max_60) / rolling_max_60

    # ── MACD ───────────────────────────────────────────────────────────────
    df['macd_diff'] = ta.trend.MACD(
        close, window_slow=26, window_fast=12, window_sign=9
    ).macd_diff()

    # ── ADX (synthetic H/L from close) ────────────────────────────────────
    df['adx'] = ta.trend.ADXIndicator(
        high=close * 1.01, low=close * 0.99, close=close, window=14
    ).adx()
    df['adx_delta_3d'] = df['adx'].diff(3)

    # ── Volume features ────────────────────────────────────────────────────
    vol_5d_mean  = volume.rolling(5,  min_periods=1).mean()
    vol_20d_mean = volume.rolling(20, min_periods=5).mean()
    vol_60d_mean = volume.rolling(60, min_periods=20).mean()
    vol_60d_std  = volume.rolling(60, min_periods=20).std()

    df['vol_ratio']     = vol_5d_mean / (vol_20d_mean + 1e-10)
    df['volume_zscore'] = (vol_5d_mean - vol_60d_mean) / (vol_60d_std + 1e-10)

    # ── Momentum (20d für Accel/BTC; weitere Fenster für rel_momentum_* in assemble) ─
    df['momentum_20d'] = close.pct_change(20)
    df['momentum_accel'] = df['momentum_20d'].diff(5)
    for _mw in cfg.REL_MOMENTUM_WINDOWS:
        _mw = int(_mw)
        if _mw == 20:
            continue
        df[f'momentum_{_mw}d'] = close.pct_change(_mw)

    # ── Regime features ────────────────────────────────────────────────────
    sma200 = close.rolling(200, min_periods=150).mean()
    df['close_vs_sma200'] = close / sma200
    df['sma200_delta_5d'] = df['close_vs_sma200'].diff(5)

    rolling_max_252 = close.rolling(252, min_periods=60).max()
    df['drawdown_252d'] = (close - rolling_max_252) / (rolling_max_252 + 1e-10)

    # ── RSI — all windows ──────────────────────────────────────────────────
    for w in cfg.RSI_WINDOWS:
        df[f'rsi_{w}'] = ta.momentum.rsi(close, window=w)
        df[f'rsi_delta_3d_{w}'] = df[f'rsi_{w}'].diff(3)

        # Weekly RSI (resample → ffill)
        try:
            weekly = close.copy()
            weekly.index = df['Date']
            weekly_c = weekly.resample('W-FRI').last().dropna()
            if len(weekly_c) >= w + 5:
                rsi_wk = ta.momentum.rsi(weekly_c, window=w)
                rsi_wk = rsi_wk.reindex(df['Date'], method='ffill')
                df[f'rsi_weekly_{w}'] = rsi_wk.values
            else:
                df[f'rsi_weekly_{w}'] = np.nan
        except Exception:
            df[f'rsi_weekly_{w}'] = np.nan

    # ── Bollinger Bands — all windows ──────────────────────────────────────
    for w in cfg.BB_WINDOWS:
        bb = ta.volatility.BollingerBands(close, window=w, window_dev=2)
        df[f'bb_pband_{w}'] = bb.bollinger_pband()
        df[f'bb_delta_3d_{w}'] = df[f'bb_pband_{w}'].diff(3)

        # Bollinger slope (rolling OLS over 5 points)
        df[f'bb_slope_5d_{w}'] = (
            df[f'bb_pband_{w}']
            .rolling(5)
            .apply(_linreg_slope, raw=True)
        )

    # ── Interaction: BB × RSI (all combinations) ──────────────────────────
    for bw in cfg.BB_WINDOWS:
        for rw in cfg.RSI_WINDOWS:
            df[f'bb_x_rsi_{bw}_{rw}'] = (
                df[f'bb_pband_{bw}'] * (df[f'rsi_{rw}'] / 100.0)
            )

    # ── SMA cross + sma_ratio (for market breadth) ─────────────────────────
    sma20 = close.rolling(20, min_periods=15).mean()
    for sw in cfg.SMA_WINDOWS:
        sma_sw = close.rolling(sw, min_periods=int(0.6 * sw)).mean()
        df[f'sma_cross_20_{sw}'] = sma20 / (sma_sw + 1e-10)
        df[f'sma_ratio_{sw}']    = close / (sma_sw + 1e-10)

    return df


def _compute_indicators_one_ticker_for_meta(df_t, rsi_w, bb_w, sma_w):
    """Nur Indikatoren für die trainierten Fenster rsi_w/bb_w/sma_w (entspricht FEAT_COLS-Technik)."""
    df = df_t.sort_values("Date").copy().reset_index(drop=True)
    close = df["close"]
    volume = df["volume"]
    log_ret = np.log(close / close.shift(1))
    df["vol_stress"] = (
        log_ret.rolling(10, min_periods=5).std() / log_ret.rolling(60, min_periods=20).std()
    )
    rolling_max_60 = close.rolling(60, min_periods=1).max()
    df["drawdown"] = (close - rolling_max_60) / rolling_max_60
    df["macd_diff"] = ta.trend.MACD(
        close, window_slow=26, window_fast=12, window_sign=9
    ).macd_diff()
    df["adx"] = ta.trend.ADXIndicator(
        high=close * 1.01, low=close * 0.99, close=close, window=14
    ).adx()
    df["adx_delta_3d"] = df["adx"].diff(3)
    vol_5d_mean = volume.rolling(5, min_periods=1).mean()
    vol_20d_mean = volume.rolling(20, min_periods=5).mean()
    vol_60d_mean = volume.rolling(60, min_periods=20).mean()
    vol_60d_std = volume.rolling(60, min_periods=20).std()
    df["vol_ratio"] = vol_5d_mean / (vol_20d_mean + 1e-10)
    df["volume_zscore"] = (vol_5d_mean - vol_60d_mean) / (vol_60d_std + 1e-10)
    df["momentum_20d"] = close.pct_change(20)
    df["momentum_accel"] = df["momentum_20d"].diff(5)
    for _mw in cfg.REL_MOMENTUM_WINDOWS:
        _mw = int(_mw)
        if _mw == 20:
            continue
        df[f"momentum_{_mw}d"] = close.pct_change(_mw)
    sma200 = close.rolling(200, min_periods=150).mean()
    df["close_vs_sma200"] = close / sma200
    df["sma200_delta_5d"] = df["close_vs_sma200"].diff(5)
    rolling_max_252 = close.rolling(252, min_periods=60).max()
    df["drawdown_252d"] = (close - rolling_max_252) / (rolling_max_252 + 1e-10)
    rw = int(rsi_w)
    df[f"rsi_{rw}"] = ta.momentum.rsi(close, window=rw)
    df[f"rsi_delta_3d_{rw}"] = df[f"rsi_{rw}"].diff(3)
    try:
        weekly = close.copy()
        weekly.index = df["Date"]
        weekly_c = weekly.resample("W-FRI").last().dropna()
        if len(weekly_c) >= rw + 5:
            rsi_wk = ta.momentum.rsi(weekly_c, window=rw)
            rsi_wk = rsi_wk.reindex(df["Date"], method="ffill")
            df[f"rsi_weekly_{rw}"] = rsi_wk.values
        else:
            df[f"rsi_weekly_{rw}"] = np.nan
    except Exception:
        df[f"rsi_weekly_{rw}"] = np.nan
    bw = int(bb_w)
    bb = ta.volatility.BollingerBands(close, window=bw, window_dev=2)
    df[f"bb_pband_{bw}"] = bb.bollinger_pband()
    df[f"bb_delta_3d_{bw}"] = df[f"bb_pband_{bw}"].diff(3)
    df[f"bb_slope_5d_{bw}"] = df[f"bb_pband_{bw}"].rolling(5).apply(_linreg_slope, raw=True)
    df[f"bb_x_rsi_{bw}_{rw}"] = df[f"bb_pband_{bw}"] * (df[f"rsi_{rw}"] / 100.0)
    sma20 = close.rolling(20, min_periods=15).mean()
    sw = int(sma_w)
    sma_sw = close.rolling(sw, min_periods=int(0.6 * sw)).mean()
    df[f"sma_cross_20_{sw}"] = sma20 / (sma_sw + 1e-10)
    df[f"sma_ratio_{sw}"] = close / (sma_sw + 1e-10)
    return df



def add_technical_indicators(df, meta_only=False):
    """Compute all indicators for all tickers in parallel.
    meta_only=True: nur rsi_w/bb_w/sma_w (wie FEAT_COLS) — für OOS neue Ticker ohne volle RSI/BB/SMA-Raster.
    """
    groups = list(df.groupby("ticker"))
    if meta_only:
        _rw = cfg.__dict__.get("rsi_w")
        _bw = cfg.__dict__.get("bb_w")
        _sw = cfg.__dict__.get("sma_w")
        if _rw is None or _bw is None or _sw is None:
            raise ValueError("meta_only=True: rsi_w, bb_w, sma_w müssen gesetzt sein (nach Meta-/Phase-4-Params).")

        def _fn(g):
            return _compute_indicators_one_ticker_for_meta(g[1], _rw, _bw, _sw)

        with ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
            results = list(ex.map(_fn, groups))
        out = pd.concat(results, ignore_index=True)
        print(
            f"Indicators computed (meta-only windows RSI={_rw}, BB={_bw}, SMA={_sw}). Shape: {out.shape}"
        )
        return out
    with ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
        results = list(ex.map(lambda g: _compute_indicators_one_ticker(g[1]), groups))
    out = pd.concat(results, ignore_index=True)
    print(f"Indicators computed. Shape: {out.shape}")
    return out