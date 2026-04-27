"""stock_rally_v10 — technische Indikatoren (Pipeline-Modul).

Eine einzige Kernfunktion ``_compute_indicators_for_ticker`` berechnet alle Indikatoren auf
übergebenen Fenster-Listen. ``add_technical_indicators`` ruft sie sowohl im vollen
Optuna-Modus als auch im ``meta_only``-Modus mit den jeweils passenden Fenstern auf.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable

import numpy as np
import pandas as pd
import ta

from lib.stock_rally_v10 import config as cfg


def _linreg_slope(values: np.ndarray) -> float:
    """Slope der OLS-Geraden (für ``rolling.apply(raw=True)``)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float) - np.mean(np.arange(n, dtype=float))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, values) / denom)


def _ensure_windows(seq: Iterable[int] | None) -> list[int]:
    return [int(w) for w in (seq or [])]


def _yang_zhang_vol_rolling(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    """Yang-Zhang annualisierte Vola (Rolling) — gap-robust, OHLC-basiert.

    σ²_YZ = σ²_overnight + k·σ²_oc + (1−k)·σ²_RS,
        k = 0.34 / (1.34 + (n+1)/(n−1)).
    """
    n = max(2, int(window))
    log_oo = np.log(open_ / close.shift(1))
    log_oc = np.log(close / open_)
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)
    rs = log_hc * log_ho + log_lc * log_lo  # Rogers-Satchell
    var_overnight = log_oo.rolling(n, min_periods=max(3, n // 2)).var()
    var_oc = log_oc.rolling(n, min_periods=max(3, n // 2)).var()
    var_rs = rs.rolling(n, min_periods=max(3, n // 2)).mean()
    k = 0.34 / (1.34 + (n + 1.0) / max(2, n - 1))
    yz_var = var_overnight + k * var_oc + (1.0 - k) * var_rs
    yz_var = yz_var.clip(lower=0.0)
    return np.sqrt(yz_var) * np.sqrt(252.0)


def _compute_indicators_for_ticker(
    df_t: pd.DataFrame,
    *,
    rsi_windows: Iterable[int],
    bb_windows: Iterable[int],
    sma_windows: Iterable[int],
    adr_windows: Iterable[int],
    breakout_windows: Iterable[int],
    vcp_windows: Iterable[int],
    rel_momentum_windows: Iterable[int],
    downside_vol_windows: Iterable[int],
    ret_moment_windows: Iterable[int],
    yz_vol_windows: Iterable[int],
    amihud_windows: Iterable[int],
    vcp_lower_low_windows: Iterable[int],
    breakout_volume_trigger_options: Iterable[float],
) -> pd.DataFrame:
    """Berechnet alle technischen Indikatoren eines Tickers.

    Wird sowohl im Voll-Modus (alle Fenster aus ``cfg.*_WINDOWS``) als auch im ``meta_only``-Modus
    (genau die trainierten Fenster) aufgerufen.
    """
    df = df_t.sort_values("Date").copy().reset_index(drop=True)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = pd.to_numeric(df.get("high", close), errors="coerce").fillna(close).astype(float)
    low = pd.to_numeric(df.get("low", close), errors="coerce").fillna(close).astype(float)
    open_ = pd.to_numeric(df.get("open", close), errors="coerce").fillna(close).astype(float)

    rsi_windows_l = _ensure_windows(rsi_windows)
    bb_windows_l = _ensure_windows(bb_windows)
    sma_windows_l = _ensure_windows(sma_windows)
    adr_windows_l = _ensure_windows(adr_windows)
    breakout_windows_l = _ensure_windows(breakout_windows)
    vcp_windows_l = _ensure_windows(vcp_windows)
    rel_momentum_windows_l = _ensure_windows(rel_momentum_windows)
    downside_vol_windows_l = _ensure_windows(downside_vol_windows)
    ret_moment_windows_l = _ensure_windows(ret_moment_windows)
    yz_vol_windows_l = _ensure_windows(yz_vol_windows)
    amihud_windows_l = _ensure_windows(amihud_windows)
    vcp_lower_low_windows_l = _ensure_windows(vcp_lower_low_windows)
    breakout_volume_trigger_l = [float(x) for x in (breakout_volume_trigger_options or [])]
    if not breakout_volume_trigger_l:
        breakout_volume_trigger_l = [1.0]

    # ── Tagesrenditen / Logreturns ──────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    ret_1d = close.pct_change()

    # ── Vol stress (kurz / lang) ───────────────────────────────────────────
    df["vol_stress"] = (
        log_ret.rolling(10, min_periods=5).std()
        / log_ret.rolling(60, min_periods=20).std()
    )

    # ── Drawdown 60d ───────────────────────────────────────────────────────
    rolling_max_60 = close.rolling(60, min_periods=1).max()
    df["drawdown"] = (close - rolling_max_60) / rolling_max_60

    # ── MACD ───────────────────────────────────────────────────────────────
    df["macd_diff"] = ta.trend.MACD(
        close, window_slow=26, window_fast=12, window_sign=9
    ).macd_diff()

    # ── ADX mit echten High/Low (zuvor synthetische ±1%-Bänder = unsinnig). ─
    df["adx"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
    df["adx_delta_3d"] = df["adx"].diff(3)

    # ── Volume features ────────────────────────────────────────────────────
    vol_5d_mean = volume.rolling(5, min_periods=1).mean()
    vol_20d_mean = volume.rolling(20, min_periods=5).mean()
    vol_60d_mean = volume.rolling(60, min_periods=20).mean()
    vol_60d_std = volume.rolling(60, min_periods=20).std()
    df["vol_ratio"] = vol_5d_mean / (vol_20d_mean + 1e-10)
    df["volume_zscore"] = (vol_5d_mean - vol_60d_mean) / (vol_60d_std + 1e-10)

    dv = (close * volume).replace([np.inf, -np.inf], np.nan)
    dv_mean20 = dv.rolling(20, min_periods=10).mean()
    dv_mean60 = dv.rolling(60, min_periods=20).mean()
    dv_std60 = dv.rolling(60, min_periods=20).std()
    df["dollar_volume_zscore"] = (dv_mean20 - dv_mean60) / (dv_std60 + 1e-10)
    df["volume_force_1d"] = df["dollar_volume_zscore"] * np.sign(ret_1d.fillna(0.0))

    # ── Momentum ───────────────────────────────────────────────────────────
    df["momentum_20d"] = close.pct_change(20)
    df["momentum_accel"] = df["momentum_20d"].diff(5)
    for _mw in rel_momentum_windows_l:
        if _mw == 20:
            continue
        df[f"momentum_{_mw}d"] = close.pct_change(_mw)

    # ── ADR / VCP-Tightness ────────────────────────────────────────────────
    ret_abs_pct = ret_1d.abs() * 100.0
    for _aw in adr_windows_l:
        df[f"adr_pct_{_aw}d"] = ret_abs_pct.rolling(
            _aw, min_periods=max(3, _aw // 2)
        ).mean()
    hl_range_pct = (high - low) / (close.abs() + 1e-10)
    for _vw in vcp_windows_l:
        short = ret_abs_pct.rolling(_vw, min_periods=max(3, _vw // 2)).mean()
        long = ret_abs_pct.rolling(_vw * 3, min_periods=max(5, _vw)).mean()
        df[f"vcp_tightness_{_vw}d"] = short / (long + 1e-10)
        short_hl = hl_range_pct.rolling(_vw, min_periods=max(3, _vw // 2)).mean()
        long_hl = hl_range_pct.rolling(_vw * 3, min_periods=max(5, _vw)).mean()
        df[f"vcp_tightness_hl_{_vw}d"] = short_hl / (long_hl + 1e-10)

    # ── VCP Lower-Lows / Trend-Slope (3b5) ─────────────────────────────────
    # Pro vcp_tightness-Fenster: in wie vielen der letzten N Tage war der Wert auf seinem
    # rollenden N-Tage-Minimum? Negative Slope der Tightness = anhaltende Kontraktion.
    for _vw in vcp_windows_l:
        for _ll in vcp_lower_low_windows_l:
            col = f"vcp_tightness_{_vw}d"
            if col not in df.columns:
                continue
            tight = df[col]
            rmin = tight.rolling(_ll, min_periods=max(5, _ll // 3)).min()
            is_at_min = (tight <= rmin + 1e-12).astype(float)
            df[f"vcp_at_period_low_frac_{_vw}_{_ll}d"] = is_at_min.rolling(
                _ll, min_periods=max(5, _ll // 3)
            ).mean()
            df[f"vcp_tightness_slope_{_vw}_{_ll}d"] = (
                tight.rolling(_ll, min_periods=max(5, _ll // 3)).apply(
                    _linreg_slope, raw=True
                )
            )

    # ── Realisierte Vola / Schiefe / Kurtosis (3b1) ────────────────────────
    for _w in downside_vol_windows_l:
        # Downside semi-vola: nur negative Tagesrenditen.
        neg = ret_1d.where(ret_1d < 0, 0.0)
        std_total = ret_1d.rolling(_w, min_periods=max(5, _w // 3)).std()
        std_down = neg.rolling(_w, min_periods=max(5, _w // 3)).std()
        df[f"downside_vol_{_w}d"] = std_down
        df[f"downside_vol_ratio_{_w}d"] = std_down / (std_total + 1e-10)
    for _w in ret_moment_windows_l:
        df[f"ret_skew_{_w}d"] = ret_1d.rolling(
            _w, min_periods=max(10, _w // 3)
        ).skew()
        df[f"ret_kurt_{_w}d"] = ret_1d.rolling(
            _w, min_periods=max(10, _w // 3)
        ).kurt()
    for _w in yz_vol_windows_l:
        df[f"yz_vol_{_w}d"] = _yang_zhang_vol_rolling(
            open_, high, low, close, window=_w
        )

    # ── Amihud-Illiquidität (3b2) ──────────────────────────────────────────
    for _w in amihud_windows_l:
        amihud_inst = ret_1d.abs() / (dv + 1e-10)
        df[f"amihud_illiquidity_{_w}d"] = amihud_inst.rolling(
            _w, min_periods=max(5, _w // 3)
        ).mean()

    # ── Regime / Long-term ─────────────────────────────────────────────────
    sma200 = close.rolling(200, min_periods=150).mean()
    df["close_vs_sma200"] = close / sma200
    df["sma200_delta_5d"] = df["close_vs_sma200"].diff(5)
    rolling_max_252 = close.rolling(252, min_periods=60).max()
    df["drawdown_252d"] = (close - rolling_max_252) / (rolling_max_252 + 1e-10)

    # ── RSI (Tag/Woche) ────────────────────────────────────────────────────
    for w in rsi_windows_l:
        df[f"rsi_{w}"] = ta.momentum.rsi(close, window=w)
        df[f"rsi_delta_3d_{w}"] = df[f"rsi_{w}"].diff(3)
        df[f"rsi_weekly_{w}"] = _rsi_weekly_aligned(close, df["Date"], window=w)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    for w in bb_windows_l:
        bb = ta.volatility.BollingerBands(close, window=w, window_dev=2)
        df[f"bb_pband_{w}"] = bb.bollinger_pband()
        df[f"bb_squeeze_factor_{w}"] = bb.bollinger_wband() / 100.0
        df[f"bb_delta_3d_{w}"] = df[f"bb_pband_{w}"].diff(3)
        df[f"bb_slope_5d_{w}"] = (
            df[f"bb_pband_{w}"].rolling(5).apply(_linreg_slope, raw=True)
        )

    # ── BB × RSI Interaktionen ─────────────────────────────────────────────
    for bw in bb_windows_l:
        for rw in rsi_windows_l:
            df[f"bb_x_rsi_{bw}_{rw}"] = (
                df[f"bb_pband_{bw}"] * (df[f"rsi_{rw}"] / 100.0)
            )

    # ── SMA / Market Breadth Bausteine ─────────────────────────────────────
    sma20 = close.rolling(20, min_periods=15).mean()
    for sw in sma_windows_l:
        sma_sw = close.rolling(sw, min_periods=int(0.6 * sw)).mean()
        df[f"sma_cross_20_{sw}"] = sma20 / (sma_sw + 1e-10)
        df[f"sma_ratio_{sw}"] = close / (sma_sw + 1e-10)

    # ── Breakouts + Volumen-Bestätigung (3b5) ──────────────────────────────
    for _bwk in breakout_windows_l:
        prior_hi = close.shift(1).rolling(_bwk, min_periods=max(20, _bwk // 3)).max()
        df[f"blue_sky_breakout_{_bwk}d"] = (close > prior_hi).astype(float)
        df[f"dist_to_prior_hi_pct_{_bwk}d"] = close / (prior_hi + 1e-10) - 1.0
        near_hi = ((prior_hi - close) / (prior_hi + 1e-10) <= 0.02).astype(float)
        df[f"volume_at_resistance_{_bwk}d"] = near_hi * df["volume_zscore"].fillna(0.0)
        # Pro Trigger eine eigene Spalte; Optuna-Trial wählt eine über cfg.bvt_z_str(z).
        vz = df["volume_zscore"].fillna(0.0)
        for _z in breakout_volume_trigger_l:
            _zs = cfg.bvt_z_str(_z)
            df[f"breakout_volume_confirmed_{_bwk}_z{_zs}d"] = (
                (df[f"blue_sky_breakout_{_bwk}d"] > 0.5) & (vz >= float(_z))
            ).astype(float)

    return df


def _rsi_weekly_aligned(
    close: pd.Series, dates: pd.Series, *, window: int
) -> pd.Series:
    """Wöchentlicher RSI (Resample → ffill) auf Tageskalender alignen."""
    try:
        weekly = close.copy()
        weekly.index = pd.to_datetime(dates)
        weekly_c = weekly.resample("W-FRI").last().dropna()
        if len(weekly_c) < window + 5:
            return pd.Series(np.nan, index=close.index)
        rsi_wk = ta.momentum.rsi(weekly_c, window=window)
        rsi_wk = rsi_wk.reindex(pd.to_datetime(dates), method="ffill")
        return pd.Series(rsi_wk.to_numpy(), index=close.index)
    except (KeyError, ValueError, TypeError) as exc:
        # Bewusste, eng gefasste Exception-Liste statt nacktem ``except`` —
        # alles andere (MemoryError, KeyboardInterrupt) muss durchschlagen.
        print(
            f"_rsi_weekly_aligned[w={window}]: {type(exc).__name__}: {exc} -> NaN-Fallback",
            flush=True,
        )
        return pd.Series(np.nan, index=close.index)


def _full_window_args() -> dict[str, object]:
    """Voller Modus: alle Fenster aus ``cfg.*_WINDOWS`` (Optuna-Raster)."""
    return dict(
        rsi_windows=cfg.RSI_WINDOWS,
        bb_windows=cfg.BB_WINDOWS,
        sma_windows=cfg.SMA_WINDOWS,
        adr_windows=cfg.ADR_WINDOWS,
        breakout_windows=cfg.BREAKOUT_LOOKBACK_WINDOWS,
        vcp_windows=cfg.VCP_WINDOWS,
        rel_momentum_windows=cfg.REL_MOMENTUM_WINDOWS,
        downside_vol_windows=getattr(cfg, "DOWNSIDE_VOL_WINDOWS", [20, 60, 120]),
        ret_moment_windows=getattr(cfg, "RET_MOMENT_WINDOWS", [20, 60, 120]),
        yz_vol_windows=getattr(cfg, "YANG_ZHANG_WINDOWS", [10, 20, 60]),
        amihud_windows=getattr(cfg, "AMIHUD_WINDOWS", [10, 20, 60]),
        vcp_lower_low_windows=getattr(cfg, "VCP_LOWER_LOW_WINDOWS", [20, 60, 120]),
        breakout_volume_trigger_options=getattr(
            cfg, "BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS", [0.5, 1.0, 1.5, 2.0]
        ),
    )


def _meta_only_window_args() -> dict[str, object]:
    """``meta_only=True``: nur die zur Laufzeit gewählten Trainings-Fenster + ein Trigger."""
    bp = getattr(cfg, "best_params", None) or {}
    sp = cfg.SEED_PARAMS
    rsi_w = cfg.__dict__.get("rsi_w")
    bb_w = cfg.__dict__.get("bb_w")
    sma_w = cfg.__dict__.get("sma_w")
    if rsi_w is None or bb_w is None or sma_w is None:
        raise ValueError(
            "meta_only=True: rsi_w/bb_w/sma_w müssen gesetzt sein (nach Meta-/Phase-4-Params)."
        )
    adr_w = int(
        cfg.__dict__.get(
            "adr_w", bp.get("adr_window", sp.get("adr_window", cfg.ADR_WINDOWS[0]))
        )
    )
    breakout_w = int(
        cfg.__dict__.get(
            "breakout_lookback_w",
            bp.get(
                "breakout_lookback_window",
                sp.get("breakout_lookback_window", cfg.BREAKOUT_LOOKBACK_WINDOWS[0]),
            ),
        )
    )
    vcp_w = int(
        cfg.__dict__.get(
            "vcp_w", bp.get("vcp_window", sp.get("vcp_window", cfg.VCP_WINDOWS[0]))
        )
    )
    rel_m_w = int(
        bp.get(
            "rel_momentum_window",
            sp.get("rel_momentum_window", cfg.REL_MOMENTUM_WINDOWS[0]),
        )
    )
    yz_w = int(bp.get("yz_vol_window", sp.get("yz_vol_window", cfg._default_yang_zhang_window())))
    dv_w = int(
        bp.get("downside_vol_window", sp.get("downside_vol_window", cfg._default_downside_vol_window()))
    )
    rm_w = int(bp.get("ret_moment_window", sp.get("ret_moment_window", cfg._default_ret_moment_window())))
    am_w = int(bp.get("amihud_window", sp.get("amihud_window", cfg._default_amihud_window())))
    ll_w = int(
        bp.get("vcp_lower_low_window", sp.get("vcp_lower_low_window", cfg._default_vcp_lower_low_window()))
    )
    bvt_z = float(
        bp.get(
            "breakout_volume_trigger_z",
            sp.get("breakout_volume_trigger_z", cfg._default_breakout_volume_trigger_z()),
        )
    )
    return dict(
        rsi_windows=[int(rsi_w)],
        bb_windows=[int(bb_w)],
        sma_windows=[int(sma_w)],
        adr_windows=[adr_w],
        breakout_windows=[breakout_w],
        vcp_windows=[vcp_w],
        rel_momentum_windows=[rel_m_w],
        downside_vol_windows=[dv_w],
        ret_moment_windows=[rm_w],
        yz_vol_windows=[yz_w],
        amihud_windows=[am_w],
        vcp_lower_low_windows=[ll_w],
        breakout_volume_trigger_options=[bvt_z],
    )


def add_technical_indicators(df: pd.DataFrame, meta_only: bool = False) -> pd.DataFrame:
    """Berechnet alle Indikatoren parallel pro Ticker.

    ``meta_only=True``: nur die trainierten RSI/BB/SMA/ADR/Breakout/VCP-Fenster (FEAT_COLS-Pfad).
    """
    groups = list(df.groupby("ticker"))
    kwargs = _meta_only_window_args() if meta_only else _full_window_args()

    def _fn(group_item: tuple[object, pd.DataFrame]) -> pd.DataFrame:
        return _compute_indicators_for_ticker(group_item[1], **kwargs)

    with ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
        results = list(ex.map(_fn, groups))
    out = pd.concat(results, ignore_index=True)
    if meta_only:
        rsi_w = kwargs["rsi_windows"][0]  # type: ignore[index]
        bb_w = kwargs["bb_windows"][0]  # type: ignore[index]
        sma_w = kwargs["sma_windows"][0]  # type: ignore[index]
        adr_w = kwargs["adr_windows"][0]  # type: ignore[index]
        brk_w = kwargs["breakout_windows"][0]  # type: ignore[index]
        vcp_w = kwargs["vcp_windows"][0]  # type: ignore[index]
        print(
            f"Indicators computed (meta-only windows RSI={rsi_w}, BB={bb_w}, SMA={sma_w}, "
            f"ADR={adr_w}, Breakout={brk_w}, VCP={vcp_w}). Shape: {out.shape}"
        )
    else:
        print(f"Indicators computed. Shape: {out.shape}")
    return out
