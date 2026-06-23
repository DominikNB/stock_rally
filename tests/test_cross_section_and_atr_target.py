"""Tests für Cross-Section-Features (Training) und ATR-normalisiertes Target."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_ohlc_ticker(n: int = 80, *, vol_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.4 * vol_scale, size=n))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01 * vol_scale, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01 * vol_scale, size=n))
    return pd.DataFrame(
        {"Date": dates, "close": close, "open": open_, "high": high, "low": low}
    )


def test_cross_section_macro_event_and_spy_relative():
    from lib.stock_rally_v10.cross_section_features import augment_df_cross_section_features

    n = 30
    dates = pd.bdate_range("2026-01-20", periods=n)
    spy_close = 400.0 + np.arange(n) * 0.5
    aapl_close = 180.0 + np.arange(n) * 0.8
    df = pd.concat(
        [
            pd.DataFrame({"ticker": "SPY", "Date": dates, "close": spy_close}),
            pd.DataFrame({"ticker": "AAPL", "Date": dates, "close": aapl_close}),
        ],
        ignore_index=True,
    )
    out = augment_df_cross_section_features(df)
    assert "macro_event_within_2bd" in out.columns
    assert "ret_vs_spy_5d" in out.columns
    macro_jan28 = out.loc[out["Date"] == pd.Timestamp("2026-01-28"), "macro_event_within_2bd"].iloc[0]
    assert macro_jan28 == 1.0
    aapl = out[out["ticker"] == "AAPL"].sort_values("Date")
    spy = out[out["ticker"] == "SPY"].sort_values("Date")
    i = 25
    expected = float(aapl["close"].pct_change(5).iloc[i] - spy["close"].pct_change(5).iloc[i])
    assert aapl["ret_vs_spy_5d"].iloc[i] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_hybrid_ceiling_never_above_fixed():
    from lib.stock_rally_v10.target import (
        _compute_atr_pct_series,
        _rally_threshold_at,
    )

    wild = _make_ohlc_ticker(60, vol_scale=3.0)
    atr_wild = _compute_atr_pct_series(
        wild["high"].to_numpy(), wild["low"].to_numpy(), wild["close"].to_numpy(), window=14
    )
    idx = 40
    rt = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="hybrid_ceiling", atr_pct=atr_wild, atr_k=1.5
    )
    assert rt <= 0.045 + 1e-12


def test_hybrid_ceiling_lowers_threshold_for_calm():
    from lib.stock_rally_v10.target import (
        _compute_atr_pct_series,
        _rally_threshold_at,
    )

    calm = _make_ohlc_ticker(60, vol_scale=0.2)
    atr_calm = _compute_atr_pct_series(
        calm["high"].to_numpy(), calm["low"].to_numpy(), calm["close"].to_numpy(), window=14
    )
    idx = 40
    rt_calm = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="hybrid_ceiling", atr_pct=atr_calm, atr_k=1.5
    )
    rt_fixed = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="fixed", atr_pct=atr_calm, atr_k=1.5
    )
    assert rt_calm < rt_fixed - 1e-6
    assert rt_calm > 0.0


def test_hybrid_ceiling_more_labels_than_fixed_on_calm(monkeypatch):
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.target import _create_target_one_ticker_fixed_bands

    df = _make_ohlc_ticker(120, vol_scale=0.35)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_WINDOW", 14, raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_K", 1.5, raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "fixed", raising=False)
    _, t_fixed = _create_target_one_ticker_fixed_bands(df)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "hybrid_ceiling", raising=False)
    _, t_ceiling = _create_target_one_ticker_fixed_bands(df)
    assert float(t_ceiling.mean()) >= float(t_fixed.mean()) - 1e-9


def test_hybrid_floor_never_below_fixed():
    from lib.stock_rally_v10.target import (
        _compute_atr_pct_series,
        _rally_threshold_at,
    )

    calm = _make_ohlc_ticker(60, vol_scale=0.2)
    atr_calm = _compute_atr_pct_series(
        calm["high"].to_numpy(), calm["low"].to_numpy(), calm["close"].to_numpy(), window=14
    )
    idx = 40
    rt = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="hybrid_floor", atr_pct=atr_calm, atr_k=1.5
    )
    assert rt >= 0.045 - 1e-12


def test_atr_threshold_scales_with_volatility():
    from lib.stock_rally_v10.target import (
        _compute_atr_pct_series,
        _rally_threshold_at,
    )

    calm = _make_ohlc_ticker(60, vol_scale=0.2)
    wild = _make_ohlc_ticker(60, vol_scale=3.0)
    atr_calm = _compute_atr_pct_series(
        calm["high"].to_numpy(), calm["low"].to_numpy(), calm["close"].to_numpy(), window=14
    )
    atr_wild = _compute_atr_pct_series(
        wild["high"].to_numpy(), wild["low"].to_numpy(), wild["close"].to_numpy(), window=14
    )
    idx = 40
    rt_calm = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="hybrid_floor", atr_pct=atr_calm, atr_k=1.5
    )
    rt_wild = _rally_threshold_at(
        idx, rt_fixed=0.045, threshold_mode="hybrid_floor", atr_pct=atr_wild, atr_k=1.5
    )
    assert rt_wild > rt_calm
    assert rt_calm == pytest.approx(0.045, abs=1e-6)
    assert np.isfinite(rt_calm) and np.isfinite(rt_wild)


def test_hybrid_floor_fewer_labels_than_pure_atr(monkeypatch):
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.target import _create_target_one_ticker_fixed_bands

    df = _make_ohlc_ticker(120, vol_scale=1.0)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_WINDOW", 14, raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_K", 1.5, raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "hybrid_floor", raising=False)
    _, t_hybrid = _create_target_one_ticker_fixed_bands(df)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "fixed", raising=False)
    _, t_fixed = _create_target_one_ticker_fixed_bands(df)
    assert float(t_hybrid.mean()) <= float(t_fixed.mean()) + 1e-9


def test_atr_target_mode_produces_valid_labels(monkeypatch):
    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.target import _create_target_one_ticker_fixed_bands

    df = _make_ohlc_ticker(120, vol_scale=1.0)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "fixed", raising=False)
    _, t_fixed = _create_target_one_ticker_fixed_bands(df)
    monkeypatch.setattr(cfg, "FIXED_Y_RALLY_THRESHOLD_MODE", "hybrid_ceiling", raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_WINDOW", 14, raising=False)
    monkeypatch.setattr(cfg, "FIXED_Y_ATR_K", 1.5, raising=False)
    _, t_hybrid = _create_target_one_ticker_fixed_bands(df)
    assert t_fixed.dtype == np.int8
    assert t_hybrid.dtype == np.int8
    assert 0.0 <= float(t_fixed.mean()) <= 1.0
    assert 0.0 <= float(t_hybrid.mean()) <= 1.0


def test_append_training_cross_section_cols():
    from lib.stock_rally_v10.cross_section_features import append_training_cross_section_cols

    df = pd.DataFrame(
        {
            "macro_event_within_2bd": [0.0, 1.0],
            "ret_vs_spy_5d": [0.01, -0.02],
            "ret_vs_spy_20d": [0.05, np.nan],
        }
    )
    out = append_training_cross_section_cols(["rsi_14"], df)
    assert "macro_event_within_2bd" in out
    assert "ret_vs_spy_5d" in out
    assert "ret_vs_spy_20d" in out
    assert out.index("rsi_14") < out.index("macro_event_within_2bd")
