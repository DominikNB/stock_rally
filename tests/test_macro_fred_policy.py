"""Tests FRED-Policy-Merge und Fear-&-Greed-Parser."""
from __future__ import annotations

import pandas as pd
import numpy as np
from types import SimpleNamespace

from lib.stock_rally_v10.macro_fear_greed import _parse_cnn_fear_greed_payload
from lib.stock_rally_v10.macro_vol_enrich import enrich_macro_volatility_features


def test_parse_cnn_fear_greed_historical():
    raw = {
        "fear_and_greed_historical": {
            "data": [
                {"x": 1_700_000_000_000, "y": 25.0, "rating": "fear"},
                {"x": 1_700_086_400_000, "y": 55.0, "rating": "greed"},
            ]
        }
    }
    ser = _parse_cnn_fear_greed_payload(raw)
    assert len(ser) == 2
    assert float(ser.iloc[0]) == 25.0


def test_enrich_fred_policy_columns_monkeypatched(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "ticker": ["A", "B"] * len(dates),
            "regime_vix_level": 20.0,
            "regime_spy_realvol_5d_ann": 0.15,
            "momentum_20d": 0.01,
        }
    )
    cfg = SimpleNamespace(
        FRED_MACRO_POLICY_ENABLED=True,
        FRED_SERIES_T10Y2Y="T10Y2Y",
        FRED_SERIES_WALCL="WALCL",
        FRED_SERIES_EFFR=("DFF",),
        FEAR_GREED_ENABLED=False,
    )

    def _fake_fred(series_id, d_min, d_max):
        idx = pd.date_range(pd.Timestamp(d_min), pd.Timestamp(d_max), freq="B")
        base = {"T10Y2Y": -0.5, "WALCL": 8e6, "DFF": 5.25}.get(series_id, 1.0)
        return pd.Series(base + np.linspace(0, 0.1, len(idx)), index=idx)

    def _fake_vol(label, d_min, d_max, fred_ids=(), yahoo_symbols=()):
        for sid in fred_ids:
            ser = _fake_fred(str(sid), d_min, d_max)
            if ser is not None and len(ser):
                return ser
        return None

    monkeypatch.setattr(
        "lib.stock_rally_v10.macro_vol_enrich._vol_index_series_with_fallback",
        _fake_vol,
    )
    monkeypatch.setattr(
        "lib.stock_rally_v10.macro_vol_enrich._vix_utils_vvix_series",
        lambda **_: None,
    )

    out = enrich_macro_volatility_features(df, cfg_mod=cfg)
    assert "mr_t10y2y" in out.columns
    assert "mr_walcl_chg_5d" in out.columns
    assert "mr_effr_ret1d" in out.columns
    assert out["mr_t10y2y"].notna().any()
