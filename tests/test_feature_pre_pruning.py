"""Tests Phase-11 statistisches Pre-Pruning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from lib.stock_rally_v10.training_phases.feature_pre_pruning import (
    _resolve_mi_sample_rows,
    _sparsity_keep_columns,
    execute_statistical_filter,
)


def test_sparsity_drops_mostly_sentinel_column():
    sentinel = -1e8
    cfg = SimpleNamespace(
        FEATURE_NUMERIC_NAN_SENTINEL=sentinel,
        STATISTICAL_PRE_PRUNE_MAX_SENTINEL_FRAC=0.98,
        STATISTICAL_PRE_PRUNE_MI_TOP_K=10,
        STATISTICAL_PRE_PRUNE_MI_MIN_SCORE=0.0,
        STATISTICAL_PRE_PRUNE_MIN_KEEP=2,
        STATISTICAL_PRE_PRUNE_WHITELIST_PREFIXES=("mr_", "regime_"),
        STATISTICAL_PRE_PRUNE_WHITELIST_EXACT=("sector_id",),
        RANDOM_STATE=42,
    )
    n = 200
    df = pd.DataFrame(
        {
            "target": np.random.randint(0, 2, n),
            "dense": np.random.randn(n),
            "sparse": np.full(n, sentinel),
            "mr_keep": np.random.randn(n),
            "sector_id": np.ones(n),
        }
    )
    out = execute_statistical_filter(
        df,
        ["dense", "sparse", "mr_keep", "sector_id"],
        cfg_mod=cfg,
    )
    assert "sparse" not in out
    assert "mr_keep" in out
    assert "sector_id" in out
    assert "dense" in out


def test_sparsity_columnwise_drops_sparse():
    sentinel = -1e8
    n = 100
    df = pd.DataFrame({"sparse": np.full(n, sentinel), "dense": np.random.randn(n)})
    keep = _sparsity_keep_columns(df, ["sparse", "dense"], sentinel=sentinel, max_sentinel_frac=0.98)
    assert keep == [False, True]


def test_resolve_mi_sample_rows_auto_caps_large_matrix():
    cfg = SimpleNamespace(
        STATISTICAL_PRE_PRUNE_MI_MAX_ROWS=None,
        STATISTICAL_PRE_PRUNE_MI_MATRIX_BUDGET_MB=180.0,
    )
    cap = _resolve_mi_sample_rows(cfg, n_rows=280_000, n_cols=440)
    assert cap is not None
    assert 8_000 <= cap < 280_000
