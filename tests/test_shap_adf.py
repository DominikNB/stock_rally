"""SHAP-ADF: Pruning-Logik und Base-vs-Meta-Trennung."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.shap_adf import apply_shap_adf, compute_shap_adf_kept


class _Cfg:
    SHAP_ADF_ENABLED = True
    SHAP_ADF_MIN_ABS_SHAP = 0.01
    SHAP_ADF_MIN_KEEP = 3
    SHAP_ADF_ALWAYS_KEEP_PREFIXES = ("mr_",)
    SHAP_ADF_ALWAYS_KEEP_EXACT = ()
    SHAP_ADF_REPLACE_FEAT_COLS = True
    META_SHAP_CUM_FRAC = 0.85
    META_SHAP_TOP_K_MIN = 1
    META_SHAP_TOP_K_MAX = 10


def test_compute_kept_drops_low_shap():
    cols = ["a", "b", "mr_x", "c"]
    shap = {"a": 0.5, "b": 0.001, "mr_x": 0.0, "c": 0.02}
    kept, dropped = compute_shap_adf_kept(
        cols, shap, min_abs_shap=0.01, min_keep=2, always_keep_prefixes=("mr_",)
    )
    assert "a" in kept and "c" in kept
    assert "mr_x" in kept
    assert "b" in dropped


def test_apply_adf_prunes_for_base_training():
    feat = ["f1", "f2", "mr_z", "f3"]
    shap_df = pd.DataFrame(
        {
            "feature_raw": feat,
            "feature": feat,
            "mean_abs_shap": [0.2, 0.0, 0.0, 0.05],
        }
    )
    res = apply_shap_adf(feat_cols=feat, shap_df=shap_df, cfg_mod=_Cfg())
    assert res.feat_cols_full == ["f1", "mr_z", "f3"]
    assert len(res.feat_cols_pruned) == 3
    assert "f2" in res.feat_cols_dropped
    assert all(n in res.feat_cols_pruned for n in res.topk_names)
