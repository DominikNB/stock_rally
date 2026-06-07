"""Phase 11 Variante A: Survivors pro News-Tag + Optuna-Schnittmenge."""
from __future__ import annotations

from types import SimpleNamespace

from lib.stock_rally_v10.optuna_train import intersect_feat_cols_with_statistical_prune
from lib.stock_rally_v10.training_phases.feature_pre_pruning import (
    news_params_from_tag,
    resolve_statistical_survivors_for_tag,
)


def test_news_params_from_tag_parses_triple():
    bp = news_params_from_tag("5_10_3", {"news_extra_zscore_w": 7})
    assert bp["news_mom_w"] == 5
    assert bp["news_vol_ma"] == 10
    assert bp["news_tone_roll"] == 3
    assert bp["news_extra_zscore_w"] == 7


def test_resolve_survivors_per_trial_tag():
    cfg = SimpleNamespace(
        FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG={
            "5_10_3": ["news_macro_5_10_3_tone", "mr_x", "rsi_21d"],
            "3_20_3": ["news_macro_3_20_3_tone", "mr_x"],
        },
        FEAT_COLS_STATISTICAL_SURVIVORS=["legacy"],
    )
    s5 = resolve_statistical_survivors_for_tag(cfg, "5_10_3")
    s3 = resolve_statistical_survivors_for_tag(cfg, "3_20_3")
    assert "news_macro_5_10_3_tone" in s5
    assert "news_macro_3_20_3_tone" not in s5
    assert "news_macro_3_20_3_tone" in s3


def test_intersect_keeps_filtered_news_for_trial_tag_only():
    cfg = SimpleNamespace(
        OPTUNA_INTERSECT_FEAT_COLS_WITH_STATISTICAL_PRUNE=True,
        FEAT_COLS_STATISTICAL_SURVIVORS_BY_TAG={
            "5_10_3": ["news_macro_5_10_3_tone", "mr_x", "rsi_21d"],
        },
        FEAT_COLS_STATISTICAL_SURVIVORS=None,
        _STATISTICAL_PRE_PRUNE_ARTIFACT=None,
        STATISTICAL_PRE_PRUNE_STRICT_NEWS_IN_OPTUNA=True,
        STATISTICAL_PRE_PRUNE_WHITELIST_PREFIXES=("mr_", "regime_"),
        STATISTICAL_PRE_PRUNE_WHITELIST_EXACT=("sector_id",),
        STATISTICAL_PRE_PRUNE_MIN_KEEP=2,
    )
    feat = [
        "news_macro_5_10_3_tone",
        "news_macro_5_10_3_vol",
        "mr_x",
        "rsi_21d",
    ]
    out = intersect_feat_cols_with_statistical_prune(
        feat, cfg_mod=cfg, trial_news_tag="5_10_3"
    )
    assert "news_macro_5_10_3_tone" in out
    assert "news_macro_5_10_3_vol" not in out
    assert "mr_x" in out
    assert len(out) == 3
