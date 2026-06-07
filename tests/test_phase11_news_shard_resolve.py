"""Phase-11 News-Shard-Auflösung."""
from __future__ import annotations

from types import SimpleNamespace

from lib.stock_rally_v10.training_phases.feature_pre_pruning import (
    resolve_news_params_for_phase11,
)


def test_resolve_fallback_from_invalid_tone_roll():
    cfg = SimpleNamespace(
        news_feat_tag=lambda m, v, t: f"{m}_{v}_{t}",
        NEWS_SHARD_MANIFEST={"3_20_3": "x.parquet", "3_20_10": "y.parquet"},
        NEWS_TONE_ROLL_WINDOWS=(3, 5, 10),
        NEWS_MOM_WINDOWS=(3,),
        NEWS_VOL_MA_WINDOWS=(20,),
        STATISTICAL_PRE_PRUNE_NEWS_TAG="5_10_3",
    )
    bp = resolve_news_params_for_phase11(
        cfg,
        {"news_mom_w": 3, "news_vol_ma": 20, "news_tone_roll": 1},
    )
    assert bp["news_tone_roll"] == 3
