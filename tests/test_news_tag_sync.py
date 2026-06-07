"""Tests für News-Tag-Synchronisation in Meta-Top-K."""
from __future__ import annotations

from types import SimpleNamespace

from lib.stock_rally_v10.news_tag_sync import (
    rewrite_topk_names_for_news_tag,
    resolve_topk_idx,
    sync_topk_for_meta,
)


def test_rewrite_topk_news_macro_tag():
    cfg = SimpleNamespace(
        news_feat_tag=lambda m, v, t: f"{m}_{v}_{t}",
    )
    bp = {"news_mom_w": 5, "news_vol_ma": 10, "news_tone_roll": 3}
    names = ["news_macro_3_20_10_tone_z_w30", "yz_vol_60d"]
    synced, tag, changes = rewrite_topk_names_for_news_tag(names, best_params=bp, cfg_mod=cfg)
    assert tag == "5_10_3"
    assert synced[0] == "news_macro_5_10_3_tone_z_w30"
    assert synced[1] == "yz_vol_60d"
    assert len(changes) == 1


def test_resolve_topk_idx_missing():
    cols = ["a", "b", "c"]
    names, idx, missing = resolve_topk_idx(["b", "x"], cols)
    assert names == ["b"]
    assert idx == [1]
    assert missing == ["x"]


def test_sync_topk_for_meta_updates_cfg():
    cfg = SimpleNamespace(
        news_feat_tag=lambda m, v, t: f"{m}_{v}_{t}",
        topk_names=["news_macro_3_20_10_vol_l5", "rsi_21"],
        best_params={"news_mom_w": 5, "news_vol_ma": 10, "news_tone_roll": 3},
        FEAT_COLS=["news_macro_5_10_3_vol_l5", "rsi_21"],
    )
    sync_topk_for_meta(cfg, verbose=False)
    assert cfg.topk_names[0] == "news_macro_5_10_3_vol_l5"
    assert list(cfg.topk_idx) == [0, 1]
