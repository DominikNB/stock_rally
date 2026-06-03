"""Leichte Invarianten für Pipeline-Fixes (ohne Yahoo/BigQuery)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_apply_news_fill_nan_and_missing_mask():
    from lib.stock_rally_v10.features import _apply_news_fill

    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "news_macro_x_tone": [0.5, np.nan],
        }
    )
    _apply_news_fill(df, ["news_macro_x_tone", "news_macro_x_vol"])
    assert df["news_macro_x_vol"].isna().all()
    assert float(df.loc[0, "news_macro_x_tone"]) == 0.5
    assert bool(df.loc[1, "news_missing"]) == 1.0
    assert bool(df.loc[0, "news_missing"]) == 0.0


def test_fill_news_sentiment_from_macro():
    from lib.signal_extra_filters import _fill_news_sentiment_column

    df = pd.DataFrame({"news_macro_3_20_10_tone": [0.12, np.nan]})
    out = _fill_news_sentiment_column(df)
    assert out["news_sentiment"].iloc[0] == pytest.approx(0.12)


def test_red_regime_llm_matches_website_badge():
    from lib.red_regime_summary import attach_red_regime_llm_columns, attach_red_regime_summary

    row = {
        "regime_vix_level": 15.0,
        "gld_ret_5d": -0.02,
        "gld_ret5_median_red_ref": 0.01,
        "news_macro_3_20_10_vol_spike": 0.9,
        "macro_vol_spike_median_red_ref": 1.0,
    }
    from lib.vix_regime_ampel import ampel_fields_from_vix

    sig = dict(row)
    sig.update(ampel_fields_from_vix(15.0))
    attach_red_regime_summary(sig)
    df = attach_red_regime_llm_columns(pd.DataFrame([row]))
    assert int(df["quality_red"].iloc[0]) == int(sig["quality_red"])
    assert str(df["red_summary_de"].iloc[0]) == str(sig["red_summary_de"])
    assert str(df["red_context_llm"].iloc[0]) == str(sig["red_summary_de"])


def test_holdout_rows_from_signals_json(tmp_path: Path):
    from holdout.build_holdout_signals_master import holdout_rows_from_signals_json

    p = tmp_path / "signals.json"
    p.write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "ticker": "AAPL",
                        "date": "2024-06-01",
                        "prob": 0.9,
                        "company": "Apple",
                        "sector": "tech",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    df = holdout_rows_from_signals_json(p)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"
    assert str(df.iloc[0]["Date"])[:10] == "2024-06-01"


def test_meta_final_fit_no_eval_set_in_source():
    src = (ROOT / "lib" / "stock_rally_v10" / "training_phases" / "meta_learner.py").read_text(
        encoding="utf-8"
    )
    block = src.split("Finales Meta-Modell trainiert")[0].split("meta_clf.fit")[-1]
    assert "eval_set" not in block.split("Finales Meta")[0]


def test_fixed_y_label_mode_is_rally_plus_entry():
    from lib.stock_rally_v10 import config as cfg

    assert cfg.fixed_y_label_mode() == "rally_plus_entry"
