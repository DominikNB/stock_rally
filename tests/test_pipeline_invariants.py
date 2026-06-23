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


def test_scoring_only_ignores_retrain_meta_for_features():
    from lib.stock_rally_v10.data_and_split import meta_only_features_for_assemble

    assert meta_only_features_for_assemble(scoring_only=True, retrain_meta_only=True) is False
    assert meta_only_features_for_assemble(scoring_only=True, retrain_meta_only=False) is False
    assert meta_only_features_for_assemble(scoring_only=False, retrain_meta_only=True) is True
    assert meta_only_features_for_assemble(scoring_only=False, retrain_meta_only=False) is False


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


def test_context_tier_llm_matches_website_for_low_vix_no_macro():
    from lib.signal_context_tier import attach_context_tier_llm_columns, context_tier_llm_fields_from_row

    row = {
        "regime_vix_level": 17.28,
        "macro_event_within_2bd": False,
    }
    fields = context_tier_llm_fields_from_row(row)
    assert fields["vix_regime_ampel"] == "yellow"
    assert fields["context_tier"] == "yellow"
    assert "rot" not in fields["red_context_llm"].lower().split("gelb")[0]
    df = attach_context_tier_llm_columns(pd.DataFrame([row]))
    assert str(df["vix_regime_ampel"].iloc[0]) == "yellow"


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


def test_ticker_ohlc_includes_high_low_for_atr_labels():
    from holdout.build_holdout_signals_master import _ticker_ohlc

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {
            ("Open", "AAPL"): [100.0, 101.0],
            ("High", "AAPL"): [102.0, 103.0],
            ("Low", "AAPL"): [99.0, 100.0],
            ("Close", "AAPL"): [101.0, 102.0],
        },
        index=idx,
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    ohlc = _ticker_ohlc(raw, "AAPL")
    assert ohlc is not None
    assert list(ohlc.columns) == ["Open", "Close", "High", "Low"]


def test_ticker_ohlc_synthesizes_high_low_when_missing():
    from holdout.build_holdout_signals_master import _ticker_ohlc

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {
            ("Open", "AAPL"): [100.0, 101.0],
            ("Close", "AAPL"): [101.0, 102.0],
        },
        index=idx,
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    ohlc = _ticker_ohlc(raw, "AAPL")
    assert ohlc is not None
    assert list(ohlc["High"]) == [101.0, 102.0]
    assert list(ohlc["Low"]) == [100.0, 101.0]

    # group_by="ticker" Layout (wie Training-Download)
    raw2 = pd.DataFrame(
        {
            ("AAPL", "Open"): [100.0, 101.0],
            ("AAPL", "High"): [102.0, 103.0],
            ("AAPL", "Low"): [99.0, 100.0],
            ("AAPL", "Close"): [101.0, 102.0],
        },
        index=idx,
    )
    raw2.columns = pd.MultiIndex.from_tuples(raw2.columns)
    ohlc2 = _ticker_ohlc(raw2, "AAPL")
    assert ohlc2 is not None
    assert list(ohlc2.columns) == ["Open", "Close", "High", "Low"]


def test_read_signals_for_latest_day_filters_daily_csv(tmp_path: Path, monkeypatch):
    import pandas as pd

    from scripts import website_analysis_common as wac

    daily = tmp_path / "master_daily_update.csv"
    pd.DataFrame(
        {
            "Date": ["2026-06-15", "2026-06-16", "2026-06-16"],
            "ticker": ["A", "B", "C"],
            "prob": [0.9, 0.8, 0.7],
            "threshold_used": [0.5, 0.5, 0.5],
        }
    ).to_csv(daily, index=False)
    monkeypatch.setattr(wac, "DAILY_CSV", daily)
    monkeypatch.setattr(wac, "COMPLETE_CSV", tmp_path / "missing_complete.csv")
    monkeypatch.setattr(wac, "HOLDOUT_CSV", tmp_path / "missing_holdout.csv")
    monkeypatch.setenv("ANALYSIS_EXPECT_SIGNAL_DATE", "2026-06-16")

    latest, sub = wac.read_signals_for_latest_day()
    assert latest == "2026-06-16"
    assert len(sub) == 2
    assert set(sub["ticker"]) == {"B", "C"}


def test_llm_analysis_is_stale_detects_older_meta(tmp_path: Path, monkeypatch):
    from scripts import website_analysis_common as wac

    daily = tmp_path / "master_daily_update.csv"
    daily.write_text("Date,ticker\n2026-06-16,X\n", encoding="utf-8")
    meta = tmp_path / "analysis_llm_last_run_meta.json"
    meta.write_text('{"signaltag_csv": "2026-06-15"}\n', encoding="utf-8")
    monkeypatch.setattr(wac, "DAILY_CSV", daily)
    monkeypatch.setattr(wac, "OUT_RUN_META", meta)

    assert wac.llm_analysis_is_stale() is True
    assert wac.llm_analysis_is_stale("2026-06-16") is True
    assert wac.llm_analysis_is_stale("2026-06-15") is False
    assert wac.llm_analysis_is_stale("2026-06-18") is True


def test_read_signals_empty_when_csv_latest_before_scoring_day(tmp_path: Path, monkeypatch):
    import pandas as pd

    from scripts import website_analysis_common as wac

    daily = tmp_path / "master_daily_update.csv"
    pd.DataFrame(
        {
            "Date": ["2026-06-16", "2026-06-16"],
            "ticker": ["A", "B"],
            "prob": [0.9, 0.8],
            "threshold_used": [0.5, 0.5],
        }
    ).to_csv(daily, index=False)
    monkeypatch.setattr(wac, "DAILY_CSV", daily)
    monkeypatch.setattr(wac, "COMPLETE_CSV", tmp_path / "missing_complete.csv")
    monkeypatch.setattr(wac, "HOLDOUT_CSV", tmp_path / "missing_holdout.csv")
    monkeypatch.setenv("ANALYSIS_EXPECT_SIGNAL_DATE", "2026-06-18")

    latest, sub = wac.read_signals_for_latest_day()
    assert latest == "2026-06-18"
    assert sub.empty


def test_read_signals_returns_yesterday_when_one_session_lag(tmp_path: Path, monkeypatch):
    import pandas as pd

    from scripts import website_analysis_common as wac

    daily = tmp_path / "master_daily_update.csv"
    pd.DataFrame(
        {
            "Date": ["2026-06-22"],
            "ticker": ["NDX1.DE"],
            "prob": [0.9],
            "threshold_used": [0.5],
        }
    ).to_csv(daily, index=False)
    monkeypatch.setattr(wac, "DAILY_CSV", daily)
    monkeypatch.setattr(wac, "COMPLETE_CSV", tmp_path / "missing_complete.csv")
    monkeypatch.setattr(wac, "HOLDOUT_CSV", tmp_path / "missing_holdout.csv")
    monkeypatch.setenv("ANALYSIS_EXPECT_SIGNAL_DATE", "2026-06-23")

    latest, sub = wac.read_signals_for_latest_day()
    assert latest == "2026-06-22"
    assert len(sub) == 1
    assert sub.iloc[0]["ticker"] == "NDX1.DE"


def test_analysis_scoring_run_end_reads_env(monkeypatch):
    from scripts import website_analysis_common as wac

    monkeypatch.setenv("ANALYSIS_SCORING_RUN_END", "2026-06-23T08:15:00+02:00")
    end = wac.analysis_scoring_run_end()
    assert end.tzinfo is not None
    assert end.year == 2026 and end.month == 6 and end.day == 23
    assert end.hour == 8 and end.minute == 15


def test_steering_block_zusatzfenster_until_scoring_run():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from scripts import website_analysis_common as wac

    scoring_end = datetime(2026, 6, 23, 9, 30, tzinfo=ZoneInfo("Europe/Berlin"))
    block, meta = wac.build_llm_steering_block(
        "2026-06-22",
        scoring_day="2026-06-23",
        scoring_end=scoring_end,
    )
    assert meta["zusatzfenster_active"] is True
    assert meta["scoring_day"] == "2026-06-23"
    assert "2026-06-23T09:30:00" in meta["zusatzfenster_end_europe_berlin_inclusive"]
    assert "Zusatzfenster Nachrichten" in block
    assert "Bewertungsregel" in block
    assert "abschließende Fazit" in block


def test_scoring_mandate_requires_integrated_fazit():
    from scripts import website_analysis_common as wac

    meta = {"zusatzfenster_active": True, "zusatzfenster_end_europe_berlin_inclusive": "2026-06-23T09:00:00+02:00"}
    text = wac.build_scoring_news_mandate_block("2026-06-22", "2026-06-23", meta)
    assert "Fazit" in text
    assert "zusammen" in text.lower() or "gesamt" in text.lower()


def test_save_base_feature_shap_report_writes_json_and_csv(tmp_path: Path):
    from lib.base_feature_shap_export import (
        build_base_feature_shap_payload,
        save_base_feature_shap_report,
    )

    shap_df = pd.DataFrame(
        {
            "feature_raw": ["a", "b"],
            "feature": ["A", "B"],
            "mean_abs_shap": [0.5, 0.1],
        }
    )
    payload = build_base_feature_shap_payload(
        shap_df=shap_df,
        feat_cols=["a", "b"],
        feat_display=["A", "B"],
        topk_names=["a"],
        topk_k=1,
        topk_mass_frac=0.9,
        shap_sample_rows=100,
        meta_shap_cum_frac=0.85,
        rsi_w=21,
        bb_w=15,
        sma_w=30,
        random_state=42,
    )
    jp = tmp_path / "base.json"
    cp = tmp_path / "base.csv"
    save_base_feature_shap_report(payload, json_paths=(jp,), csv_paths=(cp,))
    assert jp.is_file()
    assert cp.is_file()
    loaded = json.loads(jp.read_text(encoding="utf-8"))
    assert len(loaded["shap_mean_abs_sorted"]) == 2
    assert loaded["topk_names_raw"] == ["a"]


def test_meta_final_fit_no_eval_set_in_source():
    src = (ROOT / "lib" / "stock_rally_v10" / "training_phases" / "meta_learner.py").read_text(
        encoding="utf-8"
    )
    block = src.split("Finales Meta-Modell trainiert")[0].split("meta_clf.fit")[-1]
    assert "eval_set" not in block.split("Finales Meta")[0]


def test_fixed_y_label_mode_is_rally_plus_entry():
    from lib.stock_rally_v10 import config as cfg

    assert cfg.fixed_y_label_mode() == "rally_plus_entry"
