"""
Orchestrierung der V10-Pipeline (reine Python-Module).

1. ``bind_step_functions()`` — registriert Hilfsfunktionen auf ``config`` (ein gemeinsamer
   Namespace für Pipeline-Schritte und eingebettete Phasen).
2. ``run_data_download_and_split()`` — Daten & Split (``data_and_split``).
3. ``run_training_scoring_and_export()`` — Training & Export (``training_phases`` / ``post_split_embedded``).

Direkt ausführbar (Play-Button): Projektroot wird automatisch auf ``sys.path`` gelegt.
Alternativ: ``python -m lib.stock_rally_v10.pipeline_runner`` aus dem Projektroot.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Direktlauf: sys.path zeigt sonst nur auf dieses Verzeichnis — Paket ``lib`` liegt im Projektroot.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import run_data_download_and_split
from lib.stock_rally_v10.post_split_embedded import run_training_scoring_and_export


def bind_step_functions() -> None:
    """Bootstrap-Imports anwenden und Schrittfunktionen auf ``cfg`` legen."""
    from lib.stock_rally_v10 import bootstrap  # noqa: F401 — Seiteneffekt: cfg.np, cfg.pd, …

    from lib.stock_rally_v10.data import load_stock_data
    from lib.stock_rally_v10.features import assemble_features
    from lib.stock_rally_v10.helpers import _strip_tz, make_focal_objective, make_focal_objective_lgb
    from lib.stock_rally_v10.holdout_plot import (
        _rows_for_signal_calendar_day,
        apply_signal_filters,
        plot_holdout_results,
        summarize_filtered_signals_vs_target,
    )
    from lib.stock_rally_v10.indicators import add_technical_indicators
    from lib.stock_rally_v10.news import fetch_news_sentiment
    from lib.stock_rally_v10.optuna_train import (
        _OPT_MAX_CONSEC_FP,
        _apply_filters_cv,
        _peak_rsi_mask_1d,
        _rsi_from_close_1d,
        optimize_xgb,
    )
    from lib.stock_rally_v10.target import create_target, rebuild_target_for_train

    g = vars(cfg)
    g["load_stock_data"] = load_stock_data
    g["create_target"] = create_target
    g["rebuild_target_for_train"] = rebuild_target_for_train
    g["add_technical_indicators"] = add_technical_indicators
    g["fetch_news_sentiment"] = fetch_news_sentiment
    g["assemble_features"] = assemble_features
    g["optimize_xgb"] = optimize_xgb
    g["plot_holdout_results"] = plot_holdout_results
    g["apply_signal_filters"] = apply_signal_filters
    g["_rows_for_signal_calendar_day"] = _rows_for_signal_calendar_day
    g["summarize_filtered_signals_vs_target"] = summarize_filtered_signals_vs_target
    g["_apply_filters_cv"] = _apply_filters_cv
    g["_OPT_MAX_CONSEC_FP"] = _OPT_MAX_CONSEC_FP
    g["_rsi_from_close_1d"] = _rsi_from_close_1d
    g["_peak_rsi_mask_1d"] = _peak_rsi_mask_1d
    g["_strip_tz"] = _strip_tz
    g["make_focal_objective"] = make_focal_objective
    g["make_focal_objective_lgb"] = make_focal_objective_lgb


def _log_loaded_config_snapshot() -> None:
    """Pfad + mtime der geladenen ``config`` — prüft, ob der Lauf wirklich die bearbeitete Datei nutzt."""
    p = Path(cfg.__file__).resolve()
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    _opt_y = cfg.opt_optimize_y_targets()
    if _opt_y:
        _rw = getattr(cfg, "RETURN_WINDOW", None)
        _rt = getattr(cfg, "RALLY_THRESHOLD", None)
        _ld = getattr(cfg, "LEAD_DAYS", None)
        _ed = getattr(cfg, "ENTRY_DAYS", None)
        _mt = getattr(cfg, "MIN_RALLY_TAIL_DAYS", None)
        _y_line = (
            f"Y: parametric (labels per Optuna trial); cfg module defaults "
            f"rw={_rw} rt={_rt!r} lead={_ld} entry={_ed} min_rally_tail={_mt}"
        )
    else:
        _rt = getattr(cfg, "FIXED_Y_RALLY_THRESHOLD", None)
        _y_line = (
            f"Y: fixed band [{getattr(cfg, 'FIXED_Y_WINDOW_MIN', '?')},"
            f"{getattr(cfg, 'FIXED_Y_WINDOW_MAX', '?')}] rt={_rt!r} "
            f"split={getattr(cfg, 'FIXED_Y_SEGMENT_SPLIT', None)} "
            f"lead={getattr(cfg, 'FIXED_Y_LEAD_DAYS', None)} "
            f"entry={getattr(cfg, 'FIXED_Y_ENTRY_DAYS', None)} "
            f"tail_excl={getattr(cfg, 'FIXED_Y_TAIL_EXCLUDE_DAYS', None)}"
        )
    print(
        f"Pipeline: config = {p}  (mtime {mtime})\n"
        f"  OPT_OPTIMIZE_Y_TARGETS={_opt_y}  SCORING_ONLY={getattr(cfg, 'SCORING_ONLY', None)}  "
        f"UNIVERSE_FRACTION={getattr(cfg, 'UNIVERSE_FRACTION', None)}\n"
        f"  {_y_line}",
        flush=True,
    )


def run_pipeline_default() -> None:
    """Voller Lauf: Daten/Split, Training, Scoring und HTML-Export."""
    bind_step_functions()
    cfg.log_pipeline_mode_banner()
    _log_loaded_config_snapshot()
    run_data_download_and_split()
    run_training_scoring_and_export()


if __name__ == "__main__":
    run_pipeline_default()
