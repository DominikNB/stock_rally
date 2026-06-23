"""
Website-HTML neu bauen aus docs/signals.json + vorhandenen docs/charts/ — ohne Daten-Pipeline.

Nutzt PHASE17_WEBSITE_SIGNALS_OVERRIDE und verlinkt bestehende PNGs auf der Platte.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    signals_path = _ROOT / "docs" / "signals.json"
    data = json.loads(signals_path.read_text(encoding="utf-8"))
    signals = list(data.get("signals") or [])
    if not signals:
        raise SystemExit(f"Keine Signale in {signals_path}")

    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.pipeline_runner import bind_step_functions
    from lib.stock_rally_v10.training_phases.daily_scoring_html import run_phase_daily_scoring_html

    bind_step_functions()
    cfg.SCORING_ONLY = True
    cfg.RETRAIN_META_ONLY = False
    cfg.PHASE17_WEBSITE_SIGNALS_OVERRIDE = signals
    cfg.df_features = pd.DataFrame(
        {"ticker": ["_stub_"], "Date": [pd.Timestamp(cfg.END_DATE)]}
    )
    cfg.log_pipeline_mode_banner()
    cfg.load_scoring_artifacts()
    print(f"HTML-Regen: {len(signals)} Signale aus {signals_path.name}", flush=True)
    run_phase_daily_scoring_html(cfg)


if __name__ == "__main__":
    main()
