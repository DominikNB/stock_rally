"""
Nur Website-Export (Phase 17): Artefakt laden → Daten/Split/Features → HTML/JSON (+ Git wie in Phase 17).

Voraussetzung: Projektroot als CWD, ``models/scoring_artifacts.joblib`` vorhanden.
Kein Base-/Meta-Training (Phasen 12–16).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.data_and_split import run_data_download_and_split
from lib.stock_rally_v10.pipeline_runner import bind_step_functions
from lib.stock_rally_v10.training_phases.daily_scoring_html import run_phase_daily_scoring_html


def main() -> None:
    bind_step_functions()
    cfg.SCORING_ONLY = True
    cfg.log_pipeline_mode_banner()
    cfg.load_scoring_artifacts()
    run_data_download_and_split()
    run_phase_daily_scoring_html(cfg)


if __name__ == "__main__":
    main()
