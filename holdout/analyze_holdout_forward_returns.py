"""
Backward-kompatibler Einstieg: erzeugt data/master_complete.csv und master_daily_update.csv (LLM-Spalten)
(siehe build_holdout_signals_master.py; inkl. Zusatzfilter, außer `--no-filters`).

python -m holdout.analyze_holdout_forward_returns
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from holdout.build_holdout_signals_master import main

if __name__ == "__main__":
    main()
