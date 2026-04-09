"""
Backward-kompatibler Einstieg: erzeugt data/meta_holdout_signals.csv
(siehe build_holdout_signals_master.py; inkl. Zusatzfilter, außer `--no-filters`).

python analyze_holdout_forward_returns.py
"""
from __future__ import annotations

from build_holdout_signals_master import main

if __name__ == "__main__":
    main()
