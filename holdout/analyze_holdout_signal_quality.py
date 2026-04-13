"""
Erzeugt data/master_complete.csv und data/master_daily_update.csv (LLM-Spalten);
Meta, Forward-Renditen, train_target/rally,
plus Zusatzfilter aus signal_extra_filters — wie in build_holdout_signals_master).

Früher: separate holdout_signals_eval_*.csv — alles ist jetzt in der Master-Datei.

python -m holdout.analyze_holdout_signal_quality
python -m holdout.analyze_holdout_signal_quality --no-filters   # nur wenn als Arg durchgereicht
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
