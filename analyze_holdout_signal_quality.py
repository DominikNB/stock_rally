"""
Erzeugt data/meta_holdout_signals.csv (Meta, Forward-Renditen, train_target/rally,
plus Zusatzfilter aus signal_extra_filters — wie in build_holdout_signals_master).

Früher: separate holdout_signals_eval_*.csv — alles ist jetzt in der Master-Datei.

python analyze_holdout_signal_quality.py
python analyze_holdout_signal_quality.py --no-filters   # nur wenn als Arg durchgereicht
"""
from __future__ import annotations

from build_holdout_signals_master import main

if __name__ == "__main__":
    main()
