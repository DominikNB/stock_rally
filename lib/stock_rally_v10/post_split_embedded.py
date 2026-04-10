"""
Training, Meta-Modelle, Kalibrierung, Plots und HTML-Export.

Logisch identisch zur Nach-Split-Pipeline (Phasen 12–17). Der Code liegt in
``post_split_phases/cell_12.py`` … ``cell_17.py``; jede Datei wird wie zuvor mit
gemeinsamem Namespace ausgeführt (``exec(..., vars(config), vars(config))``), sodass
``globals().get('SCORING_ONLY', …)`` unverändert funktioniert.

Änderungen an der Logik: die Dateien ``post_split_phases/cell_*.py`` direkt bearbeiten.
"""
from __future__ import annotations

from pathlib import Path

from lib.stock_rally_v10 import config as cfg

_PHASE_DIR = Path(__file__).resolve().parent / "post_split_phases"
_CELL_NUMS = tuple(range(12, 18))


def run_training_scoring_and_export() -> None:
    """Optuna, Base-Modelle, Meta, Threshold, Diagnose, Daily-Scoring/HTML."""
    ns = vars(cfg)
    for num in _CELL_NUMS:
        path = _PHASE_DIR / f"cell_{num}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing phase file {path}.")
        src = path.read_text(encoding="utf-8")
        exec(compile(src, str(path), "exec"), ns, ns)


