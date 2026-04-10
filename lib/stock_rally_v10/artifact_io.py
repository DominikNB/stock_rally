"""
Optionales erneutes Schreiben des Scoring-Artefakts.

Nach vollem Training erfolgt das Speichern in der Nach-Split-Pipeline automatisch;
dieses Modul ist nur nötig, wenn du ohne den üblichen Lauf erneut auf Platte schreiben willst.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lib.scoring_persist as scoring_persist

from lib.stock_rally_v10 import config as cfg


def save_scoring_artifact_bundle(path: str | Path = "models/scoring_artifacts.joblib") -> Path:
    """Schreibt ``scoring_persist.save_scoring_artifacts`` für den aktuellen ``config``-Namespace."""
    root = Path.cwd().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    scoring_persist.save_scoring_artifacts(vars(cfg), p)
    cfg._SCORING_ARTIFACT_SAVED_THIS_SESSION = True
    return p
