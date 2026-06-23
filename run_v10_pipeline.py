"""V10-Pipeline starten.

Diese Datei im Editor öffnen und mit Run / Play ausführen (empfohlen: Workspace = Projektroot).
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from lib.stock_rally_v10.pipeline_runner import run_pipeline_default

if __name__ == "__main__":
    run_pipeline_default()
