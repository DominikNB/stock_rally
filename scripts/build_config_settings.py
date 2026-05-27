"""One-off: split tunable assignments into lib/stock_rally_v10/config_settings.py.

Run from repo root: python scripts/build_config_settings.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "lib" / "stock_rally_v10" / "config.py"
SETTINGS = ROOT / "lib" / "stock_rally_v10" / "config_settings.py"

# 1-based inclusive line ranges copied into config_settings (order preserved).
SETTINGS_RANGES: tuple[tuple[int, int], ...] = (
    (70, 103),  # Kalender, Scoring-Flags (bis _SCORING_ARTIFACT_SAVED…)
    (149, 174),  # Abschnitt 3 Kopf + N_WF… + Optuna-Toggles
    (182, 212),  # FIXED_Y-Doku + Konstanten
    (533, 853),  # EARLY_STOPPING … GKG_GCAM_USE_EXTRA_N (nur Zuweisungen)
)

HEADER = '''"""Tunable parameters — edit here.

Imported into ``lib.stock_rally_v10.config`` with ``from .config_settings import *`` so all
``cfg.*`` names and ``globals()`` in that module stay on one namespace.

``config.py`` keeps GCP bootstrap, ``save_scoring_artifacts`` / ``load_scoring_artifacts``,
target-rule helpers, ticker lists, and column builders.
"""
import datetime
import os
from pathlib import Path

'''


def main() -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    if "from .config_settings import" in raw:
        raise SystemExit(
            "config.py already imports config_settings — this script expects a monolithic "
            "backup (restore from git) before re-running."
        )
    lines = raw.splitlines(keepends=True)
    n = len(lines)

    chunks: list[str] = []
    for lo, hi in SETTINGS_RANGES:
        if lo < 1 or hi > n or lo > hi:
            raise SystemExit(f"Bad range {lo}-{hi} (file has {n} lines)")
        chunks.append("".join(lines[lo - 1 : hi]))

    SETTINGS.write_text(HEADER + "".join(chunks), encoding="utf-8")

    remove = set()
    for lo, hi in SETTINGS_RANGES:
        remove.update(range(lo, hi + 1))

    out_lines: list[str] = []
    insert_done = False
    for i, line in enumerate(lines, start=1):
        if i in remove:
            continue
        out_lines.append(line)
        if i == 69 and not insert_done:
            out_lines.append("\n")
            out_lines.append("from .config_settings import *\n")
            out_lines.append("\n")
            insert_done = True

    if not insert_done:
        raise SystemExit("Never inserted import (file shorter than 69 lines?)")

    CONFIG.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote {SETTINGS.relative_to(ROOT)}")
    print(f"Updated {CONFIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
