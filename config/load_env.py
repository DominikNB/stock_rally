"""
Lädt KEY=value-Zeilen aus lokalen Env-Dateien in os.environ (ohne python-dotenv).

Suchreihenfolge (erste gefundene Datei reicht; alle vorhandenen werden geladen, spätere überschreiben nicht):
  1) <project_root>/.env
  2) <project_root>/config/secrets.env

Bereits gesetzte Umgebungsvariablen werden nicht überschrieben (setdefault).
"""
from __future__ import annotations

import os
from pathlib import Path


def _parse_and_apply(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, val)


def load_project_env(project_root: Path | None = None) -> None:
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    for name in (".env", "config/secrets.env"):
        p = root / name
        if p.is_file():
            _parse_and_apply(p)
