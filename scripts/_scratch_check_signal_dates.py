"""Scratch: signal dates per commit + raw scoring probe."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze_commit(commit: str) -> None:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:docs/signals.json"],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sj = json.loads(raw)
    dates = Counter(s["date"][:10] for s in sj["signals"])
    recent = sorted([d for d in dates if d >= "2026-06-01"], reverse=True)
    print(f"=== {commit} generated={sj.get('generated')} total={len(sj['signals'])} ===")
    for d in recent[:10]:
        print(f"  {d}: {dates[d]}")


def main() -> None:
    for c in ["d59eeca", "f0a6ab9", "d00ebe7"]:
        analyze_commit(c)


if __name__ == "__main__":
    main()
