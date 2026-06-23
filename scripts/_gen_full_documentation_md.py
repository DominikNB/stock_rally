"""Build docs/STOCK_RALLY_DOKUMENTATION.md — single file with all project docs."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "STOCK_RALLY_DOKUMENTATION.md"
PIPELINE = ROOT / "docs" / "PIPELINE_OVERVIEW.md"
SYSTEM_REF = ROOT / "docs" / "SYSTEM_REFERENZ.md"
V11 = ROOT / "docs" / "V11_ROADMAP.md"
GEN_PIPELINE = ROOT / "scripts" / "_gen_pipeline_overview_md.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)

_H2 = re.compile(r"^(#{2,3})\s+(\d+[a-z]?\.)\s+", re.MULTILINE)
_H2_V11 = re.compile(r"^(#{2,3})\s+(\d+\.)\s+", re.MULTILINE)


def _fix_md_anchors(text: str, letter: str) -> str:
    """TOC links (#1-foo) → (#i1-foo) to match prefixed headings."""
    return re.sub(r"\]\(#(\d+)", rf"](#{letter.lower()}\1", text)


def _prefix_headings(text: str, prefix: str, *, skip_first_title: bool = True) -> str:
    lines = text.splitlines()
    out: list[str] = []
    skipped_title = False
    for line in lines:
        if skip_first_title and not skipped_title and line.startswith("# ") and not line.startswith("## "):
            skipped_title = True
            continue
        m = re.match(r"^(#{2,3})\s+(\d+[a-z]?\.)\s+(.*)$", line)
        if m:
            hashes, num, rest = m.groups()
            out.append(f"{hashes} {prefix}{num} {rest}")
        else:
            out.append(line)
    return "\n".join(out).strip()


def _strip_cross_links(text: str) -> str:
    """Replace links to split docs with in-doc anchors where possible."""
    text = text.replace("](PIPELINE_OVERVIEW.md", "](STOCK_RALLY_DOKUMENTATION.md")
    text = text.replace("](SYSTEM_REFERENZ.md", "](STOCK_RALLY_DOKUMENTATION.md")
    text = text.replace("](V11_ROADMAP.md", "](STOCK_RALLY_DOKUMENTATION.md")
    text = re.sub(
        r"\[`PIPELINE_OVERVIEW\.md`\]\([^)]+\)",
        "`STOCK_RALLY_DOKUMENTATION.md` (Teil I)",
        text,
    )
    text = re.sub(
        r"\[`V11_ROADMAP\.md`\]\([^)]+\)",
        "`STOCK_RALLY_DOKUMENTATION.md` (Teil III)",
        text,
    )
    return text


def _master_header() -> str:
    return """# stock_rally — Gesamtdokumentation

**Stand:** Juni 2026 · **Eine Datei** mit Pipeline-Übersicht, Systemreferenz, V11-Roadmap, Verbesserungsvorschlägen und SHAP-Anhang.

> Neu generieren: `python scripts/_gen_full_documentation_md.py`

---

## Gesamt-Inhaltsverzeichnis

| Teil | Inhalt | Abschnitte |
|------|--------|------------|
| **I** | Pipeline V10 (Phasen, Config, Signale, FAQ, **Verbesserungsvorschläge**, **OOS-Performance**, SHAP) | I.1 – I.18 |
| **II** | Systemreferenz (Architektur, Module, Rekonstruktion) | II.1 – II.16 |
| **III** | V11-Roadmap (Kritik, Arbeitspakete) | III.1 – III.7 |

Quellen (für Pflege): `docs/_pipeline_overview_static.md` (Teil I, §I.1–I.15), `docs/SYSTEM_REFERENZ.md` (Teil II), `docs/V11_ROADMAP.md` (Teil III). §I.16.3 Performance + §I.17 SHAP werden beim Build ergänzt.

---
"""


def build() -> str:
    for path in (SYSTEM_REF, V11):
        if not path.is_file():
            raise SystemExit(f"Missing: {path}")

    subprocess.run([str(PY), str(GEN_PIPELINE)], check=True, cwd=ROOT)
    if not PIPELINE.is_file():
        raise SystemExit(f"Pipeline doc not built: {PIPELINE}")

    pipeline_raw = PIPELINE.read_text(encoding="utf-8")
    # Drop pipeline title block through first ---
    pipeline_body = pipeline_raw.split("---", 1)[-1].strip()
    pipeline_body = _prefix_headings(pipeline_body, "I.", skip_first_title=False)
    pipeline_body = _fix_md_anchors(pipeline_body, "i")
    pipeline_body = _strip_cross_links(pipeline_body)

    system_body = _prefix_headings(
        SYSTEM_REF.read_text(encoding="utf-8"), "II.", skip_first_title=True
    )
    system_body = _fix_md_anchors(system_body, "ii")
    system_body = _strip_cross_links(system_body)

    v11_body = _prefix_headings(V11.read_text(encoding="utf-8"), "III.", skip_first_title=True)
    v11_body = _fix_md_anchors(v11_body, "iii")
    v11_body = _strip_cross_links(v11_body)

    parts = [
        _master_header(),
        "# Teil I — Pipeline-Übersicht (V10)\n",
        pipeline_body,
        "\n\n---\n\n# Teil II — Systemreferenz\n",
        system_body,
        "\n\n---\n\n# Teil III — V11-Roadmap\n",
        v11_body,
        "\n\n---\n\n*Generiert von `scripts/_gen_full_documentation_md.py` · "
        "SHAP-Daten aus letztem Phase-12/13-Lauf.*\n",
    ]
    return "\n".join(parts)


def main() -> None:
    doc = build()
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {doc.count(chr(10)) + 1:,} lines)")


if __name__ == "__main__":
    main()
