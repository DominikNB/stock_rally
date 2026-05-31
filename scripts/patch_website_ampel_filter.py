"""Ampel-Filter in bestehende docs/index.html einfügen (ohne Phase-17-Volllauf)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"
SIGNALS = ROOT / "docs" / "signals.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.website_ampel_filter import (  # noqa: E402
    ampel_counts,
    website_ampel_filter_css_block,
    website_ampel_filter_html,
    website_ampel_filter_js_block,
)


def _add_card_ampel_attrs(html: str) -> tuple[str, int]:
    n = 0

    def _repl(m: re.Match) -> str:
        nonlocal n
        tag = m.group(0)
        if "data-vix-ampel" in tag and "unknown" not in tag:
            return tag
        chunk = html[m.end() : m.end() + 4000]
        amp = "unknown"
        for c in ("red", "yellow", "green"):
            if (
                f"vix-ampel--{c}" in chunk
                or f"vix-light--{c} is-active" in chunk
                or f"vix-light vix-light--{c} is-active" in chunk
            ):
                amp = c
                break
        n += 1
        if tag.endswith(">"):
            return tag[:-1] + f' data-vix-ampel="{amp}">'
        return tag

    html = re.sub(r'<div class="sig-card[^"]*"[^>]*>', _repl, html)
    return html, n


def main() -> None:
    if not INDEX.is_file():
        sys.exit(f"Fehlt: {INDEX}")
    if not SIGNALS.is_file():
        sys.exit(f"Fehlt: {SIGNALS}")

    data = json.loads(SIGNALS.read_text(encoding="utf-8"))
    signals = data.get("signals", data) if isinstance(data, dict) else data
    counts = ampel_counts(signals)
    html = INDEX.read_text(encoding="utf-8")

    if "id=\"ampel-filter\"" not in html:
        css = website_ampel_filter_css_block()
        if ".ampel-filter-bar" not in html:
            html = html.replace("</style>", css + "\n      </style>", 1)

        block = website_ampel_filter_html(counts)
        marker = '<div class="page-wrap">'
        if marker not in html:
            sys.exit("page-wrap nicht gefunden")
        html = html.replace(marker, marker + "\n    " + block.strip() + "\n    ", 1)
    else:
        print("Hinweis: Ampel-Filter-HTML bereits vorhanden.", flush=True)

    js = website_ampel_filter_js_block().strip()
    if 'id="ampel-filter-js"' in html:
        html, n_js = re.subn(
            r'<script id="ampel-filter-js">.*?</script>',
            js,
            html,
            count=1,
            flags=re.DOTALL,
        )
        if n_js:
            print("JavaScript für Ampel-Filter aktualisiert.", flush=True)
    else:
        html = html.replace("</body>", js + "\n    </body>", 1)
        print("JavaScript für Ampel-Filter eingefügt.", flush=True)

    html, n_cards = _add_card_ampel_attrs(html)
    INDEX.write_text(html, encoding="utf-8")
    print(
        f"OK: {INDEX} — Filter Rot/Gelb/Grün/Alle "
        f"({counts['red']} / {counts['yellow']} / {counts['green']} / {counts['all']}); "
        f"{n_cards} Karten mit data-vix-ampel."
    )


if __name__ == "__main__":
    main()
