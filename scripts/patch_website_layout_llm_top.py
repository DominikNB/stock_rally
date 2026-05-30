"""Einmal-Patch: KI-Analyse nach oben, Sidebar entfernen, sig-date/vix-Markup reparieren."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.html"


def _fix_sig_cards(html: str) -> str:
    """vix-meter aus sig-date herausziehen (Rebuild-Script-Bug; Ampel steht auf einer Zeile)."""
    pattern = (
        r'(<span class="sig-date-pre">Daten bis</span>)\s*'
        r'(<div class="vix-meter"[^>]*>[^\n]*</div>)\s*'
        r'(\d{4}-\d{2}-\d{2})\s*</span>'
    )

    def _repl(m: re.Match) -> str:
        return f"{m.group(1)} {m.group(3)}</span>\n          {m.group(2)}"

    html, n = re.subn(pattern, _repl, html)
    if n:
        print(f"  sig-date/vix repariert: {n} Karten", flush=True)
    return html


def _move_llm_top(html: str) -> str:
    m_aside = re.search(
        r'<aside class="llm-sidebar"[^>]*>\s*<details[^>]*>.*?</details>\s*</aside>',
        html,
        flags=re.DOTALL,
    )
    if not m_aside:
        print("  Hinweis: keine llm-sidebar gefunden", flush=True)
        return html

    m_llm = re.search(
        r'<div class="section analysis-llm">.*?</div>\s*</div>\s*</details>',
        m_aside.group(0),
        flags=re.DOTALL,
    )
    if not m_llm:
        print("  Hinweis: analysis-llm Block in Sidebar nicht gefunden", flush=True)
        return html

    llm_inner = m_llm.group(0)
    llm_inner = re.sub(r"</div>\s*</details>\s*$", "", llm_inner).strip()
    llm_inner = re.sub(
        r"<h2>KI-Analyse \(automatisch\)</h2>",
        "",
        llm_inner,
        count=1,
    )
    llm_inner = re.sub(
        r'<p class="prompt-lead">Generiert per Gemini[^<]*</p>',
        "",
        llm_inner,
        count=1,
    )

    top = (
        '<div class="section analysis-llm-section">\n'
        "        <h2>KI-Analyse — aktuelle Signale</h2>\n"
        '        <p class="section-lead">Gemini-Einordnung der neuesten Meta-Hits '
        "(File-Upload + Google Search). Keine Anlageberatung.</p>\n"
        f"        {llm_inner.strip()}\n"
        "      </div>\n\n      "
    )

    html = html[: m_aside.start()] + html[m_aside.end() :]
    html = re.sub(r"(<main class=\"main-col\">\s*)", r"\1" + top, html, count=1)
    print("  KI-Analyse nach oben verschoben", flush=True)
    return html


def main() -> None:
    if not INDEX.is_file():
        raise SystemExit(f"Fehlt: {INDEX}")
    html = INDEX.read_text(encoding="utf-8")
    html = _fix_sig_cards(html)
    html = _move_llm_top(html)
    INDEX.write_text(html, encoding="utf-8")
    print(f"geschrieben: {INDEX}")


if __name__ == "__main__":
    main()
