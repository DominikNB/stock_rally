"""
VIX-Ampel v2 in docs/signals.json und docs/index.html nachziehen (visuelle Skala + Karten-Meter).

  python scripts/rebuild_website_vix_ampel.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.vix_regime_ampel import (
    ampel_fields_from_vix,
    vix_ampel_css_block,
    vix_ampel_html_span,
    vix_ampel_panel_html,
)

MASTER = ROOT / "data" / "master_complete.csv"
SIGNALS_JSON = ROOT / "docs" / "signals.json"
INDEX_HTML = ROOT / "docs" / "index.html"


def _vix_lookup_from_master() -> dict[tuple[str, str], dict]:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mc["ticker"] = mc["ticker"].astype(str).str.strip()
    if "regime_vix_level" not in mc.columns:
        raise SystemExit(f"{MASTER} hat keine Spalte regime_vix_level.")
    out: dict[tuple[str, str], dict] = {}
    for _, r in mc.iterrows():
        key = (str(r["ticker"]), str(r["Date"])[:10])
        try:
            v = float(r["regime_vix_level"])
            if v != v:
                v = None
        except (TypeError, ValueError):
            v = None
        out[key] = ampel_fields_from_vix(v)
    return out


def _enrich_signals(signals: list[dict], lookup: dict[tuple[str, str], dict]) -> int:
    n = 0
    for s in signals:
        key = (str(s.get("ticker", "")), str(s.get("date", ""))[:10])
        extra = lookup.get(key)
        if not extra:
            continue
        s.update(extra)
        n += 1
    return n


def _ensure_css(html: str) -> str:
    css = vix_ampel_css_block().strip()
    if ".vix-scale-track" in html:
        return html
    html = re.sub(
        r"        \.vix-ampel\{[^}]*\}.*?        \.sig-card--recent\{[^}]*\}\n",
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"        \.vix-ampel\{[^}]*\}.*?        \.vix-ampel-legend\{[^}]*\}\n",
        "",
        html,
        flags=re.DOTALL,
    )
    return html.replace("        .sig-date-pre{", css + "\n        .sig-date-pre{", 1)


def _strip_old_ampel_blocks(html: str) -> str:
    html = re.sub(
        r'(<span class="sig-date"[^>]*>.*?</span>)\s*[^<]*(?:VIX|Regime|Ruhig|Erhöht|Stress)[^<]*</span>\s*',
        r"\1\n          ",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<div class="vix-meter"[^>]*>.*?</div>\s*</div>\s*',
        "\n          ",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<span class="vix-ampel[^"]*"[^>]*>.*?</span>\s*',
        "\n          ",
        html,
        flags=re.DOTALL,
    )
    return html


def _inject_vix_panel(html: str) -> str:
    panel = vix_ampel_panel_html()
    block = (
        f'<div class="section vix-regime-section"><h2>VIX-Regime (Ampel)</h2>{panel}</div>\n\n      '
    )
    if "vix-regime-section" in html:
        return html
    html = re.sub(r'<p class="vix-ampel-legend">.*?</p>\s*', "", html, count=1, flags=re.DOTALL)
    html = re.sub(
        r"(<div class=\"section\">\s*<h2[^>]*>Letzte 30 Tage)",
        block + r"\1",
        html,
        count=1,
    )
    if "vix-regime-section" not in html:
        html = re.sub(
            r"(<div class=\"section signals-all-section\">)",
            block + r"\1",
            html,
            count=1,
        )
    return html


def _uncollapse_oos_section(html: str) -> str:
    """OOS-Archiv nicht mehr in zugeklapptem details."""
    html = re.sub(
        r"<div class=\"section\">\s*<details>\s*"
        r"<summary>OOS-Signale \(nicht in Classifier-Training\)[^<]*</summary>\s*",
        '<div class="section oos-archive-section"><h2>Frühere OOS-Signale</h2>'
        '<p class="section-lead">Älter als 30 Tage — alle mit Chart (nach unten scrollen).</p>\n        ',
        html,
        count=1,
    )
    html = re.sub(
        r"</div>\s*</details>\s*</div>\s*\n\s*<aside class=\"llm-sidebar\"",
        "</div>\n      </div>\n\n    <aside class=\"llm-sidebar\"",
        html,
        count=1,
    )
    return html


def _patch_index_html(html: str, lookup: dict[tuple[str, str], dict]) -> str:
    html = _ensure_css(html)
    html = _strip_old_ampel_blocks(html)

    pat = re.compile(
        r'(<span class="sig-ticker">([^<]+)</span>.*?'
        r'<span class="sig-date-pre">Daten bis</span>\s*(\d{4}-\d{2}-\d{2})</span>)\s*'
        r'(<div class="score-bar-bg">)',
        re.DOTALL,
    )

    def _card_repl(m: re.Match) -> str:
        ticker = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        extra = lookup.get((ticker, m.group(3).strip()))
        if not extra:
            return m.group(0)
        return f"{m.group(1)}\n          {vix_ampel_html_span(extra)}\n          {m.group(4)}"

    html, n = pat.subn(_card_repl, html)
    print(f"  index.html: {n} Karten mit VIX-Meter")

    html = _inject_vix_panel(html)
    html = _uncollapse_oos_section(html)
    return html


def main() -> None:
    if not MASTER.is_file():
        raise SystemExit(f"Fehlt: {MASTER}")

    lookup = _vix_lookup_from_master()
    print(f"VIX-Lookup: {len(lookup):,} Zeilen")

    if SIGNALS_JSON.is_file():
        payload = json.loads(SIGNALS_JSON.read_text(encoding="utf-8"))
        for key in ("signals", "signals_holdout_final"):
            if key in payload and isinstance(payload[key], list):
                n = _enrich_signals(payload[key], lookup)
                print(f"  signals.json [{key}]: {n} Signale (Ampel v2)")
        payload["vix_ampel_version"] = 2
        SIGNALS_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if INDEX_HTML.is_file():
        html = INDEX_HTML.read_text(encoding="utf-8")
        html = _patch_index_html(html, lookup)
        INDEX_HTML.write_text(html, encoding="utf-8")
        print(f"  geschrieben: {INDEX_HTML}")

    print("\nFertig. Für ein einziges Scroll-Grid aller Charts: Phase-17-Lauf (neues HTML-Layout).")


if __name__ == "__main__":
    main()
