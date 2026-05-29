"""
VIX-Ampel v2 in docs/signals.json und docs/index.html nachziehen.

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
    vix_ampel_legend_html,
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
    if ".vix-ampel--orange" in html:
        return html
    return html.replace("        .sig-date-pre{", css + "\n        .sig-date-pre{", 1)


def _patch_index_html(html: str, lookup: dict[tuple[str, str], dict]) -> str:
    html = _ensure_css(html)

    # Kaputte Reste früherer Läufe
    html = re.sub(
        r'(<span class="sig-date"[^>]*>.*?</span>)\s*[^<]*(?:VIX|Regime|Ruhig|Erhöht|Stress)[^<]*</span>\s*',
        r"\1\n          ",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'\s*<span class="vix-ampel[^"]*"[^>]*>.*?</span>\s*',
        "\n          ",
        html,
        flags=re.DOTALL,
    )

    pat = re.compile(
        r'(<span class="sig-ticker">([^<]+)</span>.*?'
        r'<span class="sig-date-pre">Daten bis</span>\s*(\d{4}-\d{2}-\d{2})</span>)\s*'
        r'(<div class="score-bar-bg">)',
        re.DOTALL,
    )

    def _card_repl(m: re.Match) -> str:
        extra = lookup.get((m.group(2).strip(), m.group(3).strip()))
        if not extra:
            return m.group(0)
        return f"{m.group(1)}\n          {vix_ampel_html_span(extra)}\n          {m.group(4)}"

    html, n = pat.subn(_card_repl, html)
    print(f"  index.html: {n} Karten mit Ampel v2")

    leg = vix_ampel_legend_html()
    if "vix-ampel-legend" not in html:
        html = re.sub(
            r"(<h2[^>]*>Letzte 30 Tage[^<]*</h2>)",
            r"\1\n        " + leg,
            html,
            count=1,
        )
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

    print("\nFertig.")


if __name__ == "__main__":
    main()
