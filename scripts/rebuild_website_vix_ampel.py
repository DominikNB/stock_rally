"""
VIX-Ampel + Rot-Kontext-Chips in docs/signals.json und docs/index.html nachziehen.

  python scripts/rebuild_website_vix_ampel.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.vix_regime_ampel import ampel_fields_from_vix, vix_ampel_html_span, vix_regime_full_css_block
from lib.vix_red_context_chips import (
    attach_red_context_to_signal,
    red_context_chips_html,
    vix_regime_guide_panel_html,
)

MASTER = ROOT / "data" / "master_complete.csv"
SIGNALS_JSON = ROOT / "docs" / "signals.json"
INDEX_HTML = ROOT / "docs" / "index.html"

_CHIP_COLS = (
    "regime_vix_level",
    "regime_vix_z_20d",
    "vix3m_vix_ratio",
    "sector_hhi_same_day",
)


def _calibrate_sector_hhi_max(mc: pd.DataFrame) -> float:
    vix = pd.to_numeric(mc.get("regime_vix_level"), errors="coerce")
    hhi = pd.to_numeric(mc.get("sector_hhi_same_day"), errors="coerce")
    sub = hhi[(vix < 20) & hhi.notna()]
    if len(sub) < 30:
        return 0.35
    return float(sub.median())


def _vix3m_ratio_by_date(dates: pd.Series) -> dict[str, float]:
    if dates.empty:
        return {}
    start = (pd.to_datetime(dates.min()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(dates.max()) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(["^VIX", "^VIX3M"], start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return {}
    vix = raw["Close"]["^VIX"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    v3 = raw["Close"]["^VIX3M"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    ratio = (v3 / vix).replace([np.inf, -np.inf], np.nan)
    ratio.index = pd.to_datetime(ratio.index).normalize()
    return {d.strftime("%Y-%m-%d"): float(ratio.loc[d]) for d in ratio.index if pd.notna(ratio.loc[d])}


def _lookup_from_master() -> dict[tuple[str, str], dict]:
    mc = pd.read_csv(MASTER)
    mc["Date"] = pd.to_datetime(mc["Date"], errors="coerce").dt.normalize()
    mc["ticker"] = mc["ticker"].astype(str).str.strip()

    hhi_max = _calibrate_sector_hhi_max(mc)
    try:
        from lib.stock_rally_v10 import config_settings as cs

        cs.VIX_RED_CHIP_SECTOR_HHI_MAX = hhi_max
    except Exception:
        pass
    print(f"  Sektor-HHI Schwelle (Median rot): {hhi_max:.4f}")

    if "vix3m_vix_ratio" not in mc.columns or mc["vix3m_vix_ratio"].notna().sum() < 10:
        by_date = _vix3m_ratio_by_date(mc["Date"])
        mc["vix3m_vix_ratio"] = mc["Date"].dt.strftime("%Y-%m-%d").map(by_date)
        print(f"  VIX3M/VIX nachgeladen: {mc['vix3m_vix_ratio'].notna().sum()} Zeilen")

    out: dict[tuple[str, str], dict] = {}
    for _, r in mc.iterrows():
        key = (str(r["ticker"]), pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"))
        row: dict = {}
        for c in _CHIP_COLS:
            if c in r.index:
                try:
                    v = float(r[c])
                    row[c] = None if v != v else v
                except (TypeError, ValueError):
                    row[c] = None
        vix = row.get("regime_vix_level")
        row.update(ampel_fields_from_vix(vix))
        attach_red_context_to_signal(row)
        out[key] = row
    return out


def _signal_key(s: dict) -> tuple[str, str]:
    return (str(s.get("ticker", "")).strip(), str(s.get("date", ""))[:10])


def _enrich_signals(
    signals: list[dict],
    master_lookup: dict[tuple[str, str], dict],
) -> int:
    n_master = 0
    for s in signals:
        key = _signal_key(s)
        extra = master_lookup.get(key)
        if extra:
            s.update(extra)
            n_master += 1
        attach_red_context_to_signal(s)
    return n_master


def _lookup_from_signals(signals: list[dict]) -> dict[tuple[str, str], dict]:
    """Lookup für index.html aus angereicherten Signal-Dicts."""
    out: dict[tuple[str, str], dict] = {}
    for s in signals:
        key = _signal_key(s)
        row = {c: s.get(c) for c in _CHIP_COLS if c in s}
        row["vix_regime_ampel"] = s.get("vix_regime_ampel")
        row["red_context_html"] = s.get("red_context_html")
        out[key] = row
    return out


def _fix_doubled_css_braces(html: str) -> str:
    """Repariert f-String-{{…}} in <style> (Browser ignoriert sonst Ampel/Chips/Layout)."""

    def _norm_style(m: re.Match) -> str:
        block = m.group(0)
        if "{{" not in block:
            return block
        return block.replace("{{", "{").replace("}}", "}")

    return re.sub(r"<style>.*?</style>", _norm_style, html, count=1, flags=re.DOTALL)


def _ensure_css(html: str) -> str:
    html = _fix_doubled_css_braces(html)
    css = vix_regime_full_css_block().strip()
    html = re.sub(
        r"        \.vix-panel\{[^}]*\}.*?        \.vix-guide-list li\{[^}]*\}\n",
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"        \.vix-panel\{\{[^}]*\}\}.*?        \.vix-guide-list li\{\{[^}]*\}\}\n",
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"        \.red-ctx-chips\{[^}]*\}.*?        \.red-ctx-chip--na\{[^}]*\}\n",
        "",
        html,
        flags=re.DOTALL,
    )
    if ".red-ctx-chip--good{" not in html:
        return html.replace("        .sig-date-pre{", css + "\n        .sig-date-pre{", 1)
    return html


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
    html = re.sub(r'\s*<div class="red-ctx-chips"[^>]*>.*?</div>\s*', "\n        ", html, flags=re.DOTALL)
    return html


def _inject_vix_panel(html: str) -> str:
    panel = vix_regime_guide_panel_html()
    block = (
        f'<div class="section vix-regime-section"><h2>VIX-Regime (Ampel)</h2>{panel}</div>\n\n      '
    )
    if "vix-regime-section" in html:
        html = re.sub(
            r'<div class="section vix-regime-section">.*?</div>\s*\n\s*',
            block,
            html,
            count=1,
            flags=re.DOTALL,
        )
        return html
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

    pat2 = re.compile(
        r'(<div class="sig-head">.*?</div>)\s*'
        r'(?=<p class="sig-gics"|<p class="yf-hint"|<a class="sig-chart|<img |<p class="sig-chart-note")',
        re.DOTALL,
    )

    def _head_repl(m: re.Match) -> str:
        head = m.group(1)
        tm = re.search(r'<span class="sig-ticker">([^<]+)', head)
        dm = re.search(
            r'<span class="sig-date-pre">Daten bis</span>\s*(\d{4}-\d{2}-\d{2})',
            head,
        )
        if not tm or not dm:
            return m.group(0)
        ticker = re.sub(r"<[^>]+>", "", tm.group(1)).strip()
        extra = lookup.get((ticker, dm.group(1)))
        if not extra:
            return m.group(0)
        amp = vix_ampel_html_span(extra)
        # Kaputtes Markup reparieren: vix-meter darf nicht innerhalb von sig-date stehen.
        head2 = re.sub(
            r'<div class="vix-meter">.*?</div>\s*</div>\s*',
            "",
            head,
            count=1,
            flags=re.DOTALL,
        )
        if amp not in head2:
            # Ampel nach geschlossenem sig-date (nie innerhalb von sig-date).
            head2, _n_amp = re.subn(
                r'(<span class="sig-date-pre">Daten bis</span>\s*\d{4}-\d{2}-\d{2}\s*</span>)',
                rf"\1\n          {amp}",
                head2,
                count=1,
            )
            if _n_amp == 0:
                head2 = re.sub(
                    r"(</span>\s*)(<div class=\"score-bar-bg\">)",
                    rf"\1\n          {amp}\n          \2",
                    head2,
                    count=1,
                    flags=re.DOTALL,
                )
        chips = extra.get("red_context_html") or ""
        if chips and str(extra.get("vix_regime_ampel", "")).lower() == "red":
            return f"{head2}\n        {chips}\n        "
        return f"{head2}\n        "

    html, n = pat2.subn(_head_repl, html)
    print(f"  index.html: {n} Karten (Ampel + Rot-Chips)")

    html = _inject_vix_panel(html)
    html = _uncollapse_oos_section(html)
    return html


def main() -> None:
    if not MASTER.is_file():
        raise SystemExit(f"Fehlt: {MASTER}")

    master_lookup = _lookup_from_master()
    print(f"Master-Lookup: {len(master_lookup):,} Zeilen")

    if SIGNALS_JSON.is_file():
        payload = json.loads(SIGNALS_JSON.read_text(encoding="utf-8"))
        html_lookup: dict[tuple[str, str], dict] = {}
        for key in ("signals", "signals_holdout_final"):
            if key in payload and isinstance(payload[key], list):
                nm = _enrich_signals(payload[key], master_lookup)
                print(f"  signals.json [{key}]: master={nm} von {len(payload[key])}")
                html_lookup.update(_lookup_from_signals(payload[key]))
        payload["vix_ampel_version"] = 4
        payload["vix_ampel_stages"] = 3
        payload["red_context_chips"] = 3
        SIGNALS_JSON.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        html_lookup = master_lookup

    if INDEX_HTML.is_file():
        html = INDEX_HTML.read_text(encoding="utf-8")
        html = _patch_index_html(html, html_lookup)
        INDEX_HTML.write_text(html, encoding="utf-8")
        print(f"  geschrieben: {INDEX_HTML}")

    print("\nFertig.")


if __name__ == "__main__":
    main()
