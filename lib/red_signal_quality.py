"""
Rot-Signal-Qualität (VIX < 20): nur OOS-validierte Komponenten.

Validierung: scripts/_scratch_validate_red_quality_tiers.py
  → data/_scratch_red_quality_validation.json

Stand 2026-06: OOS-bestätigt (META+THR + FINAL, gleiche Richtung, Δ ≥ 0.1pp):
  • liquidity_ok   — liquidity_tier == ok
  • gld_ret5_low   — gld_ret_5d unter Tages-Median (Risk-on vs. Gold-Rally)

Alpha/RS/Chips: IS oft positiv, FINAL nicht stabil → nur Anzeige der Rohwerte, kein Score.
"""
from __future__ import annotations

import html as html_mod
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_JSON = ROOT / "data" / "_scratch_red_quality_validation.json"

# Fallback falls JSON fehlt (Ergebnis Validierungslauf)
_DEFAULT_PASSED = ("liquidity_ok", "gld_ret5_low")


def _f(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def passed_component_ids() -> tuple[str, ...]:
    if VALIDATION_JSON.is_file():
        try:
            obj = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
            ids = tuple(obj.get("passed_ids") or ())
            if ids:
                return ids
        except Exception:
            pass
    return _DEFAULT_PASSED


def _liquidity_ok(row: Mapping[str, Any]) -> bool | None:
    tier = row.get("liquidity_tier")
    if tier is None or str(tier).strip() == "":
        return None
    return str(tier).strip().lower() == "ok"


def _gld_ret5_low(row: Mapping[str, Any], *, ref_median: float | None = None) -> bool | None:
    g = _f(row.get("gld_ret_5d"))
    if g is None:
        return None
    med = ref_median if ref_median is not None else _f(row.get("gld_ret5_median_same_day"))
    if med is None:
        med = _f(row.get("_gld_ret5_median_ref"))
    if med is None:
        return None
    return g < med


def evaluate_red_quality_components(row: Mapping[str, Any]) -> dict[str, bool | None]:
    """Einzelkomponenten (True = historisch günstiger in rot, OOS-validiert)."""
    out: dict[str, bool | None] = {}
    if "liquidity_ok" in passed_component_ids():
        out["liquidity_ok"] = _liquidity_ok(row)
    if "gld_ret5_low" in passed_component_ids():
        out["gld_ret5_low"] = _gld_ret5_low(row)
    return out


def red_quality_score(row: Mapping[str, Any]) -> tuple[int, int, int]:
    """
    (treffer, bekannt, max_theoretisch) über validierte Komponenten.
    max_theoretisch = len(passed_ids); bekannt = Komponenten mit Bewertung (nicht None).
    """
    comps = evaluate_red_quality_components(row)
    passed_ids = passed_component_ids()
    hits = 0
    known = 0
    for cid in passed_ids:
        v = comps.get(cid)
        if v is None:
            continue
        known += 1
        if v:
            hits += 1
    return hits, known, len(passed_ids)


def red_quality_tier(hits: int, known: int, max_pts: int) -> str:
    """hoch | mittel | niedrig | unbekannt"""
    if known <= 0:
        return "unbekannt"
    ratio = hits / known
    if ratio >= 1.0:
        return "hoch"
    if ratio >= 0.5:
        return "mittel"
    return "niedrig"


def _alpha_label(row: Mapping[str, Any]) -> str | None:
    """Informativ (nicht OOS-validiert als Qualitätsfilter)."""
    a = _f(row.get("alpha_sec_5d"))
    if a is None:
        a = _f(row.get("alpha_mkt_5d"))
    if a is None:
        return None
    return "alpha" if a > 0 else "beta"


def red_quality_fields_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    comps = evaluate_red_quality_components(row)
    hits, known, max_pts = red_quality_score(row)
    tier = red_quality_tier(hits, known, max_pts)
    alpha_l = _alpha_label(row)
    return {
        "red_quality_hits": hits,
        "red_quality_max": known if known else max_pts,
        "red_quality_tier": tier,
        "red_quality_components": comps,
        "red_quality_alpha_hint": alpha_l or "",
    }


def red_quality_badge_html(fields: Mapping[str, Any]) -> str:
    tier = str(fields.get("red_quality_tier") or "unbekannt").lower()
    hits = int(fields.get("red_quality_hits") or 0)
    mx = int(fields.get("red_quality_max") or 0)
    alpha = str(fields.get("red_quality_alpha_hint") or "")
    comps = fields.get("red_quality_components") or {}
    known = sum(1 for v in comps.values() if v is not None)
    if known <= 0 and not alpha:
        return ""
    tips = []
    if comps.get("liquidity_ok") is True:
        tips.append("Liquidität ok (OOS-validiert)")
    elif comps.get("liquidity_ok") is False:
        tips.append("Dünne Liquidität — Vorsicht")
    if comps.get("gld_ret5_low") is True:
        tips.append("Gold schwach 5d — kein Flucht-in-Gold (OOS-validiert)")
    elif comps.get("gld_ret5_low") is False:
        tips.append("Gold-Rally 5d — Risk-off-Kontext")
    if alpha == "alpha":
        tips.append("Alpha vs. Sektor/Markt (informativ, kein OOS-Filter)")
    elif alpha == "beta":
        tips.append("Eher Mitläufer vs. Markt/Sektor (informativ)")
    tip = html_mod.escape("; ".join(tips) if tips else "Rot-Qualität")
    label = {"hoch": "Qualität hoch", "mittel": "Qualität mittel", "niedrig": "Qualität niedrig"}.get(
        tier, "Qualität ?"
    )
    if mx:
        label = f"{label} ({hits}/{mx})"
    return (
        f'<span class="red-quality-badge red-quality-badge--{html_mod.escape(tier)}" '
        f'title="{tip}">{html_mod.escape(label)}</span>'
    )


def attach_red_quality_to_signal(signal: dict[str, Any]) -> dict[str, Any]:
    amp = str(signal.get("vix_regime_ampel") or "").strip().lower()
    if amp != "red":
        signal["red_quality_tier"] = ""
        signal["red_quality_html"] = ""
        signal["red_quality_hits"] = 0
        signal["red_quality_max"] = 0
        return signal
    fields = red_quality_fields_from_row(signal)
    signal.update(fields)
    signal["red_quality_html"] = red_quality_badge_html(fields)
    return signal


def attach_red_quality_llm_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Hängt OOS-validierte Rot-Qualitätsfelder an einen Signal-DataFrame."""
    import pandas as pd

    o = df.copy()
    rows: list[dict[str, Any]] = []
    for _, r in o.iterrows():
        amp = str(r.get("vix_regime_ampel") or "").strip().lower()
        if amp != "red" and r.get("regime_vix_level") is not None:
            from lib.vix_regime_ampel import classify_vix_regime

            amp = str(classify_vix_regime(r.get("regime_vix_level")).get("level") or "")
        if amp != "red":
            rows.append(
                {
                    "red_quality_tier": "",
                    "red_quality_hits": 0,
                    "red_quality_max": 0,
                    "red_quality_alpha_hint": "",
                }
            )
        else:
            rows.append(red_quality_fields_from_row(r))
    if not rows:
        for c in ("red_quality_tier", "red_quality_hits", "red_quality_max", "red_quality_alpha_hint"):
            o[c] = "" if c == "red_quality_tier" or c == "red_quality_alpha_hint" else 0
        return o
    q = pd.DataFrame(rows, index=o.index)
    for c in q.columns:
        o[c] = q[c].values
    return o


def red_quality_css_block() -> str:
    return """
        .red-quality-badge{font-size:.68em;padding:3px 8px;border-radius:8px;font-weight:600;margin-left:6px;vertical-align:middle}
        .red-quality-badge--hoch{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .red-quality-badge--mittel{background:#3d3a1a;color:#fff59d;border:1px solid #fbc02d}
        .red-quality-badge--niedrig{background:#3d2a1a;color:#ffcc80;border:1px solid #ef6c00}
        .red-quality-badge--unbekannt{background:#263238;color:#78909c;border:1px solid #455a64}
"""
