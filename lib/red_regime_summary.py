"""
Rot-Regime: Qualitäts-Badge 0/1/2 (GLD + Makro-GDELT Vol-Spike).

Score = (gld_ret_5d < gld_ref) + (news_macro_vol_spike <= spike_ref), je 0/1.

Validierung: scripts/_scratch_validate_red_research_batch.py
  macro_vol_spike_low: IS +0,94 pp | OOS +0,94 pp
  Badge >=1 vs 0:        IS −0,07 pp | OOS +0,94 pp
  Badge ==2 vs 0:        IS +1,51 pp | OOS n=17/23 (Gold-Phase)
"""
from __future__ import annotations

import html as html_mod
from typing import Any, Mapping

from lib.red_signal_quality import (
    NEWS_MACRO_VOL_SPIKE_COL,
    _gld_ret5_low,
    _macro_vol_spike_low,
    calibrate_gld_ret5_median_red_ref,
    calibrate_macro_vol_spike_median_red_ref,
    inject_gld_median_red_ref,
    inject_macro_vol_spike_median_red_ref,
)


def _score_parts(row: Mapping[str, Any]) -> tuple[int | None, bool | None, bool | None]:
    gld_low = _gld_ret5_low(row)
    spike_low = _macro_vol_spike_low(row)
    g_pt = None if gld_low is None else int(gld_low)
    s_pt = None if spike_low is None else int(spike_low)
    if g_pt is None and s_pt is None:
        return None, gld_low, spike_low
    if g_pt is None:
        return s_pt, gld_low, spike_low
    if s_pt is None:
        return g_pt, gld_low, spike_low
    return g_pt + s_pt, gld_low, spike_low


def red_quality_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    row_d = dict(row)
    inject_gld_median_red_ref(row_d)
    inject_macro_vol_spike_median_red_ref(row_d)
    score, gld_low, spike_low = _score_parts(row_d)
    gref = row_d.get("gld_ret5_median_red_ref")
    sref = row_d.get("macro_vol_spike_median_red_ref")

    if score is None:
        return {
            "quality_red": None,
            "quality_gld": None,
            "quality_spike_ok": None,
            "red_summary_tier": "unbekannt",
            "red_summary_de": "",
            "red_summary_tooltip": "GLD oder Makro-News fehlen — kein Qualitäts-Badge",
        }

    g = row_d.get("gld_ret_5d")
    sp = row_d.get(NEWS_MACRO_VOL_SPIKE_COL)
    g_pct = f"{100 * float(g):+.2f}%" if g is not None else "—"
    gref_pct = f"{100 * float(gref):+.2f}%" if gref is not None else "—"
    sp_s = f"{float(sp):.2f}" if sp is not None else "—"
    sref_s = f"{float(sref):.2f}" if sref is not None else "—"

    gld_pt = "—" if gld_low is None else ("✓" if gld_low else "✗")
    spike_pt = "—" if spike_low is None else ("✓" if spike_low else "✗")
    tip = (
        f"Score {score}/2 — GLD {gld_pt} ({g_pct} vs {gref_pct}), "
        f"Makro-Vol-Spike {spike_pt} ({sp_s} vs Median {sref_s}). "
        "OOS: News-Ruhe stabil; GLD in Gold-Phase oft one-sided."
    )

    if score >= 2:
        line = "Qualität 2 — Gold schwach & Makro-News ruhig"
        tier = "zwei"
    elif score == 1:
        if spike_low and not gld_low:
            line = "Qualität 1 — Makro-News ruhig (Gold-Rally)"
        elif gld_low and not spike_low:
            line = "Qualität 1 — Gold schwach (News-Spike)"
        else:
            line = "Qualität 1 — ein Faktor günstig"
        tier = "eins"
    else:
        line = "Qualität 0 — Risk-off (Gold-Rally und/oder News-Spike)"
        tier = "null"

    gld_q = None if gld_low is None else int(gld_low)
    spike_q = None if spike_low is None else int(spike_low)

    return {
        "quality_red": int(score),
        "quality_gld": gld_q,
        "quality_spike_ok": spike_q,
        "macro_vol_spike_median_red_ref": sref,
        "red_summary_tier": tier,
        "red_summary_de": line,
        "red_summary_tooltip": tip,
    }


def red_quality_badge_html(fields: Mapping[str, Any]) -> str:
    q = fields.get("quality_red")
    if q is None:
        return ""
    tip = html_mod.escape(str(fields.get("red_summary_tooltip") or ""))
    label = html_mod.escape(str(fields.get("red_summary_de") or f"Qualität {q}"))
    tier = str(fields.get("red_summary_tier") or "null")
    css = {"zwei": "zwei", "eins": "eins", "null": "null"}.get(tier, "null")
    return (
        f'<span class="red-gld-quality red-gld-quality--{css}" title="{tip}">{label}</span>'
    )


def attach_red_regime_summary(signal: dict[str, Any]) -> dict[str, Any]:
    """Hängt Rot-Qualitäts-Badge 0/1/2 an rot-Signale (UI + JSON)."""
    amp = str(signal.get("vix_regime_ampel") or "").strip().lower()
    if amp != "red":
        signal["quality_red"] = None
        signal["quality_gld"] = None
        signal["quality_spike_ok"] = None
        signal["red_summary_tier"] = ""
        signal["red_summary_de"] = ""
        signal["red_summary_html"] = ""
        signal["red_context_html"] = ""
        signal["red_quality_html"] = ""
        return signal

    fields = red_quality_fields(signal)
    signal.update(fields)
    signal["red_summary_html"] = red_quality_badge_html(fields)
    signal["red_quality_html"] = signal["red_summary_html"]
    signal["red_context_html"] = ""
    signal["red_context_chips"] = []
    return signal


def red_regime_summary_css_block() -> str:
    return """
        .red-gld-quality{display:inline-block;font-size:.75em;line-height:1.35;margin:6px 0 4px;padding:7px 10px;border-radius:8px;font-weight:600;max-width:100%}
        .red-gld-quality--zwei{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .red-gld-quality--eins{background:#2a3a4a;color:#90caf9;border:1px solid #546e7a}
        .red-gld-quality--null{background:#3d2a1a;color:#ffcc80;border:1px solid #ef6c00}
"""


# Abwärtskompatibilität
red_gld_quality_fields = red_quality_fields
red_gld_quality_badge_html = red_quality_badge_html
