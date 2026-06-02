"""
Rot-Regime: einfaches Qualitäts-Badge 0/1 (nur GLD).

Regel: quality_gld = 1 wenn gld_ret_5d < gld_ret5_median_red_ref (globaler Rot-Median).

Validierung: scripts/_scratch_validate_red_badge_extended.py
  META+THR +0,58 pp | FINAL +1,04 pp (Median-Split-Test, gleiche Richtung OOS)
  Feste 0/1-Schwelle (globaler Median): META+THR +0,00 pp | FINAL schwach (Gold-Rally-Phase)

Keine Chips, keine Liquidität im Score.
"""
from __future__ import annotations

import html as html_mod
from typing import Any, Mapping

from lib.red_signal_quality import (
    _gld_ret5_low,
    calibrate_gld_ret5_median_red_ref,
    inject_gld_median_red_ref,
)


def red_gld_quality_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    row_d = dict(row)
    inject_gld_median_red_ref(row_d)
    gld_low = _gld_ret5_low(row_d)
    ref = row_d.get("gld_ret5_median_red_ref")

    if gld_low is None:
        return {
            "quality_gld": None,
            "red_summary_tier": "unbekannt",
            "red_summary_de": "",
            "red_summary_tooltip": "GLD 5d-Rendite fehlt — kein Qualitäts-Badge",
        }

    q = 1 if gld_low else 0
    g = row_d.get("gld_ret_5d")
    g_pct = f"{100 * float(g):+.2f}%" if g is not None else "—"
    ref_pct = f"{100 * float(ref):+.2f}%" if ref is not None else "—"

    if q == 1:
        line = "Qualität 1 — Gold schwach (kein Flucht-in-Gold)"
        tip = (
            f"gld_ret_5d = {g_pct} unter Rot-Median {ref_pct}. "
            "In rot historisch günstiger (IS+OOS, ret_mean_5)."
        )
        tier = "eins"
    else:
        line = "Qualität 0 — Gold-Rally (Risk-off-Kontext)"
        tip = (
            f"gld_ret_5d = {g_pct} über Rot-Median {ref_pct}. "
            "In rot historisch schwächerer Kontext."
        )
        tier = "null"

    return {
        "quality_gld": q,
        "gld_ret5_median_red_ref": ref,
        "red_summary_tier": tier,
        "red_summary_de": line,
        "red_summary_tooltip": tip,
    }


def red_gld_quality_badge_html(fields: Mapping[str, Any]) -> str:
    q = fields.get("quality_gld")
    if q is None:
        return ""
    tip = html_mod.escape(str(fields.get("red_summary_tooltip") or ""))
    label = html_mod.escape(str(fields.get("red_summary_de") or f"Qualität {q}"))
    css = "eins" if int(q) == 1 else "null"
    return (
        f'<span class="red-gld-quality red-gld-quality--{css}" title="{tip}">{label}</span>'
    )


def attach_red_regime_summary(signal: dict[str, Any]) -> dict[str, Any]:
    """Hängt GLD 0/1-Badge an rot-Signale (UI + JSON)."""
    amp = str(signal.get("vix_regime_ampel") or "").strip().lower()
    if amp != "red":
        signal["quality_gld"] = None
        signal["red_summary_tier"] = ""
        signal["red_summary_de"] = ""
        signal["red_summary_html"] = ""
        signal["red_context_html"] = ""
        signal["red_quality_html"] = ""
        return signal

    fields = red_gld_quality_fields(signal)
    signal.update(fields)
    signal["red_summary_html"] = red_gld_quality_badge_html(fields)
    signal["red_quality_html"] = signal["red_summary_html"]
    signal["red_context_html"] = ""
    signal["red_context_chips"] = []
    return signal


def red_regime_summary_css_block() -> str:
    return """
        .red-gld-quality{display:inline-block;font-size:.75em;line-height:1.35;margin:6px 0 4px;padding:7px 10px;border-radius:8px;font-weight:600;max-width:100%}
        .red-gld-quality--eins{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .red-gld-quality--null{background:#3d2a1a;color:#ffcc80;border:1px solid #ef6c00}
"""
