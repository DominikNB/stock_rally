"""OOS-Kontext-Ampel: grün / gelb / rot (Makro-Event + VIX, OOS-validiert)."""
from __future__ import annotations

import html as html_mod
from typing import Any, Mapping

_CONTEXT_LEVELS = ("green", "yellow", "red")

_OLD_SIGNAL_KEYS = (
    "red_summary_tier",
    "red_summary_de",
    "red_summary_tooltip",
    "red_summary_html",
    "red_quality_html",
    "red_context_html",
    "red_context_chips",
    "red_context_llm",
    "red_quality_tier",
    "quality_red",
    "quality_gld",
    "quality_spike_ok",
    "vix_regime_label",
    "vix_regime_hint",
)


def context_vix_green_min() -> float:
    try:
        from lib.stock_rally_v10 import config_settings as cs

        return float(getattr(cs, "SIGNAL_CONTEXT_VIX_GREEN_MIN", 20.0))
    except Exception:
        return 20.0


def _truthy_macro_event(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            if val != val:
                return None
        except TypeError:
            pass
        return bool(int(val))
    s = str(val).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _parse_vix(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def classify_signal_context_tier(signal: Mapping[str, Any]) -> dict[str, Any]:
    """
    grün: kein Makro-Event in ±2 Handelstagen UND VIX ≥ SIGNAL_CONTEXT_VIX_GREEN_MIN
    rot:  Makro-Event in ±2 Handelstagen
    gelb: sonst (Standard-Modellsignal)
    """
    vix_min = context_vix_green_min()
    macro = _truthy_macro_event(signal.get("macro_event_within_2bd"))
    vix = _parse_vix(signal.get("regime_vix_level"))

    if macro is True:
        return {
            "level": "red",
            "label_de": "Makro-Risiko",
            "hint_de": (
                "Makro-Termin (z. B. FOMC, CPI) innerhalb von ±2 Handelstagen — "
                "historisch schwächere OOS-Renditen. Modell-Signal bleibt gültig."
            ),
        }
    if macro is False and vix is not None and vix >= vix_min:
        return {
            "level": "green",
            "label_de": "Kontext gut",
            "hint_de": (
                f"Kein Makro-Event in ±2 Handelstagen und VIX ≥ {vix_min:.0f} "
                f"(am Signaltag: {vix:.1f}) — historisch stärkstes OOS-Regime."
            ),
        }
    if vix is not None and vix < vix_min:
        hint = (
            f"VIX {vix:.1f} unter {vix_min:.0f} — ruhigeres Marktregime; "
            "Modell-Signal ohne Makro-Warnung."
        )
    else:
        hint = "Standard-Modellsignal ohne Makro-Warnung (VIX oder Makro-Kalender unvollständig)."
    return {
        "level": "yellow",
        "label_de": "Standard",
        "hint_de": hint,
    }


def context_tier_llm_fields_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    """
    LLM/CSV-Felder — **dieselbe** Ampel-Logik wie ``attach_signal_context_tier`` (Website).
    ``vix_regime_ampel`` = context_tier (Kompatibilitätsname für Prompt/Filter).
    """
    c = classify_signal_context_tier(row)
    tier = str(c["level"])
    label = str(c["label_de"])
    hint = str(c.get("hint_de") or "")
    vix = _parse_vix(row.get("regime_vix_level"))
    vix_s = f"{vix:.1f}" if vix is not None else "n/a"

    if tier == "red":
        llm = (
            f"Kontext-Ampel rot ({label}): {hint} "
            "Historisch schwächere OOS-Renditen bei Makro-Nähe — das Modell-Signal bleibt gültig; "
            "in Makro-Intro, Daten-Check und Fazit mit einweben (Vorsicht bei Terminen/Binary Events)."
        )
    elif tier == "green":
        llm = (
            f"Kontext-Ampel grün ({label}): {hint} "
            "Günstigeres historisches OOS-Regime — kein Makro-Warnhinweis."
        )
    else:
        llm = (
            f"Kontext-Ampel gelb ({label}): {hint} "
            f"(regime_vix_level={vix_s}). "
            "Gelb = Standard-Modellsignal — **nicht** „rot“ (Makro-Warnung). "
            "Keine veraltete „VIX unter 20 = rot“-Einordnung; keine Rot-Chip-Logik. "
            "Einordnung über News, RS, Liquidität und Makro."
        )

    return {
        "context_tier": tier,
        "context_label_de": label,
        "context_hint_de": hint,
        "vix_regime_ampel": tier,
        "red_context_llm": llm,
    }


def attach_context_tier_llm_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Setzt Kontext-Ampel + red_context_llm (Website-identisch) auf Signal-DataFrame."""
    import pandas as pd

    o = df.copy()
    fields = [context_tier_llm_fields_from_row(r) for _, r in o.iterrows()]
    if not fields:
        for col in ("context_tier", "context_label_de", "context_hint_de", "vix_regime_ampel", "red_context_llm"):
            o[col] = ""
        return o
    for col in ("context_tier", "context_label_de", "context_hint_de", "vix_regime_ampel", "red_context_llm"):
        o[col] = [f.get(col, "") for f in fields]
    return o


def attach_signal_context_tier(signal: dict[str, Any]) -> dict[str, Any]:
    """Setzt Kontext-Felder; entfernt alte Rot-/VIX-Badge-Felder aus dem Export."""
    c = classify_signal_context_tier(signal)
    tier = str(c["level"])
    signal["context_tier"] = tier
    signal["context_label_de"] = str(c["label_de"])
    signal["context_hint_de"] = str(c.get("hint_de") or "")
    # Filter/JS-Kompatibilität (bestehende data-vix-ampel / signals.json-Feld)
    signal["vix_regime_ampel"] = tier
    signal["context_tier_html"] = context_tier_html_span(signal)
    for k in _OLD_SIGNAL_KEYS:
        signal.pop(k, None)
    return signal


def context_tier_counts(signals: list[Mapping[str, Any]]) -> dict[str, int]:
    from collections import Counter

    c = Counter(
        str(s.get("context_tier") or s.get("vix_regime_ampel") or "unknown").strip().lower()
        for s in signals
    )
    return {
        "all": len(signals),
        "red": int(c.get("red", 0)),
        "yellow": int(c.get("yellow", 0)),
        "green": int(c.get("green", 0)),
        "unknown": int(c.get("unknown", 0)),
    }


def context_tier_css_block() -> str:
    return """
        .context-panel{background:#0d1117;border:1px solid #2d2d4e;border-radius:10px;padding:12px 14px;margin-bottom:4px}
        .context-panel-lead{font-size:.78em;color:#90a4ae;line-height:1.45;margin:0}
        .context-tier{display:inline-flex;align-items:center;gap:6px;font-size:.72em;padding:3px 9px;border-radius:10px;white-space:nowrap;font-weight:600}
        .context-tier-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
        .context-tier--green{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .context-tier--green .context-tier-dot{background:#66bb6a}
        .context-tier--yellow{background:#3d3520;color:#fff59d;border:1px solid #f9a825}
        .context-tier--yellow .context-tier-dot{background:#ffca28}
        .context-tier--red{background:#3d1f1f;color:#ef9a9a;border:1px solid #c62828}
        .context-tier--red .context-tier-dot{background:#ef5350}
        .context-tier--unknown{background:#263238;color:#90a4ae;border:1px solid #546e7a}
        .context-tier--unknown .context-tier-dot{background:#78909c}
        .section-lead{font-size:.8em;color:#90a4ae;line-height:1.45;margin:0 0 12px}
        .sig-card--recent{box-shadow:0 0 0 1px #43a047,0 0 12px rgba(76,175,80,.12)}
        .sig-recent-tag{font-size:.65em;background:#1b3d24;color:#a5d6a7;border:1px solid #43a047;border-radius:8px;padding:1px 6px;margin-left:4px;vertical-align:middle}
"""


def context_tier_panel_html() -> str:
    vix_min = context_vix_green_min()
    return (
        '<div class="context-panel">'
        '<p class="context-panel-lead">'
        "<strong>Kontext-Ampel</strong> pro Signal (OOS-validiert, kein zweites Modell): "
        f"<strong style=\"color:#a5d6a7\">Grün</strong> = kein Makro-Event in ±2 Handelstagen "
        f"und VIX ≥ {vix_min:.0f}; "
        "<strong style=\"color:#ef9a9a\">Rot</strong> = Makro-Event nahe; "
        "<strong style=\"color:#fff59d\">Gelb</strong> = Standard-Modellsignal."
        "</p></div>"
    )


def context_tier_html_span(signal: Mapping[str, Any]) -> str:
    if signal.get("context_tier"):
        tier = str(signal["context_tier"]).strip().lower()
        label = str(signal.get("context_label_de") or tier)
        tip = str(signal.get("context_hint_de") or "")
    else:
        c = classify_signal_context_tier(signal)
        tier = str(c["level"])
        label = str(c["label_de"])
        tip = str(c.get("hint_de") or "")
    if tier not in _CONTEXT_LEVELS:
        tier = "unknown"
    tip_esc = html_mod.escape(tip)
    label_esc = html_mod.escape(label)
    return (
        f'<span class="context-tier context-tier--{tier}" title="{tip_esc}">'
        f'<span class="context-tier-dot" aria-hidden="true"></span>'
        f"{label_esc}</span>"
    )


def context_tier_full_css_block() -> str:
    from lib.website_ampel_filter import website_ampel_filter_css_block

    return context_tier_css_block() + website_ampel_filter_css_block()
