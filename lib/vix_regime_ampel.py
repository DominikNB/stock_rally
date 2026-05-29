"""VIX-Regime-Ampel v2 (Anzeige only — filtert keine Signale).

Kalibriert auf META+THRESHOLD, auf FINAL geprüft (scripts/_scratch_validate_ampel_v2.py):
  - grün / gelb: stabiler Regime-Vorteil
  - orange (14–17) / rot_oben (17–20): schwächeres Regime, kein Einzel-Signal-Filter
  - VIX-Z-Zweig bewusst nicht enthalten (OOS nicht haltbar)
"""
from __future__ import annotations

import html as html_mod
from typing import Any


def vix_ampel_thresholds() -> tuple[float, float, float, float]:
    """(yellow_min, green_min, orange_min, red_oben_min)."""
    try:
        from lib.stock_rally_v10 import config_settings as cs

        return (
            float(getattr(cs, "VIX_AMPEL_YELLOW_MIN", 20.0)),
            float(getattr(cs, "VIX_AMPEL_GREEN_MIN", 25.0)),
            float(getattr(cs, "VIX_AMPEL_ORANGE_MIN", 14.0)),
            float(getattr(cs, "VIX_AMPEL_RED_OBEN_MIN", 17.0)),
        )
    except Exception:
        return 20.0, 25.0, 14.0, 17.0


def classify_vix_regime(
    vix_level: float | None,
    *,
    yellow_min: float | None = None,
    green_min: float | None = None,
    orange_min: float | None = None,
    red_oben_min: float | None = None,
) -> dict[str, Any]:
    """
    Ampel v2. Keys: level, label_de, regime_vix_level, hint_de.
    level ∈ green | yellow | orange | red_oben | red_tief | unknown
    """
    y_min, g_min, o_min, r_min = vix_ampel_thresholds()
    if yellow_min is not None:
        y_min = float(yellow_min)
    if green_min is not None:
        g_min = float(green_min)
    if orange_min is not None:
        o_min = float(orange_min)
    if red_oben_min is not None:
        r_min = float(red_oben_min)

    if vix_level is None:
        return {
            "level": "unknown",
            "label_de": "VIX n/a",
            "regime_vix_level": None,
            "hint_de": "",
        }
    try:
        v = float(vix_level)
    except (TypeError, ValueError):
        return {
            "level": "unknown",
            "label_de": "VIX n/a",
            "regime_vix_level": None,
            "hint_de": "",
        }
    if not (v == v):
        return {
            "level": "unknown",
            "label_de": "VIX n/a",
            "regime_vix_level": None,
            "hint_de": "",
        }

    if v >= g_min:
        level, label = "green", "Regime stark"
        hint = "Historisch bestes VIX-Regime (META+THRESHOLD/FINAL)."
    elif v >= y_min:
        level, label = "yellow", "Regime mittel"
        hint = "Mittleres VIX-Regime."
    elif v >= r_min:
        level, label = "red_oben", "VIX knapp <20"
        hint = "Schwächstes rotes Band — kein Signal-Filter."
    elif v >= o_min:
        level, label = "orange", "Regime schwach"
        hint = "Schwaches Regime; viele Einzelsignale trotzdem ok."
    else:
        level, label = "red_tief", "VIX sehr niedrig"
        hint = "Sehr ruhiger Markt-VIX."

    return {
        "level": level,
        "label_de": label,
        "regime_vix_level": round(v, 2),
        "hint_de": hint,
    }


def ampel_fields_from_vix(vix_level: float | None, **kwargs) -> dict[str, Any]:
    """Flache Felder für Signal-Dicts / JSON."""
    c = classify_vix_regime(vix_level, **kwargs)
    return {
        "regime_vix_level": c["regime_vix_level"],
        "vix_regime_ampel": c["level"],
        "vix_regime_label": c["label_de"],
        "vix_regime_hint": c.get("hint_de") or "",
    }


def vix_ampel_css_block() -> str:
    """CSS für eingebettete Website (docs/index.html)."""
    return """
        .vix-ampel{display:inline-flex;align-items:center;gap:5px;font-size:.72em;padding:2px 8px;border-radius:10px;white-space:nowrap;font-weight:500}
        .vix-ampel-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
        .vix-ampel--green{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .vix-ampel--green .vix-ampel-dot{background:#66bb6a}
        .vix-ampel--yellow{background:#3d3520;color:#fff59d;border:1px solid #f9a825}
        .vix-ampel--yellow .vix-ampel-dot{background:#ffca28}
        .vix-ampel--orange{background:#3d2a14;color:#ffcc80;border:1px solid #ef6c00}
        .vix-ampel--orange .vix-ampel-dot{background:#ffa726}
        .vix-ampel--red_oben{background:#3d1f1f;color:#ef9a9a;border:1px solid #c62828}
        .vix-ampel--red_oben .vix-ampel-dot{background:#ef5350}
        .vix-ampel--red_tief{background:#2a2438;color:#b39ddb;border:1px solid #7e57c2}
        .vix-ampel--red_tief .vix-ampel-dot{background:#9575cd}
        .vix-ampel--unknown{background:#263238;color:#90a4ae;border:1px solid #546e7a}
        .vix-ampel--unknown .vix-ampel-dot{background:#78909c}
        .vix-ampel-legend{font-size:.72em;color:#90a4ae;line-height:1.45;margin:8px 0 0}
"""


def vix_ampel_tooltip(c: dict[str, Any]) -> str:
    y_min, g_min, o_min, r_min = vix_ampel_thresholds()
    vix_txt = (
        f"{float(c['regime_vix_level']):.1f}"
        if c.get("regime_vix_level") is not None
        else "—"
    )
    hint = c.get("hint_de") or ""
    return (
        f"VIX {vix_txt} am Signaltag. Nur Regime-Einordnung (filtert keine Signale). "
        f"Bänder: <{o_min:.0f} sehr niedrig, {o_min:.0f}–{r_min:.0f} schwach, "
        f"{r_min:.0f}–{y_min:.0f} knapp <20, {y_min:.0f}–{g_min:.0f} mittel, ≥{g_min:.0f} stark. "
        f"{hint}"
    )


def vix_ampel_html_span(signal: dict[str, Any]) -> str:
    """HTML-Badge für eine Signal-Zeile (Website)."""
    if signal.get("regime_vix_level") is not None:
        c = classify_vix_regime(signal.get("regime_vix_level"))
    else:
        c = classify_vix_regime(None)
    amp = c["level"]
    vix_txt = (
        f"{float(c['regime_vix_level']):.1f}"
        if c.get("regime_vix_level") is not None
        else "—"
    )
    label = html_mod.escape(str(c.get("label_de") or "VIX"))
    tip = html_mod.escape(vix_ampel_tooltip(c))
    return (
        f'<span class="vix-ampel vix-ampel--{amp}" title="{tip}">'
        f'<span class="vix-ampel-dot" aria-hidden="true"></span>'
        f"{label} (VIX {vix_txt})</span>"
    )


def vix_ampel_legend_html() -> str:
    """Kurz-Legende für Website-Kopfbereich."""
    y_min, g_min, o_min, r_min = vix_ampel_thresholds()
    return (
        f'<p class="vix-ampel-legend">VIX-Regime (nur Kontext, kein Filter): '
        f'<span class="vix-ampel vix-ampel--green"><span class="vix-ampel-dot"></span>stark</span> ≥{g_min:.0f} · '
        f'<span class="vix-ampel vix-ampel--yellow"><span class="vix-ampel-dot"></span>mittel</span> {y_min:.0f}–{g_min:.0f} · '
        f'<span class="vix-ampel vix-ampel--orange"><span class="vix-ampel-dot"></span>schwach</span> {o_min:.0f}–{r_min:.0f} · '
        f'<span class="vix-ampel vix-ampel--red_oben"><span class="vix-ampel-dot"></span>knapp &lt;20</span> {r_min:.0f}–{y_min:.0f}'
        f"</p>"
    )
