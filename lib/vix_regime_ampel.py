"""VIX-Regime-Ampel (3 Stufen, nur Anzeige — filtert keine Signale).

Kalibriert META+THRESHOLD / FINAL (scripts/_scratch_validate_ampel_v2.py):
  - grün / gelb: stabiler Regime-Vorteil
  - rot: VIX < 20 (gesamtes schwächeres Regime, ohne Unterteilung 14–17 vs. 17–20)
"""
from __future__ import annotations

import html as html_mod
from typing import Any

_VIX_SCALE_MIN = 10.0
_VIX_SCALE_MAX = 32.0
_AMPEL_LEVELS = ("red", "yellow", "green")
_LEGACY_RED_LEVELS = frozenset({"red", "red_tief", "red_oben", "orange"})


def vix_ampel_thresholds() -> tuple[float, float]:
    """(yellow_min, green_min) — rot ist alles darunter."""
    try:
        from lib.stock_rally_v10 import config_settings as cs

        return (
            float(getattr(cs, "VIX_AMPEL_YELLOW_MIN", 20.0)),
            float(getattr(cs, "VIX_AMPEL_GREEN_MIN", 25.0)),
        )
    except Exception:
        return 20.0, 25.0


def _normalize_ampel_level(level: str) -> str:
    lv = str(level or "").strip().lower()
    if lv in _LEGACY_RED_LEVELS:
        return "red"
    if lv in _AMPEL_LEVELS:
        return lv
    return "unknown"


def classify_vix_regime(
    vix_level: float | None,
    *,
    yellow_min: float | None = None,
    green_min: float | None = None,
    orange_min: float | None = None,
    red_oben_min: float | None = None,
) -> dict[str, Any]:
    """
    Ampel 3-stufig. Keys: level, label_de, regime_vix_level, hint_de.
    level ∈ green | yellow | red | unknown
    """
    y_min, g_min = vix_ampel_thresholds()
    if yellow_min is not None:
        y_min = float(yellow_min)
    if green_min is not None:
        g_min = float(green_min)
    _ = orange_min, red_oben_min  # Legacy-Kwargs, ignoriert

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
    else:
        level, label = "red", "VIX unter 20"
        hint = (
            "Schwächeres Gesamtregime (VIX < 20) — kein Signal-Filter; "
            "viele Einzeltreffer trotzdem möglich."
        )

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


def _vix_marker_pct(vix_level: float | None) -> float:
    if vix_level is None:
        return 50.0
    v = max(_VIX_SCALE_MIN, min(_VIX_SCALE_MAX, float(vix_level)))
    return 100.0 * (v - _VIX_SCALE_MIN) / (_VIX_SCALE_MAX - _VIX_SCALE_MIN)


def _vix_scale_segments_inner(*, compact: bool = False) -> str:
    y_min, g_min = vix_ampel_thresholds()
    segs = [
        ("red", max(0.5, y_min - _VIX_SCALE_MIN)),
        ("yellow", max(0.5, g_min - y_min)),
        ("green", max(0.5, _VIX_SCALE_MAX - g_min)),
    ]
    cls_extra = " vix-scale-track--compact" if compact else ""
    parts = [f'<div class="vix-scale-track{cls_extra}" aria-hidden="true">']
    for level, flex in segs:
        parts.append(f'<span class="vix-seg vix-seg--{level}" style="flex:{flex:.2f}"></span>')
    parts.append("</div>")
    return "".join(parts)


def _vix_lights_html(active_level: str) -> str:
    active = _normalize_ampel_level(active_level)
    parts = ['<span class="vix-lights" aria-hidden="true">']
    for lv in _AMPEL_LEVELS:
        on = " is-active" if lv == active else ""
        parts.append(f'<span class="vix-light vix-light--{lv}{on}"></span>')
    parts.append("</span>")
    return "".join(parts)


def vix_ampel_css_block() -> str:
    """CSS für eingebettete Website (docs/index.html)."""
    return """
        .vix-panel{background:#0d1117;border:1px solid #2d2d4e;border-radius:10px;padding:12px 14px;margin-bottom:4px}
        .vix-panel-lead{font-size:.78em;color:#90a4ae;line-height:1.45;margin:0 0 10px}
        .vix-scale{position:relative;margin:6px 0 4px}
        .vix-scale-track{display:flex;height:14px;border-radius:7px;overflow:hidden;border:1px solid #37474f}
        .vix-scale-track--compact{height:10px;border-radius:5px}
        .vix-seg{min-width:2px}
        .vix-seg--red{background:linear-gradient(180deg,#ef5350,#b71c1c)}
        .vix-seg--yellow{background:linear-gradient(180deg,#ffeb3b,#f9a825)}
        .vix-seg--green{background:linear-gradient(180deg,#81c784,#43a047)}
        .vix-scale-marker{position:absolute;top:-3px;width:4px;height:20px;margin-left:-2px;background:#fff;border:2px solid #1a1a2e;border-radius:3px;box-shadow:0 0 6px rgba(129,212,250,.85);pointer-events:none}
        .vix-scale-track--compact+.vix-scale-marker{height:16px;top:-3px}
        .vix-scale-legend{display:flex;justify-content:space-between;gap:4px;font-size:.65em;color:#78909c;margin-top:6px;line-height:1.25}
        .vix-scale-legend span{flex:1;text-align:center;min-width:0}
        .vix-scale-legend strong{display:block;color:#b0bec5;font-size:1.05em;font-weight:600}
        .vix-meter{margin:8px 0 4px;width:100%}
        .vix-meter-row{display:flex;align-items:center;gap:10px;margin-top:6px;flex-wrap:wrap}
        .vix-meter-text{font-size:.78em;color:#b0bec5;line-height:1.3}
        .vix-meter-text strong{color:#eceff1;font-weight:600}
        .vix-meter-vix{color:#81d4fa;font-weight:600;margin-left:4px}
        .vix-lights{display:inline-flex;gap:6px;align-items:center}
        .vix-light{width:12px;height:12px;border-radius:50%;opacity:.28;border:1px solid rgba(255,255,255,.15);flex-shrink:0}
        .vix-light--red{background:#ef5350}
        .vix-light--yellow{background:#ffca28}
        .vix-light--green{background:#66bb6a}
        .vix-light.is-active{opacity:1;transform:scale(1.15);box-shadow:0 0 8px currentColor;border-color:#fff}
        .vix-ampel{display:inline-flex;align-items:center;gap:5px;font-size:.72em;padding:2px 8px;border-radius:10px;white-space:nowrap;font-weight:500}
        .vix-ampel-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
        .vix-ampel--green{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .vix-ampel--green .vix-ampel-dot{background:#66bb6a}
        .vix-ampel--yellow{background:#3d3520;color:#fff59d;border:1px solid #f9a825}
        .vix-ampel--yellow .vix-ampel-dot{background:#ffca28}
        .vix-ampel--red{background:#3d1f1f;color:#ef9a9a;border:1px solid #c62828}
        .vix-ampel--red .vix-ampel-dot{background:#ef5350}
        .vix-ampel--unknown{background:#263238;color:#90a4ae;border:1px solid #546e7a}
        .vix-ampel--unknown .vix-ampel-dot{background:#78909c}
        .section-lead{font-size:.8em;color:#90a4ae;line-height:1.45;margin:0 0 12px}
        .sig-card--recent{box-shadow:0 0 0 1px #43a047,0 0 12px rgba(76,175,80,.12)}
        .sig-recent-tag{font-size:.65em;background:#1b3d24;color:#a5d6a7;border:1px solid #43a047;border-radius:8px;padding:1px 6px;margin-left:4px;vertical-align:middle}
"""

def vix_regime_full_css_block() -> str:
    """Ampel + Rot-Kontext-Chips."""
    from lib.vix_red_context_chips import red_context_chips_css_block

    return vix_ampel_css_block() + red_context_chips_css_block()


def vix_ampel_tooltip(c: dict[str, Any]) -> str:
    y_min, g_min = vix_ampel_thresholds()
    vix_txt = (
        f"{float(c['regime_vix_level']):.1f}"
        if c.get("regime_vix_level") is not None
        else "—"
    )
    hint = c.get("hint_de") or ""
    return (
        f"VIX {vix_txt} am Signaltag. Nur Regime-Einordnung (filtert keine Signale). "
        f"Drei Stufen: rot <{y_min:.0f}, gelb {y_min:.0f}–{g_min:.0f}, grün ≥{g_min:.0f}. "
        f"{hint}"
    )


def vix_ampel_scale_legend_labels_html() -> str:
    y_min, g_min = vix_ampel_thresholds()
    labels = [
        ("red", f"<{y_min:.0f}", "VIX unter 20"),
        ("yellow", f"{y_min:.0f}–{g_min:.0f}", "mittel"),
        ("green", f"≥{g_min:.0f}", "stark"),
    ]
    flexes = [
        max(0.5, y_min - _VIX_SCALE_MIN),
        max(0.5, g_min - y_min),
        max(0.5, _VIX_SCALE_MAX - g_min),
    ]
    parts = ['<div class="vix-scale-legend">']
    for (_, band, name), flex in zip(labels, flexes):
        parts.append(
            f'<span style="flex:{flex:.2f}"><strong>{html_mod.escape(band)}</strong>'
            f"{html_mod.escape(name)}</span>"
        )
    parts.append("</div>")
    return "".join(parts)


def vix_ampel_panel_html() -> str:
    """Große VIX-Skala + Legende für den Seitenkopf."""
    y_min, g_min = vix_ampel_thresholds()
    return (
        '<div class="vix-panel">'
        '<p class="vix-panel-lead">Einordnung des <strong>VIX am Signaltag</strong> — '
        "<strong>drei Stufen</strong> (grün / gelb / rot). Nur Regime-Kontext, "
        "<strong>kein Filter</strong> für einzelne Signale.</p>"
        '<div class="vix-scale">'
        + _vix_scale_segments_inner()
        + vix_ampel_scale_legend_labels_html()
        + "</div>"
        f'<p class="vix-panel-lead" style="margin:10px 0 0">Skala ca. '
        f"{_VIX_SCALE_MIN:.0f}–{_VIX_SCALE_MAX:.0f} · "
        f"rot &lt;{y_min:.0f} · gelb {y_min:.0f}–{g_min:.0f} · grün ≥{g_min:.0f}.</p>"
        "</div>"
    )


def vix_ampel_html_span(signal: dict[str, Any]) -> str:
    """Visuelles Regime-Badge pro Signal-Karte (Skala + 3 Lichter)."""
    if signal.get("regime_vix_level") is not None:
        c = classify_vix_regime(signal.get("regime_vix_level"))
    else:
        c = classify_vix_regime(None)
    amp = str(c["level"])
    vix_txt = (
        f"{float(c['regime_vix_level']):.1f}"
        if c.get("regime_vix_level") is not None
        else "—"
    )
    label = html_mod.escape(str(c.get("label_de") or "VIX"))
    tip = html_mod.escape(vix_ampel_tooltip(c))
    pct = _vix_marker_pct(c.get("regime_vix_level"))
    return (
        f'<div class="vix-meter" title="{tip}">'
        f'<div class="vix-scale">'
        f"{_vix_scale_segments_inner(compact=True)}"
        f'<span class="vix-scale-marker" style="left:{pct:.1f}%"></span>'
        f"</div>"
        f'<div class="vix-meter-row">'
        f"{_vix_lights_html(amp)}"
        f'<span class="vix-meter-text"><strong>{label}</strong>'
        f'<span class="vix-meter-vix">VIX {html_mod.escape(vix_txt)}</span></span>'
        f"</div></div>"
    )


def vix_ampel_legend_html() -> str:
    """Alias: Kopf-Panel (ersetzt die alte Textzeile)."""
    return vix_ampel_panel_html()
