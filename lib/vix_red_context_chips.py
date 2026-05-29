"""
Kontext-Chips nur fuer VIX-Regime rot (VIX < 20).

Vier nicht-redundante Hinweise, OOS auf META+THRESHOLD + FINAL validiert
(scripts/_scratch_validate_all_context.py). Kein Filter — nur Anzeige.

  1. VIX unter 20-Tage-Mittel (regime_vix_z_20d < 0)
  2. VIX-Termstruktur entspannt (VIX3M/VIX unter Schwelle)
  3. Wenig Sektor-Crowding (sector_hhi_same_day unter Median-Referenz)
  4. Sektor-News positiver als Makro (news_sec_minus_macro_tone > 0)
"""
from __future__ import annotations

import html as html_mod
from typing import Any, Mapping

_CHIP_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "vix_z",
        "label": "VIX vs. 20d-Mittel",
        "tooltip": (
            "Grün: VIX unter dem 20-Tage-Mittel (Z < 0). "
            "Orange: darueber. In rot historisch guenstiger wenn gruen (META+THR/FINAL)."
        ),
    },
    {
        "id": "vix_term",
        "label": "VIX-Term 3M/VIX",
        "tooltip": (
            "Grün: niedrige Ratio (Termstruktur entspannt). "
            "Orange: hohe Ratio. In rot historisch schlechter wenn orange."
        ),
    },
    {
        "id": "crowding",
        "label": "Sektor-Crowding",
        "tooltip": (
            "Grün: wenig parallele Signale im gleichen Sektor (niedriger HHI). "
            "Orange: viel Crowding am Signaltag."
        ),
    },
    {
        "id": "news_sec",
        "label": "News Sektor vs. Makro",
        "tooltip": (
            "Grün: Sektor-News-Ton ueber Makro-News. "
            "Orange: Makro dominiert. In rot historisch guenstiger wenn gruen."
        ),
    },
)


def _chip_thresholds() -> dict[str, float]:
    try:
        from lib.stock_rally_v10 import config_settings as cs

        return {
            "vix3m_vix_max": float(getattr(cs, "VIX_RED_CHIP_VIX3M_RATIO_MAX", 1.16)),
            "sector_hhi_max": float(getattr(cs, "VIX_RED_CHIP_SECTOR_HHI_MAX", 0.35)),
        }
    except Exception:
        return {"vix3m_vix_max": 1.16, "sector_hhi_max": 0.35}


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


def _news_sec_minus_macro(row: Mapping[str, Any]) -> float | None:
    if "news_sec_minus_macro_tone" in row:
        return _f(row.get("news_sec_minus_macro_tone"))
    sec = macro = None
    for k, v in row.items():
        ks = str(k)
        if ks.startswith("news_sec") and ks.endswith("_tone"):
            sec = _f(v)
        if ks.startswith("news_macro") and ks.endswith("_tone"):
            macro = _f(v)
    if sec is None or macro is None:
        return None
    return sec - macro


def evaluate_red_context_chips(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Liefert 4 Chips mit state in good | warn | na.
    good = historisch guenstig in rot; warn = unguenstig; na = Daten fehlen.
    """
    thr = _chip_thresholds()
    vix_z = _f(row.get("regime_vix_z_20d"))
    vix_ratio = _f(row.get("vix3m_vix_ratio"))
    hhi = _f(row.get("sector_hhi_same_day"))
    news_diff = _news_sec_minus_macro(row)

    states: dict[str, str] = {}

    if vix_z is None:
        states["vix_z"] = "na"
    elif vix_z < 0:
        states["vix_z"] = "good"
    else:
        states["vix_z"] = "warn"

    if vix_ratio is None:
        states["vix_term"] = "na"
    elif vix_ratio < thr["vix3m_vix_max"]:
        states["vix_term"] = "good"
    else:
        states["vix_term"] = "warn"

    if hhi is None:
        states["crowding"] = "na"
    elif hhi < thr["sector_hhi_max"]:
        states["crowding"] = "good"
    else:
        states["crowding"] = "warn"

    if news_diff is None:
        states["news_sec"] = "na"
    elif news_diff > 0:
        states["news_sec"] = "good"
    else:
        states["news_sec"] = "warn"

    out: list[dict[str, Any]] = []
    for spec in _CHIP_SPECS:
        cid = spec["id"]
        out.append(
            {
                "id": cid,
                "label": spec["label"],
                "tooltip": spec["tooltip"],
                "state": states.get(cid, "na"),
            }
        )
    return out


def chip_fields_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flache Felder fuer JSON / Signal-Dict."""
    chips = evaluate_red_context_chips(row)
    good = sum(1 for c in chips if c["state"] == "good")
    warn = sum(1 for c in chips if c["state"] == "warn")
    na = sum(1 for c in chips if c["state"] == "na")
    return {
        "red_context_chips": chips,
        "red_context_good": good,
        "red_context_warn": warn,
        "red_context_na": na,
    }


def red_context_chips_html(chips: list[dict[str, Any]] | None) -> str:
    if not chips:
        return ""
    parts = ['<div class="red-ctx-chips" aria-label="Kontext in rot">']
    for c in chips:
        st = str(c.get("state") or "na")
        label = html_mod.escape(str(c.get("label") or ""))
        tip = html_mod.escape(str(c.get("tooltip") or ""))
        parts.append(
            f'<span class="red-ctx-chip red-ctx-chip--{st}" title="{tip}">{label}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def red_context_chips_css_block() -> str:
    return """
        .red-ctx-chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}
        .red-ctx-chip{font-size:.68em;padding:3px 8px;border-radius:8px;font-weight:500;line-height:1.25;max-width:100%}
        .red-ctx-chip--good{background:#1b3d24;color:#a5d6a7;border:1px solid #43a047}
        .red-ctx-chip--warn{background:#3d2a1a;color:#ffcc80;border:1px solid #ef6c00}
        .red-ctx-chip--na{background:#263238;color:#78909c;border:1px solid #455a64}
        .vix-guide-list{margin:0;padding-left:1.15em;font-size:.78em;color:#b0bec5;line-height:1.5}
        .vix-guide-list li{margin:.35em 0}
        .vix-guide-list strong{color:#eceff1}
"""


def vix_regime_guide_panel_html() -> str:
    """Ampel-Legende + Nutzung der Rot-Chips (Seitenkopf)."""
    from lib.vix_regime_ampel import vix_ampel_panel_html, vix_ampel_thresholds

    y_min, g_min = vix_ampel_thresholds()
    ampel = vix_ampel_panel_html()
    return (
        f'{ampel}'
        '<div class="vix-panel" style="margin-top:10px">'
        "<p class=\"vix-panel-lead\"><strong>So nutzen Sie die Ampel</strong></p>"
        "<ul class=\"vix-guide-list\">"
        f"<li><strong>Grün (VIX ≥ {g_min:.0f}):</strong> historisch stärkstes Regime — Signale im Schnitt am besten.</li>"
        f"<li><strong>Gelb ({y_min:.0f}–{g_min:.0f}):</strong> mittleres Regime.</li>"
        f"<li><strong>Rot (VIX &lt; {y_min:.0f}):</strong> schwächeres Gesamtregime — "
        "<em>kein Ausschluss</em> einzelner Trades; viele Treffer bleiben möglich.</li>"
        "</ul>"
        "<p class=\"vix-panel-lead\" style=\"margin-top:10px\">"
        "<strong>Nur bei rot:</strong> vier Zusatz-Chips pro Signal (kein zweites Scoring). "
        "Sie ersetzen <em>nicht</em> Modell-Wahrscheinlichkeit und Threshold.</p>"
        "<ul class=\"vix-guide-list\">"
        "<li><strong>Grüner Chip</strong> = Kontext historisch günstig in rot (Train + Test).</li>"
        "<li><strong>Oranger Chip</strong> = eher Vorsicht (Size, Stop, Skepsis).</li>"
        "<li><strong>Grau</strong> = Daten fehlen — Chip ignorieren.</li>"
        "<li>Entscheidung: erst Signal ja/nein (prob/Setup), dann in rot: "
        "mehr grüne Chips → mehr Überzeugung; viele orange → vorsichtiger handeln.</li>"
        "</ul></div>"
    )


def attach_red_context_to_signal(signal: dict[str, Any]) -> dict[str, Any]:
    """Erweitert Signal-Dict um Chips; nur sinnvoll wenn Ampel rot."""
    amp = str(signal.get("vix_regime_ampel") or "").strip().lower()
    if amp not in ("red",) and _f(signal.get("regime_vix_level")) is not None:
        from lib.vix_regime_ampel import classify_vix_regime

        amp = str(classify_vix_regime(signal.get("regime_vix_level")).get("level") or "")
    if amp != "red":
        signal["red_context_chips"] = []
        signal["red_context_html"] = ""
        return signal
    fields = chip_fields_from_row(signal)
    signal.update(fields)
    signal["red_context_html"] = red_context_chips_html(fields["red_context_chips"])
    return signal
