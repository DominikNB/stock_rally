"""
Konfiguration und Hilfsfunktionen für Spaltennamen.

Alle Pipeline-Zustände (DataFrames, Modelle, ``best_params``, …) werden zur Laufzeit
als Attribute dieses Moduls gesetzt (gemeinsamer Namespace für alle Schritte).
``save_scoring_artifacts`` / ``load_scoring_artifacts`` delegieren an ``lib.scoring_persist``
und verwenden ``globals()`` dieses Moduls, solange die Pipeline über dieses Paket läuft.

Einstellbare Konstanten (Kalender, Scoring, Optuna, FIXED_Y, News, …) stehen in
``config_settings.py`` und werden mit ``from .config_settings import *`` in diesen
Modul-Namespace geladen — dort anpassen, nicht hier duplizieren.

Datei-Gliederung (Lesereihenfolge):
  1. GCP-Auth (optional), dann Import der Settings
  2. Scoring-Artefakt (save/load, Phase-17-Logik)
  3. Pipeline-Hilfen (Zielregeln, Optuna-Getter)
  4. News: BigQuery/GKG, Kanäle, Geo, GCAM, GDELT-API-Fallback
  5. Universum: TICKERS_BY_SECTOR, COMPANY_NAMES
  6. Hilfsfunktionen (News-/Technik-Spalten, Rename-Map)
"""

import datetime
import functools
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path.cwd().resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Live-Terminal-Logs (Windows/IDE): sonst bleiben viele print-Ausgaben blockgepuffert.
# Zusätzlich Encoding auf UTF-8 stellen, damit Sonderzeichen (→, —, …) auf Windows
# (cp1252) nicht zu UnicodeEncodeError im News-Code-Pfad führen.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, OSError, ValueError):
        try:
            _stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass

# ── Google Cloud / BigQuery: ADC (optional, vor BQ_PROJECT_ID) ───────────
def _default_application_default_credentials_path() -> Path | None:
    """Typische gcloud-ADC-Pfade (Windows: %APPDATA%\\gcloud\\…, sonst ~/.config/gcloud/…)."""
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        p = Path(appdata) / "gcloud" / "application_default_credentials.json"
        if p.is_file():
            return p
    for candidate in (Path.home() / ".config" / "gcloud" / "application_default_credentials.json",):
        if candidate.is_file():
            return candidate
    return None


_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
if _cred:
    if os.path.isfile(_cred):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _cred
        print(f"Credentials: GOOGLE_APPLICATION_CREDENTIALS -> {_cred}")
    else:
        print(
            "WARNUNG: GOOGLE_APPLICATION_CREDENTIALS gesetzt, Datei fehlt —",
            _cred,
        )
else:
    _adc = _default_application_default_credentials_path()
    if _adc is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)
        print(f"Credentials: ADC -> {_adc}")
    else:
        print(
            "Hinweis: Keine ADC-JSON gefunden — für BigQuery: gcloud auth application-default login "
            "oder GOOGLE_APPLICATION_CREDENTIALS setzen."
        )


from .config_settings import *

import lib.scoring_persist as scoring_persist


def save_scoring_artifacts(path=None):
    """Speichert Base-Modelle, Meta-Classifier, kalibrierten Threshold und Metadaten."""
    p = Path(path) if path is not None else None
    out = scoring_persist.save_scoring_artifacts(globals(), p)
    globals()["_SCORING_ARTIFACT_SAVED_THIS_SESSION"] = True
    return out


def load_scoring_artifacts(path=None):
    """Lädt Artefakt; setzt u.a. base_models, meta_clf, best_threshold, FEAT_COLS, topk_idx."""
    p = Path(path) if path is not None else None
    r = scoring_persist.load_scoring_artifacts(globals(), p)
    globals()["_SCORING_ARTIFACT_SAVED_THIS_SESSION"] = True
    return r


@functools.cache
def log_pipeline_mode_banner() -> None:
    """Schreibt einmal pro Prozess, ob nur gescored wird (``SCORING_ONLY``) oder trainiert wird."""
    sc = bool(globals().get("SCORING_ONLY", False))
    meta_only = bool(globals().get("RETRAIN_META_ONLY", False))
    if sc:
        print(
            "Pipeline-Modus: SCORING_ONLY — nur Scoring/Export "
            "(Phasen 12–16 Training übersprungen; Artefakt wird geladen).",
            flush=True,
        )
    elif meta_only:
        print(
            "Pipeline-Modus: RETRAIN_META_ONLY — Base-Training übersprungen; "
            "lade Base-Artefakt und starte direkt bei Meta (Phasen 13–17).",
            flush=True,
        )
    else:
        print(
            "Pipeline-Modus: volles Training — Optuna/Meta/Kalibrierung (12–16), danach Scoring/Export.",
            flush=True,
        )


def opt_optimize_y_targets() -> bool:
    """True = Optuna sucht Rally-/Label-Parameter (Modulattribut ``OPT_OPTIMIZE_Y_TARGETS``)."""
    return bool(OPT_OPTIMIZE_Y_TARGETS)


def y_label_rule() -> str:
    """Bei festem Y: ``fixed_bands`` oder ``cross_sectional_top_q`` (Modulattribut ``Y_LABEL_RULE``)."""
    m = sys.modules[__name__]
    return str(getattr(m, "Y_LABEL_RULE", "fixed_bands")).strip().lower()


def fixed_y_rule_params() -> tuple[int, int, float, int, int, int, int]:
    """(w_min, w_max, rally_threshold, segment_split, lead_days, entry_days, tail_exclude_days).

    Wie ``target._create_target_one_ticker_fixed_bands``. Nur Modulattribute ``FIXED_Y_*``.
    """
    m = sys.modules[__name__]
    w_lo = int(getattr(m, "FIXED_Y_WINDOW_MIN", 3))
    w_hi = int(getattr(m, "FIXED_Y_WINDOW_MAX", 10))
    rt = float(getattr(m, "FIXED_Y_RALLY_THRESHOLD", 0.06))
    split = int(getattr(m, "FIXED_Y_SEGMENT_SPLIT", 5))
    ld = int(getattr(m, "FIXED_Y_LEAD_DAYS", 3))
    ed = int(getattr(m, "FIXED_Y_ENTRY_DAYS", 2))
    tail_ex = int(getattr(m, "FIXED_Y_TAIL_EXCLUDE_DAYS", 3))
    return w_lo, w_hi, rt, split, ld, ed, tail_ex


def fixed_y_require_strict_daily_up_in_rally() -> bool:
    """True: feste Band-Regel lässt nur Fenster mit strikt positiven Tages-Returns zu."""
    m = sys.modules[__name__]
    return bool(getattr(m, "FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY", False))


def fixed_y_max_dip_below_entry_fraction() -> float:
    """Zulässiger Anteil f unter dem Entry-Open-Preis: Close >= open * (1-f) bis Exit. f∈[0,1] nach Clamping."""
    m = sys.modules[__name__]
    v = getattr(m, "FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION", None)
    if v is not None:
        return float(v)
    # Legacy-Notebooks mit altem bool
    if bool(getattr(m, "FIXED_Y_REQUIRE_NO_DIP_BELOW_ENTRY_UNTIL_THRESHOLD", False)):
        return 0.0
    return 1.0


def fixed_y_long_segment_label_params() -> tuple[str, int]:
    """(mode, long_entry_days) für L >= FIXED_Y_SEGMENT_SPLIT."""
    m = sys.modules[__name__]
    mode = str(getattr(m, "FIXED_Y_LONG_SEGMENT_LABEL_MODE", "tail_exclude")).strip().lower()
    if mode not in {"tail_exclude", "early_only"}:
        mode = "tail_exclude"
    long_ed = int(getattr(m, "FIXED_Y_LONG_ENTRY_DAYS", 2))
    return mode, max(0, long_ed)


def fixed_y_label_mode() -> str:
    """Label-Modus der festen Band-Regel: segment_based, entry_direct oder rally_plus_entry."""
    m = sys.modules[__name__]
    mode = str(getattr(m, "FIXED_Y_LABEL_MODE", "segment_based")).strip().lower()
    if mode not in {"segment_based", "entry_direct", "rally_plus_entry"}:
        mode = "segment_based"
    return mode


def describe_target_rule_fixed_bands() -> str:
    """Feste Band-Regel (OPT_OPTIMIZE_Y_TARGETS=False); Zahlen = ``fixed_y_rule_params()`` wie in der Pipeline."""
    w_lo, w_hi, rt, split, ld, ed, tail_ex = fixed_y_rule_params()
    strict_up = fixed_y_require_strict_daily_up_in_rally()
    mdf = max(0.0, min(float(fixed_y_max_dip_below_entry_fraction()), 1.0))
    label_mode = fixed_y_label_mode()
    long_mode, long_ed = fixed_y_long_segment_label_params()
    strict_txt = " Ja (jeder Tages-Return im Rally-Fenster muss > 0 sein)." if strict_up else " Nein."
    if mdf >= 1.0 - 1e-12:
        no_dip_txt = " f=1.0: kein Mindest-Preis-Floor unter Entry-Open (nur Close ≥ 0 sinnvoll)."
    elif mdf <= 1e-12:
        no_dip_txt = " f=0.0: jeder Close auf dem Haltetags-Pfad muss mindestens dem Entry-Open entsprechen."
    else:
        no_dip_txt = (
            f" f={mdf:.4f} ({mdf:.2%}): jeder Close von Entry bis Exit muss mindestens "
            f"open_entry·(1−f) = open_entry·{1.0 - mdf:.6f} sein."
        )
    long_txt = (
        f"bei L ≥ {split} nur die ersten {long_ed} grünen Tage (early_only)."
        if long_mode == "early_only"
        else f"bei L ≥ {split} alle grünen Tage außer den letzten {tail_ex} (tail_exclude)."
    )
    if label_mode == "entry_direct":
        return (
            "Zieldefinition (feste Band-Regel — dieselben Konstanten wie in der Pipeline):\n\n"
            "(1) Grüne Rally-Phase: Ein Handelstag ist „grün“, wenn es mindestens ein Fenster "
            "mit Länge w ∈ [{w_lo}, {w_hi}] Handelstagen gibt, über das die kumulierte Rendite "
            "(Produkt der Tagesrenditen) mindestens {rt_str} beträgt; alle Tage solcher Fenster werden "
            "zu grünen Segmenten vereinigt.\n"
            "Zusatz-Constraint „strict daily up“ aktiv?{strict_txt}\n\n"
            "Dip-Constraint (max. Anteil unter Entry-Open, Close ≥ open·(1−f)): {no_dip_txt}\n\n"
            "(2) Positives Klassenlabel (entry_direct): target[t0]=1 genau dann, wenn ein Entry "
            "am nächsten Open (t1=t0+1) mindestens ein Fenster w ∈ [{w_lo}, {w_hi}] mit "
            "Trade-Return close[t1+w]/open[t1]-1 >= {rt_str} erreicht."
        ).format(
            w_lo=w_lo,
            w_hi=w_hi,
            rt_str=f"{rt:.2%}",
            strict_txt=strict_txt,
            no_dip_txt=no_dip_txt,
        )
    return (
        "Zieldefinition (feste Band-Regel — dieselben Konstanten wie in der Pipeline):\n\n"
        "(1) Grüne Rally-Phase: Ein Handelstag ist „grün“, wenn es mindestens ein Fenster "
        "mit Länge w ∈ [{w_lo}, {w_hi}] Handelstagen gibt, über das die kumulierte Rendite "
        "(Produkt der Tagesrenditen) mindestens {rt_str} beträgt; alle Tage solcher Fenster werden "
        "zu grünen Segmenten vereinigt.\n"
        "Zusatz-Constraint „strict daily up“ aktiv?{strict_txt}\n\n"
        "Dip-Constraint (max. Anteil unter Entry-Open, Close ≥ open·(1−f)): {no_dip_txt}\n\n"
        "(2) Positives Klassenlabel: Am ersten Tag eines neuen grünen Segments sei L die "
        "Segmentlänge in Handelstagen. Label 1 erhalten die {ld} Handelstage unmittelbar vor "
        "Segmentbeginn; bei L < {split} zusätzlich die ersten {ed} grünen Tage des Segments; "
        "{long_txt}"
    ).format(
        w_lo=w_lo,
        w_hi=w_hi,
        rt_str=f"{rt:.2%}",
        split=split,
        ld=ld,
        ed=ed,
        tail_ex=tail_ex,
        long_txt=long_txt,
        strict_txt=strict_txt,
        no_dip_txt=no_dip_txt,
    )


def describe_fixed_y_target_rule() -> str:
    """Alias für ``describe_target_rule_fixed_bands`` (Logs, Optuna)."""
    return describe_target_rule_fixed_bands()


def resolved_opt_y_label_params(c=None) -> dict[str, int | float]:
    """Rally-/Label-Parameter wie beim Training: aus ``c.best_params``, sonst Modul-Defaults."""
    m = sys.modules[__name__]
    bp = getattr(c, "best_params", None) or {} if c is not None else {}
    return {
        "return_window": int(bp.get("return_window", getattr(m, "RETURN_WINDOW", 4))),
        "rally_threshold": float(bp.get("rally_threshold", getattr(m, "RALLY_THRESHOLD", 0.06))),
        "lead_days": int(bp.get("lead_days", getattr(m, "LEAD_DAYS", 3))),
        "entry_days": int(bp.get("entry_days", getattr(m, "ENTRY_DAYS", 2))),
        "min_rally_tail_days": int(
            bp.get("min_rally_tail_days", getattr(m, "MIN_RALLY_TAIL_DAYS", 5))
        ),
    }


def describe_target_rule_opt_y_from_params(p: dict[str, int | float]) -> str:
    """Parametrische Regel (OPT_OPTIMIZE_Y_TARGETS=True); Werte = tatsächlich genutzte Parameter."""
    rw = int(p["return_window"])
    rt = float(p["rally_threshold"])
    ld = int(p["lead_days"])
    ed = int(p["entry_days"])
    mt = int(p["min_rally_tail_days"])
    rt_str = f"{rt:.2%}"
    return (
        "Zieldefinition (parametrische Rally-/Label-Regel — dieselbe Logik wie "
        "``target._create_target_one_ticker``; Werte = tatsächlich verwendete Parameter):\n\n"
        "(1) Rally: An einem Tag endet ein Fenster von genau {rw} aufeinanderfolgenden Handelstagen, "
        "über das die kumulierte Rendite (Produkt der Tagesrenditen) mindestens {rt_str} beträgt; "
        "dann markiert dieses Fenster eine Rally-Phase, und alle Tage darin werden zur "
        "zusammenhängenden grünen Phase vereinigt (wie im Code über das Ende des Fensters iteriert).\n\n"
        "(2) Positives Klassenlabel: Am ersten Handelstag eines neuen Rally-Segments (nach Rally-Pause) "
        "sei das Segment [start … end]. Nur wenn die Segmentlänge mindestens {mt} Rally-Tage beträgt, "
        "erhalten die bis zu {ld} Handelstage unmittelbar vor „start“ Label 1 (Vorlauf). "
        "Innerhalb des Segments erhalten höchstens die ersten {ed} Tage ab „start“ Label 1, "
        "jeweils nur wenn vom jeweiligen Tag bis zum Segmentende noch mindestens {mt} Rally-Tage "
        "übrig sind (kein positives Label am Rally-Ende).\n\n"
        "Konkret verwendet: return_window={rw}, rally_threshold={rt_str}, "
        "lead_days={ld}, entry_days={ed}, min_rally_tail_days={mt}."
    ).format(
        rw=rw,
        rt_str=rt_str,
        ld=ld,
        ed=ed,
        mt=mt,
    )


def _target_rule_use_opt_y(c=None) -> bool:
    if c is not None and getattr(c, "OPT_OPTIMIZE_Y_TARGETS", None) is not None:
        return bool(getattr(c, "OPT_OPTIMIZE_Y_TARGETS"))
    return opt_optimize_y_targets()


def describe_target_rule_text(c=None) -> str:
    """
    Beschreibung der Label-Regel so, wie sie zur Laufzeit gilt: feste Bände oder Werte aus ``best_params``.
    Für die Website ``c`` übergeben (Namespace mit ``best_params``, ``OPT_OPTIMIZE_Y_TARGETS``).
    """
    if not _target_rule_use_opt_y(c):
        return describe_target_rule_fixed_bands()
    return describe_target_rule_opt_y_from_params(resolved_opt_y_label_params(c))


def describe_target_rule_narrative_for_website(c=None) -> str:
    """
    Verständliche Fließtexte für docs/index.html: was „positives Label“ bedeutet und wie es gebaut wird;
    Parameter nur eingebettet, kein reiner Parameterblock.
    """
    if not _target_rule_use_opt_y(c):
        return _describe_target_rule_fixed_bands_narrative()
    return _describe_target_rule_opt_y_narrative(resolved_opt_y_label_params(c))


def _describe_target_rule_fixed_bands_narrative() -> str:
    w_lo, w_hi, rt, split, ld, ed, tail_ex = fixed_y_rule_params()
    strict_up = fixed_y_require_strict_daily_up_in_rally()
    mdf = max(0.0, min(float(fixed_y_max_dip_below_entry_fraction()), 1.0))
    label_mode = fixed_y_label_mode()
    rt_str = f"{rt:.2%}"
    strict_sentence = (
        " Zusätzlich ist ein strikter Tagesanstiegs-Constraint aktiv: Ein Rally-Fenster zählt nur, "
        "wenn jeder enthaltene Tages-Return > 0 ist (keine Tagesrückgänge innerhalb des Fensters)."
        if strict_up
        else ""
    )
    if mdf >= 1.0 - 1e-12:
        no_dip_sentence = ""
    elif mdf <= 1e-12:
        no_dip_sentence = (
            " Zusätzlich darf der Schlusskurs auf dem Haltetags-Pfad bis zum qualifizierenden Exit "
            "nie unter den Einstiegskurs (Entry-Open) fallen."
        )
    else:
        no_dip_sentence = (
            f" Zusätzlich muss jeder Schlusskurs auf dem Haltetags-Pfad mindestens "
            f"open_entry·(1−{mdf:.4f}) betragen (höchstens {mdf:.2%} unter dem Entry-Open)."
        )
    if label_mode == "entry_direct":
        return (
            "Was das Modell vorhersagen soll: Das positive Label markiert konkrete Einstiegstage. "
            "Ein Tag t0 erhält target=1 genau dann, wenn ein Einstieg am nächsten Open (t1=t0+1) "
            "in mindestens einem Fenster aus {w_lo} bis {w_hi} Handelstagen eine Rendite von "
            "mindestens {rt_str} erreicht.{strict_sentence}{no_dip_sentence}\n\n"
            "Unabhängig davon wird weiterhin die grüne Rally-Spur als Vereinigungsmenge aller "
            "qualifizierenden Haltetage geführt (für Diagnose/Visualisierung). Die Klassenzuordnung "
            "selbst ist im entry_direct-Modus jedoch nicht mehr segmentbasiert, sondern direkt "
            "an die Entry-Möglichkeit gekoppelt."
        ).format(
            w_lo=w_lo,
            w_hi=w_hi,
            rt_str=rt_str,
            strict_sentence=strict_sentence,
            no_dip_sentence=no_dip_sentence,
        )
    return (
        "Was das Modell vorhersagen soll: Das Klassenlabel „positiv“ markiert Tage, an denen sich ein "
        "Einstieg vor oder am Anfang einer späteren Kursaufwertsphase lohnt — nicht die gesamte Rally, "
        "sondern Vorlauf und frühe grüne Tage. Die Regel ist fest (ohne Optuna für Rally-Parameter) "
        "und entspricht exakt der Pipeline.\n\n"
        "Zuerst wird festgelegt, wann der Kurs in einer „grünen“ Rally-Phase steht: Ein Tag zählt dazu, "
        "wenn es mindestens ein Fenster aus {w_lo} bis {w_hi} aufeinanderfolgenden Handelstagen gibt, "
        "über das die kumulierte Rendite (Produkt der Tagesrenditen) mindestens {rt_str} beträgt. "
        "Alle Tage, die in ein solches Fenster fallen, werden zu zusammenhängenden grünen Segmenten "
        "vereinigt — so entsteht die Rally-Spur.{strict_sentence}{no_dip_sentence}\n\n"
        "Daraus werden die Trainings-Labels abgeleitet: Betrachtet wird jeweils der erste Tag "
        "eines neuen grünen Segments (nach einer Rally-Pause). Seine Länge sei L Handelstage. "
        "Dann erhalten die {ld} Handelstage unmittelbar vor Segmentbeginn das positive Label; "
        "zusätzlich — je nach Länge — entweder nur die ersten {ed} grünen Tage des Segments "
        "(wenn L kleiner als {split} ist), oder alle grünen Tage des Segments außer "
        "den letzten {tail_ex} (wenn L mindestens {split} beträgt). So werden frühe Einstiege "
        "bevorzugt und sehr späte Rally-Tage nicht als Trainingsziel geführt."
    ).format(
        w_lo=w_lo,
        w_hi=w_hi,
        rt_str=rt_str,
        split=split,
        ld=ld,
        ed=ed,
        tail_ex=tail_ex,
        strict_sentence=strict_sentence,
        no_dip_sentence=no_dip_sentence,
    )


def _describe_target_rule_opt_y_narrative(p: dict[str, int | float]) -> str:
    rw = int(p["return_window"])
    rt = float(p["rally_threshold"])
    ld = int(p["lead_days"])
    ed = int(p["entry_days"])
    mt = int(p["min_rally_tail_days"])
    rt_str = f"{rt:.2%}"
    return (
        "Was das Modell vorhersagen soll: Das positive Label steht für Tage, an denen ein Einstieg "
        "in eine beginnende oder anhaltende Rally sinnvoll erscheint — mit Vorlauf vor dem Rally-Start "
        "und einer begrenzten Einstiegszone am Anfang der grünen Phase. Die Schwellen stammen aus Optuna "
        "(ein Satz gewählter Parameter) und werden beim Training und in diesem Artefakt genau so verwendet.\n\n"
        "Die Rally wird so definiert: Liegt an einem Tag die kumulierte Rendite über genau {rw} "
        "aufeinanderfolgende Handelstage mindestens bei {rt_str}, markiert dieses Fenster grüne Tage; "
        "alle solchen Fenster werden zu durchgehenden Rally-Segmenten zusammengezogen.\n\n"
        "Das positive Klassenlabel setzt nur dann, wenn das Rally-Segment lang genug ist: "
        "Mindestens {mt} Rally-Handelstage müssen im Segment noch „übrig“ sein, damit Vorlauf und "
        "Einstieg nicht am Ende einer kurzen Erholung hängen. Konkret: Bis zu {ld} Handelstage "
        "vor dem ersten grünen Tag eines neuen Segments erhalten Label 1; innerhalb des Segments "
        "höchstens die ersten {ed} Tage — aber jeweils nur, wenn vom jeweiligen Tag bis zum "
        "Rally-Ende noch mindestens {mt} grüne Tage folgen. So werden Späteinsteige am Rally-Ende "
        "vermieden. Die genannten Werte sind die optimierten Parameter (Rally-Fenster, Schwelle, "
        "Vorlauf, Einstiegs-Tage, Mindest-Rally-Rest) und entsprechen den im Artefakt gespeicherten "
        "Einträgen return_window, rally_threshold, lead_days, entry_days und min_rally_tail_days."
    ).format(
        rw=rw,
        rt_str=rt_str,
        ld=ld,
        ed=ed,
        mt=mt,
    )


def html_target_definition_section(c=None) -> str:
    """HTML-Block für docs/index.html: Zieldefinition (Trainings-Labels), sicher escaped."""
    import html as html_module

    raw = describe_target_rule_narrative_for_website(c)
    inner = "".join(
        f"<p>{html_module.escape(b.strip())}</p>"
        for b in raw.split("\n\n")
        if b.strip()
    )
    return (
        '<div class="section target-def"><h2>Zieldefinition (Trainings-Labels)</h2>'
        f'<div class="target-def-body">{inner}</div></div>'
    )



def gcam_series_colname(key: str) -> str:
    """Spaltenname in sentiment_df / BQ-Alias-Suffix: gcam_c12_1, gcam_vnt_success."""
    k = str(key).strip().replace(".", "_").replace("-", "_").replace(":", "_")
    return "gcam_" + k


def _gkg_gcam_keys_clean() -> tuple[str, ...]:
    return tuple(str(k).strip() for k in (GKG_GCAM_METRIC_KEYS or ()) if str(k).strip())


def _gkg_gcam_must_have_keys_clean() -> tuple[str, ...]:
    return tuple(str(k).strip() for k in (GKG_GCAM_MUST_HAVE_KEYS or ()) if str(k).strip())


# Makro: meist weltweit. Sektor-Geo: nur Events-Tabellen (ActionGeo_*), nicht GKG.
# AUTO: Wenn ≥ BQ_SECTOR_GEO_MAJORITY_THRESHOLD der Ticker derselben Region (EU/US-Heuristik
# über Kürzel/Suffix) zugeordnet sind → nur diese Region; sonst kein Sektor-Geo-Filter.
# Bei AUTO=False: fest BQ_SECTOR_GEO_COUNTRIES für alle Sektoren (Legacy).
BQ_MACRO_GEO_COUNTRIES = None
BQ_SECTOR_USE_GEO_FILTER = True
BQ_SECTOR_GEO_AUTO = True
BQ_SECTOR_GEO_MAJORITY_THRESHOLD = 0.5
BQ_SECTOR_GEO_COUNTRIES = ["EU", "US"]
# Optional: Ticker ohne passendes Suffix manuell — Werte 'EU' oder 'US'.
TICKER_BQ_GEO_OVERRIDES = {}
GDELT_REQUEST_DELAY_SEC = 6.0  # nur gdelt_api: ~1 Request / 5s
NEWS_GDELT_SECONDS_PER_CALL = 6.0  # nur gdelt_api: Throttle + Zeit-Schätzung

# Legacy GDELT-Doc-API (nicht BigQuery): Suchstrings pro Kanal
MACRO_NEWS_QUERY = "heat pump OR HVAC OR refrigerant OR air conditioning"

_SECTOR_KEYWORDS_PRIMARY = {
    "technology": ["semiconductor", "AI chip", "cloud computing", "software"],
    "financial_services": ["interest rate", "bank earnings", "Fed", "credit"],
    "healthcare": ["FDA approval", "clinical trial", "pharma", "biotech"],
    "consumer_cyclical": ["consumer spending", "retail sales", "inflation", "automotive industry", "tariff"],
    "industrials": ["manufacturing PMI", "industrial output", "capex", "HVAC"],
    "energy": ["oil price", "crude", "OPEC", "natural gas"],
    "basic_materials": ["chemical industry", "steel", "commodities"],
    "real_estate": ["real estate", "mortgage rates", "housing market"],
    "communication_services": ["telecom", "5G", "broadband", "streaming", "advertising"],
    "crypto": ["Bitcoin", "Ethereum", "crypto regulation", "DeFi"],
    "consumer_defensive": ["consumer staples", "food prices", "dividend", "household products"],
    "utilities": ["electric utility", "power grid", "regulated utility", "natural gas utility"],
}
SECTOR_KEYWORDS = dict(_SECTOR_KEYWORDS_PRIMARY)

# BigQuery GKG: grober V2Themes-Filter pro Kanal (Makro + Sektor).
# Wird von news.py in jeder Tagesaggregation mitbenutzt — nicht durch GCAM/GKG_THEME_SQL_TRIPLES
# „ersetzt“; Tripel/GCAM/Anker-Orgs sind zusätzliche Schichten auf derselben Pipeline.
# Bei load_scoring_artifacts() kann SECTOR_BQ_THEME_WHERE aus dem Artefakt-Manifest überschrieben werden
# (Trainingsstand) — dann ist dieser Block in der Datei nur Default vor dem Laden.
MACRO_BQ_THEME_WHERE = (
    "(V2Themes LIKE '%ECON_INTERESTRATES%' OR V2Themes LIKE '%ECON_INFLATION%' "
    "OR V2Themes LIKE '%MONETARY%' OR V2Themes LIKE '%ECON_CENTRALBANK%' "
    "OR V2Themes LIKE '%ECON_COST_OF_BORROWING%' OR V2Themes LIKE '%ECON_DEBT%')"
)

_SECTOR_BQ_THEME_WHERE_PRIMARY = {
    "technology": (
        "(V2Themes LIKE '%TECH%' OR V2Themes LIKE '%AI_%' OR V2Themes LIKE '%SEMICONDUCTOR%' "
        "OR V2Themes LIKE '%ECON_ENTREPRENEURSHIP%' OR V2Themes LIKE '%ECON_TAXES_CORPORATE%')"
    ),
    "financial_services": (
        "(V2Themes LIKE '%ECON_%' OR V2Themes LIKE '%FINANCE%' OR V2Themes LIKE '%BANK%' "
        "OR V2Themes LIKE '%MONETARY%' OR V2Themes LIKE '%INFLATION%')"
    ),
    "healthcare": (
        "(V2Themes LIKE '%HEALTH%' OR V2Themes LIKE '%MEDICAL%' OR V2Themes LIKE '%FDA%' "
        "OR V2Themes LIKE '%PHARMACEUTICAL%' OR V2Themes LIKE '%CLINICAL_TRIAL%')"
    ),
    "consumer_cyclical": (
        "(V2Themes LIKE '%CONSUMER%' OR V2Themes LIKE '%RETAIL%' OR V2Themes LIKE '%AUTO%' "
        "OR V2Themes LIKE '%VEHICLE%' OR V2Themes LIKE '%AUTOMOTIVE%' "
        "OR V2Themes LIKE '%ECON_AUTOS%' OR V2Themes LIKE '%TRADE_DISPUTE%')"
    ),
    "industrials": (
        "(V2Themes LIKE '%MANUFACTURING%' OR V2Themes LIKE '%ECON_%' OR V2Themes LIKE '%INDUSTRY%' "
        "OR V2Themes LIKE '%ENV_GREEN_ENERGY%' OR V2Themes LIKE '%ECON_SUBSIDIES%')"
    ),
    "energy": (
        "(V2Themes LIKE '%ENERGY%' OR V2Themes LIKE '%OIL%' OR V2Themes LIKE '%GAS%' "
        "OR V2Themes LIKE '%PETROLEUM%' OR V2Themes LIKE '%OPEC%')"
    ),
    "basic_materials": (
        "(V2Themes LIKE '%STEEL%' OR V2Themes LIKE '%COMMODITY%' OR V2Themes LIKE '%CHEMICAL%')"
    ),
    "real_estate": (
        "(V2Themes LIKE '%REALESTATE%' OR V2Themes LIKE '%HOUSING%' OR V2Themes LIKE '%MORTGAGE%' "
        "OR V2Themes LIKE '%PROPERTY%')"
    ),
    "communication_services": (
        "(V2Themes LIKE '%TELECOM%' OR V2Themes LIKE '%5G%' OR V2Themes LIKE '%BROADBAND%' "
        "OR V2Themes LIKE '%MEDIA%' OR V2Themes LIKE '%STREAMING%' OR V2Themes LIKE '%ADVERTISING%')"
    ),
    "crypto": (
        "(V2Themes LIKE '%CRYPTO%' OR V2Themes LIKE '%BITCOIN%' OR V2Themes LIKE '%BLOCKCHAIN%' "
        "OR V2Themes LIKE '%ETHEREUM%')"
    ),
    "consumer_defensive": (
        "(V2Themes LIKE '%CONSUMER%' OR V2Themes LIKE '%FOOD%' OR V2Themes LIKE '%RETAIL%' "
        "OR V2Themes LIKE '%GROCERY%' OR V2Themes LIKE '%INFLATION%')"
    ),
    "utilities": (
        "(V2Themes LIKE '%ENERGY%' OR V2Themes LIKE '%ELECTRIC%' OR V2Themes LIKE '%POWER%' "
        "OR V2Themes LIKE '%UTILITY%' OR V2Themes LIKE '%GRID%' OR V2Themes LIKE '%GAS%')"
    ),
}
SECTOR_BQ_THEME_WHERE = dict(_SECTOR_BQ_THEME_WHERE_PRIMARY)

# =============================================================================
# 5. Universum: Ticker, Modell-Cluster / News-Kanäle, Anzeigenamen
# =============================================================================
# Die Keys von TICKERS_BY_SECTOR folgen den Yahoo-nahen, normalisierten Sektor-Keys.
# Für das Modell ist das der primäre Kanal-Schlüssel (News, sector_id, Reports).
# ``gics_sector``/``gics_industry`` kommen weiterhin live aus yfinance in ``assemble_features``.

TICKERS_BY_SECTOR = {
    # ── Technology ─────────────────────────────────────────────────────────
    # DAX: SAP, IFX  |  MDAX: NEM(Nemetschek), BC8  |  SDAX: TMV, COK, YSN, GFT, NA9, AOF, ELG, AIXA, SMHN, SANT
    'technology': ['AAPL', 'MSFT', 'NVDA', 'CRM', 'CSCO', 'INTC', 'IBM', 'ASML',
                   'TSM', 'AMD', 'PANW', 'ADBE', 'STX', 'WDC', 'MU', 'NXPI', 'TER',
                   'LRCX', 'NOW', 'ANET', 'FI',
                   'GOOGL', 'META', 'AVGO', 'ARM',
                   'SAP.DE', 'IFX.DE',
                   'NEM.DE', 'BC8.DE',
                   'TMV.DE', 'COK.DE', 'YSN.DE', 'GFT.DE', 'NA9.DE',
                   'AOF.DE', 'ELG.DE', 'AIXA.DE', 'SMHN.DE', 'KTN.DE'],

    # ── Financial Services ─────────────────────────────────────────────────
    # DAX: DB1, HNR1  |  MDAX: TLX  |  SDAX: MLP, GLJ, PBB, HABA, WUW, DBAG, HYQ
    'financial_services': ['JPM', 'GS', 'AXP', 'V', 'MS', 'MCO', 'SPGI', 'SCHW', 'TROW', 'AMP',
                           'BAC', 'C', 'WFC',
                           'ALV.DE', 'MUV2.DE', 'DBK.DE', 'CBK.DE', 'BNP.PA',
                           'DB1.DE', 'HNR1.DE',
                           'TLX.DE',
                           'MLP.DE', 'GLJ.DE', 'PBB.DE', 'WUW.DE', 'DBAN.DE', 'HYQ.DE'],

    # ── Healthcare ─────────────────────────────────────────────────────────
    # DAX: FME, SHL  |  MDAX: SRT3  |  SDAX: AFX, EUZ, DMP, GXI, DRW3, O2BC, EVT
    'healthcare': ['JNJ', 'UNH', 'MRK', 'AMGN', 'NVS', 'REGN', 'VRTX', 'BIIB', 'GILD', 'ALGN', 'IDXX',
                   'ISRG', 'ILMN', 'LLY', 'ABBV', 'TMO', 'DHR',
                   'BAYN.DE', 'FRE.DE', 'MRK.DE', 'FME.DE', 'SHL.DE',
                   'SRT3.DE',
                   'AFX.DE', 'EUZ.DE', 'DMP.DE', 'GXI.DE', 'DRW3.DE', 'EVT.DE'],

    # ── Consumer Cyclical (inkl. Automotive-Basket) ───────────────────────
    # DAX: BEI  |  SDAX: FIE, HBH, DOU, SZU, HFG, KWS, EVD, DHER
    # DAX Automotive: VOW3, BMW, MBG, CON, PAH3, P911, DTG  |  MDAX: TGR, SHA
    'consumer_cyclical': ['MCD', 'NKE', 'WMT', 'HD', 'DIS',
                          'AMZN', 'NFLX', 'BKNG', 'TSLA', 'TJX', 'ORLY', 'ULTA', 'LULU',
                          'ADS.DE', 'BOSS.DE', 'HEN3.DE', 'PUM.DE', 'ZAL.DE', 'MC.PA',
                          'BEI.DE',
                          'FIE.DE', 'HBH.DE', 'DOU.DE', 'SZU.DE', 'HFG.DE', 'KWS.DE',
                          'EVD.DE', 'DHER.DE',
                          'VOW3.DE', 'BMW.DE', 'MBG.DE', 'CON.DE', 'PAH3.DE', 'P911.DE',
                          'DTG.DE',
                          'TM', 'STLA', 'GM', 'F'],

    # ── Industrials (inkl. HVAC/Heat-Pump-Basket) ─────────────────────────
    # DAX: SIE, RHM, MTX, AIR, DHL, G1A  |  MDAX: KBX, HOC, KGX, JUN3, TKA, HAG, R3NK, LHA, FRA, HLAG, RAA
    # SDAX: DUE, JST, SFQ, NOEJ, VOS, WAC, INH, MUX, STM, KSB, HDD
    'industrials': ['CAT', 'BA', 'HON', 'MMM', 'SHW', 'TRV', 'DOW', 'FDX',
                    'URI', 'DE', 'FAST', 'ODFL', 'ETN',
                    'RTX', 'LMT', 'UNP', 'UPS',
                    'SIE.DE', 'RHM.DE', 'MTX.DE', 'AIR.DE', 'DHL.DE', 'G1A.DE',
                    'KBX.DE', 'HOC.F', 'KGX.DE', 'JUN3.DE', 'TKA.DE', 'HAG.DE',
                    'R3NK.DE', 'LHA.DE', 'FRA.DE', 'HLAG.DE', 'RAA.DE',
                    'DUE.DE', 'JST.DE', 'SFQ.DE', 'NOEJ.DE', 'VOS.DE',
                    'WAC.DE', 'INH.DE', 'MUX.DE', 'STM.DE', 'KSB.DE', 'HDD.DE',
                    'NIBE-B.ST', '6367.T', 'CARR', 'TT', 'JCI', 'LII',
                    'AOS', 'WSO', '6503.T', 'AALB.AS'],

    # ── Energy ─────────────────────────────────────────────────────────────
    # DAX: ENR  |  MDAX: NDX1  |  SDAX: S92, VBK, EKT, PNE3, F3C, VH2
    'energy':     ['CVX', 'XOM', 'BP',
                   'RWE.DE', 'EOAN.DE', 'NEE', 'DUK', 'IBE.MC',
                   'ENR.DE',
                   'NDX1.DE',
                   'S92.DE', 'VBK.DE', 'EKT.DE', 'PNE3.DE', 'F3C.DE', 'VH2.DE'],

    # ── Crypto (nicht Yahoo-GICS) ──────────────────────────────────────────
    'crypto':     ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'],

    # ── Basic Materials ─────────────────────────────────────────────────────
    # DAX: BAS, BNR, HEI, SY1  |  MDAX: EVK, LXS, NDA, SDO, FPE3  |  SDAX: ACT, WAF, BFSA, KCO
    'basic_materials': ['BAS.DE', 'LIN', 'WCH.DE', 'APD', 'LYB', 'BNR.DE', 'NEM', 'RIO',
                        'ALB', 'FMC', 'FCX',
                        'HEI.DE', 'SY1.DE',
                        'EVK.DE', 'LXS.DE', 'NDA.DE', 'SZG.DE', 'FPE3.DE',
                        'ACT.DE', 'WAF.DE', 'BFSA.DE', 'KCO.DE'],

    # ── Real Estate ────────────────────────────────────────────────────────
    # DAX: VNA  |  MDAX: LEG  |  SDAX: GYC, DEQ, HABA, PAT
    'real_estate':['VNA.DE', 'LEG.DE', 'O', 'AMT', 'SPG', 'WDP.BR',
                   'GYC.DE', 'DEQ.DE', 'HABA.DE', 'PAT.DE'],

    # ── Communication Services (Telecom + Media) ───────────────────────────
    # DAX: DTE  |  MDAX: O2D, UTDI, FNTN  |  SDAX: 1U1
    # DAX: G24  |  MDAX: RRTL  |  SDAX: SPG, BVB, PSM, CWC
    'communication_services': ['DTE.DE', 'T', 'VZ', 'TMUS', 'ORA.PA',
                               'TEF.MC', 'UTDI.DE', 'FNTN.DE', '1U1.DE',
                               'RRTL.DE', 'SPG.DE', 'BVB.DE', 'PSM.DE', 'CWC.DE', 'G24.DE'],

    # ── Consumer Defensive / Staples (2026-05 Erweiterung) ─────────────────
    'consumer_defensive': ['KO', 'PG', 'PEP', 'COST', 'CL', 'MDLZ', 'GIS', 'KHC', 'KMB',
                           'MO', 'PM', 'STZ', 'SYY', 'HSY', 'MKC'],

    # ── Utilities (2026-05 Erweiterung) ────────────────────────────────────
    'utilities': ['SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC', 'ES', 'PPL', 'AES',
                  'ENEL.MI'],
}

# Alphabetically sorted sector labels (0-based)
SECTOR_LABELS = {s: i for i, s in enumerate(sorted(TICKERS_BY_SECTOR.keys()))}
# basic_materials=0, communication_services=1, consumer_cyclical=2, consumer_defensive=3,
# crypto=4, energy=5, financial_services=6, healthcare=7, industrials=8, real_estate=9,
# technology=10, utilities=11

TICKER_TO_SECTOR = {t: s for s, tl in TICKERS_BY_SECTOR.items() for t in tl}
ALL_TICKERS = [t for tl in TICKERS_BY_SECTOR.values() for t in tl]


# ──────────────────────────────────────────────────────────────────────────────
# Sektor-Key Normalisierung
# ──────────────────────────────────────────────────────────────────────────────
# Wir mischen drei Quellen für Sektor-Strings:
#  1. ``TICKERS_BY_SECTOR.keys()`` (Internformat): lowercase + underscore
#     → "technology", "consumer_cyclical", "communication_services" …
#  2. yfinance/Yahoo GICS (Anzeigeformat): Title-Case + Leerzeichen
#     → "Technology", "Consumer Cyclical", "Communication Services" …
#  3. Externe Quellen (Reports/Manifeste): gelegentlich Sondervarianten.
#
# News-Cache, BigQuery-Channels und die GCAM/Theme-Konfiguration laufen auf Variante 1.
# Würden wir Variante 2 ungefiltert an einen Merge auf ``["Date", "sector"]`` geben,
# matched NUR ``crypto`` (zufällig identisch). Die Folge: 237 von 243 Equity-Ticker
# bekommen 0 % News-Fill (Diagnose bestätigt). Lösung: vor jedem Merge, jeder
# Persistenz und jeder Sektor-Lookup ``normalize_sector_key`` aufrufen.
_SECTOR_KEY_ALIASES = {
    # Yahoo-Sonderfälle / abweichende Schreibweisen → Internformat.
    "consumer defensive": "consumer_defensive",
    "consumer staples": "consumer_defensive",
    "utilities": "utilities",
    "cryptocurrency": "crypto",
    "crypto-asset": "crypto",
    "telecommunication services": "communication_services",
    "telecom services": "communication_services",
    "telecommunications": "communication_services",
    # Volle Sicherheits-Aliasse (idempotent).
    **{k: k for k in TICKERS_BY_SECTOR.keys()},
}


def normalize_sector_key(value) -> str:
    """yfinance-/Display-Sektor → internes Cache-/Channel-Schema.

    Idempotent: bereits normalisierte Keys (``technology``) bleiben unverändert.
    Unbekannte Sektoren werden auf ``"unknown"`` gemappt, damit ein nachfolgender
    News-Merge sie eindeutig ignoriert (NaN-Block) statt einen Mismatch zu erzeugen.

    Beispiele:
        >>> normalize_sector_key("Consumer Cyclical")
        'consumer_cyclical'
        >>> normalize_sector_key("Technology")
        'technology'
        >>> normalize_sector_key("technology")
        'technology'
        >>> normalize_sector_key(None)
        'unknown'
    """
    if value is None:
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    s_low = s.lower()
    # 1. Alias-Treffer (vor der Umwandlung Leerzeichen→Unterstrich).
    if s_low in _SECTOR_KEY_ALIASES:
        return _SECTOR_KEY_ALIASES[s_low]
    # 2. Standardweg: "Consumer Cyclical" → "consumer_cyclical".
    s_norm = s_low.replace(" ", "_").replace("-", "_")
    if s_norm in TICKERS_BY_SECTOR:
        return s_norm
    if s_norm in _SECTOR_KEY_ALIASES:
        return _SECTOR_KEY_ALIASES[s_norm]
    # 3. Konnte nicht zugeordnet werden — als unknown markieren.
    return "unknown"

# Company display names
COMPANY_NAMES = {
    # Heat pump
    'NIBE-B.ST': 'NIBE', '6367.T': 'Daikin', 'CARR': 'Carrier', 'TT': 'Trane Tech',
    'JCI': 'Johnson Controls', 'LII': 'Lennox', 'AOS': 'A.O. Smith', 'WSO': 'Watsco',
    '6503.T': 'Mitsubishi', 'AALB.AS': 'Aalberts',
    # Tech
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'CRM': 'Salesforce',
    'CSCO': 'Cisco', 'INTC': 'Intel', 'IBM': 'IBM',
    'TSM': 'TSMC', 'AMD': 'AMD', 'PANW': 'Palo Alto Networks',
    'ADBE': 'Adobe', 'STX': 'Seagate', 'WDC': 'Western Digital', 'MU': 'Micron',
    'NXPI': 'NXP Semiconductors', 'TER': 'Teradyne', 'LRCX': 'Lam Research',
    'NOW': 'ServiceNow', 'ANET': 'Arista Networks', 'FI': 'Fiserv',
    'GOOGL': 'Alphabet', 'META': 'Meta Platforms', 'AVGO': 'Broadcom', 'ARM': 'Arm Holdings',
    'SAP.DE': 'SAP', 'IFX.DE': 'Infineon', 'ASML': 'ASML',
    # Finance
    'JPM': 'JPMorgan', 'GS': 'Goldman Sachs', 'AXP': 'AmEx', 'V': 'Visa',
    'MS': 'Morgan Stanley', 'MCO': "Moody's", 'SPGI': 'S&P Global',
    'SCHW': 'Charles Schwab', 'TROW': 'T. Rowe Price', 'AMP': 'Ameriprise',
    'BAC': 'Bank of America', 'C': 'Citigroup', 'WFC': 'Wells Fargo',
    'ALV.DE': 'Allianz', 'MUV2.DE': 'Munich Re', 'DBK.DE': 'Deutsche Bank',
    'CBK.DE': 'Commerzbank', 'BNP.PA': 'BNP Paribas',
    # Healthcare
    'JNJ': 'J&J', 'UNH': 'UnitedHealth', 'MRK': 'Merck US', 'AMGN': 'Amgen',
    'ISRG': 'Intuitive Surgical', 'ILMN': 'Illumina',
    'REGN': 'Regeneron', 'VRTX': 'Vertex', 'BIIB': 'Biogen',
    'GILD': 'Gilead', 'ALGN': 'Align Technology', 'IDXX': 'IDEXX Labs',
    'LLY': 'Eli Lilly', 'ABBV': 'AbbVie', 'TMO': 'Thermo Fisher', 'DHR': 'Danaher',
    'BAYN.DE': 'Bayer', 'FRE.DE': 'Fresenius', 'MRK.DE': 'Merck KGaA',
    'SRT3.DE': 'Sartorius', 'NVS': 'Novartis',
    # Consumer
    'KO': 'Coca-Cola', 'MCD': "McDonald's", 'NKE': 'Nike', 'PG': 'P&G',
    'WMT': 'Walmart', 'HD': 'Home Depot', 'DIS': 'Disney',
    'AMZN': 'Amazon', 'NFLX': 'Netflix', 'BKNG': 'Booking', 'TSLA': 'Tesla',
    'TJX': 'TJX', 'ORLY': "O'Reilly Auto", 'ULTA': 'Ulta Beauty', 'LULU': 'Lululemon',
    'ADS.DE': 'Adidas', 'BOSS.DE': 'Hugo Boss', 'HEN3.DE': 'Henkel',
    'PUM.DE': 'PUMA', 'ZAL.DE': 'Zalando', 'MC.PA': 'LVMH',
    # Industrial (DAX)
    'CAT': 'Caterpillar', 'BA': 'Boeing', 'HON': 'Honeywell', 'MMM': '3M',
    'FDX': 'FedEx',
    'URI': 'United Rentals', 'DE': 'Deere', 'FAST': 'Fastenal',
    'ODFL': 'Old Dominion Freight Line', 'ETN': 'Eaton',
    'RTX': 'RTX', 'LMT': 'Lockheed Martin', 'UNP': 'Union Pacific', 'UPS': 'UPS',
    'SHW': 'Sherwin-Williams', 'TRV': 'Travelers', 'DOW': 'Dow Inc.',
    'SIE.DE': 'Siemens', 'RHM.DE': 'Rheinmetall', 'MTX.DE': 'MTU Aero',
    'AIR.DE': 'Airbus', 'DHL.DE': 'DHL Group', 'G1A.DE': 'GEA Group',
    # Industrial (MDAX)
    'KBX.DE': 'Knorr-Bremse', 'HOC.F': 'Hochtief', 'KGX.DE': 'KION Group',
    'JUN3.DE': 'Jungheinrich', 'TKA.DE': 'Thyssenkrupp', 'HAG.DE': 'Hensoldt',
    'R3NK.DE': 'RENK Group', 'LHA.DE': 'Lufthansa', 'FRA.DE': 'Fraport',
    'HLAG.DE': 'Hapag-Lloyd', 'RAA.DE': 'Rational',
    # Industrial (SDAX)
    'DUE.DE': 'Dürr', 'JST.DE': 'JOST Werke', 'SFQ.DE': 'SAF-Holland',
    'NOEJ.DE': 'NORMA Group', 'VOS.DE': 'Vossloh', 'WAC.DE': 'Wacker Neuson',
    'INH.DE': 'INDUS Holding', 'MUX.DE': 'Mutares', 'STM.DE': 'Stabilus',
    'KSB.DE': 'KSB', 'HDD.DE': 'Heidelberger Druck',
    # Energy
    'CVX': 'Chevron', 'XOM': 'ExxonMobil', 'BP': 'BP',
    'RWE.DE': 'RWE', 'EOAN.DE': 'E.ON', 'NEE': 'NextEra', 'DUK': 'Duke Energy',
    'IBE.MC': 'Iberdrola',
    'ENR.DE': 'Siemens Energy', 'NDX1.DE': 'Nordex',
    'S92.DE': 'SMA Solar', 'VBK.DE': 'Verbio', 'EKT.DE': 'Energiekontor',
    'PNE3.DE': 'PNE', 'F3C.DE': 'SFC Energy', 'VH2.DE': 'Friedrich Vorwerk',
    # Crypto
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'BNB',
    'SOL-USD': 'Solana', 'XRP-USD': 'XRP', 'ADA-USD': 'Cardano',
    # Automotive (DAX)
    'VOW3.DE': 'Volkswagen', 'BMW.DE': 'BMW', 'MBG.DE': 'Mercedes-Benz',
    'CON.DE': 'Continental', 'PAH3.DE': 'Porsche SE', 'P911.DE': 'Porsche AG',
    'DTG.DE': 'Daimler Truck',
    'TM': 'Toyota', 'STLA': 'Stellantis', 'GM': 'General Motors', 'F': 'Ford',
    # Materials (DAX)
    'BAS.DE': 'BASF', 'LIN': 'Linde', 'WCH.DE': 'Wacker Chemie',
    'APD': 'Air Products', 'LYB': 'LyondellBasell', 'BNR.DE': 'Brenntag',
    'NEM': 'Newmont', 'RIO': 'Rio Tinto', 'ALB': 'Albemarle', 'FMC': 'FMC Corp', 'FCX': 'Freeport-McMoRan',
    'HEI.DE': 'Heidelberg Materials', 'SY1.DE': 'Symrise',
    # Materials (MDAX)
    'EVK.DE': 'Evonik', 'LXS.DE': 'LANXESS', 'NDA.DE': 'Aurubis',
    'SZG.DE': 'Salzgitter', 'FPE3.DE': 'Fuchs SE',
    # Materials (SDAX)
    'ACT.DE': 'AlzChem', 'WAF.DE': 'Siltronic', 'BFSA.DE': 'Befesa',
    'KCO.DE': 'Klöckner & Co',
    # Real Estate
    'VNA.DE': 'Vonovia', 'LEG.DE': 'LEG Immobilien', 'O': 'Realty Income',
    'AMT': 'American Tower', 'SPG': 'Simon Property', 'WDP.BR': 'WDP',
    'GYC.DE': 'Grand City Prop.', 'DEQ.DE': 'Deutsche EuroShop',
    'HABA.DE': 'Hamborner REIT', 'PAT.DE': 'PATRIZIA',
    # Telecom
    'DTE.DE': 'Deutsche Telekom', 'T': 'AT&T', 'VZ': 'Verizon',
    'TMUS': 'T-Mobile', 'ORA.PA': 'Orange',
    'TEF.MC': 'Telefonica', 'UTDI.DE': 'United Internet',
    'FNTN.DE': 'freenet', '1U1.DE': '1&1',
    # Media
    'RRTL.DE': 'RTL Group', 'SPG.DE': 'Springer Nature', 'BVB.DE': 'Borussia Dortmund',
    'PSM.DE': 'ProSieben', 'CWC.DE': 'CEWE', 'G24.DE': 'Scout24',
    # Finance (DAX additions)
    'DB1.DE': 'Deutsche Börse', 'HNR1.DE': 'Hannover Rück',
    # Finance (MDAX)
    'TLX.DE': 'Talanx',
    # Finance (SDAX)
    'MLP.DE': 'MLP', 'GLJ.DE': 'GRENKE', 'PBB.DE': 'Deutsche Pfandbriefbank',
    'WUW.DE': 'Wüstenrot & W.',
    'DBAN.DE': 'Deutsche Beteiligungs', 'HYQ.DE': 'Hypoport',
    # Healthcare (DAX additions)
    'FME.DE': 'Fresenius Med.', 'SHL.DE': 'Siemens Healthineers',
    # Healthcare (SDAX)
    'AFX.DE': 'Carl Zeiss Meditec', 'EUZ.DE': 'Eckert & Ziegler',
    'DMP.DE': 'Dermapharm', 'GXI.DE': 'Gerresheimer',
    'DRW3.DE': 'Drägerwerk', 'EVT.DE': 'Evotec',
    # Consumer (DAX additions)
    'BEI.DE': 'Beiersdorf',
    # Consumer (SDAX)
    'FIE.DE': 'Fielmann', 'HBH.DE': 'Hornbach', 'DOU.DE': 'Douglas',
    'SZU.DE': 'Südzucker', 'HFG.DE': 'HelloFresh', 'KWS.DE': 'KWS Saat',
    'EVD.DE': 'CTS Eventim', 'DHER.DE': 'Delivery Hero',
    # Tech (MDAX)
    'NEM.DE': 'Nemetschek', 'BC8.DE': 'Bechtle',
    # Tech (SDAX)
    'TMV.DE': 'TeamViewer', 'COK.DE': 'CANCOM', 'YSN.DE': 'secunet',
    'GFT.DE': 'GFT Technologies', 'NA9.DE': 'Nagarro', 'AOF.DE': 'ATOSS Software',
    'ELG.DE': 'Elmos Semiconductor', 'AIXA.DE': 'Aixtron',
    'SMHN.DE': 'SÜSS MicroTec', 'KTN.DE': 'Kontron',
    # Consumer defensive / staples
    'PEP': 'PepsiCo', 'COST': 'Costco', 'CL': 'Colgate-Palmolive',
    'MDLZ': 'Mondelez', 'GIS': 'General Mills', 'KHC': 'Kraft Heinz',
    'KMB': 'Kimberly-Clark', 'MO': 'Altria', 'PM': 'Philip Morris',
    'STZ': 'Constellation Brands', 'SYY': 'Sysco', 'HSY': "Hershey's", 'MKC': 'McCormick',
    # Utilities
    'SO': 'Southern Company', 'D': 'Dominion Energy', 'AEP': 'American Electric Power',
    'EXC': 'Exelon', 'SRE': 'Sempra', 'XEL': 'Xcel Energy', 'ED': 'Consolidated Edison',
    'WEC': 'WEC Energy', 'ES': 'Eversource', 'PPL': 'PPL', 'AES': 'AES',
    'ENEL.MI': 'Enel',
}

# Rollierende Anker-Ticker (Top-N nach Marktkap): Spillover-Signal, weniger GDELT-Rauschen.
# Zeitplan: JSON mit Vierteljahres-Zeilen (siehe scripts/build_sector_anchor_schedule.py).
# BigQuery nutzt Chunk-Enddatum ``ref_date`` — bei sehr großen Zeitscheiben ggf. Chunk-Dauer < Quartal.
NEWS_ANCHOR_ORG_FILTER = True
NEWS_ANCHOR_SCHEDULE_PATH = Path("data") / "sector_anchor_quarters.json"
NEWS_ANCHOR_TOP_N = 3
NEWS_ANCHOR_FETCH_SLEEP_SEC = 0.05

# =============================================================================
# 6. Hilfsfunktionen (Feature-Spalten, Rename-Map)
# =============================================================================


def sector_news_theme_sql(sector_key: str, ref_date) -> str:
    """Sektor-``V2Themes``-Block; optional zusätzlich ``V2Organizations`` (Ankerliste zum Datum)."""
    import datetime as _dt

    base = str(SECTOR_BQ_THEME_WHERE.get(sector_key) or "(1=0)")
    if not NEWS_ANCHOR_ORG_FILTER:
        return base
    from lib.stock_rally_v10.anchor_tickers import (
        load_anchor_schedule,
        tickers_for_sector_on_date,
        v2organizations_or_clause,
    )

    sched = load_anchor_schedule(NEWS_ANCHOR_SCHEDULE_PATH)
    if not sched:
        return base
    if isinstance(ref_date, _dt.datetime):
        d = ref_date.date()
    elif isinstance(ref_date, _dt.date):
        d = ref_date
    elif isinstance(ref_date, str) and len(ref_date) >= 10:
        d = _dt.date.fromisoformat(ref_date[:10])
    else:
        d = _dt.date.today()
    tks = tickers_for_sector_on_date(sched, sector_key, d)
    if not tks:
        return base
    org = v2organizations_or_clause(tks, COMPANY_NAMES)
    if not org:
        return base
    return f"(({base}) AND ({org}))"


def sector_anchor_org_active(sector_key: str, ref_date) -> bool:
    """True, wenn für den Sektor eine echte V2Organizations-Ankerbedingung aktiv ist."""
    if not NEWS_ANCHOR_ORG_FILTER:
        return False
    base = str(SECTOR_BQ_THEME_WHERE.get(sector_key) or "(1=0)")
    return sector_news_theme_sql(sector_key, ref_date) != base


# Für ``anchor_quality_idx``: Mittelwert dieser Anker-GCAM-Rohwerte als eine „Tone“-Serie.
GCAM_ANCHOR_QUALITY_MEAN_KEYS: tuple[str, ...] = ("c18.159", "c18.158", "c18.161")


def news_feat_tag(mom_w, vol_ma, tone_roll):
    return f"{int(mom_w)}_{int(vol_ma)}_{int(tone_roll)}"


def _news_macro_variant_cols(mid: str):
    return [
        f'news_macro_{mid}_tone',
        f'news_macro_{mid}_vol',
        f'news_macro_{mid}_tone_mom',
        f'news_macro_{mid}_vol_spike',
        f'news_macro_{mid}_tone_l1', f'news_macro_{mid}_tone_l3', f'news_macro_{mid}_tone_l5',
        f'news_macro_{mid}_vol_l1', f'news_macro_{mid}_vol_l3', f'news_macro_{mid}_vol_l5',
        f'news_macro_{mid}_tone_x_log1p_artcount',
        f'news_macro_{mid}_tone_d1',
    ]


def _news_sec_variant_cols(mid: str):
    return [
        f'news_sec_{mid}_tone',
        f'news_sec_{mid}_vol',
        f'news_sec_{mid}_tone_mom',
        f'news_sec_{mid}_vol_spike',
        f'news_sec_{mid}_tone_l1', f'news_sec_{mid}_tone_l3', f'news_sec_{mid}_tone_l5',
        f'news_sec_{mid}_vol_l1', f'news_sec_{mid}_vol_l3', f'news_sec_{mid}_vol_l5',
        f'news_sec_{mid}_tone_x_log1p_artcount',
        f'news_sec_{mid}_tone_d1',
    ]


# Nach Shard-Merge: Macro-/Sektor-Tone × Regime-Spalten (Interaktion „Stress × News“).
NEWS_ADD_REGIME_TONE_INTERACTIONS = True
NEWS_REGIME_INTERACTION_BASE_COLS: tuple[str, ...] = ("mr_vvix_div_vix", "regime_vix_level")
# Nach Shard-Merge: Tone × sign(Tagesrendite) als Bestätigungs-/Widerspruchssignal.
# Liefert ``news_*_{tag}_tone_x_sign_ret_1d`` und ``..._x_sign_ret_lag1`` pro Macro/Sec/Anchor.
# ``NEWS_ADD_SIGN_CONFIRMATION`` ist der Default (Phase 12 / Scoring); Optuna wählt je Trial aus
# ``NEWS_ADD_SIGN_CONFIRMATION_OPTIONS`` (gleiche Mechanik wie ``NEWS_EXTRA_TONE_ACCEL_OPTIONS``).
NEWS_ADD_SIGN_CONFIRMATION = True
NEWS_ADD_SIGN_CONFIRMATION_OPTIONS = [False, True]


def _news_regime_interaction_colnames(tag: str) -> list[str]:
    if not USE_NEWS_SENTIMENT or not NEWS_ADD_REGIME_TONE_INTERACTIONS:
        return []
    out: list[str] = []
    for base in NEWS_REGIME_INTERACTION_BASE_COLS:
        safe = str(base).replace(".", "_")
        out.append(f"news_macro_{tag}_tone_x_{safe}")
        out.append(f"news_sec_{tag}_tone_x_{safe}")
    return out


def _news_sign_confirmation_colnames(
    tag: str, *, enabled: bool | None = None
) -> list[str]:
    """Spaltennamen für Tone × sign(Tagesrendite). ``enabled`` überschreibt den globalen Default."""
    if not USE_NEWS_SENTIMENT:
        return []
    flag = bool(NEWS_ADD_SIGN_CONFIRMATION) if enabled is None else bool(enabled)
    if not flag:
        return []
    out: list[str] = []
    for prefix in ("news_macro", "news_sec", "news_anchor"):
        out.append(f"{prefix}_{tag}_tone_x_sign_ret_1d")
        out.append(f"{prefix}_{tag}_tone_x_sign_ret_lag1")
    return out


def _news_tone_z_derived_colnames(feature_base: str, sw: str) -> list[str]:
    """Aus tone_z/vol_z (gleiches Roll-Fenster sw): Shock vs. 3d-Basis, ΔZ, Z×relu(vol_z)."""
    return [
        f"{feature_base}_tone_z{sw}_shock",
        f"{feature_base}_tone_z{sw}_dz1",
        f"{feature_base}_tone_z{sw}_x_volz_pos",
    ]


def _news_base_cols(tag):
    return _news_macro_variant_cols(tag) + _news_sec_variant_cols(tag)


def _append_gcam_news_cols(out, tag, zscore_w, use_accel):
    """Nur Sektor-GCAM pro Key (kein news_macro_* auf GCAM-Ebene — Macro bleibt V2-Basis)."""
    zw = int(zscore_w) if zscore_w is not None else 0
    for gk in _gkg_gcam_keys_clean():
        mid = f"{tag}_{gcam_series_colname(gk)}"
        out.extend(_news_sec_variant_cols(mid))
        if zw > 0:
            sw = f"_w{zw}"
            out.extend(
                [
                    f"news_sec_{mid}_tone_z{sw}",
                    f"news_sec_{mid}_vol_z{sw}",
                ]
            )
            out.extend(_news_tone_z_derived_colnames(f"news_sec_{mid}", sw))
        if use_accel:
            out.append(f"news_sec_{mid}_tone_accel")


def _append_anchor_gcam_sec_cols(out, tag, zscore_w, use_accel):
    """Nur Sektor: GCAM aus Anker-Zeilen (Theme ∧ Orgs), gleiche Keys wie Sektor-GCAM."""
    zw = int(zscore_w) if zscore_w is not None else 0
    for gk in _gkg_gcam_keys_clean():
        cn = gcam_series_colname(gk)
        mid = f"{tag}_anchor_{cn}"
        out.extend(_news_sec_variant_cols(mid))
        if zw > 0:
            sw = f"_w{zw}"
            out.extend(
                [
                    f"news_sec_{mid}_tone_z{sw}",
                    f"news_sec_{mid}_vol_z{sw}",
                ]
            )
            out.extend(_news_tone_z_derived_colnames(f"news_sec_{mid}", sw))
        if use_accel:
            out.append(f"news_sec_{mid}_tone_accel")


def _append_gcam_div_sec_cols(out, tag, zscore_w, use_accel):
    """Sektor: Anker-GCAM minus breites Sektor-GCAM (Divergenz pro Key)."""
    zw = int(zscore_w) if zscore_w is not None else 0
    for gk in _gkg_gcam_keys_clean():
        cn = gcam_series_colname(gk)
        mid = f"{tag}_div_{cn}"
        out.extend(_news_sec_variant_cols(mid))
        if zw > 0:
            sw = f"_w{zw}"
            out.extend(
                [
                    f"news_sec_{mid}_tone_z{sw}",
                    f"news_sec_{mid}_vol_z{sw}",
                ]
            )
            out.extend(_news_tone_z_derived_colnames(f"news_sec_{mid}", sw))
        if use_accel:
            out.append(f"news_sec_{mid}_tone_accel")


def _append_anchor_quality_sec_cols(out, tag, zscore_w, use_accel):
    """Sektor: Mittelwert der Anker-GCAM-Rohwerte (LMT-Kern) als eine Zeitreihe."""
    zw = int(zscore_w) if zscore_w is not None else 0
    mid = f"{tag}_anchor_quality_idx"
    out.extend(_news_sec_variant_cols(mid))
    if zw > 0:
        sw = f"_w{zw}"
        out.extend(
            [
                f"news_sec_{mid}_tone_z{sw}",
                f"news_sec_{mid}_vol_z{sw}",
            ]
        )
        out.extend(_news_tone_z_derived_colnames(f"news_sec_{mid}", sw))
    if use_accel:
        out.append(f"news_sec_{mid}_tone_accel")


def build_news_model_cols(
    tag, zscore_w=0, use_accel=False, use_cross=False, *,
    add_sign_confirmation: bool | None = None,
):
    """Spaltenliste für ein gewähltes Tag-Tripel; zscore_w=0 schließt tone_z/vol_z aus.

    ``add_sign_confirmation``: optionaler Optuna-Override für die globale
    ``NEWS_ADD_SIGN_CONFIRMATION``-Einstellung (None = Default verwenden).
    """
    out = _news_base_cols(tag)
    zw = int(zscore_w) if zscore_w is not None else 0
    if zw > 0:
        sw = f'_w{zw}'
        out.extend([
            f'news_macro_{tag}_tone_z{sw}', f'news_macro_{tag}_vol_z{sw}',
            f'news_sec_{tag}_tone_z{sw}', f'news_sec_{tag}_vol_z{sw}',
        ])
        out.extend(_news_tone_z_derived_colnames(f"news_macro_{tag}", sw))
        out.extend(_news_tone_z_derived_colnames(f"news_sec_{tag}", sw))
    if use_accel:
        out.extend([f'news_macro_{tag}_tone_accel', f'news_sec_{tag}_tone_accel'])
    if use_cross:
        out.append(f'news_cross_{tag}_macro_minus_sec_tone')
    _append_gcam_news_cols(out, tag, zscore_w, use_accel)
    if NEWS_ANCHOR_ORG_FILTER:
        _append_anchor_gcam_sec_cols(out, tag, zscore_w, use_accel)
        _append_gcam_div_sec_cols(out, tag, zscore_w, use_accel)
        _append_anchor_quality_sec_cols(out, tag, zscore_w, use_accel)
    out.extend(_news_regime_interaction_colnames(tag))
    out.extend(
        _news_sign_confirmation_colnames(tag, enabled=add_sign_confirmation)
    )
    return out


def optuna_training_news_column_union() -> list[str]:
    """Alle ``news_*``-Spalten, die Optuna je Trial höchstens anfasst (Superset wie ``optuna_train``).

    Unterscheidet sich von ``all_news_model_cols()``: dort werden pro Tag u. a. **alle**
    ``NEWS_EXTRA_ZSCORE_WINDOWS`` gleichzeitig in der Namensliste geführt; Optuna wählt aber nur **ein**
    Z-Score-Fenster pro Trial — die Union über das Raster ist schmaler und vermeidet tausende Null-Spalten.
    """
    if not USE_NEWS_SENTIMENT:
        return []
    names: set[str] = set()
    sign_options = NEWS_ADD_SIGN_CONFIRMATION_OPTIONS or [bool(NEWS_ADD_SIGN_CONFIRMATION)]
    for mom in NEWS_MOM_WINDOWS:
        for vma in NEWS_VOL_MA_WINDOWS:
            for tr in NEWS_TONE_ROLL_WINDOWS:
                tag = news_feat_tag(mom, vma, tr)
                for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                    zw = int(z) if z is not None else 0
                    for acc in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        for cr in NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                            for sc in sign_options:
                                for c in build_news_model_cols(
                                    tag,
                                    zscore_w=zw,
                                    use_accel=bool(acc),
                                    use_cross=bool(cr),
                                    add_sign_confirmation=bool(sc),
                                ):
                                    names.add(c)
    return sorted(names)


def all_news_model_cols():
    cols = []
    for m in NEWS_MOM_WINDOWS:
        for v in NEWS_VOL_MA_WINDOWS:
            for r in NEWS_TONE_ROLL_WINDOWS:
                tag = news_feat_tag(m, v, r)
                cols.extend(_news_base_cols(tag))
                for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                    if z > 0:
                        sw = f'_w{int(z)}'
                        cols.extend([
                            f'news_macro_{tag}_tone_z{sw}', f'news_macro_{tag}_vol_z{sw}',
                            f'news_sec_{tag}_tone_z{sw}', f'news_sec_{tag}_vol_z{sw}',
                        ])
                        cols.extend(_news_tone_z_derived_colnames(f"news_macro_{tag}", sw))
                        cols.extend(_news_tone_z_derived_colnames(f"news_sec_{tag}", sw))
                if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                    cols.extend([f'news_macro_{tag}_tone_accel', f'news_sec_{tag}_tone_accel'])
                if True in NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                    cols.append(f'news_cross_{tag}_macro_minus_sec_tone')
                for gk in _gkg_gcam_keys_clean():
                    mid = f"{tag}_{gcam_series_colname(gk)}"
                    cols.extend(_news_sec_variant_cols(mid))
                    for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                        if z > 0:
                            sw = f'_w{int(z)}'
                            cols.extend(
                                [
                                    f'news_sec_{mid}_tone_z{sw}',
                                    f'news_sec_{mid}_vol_z{sw}',
                                ]
                            )
                            cols.extend(_news_tone_z_derived_colnames(f"news_sec_{mid}", sw))
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        cols.append(f'news_sec_{mid}_tone_accel')
                if NEWS_ANCHOR_ORG_FILTER:
                    for gk in _gkg_gcam_keys_clean():
                        cn = gcam_series_colname(gk)
                        mid_a = f"{tag}_anchor_{cn}"
                        cols.extend(_news_sec_variant_cols(mid_a))
                        for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                            if z > 0:
                                sw = f'_w{int(z)}'
                                cols.extend(
                                    [
                                        f'news_sec_{mid_a}_tone_z{sw}',
                                        f'news_sec_{mid_a}_vol_z{sw}',
                                    ]
                                )
                                cols.extend(_news_tone_z_derived_colnames(f"news_sec_{mid_a}", sw))
                        if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                            cols.append(f'news_sec_{mid_a}_tone_accel')
                    for gk in _gkg_gcam_keys_clean():
                        cn = gcam_series_colname(gk)
                        mid_d = f"{tag}_div_{cn}"
                        cols.extend(_news_sec_variant_cols(mid_d))
                        for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                            if z > 0:
                                sw = f'_w{int(z)}'
                                cols.extend(
                                    [
                                        f'news_sec_{mid_d}_tone_z{sw}',
                                        f'news_sec_{mid_d}_vol_z{sw}',
                                    ]
                                )
                                cols.extend(_news_tone_z_derived_colnames(f"news_sec_{mid_d}", sw))
                        if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                            cols.append(f'news_sec_{mid_d}_tone_accel')
                    mid_q = f"{tag}_anchor_quality_idx"
                    cols.extend(_news_sec_variant_cols(mid_q))
                    for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                        if z > 0:
                            sw = f'_w{int(z)}'
                            cols.extend(
                                [
                                    f'news_sec_{mid_q}_tone_z{sw}',
                                    f'news_sec_{mid_q}_vol_z{sw}',
                                ]
                            )
                            cols.extend(_news_tone_z_derived_colnames(f"news_sec_{mid_q}", sw))
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        cols.append(f'news_sec_{mid_q}_tone_accel')
                cols.extend(_news_regime_interaction_colnames(tag))
    return cols

def _default_btc_momentum_z_window():
    w = BTC_MOMENTUM_Z_WINDOWS
    return int(w[len(w) // 2])


def _default_market_breadth_z_window():
    w = MARKET_BREADTH_Z_WINDOWS
    return int(w[len(w) // 2])


def _default_rel_momentum_window():
    w = REL_MOMENTUM_WINDOWS
    return int(w[len(w) // 2])


def _default_adr_window():
    w = ADR_WINDOWS
    return int(w[len(w) // 2])


def _default_breakout_lookback_window():
    w = BREAKOUT_LOOKBACK_WINDOWS
    return int(w[len(w) // 2])


def _default_vcp_window():
    w = VCP_WINDOWS
    return int(w[len(w) // 2])


def _default_btc_corr_window():
    w = BTC_CORR_WINDOWS
    return int(w[len(w) // 2])


def _default_yang_zhang_window():
    w = YANG_ZHANG_WINDOWS or [20]
    return int(w[len(w) // 2])


def _default_downside_vol_window():
    w = DOWNSIDE_VOL_WINDOWS or [60]
    return int(w[len(w) // 2])


def _default_ret_moment_window():
    w = RET_MOMENT_WINDOWS or [60]
    return int(w[len(w) // 2])


def _default_amihud_window():
    w = AMIHUD_WINDOWS or [20]
    return int(w[len(w) // 2])


def _default_vcp_lower_low_window():
    w = VCP_LOWER_LOW_WINDOWS or [60]
    return int(w[len(w) // 2])


def _default_breakout_volume_trigger_z():
    opts = BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS or [1.0]
    return float(opts[len(opts) // 2])


def bvt_z_str(z: float) -> str:
    """Trigger-Z als kurzer, dateinamen-tauglicher String (0.5 → "0p5", 1.0 → "1p0")."""
    return f"{float(z):.1f}".replace('.', 'p')


def build_technical_cols(
    rsi_w,
    bb_w,
    sma_w,
    *,
    btc_momentum_z_window=None,
    market_breadth_z_window=None,
    rel_momentum_window=None,
    adr_window=None,
    breakout_lookback_window=None,
    vcp_window=None,
    btc_corr_window=None,
    yz_vol_window=None,
    downside_vol_window=None,
    ret_moment_window=None,
    amihud_window=None,
    vcp_lower_low_window=None,
    breakout_volume_trigger_z=None,
):
    bz = int(
        btc_momentum_z_window
        if btc_momentum_z_window is not None
        else _default_btc_momentum_z_window()
    )
    brz = int(
        market_breadth_z_window
        if market_breadth_z_window is not None
        else _default_market_breadth_z_window()
    )
    rm = int(
        rel_momentum_window
        if rel_momentum_window is not None
        else _default_rel_momentum_window()
    )
    aw = int(adr_window if adr_window is not None else _default_adr_window())
    bwk = int(
        breakout_lookback_window
        if breakout_lookback_window is not None
        else _default_breakout_lookback_window()
    )
    vw = int(vcp_window if vcp_window is not None else _default_vcp_window())
    bcw = int(btc_corr_window if btc_corr_window is not None else _default_btc_corr_window())
    yz_w = int(
        yz_vol_window if yz_vol_window is not None else _default_yang_zhang_window()
    )
    dv_w = int(
        downside_vol_window
        if downside_vol_window is not None
        else _default_downside_vol_window()
    )
    rm_w = int(
        ret_moment_window
        if ret_moment_window is not None
        else _default_ret_moment_window()
    )
    am_w = int(amihud_window if amihud_window is not None else _default_amihud_window())
    ll_w = int(
        vcp_lower_low_window
        if vcp_lower_low_window is not None
        else _default_vcp_lower_low_window()
    )
    bvt_z_val = float(
        breakout_volume_trigger_z
        if breakout_volume_trigger_z is not None
        else _default_breakout_volume_trigger_z()
    )
    bvt_str = bvt_z_str(bvt_z_val)
    return [
        f'rsi_{rsi_w}',
        f'bb_pband_{bb_w}',
        'macd_diff',
        'vol_stress',
        'drawdown',
        'adx',
        'vol_ratio',
        f'bb_x_rsi_{bb_w}_{rsi_w}',
        f'rsi_delta_3d_{rsi_w}',
        f'bb_slope_5d_{bb_w}',
        f'bb_delta_3d_{bb_w}',
        'adx_delta_3d',
        'momentum_accel',
        f'rsi_weekly_{rsi_w}',
        f'sma_cross_20_{sma_w}',
        'close_vs_sma200',
        'sma200_delta_5d',
        'drawdown_252d',
        f'market_breadth_{sma_w}',
        f'market_breadth_z_{sma_w}_w{brz}',
        'volume_zscore',
        f'sector_avg_rsi_{rsi_w}',
        'btc_momentum',
        f'btc_momentum_z_w{bz}',
        f'corr_stock_btc_{bcw}d',
        f'rel_momentum_{rm}d',
        f'adr_pct_{aw}d',
        f'vcp_tightness_{vw}d',
        f'vcp_tightness_hl_{vw}d',
        f'blue_sky_breakout_{bwk}d',
        f'dist_to_prior_hi_pct_{bwk}d',
        f'volume_at_resistance_{bwk}d',
        'dollar_volume_zscore',
        'volume_force_1d',
        f'bb_squeeze_factor_{bb_w}',
        # Erweiterte Risikofeatures (3b1/3b2/3b5) — Optuna-tunbar (eine Spalte je gewähltem Fenster):
        f'yz_vol_{yz_w}d',
        f'downside_vol_{dv_w}d',
        f'downside_vol_ratio_{dv_w}d',
        f'ret_skew_{rm_w}d',
        f'ret_kurt_{rm_w}d',
        f'amihud_illiquidity_{am_w}d',
        f'vcp_at_period_low_frac_{vw}_{ll_w}d',
        f'vcp_tightness_slope_{vw}_{ll_w}d',
        f'breakout_volume_confirmed_{bwk}_z{bvt_str}d',
        'month',
        'sector_id',
        'gics_sector_id',
        'gics_industry_id',
    ]


def effective_window_grid(name: str) -> list:
    """Optuna-Grid für ``name`` — vom Pre-Screen-Artefakt ggf. eingeschränkt.

    Liest ``cfg._FEATURE_PRESCREEN_ARTIFACT['allowed_windows'][name]`` falls gesetzt
    und nicht leer; sonst Fallback auf ``getattr(cfg, name)``. Reihenfolge der
    Original-Liste wird beibehalten (Optuna-Stabilität).
    """
    art = globals().get("_FEATURE_PRESCREEN_ARTIFACT")
    if isinstance(art, dict):
        allowed = art.get("allowed_windows") or {}
        if isinstance(allowed, dict):
            v = allowed.get(name)
            if v:
                return list(v)
    return list(globals().get(name, []))


def all_model_tech_col_names_for_assemble_dropna() -> list[str]:
    """Alle technischen Modell-Spaltennamen über das volle Optuna-Raster (für dropna nach assemble)."""
    names: set[str] = set()
    yz_grid = YANG_ZHANG_WINDOWS or [_default_yang_zhang_window()]
    dv_grid = DOWNSIDE_VOL_WINDOWS or [_default_downside_vol_window()]
    rm_grid = RET_MOMENT_WINDOWS or [_default_ret_moment_window()]
    am_grid = AMIHUD_WINDOWS or [_default_amihud_window()]
    ll_grid = VCP_LOWER_LOW_WINDOWS or [_default_vcp_lower_low_window()]
    bvt_grid = BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS or [_default_breakout_volume_trigger_z()]
    for rw in RSI_WINDOWS:
        for bw in BB_WINDOWS:
            for sw in SMA_WINDOWS:
                for bz in BTC_MOMENTUM_Z_WINDOWS:
                    for brz in MARKET_BREADTH_Z_WINDOWS:
                        for relm in REL_MOMENTUM_WINDOWS:
                            for aw in ADR_WINDOWS:
                                for bwk in BREAKOUT_LOOKBACK_WINDOWS:
                                    for vw in VCP_WINDOWS:
                                        for bcw in BTC_CORR_WINDOWS:
                                            for yzw in yz_grid:
                                                for dvw in dv_grid:
                                                    for rmw in rm_grid:
                                                        for amw in am_grid:
                                                            for llw in ll_grid:
                                                                for bvt in bvt_grid:
                                                                    for c in build_technical_cols(
                                                                        rw,
                                                                        bw,
                                                                        sw,
                                                                        btc_momentum_z_window=bz,
                                                                        market_breadth_z_window=brz,
                                                                        rel_momentum_window=relm,
                                                                        adr_window=aw,
                                                                        breakout_lookback_window=bwk,
                                                                        vcp_window=vw,
                                                                        btc_corr_window=bcw,
                                                                        yz_vol_window=yzw,
                                                                        downside_vol_window=dvw,
                                                                        ret_moment_window=rmw,
                                                                        amihud_window=amw,
                                                                        vcp_lower_low_window=llw,
                                                                        breakout_volume_trigger_z=bvt,
                                                                    ):
                                                                        names.add(c)
    return sorted(names)


def build_feature_cols(
    rsi_w, bb_w, sma_w,
    news_mom_w=None, news_vol_ma=None, news_tone_roll=None,
    news_extra_zscore_w=None, news_extra_tone_accel=None, news_extra_macro_sec_diff=None,
    btc_momentum_z_window=None,
    market_breadth_z_window=None,
    rel_momentum_window=None,
    adr_window=None,
    breakout_lookback_window=None,
    vcp_window=None,
    btc_corr_window=None,
    yz_vol_window=None,
    downside_vol_window=None,
    ret_moment_window=None,
    amihud_window=None,
    vcp_lower_low_window=None,
    breakout_volume_trigger_z=None,
    news_add_sign_confirmation: bool | None = None,
):
    out = build_technical_cols(
        rsi_w,
        bb_w,
        sma_w,
        btc_momentum_z_window=btc_momentum_z_window,
        market_breadth_z_window=market_breadth_z_window,
        rel_momentum_window=rel_momentum_window,
        adr_window=adr_window,
        breakout_lookback_window=breakout_lookback_window,
        vcp_window=vcp_window,
        btc_corr_window=btc_corr_window,
        yz_vol_window=yz_vol_window,
        downside_vol_window=downside_vol_window,
        ret_moment_window=ret_moment_window,
        amihud_window=amihud_window,
        vcp_lower_low_window=vcp_lower_low_window,
        breakout_volume_trigger_z=breakout_volume_trigger_z,
    )
    if USE_NEWS_SENTIMENT:
        if news_mom_w is None:
            news_mom_w = NEWS_MOM_WINDOWS[len(NEWS_MOM_WINDOWS) // 2]
            news_vol_ma = NEWS_VOL_MA_WINDOWS[len(NEWS_VOL_MA_WINDOWS) // 2]
            news_tone_roll = NEWS_TONE_ROLL_WINDOWS[0]
        if news_extra_zscore_w is None:
            nz = [z for z in NEWS_EXTRA_ZSCORE_WINDOWS if z > 0]
            news_extra_zscore_w = nz[len(nz) // 2] if nz else 0
        if news_extra_tone_accel is None:
            news_extra_tone_accel = True if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS else False
        if news_extra_macro_sec_diff is None:
            news_extra_macro_sec_diff = True if True in NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS else False
        tag = news_feat_tag(news_mom_w, news_vol_ma, news_tone_roll)
        out = out + build_news_model_cols(
            tag,
            zscore_w=news_extra_zscore_w,
            use_accel=bool(news_extra_tone_accel),
            use_cross=bool(news_extra_macro_sec_diff),
            add_sign_confirmation=news_add_sign_confirmation,
        )
    return out

def build_rename_map(
    rsi_w, bb_w, sma_w,
    news_mom_w=None, news_vol_ma=None, news_tone_roll=None,
    news_extra_zscore_w=None, news_extra_tone_accel=None, news_extra_macro_sec_diff=None,
    btc_momentum_z_window=None,
    market_breadth_z_window=None,
    rel_momentum_window=None,
    adr_window=None,
    breakout_lookback_window=None,
    vcp_window=None,
    btc_corr_window=None,
    yz_vol_window=None,
    downside_vol_window=None,
    ret_moment_window=None,
    amihud_window=None,
    vcp_lower_low_window=None,
    breakout_volume_trigger_z=None,
    news_add_sign_confirmation: bool | None = None,
):
    bz = int(
        btc_momentum_z_window
        if btc_momentum_z_window is not None
        else _default_btc_momentum_z_window()
    )
    brz = int(
        market_breadth_z_window
        if market_breadth_z_window is not None
        else _default_market_breadth_z_window()
    )
    rm = int(
        rel_momentum_window
        if rel_momentum_window is not None
        else _default_rel_momentum_window()
    )
    aw = int(adr_window if adr_window is not None else _default_adr_window())
    bwk = int(
        breakout_lookback_window
        if breakout_lookback_window is not None
        else _default_breakout_lookback_window()
    )
    vw = int(vcp_window if vcp_window is not None else _default_vcp_window())
    bcw = int(btc_corr_window if btc_corr_window is not None else _default_btc_corr_window())
    yz_w = int(
        yz_vol_window if yz_vol_window is not None else _default_yang_zhang_window()
    )
    dv_w = int(
        downside_vol_window
        if downside_vol_window is not None
        else _default_downside_vol_window()
    )
    rm_w = int(
        ret_moment_window
        if ret_moment_window is not None
        else _default_ret_moment_window()
    )
    am_w = int(amihud_window if amihud_window is not None else _default_amihud_window())
    ll_w = int(
        vcp_lower_low_window
        if vcp_lower_low_window is not None
        else _default_vcp_lower_low_window()
    )
    bvt_z_val = float(
        breakout_volume_trigger_z
        if breakout_volume_trigger_z is not None
        else _default_breakout_volume_trigger_z()
    )
    bvt_str = bvt_z_str(bvt_z_val)
    m = {
        f'rsi_{rsi_w}':              f'RSI ({rsi_w}d)',
        f'bb_pband_{bb_w}':          f'Bollinger %B ({bb_w}d)',
        f'bb_x_rsi_{bb_w}_{rsi_w}': f'BB({bb_w}) × RSI({rsi_w})',
        f'sma_cross_20_{sma_w}':     f'SMA20 / SMA{sma_w}',
        f'market_breadth_{sma_w}':   f'Breadth (SMA{sma_w})',
        f'market_breadth_z_{sma_w}_w{brz}': f'Breadth Z SMA{sma_w} roll{brz}',
        f'sector_avg_rsi_{rsi_w}':   f'Sector Avg RSI ({rsi_w}d)',
        f'rsi_delta_3d_{rsi_w}':     f'RSI Δ3d ({rsi_w}d)',
        f'bb_slope_5d_{bb_w}':       f'BB Slope 5d ({bb_w}d)',
        f'bb_delta_3d_{bb_w}':       f'BB Δ3d ({bb_w}d)',
        f'rsi_weekly_{rsi_w}':       f'RSI Weekly ({rsi_w}d)',
        'macd_diff':       'MACD Histogram',
        'vol_stress':      'Vol Stress',
        'drawdown':        'Drawdown 60d',
        'adx':             'ADX',
        'vol_ratio':       'Vol Ratio 5/20d',
        'adx_delta_3d':    'ADX Δ3d',
        'momentum_accel':  'Momentum Accel',
        'close_vs_sma200': 'Close / SMA200',
        'sma200_delta_5d': 'SMA200 Ratio Δ 5d',
        'drawdown_252d':   'Drawdown 252d',
        'volume_zscore':   'Volume Z-Score',
        'btc_momentum':    'BTC Momentum',
        f'btc_momentum_z_w{bz}': f'BTC Mom Z roll{bz}',
        f'corr_stock_btc_{bcw}d': f'Corr Stock/BTC {bcw}d',
        f'rel_momentum_{rm}d': f'Rel. Mom {rm}d vs sector',
        f'adr_pct_{aw}d': f'ADR Proxy {aw}d',
        f'vcp_tightness_{vw}d': f'VCP Tightness {vw}d',
        f'vcp_tightness_hl_{vw}d': f'VCP Tightness HL {vw}d',
        f'blue_sky_breakout_{bwk}d': f'Blue Sky Breakout {bwk}d',
        f'dist_to_prior_hi_pct_{bwk}d': f'Dist to Prior High {bwk}d',
        f'volume_at_resistance_{bwk}d': f'Vol@Resistance {bwk}d',
        'dollar_volume_zscore': 'Dollar Volume Z-Score',
        'volume_force_1d': 'Volume Force 1d',
        f'bb_squeeze_factor_{bb_w}': f'BB Squeeze Factor ({bb_w}d)',
        f'yz_vol_{yz_w}d': f'Yang-Zhang Vol {yz_w}d',
        f'downside_vol_{dv_w}d': f'Downside Vol {dv_w}d',
        f'downside_vol_ratio_{dv_w}d': f'Downside Vol Ratio {dv_w}d',
        f'ret_skew_{rm_w}d': f'Return Skew {rm_w}d',
        f'ret_kurt_{rm_w}d': f'Return Kurt {rm_w}d',
        f'amihud_illiquidity_{am_w}d': f'Amihud Illiquidity {am_w}d',
        f'vcp_at_period_low_frac_{vw}_{ll_w}d': f'VCP at Period-Low Frac vw{vw}/{ll_w}d',
        f'vcp_tightness_slope_{vw}_{ll_w}d': f'VCP Tightness Slope vw{vw}/{ll_w}d',
        f'breakout_volume_confirmed_{bwk}_z{bvt_str}d': f'Breakout Vol-Confirmed {bwk}d (z≥{bvt_z_val:g})',
        'month':           'Month',
        'sector_id':          'Sector ID (Research-Cluster)',
        'gics_sector_id':     'Yahoo GICS Sector ID',
        'gics_industry_id':   'Yahoo GICS Industry ID',
    }
    if USE_NEWS_SENTIMENT and news_mom_w is not None and news_vol_ma is not None and news_tone_roll is not None:
        if news_extra_zscore_w is None:
            nz = [z for z in NEWS_EXTRA_ZSCORE_WINDOWS if z > 0]
            news_extra_zscore_w = nz[len(nz) // 2] if nz else 0
        if news_extra_tone_accel is None:
            news_extra_tone_accel = True if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS else False
        if news_extra_macro_sec_diff is None:
            news_extra_macro_sec_diff = True if True in NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS else False
        tag = news_feat_tag(news_mom_w, news_vol_ma, news_tone_roll)
        for c in build_news_model_cols(
            tag,
            zscore_w=news_extra_zscore_w,
            use_accel=bool(news_extra_tone_accel),
            use_cross=bool(news_extra_macro_sec_diff),
            add_sign_confirmation=news_add_sign_confirmation,
        ):
            m[c] = c.replace('_', ' ')
    return m

print('Configuration loaded.')
print(f'Total tickers: {len(ALL_TICKERS)}')
_uf_tr = float(UNIVERSE_FRACTION)
if SCORING_ONLY:
    print(
        'Universum (Training): SCORING_ONLY=True — Symbole aus Artefakt beim Laden, '
        'nicht über UNIVERSE_FRACTION.',
        flush=True,
    )
elif _uf_tr >= 1.0:
    print(
        f'Universum (Training): 100.0% — alle {len(ALL_TICKERS)} Symbole (UNIVERSE_FRACTION={_uf_tr}).',
        flush=True,
    )
elif _uf_tr <= 0.0:
    print(
        f'Universum (Training): UNIVERSE_FRACTION={_uf_tr} ungültig (muss > 0 und ≤ 1).',
        flush=True,
    )
else:
    _n_plan = max(1, int(round(len(ALL_TICKERS) * _uf_tr)))
    print(
        f'Universum (Training): {100.0 * _uf_tr:.1f}% — geplant ca. {_n_plan}/{len(ALL_TICKERS)} Symbole '
        f'(UNIVERSE_FRACTION={_uf_tr}; Auswahl per Zufall, UNIVERSE_SAMPLE_SEED={UNIVERSE_SAMPLE_SEED}).',
        flush=True,
    )
print(f'Sector labels: {SECTOR_LABELS}')
