"""
Konfiguration und Hilfsfunktionen für Spaltennamen.

Alle Pipeline-Zustände (DataFrames, Modelle, ``best_params``, …) werden zur Laufzeit
als Attribute dieses Moduls gesetzt (gemeinsamer Namespace für alle Schritte).
``save_scoring_artifacts`` / ``load_scoring_artifacts`` delegieren an ``lib.scoring_persist``
und verwenden ``globals()`` dieses Moduls, solange die Pipeline über dieses Paket läuft.

Datei-Gliederung (Lesereihenfolge):
  1. GCP-Auth (optional), Kalender & Laufmodus
  2. Scoring-Artefakt (SCORING_ONLY, save/load, Phase-17-Jobs)
  3. Pipeline: CV/Optuna, Labels, Anti-Peak, Split, Indikator-Grids, SEED_PARAMS
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
for _stream in (sys.stdout, sys.stderr):
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

# =============================================================================
# 1. Kalender & Laufmodus (Kurse / Training; GCP-Auth siehe oben)
# =============================================================================
# ── Dates ──────────────────────────────────────────────────────────────────
# Walk-forward: BASE → META → THRESHOLD → FINAL = disjunkte Zeitfenster (Purge zwischen Stufen).
# FINAL = letztes Fenster (Final-Test); TRAIN_END_DATE = END_DATE = Kalenderende des Laufs (aktueller Tag).
START_DATE = '2015-01-01'  # Kurs-Download / Labels; News siehe NEWS_EXTRA_HISTORY_* und NEWS_BQ_*
END_DATE        = datetime.date.today().strftime('%Y-%m-%d')           # Download bis heute
TRAIN_END_DATE  = datetime.date.today().strftime('%Y-%m-%d')           # gleiches Kalenderende wie END_DATE
print(f'Download:  {START_DATE} -> {END_DATE}')
print(f'Training:  {START_DATE} -> {TRAIN_END_DATE}  (bis zum aktuellen Datum)')


# =============================================================================
# 2. Scoring-Artefakt & parallele Filter (Phase 17)
# =============================================================================
# ── SCORING_ONLY: gespeicherte Modelle + Threshold, ohne Neu-Training ───────
# True  → Daten + Scoring/HTML; Training (``training_phases`` 12–16) übersprungen.
# False → volles Training; Artefakt nach Meta-/Threshold-Phase automatisch schreiben.
# Nur diese Datei (bzw. ``cfg.SCORING_ONLY`` nach ``import``) — nicht nur eine Notebook-Variable.
SCORING_ONLY = False
SCORING_ARTIFACT_PATH = Path("models") / "scoring_artifacts.joblib"
# Phase 17: Signal-Filter pro Ticker parallel (joblib loky). -1 = alle Kerne, 1 = seriell.
PHASE17_SIGNAL_FILTER_JOBS = -1
# Fortschrittslog alle N Ticker; 0 = ca. 10 Stufen (nt//10).
PHASE17_SIGNAL_FILTER_PROGRESS_BATCH = 0
# Website + signals.json „signals“: nur FINAL-OOS (Classifier) — nie TRAIN/THRESHOLD-Kalender.
# Wird von save_scoring_artifacts / load_scoring_artifacts gesetzt, wenn in dieser Session gespeichert wurde.
_SCORING_ARTIFACT_SAVED_THIS_SESSION = False


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
    if sc:
        print(
            "Pipeline-Modus: SCORING_ONLY — nur Scoring/Export "
            "(Phasen 12–16 Training übersprungen; Artefakt wird geladen).",
            flush=True,
        )
    else:
        print(
            "Pipeline-Modus: volles Training — Optuna/Meta/Kalibrierung (12–16), danach Scoring/Export.",
            flush=True,
        )


# =============================================================================
# 3. Pipeline: CV, Optuna, Zielvariablen, Split, Indikatoren, SEED_PARAMS
# =============================================================================
# (Ticker-Universum: Abschnitt 5 — hier nur Hyperparameter & Grids.)

# ── Fixed pipeline constants ─────────────────────────────────────────────────────
N_WF_SPLITS           = 5     # Walk-forward CV folds
GDELT_CHUNK_DAYS      = 45    # Nur bei NEWS_SOURCE="gdelt_api": Chunk-Länge in Tagen
OPT_MIN_PRECISION_BASE   = 0.6   # Phase 1 Base-XGB
OPT_MIN_PRECISION_META   = 0.85  # Phase 4 Meta-Learner (inkl. produktivem Threshold aus Meta-Optuna)
OPT_MIN_PRECISION = OPT_MIN_PRECISION_META  # Reports/PR-Plots: gleiches Gate wie Meta
# Phase 5: Mindestanzahl positiver Roh-Vorhersagen (prob>=t), damit Precision nicht trivial ist
PHASE5_MIN_SIGNALS    = 5

# ── Optuna mode toggle ────────────────────────────────────────────────────────
# True  (Option A): Optuna also searches XGBoost model hyperparameters.
#                   More expressive search, but only XGBoost is tuned — other
#                   base models (LGB/RF/ET/LR) are not — asymmetric optimisation.
# False (Option B): Optuna only searches target/filter params (ohne threshold — der kommt in Phase 5).
#                   All base models use fixed SEED_PARAMS — consistent and faster.
OPT_MODEL_HYPERPARAMS = True
# True: Phase-1-Optuna optimiert Rally-/Label-Parameter (return_window, rally_threshold, …).
# False: feste Band-Regel (FIXED_Y_*), siehe ``target._create_target_one_ticker_fixed_bands``.
OPT_OPTIMIZE_Y_TARGETS = False


def opt_optimize_y_targets() -> bool:
    """True = Optuna sucht Rally-/Label-Parameter (Modulattribut ``OPT_OPTIMIZE_Y_TARGETS``)."""
    return bool(OPT_OPTIMIZE_Y_TARGETS)


# Festes Y (OPT_OPTIMIZE_Y_TARGETS=False) — ``target._create_target_one_ticker_fixed_bands``:
# (1) Grün: ∃ Fenster w ∈ [FIXED_Y_WINDOW_MIN, FIXED_Y_WINDOW_MAX] mit kum. Rendite ≥ FIXED_Y_RALLY_THRESHOLD;
#     alle Tage solcher Fenster werden zu grünen Segmenten vereinigt.
# (2) Label 1: FIXED_Y_LEAD_DAYS vor Segmentbeginn; bei L < FIXED_Y_SEGMENT_SPLIT zusätzlich die ersten
#     FIXED_Y_ENTRY_DAYS grüne Tage; bei L ≥ split alle grünen außer den letzten FIXED_Y_TAIL_EXCLUDE_DAYS.
# Entspricht: w ∈ [3, 10], 6 %, 3 Vorlauf-Tage, Split 5, 2 Einstiegs-Tage auf Grün, 3 Tage Rally-Ende ausgeschlossen.
FIXED_Y_WINDOW_MIN = 3
FIXED_Y_WINDOW_MAX = 8
FIXED_Y_RALLY_THRESHOLD = 0.09  # 8.00 %
FIXED_Y_SEGMENT_SPLIT = 5
FIXED_Y_LEAD_DAYS = 3
FIXED_Y_ENTRY_DAYS = 2
FIXED_Y_TAIL_EXCLUDE_DAYS = 3


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


def describe_target_rule_fixed_bands() -> str:
    """Feste Band-Regel (OPT_OPTIMIZE_Y_TARGETS=False); Zahlen = ``fixed_y_rule_params()`` wie in der Pipeline."""
    w_lo, w_hi, rt, split, ld, ed, tail_ex = fixed_y_rule_params()
    return (
        "Zieldefinition (feste Band-Regel — dieselben Konstanten wie in der Pipeline):\n\n"
        "(1) Grüne Rally-Phase: Ein Handelstag ist „grün“, wenn es mindestens ein Fenster "
        "mit Länge w ∈ [{w_lo}, {w_hi}] Handelstagen gibt, über das die kumulierte Rendite "
        "(Produkt der Tagesrenditen) mindestens {rt_str} beträgt; alle Tage solcher Fenster werden "
        "zu grünen Segmenten vereinigt.\n\n"
        "(2) Positives Klassenlabel: Am ersten Tag eines neuen grünen Segments sei L die "
        "Segmentlänge in Handelstagen. Label 1 erhalten die {ld} Handelstage unmittelbar vor "
        "Segmentbeginn; bei L < {split} zusätzlich die ersten {ed} grünen Tage des Segments; "
        "bei L ≥ {split} alle grünen Tage des Segments außer den letzten {tail_ex}."
    ).format(
        w_lo=w_lo,
        w_hi=w_hi,
        rt_str=f"{rt:.2%}",
        split=split,
        ld=ld,
        ed=ed,
        tail_ex=tail_ex,
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
    rt_str = f"{rt:.2%}"
    return (
        "Was das Modell vorhersagen soll: Das Klassenlabel „positiv“ markiert Tage, an denen sich ein "
        "Einstieg vor oder am Anfang einer späteren Kursaufwertsphase lohnt — nicht die gesamte Rally, "
        "sondern Vorlauf und frühe grüne Tage. Die Regel ist fest (ohne Optuna für Rally-Parameter) "
        "und entspricht exakt der Pipeline.\n\n"
        "Zuerst wird festgelegt, wann der Kurs in einer „grünen“ Rally-Phase steht: Ein Tag zählt dazu, "
        "wenn es mindestens ein Fenster aus {w_lo} bis {w_hi} aufeinanderfolgenden Handelstagen gibt, "
        "über das die kumulierte Rendite (Produkt der Tagesrenditen) mindestens {rt_str} beträgt. "
        "Alle Tage, die in ein solches Fenster fallen, werden zu zusammenhängenden grünen Segmenten "
        "vereinigt — so entsteht die Rally-Spur.\n\n"
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


EARLY_STOPPING_ROUNDS = 30
N_OPTUNA_TRIALS       = 35  # erhöht wegen News-Feature-Grid
# Nur Optuna Phase 1 (Base-XGB): drosseln, wenn Trials trotz kleinem Universum langsam sind.
# UNIVERSE_FRACTION verkürzt nur Zeilen/Ticker in df_train; Laufzeit dominiert oft
# N_OPTUNA_TRIALS × OPTUNA_WF_SPLITS × (rebuild_target + XGB pro Fold).
OPTUNA_WF_SPLITS      = None  # None → N_WF_SPLITS; z.B. 3 weniger Folds pro Trial
# tqdm-Balken in Phase 1: None oder True = an; False = aus (ruhigeres Log, weniger Sonderzeichen | % in der Konsole).
OPTUNA_SHOW_PROGRESS_BAR = None
N_META_TRIALS         = 250
# Meta-Stacking: Roh-Features neben Base-Wahrscheinlichkeiten (s. ``optuna_base_models``).
# Standard: feste Anzahl META_SHAP_TOP_K. Alternativ: META_SHAP_CUM_FRAC z. B. 0.75 =
# kleinstes K, sodass Summe der K größten mean|SHAP| ≥ 75 % der Summe über alle Spalten.
META_SHAP_TOP_K = 10  # Fallback, wenn Summe mean|SHAP| = 0 (nur dann)
META_SHAP_CUM_FRAC = 0.85  # kleinstes K mit kumulierter SHAP-Masse ≥ 75 %
META_SHAP_TOP_K_MIN = 5
META_SHAP_TOP_K_MAX = 35
RANDOM_STATE          = 42
N_WORKERS             = os.cpu_count()

# ── Parametrisches Y (nur wenn OPT_OPTIMIZE_Y_TARGETS=True) ─────────────────
# Bei festem Y (False) nutzt ``create_target`` nur FIXED_Y_*; diese Werte sind dann irrelevant,
# werden aber an die Band-Logik und SEED_PARAMS-Y-Keys angeglichen (Lesbarkeit / späteres Umschalten).
RETURN_WINDOW        = FIXED_Y_WINDOW_MAX  # Oberkante w; Band-Regel nutzt w ∈ [FIXED_Y_WINDOW_MIN, …]
RALLY_THRESHOLD      = FIXED_Y_RALLY_THRESHOLD
LEAD_DAYS            = FIXED_Y_LEAD_DAYS   # bei festem Y: dieselben wie in fixed_y_rule_params()
ENTRY_DAYS           = FIXED_Y_ENTRY_DAYS
MIN_RALLY_TAIL_DAYS  = 5 # nur parametrische Regel; bei festem Y unbenutzt
CONSECUTIVE_DAYS     = 2     # wie SEED_PARAMS (Post-Filter)
SIGNAL_COOLDOWN_DAYS = 2

# ── Anti-Peak (nur apply_signal_filters / Website — nicht in Optuna-CV) ─────
# Viele Fehlsignale sitzen direkt am lokalen Kurs-High; Filter verlangt Abstand zum N-Tage-High.
SIGNAL_SKIP_NEAR_PEAK = True
PEAK_LOOKBACK_DAYS = 20
PEAK_MIN_DIST_FROM_HIGH_PCT = 0.012   # mind. 1.2 % unter Rolling-High (tunable 0.008–0.02)
SIGNAL_MAX_RSI = 78.0                 # kein Kauf-Signal wenn RSI darüber; None = aus

# ── Ticker-Universum (Symbole) ──────────────────────────────────────────────
# Maximale Anzahl Wertpapiere im TRAIN_BASE-Set — nur bei SPLIT_MODE="ticker" (Legacy).
# Bei SPLIT_MODE="time" nutzen alle Stufen dieselben Ticker; unten steht der Zeit-Split.
# Empfehlung: 30–60. Höher = langsamer, aber potenziell besseres Modell.
MAX_TRAIN_TICKERS   = 45
# Anteil von ALL_TICKERS, der geladen wird — VOR Download/Target/News/Features (s. data_and_split).
# 2.5 % = 0.025 (nicht 2.5 — sonst greift die Auswahl nicht).
# Nicht verwechseln mit TIME_SPLIT_FRAC_*: das sind Anteile der Handelstage, nicht der Ticker.
UNIVERSE_FRACTION = 1  # 1.0 = alle; <1 = zufällige Teilmenge (UNIVERSE_SAMPLE_SEED)
UNIVERSE_SAMPLE_SEED = 42

# ── Zeit-Split (Kalender) ────────────────────────────────────────────────────
# "time" = gleiche Ticker in allen Stufen, disjunkte Zeiträume (empfohlen).
# "ticker" = Legacy: verschiedene Ticker pro Stufe, gleicher Kalender bis TRAIN_END_DATE.
SPLIT_MODE = "time"
# Anteile der „Inhalts“-Handelstage (n − 3×Purge) für SPLIT_MODE=time: **BASE** (nur Base-Optuna/Modelle),
# **META** (nur Meta-Learner; Base-Vorhersagen OOS), **THRESHOLD** (Schwellen-Kalibrierung), **FINAL** (OOS).
# Zwischen je zwei Blöcken liegen TIME_PURGE_TRADING_DAYS (keine Überlappung / Embargo).
# Alle drei Fraktionen müssen > 0 sein; Summe BASE+META+THRESHOLD muss < 1 (Rest = FINAL).
# Da der produktive Threshold jetzt direkt aus Meta-Optuna kommt, kann META größer und THRESHOLD kleiner sein.
# THRESHOLD bleibt als separates Diagnose-/Monitoring-Fenster bestehen.
# SPLIT_MODE=ticker: dieselben BASE/META-Anteile + Purge teilen nur den TRAIN-Kalender (gleiche Ticker).
TIME_SPLIT_FRAC_BASE = 0.45
TIME_SPLIT_FRAC_META = 0.35
TIME_SPLIT_FRAC_THRESHOLD = 0.05
TIME_PURGE_TRADING_DAYS = 5

# ── Indicator parameter grids ───────────────────────────────────────────────
RSI_WINDOWS  = [7, 10, 14, 21]
BB_WINDOWS   = [15, 20, 25]
SMA_WINDOWS  = [30, 50, 70]

# ── Seed params für enqueue_trial (Optuna Base-XGB) ─────────────────────────
# Fest eingetragen aus ``best_params`` des letzten gespeicherten Trainings
# (``models/scoring_artifacts.joblib``) — kein Laufzeit-Laden des Artefakts.
SEED_PARAMS = dict(
    # Y-Keys: an feste Band-Regel angeglichen (bei OPT_OPTIMIZE_Y_TARGETS=False ohnehin aus enqueue entfernt).
    return_window=10,
    rally_threshold=0.06,
    lead_days=3,
    entry_days=2,
    min_rally_tail_days=5,
    consecutive_days=2,
    signal_cooldown_days=2,
    threshold=0.6338651351186617,
    rsi_window=21,
    bb_window=20,
    sma_window=50,
    news_mom_w=3,
    news_vol_ma=20,
    news_tone_roll=3,
    news_extra_zscore_w=20,
    news_extra_tone_accel=True,
    news_extra_macro_sec_diff=True,
    btc_momentum_z_window=60,
    market_breadth_z_window=60,
    rel_momentum_window=20,
    grow_policy='lossguide',
    max_leaves=686,
    max_bin=256,
    max_depth=3,
    min_child_weight=22,
    gamma=8.955413122339955,
    reg_alpha=9.617691306887567e-08,
    reg_lambda=2.1395039834782624e-08,
    learning_rate=0.024079682452395383,
    n_estimators=363,
    subsample=0.7803623012672697,
    colsample_bytree=0.5792981006909539,
    focal_gamma=0.8076637868641594,
    focal_alpha=0.3869335839666591,
    signal_skip_near_peak=True,
    peak_lookback_days=20,
    peak_min_dist_from_high_pct=0.012,
    signal_max_rsi=78.0,
)

# =============================================================================
# 4. News / BigQuery (GKG) / GDELT-API-Fallback (V2Themes, GCAM, optional Anker-Orgs)
# =============================================================================

# ── News / GDELT grids (Optuna wählt ein Tag-Tripel) ─────────────────────────
USE_NEWS_SENTIMENT = True
# Momentum auf geglätteter Ton-Serie; Vol-MA für Spike; zusätzliche Ton-Glättung vor Momentum
NEWS_MOM_WINDOWS = [3, 5, 7]
NEWS_VOL_MA_WINDOWS = [10, 20]
NEWS_TONE_ROLL_WINDOWS = [0, 3]  # 0 = keine zusätzliche Ton-Glättung

# Zusätzliche News-Spalten — Optuna Phase 1 wählt je Trial ein Tripel (wie news_mom_w …)
NEWS_EXTRA_ZSCORE_WINDOWS = [0, 10, 20, 30]   # 0 = keine tone_z/vol_z; >0 → Spalten …_tone_z_w{w}
NEWS_EXTRA_TONE_ACCEL_OPTIONS = [False, True]
NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS = [False, True]
# assemble_features: welche News-Spalten am Ende per Fillna/0 angelegt werden.
# "optuna_union" = Vereinigung wie Optuna-Raster (bei gleicher Config oft gleiche Anzahl wie all_news_model).
# "all_news_model" = alle Namen aus all_news_model_cols() (Legacy).
# "none" = keinen News-Fill (nur was die Merges liefern; für Experten).
FEATURE_ASSEMBLE_NEWS_FILL = "optuna_union"  # "all_news_model" | "optuna_union" | "none"

# Kontext-Features: Optuna wählt je Trial ein Fenster (Indizes siehe SEED_PARAMS).
BTC_MOMENTUM_Z_WINDOWS = [20, 40, 60, 120]
MARKET_BREADTH_Z_WINDOWS = [20, 40, 60, 120]
REL_MOMENTUM_WINDOWS = [10, 20, 60]

# Volles News-Fenster-Grid: News liegt nur in Shard-Dateien (FEATURE_SHARD_DIR), nicht als breite Spalten in df_features.
FEATURE_SHARD_DIR = os.path.join(os.getcwd(), "data", "feature_shards_news")
NEWS_SHARD_MANIFEST: dict[str, str] = {}
_FEATURE_NEWS_SHARDS_ACTIVE = False

NEWS_CACHE_DIR = os.path.join(os.getcwd(), 'data')
NEWS_CACHE_FILE = os.path.join(NEWS_CACHE_DIR, 'news_gdelt_cache.pkl')
# News: BigQuery (GDELT events, kein API-Rate-Limit) oder Legacy HTTP-Doc-API
NEWS_SOURCE = "bigquery"  # "bigquery" | "gdelt_api"
# GCP-Projekt mit BigQuery-Billing (Abrechnung). Auth siehe Modulkopf.
BQ_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "gedelt-calls").strip() or "gedelt-calls"
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", BQ_PROJECT_ID)
# Öffentliches GDELT (Projekt gdelt-bq). **events** hat keine Theme-Spalten — News-Features nutzen **GKG**.
BQ_USE_GKG_TABLE = True
GDELT_BQ_EVENTS_TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"
# Kosten: GKG + BQ_SINGLE_SCAN → 1 Query pro Lauf (Vereinigung der Lücken); sonst 1 Query pro Kanal/Lücke.
# News vs. Kurs: optional länger (Kalenderjahre). Überschreibt NEWS_EXTRA_* wenn gesetzt.
# Vor Kurs-Start: Warmup für Rollings/Lags/Z-Scores am ersten Kurstag (empfohlen: 1).
NEWS_EXTRA_HISTORY_YEARS_BEFORE = 1
# Nach Kurs-Ende: selten nötig; z. B. wenn Nachrichtenlage mit Verzug zur letzten Kurszeile passen soll.
NEWS_EXTRA_HISTORY_YEARS_AFTER = 0
# Explizite Grenzen (setzt NEWS_EXTRA_* außer Kraft): None = automatisch wie oben + Kurs-START/END
NEWS_BQ_START_DATE = None  # z.B. "2023-01-01" — kürzer = weniger gescannte GB
NEWS_BQ_END_DATE = None    # z.B. None oder gleiches END_DATE
# Nur wenn BQ_USE_GKG_TABLE False und du eine andere Tabelle meinst (Legacy):
BQ_EVENT_DATE_FIELD = "SQLDATE"
BQ_THEMES_COLUMN = "V2Themes"
BQ_USE_PARTITION_FILTER = True  # Pflicht: _PARTITIONTIME — sonst Full-Table-Scan
BQ_SINGLE_SCAN = True          # True: GKG-Query mit IF/COUNTIF pro Kanal (nicht 14×)
# Größere Fenster + viele Theme-Spalten → BigQuery „Query too big“ / Ressourcenlimit.
# 0 = ein einziger Scan über [a,b]; sonst höchstens so viele Kalendertage pro Teil-Query (Ergebnisse werden aneinandergehängt).
BQ_SINGLE_SCAN_CHUNK_DAYS = 120
# Wenn True: sektorale GKG-Bedingung = (Theme-Filter) AND (Keyword-OR über SECTOR_KEYWORDS[sector]).
# Leere Channel-Liste bedeutet: auf alle Sektorkanäle anwenden.
BQ_SECTOR_THEME_KEYWORD_CONJUNCTION = True
BQ_SECTOR_THEME_KEYWORD_CHANNELS: tuple[str, ...] = ()

# Feste GKG-Theme-Tripel für SQL (leer = keine zusätzlichen Theme-Kanäle im GKG-Scan).
# Hinweis: Es werden keine ``news_*_th_*``-Modellspalten mehr gebaut — nur V2 Macro/Sektor + Sektor-GCAM.
GKG_THEME_SQL_TRIPLES = []

# GCAM-Schlüssel: nicht leer = feste Liste (überschreibt Auto). Sonst: Must-Have + optional Datei/Exploration.
# GKG_GCAM_EXPLORE_KEYS=True: BigQuery holt zusätzliche Top-c* (Beifang), ohne Must-Haves zu duplizieren.
# Nach abgeschlossener Exploration: GKG_GCAM_EXPLORE_KEYS=False stellen — dann KEIN erneutes BQ-Scouting;
# die ermittelten Keys bleiben in GKG_GCAM_KEYS_PATH (Must-Haves + Datei). GKG_GCAM_EXPLORE_EXTRA_N wird
# nur bei EXPLORE_KEYS=True gelesen; auf „8 → 3“ zu ändern löst ohne Exploration nichts aus und kürzt
# die gespeicherte Liste nicht automatisch. Zum harten Einfrieren: volle Tuple hier in GKG_GCAM_METRIC_KEYS.
# GKG_GCAM_USE_EXTRA_N: nach Resolve nur so viele *Nicht-Must-Have*-Keys nutzen (weniger Features/Optuna).
# Unterscheidet sich von GKG_GCAM_EXPLORE_EXTRA_N (nur BQ-Exploration). None = alle Extras aus Datei.
# Keys mit Doppelpunkt (z. B. vnt:success): Regex in news._bq_gcam_regexp_for_key (RE2) — Wert als :<zahl>.
GKG_GCAM_METRIC_KEYS: tuple[str, ...] = ()
GKG_GCAM_MUST_HAVE_KEYS: tuple[str, ...] = (
    "c18.158",  # LMT_Negative
    "c18.159",  # LMT_Positive
    "c18.161",  # LMT_Uncertainty
    "c12.152",  # Harvard_Weakness
    "c12.1",  # Harvard_Negative
    "vnt:success",  # Lasswell_Success (Kontext Erfolg / Meilensteine)
    "c12.181",  # Harvard_Active (Dynamik / Momentum im Text)
)
GKG_GCAM_AUTO_RESOLVE = True
# True = BigQuery holt *neue* zusätzliche c*-Keys (Key-Exploration). False = nur Must-Haves + JSON, kein BQ-Scout.
# (Lange BQ-Läufe bei False kommen vom News-GKG-Cache — nicht von GCAM-Key-Exploration.)
GKG_GCAM_EXPLORE_KEYS = False
# Nur wenn GKG_GCAM_EXPLORE_KEYS True: Anzahl der zusätzlichen Top-c*-Keys (Must-Haves extra).
GKG_GCAM_EXPLORE_EXTRA_N = 8
GKG_GCAM_AUTO_TOP_N = 8  # Fallback wenn GKG_GCAM_EXPLORE_EXTRA_N fehlt
GKG_GCAM_KEYS_PATH = Path("data") / "gkg_gcam_metric_keys.json"
# Nach Resolve (Must-Haves + JSON): höchstens so viele *zusätzliche* GCAM-Keys für Features/Training.
# Z. B. 8 in der Datei exploriert, hier 2 → nur Must-Haves + 2 Extras. None = keine Kürzung; 0 = nur Must-Haves.
# Nur *schmalere* Key-Liste löst keinen News-BigQuery-Neu-Scan aus, solange der Pickle-Cache die Keys schon kennt.
GKG_GCAM_USE_EXTRA_N = 2  # schlankere Feature-Matrix; None = alle Extras aus Datei


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
                   'TSM', 'AMD', 'PANW',
                   'SAP.DE', 'IFX.DE',
                   'NEM.DE', 'BC8.DE',
                   'TMV.DE', 'COK.DE', 'YSN.DE', 'GFT.DE', 'NA9.DE',
                   'AOF.DE', 'ELG.DE', 'AIXA.DE', 'SMHN.DE', 'SANT.DE'],

    # ── Financial Services ─────────────────────────────────────────────────
    # DAX: DB1, HNR1  |  MDAX: TLX  |  SDAX: MLP, GLJ, PBB, HABA, WUW, DBAG, HYQ
    'financial_services': ['JPM', 'GS', 'AXP', 'V',
                           'ALV.DE', 'MUV2.DE', 'DBK.DE', 'CBK.DE', 'BNP.PA',
                           'DB1.DE', 'HNR1.DE',
                           'TLX.DE',
                           'MLP.DE', 'GLJ.DE', 'PBB.DE', 'WUW.DE', 'DBAG.DE', 'HYQ.DE'],

    # ── Healthcare ─────────────────────────────────────────────────────────
    # DAX: FME, SHL  |  MDAX: SRT3  |  SDAX: AFX, EUZ, DMP, GXI, DRW3, O2BC, EVT
    'healthcare': ['JNJ', 'UNH', 'MRK', 'AMGN', 'NVS',
                   'ISRG', 'ILMN',
                   'BAYN.DE', 'FRE.DE', 'MRK.DE', 'FME.DE', 'SHL.DE',
                   'SRT3.DE',
                   'AFX.DE', 'EUZ.DE', 'DMP.DE', 'GXI.DE', 'DRW3.DE', 'O2BC.DE', 'EVT.DE'],

    # ── Consumer Cyclical (inkl. Automotive-Basket) ───────────────────────
    # DAX: BEI  |  SDAX: FIE, HBH, DOU, SZU, HFG, KWS, EVD, DHER
    # DAX Automotive: VOW3, BMW, MBG, CON, PAH3, P911, DTG  |  MDAX: TGR, SHA
    'consumer_cyclical': ['KO', 'MCD', 'NKE', 'PG', 'WMT', 'HD', 'DIS',
                          'ADS.DE', 'BOSS.DE', 'HEN3.DE', 'PUM.DE', 'ZAL.DE', 'LVMH.PA',
                          'BEI.DE',
                          'FIE.DE', 'HBH.DE', 'DOU.DE', 'SZU.DE', 'HFG.DE', 'KWS.DE',
                          'EVD.DE', 'DHER.DE',
                          'VOW3.DE', 'BMW.DE', 'MBG.DE', 'CON.DE', 'PAH3.DE', 'P911.DE',
                          'DTG.DE',
                          'TM', 'STLA', 'GM', 'F',
                          'TGR.DE', 'SHA.DE'],

    # ── Industrials (inkl. HVAC/Heat-Pump-Basket) ─────────────────────────
    # DAX: SIE, RHM, MTX, AIR, DHL, G1A  |  MDAX: KBX, HOC, KGX, JUN3, TKA, HAG, R3NK, LHA, FRA, HLAG, RAA
    # SDAX: DUE, JST, SFQ, NOEJ, VOS, WAC, INH, MUX, STM, KSB, HDD
    'industrials': ['CAT', 'BA', 'HON', 'MMM', 'SHW', 'TRV', 'DOW', 'FDX',
                    'SIE.DE', 'RHM.DE', 'MTX.DE', 'AIR.DE', 'DHL.DE', 'G1A.DE',
                    'KBX.DE', 'HOC.DE', 'KGX.DE', 'JUN3.DE', 'TKA.DE', 'HAG.DE',
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
                        'HEI.DE', 'SY1.DE',
                        'EVK.DE', 'LXS.DE', 'NDA.DE', 'SDO.DE', 'FPE3.DE',
                        'ACT.DE', 'WAF.DE', 'BFSA.DE', 'KCO.DE'],

    # ── Real Estate ────────────────────────────────────────────────────────
    # DAX: VNA  |  MDAX: LEG  |  SDAX: GYC, DEQ, HABA, PAT
    'real_estate':['VNA.DE', 'LEG.DE', 'O', 'AMT', 'SPG', 'WDP.BR',
                   'GYC.DE', 'DEQ.DE', 'HABA.DE', 'PAT.DE'],

    # ── Communication Services (Telecom + Media) ───────────────────────────
    # DAX: DTE  |  MDAX: O2D, UTDI, FNTN  |  SDAX: 1U1
    # DAX: G24  |  MDAX: RRTL  |  SDAX: SPG, BVB, PSM, CWC
    'communication_services': ['DTE.DE', 'T', 'VZ', 'TMUS', 'ORAN',
                               'O2D.DE', 'UTDI.DE', 'FNTN.DE', '1U1.DE',
                               'RRTL.DE', 'SPG.DE', 'BVB.DE', 'PSM.DE', 'CWC.DE', 'G24.DE'],
}

# Alphabetically sorted sector labels (0-based)
SECTOR_LABELS = {s: i for i, s in enumerate(sorted(TICKERS_BY_SECTOR.keys()))}
# basic_materials=0, communication_services=1, consumer_cyclical=2, crypto=3,
# energy=4, financial_services=5, healthcare=6, industrials=7, real_estate=8, technology=9

TICKER_TO_SECTOR = {t: s for s, tl in TICKERS_BY_SECTOR.items() for t in tl}
ALL_TICKERS = [t for tl in TICKERS_BY_SECTOR.values() for t in tl]

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
    'SAP.DE': 'SAP', 'IFX.DE': 'Infineon', 'ASML': 'ASML',
    # Finance
    'JPM': 'JPMorgan', 'GS': 'Goldman Sachs', 'AXP': 'AmEx', 'V': 'Visa',
    'ALV.DE': 'Allianz', 'MUV2.DE': 'Munich Re', 'DBK.DE': 'Deutsche Bank',
    'CBK.DE': 'Commerzbank', 'BNP.PA': 'BNP Paribas',
    # Healthcare
    'JNJ': 'J&J', 'UNH': 'UnitedHealth', 'MRK': 'Merck US', 'AMGN': 'Amgen',
    'ISRG': 'Intuitive Surgical', 'ILMN': 'Illumina',
    'BAYN.DE': 'Bayer', 'FRE.DE': 'Fresenius', 'MRK.DE': 'Merck KGaA',
    'SRT3.DE': 'Sartorius', 'NVS': 'Novartis',
    # Consumer
    'KO': 'Coca-Cola', 'MCD': "McDonald's", 'NKE': 'Nike', 'PG': 'P&G',
    'WMT': 'Walmart', 'HD': 'Home Depot', 'DIS': 'Disney',
    'ADS.DE': 'Adidas', 'BOSS.DE': 'Hugo Boss', 'HEN3.DE': 'Henkel',
    'PUM.DE': 'PUMA', 'ZAL.DE': 'Zalando', 'LVMH.PA': 'LVMH',
    # Industrial (DAX)
    'CAT': 'Caterpillar', 'BA': 'Boeing', 'HON': 'Honeywell', 'MMM': '3M',
    'FDX': 'FedEx',
    'SHW': 'Sherwin-Williams', 'TRV': 'Travelers', 'DOW': 'Dow Inc.',
    'SIE.DE': 'Siemens', 'RHM.DE': 'Rheinmetall', 'MTX.DE': 'MTU Aero',
    'AIR.DE': 'Airbus', 'DHL.DE': 'DHL Group', 'G1A.DE': 'GEA Group',
    # Industrial (MDAX)
    'KBX.DE': 'Knorr-Bremse', 'HOC.DE': 'Hochtief', 'KGX.DE': 'KION Group',
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
    # Automotive (MDAX)
    'TGR.DE': 'Traton', 'SHA.DE': 'Schaeffler',
    # Materials (DAX)
    'BAS.DE': 'BASF', 'LIN': 'Linde', 'WCH.DE': 'Wacker Chemie',
    'APD': 'Air Products', 'LYB': 'LyondellBasell', 'BNR.DE': 'Brenntag',
    'NEM': 'Newmont', 'RIO': 'Rio Tinto', 'HEI.DE': 'Heidelberg Materials', 'SY1.DE': 'Symrise',
    # Materials (MDAX)
    'EVK.DE': 'Evonik', 'LXS.DE': 'LANXESS', 'NDA.DE': 'Aurubis',
    'SDO.DE': 'K+S', 'FPE3.DE': 'Fuchs SE',
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
    'TMUS': 'T-Mobile', 'ORAN': 'Orange',
    'O2D.DE': 'Telefónica Dtl.', 'UTDI.DE': 'United Internet',
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
    'DBAG.DE': 'Deutsche Beteiligungs', 'HYQ.DE': 'Hypoport',
    # Healthcare (DAX additions)
    'FME.DE': 'Fresenius Med.', 'SHL.DE': 'Siemens Healthineers',
    # Healthcare (SDAX)
    'AFX.DE': 'Carl Zeiss Meditec', 'EUZ.DE': 'Eckert & Ziegler',
    'DMP.DE': 'Dermapharm', 'GXI.DE': 'Gerresheimer',
    'DRW3.DE': 'Drägerwerk', 'O2BC.DE': 'Ottobock', 'EVT.DE': 'Evotec',
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
    'SMHN.DE': 'SÜSS MicroTec', 'SANT.DE': 'Kontron',
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
    if isinstance(ref_date, datetime.datetime):
        d = ref_date.date()
    elif isinstance(ref_date, datetime.date):
        d = ref_date
    elif isinstance(ref_date, str) and len(ref_date) >= 10:
        d = datetime.date.fromisoformat(ref_date[:10])
    else:
        d = datetime.date.today()
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
    ]


def _news_sec_variant_cols(mid: str):
    return [
        f'news_sec_{mid}_tone',
        f'news_sec_{mid}_vol',
        f'news_sec_{mid}_tone_mom',
        f'news_sec_{mid}_vol_spike',
        f'news_sec_{mid}_tone_l1', f'news_sec_{mid}_tone_l3', f'news_sec_{mid}_tone_l5',
        f'news_sec_{mid}_vol_l1', f'news_sec_{mid}_vol_l3', f'news_sec_{mid}_vol_l5',
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
    if use_accel:
        out.append(f"news_sec_{mid}_tone_accel")


def build_news_model_cols(tag, zscore_w=0, use_accel=False, use_cross=False):
    """Spaltenliste für ein gewähltes Tag-Tripel; zscore_w=0 schließt tone_z/vol_z aus."""
    out = _news_base_cols(tag)
    zw = int(zscore_w) if zscore_w is not None else 0
    if zw > 0:
        sw = f'_w{zw}'
        out.extend([
            f'news_macro_{tag}_tone_z{sw}', f'news_macro_{tag}_vol_z{sw}',
            f'news_sec_{tag}_tone_z{sw}', f'news_sec_{tag}_vol_z{sw}',
        ])
    if use_accel:
        out.extend([f'news_macro_{tag}_tone_accel', f'news_sec_{tag}_tone_accel'])
    if use_cross:
        out.append(f'news_cross_{tag}_macro_minus_sec_tone')
    _append_gcam_news_cols(out, tag, zscore_w, use_accel)
    if NEWS_ANCHOR_ORG_FILTER:
        _append_anchor_gcam_sec_cols(out, tag, zscore_w, use_accel)
        _append_gcam_div_sec_cols(out, tag, zscore_w, use_accel)
        _append_anchor_quality_sec_cols(out, tag, zscore_w, use_accel)
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
    for mom in NEWS_MOM_WINDOWS:
        for vma in NEWS_VOL_MA_WINDOWS:
            for tr in NEWS_TONE_ROLL_WINDOWS:
                tag = news_feat_tag(mom, vma, tr)
                for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                    zw = int(z) if z is not None else 0
                    for acc in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        for cr in NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS:
                            for c in build_news_model_cols(
                                tag,
                                zscore_w=zw,
                                use_accel=bool(acc),
                                use_cross=bool(cr),
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
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        cols.append(f'news_sec_{mid_q}_tone_accel')
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


def build_technical_cols(
    rsi_w,
    bb_w,
    sma_w,
    *,
    btc_momentum_z_window=None,
    market_breadth_z_window=None,
    rel_momentum_window=None,
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
        f'rel_momentum_{rm}d',
        'month',
        'sector_id',
        'gics_sector_id',
        'gics_industry_id',
    ]


def all_model_tech_col_names_for_assemble_dropna() -> list[str]:
    """Alle technischen Modell-Spaltennamen über RSI/BB/SMA- und Kontext-Fenster-Raster (für dropna nach assemble)."""
    names: set[str] = set()
    for rw in RSI_WINDOWS:
        for bw in BB_WINDOWS:
            for sw in SMA_WINDOWS:
                for bz in BTC_MOMENTUM_Z_WINDOWS:
                    for brz in MARKET_BREADTH_Z_WINDOWS:
                        for relm in REL_MOMENTUM_WINDOWS:
                            for c in build_technical_cols(
                                rw,
                                bw,
                                sw,
                                btc_momentum_z_window=bz,
                                market_breadth_z_window=brz,
                                rel_momentum_window=relm,
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
):
    out = build_technical_cols(
        rsi_w,
        bb_w,
        sma_w,
        btc_momentum_z_window=btc_momentum_z_window,
        market_breadth_z_window=market_breadth_z_window,
        rel_momentum_window=rel_momentum_window,
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
        )
    return out

def build_rename_map(
    rsi_w, bb_w, sma_w,
    news_mom_w=None, news_vol_ma=None, news_tone_roll=None,
    news_extra_zscore_w=None, news_extra_tone_accel=None, news_extra_macro_sec_diff=None,
    btc_momentum_z_window=None,
    market_breadth_z_window=None,
    rel_momentum_window=None,
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
        f'rel_momentum_{rm}d': f'Rel. Mom {rm}d vs sector',
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
