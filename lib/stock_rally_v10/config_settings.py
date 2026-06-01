"""Tunable parameters — edit here.

Imported into ``lib.stock_rally_v10.config`` with ``from .config_settings import *`` so all
``cfg.*`` names and ``globals()`` in that module stay on one namespace.

``config.py`` keeps GCP bootstrap, ``save_scoring_artifacts`` / ``load_scoring_artifacts``,
target-rule helpers, ticker lists, and column builders.
"""
import datetime
import os
from pathlib import Path

# =============================================================================
# 1. Kalender & Laufmodus (Kurse / Training; GCP-Auth siehe oben)
# =============================================================================
# ── Dates ──────────────────────────────────────────────────────────────────
# Walk-forward: BASE → META → THRESHOLD → FINAL = disjunkte Zeitfenster (Purge zwischen Stufen).
# FINAL = letztes Fenster (Final-Test); TRAIN_END_DATE = END_DATE = Kalenderende des Laufs (aktueller Tag).
START_DATE = '2015-01-01'  # Kurs-Download / Labels; News siehe NEWS_EXTRA_HISTORY_* und NEWS_BQ_*
END_DATE        = datetime.date.today().strftime('%Y-%m-%d')           # Download bis heute
TRAIN_END_DATE  = datetime.date.today().strftime('%Y-%m-%d')           # gleiches Kalenderende wie END_DATE
YF_DOWNLOAD_BATCH_SIZE = 20
YF_DOWNLOAD_BATCH_SLEEP_SEC = 2.0
print(f'Download:  {START_DATE} -> {END_DATE}')
print(f'Training:  {START_DATE} -> {TRAIN_END_DATE}  (bis zum aktuellen Datum)')


# =============================================================================
# 2. Scoring-Artefakt & parallele Filter (Phase 17)
# =============================================================================
# ── SCORING_ONLY: gespeicherte Modelle + Threshold, ohne Neu-Training ───────
# True  → Daten + Scoring/HTML; Training (``training_phases`` 12–16) übersprungen.
# False → volles Training; Artefakt nach Meta-/Threshold-Phase automatisch schreiben.
# Nur diese Datei (bzw. ``cfg.SCORING_ONLY`` nach ``import``) — nicht nur eine Notebook-Variable.
SCORING_ONLY = True
# True → Base-Optuna/Base-Modelle bleiben aus Artefakt bestehen; Training startet direkt bei Meta.
# Voraussetzung: SCORING_ONLY=False und ein vorhandenes ``SCORING_ARTIFACT_PATH``.
RETRAIN_META_ONLY = False
SCORING_ARTIFACT_PATH = Path("models") / "scoring_artifacts.joblib"
# Phase 17: Signal-Filter pro Ticker parallel (joblib loky). -1 = alle Kerne, 1 = seriell.
PHASE17_SIGNAL_FILTER_JOBS = -1
# Fortschrittslog alle N Ticker; 0 = ca. 10 Stufen (nt//10).
PHASE17_SIGNAL_FILTER_PROGRESS_BATCH = 0
# docs/index.html: GitHub lehnt Dateien >100 MiB ab (nur diese Datei — nicht docs/charts/).
DOCS_INDEX_HTML_MAX_BYTES = 98 * 1024 * 1024
DOCS_WEBSITE_MAX_CHART_SIGNALS_CAP = 1200
DOCS_WEBSITE_HTML_SHRINK_STEP = 35
# Website-Charts: "external" = PNG unter docs/charts/{thumb,full}/ (viele Plots, kleines HTML).
# "inline" = Base64 in index.html (Shrink-Loop bis unter DOCS_INDEX_HTML_MAX_BYTES).
WEBSITE_CHART_STORAGE = "external"
WEBSITE_CHART_THUMB_WIDTH_PX = 520
WEBSITE_CHART_FULL_DPI = 96
# Website-Charts: "compact" = Preis+RSI, kleinere PNG, 2-Spalten-Layout (mehr Signale sichtbar).
# "full" = 4 Panels (Preis, BB-Breite, RSI, Volumen) wie bisher.
WEBSITE_CHART_LAYOUT = "compact"
WEBSITE_CHART_DPI = 72
WEBSITE_CHART_DAYS_EACH_SIDE = 21
# Website + signals.json „signals“: nur FINAL-OOS (Classifier) — nie TRAIN/THRESHOLD-Kalender.
# Wird von save_scoring_artifacts / load_scoring_artifacts gesetzt, wenn in dieser Session gespeichert wurde.
_SCORING_ARTIFACT_SAVED_THIS_SESSION = False

# VIX-Regime-Ampel (3 Stufen, nur Website — filtert keine Signale).
# rot: VIX < VIX_AMPEL_YELLOW_MIN | gelb: bis GREEN | grün: ab GREEN
VIX_AMPEL_YELLOW_MIN = 20.0
VIX_AMPEL_GREEN_MIN = 25.0
# Rot-Kontext-Chips (OOS META+THR + FINAL, scripts/_scratch_validate_all_context.py)
VIX_RED_CHIP_VIX3M_RATIO_MAX = 1.16
VIX_RED_CHIP_SECTOR_HHI_MAX = 0.35
# =============================================================================
# 3. Pipeline: CV, Optuna, Zielvariablen, Split, Indikatoren, SEED_PARAMS
# =============================================================================
# (Ticker-Universum: Abschnitt 5 — hier nur Hyperparameter & Grids.)

# ── Fixed pipeline constants ─────────────────────────────────────────────────────
N_WF_SPLITS           = 5     # Walk-forward CV folds
GDELT_CHUNK_DAYS      = 45    # Nur bei NEWS_SOURCE="gdelt_api": Chunk-Länge in Tagen
OPT_MIN_PRECISION_BASE   = 0.64   # Phase 1 Base-XGB
OPT_MIN_PRECISION_META   = 0.99  # Phase 4 Meta-Learner (inkl. produktivem Threshold aus Meta-Optuna)
OPT_MIN_PRECISION = OPT_MIN_PRECISION_META  # Reports/PR-Plots: gleiches Gate wie Meta
# Phase 5: Mindestanzahl positiver Roh-Vorhersagen (prob>=t), damit Precision nicht trivial ist
PHASE5_MIN_SIGNALS    = 5

#

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
# Label-Logik bei festem Y (OPT_OPTIMIZE_Y_TARGETS=False). Standard: feste Band-Regel — **keine**
# Querschnitts-Labels. Nur bei Bedarf Y_LABEL_RULE auf "cross_sectional_top_q" setzen; dann greifen
# die CS_TARGET_*-Parameter unten.
Y_LABEL_RULE = "fixed_bands"
# --- Nur wenn Y_LABEL_RULE == "cross_sectional_top_q" (sonst ignoriert) ---
CS_TARGET_FORWARD_HORIZON = 20
CS_TARGET_TOP_Q = 0.1
CS_TARGET_MIN_TICKERS_PER_GROUP = 5
CS_TARGET_GROUPBY = "calendar_day"  # "calendar_day" | "sector_day"
# Festes Y (OPT_OPTIMIZE_Y_TARGETS=False) — ``target._create_target_one_ticker_fixed_bands``:
# (1) Grün: ∃ Fenster w ∈ [FIXED_Y_WINDOW_MIN, FIXED_Y_WINDOW_MAX] mit kum. Rendite ≥ FIXED_Y_RALLY_THRESHOLD;
#     alle Tage solcher Fenster werden zu grünen Segmenten vereinigt.
#     Optionaler Zusatz-Constraint: FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY=True erzwingt,
#     dass innerhalb eines qualifizierenden Fensters jeder Tages-Return strikt > 0 ist
#     (keine Einbrüche/Seitwärts-Tage im Rally-Fenster).
#     Zusatz-Constraint: FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION f in [0,1]: auf dem Haltetags-
#     Pfad (Entry-Open bis Exit) muss jeder Close >= open_entry * (1 - f) sein (f=0: streng wie
#     zuvor „kein Abverkauf“; f=0.01: max. 1% unter dem Entry-Open; f=1: keine Dipschranke).
# (2) Label 1:
#     FIXED_Y_LABEL_MODE = "segment_based" (Legacy): FIXED_Y_LEAD_DAYS vor Segmentbeginn; bei
#       L < FIXED_Y_SEGMENT_SPLIT zusätzlich die ersten FIXED_Y_ENTRY_DAYS grünen Tage.
#       Bei L ≥ split steuert FIXED_Y_LONG_SEGMENT_LABEL_MODE:
#         - "tail_exclude": alle grünen außer den letzten FIXED_Y_TAIL_EXCLUDE_DAYS
#         - "early_only": nur die ersten FIXED_Y_LONG_ENTRY_DAYS grünen Tage
#     FIXED_Y_LABEL_MODE = "entry_direct": target[t0]=1 genau dann, wenn ein Einstieg
#       am nächsten Open (t1) ein Fenster w∈[w_min,w_max] mit Rendite >= FIXED_Y_RALLY_THRESHOLD hat.
# w∈[2,8], Zielkumulatrendite >= 4 % im Fenster; f siehe unten.
FIXED_Y_WINDOW_MIN = 2
FIXED_Y_WINDOW_MAX = 8
FIXED_Y_RALLY_THRESHOLD = 0.045
FIXED_Y_LABEL_MODE = "rally_plus_entry"  # "segment_based" | "entry_direct" | "rally_plus_entry"
FIXED_Y_SEGMENT_SPLIT = 4
FIXED_Y_LEAD_DAYS = 2
FIXED_Y_ENTRY_DAYS = 1
FIXED_Y_TAIL_EXCLUDE_DAYS = 4
FIXED_Y_REQUIRE_STRICT_DAILY_UP_IN_RALLY = False
# 0.0: streng (kein Close unter Entry-Open); 0.01: max. 1% unter Entry-Open; 1.0: kein Dip-Floor.
FIXED_Y_MAX_DIP_BELOW_ENTRY_FRACTION = 0.015
FIXED_Y_LONG_SEGMENT_LABEL_MODE = "early_only"  # "tail_exclude" | "early_only"
FIXED_Y_LONG_ENTRY_DAYS = 2  # nur relevant bei FIXED_Y_LONG_SEGMENT_LABEL_MODE="early_only"
# Wieviel der Rally als target=1? — hängt vom Label-Modus ab:
# - entry_direct: kein Rally-Anteil; target=1 nur an Signal-Tagen t0 (gültiger Einstieg am Open t1).
# - segment_based: keine Prozentzahl; stattdessen LEAD/ENTRY/SPLIT/TAIL/LONG_* (Handelstage).
# - rally_plus_entry: Anteil je qualifizierendem Fenster [entry..exit] — erste
#   ceil(Fensterlänge · HEAD_FRACTION) Tage dieses Fensters (+ Vorlauf RALLY_SIGNAL_ENTRY_DAYS).
FIXED_Y_RALLY_PLUS_TARGET_SEGMENT_HEAD_FRACTION = 0.35
FIXED_Y_RALLY_SIGNAL_ENTRY_DAYS = 2
FIXED_Y_RALLY_PLUS_TARGET_OVERLAP_MODE = "union"  # "greedy_first" | "union"
EARLY_STOPPING_ROUNDS = 30
N_OPTUNA_TRIALS       = 150  # mehr Coverage im großen Kombinationsraum (TPE warm-up + Exploitation)
# Nur Optuna Phase 1 (Base-XGB): drosseln, wenn Trials trotz kleinem Universum langsam sind.
# UNIVERSE_FRACTION verkürzt nur Zeilen/Ticker in df_train; Laufzeit dominiert oft
# N_OPTUNA_TRIALS × OPTUNA_WF_SPLITS × (rebuild_target + XGB pro Fold).
OPTUNA_WF_SPLITS      = None  # None → N_WF_SPLITS; z.B. 3 weniger Folds pro Trial
# tqdm-Balken in Phase 1: None oder True = an; False = aus (ruhigeres Log, weniger Sonderzeichen | % in der Konsole).
OPTUNA_SHOW_PROGRESS_BAR = None
N_META_TRIALS         = 150
# Meta-Objective in den Fold-Validierungen:
# - "tp_precision": bisherige Zielfunktion (TP/Precision/FP-Run Constraints).
# - "signal_mean_return": mittlere Signal-Rendite über Haltehorizonte (z. B. 1..5 Tage).
# - "signal_win_rate": Anteil der Signale mit positivem mittlerem Horizon-Return.
META_OBJECTIVE_MODE = "tp_precision"
# Für signal_mean_return: pro Signal Mittelwert dieser Forward-Renditen, dann ZENTRAL-Aggregat
# (Mean oder Median, gesteuert via META_OBJECTIVE_SIGNAL_AGGREGATION) über alle Signale.
META_SIGNAL_RETURN_HORIZONS = (1, 2, 3, 4, 5)
# Zentral-Aggregat über die Signal-Returns innerhalb eines Folds (und auf dem THRESHOLD-Set in
# Phase 5). Default "median": robust gegen einzelne Mega-Gewinner/-Verlierer, die einen einzelnen
# Optuna-Trial sonst künstlich nach oben (oder unten) ziehen können — insbesondere bei den
# typischerweise ~5–30 Signalen je Fold dominiert ein Outlier sonst den Mean. "mean" reproduziert
# das alte Verhalten (literarisches signal_mean_return).
META_OBJECTIVE_SIGNAL_AGGREGATION = "median"

# Aggregation der Forward-Returns ÜBER DIE HORIZONTE pro einzelnem Signal
# (rlist = [close(t+h)/open(t+1)−1 für h ∈ META_SIGNAL_RETURN_HORIZONS]).
# Ein einzelner h kann durch Earnings-Reaktion oder Lücken extrem ausschlagen und dominiert
# sonst den per-Signal-Score, der wiederum in die Optuna-Zielfunktion einfließt.
#  - "mean": Mittelwert (Legacy, Outlier-empfindlich).
#  - "median": mittlerer Wert; bei 5 Horizonten = Tag-3-Return (sehr konservativ).
#  - "trimmed_mean": Mean nach Verwerfen je TRIM_FRAC am oberen+unteren Ende.
#    Bei 5 Horizonten und TRIM_FRAC=0.20 → je 1 oben/unten weg, Mean der mittleren 3.
#    Empfohlener Default für 5–8 Horizonte: robust ohne ganz so aggressiv wie Median.
META_OBJECTIVE_HORIZON_AGGREGATION = "trimmed_mean"
META_OBJECTIVE_HORIZON_TRIM_FRAC = 0.20
# Für signal_win_rate: optionaler Tiebreaker auf mean return (0.0 = nur Win-Rate).
META_OBJECTIVE_WINRATE_RETURN_TIEBREAKER = 0.1
# Optionaler Mindest-Count gefilterter Signale je Fold (darunter leichter Penalty).
META_OBJECTIVE_MIN_SIGNALS_PER_FOLD = 1
# Für signal_mean_return: Mindest-Signaldichte je Fold (Signale pro Handelstag, nach allen Filtern).
# 1.0 bedeutet im Schnitt mindestens 1 Signal pro Handelstag im Validierungs-Fold.
META_OBJECTIVE_MIN_SIGNALS_PER_DAY_PER_FOLD = 0.25
# Phase-5 Threshold-Wahl auf THRESHOLD-Set: optionale weiche Coverage-Behandlung.
# Default = False (hartes Gate wie in der Meta-CV bleibt aktiv → Repräsentativitätsschutz:
# zu wenige Signale = unzuverlässige Mean-Return-Schätzung). Falls True, ersetzt die
# nachstehende dreistufige Logik den harten Cut-off:
#   * coverage >= MIN_SIGNALS_PER_DAY                             → fs = 1 + return.
#   * FLOOR_FRACTION * MIN_SIGNALS_PER_DAY <= coverage < target   → fs = 1 + return − SOFT_PENALTY_WEIGHT * (target − coverage).
#   * coverage < FLOOR_FRACTION * MIN_SIGNALS_PER_DAY             → hart abgestraft.
META_PHASE5_SOFT_COVERAGE_ENABLED = False
META_PHASE5_DENSITY_FLOOR_FRACTION = 0.5
META_PHASE5_DENSITY_SOFT_PENALTY_WEIGHT = 1.0
# Phase-5 Seed: bei True nutzt die Threshold-Wahl als Seed den Median der nested
# Thresholds aus dem besten Meta-CV-Trial (mehr Datenbasis als der reine Trial-Parameter).
# Der Seed wird – wenn die Meta-Probabilities kalibriert werden – durch dieselbe
# Kalibrierung geschickt, damit Seed und Threshold-Grid auf derselben Skala leben.
META_PHASE5_SEED_USE_NESTED_THR_MEAN = True
# Wahrscheinlichkeitsskalierung für Meta-Classifier-Ausgaben:
# - "none": rohe predict_proba
# - "sigmoid": Platt-Skalierung (logistische Kalibrierung)
# - "isotonic": isotone Kalibrierung
META_PROBA_CALIBRATION_METHOD = "sigmoid"
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
CONSECUTIVE_DAYS     = 1     # wie SEED_PARAMS (Post-Filter)
SIGNAL_COOLDOWN_DAYS = 2

# ── Anti-Peak (nur apply_signal_filters / Website — nicht in Optuna-CV) ─────
# Viele Fehlsignale sitzen direkt am lokalen Kurs-High; Filter verlangt Abstand zum N-Tage-High.
SIGNAL_SKIP_NEAR_PEAK = True
PEAK_LOOKBACK_DAYS = 20
PEAK_MIN_DIST_FROM_HIGH_PCT = 0.012   # mind. 1.2 % unter Rolling-High (tunable 0.008–0.02)
SIGNAL_MAX_RSI = 78.0                 # kein Kauf-Signal wenn RSI darüber; None = aus
# Hard-Filter: wenn Vortag positiv und vol_stress als 20d-ZScore darüber liegt -> Signal löschen.
SIGNAL_MAX_VOL_STRESS_Z = 2.0         # None = aus
# Blue-Sky-Schutz: Breakout ohne Volumen-Bestätigung (Vortag) wird verworfen.
SIGNAL_MIN_BLUE_SKY_VOLUME_Z = 0.5    # None = aus
# Dynamischer Threshold-Veto-Block (Meta/Scoring): baseline threshold * multipliers.
MULT_FINAL_THRESHOLD_1 = 1.0
MULT_FINAL_THRESHOLD_2 = 1.0
MULT_FINAL_THRESHOLD_3 = 1.0
DYN_VVIX_TRIGGER = 8.2
DYN_RSI_TRIGGER = 75.0
DYN_BB_PBAND_TRIGGER = 1.02

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

# Base-XGB: Klassenimbalance über scale_pos_weight (neg/pos) je Fold/Bootstrap ausgleichen.
BASE_USE_SCALE_POS_WEIGHT = True
# Meta-XGB: Klassenimbalance über scale_pos_weight (neg/pos) je Inner-Train-Split ausgleichen.
META_USE_SCALE_POS_WEIGHT = True

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
    news_tone_roll=1,
    news_extra_zscore_w=20,
    news_extra_tone_accel=True,
    news_extra_macro_sec_diff=True,
    btc_momentum_z_window=60,
    market_breadth_z_window=60,
    rel_momentum_window=20,
    adr_window=20,
    breakout_lookback_window=252,
    vcp_window=10,
    btc_corr_window=20,
    # Optuna-tunbare Risk-/Liquidity-/VCP-/Breakout-Fenster (Default = robuster Mittelweg).
    yz_vol_window=20,
    downside_vol_window=60,
    ret_moment_window=60,
    amihud_window=20,
    vcp_lower_low_window=60,
    breakout_volume_trigger_z=1.0,
    news_add_sign_confirmation=True,
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
    signal_max_vol_stress_z=2.0,
    signal_min_blue_sky_volume_z=0.5,
    mult_final_threshold_1=1.0,
    mult_final_threshold_2=1.0,
    mult_final_threshold_3=1.0,
    dyn_vvix_trigger=8.2,
    dyn_rsi_trigger=75.0,
    dyn_bb_pband_trigger=1.02,
)

# =============================================================================
# 4. News / BigQuery (GKG) / GDELT-API-Fallback (V2Themes, GCAM, optional Anker-Orgs)
# =============================================================================

# ── News / GDELT grids (Optuna wählt ein Tag-Tripel) ─────────────────────────
USE_NEWS_SENTIMENT = True
# Momentum auf geglätteter Ton-Serie; Vol-MA für Spike; zusätzliche Ton-Glättung vor Momentum
NEWS_MOM_WINDOWS = [3, 5, 7]
NEWS_VOL_MA_WINDOWS = [10, 20]
# Fokus auf stabilere Drift-/Trendreaktionen (T+1/T+2), weniger ultra-kurzfristiges Rauschen.
NEWS_TONE_ROLL_WINDOWS = [3, 5, 10]

# Zusätzliche News-Spalten — Optuna Phase 1 wählt je Trial ein Tripel (wie news_mom_w …)
# 5 = kurzes „Shock“-Fenster (relativ zur lokalen 5d-Basis); 0 = kein Z-Block in FEAT_COLS (Optuna darf das wählen)
NEWS_EXTRA_ZSCORE_WINDOWS = [0, 5, 10, 20, 30]
NEWS_EXTRA_TONE_ACCEL_OPTIONS = [False, True]
NEWS_EXTRA_MACRO_SEC_DIFF_OPTIONS = [False, True]
# assemble_features: welche News-Spalten am Ende per Fillna/0 angelegt werden.
# "optuna_union" = Vereinigung wie Optuna-Raster (bei gleicher Config oft gleiche Anzahl wie all_news_model).
# "all_news_model" = alle Namen aus all_news_model_cols() (Legacy).
# "none" = keinen News-Fill (nur was die Merges liefern; für Experten).
FEATURE_ASSEMBLE_NEWS_FILL = "optuna_union"  # "all_news_model" | "optuna_union" | "none"
# Fehlende/ungültige Werte als klarer Sentinel (statt 0.0), damit „missing“ getrennt von „neutral“ bleibt.
FEATURE_NUMERIC_NAN_SENTINEL = -1e8

# Kontext-Features: Optuna wählt je Trial ein Fenster (Indizes siehe SEED_PARAMS).
BTC_MOMENTUM_Z_WINDOWS = [20, 60, 120]
MARKET_BREADTH_Z_WINDOWS = [20, 60, 120]
REL_MOMENTUM_WINDOWS = [10, 20, 60]
ADR_WINDOWS = [10, 20]
BREAKOUT_LOOKBACK_WINDOWS = [60, 120, 252]
VCP_WINDOWS = [5, 10, 20]
BTC_CORR_WINDOWS = [20, 60, 120]

# Realisierte Vola / Schiefe / Kurtosis: pro Ticker rollende Statistiken auf Tagesrenditen.
# Kleinere Fenster sind bei kurzen Rallies oft trennschärfer; 60d als robuster Anker.
# Optuna wählt je Trial **ein** Fenster pro Liste (analog RSI/BB/SMA-Raster).
DOWNSIDE_VOL_WINDOWS = [20, 60, 120]
RET_MOMENT_WINDOWS = [20, 60, 120]
YANG_ZHANG_WINDOWS = [10, 20, 60]
# Amihud-Illiquidität: Mittelwert |ret| / Dollarvolumen über das Fenster (höher = illiquider).
AMIHUD_WINDOWS = [10, 20, 60]
# VCP-Lower-Lows: in wie vielen der letzten N Tage hat ``vcp_tightness_{vw}d`` ein neues
# Minimum gegenüber dem laufenden Tief markiert. Stärker = engere/anhaltendere Kontraktion.
VCP_LOWER_LOW_WINDOWS = [20, 60, 120]
# Breakout-Bestätigung: ``blue_sky_breakout_{w}d`` AND ``volume_zscore`` >= Trigger.
# Optuna wählt einen Trigger aus dem Raster; die Indikator-Pipeline berechnet je Kombination
# ``(bwk, trigger)`` eine eigene Spalte ``breakout_volume_confirmed_{bwk}_z{trigger}d``.
BREAKOUT_VOLUME_TRIGGER_Z_OPTIONS = [0.5, 1.0, 1.5, 2.0]

# ── Feature pre-screening (vor Optuna Phase 1) ───────────────────────────────
# SHAP-Importance über 5 walk-forward Folds + Boruta-Schatten als Rausch-Floor +
# familien-bewusste Auswahl (Top-2 pro Familie immer behalten). Schreibt ein
# Artefakt nach ``FEATURE_PRESCREEN_DIR`` (JSON) und ``cfg._FEATURE_PRESCREEN_ARTIFACT``.
# Optuna nutzt anschließend ``effective_window_grid(name)`` statt direkt ``cfg.<list>``.
FEATURE_PRESCREEN_ENABLED = True  # True = Pre-Screen vor Phase 1 ausführen (User-Default).
FEATURE_PRESCREEN_N_SHADOWS = 20  # Anzahl Schatten-Features pro Fold für Rausch-Floor.
FEATURE_PRESCREEN_NOISE_QUANTILE = 0.95  # Quantil der Schatten-Importance = Floor.
FEATURE_PRESCREEN_FAMILY_TOPK = 2  # Top-K pro Familie immer behalten.
FEATURE_PRESCREEN_GLOBAL_TOPK = 30  # Top-K global immer behalten (über alle Familien).
FEATURE_PRESCREEN_MACRO_TOPK = 5  # Top-K der mr_*/regime_*-Gruppe immer behalten.
FEATURE_PRESCREEN_SHAP_SAMPLE_PER_FOLD = 20000  # max. Zeilen für SHAP pro Fold (Speed).
FEATURE_PRESCREEN_AUC_TOL = 0.005  # Abbruch wenn AUC(kept) < AUC(all) − tol.
FEATURE_PRESCREEN_PRAUC_TOL = 0.005  # Abbruch wenn PR-AUC(kept) < PR-AUC(all) − tol.
FEATURE_PRESCREEN_DIR = os.path.join(os.getcwd(), "data")
FEATURE_PRESCREEN_ARTIFACT_NAME = "feature_prescreen_v1.json"
FEATURE_PRESCREEN_REUSE_SAME_CALENDAR_DAY = True
# Wird zur Laufzeit von ``feature_prescreen.run_feature_prescreen`` befüllt.
_FEATURE_PRESCREEN_ARTIFACT: dict | None = None

# Volles News-Fenster-Grid: News liegt nur in Shard-Dateien (FEATURE_SHARD_DIR), nicht als breite Spalten in df_features.
FEATURE_SHARD_DIR = os.path.join(os.getcwd(), "data", "feature_shards_news")
# True: Läuft am selben Kalendertag kein vollständiger Rebuild, wenn news_shards_manifest.json
# ``built_on``=heute trägt und alle benötigten Shard-Dateien existieren (ungeachtet exakter input_signature).
# Praktisch „max. einmal pro Tag“ neu bauen; Später-Runs/Tag sparen Zeit (Shards können minimal älter sein).
NEWS_SHARDS_REUSE_SAME_CALENDAR_DAY = True
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
# Wenn True: Keyword-OR (SECTOR_KEYWORDS) kann in SQL mit dem Theme kombiniert werden.
# Leere Channel-Liste bedeutet: auf alle Sektorkanäle anwenden (siehe unten FOR_BASE_AGG).
BQ_SECTOR_THEME_KEYWORD_CONJUNCTION = True
BQ_SECTOR_THEME_KEYWORD_CHANNELS: tuple[str, ...] = ()
# Wenn True UND BQ_SECTOR_THEME_KEYWORD_CONJUNCTION True: Basis-Sektorkanal ({ch}_tone/{ch}_vol/GCAM)
# nutzt (Theme) AND (Keywords) — sehr streng, oft wenig Artikel -> flache Tone-Features.
# False: Basis-Kanal nur Theme (wie früher, mehr Varianz); Keywords bleiben bei Anker-Spalten
# (_anchor_tone etc.) aktiv, sofern CONJUNCTION an ist.
BQ_SECTOR_KEYWORD_CONJUNCTION_FOR_BASE_AGG = False

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

