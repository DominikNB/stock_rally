"""
Konfiguration und Hilfsfunktionen für Spaltennamen.

Alle Pipeline-Zustände (DataFrames, Modelle, ``best_params``, …) werden zur Laufzeit
als Attribute dieses Moduls gesetzt (gemeinsamer Namespace für alle Schritte).
``save_scoring_artifacts`` / ``load_scoring_artifacts`` delegieren an ``lib.scoring_persist``
und verwenden ``globals()`` dieses Moduls, solange die Pipeline über dieses Paket läuft.

Datei-Gliederung (alles in Lesereihenfolge von oben nach unten):
  1. Pfade, Google-ADC, Kalenderdaten
  2. Scoring-Artefakt (SCORING_ONLY, load/save)
  3. Pipeline: CV/Optuna, Labels, Anti-Peak, Split, Indikator-Grids, SEED_PARAMS
  4. News: GDELT/BigQuery, GKG-Kanäle, Geo, V2Themes-SQL, API-Stichwörter
  5. Universum: TICKERS_BY_SECTOR, Ableitungen, COMPANY_NAMES
  6. Hilfsfunktionen (Feature-Spalten, Rename-Map)
"""

import datetime
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path.cwd().resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Google Cloud / BigQuery: ADC + Projekt (vor BQ_PROJECT_ID) ────────────
ADC_PATH = r"C:\Users\HP\AppData\Roaming\gcloud\application_default_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ADC_PATH
os.environ["GOOGLE_CLOUD_PROJECT"] = "gedelt-calls"
print(f"Credentials geladen von: {ADC_PATH}")
if not os.path.isfile(ADC_PATH):
    print("WARNUNG: Datei nicht gefunden — Pfad prüfen oder gcloud auth application-default login")

# =============================================================================
# 1. Kalender & Laufmodus (Kurse / Training)
# =============================================================================
# ── Dates ──────────────────────────────────────────────────────────────────
# Walk-forward: BASE → META → THRESHOLD → FINAL = disjunkte Zeitfenster (Purge zwischen Stufen).
# FINAL = letztes Fenster (Final-Test); TRAIN_END_DATE = END_DATE = Kalenderende des Laufs (aktueller Tag).
START_DATE = '2018-01-01'  # Kurs-Download / Labels; News siehe NEWS_EXTRA_HISTORY_* und NEWS_BQ_*
END_DATE        = datetime.date.today().strftime('%Y-%m-%d')           # Download bis heute
TRAIN_END_DATE  = datetime.date.today().strftime('%Y-%m-%d')           # gleiches Kalenderende wie END_DATE
print(f'Download:  {START_DATE} → {END_DATE}')
print(f'Training:  {START_DATE} → {TRAIN_END_DATE}  (bis zum aktuellen Datum)')


# ── SCORING_ONLY: gespeicherte Modelle + Threshold, ohne Neu-Training ───────
# (gehört logisch zu Abschnitt 1 — Laufmodus)
# True  → Datenpfad + Scoring/HTML; Training (Phasen 12–16) wird übersprungen.
# False → volles Training; Artefakt wird nach Meta/Threshold-Phase automatisch geschrieben.
SCORING_ONLY = True
SCORING_ARTIFACT_PATH = Path("models") / "scoring_artifacts.joblib"
# Phase 17: Signal-Filter pro Ticker parallel (joblib loky). -1 = alle Kerne, 1 = seriell.
PHASE17_SIGNAL_FILTER_JOBS = -1
# Fortschrittslog alle N Ticker; 0 = ca. 10 Stufen (nt//10).
PHASE17_SIGNAL_FILTER_PROGRESS_BATCH = 0
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


# =============================================================================
# 3. Pipeline: CV, Optuna, Zielvariablen, Split, Indikatoren, SEED_PARAMS
# =============================================================================
# (Ticker-Universum steht unten in Abschnitt 5 — hier nur Hyperparameter & Grids.)

# ── Fixed pipeline constants ─────────────────────────────────────────────────────
N_WF_SPLITS           = 5     # Walk-forward CV folds
GDELT_CHUNK_DAYS      = 45    # GDELT API: kürzere Sub-Chunks (vorher fest 90d)
OPT_MIN_PRECISION_BASE   = 0.6   # Phase 1 Base-XGB
OPT_MIN_PRECISION_META   = 0.7   # Phase 4 Meta-Learner
OPT_MIN_PRECISION_THRESHOLD = 0.95  # Phase 5 Threshold-Kalibrierung / PR-Plots
OPT_MIN_PRECISION = OPT_MIN_PRECISION_THRESHOLD  # Alias: find_precision_threshold, Reports
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
# False: feste Regel — grüner Bereich = Rendite > FIXED_Y_RALLY_THRESHOLD innerhalb von
# FIXED_Y_WINDOW_MIN..FIXED_Y_WINDOW_MAX Handelstagen; Targets nur nach fester Vorschrift.
OPT_OPTIMIZE_Y_TARGETS = False
FIXED_Y_WINDOW_MIN = 3
FIXED_Y_WINDOW_MAX = 10
FIXED_Y_RALLY_THRESHOLD = 0.06
FIXED_Y_SEGMENT_SPLIT = 5
EARLY_STOPPING_ROUNDS = 30
N_OPTUNA_TRIALS       = 300  # erhöht wegen News-Feature-Grid
# Nur Phase-1-Studie (Base-XGB): hier drosseln, wenn Optuna trotz kleinem Universum langsam ist.
# UNIVERSE_FRACTION verkürzt Datenpipeline; jeder Trial kostet noch rebuild_target + WF×XGB.
OPTUNA_TRIALS         = None  # None → N_OPTUNA_TRIALS; z.B. 40 für schnelle Tests
OPTUNA_WF_SPLITS      = None  # None → N_WF_SPLITS; z.B. 3 weniger Folds pro Trial
N_META_TRIALS         = 300
# Cell 13a: SHAP → Meta-Stacking — Top-K Roh-Features neben Base-Probabilities (Cell 14)
META_SHAP_TOP_K       = 10
RANDOM_STATE          = 42
N_WORKERS             = os.cpu_count()

# ── Optuna initial values (überschrieben durch Cell 13 nach Optimierung) ─────────
# create_target() in Cell 11 läuft VOR Optuna — Werte sollten mit SEED_PARAMS konsistent sein.
RETURN_WINDOW        = 4     # zuletzt: Optuna trial 154 (Base-XGB)
RALLY_THRESHOLD      = 0.07423891180366396
# „Early-only“-Label: positives = Vorlauf (LEAD_DAYS) + kurze Einstiegszone (ENTRY_DAYS),
# nicht die gesamte grüne Rally. Optuna sucht entry_days in [1, ENTRY_DAYS_OPT_MAX].
ENTRY_DAYS_OPT_MAX   = 4
LEAD_DAYS            = 4
ENTRY_DAYS           = 3
# Positives Label nur, wenn ab diesem Tag noch mind. so viele Rally-Handelstage „übrig“ sind
# (inkl. heute bis Segmentende) — kurze grüne Reste / Rally-Ende werden nicht als Ziel markiert.
MIN_RALLY_TAIL_DAYS  = 5
CONSECUTIVE_DAYS     = 2
SIGNAL_COOLDOWN_DAYS = 4

# ── Anti-Peak (nur apply_signal_filters / Website — nicht in Optuna-CV) ─────
# Viele Fehlsignale sitzen direkt am lokalen Kurs-High; Filter verlangt Abstand zum N-Tage-High.
SIGNAL_SKIP_NEAR_PEAK = True
PEAK_LOOKBACK_DAYS = 20
PEAK_MIN_DIST_FROM_HIGH_PCT = 0.012   # mind. 1.2 % unter Rolling-High (tunable 0.008–0.02)
SIGNAL_MAX_RSI = 78.0                 # kein Kauf-Signal wenn RSI darüber; None = aus

# ── Trainings-Universum ─────────────────────────────────────────────────────
# Maximale Anzahl Wertpapiere im TRAIN_BASE-Set — nur bei SPLIT_MODE="ticker" (Legacy).
# Bei SPLIT_MODE="time" nutzen alle Stufen dieselben Ticker (TIME_SPLIT_FRAC_*).
# Empfehlung: 30–60. Höher = langsamer, aber potenziell besseres Modell.
MAX_TRAIN_TICKERS   = 45
# Schneller Run: nur einen zufälligen Anteil der geladenen Ticker (nach assemble_features)
UNIVERSE_FRACTION = 0.05  # 1.0 = alle; <1 wählt Tickers VOR Download/Features (nicht erst danach)
UNIVERSE_SAMPLE_SEED = 42
# Train/Test-Split: "time" = gleiches Ticker-Universum, disjunkte Zeiträume (empfohlen).
# "ticker" = Legacy: verschiedene Ticker pro Stufe, gleicher Kalender bis TRAIN_END_DATE.
SPLIT_MODE = "time"
# Anteile auf allen Handelstagen [START … TRAIN_END_DATE]; walk-forward, keine Überlappung der Stufen (Purge Base→Meta).
TIME_SPLIT_FRAC_BASE = 0.45
TIME_SPLIT_FRAC_META = 0.25
TIME_SPLIT_FRAC_THRESHOLD = 0.15
# Rest = FINAL: letztes Fenster = eigentlicher Final-Test bis zum Kalenderende (s. TRAIN_END_DATE).
TIME_PURGE_TRADING_DAYS = 5

# ── Indicator parameter grids ───────────────────────────────────────────────
RSI_WINDOWS  = [7, 10, 14, 21]
BB_WINDOWS   = [15, 20, 25]
SMA_WINDOWS  = [30, 50, 70]

# ── Seed params für enqueue_trial (Optuna Base-XGB) ─────────────────────────
# Referenz-Seed für enqueue_trial (früherer Optuna-Lauf, z. B. Trial 154).
SEED_PARAMS = dict(
    return_window=4,
    rally_threshold=0.07423891180366396,
    lead_days=4,
    entry_days=3,
    min_rally_tail_days=5,
    consecutive_days=2,
    signal_cooldown_days=4,
    threshold=0.6338651351186617,
    rsi_window=14,
    bb_window=25,
    sma_window=50,
    news_mom_w=3,
    news_vol_ma=20,
    news_tone_roll=0,
    news_extra_zscore_w=20,
    news_extra_tone_accel=True,
    news_extra_macro_sec_diff=True,
    grow_policy='depthwise',
    max_leaves=910,
    max_bin=64,
    max_depth=6,
    min_child_weight=9,
    gamma=5.7204046577916365,
    reg_alpha=0.00044267707418928947,
    reg_lambda=1.1663163488001458,
    learning_rate=0.013316606543583374,
    n_estimators=1263,
    subsample=0.5896905968497552,
    colsample_bytree=0.5602713933350929,
    focal_gamma=0.5063350144294034,
    focal_alpha=0.38191769324282254,
    signal_skip_near_peak=True,
    peak_lookback_days=20,
    peak_min_dist_from_high_pct=0.012,
    signal_max_rsi=78.0,
)

# =============================================================================
# 4. News / GDELT / BigQuery / GKG (Zeitreihen + Kanal-SQL auf V2Themes)
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

NEWS_CACHE_DIR = os.path.join(os.getcwd(), 'data')
NEWS_CACHE_FILE = os.path.join(NEWS_CACHE_DIR, 'news_gdelt_cache.pkl')
# News: BigQuery (GDELT events, kein API-Rate-Limit) oder Legacy HTTP-Doc-API
NEWS_SOURCE = "bigquery"  # "bigquery" | "gdelt_api"
# GCP-Projekt mit BigQuery-Billing. Auth: gcloud auth application-default login
# oder GOOGLE_APPLICATION_CREDENTIALS. Projekt sonst über gcloud/ADC.
BQ_PROJECT_ID = "gedelt-calls"
os.environ["GOOGLE_CLOUD_PROJECT"] = "gedelt-calls"
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

# Feste GKG-Theme-Tripel für SQL (leer = keine zusätzlichen Theme-Spalten).
GKG_THEME_SQL_TRIPLES = []

# GCAM-Schlüssel: nicht leer = feste Liste (überschreibt Auto). Sonst: Must-Have + optional Datei/Exploration.
# GKG_GCAM_EXPLORE_KEYS=True: BigQuery holt zusätzliche Top-c* (Beifang), ohne Must-Haves zu duplizieren.
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
GKG_GCAM_EXPLORE_KEYS = True
# Zusätzliche explorative c*-Keys (nach Häufigkeit), Must-Haves nicht gezählt.
GKG_GCAM_EXPLORE_EXTRA_N = 8
GKG_GCAM_AUTO_TOP_N = 8  # Fallback wenn GKG_GCAM_EXPLORE_EXTRA_N fehlt
GKG_GCAM_KEYS_PATH = Path("data") / "gkg_gcam_metric_keys.json"

# Abgeleitete GCAM-Differenzen (features.py): nur wenn beide Basis-Spalten in sentiment_df existieren.
# (slug, minuend_key, subtrahend_key) — z. B. fin. Positivität − Negativität; Erfolg − Unsicherheit.
GCAM_INTERACTION_DIFFS: tuple[tuple[str, str, str], ...] = (
    ("inter_fin_spread", "c18.159", "c18.158"),
    ("inter_confidence", "vnt:success", "c18.161"),
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
SECTOR_KEYWORDS = {
    'heat_pump':  ['heat pump', 'HVAC', 'refrigerant', 'air conditioning', 'Daikin', 'Carrier'],
    'tech':       ['semiconductor', 'AI chip', 'cloud computing', 'software'],
    'finance':    ['interest rate', 'bank earnings', 'Fed', 'credit'],
    'healthcare': ['FDA approval', 'clinical trial', 'pharma', 'biotech'],
    'consumer':   ['consumer spending', 'retail sales', 'inflation'],
    'industrial': ['manufacturing PMI', 'industrial output', 'capex'],
    'energy':     ['oil price', 'crude', 'OPEC', 'natural gas'],
    'crypto':     ['Bitcoin', 'Ethereum', 'crypto regulation', 'DeFi'],
    'automotive': ['automotive industry', 'electric vehicle', 'car manufacturing'],
    'materials':  ['chemical industry', 'steel', 'commodities'],
    'real_estate': ['real estate', 'mortgage rates', 'housing market'],
    'telecom':    ['telecom', '5G', 'broadband'],
    'media':      ['media earnings', 'streaming', 'advertising'],
}

# BigQuery GKG: V2Themes-Filter (Makro + pro Sektor-Kanal)
MACRO_BQ_THEME_WHERE = (
    "(V2Themes LIKE '%ECON_INTERESTRATES%' OR V2Themes LIKE '%ECON_INFLATION%' "
    "OR V2Themes LIKE '%MONETARY%' OR V2Themes LIKE '%ECON_CENTRALBANK%')"
)

SECTOR_BQ_THEME_WHERE = {
    'heat_pump': (
        "(V2Themes LIKE '%ENV_GREEN_ENERGY%' OR V2Themes LIKE '%ECON_SUBSIDIES%' "
        "OR V2Themes LIKE '%ENV_ENERGY%')"
    ),
    'tech': (
        "(V2Themes LIKE '%TECH%' OR V2Themes LIKE '%ECON_%' OR V2Themes LIKE '%AI_%' "
        "OR V2Themes LIKE '%SEMICONDUCTOR%')"
    ),
    'finance': (
        "(V2Themes LIKE '%ECON_%' OR V2Themes LIKE '%FINANCE%' OR V2Themes LIKE '%BANK%' "
        "OR V2Themes LIKE '%MONETARY%' OR V2Themes LIKE '%INFLATION%')"
    ),
    'healthcare': (
        "(V2Themes LIKE '%HEALTH%' OR V2Themes LIKE '%MEDICAL%' OR V2Themes LIKE '%FDA%' "
        "OR V2Themes LIKE '%PHARMACEUTICAL%')"
    ),
    'consumer': (
        "(V2Themes LIKE '%CONSUMER%' OR V2Themes LIKE '%RETAIL%' OR V2Themes LIKE '%ECON_%' "
        "OR V2Themes LIKE '%INFLATION%')"
    ),
    'industrial': (
        "(V2Themes LIKE '%MANUFACTURING%' OR V2Themes LIKE '%ECON_%' OR V2Themes LIKE '%INDUSTRY%')"
    ),
    'energy': (
        "(V2Themes LIKE '%ENERGY%' OR V2Themes LIKE '%OIL%' OR V2Themes LIKE '%GAS%' "
        "OR V2Themes LIKE '%PETROLEUM%' OR V2Themes LIKE '%OPEC%')"
    ),
    'crypto': (
        "(V2Themes LIKE '%CRYPTO%' OR V2Themes LIKE '%BITCOIN%' OR V2Themes LIKE '%BLOCKCHAIN%' "
        "OR V2Themes LIKE '%ETHEREUM%')"
    ),
    'automotive': (
        "(V2Themes LIKE '%AUTO%' OR V2Themes LIKE '%VEHICLE%' OR V2Themes LIKE '%ELECTRIC%' "
        "OR V2Themes LIKE '%AUTOMOTIVE%')"
    ),
    'materials': (
        "(V2Themes LIKE '%STEEL%' OR V2Themes LIKE '%COMMODITY%' OR V2Themes LIKE '%CHEMICAL%')"
    ),
    'real_estate': (
        "(V2Themes LIKE '%REALESTATE%' OR V2Themes LIKE '%HOUSING%' OR V2Themes LIKE '%MORTGAGE%' "
        "OR V2Themes LIKE '%PROPERTY%')"
    ),
    'telecom': (
        "(V2Themes LIKE '%TELECOM%' OR V2Themes LIKE '%5G%' OR V2Themes LIKE '%BROADBAND%')"
    ),
    'media': (
        "(V2Themes LIKE '%MEDIA%' OR V2Themes LIKE '%STREAMING%' OR V2Themes LIKE '%ADVERTISING%')"
    ),
}

# =============================================================================
# 5. Universum: Ticker, Sektoren, Anzeigenamen
# =============================================================================
# DAX40 / MDAX50 / SDAX70 vollständig + internationale Referenzwerte
# (Keys müssen zu SECTOR_BQ_THEME_WHERE passen.)

TICKERS_BY_SECTOR = {
    # ── Wärmepumpen & HVAC ─────────────────────────────────────────────────
    'heat_pump':  ['NIBE-B.ST', '6367.T', 'CARR', 'TT', 'JCI', 'LII',
                   'AOS', 'WSO', '6503.T', 'AALB.AS'],

    # ── Technologie ────────────────────────────────────────────────────────
    # DAX: SAP, IFX  |  MDAX: NEM(Nemetschek), BC8  |  SDAX: TMV, COK, YSN, GFT, NA9, AOF, ELG, AIXA, SMHN, SANT
    'tech':       ['AAPL', 'MSFT', 'NVDA', 'CRM', 'CSCO', 'INTC', 'IBM', 'ASML',
                   'SAP.DE', 'IFX.DE',
                   'NEM.DE', 'BC8.DE',
                   'TMV.DE', 'COK.DE', 'YSN.DE', 'GFT.DE', 'NA9.DE',
                   'AOF.DE', 'ELG.DE', 'AIXA.DE', 'SMHN.DE', 'SANT.DE'],

    # ── Finanzen ───────────────────────────────────────────────────────────
    # DAX: DB1, HNR1  |  MDAX: TLX  |  SDAX: MLP, GLJ, PBB, HABA, WUW, DBAG, HYQ
    'finance':    ['JPM', 'GS', 'AXP', 'V',
                   'ALV.DE', 'MUV2.DE', 'DBK.DE', 'CBK.DE', 'BNP.PA',
                   'DB1.DE', 'HNR1.DE',
                   'TLX.DE',
                   'MLP.DE', 'GLJ.DE', 'PBB.DE', 'WUW.DE', 'DBAG.DE', 'HYQ.DE'],

    # ── Gesundheit ─────────────────────────────────────────────────────────
    # DAX: FME, SHL  |  MDAX: SRT3  |  SDAX: AFX, EUZ, DMP, GXI, DRW3, O2BC, EVT
    'healthcare': ['JNJ', 'UNH', 'MRK', 'AMGN', 'NVS',
                   'BAYN.DE', 'FRE.DE', 'MRK.DE', 'FME.DE', 'SHL.DE',
                   'SRT3.DE',
                   'AFX.DE', 'EUZ.DE', 'DMP.DE', 'GXI.DE', 'DRW3.DE', 'O2BC.DE', 'EVT.DE'],

    # ── Konsumgüter ────────────────────────────────────────────────────────
    # DAX: BEI  |  SDAX: FIE, HBH, DOU, SZU, HFG, KWS, EVD, DHER
    'consumer':   ['KO', 'MCD', 'NKE', 'PG', 'WMT', 'HD', 'DIS',
                   'ADS.DE', 'BOSS.DE', 'HEN3.DE', 'PUM.DE', 'ZAL.DE', 'LVMH.PA',
                   'BEI.DE',
                   'FIE.DE', 'HBH.DE', 'DOU.DE', 'SZU.DE', 'HFG.DE', 'KWS.DE',
                   'EVD.DE', 'DHER.DE'],

    # ── Industrie ──────────────────────────────────────────────────────────
    # DAX: SIE, RHM, MTX, AIR, DHL, G1A  |  MDAX: KBX, HOC, KGX, JUN3, TKA, HAG, R3NK, LHA, FRA, HLAG, RAA
    # SDAX: DUE, JST, SFQ, NOEJ, VOS, WAC, INH, MUX, STM, KSB, HDD
    'industrial': ['CAT', 'BA', 'HON', 'MMM', 'SHW', 'TRV', 'DOW',
                   'SIE.DE', 'RHM.DE', 'MTX.DE', 'AIR.DE', 'DHL.DE', 'G1A.DE',
                   'KBX.DE', 'HOC.DE', 'KGX.DE', 'JUN3.DE', 'TKA.DE', 'HAG.DE',
                   'R3NK.DE', 'LHA.DE', 'FRA.DE', 'HLAG.DE', 'RAA.DE',
                   'DUE.DE', 'JST.DE', 'SFQ.DE', 'NOEJ.DE', 'VOS.DE',
                   'WAC.DE', 'INH.DE', 'MUX.DE', 'STM.DE', 'KSB.DE', 'HDD.DE'],

    # ── Energie ────────────────────────────────────────────────────────────
    # DAX: ENR  |  MDAX: NDX1  |  SDAX: S92, VBK, EKT, PNE3, F3C, VH2
    'energy':     ['CVX', 'XOM', 'BP',
                   'RWE.DE', 'EOAN.DE', 'NEE', 'DUK', 'IBE.MC',
                   'ENR.DE',
                   'NDX1.DE',
                   'S92.DE', 'VBK.DE', 'EKT.DE', 'PNE3.DE', 'F3C.DE', 'VH2.DE'],

    # ── Krypto ─────────────────────────────────────────────────────────────
    'crypto':     ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'],

    # ── Automotive ─────────────────────────────────────────────────────────
    # DAX: VOW3, BMW, MBG, CON, PAH3, P911, DTG  |  MDAX: TGR, SHA
    'automotive': ['VOW3.DE', 'BMW.DE', 'MBG.DE', 'CON.DE', 'PAH3.DE', 'P911.DE',
                   'DTG.DE',
                   'TM', 'STLA', 'GM', 'F',
                   'TGR.DE', 'SHA.DE'],

    # ── Rohstoffe & Chemie ─────────────────────────────────────────────────
    # DAX: BAS, BNR, HEI, SY1  |  MDAX: EVK, LXS, NDA, SDO, FPE3  |  SDAX: ACT, WAF, BFSA, KCO
    'materials':  ['BAS.DE', 'LIN', 'WCH.DE', 'APD', 'LYB', 'BNR.DE', 'NEM',
                   'HEI.DE', 'SY1.DE',
                   'EVK.DE', 'LXS.DE', 'NDA.DE', 'SDO.DE', 'FPE3.DE',
                   'ACT.DE', 'WAF.DE', 'BFSA.DE', 'KCO.DE'],

    # ── Immobilien ─────────────────────────────────────────────────────────
    # DAX: VNA  |  MDAX: LEG  |  SDAX: GYC, DEQ, HABA, PAT
    'real_estate':['VNA.DE', 'LEG.DE', 'O', 'AMT', 'SPG', 'WDP.BR',
                   'GYC.DE', 'DEQ.DE', 'HABA.DE', 'PAT.DE'],

    # ── Telekommunikation ──────────────────────────────────────────────────
    # DAX: DTE  |  MDAX: O2D, UTDI, FNTN  |  SDAX: 1U1
    'telecom':    ['DTE.DE', 'T', 'VZ', 'TMUS', 'ORAN',
                   'O2D.DE', 'UTDI.DE', 'FNTN.DE', '1U1.DE'],

    # ── Medien (neu) ───────────────────────────────────────────────────────
    # DAX: G24  |  MDAX: RRTL  |  SDAX: SPG, BVB, PSM, CWC
    'media':      ['RRTL.DE', 'SPG.DE', 'BVB.DE', 'PSM.DE', 'CWC.DE', 'G24.DE'],
}

# Alphabetically sorted sector labels (0-based)
SECTOR_LABELS = {s: i for i, s in enumerate(sorted(TICKERS_BY_SECTOR.keys()))}
# automotive=0, consumer=1, crypto=2, energy=3, finance=4, healthcare=5,
# heat_pump=6, industrial=7, materials=8, media=9, real_estate=10, tech=11, telecom=12

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
    'SAP.DE': 'SAP', 'IFX.DE': 'Infineon', 'ASML': 'ASML',
    # Finance
    'JPM': 'JPMorgan', 'GS': 'Goldman Sachs', 'AXP': 'AmEx', 'V': 'Visa',
    'ALV.DE': 'Allianz', 'MUV2.DE': 'Munich Re', 'DBK.DE': 'Deutsche Bank',
    'CBK.DE': 'Commerzbank', 'BNP.PA': 'BNP Paribas',
    # Healthcare
    'JNJ': 'J&J', 'UNH': 'UnitedHealth', 'MRK': 'Merck US', 'AMGN': 'Amgen',
    'BAYN.DE': 'Bayer', 'FRE.DE': 'Fresenius', 'MRK.DE': 'Merck KGaA',
    'SRT3.DE': 'Sartorius', 'NVS': 'Novartis',
    # Consumer
    'KO': 'Coca-Cola', 'MCD': "McDonald's", 'NKE': 'Nike', 'PG': 'P&G',
    'WMT': 'Walmart', 'HD': 'Home Depot', 'DIS': 'Disney',
    'ADS.DE': 'Adidas', 'BOSS.DE': 'Hugo Boss', 'HEN3.DE': 'Henkel',
    'PUM.DE': 'PUMA', 'ZAL.DE': 'Zalando', 'LVMH.PA': 'LVMH',
    # Industrial (DAX)
    'CAT': 'Caterpillar', 'BA': 'Boeing', 'HON': 'Honeywell', 'MMM': '3M',
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
    'NEM': 'Newmont', 'HEI.DE': 'Heidelberg Materials', 'SY1.DE': 'Symrise',
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
    'HABA.DE': 'Hamborner REIT', 'WUW.DE': 'Wüstenrot & W.',
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
NEWS_ANCHOR_ORG_FILTER = False
NEWS_ANCHOR_SCHEDULE_PATH = Path("data") / "sector_anchor_quarters.json"
NEWS_ANCHOR_TOP_N = 3

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


def _gkg_theme_triples() -> list[tuple[str, str, str]]:
    return list(GKG_THEME_SQL_TRIPLES or [])


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
    zw = int(zscore_w) if zscore_w is not None else 0
    for gk in _gkg_gcam_keys_clean():
        mid = f"{tag}_{gcam_series_colname(gk)}"
        out.extend(_news_macro_variant_cols(mid))
        out.extend(_news_sec_variant_cols(mid))
        if zw > 0:
            sw = f"_w{zw}"
            out.extend(
                [
                    f"news_macro_{mid}_tone_z{sw}",
                    f"news_macro_{mid}_vol_z{sw}",
                    f"news_sec_{mid}_tone_z{sw}",
                    f"news_sec_{mid}_vol_z{sw}",
                ]
            )
        if use_accel:
            out.extend([f"news_macro_{mid}_tone_accel", f"news_sec_{mid}_tone_accel"])


def _append_gcam_interaction_news_cols(out, tag, zscore_w, use_accel):
    zw = int(zscore_w) if zscore_w is not None else 0
    for slug, _a, _b in GCAM_INTERACTION_DIFFS:
        mid = f"{tag}_gcam_{slug}"
        out.extend(_news_macro_variant_cols(mid))
        out.extend(_news_sec_variant_cols(mid))
        if zw > 0:
            sw = f"_w{zw}"
            out.extend(
                [
                    f"news_macro_{mid}_tone_z{sw}",
                    f"news_macro_{mid}_vol_z{sw}",
                    f"news_sec_{mid}_tone_z{sw}",
                    f"news_sec_{mid}_vol_z{sw}",
                ]
            )
        if use_accel:
            out.extend([f"news_macro_{mid}_tone_accel", f"news_sec_{mid}_tone_accel"])


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


def _append_gkg_theme_news_cols(out, tag, zscore_w, use_accel):
    zw = int(zscore_w) if zscore_w is not None else 0
    for ch, alias, _ in _gkg_theme_triples():
        mid = f"{tag}_th_{alias}"
        if ch == "macro":
            out.extend(_news_macro_variant_cols(mid))
            if zw > 0:
                sw = f'_w{zw}'
                out.extend([
                    f'news_macro_{mid}_tone_z{sw}', f'news_macro_{mid}_vol_z{sw}',
                ])
            if use_accel:
                out.append(f'news_macro_{mid}_tone_accel')
        else:
            out.extend(_news_sec_variant_cols(mid))
            if zw > 0:
                sw = f'_w{zw}'
                out.extend([
                    f'news_sec_{mid}_tone_z{sw}', f'news_sec_{mid}_vol_z{sw}',
                ])
            if use_accel:
                out.append(f'news_sec_{mid}_tone_accel')


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
    _append_gcam_interaction_news_cols(out, tag, zscore_w, use_accel)
    if NEWS_ANCHOR_ORG_FILTER:
        _append_anchor_gcam_sec_cols(out, tag, zscore_w, use_accel)
        _append_gcam_div_sec_cols(out, tag, zscore_w, use_accel)
        _append_anchor_quality_sec_cols(out, tag, zscore_w, use_accel)
    _append_gkg_theme_news_cols(out, tag, zscore_w, use_accel)
    return out


def all_news_model_cols():
    cols = []
    trips = _gkg_theme_triples()
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
                    cols.extend(_news_macro_variant_cols(mid))
                    cols.extend(_news_sec_variant_cols(mid))
                    for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                        if z > 0:
                            sw = f'_w{int(z)}'
                            cols.extend(
                                [
                                    f'news_macro_{mid}_tone_z{sw}',
                                    f'news_macro_{mid}_vol_z{sw}',
                                    f'news_sec_{mid}_tone_z{sw}',
                                    f'news_sec_{mid}_vol_z{sw}',
                                ]
                            )
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        cols.extend(
                            [f'news_macro_{mid}_tone_accel', f'news_sec_{mid}_tone_accel']
                        )
                for slug, _ka, _kb in GCAM_INTERACTION_DIFFS:
                    mid_i = f"{tag}_gcam_{slug}"
                    cols.extend(_news_macro_variant_cols(mid_i))
                    cols.extend(_news_sec_variant_cols(mid_i))
                    for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                        if z > 0:
                            sw = f'_w{int(z)}'
                            cols.extend(
                                [
                                    f'news_macro_{mid_i}_tone_z{sw}',
                                    f'news_macro_{mid_i}_vol_z{sw}',
                                    f'news_sec_{mid_i}_tone_z{sw}',
                                    f'news_sec_{mid_i}_vol_z{sw}',
                                ]
                            )
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        cols.extend(
                            [
                                f'news_macro_{mid_i}_tone_accel',
                                f'news_sec_{mid_i}_tone_accel',
                            ]
                        )
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
                for ch, alias, _ in trips:
                    mid = f"{tag}_th_{alias}"
                    if ch == "macro":
                        cols.extend(_news_macro_variant_cols(mid))
                    else:
                        cols.extend(_news_sec_variant_cols(mid))
                    for z in NEWS_EXTRA_ZSCORE_WINDOWS:
                        if z > 0:
                            sw = f'_w{int(z)}'
                            if ch == "macro":
                                cols.extend([
                                    f'news_macro_{mid}_tone_z{sw}', f'news_macro_{mid}_vol_z{sw}',
                                ])
                            else:
                                cols.extend([
                                    f'news_sec_{mid}_tone_z{sw}', f'news_sec_{mid}_vol_z{sw}',
                                ])
                    if True in NEWS_EXTRA_TONE_ACCEL_OPTIONS:
                        if ch == "macro":
                            cols.append(f'news_macro_{mid}_tone_accel')
                        else:
                            cols.append(f'news_sec_{mid}_tone_accel')
    return cols

def build_technical_cols(rsi_w, bb_w, sma_w):
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
        'volume_zscore',
        f'sector_avg_rsi_{rsi_w}',
        'btc_momentum',
        'month',
        'sector_id',
    ]

def build_feature_cols(
    rsi_w, bb_w, sma_w,
    news_mom_w=None, news_vol_ma=None, news_tone_roll=None,
    news_extra_zscore_w=None, news_extra_tone_accel=None, news_extra_macro_sec_diff=None,
):
    out = build_technical_cols(rsi_w, bb_w, sma_w)
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
):
    m = {
        f'rsi_{rsi_w}':              f'RSI ({rsi_w}d)',
        f'bb_pband_{bb_w}':          f'Bollinger %B ({bb_w}d)',
        f'bb_x_rsi_{bb_w}_{rsi_w}': f'BB({bb_w}) × RSI({rsi_w})',
        f'sma_cross_20_{sma_w}':     f'SMA20 / SMA{sma_w}',
        f'market_breadth_{sma_w}':   f'Breadth (SMA{sma_w})',
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
        'month':           'Month',
        'sector_id':       'Sector ID',
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
print(f'Sector labels: {SECTOR_LABELS}')
