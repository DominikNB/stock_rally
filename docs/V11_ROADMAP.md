# V11 Roadmap — Antwort auf Pipeline-Kritik (Stand V10)

> **Alles in einer Datei:** [`STOCK_RALLY_DOKUMENTATION.md`](STOCK_RALLY_DOKUMENTATION.md) (Teil III) · Neu bauen: `python scripts/_gen_full_documentation_md.py`

Dieses Dokument ordnet externe/ interne Kritikpunkte am V10-Stack ein, korrigiert Fakten am **Ist-Code** und leitet konkrete V11-Arbeitspakete ab.

Verwandt: [`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md) (inkl. **§15 Verbesserungsvorschläge**), [`SYSTEM_REFERENZ.md`](SYSTEM_REFERENZ.md).

---

## Kurzfassung

| Bereich | Kritik (vereinfacht) | V10-Ist (Code) | V11-Ziel |
|---------|----------------------|----------------|----------|
| Signal / Target | Meta kennt Post-Filter nicht | **Teilweise falsch** — Filter sind in Meta-Optuna-CV drin; Schwelle auf THRESHOLD ist getrennt | Joint Optimierung Modell + Threshold + ein Kalender |
| News-Tag-Sync | `3_20_10` vs `5_10_3` = Silent Corruption | **Teilweise behoben:** `news_tag_sync.sync_topk_for_meta` (Phase 13 + Artefakt-Load) | Zusätzlich `NEWS_TAG` in Artefakt + harte Validierung |
| Phase-11 Pre-Prune | 590 Features vor Optuna | **Implementiert:** `feature_pre_pruning.py`, nur `df_train` | Feintuning `MI_TOP_K` / Split META↑ |
| Datenbreite | Nur GDELT + Ticker-Tech | **Erweitert:** FRED T10Y2Y/WALCL/EFFR + CNN Fear & Greed (`mr_fear_greed*`) | CoT optional |
| Feature-Pruning | 428/590 SHAP≈0 | **Implementiert:** `SHAP_ADF_*` in `config_settings.py`, `lib/shap_adf.py` | Nächster Base-Lauf optional `SHAP_ADF_REPLACE_FEAT_COLS=True` |

---

## 1. Signallogik & Target

### Was die Kritik trifft

1. **Zwei Kalender, zwei Optimierungsprobleme**
   - Meta-XGB wird auf **Klassifikation** (`target`) über den META-Kalender gefittet.
   - **`best_threshold`** wird danach auf dem **THRESHOLD**-Kalender per Grid (`_phase5_score_grid`) gewählt — nicht gemeinsam mit den letzten Boosting-Runden des Meta-Modells.
   - Phase 15 (`threshold_pr_filters.py`) ist vor allem **Report/Kalibrierungs-Visualisierung**, nicht die Quelle der produktiven Schwelle (die kommt aus Phase 13).

2. **Suboptimierung bleibt real**, aber anders formuliert:
   - Der Meta-Learner optimiert **Wahrscheinlichkeiten vs. Label**, während das Produktions-KPI **gefilterte Signale** (Threshold + Cooldown + …) ist.
   - Selbst wenn Filter in Optuna-CV laufen, ist das **Modellgewicht** nicht das gleiche Optimierungsziel wie „finale Signal-PR auf THRESHOLD“.

### Was die Kritik am Ist-Code **überzeichnet**

In `meta_learner.py` nutzt **`meta_objective`** pro Trial u. a.:

- `signal_skip_near_peak`, `peak_*`, `signal_max_rsi`, `signal_max_vol_stress_z`
- dynamische Threshold-Multiplikatoren (`mult_final_threshold_*`, `dyn_*_trigger`)
- `_apply_filters_cv` — dieselbe Filterkette wie im Scoring

→ Post-Filter sind **keine rein statischen Nachbesserungen**, die Meta beim Training „nicht kennt“. Sie sind **Hyperparameter der Meta-Optuna** und fließen in den CV-Loss ein.

**Konsequenz für V11:** Nicht „Filter erst einführen“, sondern **Trennung Meta-Fit vs. Phase-5-Threshold** und **BASE vs. META vs. THRESHOLD** enger koppeln.

### V11-Vorschlag: End-to-End (pragmatisch)

| Stufe | Maßnahme |
|-------|----------|
| A | **Ein Objective** auf dem Kalender, der auch produktiv zählt (z. B. THRESHOLD oder META+THRESHOLD mit Purge): `score = f(proba, threshold, filter_params, returns)` |
| B | Optuna optimiert **gemeinsam**: Meta-Hyperparameter, `meta_eval_threshold`-Seed, produktiver `best_threshold`, Filter-Grenzen (bereits teilweise vorhanden) |
| C | Meta-**Fit** erst **nach** Trial-Auswahl oder mit **nested** Threshold pro Fold (nicht nur Post-hoc Phase 5) |
| D | Optional: Filter als **explizite Meta-Features** (`rsi_above_kill`, `dist_to_high_pct`, …) statt nur Post-Processing — nur wenn A/B nicht reichen |

**Tests:** `tests/test_pipeline_invariants.py` um „Threshold + Filter auf THRESHOLD reproduzierbar aus Artefakt“ erweitern; bereits abgedeckt: `SCORING_ONLY` ignoriert `RETRAIN_META_ONLY` bei `assemble_features`.

**Siehe auch:** Priorisierte Backlog-Liste in [`PIPELINE_OVERVIEW.md` §15](PIPELINE_OVERVIEW.md#15-verbesserungsvorschläge-holdout--pipeline) (Top-Prioritäten #1 Cross-Section-Features, #2 ATR-Target; Stufen A/B/C).

### V11-Ergänzung: Label & Cross-Section

| Stufe | Maßnahme |
|-------|----------|
| **Quick** | `macro_event_within_2bd`, `ret_vs_spy_5d` in `FEAT_COLS` (Code in `signal_extra_filters.py` portieren) |
| **Mittel** | ATR-normalisiertes Rally-Target — erfordert neuen V11-Branch + volles Retraining |

---

## 2. Feature-Sync & News-Tag

### Problem (bestätigt)

- `topk_names` und `FEAT_COLS` werden in `scoring_artifacts.joblib` **eingefroren** (`scoring_persist.py`).
- Phase 12 mit neuem `news_feat_tag(mom, vol, tone)` erzeugt Spalten wie `news_*_5_10_3_*`.
- `RETRAIN_META_ONLY=True` lädt altes Artefakt mit `news_*_3_20_10_*` → Meta sucht Spalten, die im aktuellen `df_test` fehlen oder semantisch andere Signale tragen.

Das ist **kein Kosmetik-Thema**, sondern Risiko für **stille Fehl-Spalten** (0/NaN-Fill), wenn keine harte Validierung greift.

### V10.1 umgesetzt (2026-05)

- `lib/stock_rally_v10/news_tag_sync.py` — Top-K-Spaltennamen werden auf `best_params`-News-Tag gemappt.
- `lib/stock_rally_v10/training_phases/feature_pre_pruning.py` — Phase 11 (Sparsity + MI, nur BASE).
- `pipeline_runner.run_statistical_pre_pruning()` nach Split.
- `BASE_MODELS_META_EXCLUDE = ("XGB-1", "LR")` — XGB-1 bleibt für SHAP/ADF, nicht im Meta-Stack.
- `TIME_SPLIT_FRAC_BASE/META` → 0.40 / 0.40 (mehr META-Zeilen bei gleichem Purge).

### V11-Vorschlag (Rest)

```python
# config_settings.py (Beispiel)
NEWS_MOM_W = 5
NEWS_VOL_MA_W = 10
NEWS_TONE_ROLL_W = 3
# abgeleitet, nie handisch in Feature-Listen:
NEWS_TAG = f"{NEWS_MOM_W}_{NEWS_VOL_MA_W}_{NEWS_TONE_ROLL_W}"
```

| Maßnahme | Ort |
|----------|-----|
| Zentrale `NEWS_TAG` / Tripel | `config_settings.py` |
| `build_news_model_cols(tag=NEWS_TAG)` überall | `config.py`, `features.py`, Optuna |
| `topk_names` nur aus **aktueller** `FEAT_COLS` + SHAP | Phase 12 Ende |
| **Pipeline-Gate** vor Phase 13 | `assert all(c in df.columns for c in topk_names)` + `news_tag` in Artefakt == `cfg.NEWS_TAG` |
| Artefakt-Versionierung | `artifact_schema_version`, `news_tag`, `feat_cols_hash` in `joblib` |

---

## 3. Externe Daten (kostenfrei)

### FRED — bereits in V10 (`lib/stock_rally_v10/macro_vol_enrich.py`)

- Loader: `fredapi` + `FRED_API_KEY` (`.env`), Fallback: FRED-CSV-URL
- **Bereits über FRED:** `VVIXCLS`, `VXNCLS`, `RVXCLS`, `VXVCLS`, `BAMLH0A0HYM2` (HY-OAS → `mr_hy_spread*`), `DTWEXBGS` (DXY-Fallback → `mr_dxy_*`)
- **Neu (V10.1):** `T10Y2Y`, `WALCL`, `DFF`/`EFFR` → `mr_t10y2y*`, `mr_walcl*`, `mr_effr*`
- **Fear & Greed:** CNN JSON (`FEAR_GREED_*`, Cache `data/fear_greed_cache.json`) → `mr_fear_greed*`
- **`regime_tnx_ret_5d`:** Yahoo `^TNX` (`lib/signal_extra_filters.py`), nicht FRED `DGS10`/`T10Y2Y`

Priorität nach Meta-SHAP-Lücke (Makro/Regime dominant):

| Quelle | API | Status V10 | V11-Ziel |
|--------|-----|------------|----------|
| **FRED (erweitern)** | API-Key | Teilweise | `T10Y2Y`, `WALCL`, `EFFR` → neue `mr_*` / `regime_*` |
| **CFTC CoT** | öffentlich / Nasdaq Data Link | Netto-Positionen SPX/NQ „non-commercial“ | mittel-hoch (wöchentlich, Forward-fill) |
| Fear & Greed | freie JSON-APIs | Index + Δ5d → Website-Ampel + Meta | niedrig |
| Alpha Vantage / EOD | Free Tier | Fundamentals — nur wenn Label es braucht | optional |

**Integration:** wie bestehende `mr_*`: merge auf `(Date)` (global) oder `(Date, sector)` — **kein** Ticker-Leakage in FINAL.

---

## 4. Feature-Pruning (ADF)

### Ist

- ~428/590 Features mit SHAP ≈ 0 im Base-Report (META-Stichprobe, XGB-1).
- `feature_prescreen` + `news_correlation_prescreen` reduzieren **vor** Optuna, aber **nicht** post-SHAP.
- Meta nutzt ohnehin nur **35** Roh-Spalten + 10 Probas.

### V11: Automated Drop Features (nach Phase 12)

```
FEAT_COLS_PRUNED = [c for c in FEAT_COLS if shap_mass[c] >= SHAP_PRUNE_MIN]
```

| Parameter | Vorschlag |
|-----------|-----------|
| `SHAP_PRUNE_MIN` | z. B. 1e-4 oder unteres Quantil der positiven SHAP-Masse |
| `SHAP_PRUNE_KEEP_TOP_N` | optional Deckel 120–150 |
| Base-Training | nur `FEAT_COLS_PRUNED` |
| Meta `topk_names` | aus **geprünter** Liste neu berechnen |

**Effekt:** schnelleres Base-Optuna, weniger RAM, weniger Scheinkorrelationen bei ~2k META-Zeilen.

---

## 5. Base-Stack verschlanken

Meta-SHAP (Phase 13) zeigt:

| Modell | mean \|SHAP\| (Meta) |
|--------|---------------------|
| LGB-3_prob | 0.173 |
| RF_prob | 0.127 |
| XGB-2/4_prob | 0.112 / 0.083 |
| XGB-1_prob | 0.014 |
| LR_prob | 0.000 |

**V11-Option (konfigurierbar):**

```python
BASE_MODELS_ACTIVE = ("XGB-2", "XGB-3", "XGB-4", "LGB-2", "LGB-3", "RF", "ET")  # ohne LR, ohne XGB-1
```

Oder: Stack nur Modelle mit Meta-SHAP > ε nach erstem Meta-Lauf.

---

## 6. Umsetzungsreihenfolge (empfohlen)

1. **News-Tag-Gate + `NEWS_TAG`** — geringer Aufwand, hoher Sicherheitsgewinn (blockiert Silent Bugs sofort).
2. **SHAP-ADF-Pruning** — Performance + Generalisierung.
3. **Phase-5 / Threshold in Meta-Objective verzahnen** — größter KPI-Hebel.
4. **FRED erweitern** (T10Y2Y, WALCL, EFFR) — Basis-Loader existiert; nur neue Serien + Merge.
5. **CoT + Fear/Greed** — optional danach.
6. **Base-Modell-Subset** — nach 1–3 messen, ob Laufzeit ↓ ohne Score-Verlust.

---

## 7. Was wir in V10 **nicht** ändern (ohne expliziten V11-Branch)

- Zieldefinition `rally_plus_entry` bleibt, bis Label-Studien es erzwingen.
- 99%-Precision-Gate in Meta-CV bleibt Produktions-Constraint (Kritik war nicht „Gate abschaffen“, sondern Konsistenz der Pipeline).

---

*Bei neuen Phase-12/13-Läufen: `PIPELINE_OVERVIEW.md` via `python scripts/_gen_pipeline_overview_md.py` aktualisieren.*
