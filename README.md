# stock_rally

## Verzeichnis

| Pfad | Inhalt |
|------|--------|
| `lib/` | Gemeinsame Module: `scoring_persist`, `signal_extra_filters`, `website_rally_prompt` |
| `holdout/` | Pipeline-Skripte für Holdout-CSVs (`build_holdout_signals_master`, Analysen, KDE-Plot, …) |
| `scripts/` | Website-Gemini-Lauf, `website_analysis_common`, Hilfsskripte |
| `config/` | `load_env`, `env.example` |
| `data/`, `docs/`, `figures/`, `models/` | Daten, GitHub Pages, Abbildungen, Artefakte — technische Gesamtbeschreibung: `docs/SYSTEM_REFERENZ.md` |
| `stock_rally_v10.ipynb` | Haupt-Notebook (Arbeitsverzeichnis = Projektroot) |

## Training → Artefakt

Nach **Phase 5** (Meta-Learner + Threshold) **speichert Cell 14** automatisch `models/scoring_artifacts.joblib` (über `save_scoring_artifacts()` aus Cell 2). Cell 18 ist nur noch bei Bedarf nötig (z. B. manuell erneut speichern im gleichen Kernel).

## Imports

Python-Pfade gehen vom **Projektroot** aus (wie im Notebook: `sys.path` + `import lib…` / `import holdout…`).

Holdout-CLI von dort z. B.:

```text
python -m holdout.build_holdout_signals_master
python -m holdout.analyze_holdout_forward_returns
```

Website-Analyse:

```text
python scripts/run_website_analysis_gemini.py
```
