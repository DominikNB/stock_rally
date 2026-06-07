"""Build docs/PIPELINE_OVERVIEW.md = static body + dynamic SHAP appendix."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "docs" / "_pipeline_overview_static.md"
BASE_REPORT = ROOT / "models" / "base_feature_shap_report.json"
META_REPORT = ROOT / "data" / "meta_feature_shap_report.json"
OUT = ROOT / "docs" / "PIPELINE_OVERVIEW.md"
TABLE_FRAG = ROOT / "data" / "_shap_top100_table.md"
META_TABLE_FRAG = ROOT / "data" / "_shap_meta_table.md"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)


def _ensure_table_frag(path: Path, script: str) -> str:
    if not path.is_file():
        subprocess.run([str(PY), str(ROOT / "scripts" / script), str(path)], check=True, cwd=ROOT)
    return path.read_text(encoding="utf-8").strip()


def _build_header(p: dict, meta_extra: str) -> str:
    return (
        f"**Stand Base-SHAP:** `{p.get('built_on_utc', '?')}` · "
        f"**Base-Features (Report):** {p.get('feature_count', '?')} · "
        f"**Modell:** {p.get('shap_model', '?')} auf {p.get('shap_sample_rows', '?')} Zeilen "
        f"({p.get('shap_dataset', '?')}){meta_extra}\n\n"
    )


def _build_base_shap_section(p: dict, table: str, topk_lines: str) -> str:
    adf = p.get("shap_adf") or {}
    adf_block = ""
    if adf:
        adf_block = f"""
### SHAP-ADF (letzter Lauf)

| Kennzahl | Wert |
|----------|------|
| Aktiv | `{adf.get('enabled', '?')}` |
| Features vor ADF | {adf.get('n_before', '?')} |
| Features nach ADF | {adf.get('n_after', '?')} |
| Gedroppt | {adf.get('n_dropped', '?')} |
| Min. mean \\|SHAP\\| | {adf.get('min_abs_shap', '?')} |
| Min. Keep | {adf.get('min_keep', '?')} |

"""
    return f"""### 15.1 Base-Optuna-Gewinner (aus SHAP-Report)

| Parameter | Wert |
|-----------|------|
| `rsi_w` | {p.get('rsi_w', '?')} |
| `bb_w` | {p.get('bb_w', '?')} |
| `sma_w` | {p.get('sma_w', '?')} |
| News-Tag (Spaltenpräfix) | `{p.get('news_tag', '5_10_3')}` |
| `topk_k` | {p.get('topk_k', '?')} |
| `META_SHAP_CUM_FRAC` | {p.get('meta_shap_cum_frac', 0.85)} |
| Kum. SHAP-Masse Top-K | {100 * float(p.get('topk_mass_frac') or 0):.1f} % |
| Features mit \\|SHAP\\| ≈ 0 | {p.get('shap_zeroish_count', '?')} |

{adf_block}
### 15.2 `topk_names_raw` (Meta-Roh-Features, Reihenfolge)

{topk_lines}

### 15.3 Top 100 Base-SHAP

**Quelle:** `models/base_feature_shap_report.json` · **Metrik:** mean |SHAP| · **Modell:** XGB-1 nach Phase 12 (finale SHAP auf geprüfter Liste).

| Rank | mean \\|SHAP\\| | Anzeigename | Raw-Spalte |
|------|-------------|-------------|------------|
{table}

**Lesart:** In den Top 20 dominieren Volatilität/ADR/RSI-Delta, Makro-Vol-Regime (`mr_*`, `regime_*`) und News×Makro-Kreuzterme.
"""


def _build_meta_shap_section(mp: dict, meta_table: str) -> str:
    mrows = mp.get("shap_mean_abs_sorted") or []
    prob_rows = [r for r in mrows if str(r.get("feature", "")).endswith("_prob")]
    raw_rows = [r for r in mrows if not str(r.get("feature", "")).endswith("_prob")]
    prob_mass = sum(float(r["mean_abs_shap"]) for r in prob_rows)
    raw_mass = sum(float(r["mean_abs_shap"]) for r in raw_rows)
    total_mass = prob_mass + raw_mass or 1.0
    zeroish = sum(1 for r in mrows if float(r.get("mean_abs_shap", 0)) <= 1e-12)
    news_hint = ""
    for r in raw_rows:
        feat = str(r.get("feature", ""))
        if "news_macro_" in feat:
            rest = feat.split("news_macro_", 1)[1]
            tag = "_".join(rest.split("_")[:3])
            news_hint = (
                f"Meta-Roh-News-Präfix: **`{tag}`** — muss mit Phase-12-News-Tag in `best_params` übereinstimmen."
            )
            break

    return f"""
### 15.4 Meta-Learner SHAP (Phase 13)

**Quelle:** `data/meta_feature_shap_report.json` · **Metrik:** mean |SHAP| auf **gesamtem META** (`df_test`) ·
**Modell:** Meta-XGB nach Optuna + finalem Fit.

| Kennzahl | Wert |
|----------|------|
| Meta-Inputs gesamt | {mp.get('feature_count', len(mrows))} |
| Davon Base-`prob` | {len(prob_rows)} |
| Davon Roh-Features (Top-K) | {len(raw_rows)} |
| Summe mean \\|SHAP\\| Base-`prob` | {prob_mass:.4f} ({100 * prob_mass / total_mass:.1f} % der Gesamtmasse) |
| Summe mean \\|SHAP\\| Roh-Features | {raw_mass:.4f} ({100 * raw_mass / total_mass:.1f} % der Gesamtmasse) |
| Features mit \\|SHAP\\| ≈ 0 | {zeroish} |

#### Lesart

- Der Meta-Classifier **gewichtet vor allem die Base-Ensemble-Probas** — typisch **LGB-3**, **RF**, **XGB-2/4**.
- Roh-Features: BTC-Momentum-Z, News-Makro-Tone, RVX/DXY, Zinsregime.
- **XGB-1_prob** / **LR_prob** oft nahe 0 SHAP — Meta bevorzugt andere Basen.
- {news_hint or 'News-Tag aus Spaltennamen mit Artefakt `best_params` abgleichen.'}

#### Vollständige Rangliste (alle {len(mrows)} Meta-Features)

| Rank | mean \\|SHAP\\| | Typ | Feature |
|------|-------------|-----|---------|
{meta_table}
"""


def main() -> None:
    if not STATIC.is_file():
        raise SystemExit(f"Missing static template: {STATIC}")
    if not BASE_REPORT.is_file():
        raise SystemExit(f"Missing base SHAP report: {BASE_REPORT}")

    p = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    topk = p.get("topk_names_raw") or []
    topk_lines = "\n".join(f"{i}. `{name}`" for i, name in enumerate(topk, 1))
    table = _ensure_table_frag(TABLE_FRAG, "_gen_shap_table_md.py")

    meta_extra = ""
    meta_section = """
### 15.4 Meta-Learner SHAP

*Noch nicht vorhanden — entsteht nach Phase 13 als `data/meta_feature_shap_report.json`.*
"""
    if META_REPORT.is_file():
        mp = json.loads(META_REPORT.read_text(encoding="utf-8"))
        mrows = mp.get("shap_mean_abs_sorted") or []
        meta_extra = (
            f" · **Meta-SHAP:** {mp.get('feature_count', len(mrows))} Inputs"
        )
        meta_table = _ensure_table_frag(META_TABLE_FRAG, "_gen_shap_meta_table_md.py")
        meta_section = _build_meta_shap_section(mp, meta_table)

    static = STATIC.read_text(encoding="utf-8")
    marker = "<!-- SHAP_APPENDIX_START -->"
    if marker not in static:
        raise SystemExit(f"Marker {marker!r} not found in {STATIC}")

    before, _ = static.split(marker, 1)
    appendix = _build_base_shap_section(p, table, topk_lines) + meta_section

    footer = """
---

## 16. Dateien & Befehle

| Pfad | Inhalt |
|------|--------|
| `models/base_feature_shap_report.json` | Volle Base-SHAP (JSON) |
| `models/base_feature_shap_report.csv` | Volle Base-SHAP (CSV) |
| `data/base_feature_shap_report.json` | Spiegel nach `data/` |
| `data/meta_feature_shap_report.json` | Meta-SHAP nach Phase 13 |
| `models/scoring_artifacts.joblib` | Produktions-Artefakt |
| `models/meta_optuna_poststudy_checkpoint.json` | Meta-Optuna-Resume |
| `docs/_pipeline_overview_static.md` | Statischer Hauptteil (von Hand pflegen) |

```bash
# Pipeline (Projektroot)
python -m lib.stock_rally_v10.pipeline_runner

# Gesamtdokument neu bauen (statisch + SHAP-Anhang)
python scripts/_gen_pipeline_overview_md.py

# Nur SHAP-Tabellen-Fragmente
python scripts/_gen_shap_table_md.py data/_shap_top100_table.md
python scripts/_gen_shap_meta_table_md.py data/_shap_meta_table.md
```

---

*Abschnitt 15 (SHAP) wird aus Reports generiert. Abschnitte 1–14: `docs/_pipeline_overview_static.md`.*
"""

    doc = before + marker + "\n\n" + _build_header(p, meta_extra) + appendix + footer
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
