"""SHAP-Top-Listen aus data/*.json + rename_map."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.config import build_rename_map

_sp = cfg.SEED_PARAMS
rm = build_rename_map(int(_sp["rsi_window"]), int(_sp["bb_window"]), int(_sp["sma_window"]))


def _disp(raw: str) -> str:
    return str(rm.get(raw, raw))


def main() -> None:
    from lib.base_feature_shap_export import load_base_feature_shap_report

    meta_path = ROOT / "data" / "meta_feature_shap_report.json"
    pres_path = ROOT / "data" / "feature_prescreen_v1.json"

    base_payload = load_base_feature_shap_report()
    if base_payload:
        rows = base_payload.get("shap_mean_abs_sorted") or []
        print("=== BASE (Phase 12, XGB-1, volle FEAT_COLS, mean |SHAP|) ===")
        print(
            f"Features: {base_payload.get('feature_count')}  "
            f"Top-K: {base_payload.get('topk_k')}  "
            f"Stand: {base_payload.get('built_on_utc', '')}\n"
        )
        for r in rows[:30]:
            print(
                f"{r['rank']:3d}. {_disp(str(r['feature_raw'])):44s}  "
                f"{float(r['mean_abs_shap']):.4f}"
            )
        if len(rows) > 30:
            print(f"  … +{len(rows) - 30} weitere (siehe models/base_feature_shap_report.csv)")
        n0 = int(base_payload.get("shap_zeroish_count", 0))
        if n0:
            print(f"\n  |SHAP|≈0: {n0} Features")
    else:
        print("Base-SHAP-Report fehlt (models/ oder data/base_feature_shap_report.json)")
    print()

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("=== META (Phase 13, XGB Meta auf TEST, mean |SHAP|) ===")
        print(f"Features: {meta.get('feature_count')}\n")
        for r in meta["shap_mean_abs_sorted"][:25]:
            print(f"{r['rank']:2d}. {r['feature']:44s}  {r['mean_abs_shap']:.4f}")
        probs = [r for r in meta["shap_mean_abs_sorted"] if r["feature"].endswith("_prob")]
        raw = [r for r in meta["shap_mean_abs_sorted"] if not r["feature"].endswith("_prob")]
        print(f"\n  Davon Base-Modell-Probas: {len(probs)} | Roh-Features (Top-K): {len(raw)}")
    else:
        print("meta_feature_shap_report.json fehlt")

    print()
    if pres_path.is_file():
        pres = json.loads(pres_path.read_text(encoding="utf-8"))
        print(
            f"=== BASE (Prescreen-XGB, mean |SHAP|, Stand {pres.get('built_on')}) ==="
        )
        print(
            f"Stichprobe: {pres.get('input_signature', {}).get('n_rows')} Zeilen, "
            f"{pres.get('input_signature', {}).get('n_features')} Features\n"
        )
        for i, row in enumerate(pres.get("importance_top_50", [])[:25], 1):
            raw = str(row["feature"])
            print(f"{i:2d}. {_disp(raw):44s}  {row['mean_abs_shap']:.4f}")
    else:
        print("feature_prescreen_v1.json fehlt")

    print()
    print("=== BASE Top-K für Meta-Stack (aus Artefakt topk_names, Reihenfolge SHAP-Auswahl) ===")
    try:
        import joblib

        art = joblib.load(ROOT / "models" / "scoring_artifacts.joblib")
        for i, raw in enumerate(art.get("topk_names") or [], 1):
            print(f"{i:2d}. {_disp(str(raw))}")
    except Exception as e:
        print(f"  (Artefakt nicht ladbar: {e})")


if __name__ == "__main__":
    main()
