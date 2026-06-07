"""
Nur für Legacy-Artefakte: topk aus SHAP-Report aktualisieren, ohne Base-Neutraining.

Voller ADF-Lauf (prune + alle Base-Modelle neu): Phase 12 mit SHAP_ADF_ENABLED=True.

  python scripts/apply_shap_adf_to_artifact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd

from lib.base_feature_shap_export import load_base_feature_shap_report
from lib.shap_adf import apply_shap_adf, enrich_shap_payload_with_adf
from lib.stock_rally_v10 import config as cfg


def main() -> None:
    path = Path(cfg.SCORING_ARTIFACT_PATH)
    if not path.is_file():
        raise SystemExit(f"Artefakt fehlt: {path}")
    report = load_base_feature_shap_report()
    if not report:
        raise SystemExit("base_feature_shap_report.json fehlt (models/ oder data/).")
    bundle = joblib.load(path)
    feat_cols = list(bundle["FEAT_COLS"])
    rows = report.get("shap_mean_abs_sorted") or []
    shap_df = pd.DataFrame(rows).rename(columns={"feature_display": "feature"})
    if "feature_raw" not in shap_df.columns:
        raise SystemExit("SHAP-Report ohne feature_raw.")

    adf = apply_shap_adf(feat_cols=feat_cols, shap_df=shap_df, cfg_mod=cfg)
    bundle["topk_names"] = list(adf.topk_names)
    bundle["topk_idx"] = np.asarray(adf.topk_idx)
    bundle["FEAT_COLS_PRUNED"] = list(adf.feat_cols_pruned)
    if bundle.get("base_feature_shap_report"):
        bundle["base_feature_shap_report"] = enrich_shap_payload_with_adf(
            dict(bundle["base_feature_shap_report"]), adf, cfg
        )
    joblib.dump(bundle, path)
    print(f"Aktualisiert: {path.resolve()}")


if __name__ == "__main__":
    main()
