"""
Base-SHAP-Rangliste als CSV exportieren (aus JSON oder scoring_artifacts.joblib).

  python scripts/export_base_shap_from_artifact.py
  python scripts/export_base_shap_from_artifact.py --out reports/base_shap.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_payload() -> dict:
    from lib.base_feature_shap_export import load_base_feature_shap_report

    payload = load_base_feature_shap_report()
    if payload is not None:
        return payload
    art_path = ROOT / "models" / "scoring_artifacts.joblib"
    if art_path.is_file():
        import joblib

        art = joblib.load(art_path)
        payload = art.get("base_feature_shap_report")
        if isinstance(payload, dict) and payload.get("shap_mean_abs_sorted"):
            return payload
    raise FileNotFoundError(
        "Kein Base-SHAP-Report — zuerst Phase 12 (volles Base-Training) ausführen."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Base-SHAP-CSV aus gespeichertem Report")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "base_feature_shap_report.csv",
        help="Ziel-CSV (Default: models/base_feature_shap_report.csv)",
    )
    args = ap.parse_args()
    payload = _load_payload()
    rows = payload.get("shap_mean_abs_sorted") or []
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    n = len(df)
    n0 = int((pd.to_numeric(df.get("mean_abs_shap"), errors="coerce") <= 1e-12).sum()) if n else 0
    print(f"Geschrieben: {args.out.resolve()}  ({n} Zeilen, davon {n0} mit |SHAP|≈0)")
    built = payload.get("built_on_utc", "")
    if built:
        print(f"  Stand: {built}")


if __name__ == "__main__":
    main()
