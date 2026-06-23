"""Einmal-Summary Base/Meta aus models/* — nicht Teil der Pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    meta = json.loads((ROOT / "models/meta_optuna_poststudy_checkpoint.json").read_text(encoding="utf-8"))
    shap = json.loads((ROOT / "models/base_feature_shap_report.json").read_text(encoding="utf-8"))
    base = joblib.load(ROOT / "models/base_optuna_checkpoint.joblib")
    art = joblib.load(ROOT / "models/scoring_artifacts.joblib")

    print("=== BASE (Phase 12 / Optuna + SHAP) ===")
    print(f"built_on_utc: {shap.get('built_on_utc')}")
    print(f"best trial value: {base.get('best_value')}")
    bbp = base.get("best_params") or {}
    print("best_params:")
    for k in sorted(bbp):
        print(f"  {k}: {bbp[k]}")
    print(f"windows: rsi={shap.get('rsi_w')} bb={shap.get('bb_w')} sma={shap.get('sma_w')}")
    print(f"topk: k={shap.get('topk_k')} mass_frac={shap.get('topk_mass_frac'):.3f}")
    print(f"features: {shap.get('feature_count')} (before ADF: {shap.get('feature_count_before_adf')})")
    cs_topk = [x for x in shap.get("topk_names_raw", []) if "ret_vs_spy" in x or "macro_event" in x]
    print(f"cross-section in topk: {cs_topk}")
    print("top SHAP (15):")
    for row in shap.get("shap_mean_abs_sorted", [])[:15]:
        print(f"  {row['rank']:2d} {row['feature_raw']:42s} {row['mean_abs_shap']:.5f}")

    print("\n=== META (Phase 13 / Optuna) ===")
    print(f"best_value (tp_precision objective): {meta.get('best_value')}")
    mbp = meta.get("best_params") or {}
    for k in sorted(mbp):
        print(f"  {k}: {mbp[k]}")

    print("\n=== SCORING ARTIFACT (deploy) ===")
    print(f"best_threshold: {art.get('best_threshold')}")
    print(f"meta_optuna_best_value: {art.get('meta_optuna_best_value')}")
    fc = art.get("FEAT_COLS") or []
    print(f"FEAT_COLS: {len(fc)}")
    print(f"cross-section in FEAT_COLS: {[c for c in fc if 'ret_vs_spy' in c or 'macro_event' in c]}")

    snap = ROOT / "data/model_snapshots/pre_meta_optimization_20260614_210323/scoring_artifacts.joblib"
    if snap.is_file():
        old = joblib.load(snap)
        print("\n=== DELTA vs Snapshot (vor ATR+Cross-Section) ===")
        print(f"threshold: {old.get('best_threshold')} -> {art.get('best_threshold')}")
        print(f"meta objective: {old.get('meta_optuna_best_value')} -> {art.get('meta_optuna_best_value')}")
        print(f"FEAT_COLS count: {len(old.get('FEAT_COLS', []))} -> {len(fc)}")

    sj = json.loads((ROOT / "docs/signals.json").read_text(encoding="utf-8"))
    sigs = sj.get("signals", sj if isinstance(sj, list) else [])
    print(f"\nOOS signals (signals.json): {len(sigs)} @ threshold {sj.get('threshold')} generated {sj.get('generated')}")


if __name__ == "__main__":
    main()
