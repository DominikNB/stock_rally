"""Persistiert die volle Base-XGB-1-SHAP-Rangliste nach Phase 12 (JSON + CSV)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_JSON_PATHS = (
    Path("data") / "base_feature_shap_report.json",
    Path("models") / "base_feature_shap_report.json",
)
_DEFAULT_CSV_PATHS = (
    Path("data") / "base_feature_shap_report.csv",
    Path("models") / "base_feature_shap_report.csv",
)


def build_base_feature_shap_payload(
    *,
    shap_df: pd.DataFrame,
    feat_cols: list[str],
    feat_display: list[str],
    topk_names: list[str],
    topk_k: int,
    topk_mass_frac: float,
    shap_model: str = "XGB-1",
    shap_dataset: str = "df_test (META calendar)",
    shap_sample_rows: int,
    meta_shap_cum_frac: float | None,
    rsi_w: int,
    bb_w: int,
    sma_w: int,
    random_state: int,
) -> dict[str, Any]:
    """Payload für JSON, CSV und optional ``scoring_artifacts.joblib``."""
    records: list[dict[str, Any]] = []
    for i, row in shap_df.iterrows():
        records.append(
            {
                "rank": int(i + 1),
                "feature_display": str(row["feature"]),
                "feature_raw": str(row["feature_raw"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
            }
        )
    zeroish = int((shap_df["mean_abs_shap"] <= 1e-12).sum())
    return {
        "phase": "base_training_phase12",
        "built_on_utc": datetime.now(timezone.utc).isoformat(),
        "shap_model": shap_model,
        "shap_dataset": shap_dataset,
        "shap_sample_rows": int(shap_sample_rows),
        "feature_count": int(len(feat_cols)),
        "features_raw": [str(x) for x in feat_cols],
        "features_display": [str(x) for x in feat_display],
        "topk_k": int(topk_k),
        "topk_mass_frac": float(topk_mass_frac),
        "meta_shap_cum_frac": meta_shap_cum_frac,
        "topk_names_raw": [str(x) for x in topk_names],
        "rsi_w": int(rsi_w),
        "bb_w": int(bb_w),
        "sma_w": int(sma_w),
        "random_state": int(random_state),
        "shap_zeroish_count": zeroish,
        "shap_mean_abs_sorted": records,
    }


def save_base_feature_shap_report(
    payload: dict[str, Any],
    *,
    json_paths: tuple[Path, ...] | None = None,
    csv_paths: tuple[Path, ...] | None = None,
) -> list[Path]:
    """
    Schreibt volle SHAP-Rangliste nach ``data/`` und ``models/`` (JSON + CSV).

    Raises:
        RuntimeError: wenn kein Zielpfad geschrieben werden konnte.
    """
    json_paths = json_paths or _DEFAULT_JSON_PATHS
    csv_paths = csv_paths or _DEFAULT_CSV_PATHS
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    rows = payload.get("shap_mean_abs_sorted") or []
    csv_df = pd.DataFrame(rows)

    written: list[Path] = []
    errors: list[str] = []

    for jp in json_paths:
        try:
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(text, encoding="utf-8")
            written.append(jp.resolve())
        except OSError as e:
            errors.append(f"{jp}: {e}")

    for cp in csv_paths:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            csv_df.to_csv(cp, index=False, encoding="utf-8")
            written.append(cp.resolve())
        except OSError as e:
            errors.append(f"{cp}: {e}")

    if not written:
        raise RuntimeError(
            "Base-SHAP-Export fehlgeschlagen — keine Datei geschrieben: " + "; ".join(errors)
        )
    if errors:
        print(
            "Warnung Base-SHAP-Export: Teilpfade fehlgeschlagen: " + "; ".join(errors),
            flush=True,
        )
    n = int(payload.get("feature_count", len(rows)))
    print(
        f"Base-SHAP-Export: {n} Features, Top-K={payload.get('topk_k')} "
        f"({payload.get('topk_mass_frac', 0):.1%} SHAP-Masse) →",
        flush=True,
    )
    for p in written:
        print(f"  {p}", flush=True)
    return written


def load_base_feature_shap_report(
    *,
    prefer_models: bool = True,
) -> dict[str, Any] | None:
    """Lädt den zuletzt gespeicherten Base-SHAP-Report (``models/`` dann ``data/``)."""
    order = (
        (Path("models") / "base_feature_shap_report.json", Path("data") / "base_feature_shap_report.json")
        if prefer_models
        else (Path("data") / "base_feature_shap_report.json", Path("models") / "base_feature_shap_report.json")
    )
    for p in order:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None
