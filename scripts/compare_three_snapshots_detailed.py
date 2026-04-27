"""Detailed three-way snapshot comparison (master_complete + optional joblib meta)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SNAPS = {
    "A_baseline_vor_grossen_aenderungen": ROOT
    / "data/model_snapshots/PROTECTED_pre_meta_optimization_20260424_185617",
    "B_zwischenstand_vor_chase_kalibrierung": ROOT
    / "data/model_snapshots/pre_meta_optimization_20260424_213141",
    "C_neuester_lauf": ROOT / "data/model_snapshots/post_meta_optimization_20260424_235955",
}


def _prob_col(df: pd.DataFrame) -> str | None:
    for c in ("meta_prob", "prob", "meta_score"):
        if c in df.columns:
            return c
    return None


def _numeric_summary(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "win_rate_gt0": float((s > 0).mean()),
    }


def _load_joblib_meta(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return {"error": "missing"}
    try:
        import joblib

        art = joblib.load(path)
    except Exception as e:
        return {"error": str(e)}
    if not isinstance(art, dict):
        return {"type": type(art).__name__}
    for k in (
        "best_threshold",
        "meta_calibration_mode",
        "meta_prob_calibrator",
        "meta_optuna_best_value",
        "meta_optuna_best_params",
        "meta_best_params",
        "meta_utility_params",
    ):
        if k not in art:
            continue
        v = art[k]
        if k == "meta_prob_calibrator" and v is not None:
            out[k] = type(v).__name__
        elif k == "meta_optuna_best_params" and isinstance(v, dict):
            out[k] = v
        elif hasattr(v, "get_params"):
            try:
                out[k] = v.get_params()
            except Exception:
                out[k] = str(v)
        else:
            out[k] = v
    return out


def _analyze_snapshot(label: str, snap_dir: Path) -> dict:
    mc = snap_dir / "master_complete.csv"
    art = snap_dir / "scoring_artifacts.joblib"
    sig = snap_dir / "signals.json"
    out: dict = {"label": label, "dir": str(snap_dir.relative_to(ROOT))}
    if not mc.exists():
        out["master_complete"] = {"error": "missing"}
        return out
    df = pd.read_csv(mc)
    d = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else None
    out["master_complete"] = {
        "rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "date_min": str(d.min()) if d is not None else None,
        "date_max": str(d.max()) if d is not None else None,
        "unique_days": int(d.dt.normalize().nunique()) if d is not None else None,
    }
    pc = _prob_col(df)
    out["master_complete"]["prob_column"] = pc
    if pc:
        out["prob"] = _numeric_summary(df[pc])
    ret_cols = [c for c in df.columns if c.startswith("ret_") and c[4:].replace("d", "").isdigit()]
    ret_cols = sorted(set(ret_cols), key=lambda x: (len(x), x))
    out["forward_returns"] = {c: _numeric_summary(df[c]) for c in ret_cols if c in df.columns}
    extras = [
        "meta_prob_margin",
        "runup_3d",
        "runup_5d",
        "regime_vix_level",
        "regime_vix_z_20d",
    ]
    for c in extras:
        if c in df.columns:
            out[c] = _numeric_summary(df[c])
    out["artifacts_meta"] = _load_joblib_meta(art)
    if sig.exists():
        try:
            payload = json.loads(sig.read_text(encoding="utf-8"))
            n = len(payload) if isinstance(payload, list) else None
            if n is None and isinstance(payload, dict):
                for k in ("signals", "rows", "data"):
                    if isinstance(payload.get(k), list):
                        n = len(payload[k])
                        break
            out["signals_json_count"] = n
        except Exception as e:
            out["signals_json_count"] = f"error:{e}"
    return out


def _pair_keys(df: pd.DataFrame) -> pd.Series:
    return df["ticker"].astype(str) + "|" + pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")


def _set_overlap(a: Path, b: Path, c: Path) -> dict:
    def load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p)

    da, db, dc = load(a), load(b), load(c)
    for d in (da, db, dc):
        d["_k"] = _pair_keys(d)
    sa, sb, sc = set(da["_k"]), set(db["_k"]), set(dc["_k"])
    return {
        "A_only": len(sa - sb - sc),
        "B_only": len(sb - sa - sc),
        "C_only": len(sc - sa - sb),
        "AB_not_C": len((sa & sb) - sc),
        "AC_not_B": len((sa & sc) - sb),
        "BC_not_A": len((sb & sc) - sa),
        "ABC": len(sa & sb & sc),
    }


def main() -> None:
    paths = {k: v for k, v in SNAPS.items()}
    for k, p in paths.items():
        if not p.is_dir():
            print(f"Missing dir {k}: {p}", file=sys.stderr)
            sys.exit(1)
    reports = {k: _analyze_snapshot(k, p) for k, p in paths.items()}
    mc_paths = {k: paths[k] / "master_complete.csv" for k in paths}
    overlap = _set_overlap(mc_paths["A_baseline_vor_grossen_aenderungen"], mc_paths["B_zwischenstand_vor_chase_kalibrierung"], mc_paths["C_neuester_lauf"])
    full = {"snapshots": reports, "signal_key_overlap_ticker_date": overlap}
    out_path = ROOT / "data" / "three_way_snapshot_comparison_20260424.json"
    out_path.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(full, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
