from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _latest_snapshot(prefix: str) -> Path | None:
    base = Path("data/model_snapshots")
    if not base.exists():
        return None
    cands = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    return cands[-1] if cands else None


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_meta_params(artifact: dict) -> dict:
    p = artifact.get("meta_optuna_best_params", {}) if isinstance(artifact, dict) else {}
    return p if isinstance(p, dict) else {}


def _risk_off_mask(df: pd.DataFrame, m_cut: float, v_cut: float) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    idx = df.index
    if "market_ret_1d" in df.columns:
        m = pd.to_numeric(df["market_ret_1d"], errors="coerce")
    else:
        m = pd.Series(np.nan, index=idx)
    if "regime_vix_z_20d" in df.columns:
        v = pd.to_numeric(df["regime_vix_z_20d"], errors="coerce")
    else:
        v = pd.Series(np.nan, index=idx)
    return ((m <= m_cut) | (v >= v_cut)).fillna(False)


def _load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    payload = _read_json(path)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for k in ("signals", "rows", "data"):
            if isinstance(payload.get(k), list):
                return pd.DataFrame(payload[k])
    return pd.DataFrame()


def main() -> None:
    out = []
    art_path = Path("models/scoring_artifacts.joblib")
    if not art_path.exists():
        raise SystemExit(f"Missing artifact: {art_path}")
    artifact = joblib.load(art_path)
    params = _extract_meta_params(artifact)

    m_cut = float(params.get("regime_riskoff_market_ret1d_cut", -0.003))
    v_cut = float(params.get("regime_riskoff_vix_z_cut", 1.0))
    off_mult = float(params.get("regime_riskoff_threshold_mult", 1.15))
    on_mult = float(params.get("regime_riskon_threshold_mult", 0.98))
    thr = float(params.get("meta_eval_threshold", artifact.get("best_threshold", np.nan)))

    out.append("# Post-Run Objective Compensation Report")
    out.append("## 1) Learned regime parameters")
    out.append(f"- regime_riskoff_market_ret1d_cut = {m_cut:.6f}")
    out.append(f"- regime_riskoff_vix_z_cut         = {v_cut:.6f}")
    out.append(f"- regime_riskoff_threshold_mult    = {off_mult:.6f}")
    out.append(f"- regime_riskon_threshold_mult     = {on_mult:.6f}")
    out.append(f"- meta_eval_threshold              = {thr:.6f}")
    out.append(f"- distance off_mult from 1.0       = {abs(off_mult - 1.0):.4f}")
    out.append(f"- distance on_mult from 1.0        = {abs(on_mult - 1.0):.4f}")

    mc = Path("data/master_complete.csv")
    if mc.exists():
        df = pd.read_csv(mc)
        ro = _risk_off_mask(df, m_cut, v_cut)
        out.append("\n## 2) Regime activation in current data")
        out.append(f"- rows = {len(df)}")
        out.append(f"- risk_off_share_all_rows = {float(ro.mean()):.3f}")

    sig = _load_signals(Path("docs/signals.json"))
    if not sig.empty:
        ro_s = _risk_off_mask(sig, m_cut, v_cut)
        out.append(f"- risk_off_share_signals = {float(ro_s.mean()):.3f}")
        if "prob" in sig.columns:
            p = pd.to_numeric(sig["prob"], errors="coerce")
            out.append(f"- mean_prob_signals = {float(p.mean()):.4f}")
    else:
        out.append("- signals.json not available/parsible")

    pre = _latest_snapshot("pre_meta_optimization_")
    post = _latest_snapshot("post_meta_optimization_")
    out.append("\n## 3) Pre vs post snapshot comparison")
    out.append(f"- latest_pre_snapshot  = {pre.as_posix() if pre else 'n/a'}")
    out.append(f"- latest_post_snapshot = {post.as_posix() if post else 'n/a'}")
    if post and (post / "comparison_summary.json").exists():
        comp = _read_json(post / "comparison_summary.json") or {}
        c = comp.get("comparison_vs_latest_pre", {})
        for k in (
            "delta_signals_json_count",
            "delta_rows",
            "delta_unique_tickers",
            "delta_unique_days",
            "delta_ret4_mean",
            "delta_ret4_win_rate",
        ):
            if k in c:
                out.append(f"- {k} = {c[k]}")
    else:
        out.append("- no post snapshot comparison yet")

    # heuristic decision
    comp_hint = []
    if abs(off_mult - 1.0) < 0.03 and abs(on_mult - 1.0) < 0.03:
        comp_hint.append("regime multipliers close to 1.0")
    if not sig.empty and "prob" in sig.columns:
        pass
    out.append("\n## 4) Interpretation heuristic")
    if comp_hint:
        out.append("- likely compensation/minimal net effect: " + "; ".join(comp_hint))
        out.append("- recommendation: simplify objective OR widen regime multiplier search range")
    else:
        out.append("- regime layer likely active with non-trivial effect")
        out.append("- recommendation: keep, then validate with 2-3 additional OOS cycles")

    rep = Path("data/postrun_objective_compensation_report.txt")
    rep.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote: {rep}")


if __name__ == "__main__":
    main()

