"""OOS-Holdout-Performance aus master_complete.csv — Summary für JSON + Doku."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

HORIZONS = (2, 4, 6, 8, 10)


def _pct_mean(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _pct_win(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return float((s > 0).mean())


def _horizon_block(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {}
    s = pd.to_numeric(df[col], errors="coerce")
    nn = int(s.notna().sum())
    if nn == 0:
        return {"n": 0}
    return {
        "n": nn,
        "mean_pct": round(100 * float(s.mean()), 4),
        "median_pct": round(100 * float(s.median()), 4),
        "win_rate_pct": round(100 * float((s > 0).mean()), 2),
        "std_pct": round(100 * float(s.std()), 4),
        "p10_pct": round(100 * float(s.quantile(0.1)), 4),
        "p90_pct": round(100 * float(s.quantile(0.9)), 4),
    }


def _context_tier_block(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sub = df.dropna(subset=["ret_mean_5"]).copy()
    if sub.empty:
        return out
    if "macro_event_within_2bd" in sub.columns:
        macro = sub["macro_event_within_2bd"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "1.0", "yes"}
            if v is not None and not (isinstance(v, float) and v != v)
            else bool(v)
        )
        for label, mask in [
            ("red_macro_event", macro),
            ("yellow_green_no_macro", ~macro),
        ]:
            g = sub.loc[mask]
            if len(g):
                out[label] = {
                    "n": int(len(g)),
                    **_horizon_block(g, "ret_mean_5"),
                }
    if "regime_vix_level" in sub.columns:
        vix = pd.to_numeric(sub["regime_vix_level"], errors="coerce")
        for label, mask in [
            ("green_vix_ge_20", vix >= 20),
            ("yellow_vix_lt_20", vix < 20),
        ]:
            g = sub.loc[mask & vix.notna()]
            if len(g):
                out[label] = {
                    "n": int(len(g)),
                    **_horizon_block(g, "ret_mean_5"),
                }
    return out


def summarize_holdout_frame(
    df: pd.DataFrame,
    *,
    source: str = "master_complete.csv",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregierte OOS-Kennzahlen (FINAL-Signale mit Forward-Renditen)."""
    out_df = df.copy()
    if "Date" in out_df.columns:
        out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce")

    ret_cols = [f"ret_{h}d" for h in HORIZONS]
    for c in ret_cols + ["ret_mean_5", "prob", "train_target"]:
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    sub = out_df.dropna(subset=["ret_mean_5"])
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_signals_total": int(len(out_df)),
        "n_signals_with_ret_mean_5": int(len(sub)),
        "n_signals_all_horizons_complete": int(
            out_df[ret_cols].notna().all(axis=1).sum() if all(c in out_df.columns for c in ret_cols) else 0
        ),
        "forward_returns": {
            "ret_mean_5": _horizon_block(sub, "ret_mean_5"),
            **{f"ret_{h}d": _horizon_block(sub, f"ret_{h}d") for h in HORIZONS},
        },
        "context_tiers": _context_tier_block(out_df),
    }

    if "train_target" in sub.columns and len(sub):
        payload["label_diag"] = {
            "train_target_eq_1_pct": round(100 * float((sub["train_target"] == 1).mean()), 2),
            "note": "train_target aus Legacy-Band-Label im Holdout-Export — nicht identisch mit rally_plus_entry live.",
        }

    if "Date" in sub.columns and len(sub):
        by_year: dict[str, Any] = {}
        for yr, g in sub.groupby(sub["Date"].dt.year):
            rr = g["ret_mean_5"]
            by_year[str(int(yr))] = {
                "n": int(len(g)),
                "mean_ret_mean_5_pct": round(100 * float(rr.mean()), 4),
                "win_rate_pct": round(100 * float((rr > 0).mean()), 2),
            }
        payload["by_year"] = by_year
        payload["date_min"] = str(sub["Date"].min().date())
        payload["date_max"] = str(sub["Date"].max().date())

    if extra:
        payload.update(extra)
    return payload


def load_run_meta(
    root: Path,
    *,
    signals_json: Path | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    sj = signals_json or root / "docs" / "signals.json"
    if sj.is_file():
        try:
            payload = json.loads(sj.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                meta["website_generated"] = payload.get("generated")
                meta["best_threshold"] = payload.get("threshold")
                sigs = payload.get("signals") or payload.get("signals_holdout_final") or []
                meta["n_oos_signals_json"] = len(sigs) if isinstance(sigs, list) else None
        except Exception as exc:
            meta["signals_json_error"] = str(exc)

    art = artifact_path or root / "models" / "scoring_artifacts.joblib"
    if art.is_file():
        try:
            bundle = joblib.load(art)
            meta["artifact"] = {
                "best_threshold": float(bundle.get("best_threshold", float("nan"))),
                "meta_optuna_best_value": float(bundle.get("meta_optuna_best_value", float("nan"))),
                "meta_objective_mode": str(bundle.get("meta_objective_mode", "tp_precision")),
                "threshold_calibration_end_date": bundle.get("threshold_calibration_end_date"),
            }
        except Exception as exc:
            meta["artifact_error"] = str(exc)
    return meta


def build_performance_report(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[1]
    mc_path = root / "data" / "master_complete.csv"
    if not mc_path.is_file():
        return {"error": f"missing {mc_path.as_posix()}"}
    df = pd.read_csv(mc_path)
    meta = load_run_meta(root)
    return summarize_holdout_frame(df, source="data/master_complete.csv", extra={"run_meta": meta})


def write_performance_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def performance_markdown_table(payload: dict[str, Any]) -> str:
    """Markdown-Fragment für Doku-Generator."""
    if payload.get("error"):
        return f"*Keine Performance-Daten: {payload['error']}*\n"

    lines = [
        f"**Stand:** `{payload.get('generated_at_utc', '?')}` · "
        f"Quelle: `{payload.get('source', '?')}` · "
        f"Signale: **{payload.get('n_signals_total', '?')}** "
        f"(davon **{payload.get('n_signals_with_ret_mean_5', '?')}** mit `ret_mean_5`)",
        "",
    ]
    rm = run_meta = payload.get("run_meta") or {}
    if run_meta.get("website_generated"):
        lines.append(f"Website-Lauf: `{run_meta['website_generated']}` · Schwelle: `{run_meta.get('best_threshold')}`")
    art = run_meta.get("artifact") or {}
    if art:
        lines.append(
            f"Artefakt: `best_threshold={art.get('best_threshold')}` · "
            f"Meta-Optuna-Score=`{art.get('meta_optuna_best_value')}` · "
            f"Kalibrier-Ende=`{art.get('threshold_calibration_end_date')}`"
        )
    lines.append("")

    fwd = payload.get("forward_returns") or {}
    lines.extend(
        [
            "| Horizont | n | Mittel | Median | Win-Rate |",
            "|----------|---|--------|--------|----------|",
        ]
    )
    for key in ["ret_mean_5"] + [f"ret_{h}d" for h in HORIZONS]:
        block = fwd.get(key) or {}
        if not block.get("n"):
            continue
        lines.append(
            f"| `{key}` | {block['n']} | {block.get('mean_pct', '—')}% | "
            f"{block.get('median_pct', '—')}% | {block.get('win_rate_pct', '—')}% |"
        )

    ctx = payload.get("context_tiers") or {}
    if ctx:
        lines.extend(["", "### Kontext-Ampel (ret_mean_5)", "", "| Segment | n | Mittel | Win-Rate |", "|---------|---|--------|----------|"])
        labels = {
            "red_macro_event": "Rot (Makro-Event ±2bd)",
            "yellow_green_no_macro": "Gelb/Grün (kein Makro-Event)",
            "green_vix_ge_20": "Grün (VIX ≥ 20)",
            "yellow_vix_lt_20": "Gelb (VIX < 20)",
        }
        for k, label in labels.items():
            b = ctx.get(k) or {}
            if b.get("n"):
                lines.append(
                    f"| {label} | {b['n']} | {b.get('mean_pct', '—')}% | {b.get('win_rate_pct', '—')}% |"
                )

    by_year = payload.get("by_year") or {}
    if by_year:
        lines.extend(["", "### Nach Jahr", "", "| Jahr | n | Mittel ret_mean_5 | Win-Rate |", "|------|---|-----------------|----------|"])
        for yr in sorted(by_year.keys()):
            b = by_year[yr]
            lines.append(
                f"| {yr} | {b['n']} | {b.get('mean_ret_mean_5_pct', '—')}% | {b.get('win_rate_pct', '—')}% |"
            )

    lines.append("")
    lines.append(
        "*Neuere Signale ohne vollständiges Forward-Fenster fehlen in `ret_mean_5` "
        "(typisch die letzten ~10 Handelstage).*"
    )
    return "\n".join(lines)
