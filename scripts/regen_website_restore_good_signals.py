"""
Website aus gutem signals.json (226b1dd) + Kontext-Ampel — ohne Meta-Rescoring.

1. Signale aus Git-Stand 2026-06-06 laden
2. Holdout-Anreicherung (macro_event, VIX) + Kontext-Tier
3. Charts aus gleichem Commit wiederherstellen (falls fehlend)
4. Phase 17 nur HTML/Charts (PHASE17_WEBSITE_SIGNALS_OVERRIDE)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

GOOD_COMMIT = "226b1dd"


def _load_good_signals() -> list[dict]:
    raw = subprocess.check_output(
        ["git", "show", f"{GOOD_COMMIT}:docs/signals.json"],
        cwd=str(_ROOT),
    )
    data = json.loads(raw.decode("utf-8"))
    return list(data.get("signals") or [])


def _enrich_with_context_tier(signals: list[dict]) -> list[dict]:
    from holdout.build_holdout_signals_master import main as build_holdout_master
    from lib.signal_context_tier import attach_signal_context_tier
    from lib.stock_rally_v10.equity_classification import CLASSIFICATION_COLUMN_KEYS

    rows = [
        {
            "ticker": s["ticker"],
            "Date": s["date"],
            "prob": float(s["prob"]),
            "threshold_used": float(s.get("threshold_used", 0.9) or 0.9),
            "company": s.get("company", s["ticker"]),
            "sector": s.get("sector", "—"),
            **{k: s.get(k, "") for k in CLASSIFICATION_COLUMN_KEYS},
        }
        for s in signals
    ]
    exported = build_holdout_master(holdout_df=pd.DataFrame(rows))
    if exported is None or len(exported) == 0:
        raise RuntimeError("build_holdout_signals_master lieferte 0 Zeilen")

    out: list[dict] = []
    for _, r in exported.iterrows():
        sig = {
            "ticker": str(r["ticker"]),
            "company": str(r.get("company", r["ticker"])),
            "sector": str(r.get("sector", "—")),
            **{k: str(r.get(k, "")) for k in CLASSIFICATION_COLUMN_KEYS},
            "date": str(r["Date"])[:10],
            "prob": float(r["prob"]),
        }
        if "regime_vix_level" in r.index:
            v = r.get("regime_vix_level")
            if v is not None and not (isinstance(v, float) and v != v):
                sig["regime_vix_level"] = float(v)
        if "macro_event_within_2bd" in r.index:
            v = r.get("macro_event_within_2bd")
            if v is not None and not (isinstance(v, float) and v != v):
                if isinstance(v, (bool,)):
                    sig["macro_event_within_2bd"] = bool(v)
                elif isinstance(v, (int, float)):
                    sig["macro_event_within_2bd"] = bool(int(v))
                else:
                    sig["macro_event_within_2bd"] = str(v).strip().lower() in {"true", "1", "yes"}
        attach_signal_context_tier(sig)
        out.append(sig)

    def _key(s: dict) -> tuple[int, float]:
        return (int(pd.Timestamp(s["date"]).value), float(s.get("prob", 0.0)))

    out.sort(key=_key, reverse=True)
    return out


def main() -> None:
    print(f"Lade Signale aus {GOOD_COMMIT} …", flush=True)
    signals = _load_good_signals()
    print(f"  {len(signals)} Signale aus gutem Export.", flush=True)

    print("Holdout-Anreicherung + Kontext-Ampel …", flush=True)
    signals = _enrich_with_context_tier(signals)
    print(f"  {len(signals)} Signale angereichert.", flush=True)

    print(f"Stelle docs/charts aus {GOOD_COMMIT} wieder her …", flush=True)
    subprocess.run(
        ["git", "checkout", GOOD_COMMIT, "--", "docs/charts"],
        cwd=str(_ROOT),
        check=True,
    )

    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.data_and_split import run_data_download_and_split
    from lib.stock_rally_v10.pipeline_runner import bind_step_functions
    from lib.stock_rally_v10.training_phases.daily_scoring_html import run_phase_daily_scoring_html

    bind_step_functions()
    cfg.SCORING_ONLY = True
    cfg.RETRAIN_META_ONLY = False
    cfg.PHASE17_WEBSITE_SIGNALS_OVERRIDE = signals
    cfg.log_pipeline_mode_banner()
    cfg.load_scoring_artifacts()
    run_data_download_and_split()
    run_phase_daily_scoring_html(cfg)


if __name__ == "__main__":
    main()
