"""
Kurztest: vorgefertigter Feature-Subset-Pool + Phase-1-Optuna (nur optimize_xgb).

Lädt ``config``, setzt **nur im Prozess** Test-Overrides, baut Daten/Split wie die Pipeline,
bindet ``optimize_xgb`` und führt wenige Trials aus. Keine Phasen 13–17.

Vom Projektroot:

  .venv\\Scripts\\python.exe scripts/smoke_test_feature_subset_optuna.py
  .venv\\Scripts\\python.exe scripts/smoke_test_feature_subset_optuna.py --trials 8

Hinweis: Erster Lauf kann wie ``pipeline_runner`` mehrere Minuten für assemble/split brauchen;
Prescreens sind für den Smoke-Lauf ausgeschaltet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5, help="Optuna-Trials (Default: 5)")
    args = ap.parse_args()

    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.data_and_split import run_data_download_and_split
    from lib.stock_rally_v10.pipeline_runner import bind_step_functions

    # Nur dieser Prozess — config.py auf Disk unverändert.
    cfg.OPTUNA_FEATURE_SUBSET_POOL_ENABLED = True
    cfg.N_OPTUNA_TRIALS = max(1, int(args.trials))
    cfg.OPTUNA_WF_SPLITS = 2
    # Kleiner Pool = schneller Aufbau; >= 100 (Mindestanforderung in optimize_xgb).
    cfg.OPTUNA_FEATURE_SUBSET_POOL_SIZE = min(
        2500, max(150, int(getattr(cfg, "OPTUNA_FEATURE_SUBSET_POOL_SIZE", 10000)))
    )
    cfg.FEATURE_PRESCREEN_ENABLED = False
    cfg.NEWS_CORRELATION_PRESCREEN_ENABLED = False

    print(
        "[smoke] OPTUNA_FEATURE_SUBSET_POOL_ENABLED=True, "
        f"N_OPTUNA_TRIALS={cfg.N_OPTUNA_TRIALS}, OPTUNA_WF_SPLITS={cfg.OPTUNA_WF_SPLITS}, "
        f"POOL_SIZE={cfg.OPTUNA_FEATURE_SUBSET_POOL_SIZE}, Prescreens off",
        flush=True,
    )

    bind_step_functions()
    run_data_download_and_split()

    best = cfg.optimize_xgb(cfg.df_train, n_trials=int(args.trials), seed_params=cfg.SEED_PARAMS)
    fcols = best.get("feature_subset_columns")
    sid = best.get("feature_subset_id")
    print(
        "\n[smoke] optimize_xgb fertig.",
        f"best feature_subset_id={sid!r}, "
        f"len(feature_subset_columns)={len(fcols) if fcols else 0}",
        flush=True,
    )
    if fcols:
        print(f"  erste Spalten: {fcols[:8]!r}", flush=True)


if __name__ == "__main__":
    main()
