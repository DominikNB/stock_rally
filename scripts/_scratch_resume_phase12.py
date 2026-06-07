"""
Phase 12 fortsetzen: Optuna-Checkpoint + Phase-11-Artefakt, keine neuen Trials.

Vom Projektroot:
  .venv\\Scripts\\python.exe scripts/_scratch_resume_phase12.py

Setzt nur im Prozess:
  N_OPTUNA_TRIALS=0, STATISTICAL_PRE_PRUNE_REUSE_ARTIFACT=True,
  optional kleineres Universum (--universe-frac) für schnelleren Smoke.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Kein blockierendes plt.show() im Headless-/CLI-Test
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--universe-frac",
        type=float,
        default=None,
        help="Optional z. B. 0.05 für schnelleren Daten-Split (Default: cfg.UNIVERSE_FRACTION)",
    )
    args = ap.parse_args()

    from lib.stock_rally_v10 import config as cfg
    from lib.stock_rally_v10.data_and_split import run_data_download_and_split
    from lib.stock_rally_v10.pipeline_runner import bind_step_functions
    from lib.stock_rally_v10.training_phases.feature_pre_pruning import run_statistical_pre_pruning
    from lib.stock_rally_v10.training_phases.optuna_base_models import run_phase_optuna_base_models
    from lib.stock_rally_v10.optuna_train import _base_optuna_checkpoint_path

    cfg.N_OPTUNA_TRIALS = 0
    cfg.STATISTICAL_PRE_PRUNE_REUSE_ARTIFACT = True
    if args.universe_frac is not None:
        cfg.UNIVERSE_FRACTION = float(args.universe_frac)

    ckpt = _base_optuna_checkpoint_path(cfg)
    print(
        f"[resume] N_OPTUNA_TRIALS=0, Checkpoint={ckpt.resolve()!s}, "
        f"exists={ckpt.is_file()}, UNIVERSE_FRACTION={cfg.UNIVERSE_FRACTION}",
        flush=True,
    )
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint fehlt: {ckpt}")

    bind_step_functions()
    run_data_download_and_split()
    run_statistical_pre_pruning(cfg)
    run_phase_optuna_base_models(cfg)
    print("[resume] Phase 12 fertig — Meta-Stack-Ausschluss und Base-Modelle OK.", flush=True)


if __name__ == "__main__":
    main()
