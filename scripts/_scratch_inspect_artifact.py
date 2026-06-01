"""Inspect scoring artifact for analysis."""
import joblib

a = joblib.load("models/scoring_artifacts.joblib")
print("Meta best value:", a.get("meta_optuna_best_value"))
print()
mp = a.get("meta_optuna_best_params", {})
print("Meta best params (filter/threshold-relevant):")
for k in [
    "meta_eval_threshold",
    "dyn_rsi_trigger",
    "signal_max_rsi",
    "signal_max_vol_stress_z",
    "mult_final_threshold_1",
    "mult_final_threshold_2",
    "mult_final_threshold_3",
    "dyn_vvix_trigger",
    "dyn_bb_pband_trigger",
    "signal_min_blue_sky_volume_z",
    "signal_skip_near_peak",
    "peak_lookback_days",
    "peak_min_dist_from_high_pct",
]:
    print(f"  {k}: {mp.get(k)}")
print()
print("Meta user attrs:")
for k, v in (a.get("meta_optuna_best_user_attrs") or {}).items():
    print(f"  {k}: {v}")
print()
print("Final best_threshold:", a.get("best_threshold"))
print()
bp = a.get("best_params", {})
print("Base best_params (selected):")
for k in [
    "return_window",
    "rally_threshold",
    "lead_days",
    "entry_days",
    "min_rally_tail_days",
    "consecutive_days",
    "signal_cooldown_days",
    "base_eval_threshold",
    "rsi_window",
    "bb_window",
    "sma_window",
    "max_depth",
    "learning_rate",
    "n_estimators",
]:
    if k in bp:
        print(f"  {k}: {bp.get(k)}")
print()
print("Base best value:", a.get("base_optuna_best_value"))
print("CONSECUTIVE_DAYS:", a.get("CONSECUTIVE_DAYS"))
print("SIGNAL_COOLDOWN_DAYS:", a.get("SIGNAL_COOLDOWN_DAYS"))
print()
print("Phase5 multipliers in artifact:", a.get("mult_final_threshold_1"), a.get("mult_final_threshold_2"), a.get("mult_final_threshold_3"))
print("Phase5 signal_max_rsi:", a.get("signal_max_rsi"))
print("Phase5 dyn_*: vvix", a.get("dyn_vvix_trigger"), " rsi", a.get("dyn_rsi_trigger"), " bb", a.get("dyn_bb_pband_trigger"))
