"""Regime + trainierbare ``mr_*`` für Base-Classifier (festes Spalten-Set)."""
from __future__ import annotations

import pandas as pd

from lib.stock_rally_v10.macro_vol_enrich import (
    TRAINING_MACRO_VOL_COLS,
    enrich_macro_volatility_features,
)
from lib.signal_extra_filters import add_short_horizon_macro_regime_columns


def augment_df_macro_regime_and_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Kurzfrist-Regime mergen (falls noch nicht da), danach nur trainierbare ``mr_*``."""
    out = df
    if "regime_vix_level" not in out.columns:
        print(
            "assemble_features: Kurzfrist-Regime (^VIX/^TNX/SPY-RV) …",
            flush=True,
        )
        out = add_short_horizon_macro_regime_columns(out)
    print(
        "assemble_features: enrich_macro_volatility_features (nur Training-spf. mr_*) …",
        flush=True,
    )
    return enrich_macro_volatility_features(out)


def append_macro_regime_vol_numeric_cols(
    feat_cols: list[str],
    df_train: pd.DataFrame,
) -> list[str]:
    """Hängt nur ``TRAINING_MACRO_VOL_COLS`` an, wenn Spalte existiert und numerisch (oder bool) ist."""
    seen = set(feat_cols)
    out = list(feat_cols)
    added = 0
    for c in TRAINING_MACRO_VOL_COLS:
        if c not in df_train.columns or c in seen:
            continue
        s = df_train[c]
        if pd.api.types.is_bool_dtype(s):
            pass
        elif not pd.api.types.is_numeric_dtype(s):
            continue
        out.append(c)
        seen.add(c)
        added += 1
    if added:
        print(
            f"Phase 12: +{added} Macro-Vola-Spalte(n) (festes Training-Set, max. {len(TRAINING_MACRO_VOL_COLS)}).",
            flush=True,
        )
    return out
