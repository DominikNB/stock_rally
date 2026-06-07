"""SHAP-ADF: Automated Drop Features nach Base-SHAP (Phase 12)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapAdfResult:
    feat_cols_full: list[str]
    feat_cols_pruned: list[str]
    feat_cols_dropped: list[str]
    topk_names: list[str]
    topk_idx: list[int]
    topk_k: int
    topk_mass_frac: float
    replace_feat_cols: bool


def _should_always_keep(
    col: str,
    *,
    always_keep_prefixes: tuple[str, ...],
    always_keep_exact: frozenset[str],
) -> bool:
    if col in always_keep_exact:
        return True
    return any(str(col).startswith(p) for p in always_keep_prefixes)


def compute_shap_adf_kept(
    feat_cols: list[str],
    shap_by_col: dict[str, float],
    *,
    min_abs_shap: float,
    min_keep: int,
    always_keep_prefixes: tuple[str, ...] = (),
    always_keep_exact: frozenset[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Liefert (kept, dropped) in Original-Reihenfolge von ``feat_cols``.

    Spalten mit mean |SHAP| < ``min_abs_shap`` werden verworfen, außer Always-Keep.
    Wenn danach weniger als ``min_keep``: die besten verworfenen wieder aufnehmen.
    """
    exact = always_keep_exact or frozenset()
    kept: list[str] = []
    dropped: list[str] = []
    borderline: list[tuple[float, str]] = []

    for col in feat_cols:
        imp = float(shap_by_col.get(col, 0.0))
        if _should_always_keep(col, always_keep_prefixes=always_keep_prefixes, always_keep_exact=exact):
            kept.append(col)
        elif imp >= float(min_abs_shap):
            kept.append(col)
        else:
            dropped.append(col)
            borderline.append((imp, col))

    if len(kept) < int(min_keep) and borderline:
        borderline.sort(key=lambda x: -x[0])
        need = int(min_keep) - len(kept)
        for _, col in borderline[:need]:
            if col in dropped:
                dropped.remove(col)
                kept.append(col)

    # Reihenfolge wie feat_cols
    kept_set = set(kept)
    kept_ordered = [c for c in feat_cols if c in kept_set]
    dropped_ordered = [c for c in feat_cols if c not in kept_set]
    return kept_ordered, dropped_ordered


def _resolve_meta_shap_top_k(cfg_mod: Any, mean_shap: np.ndarray) -> tuple[int, float]:
    """Kleinstes K mit kumulierter SHAP-Masse >= META_SHAP_CUM_FRAC (wie Phase 12)."""
    n_feat = int(len(mean_shap))
    k_min = max(1, int(getattr(cfg_mod, "META_SHAP_TOP_K_MIN", 1)))
    k_cap = getattr(cfg_mod, "META_SHAP_TOP_K_MAX", None)
    k_max = n_feat if k_cap is None else max(k_min, min(int(k_cap), n_feat))
    total = float(np.sum(mean_shap))
    cum_frac = getattr(cfg_mod, "META_SHAP_CUM_FRAC", None)
    if cum_frac is not None:
        f = float(cum_frac)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"META_SHAP_CUM_FRAC muss in (0, 1] liegen, nicht {f!r}")
        if total <= 0.0:
            k = max(k_min, min(int(getattr(cfg_mod, "META_SHAP_TOP_K", 10)), k_max))
        else:
            order = np.argsort(mean_shap)[::-1]
            vcum = np.cumsum(mean_shap[order])
            thr = f * total
            k = int(np.searchsorted(vcum, thr, side="left")) + 1
            k = max(k_min, min(k, k_max))
    else:
        k = max(k_min, min(int(getattr(cfg_mod, "META_SHAP_TOP_K", 10)), k_max))
    order = np.argsort(mean_shap)[::-1][:k]
    mass = float(np.sum(mean_shap[order])) / total if total > 0 else 0.0
    return k, mass


def resolve_topk_from_mean_shap(
    feat_cols: list[str],
    mean_shap: np.ndarray,
    cfg_mod: Any,
) -> tuple[list[str], list[int], int, float]:
    topk_k, mass_frac = _resolve_meta_shap_top_k(cfg_mod, mean_shap)
    order = np.argsort(mean_shap)[::-1][:topk_k]
    topk_names = [feat_cols[int(i)] for i in order]
    topk_idx = [int(i) for i in order]
    return topk_names, topk_idx, int(topk_k), float(mass_frac)


def apply_shap_adf(
    *,
    feat_cols: list[str],
    shap_df: pd.DataFrame,
    cfg_mod: Any,
) -> ShapAdfResult:
    """
    ADF nach SHAP-Probe (volle ``feat_cols``) vor dem Training aller Base-Modelle.

    Bei ``SHAP_ADF_ENABLED=True``: ``feat_cols_full`` = geprünte Liste für Base-Training;
    ``feat_cols_pruned`` identisch; ``topk_names`` aus dem pruned-Pool.
    """
    if not bool(getattr(cfg_mod, "SHAP_ADF_ENABLED", False)):
        mean_shap = shap_df["mean_abs_shap"].to_numpy(dtype=np.float64)
        topk_names, topk_idx, topk_k, mass = resolve_topk_from_mean_shap(feat_cols, mean_shap, cfg_mod)
        return ShapAdfResult(
            feat_cols_full=list(feat_cols),
            feat_cols_pruned=list(feat_cols),
            feat_cols_dropped=[],
            topk_names=topk_names,
            topk_idx=topk_idx,
            topk_k=topk_k,
            topk_mass_frac=mass,
            replace_feat_cols=False,
        )

    min_shap = float(getattr(cfg_mod, "SHAP_ADF_MIN_ABS_SHAP", 1e-6))
    min_keep = int(getattr(cfg_mod, "SHAP_ADF_MIN_KEEP", 80))
    prefixes = tuple(getattr(cfg_mod, "SHAP_ADF_ALWAYS_KEEP_PREFIXES", ("mr_", "regime_")))
    exact = frozenset(getattr(cfg_mod, "SHAP_ADF_ALWAYS_KEEP_EXACT", ()))
    replace = bool(getattr(cfg_mod, "SHAP_ADF_REPLACE_FEAT_COLS", True))

    shap_by_col = {
        str(r["feature_raw"]): float(r["mean_abs_shap"])
        for _, r in shap_df.iterrows()
    }
    kept, dropped = compute_shap_adf_kept(
        feat_cols,
        shap_by_col,
        min_abs_shap=min_shap,
        min_keep=min_keep,
        always_keep_prefixes=prefixes,
        always_keep_exact=exact,
    )

    # Top-K nur aus kept (Meta-Roh-Features ohne tote Spalten)
    pruned_order = [feat_cols.index(c) for c in kept]
    mean_kept = shap_df.set_index("feature_raw").loc[kept, "mean_abs_shap"].to_numpy(dtype=np.float64)
    topk_names, topk_idx_rel, topk_k, mass = resolve_topk_from_mean_shap(kept, mean_kept, cfg_mod)
    topk_idx = [pruned_order[i] for i in topk_idx_rel]

    feat_out = list(kept) if replace else list(feat_cols)

    print(
        f"SHAP-ADF: {len(feat_cols)} → pruned {len(kept)} "
        f"(dropped {len(dropped)}, min|SHAP|={min_shap:g}, min_keep={min_keep})",
        flush=True,
    )
    if replace:
        print(
            "  Alle Base-Modelle trainieren/scoren auf der geprünten FEAT_COLS-Liste.",
            flush=True,
        )
    else:
        print(
            "  SHAP_ADF_REPLACE_FEAT_COLS=False: FEAT_COLS unverändert (nur Dokumentation/topk aus ADF-Pool).",
            flush=True,
        )
    print(
        f"  Meta Top-K={topk_k} (SHAP-Masse {mass:.1%} auf pruned-Pool), "
        f"Beispiele: {topk_names[:5]}{'…' if len(topk_names) > 5 else ''}",
        flush=True,
    )

    return ShapAdfResult(
        feat_cols_full=feat_out,
        feat_cols_pruned=kept,
        feat_cols_dropped=dropped,
        topk_names=topk_names,
        topk_idx=topk_idx,
        topk_k=topk_k,
        topk_mass_frac=mass,
        replace_feat_cols=replace,
    )


def enrich_shap_payload_with_adf(payload: dict, adf: ShapAdfResult, cfg_mod: Any) -> dict:
    out = dict(payload)
    out["shap_adf"] = {
        "enabled": bool(getattr(cfg_mod, "SHAP_ADF_ENABLED", False)),
        "min_abs_shap": float(getattr(cfg_mod, "SHAP_ADF_MIN_ABS_SHAP", 1e-6)),
        "min_keep": int(getattr(cfg_mod, "SHAP_ADF_MIN_KEEP", 80)),
        "replace_feat_cols": bool(adf.replace_feat_cols),
        "feature_count_pruned": len(adf.feat_cols_pruned),
        "feature_count_dropped": len(adf.feat_cols_dropped),
        "feat_cols_pruned": list(adf.feat_cols_pruned),
        "feat_cols_dropped_sample": list(adf.feat_cols_dropped[:40]),
    }
    out["topk_names_raw"] = list(adf.topk_names)
    out["topk_k"] = int(adf.topk_k)
    out["topk_mass_frac"] = float(adf.topk_mass_frac)
    return out
