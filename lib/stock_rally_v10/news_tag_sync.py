"""Synchronisiert News-Feature-Namen zwischen Base-Optuna-Tag und Meta-Top-K."""
from __future__ import annotations

from typing import Any


def news_tag_from_params(best_params: dict[str, Any], cfg_mod: Any) -> str:
    """Aktuelles News-Fenster-Tripel als ``mom_vol_tone``-String."""
    fn = getattr(cfg_mod, "news_feat_tag", None)
    if not callable(fn):
        raise AttributeError("cfg_mod.news_feat_tag fehlt")
    return str(
        fn(
            int(best_params["news_mom_w"]),
            int(best_params["news_vol_ma"]),
            int(best_params["news_tone_roll"]),
        )
    )


def _replace_news_tag_in_name(col: str, new_tag: str) -> str:
    """Ersetzt ``news_macro_X_Y_Z`` / ``news_sec_X_Y_Z`` durch ``new_tag``."""
    for prefix in ("news_macro_", "news_sec_"):
        if not col.startswith(prefix):
            continue
        rest = col[len(prefix) :]
        parts = rest.split("_")
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            suffix = "_".join(parts[3:])
            base = f"{prefix}{new_tag}"
            return f"{base}_{suffix}" if suffix else base
    return col


def rewrite_topk_names_for_news_tag(
    topk_names: list[str],
    *,
    best_params: dict[str, Any],
    cfg_mod: Any,
    artifact_news_tag: str | None = None,
) -> tuple[list[str], str, list[str]]:
    """
    Mappt gespeicherte Top-K-Namen auf den News-Tag aus ``best_params``.

    Returns:
        (synced_names, current_tag, changed_pairs as "old -> new")
    """
    current = news_tag_from_params(best_params, cfg_mod)
    out: list[str] = []
    changes: list[str] = []
    for name in topk_names:
        new_name = _replace_news_tag_in_name(str(name), current)
        if new_name != name:
            changes.append(f"{name} -> {new_name}")
        out.append(new_name)
    if artifact_news_tag and artifact_news_tag != current:
        changes.insert(0, f"artifact_news_tag={artifact_news_tag} -> current={current}")
    return out, current, changes


def resolve_topk_idx(
    topk_names: list[str],
    feat_cols: list[str],
) -> tuple[list[str], list[int], list[str]]:
    """Indizes in ``feat_cols``; fehlende Spalten werden übersprungen (mit leerem Index-Set warnen)."""
    col_index = {str(c): i for i, c in enumerate(feat_cols)}
    names: list[str] = []
    idx: list[int] = []
    missing: list[str] = []
    for n in topk_names:
        i = col_index.get(str(n))
        if i is None:
            missing.append(str(n))
            continue
        names.append(str(n))
        idx.append(int(i))
    return names, idx, missing


def sync_topk_for_meta(
    cfg_mod: Any,
    *,
    feat_cols: list[str] | None = None,
    best_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[list[str], Any]:
    """
    Schreibt ``topk_names`` / ``topk_idx`` auf ``cfg_mod`` (News-Tag aus ``best_params``).
    """
    import numpy as np

    topk_raw = list(getattr(cfg_mod, "topk_names", None) or [])
    if not topk_raw:
        return [], np.asarray([], dtype=np.int64)
    bp = dict(best_params or getattr(cfg_mod, "best_params", None) or {})
    if not bp.get("news_mom_w"):
        sp = getattr(cfg_mod, "SEED_PARAMS", None) or {}
        for k in ("news_mom_w", "news_vol_ma", "news_tone_roll"):
            if k not in bp and k in sp:
                bp[k] = sp[k]
    artifact_tag = None
    shap = getattr(cfg_mod, "base_feature_shap_report", None)
    if isinstance(shap, dict):
        artifact_tag = shap.get("news_tag")
    synced, tag, changes = rewrite_topk_names_for_news_tag(
        topk_raw,
        best_params=bp,
        cfg_mod=cfg_mod,
        artifact_news_tag=str(artifact_tag) if artifact_tag else None,
    )
    cols = list(feat_cols or getattr(cfg_mod, "FEAT_COLS", None) or [])
    names, idx, missing = resolve_topk_idx(synced, cols)
    if verbose and changes:
        print(
            f"[News-Tag-Sync] Top-K auf Tag `{tag}` gemappt ({len(changes)} Umbenennung(en)).",
            flush=True,
        )
        for line in changes[:8]:
            print(f"  {line}", flush=True)
        if len(changes) > 8:
            print(f"  … +{len(changes) - 8} weitere", flush=True)
    if verbose and missing:
        print(
            f"[News-Tag-Sync] WARN: {len(missing)} Top-K-Spalte(n) fehlen in FEAT_COLS "
            f"(z. B. {missing[:3]}) — Meta-Stack verkürzt sich.",
            flush=True,
        )
    cfg_mod.topk_names = names
    cfg_mod.topk_idx = np.asarray(idx, dtype=np.int64)
    cfg_mod.topk_news_tag_synced = tag
    return names, cfg_mod.topk_idx
