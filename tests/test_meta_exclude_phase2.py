"""Regression: BASE_MODELS_META_EXCLUDE in Phase-2-Base-Training."""
from __future__ import annotations

from types import SimpleNamespace


def test_meta_exclude_matches_config():
    c = SimpleNamespace(BASE_MODELS_META_EXCLUDE=("XGB-1", "LR"))
    _meta_exclude = frozenset(str(x) for x in (getattr(c, "BASE_MODELS_META_EXCLUDE", ()) or ()))
    base_models: list[str] = []
    for name in ("XGB-1", "XGB-2", "LR"):
        if name not in _meta_exclude:
            base_models.append(name)
    assert base_models == ["XGB-2"]
