"""
Offline-Analyse der News-Feature-Shards (Parquet unter data/feature_shards_news).

Schreibt CSVs nach data/news_feature_analysis/ und druckt eine Kurz-Zusammenfassung.
Lauf unabhängig von der Pipeline (keine Imports aus Training-Phasen nötig).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _load_manifest(shard_dir: Path) -> dict[str, str]:
    p = shard_dir / "news_shards_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Manifest fehlt: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    tags = obj.get("tags") or {}
    return {str(k): str(v) for k, v in tags.items()}


def _analyze_one_parquet(path: Path, tag: str) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    df = pd.read_parquet(path)
    n_rows = len(df)
    news_cols = [c for c in df.columns if str(c).startswith("news_")]
    if not news_cols:
        return pd.DataFrame(), {
            "tag": tag,
            "n_rows": n_rows,
            "n_news_cols": 0,
            "n_constant": 0,
            "n_constant_zero": 0,
            "n_constant_nonzero": 0,
            "n_all_nan": 0,
            "n_varying": 0,
            "frac_cols_constant": 0.0,
            "seconds": time.perf_counter() - t0,
        }

    mat = df[news_cols].apply(pd.to_numeric, errors="coerce")
    arr = mat.to_numpy(dtype=np.float64, copy=False)
    fin = np.isfinite(arr)
    n_finite = fin.sum(axis=0)
    n_nan = (~fin).sum(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            stds = np.nanstd(arr, axis=0, ddof=0)
            amin = np.nanmin(arr, axis=0)
            amax = np.nanmax(arr, axis=0)
            mu = np.where(n_finite > 0, np.nanmean(arr, axis=0), np.nan)
        # Eine Beobachtung -> Std per Definition 0 (nanstd liefert nan)
        stds = np.where(n_finite > 1, stds, np.where(n_finite == 1, 0.0, np.nan))
        spread = amax - amin
        pct_zero = np.where(n_finite > 0, ((arr == 0.0) & fin).sum(axis=0) / n_finite, np.nan)

    eps_s = 1e-12
    eps_std = 1e-12
    all_nan = n_finite == 0
    const = (~all_nan) & (spread <= eps_s) & (stds <= eps_std)
    const_zero = const & (np.abs(mu) <= eps_s)
    const_nz = const & (np.abs(mu) > eps_s)
    varying = (~all_nan) & ~const

    n_news = len(news_cols)
    summary = {
        "tag": tag,
        "n_rows": int(n_rows),
        "n_news_cols": int(n_news),
        "n_constant": int(const.sum()),
        "n_constant_zero": int(const_zero.sum()),
        "n_constant_nonzero": int(const_nz.sum()),
        "n_all_nan": int(all_nan.sum()),
        "n_varying": int(varying.sum()),
        "frac_cols_constant": float(const.sum()) / float(n_news) if n_news else 0.0,
        "seconds": time.perf_counter() - t0,
    }

    detail = pd.DataFrame(
        {
            "tag": tag,
            "column": news_cols,
            "n_rows": n_rows,
            "n_finite": n_finite.astype(np.int64),
            "n_nan": n_nan.astype(np.int64),
            "std": stds,
            "min": amin,
            "max": amax,
            "spread": spread,
            "pct_zero": pct_zero,
            "is_all_nan": all_nan,
            "is_constant": const,
            "is_constant_zero": const_zero,
            "is_constant_nonzero": const_nz,
        }
    )
    return detail, summary


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="News-Feature-Shards analysieren")
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=root / "data" / "feature_shards_news",
        help="Verzeichnis mit news_shards_manifest.json und Parquet-Dateien",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "news_feature_analysis",
        help="Ausgabe für CSVs",
    )
    ap.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Nur diese feat_tags (z.B. 5_10_3). Standard: alle im Manifest.",
    )
    args = ap.parse_args()
    shard_dir = args.shard_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_to_file = _load_manifest(shard_dir)
    if args.tags:
        wanted = set(args.tags)
        tag_to_file = {t: f for t, f in tag_to_file.items() if t in wanted}
        missing = wanted - set(tag_to_file.keys())
        if missing:
            print(f"[analyze_news_features] Unbekannte Tags (ignoriert): {sorted(missing)}", file=sys.stderr)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_prefix = f"run_{ts}"

    summaries: list[dict] = []
    detail_parts: list[pd.DataFrame] = []

    for tag in sorted(tag_to_file.keys()):
        fname = tag_to_file[tag]
        path = shard_dir / fname
        if not path.is_file():
            print(f"[analyze_news_features] Fehlt: {path}", file=sys.stderr)
            continue
        print(f"[analyze_news_features] {tag} <- {path.name} …", flush=True)
        detail, summ = _analyze_one_parquet(path, tag)
        summaries.append(summ)
        if not detail.empty:
            detail_parts.append(detail)

    by_tag = pd.DataFrame(summaries)
    by_tag_path = out_dir / f"{run_prefix}_by_tag.csv"
    by_tag.to_csv(by_tag_path, index=False)

    if detail_parts:
        columns_path = out_dir / f"{run_prefix}_columns.csv"
        pd.concat(detail_parts, ignore_index=True).to_csv(columns_path, index=False)
    else:
        columns_path = None

    meta = {
        "created_utc": ts,
        "shard_dir": str(shard_dir),
        "by_tag_csv": str(by_tag_path),
        "columns_csv": str(columns_path) if columns_path else None,
        "tags_analyzed": list(by_tag["tag"]),
    }
    meta_path = out_dir / f"{run_prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== News-Feature-Analyse (Kurz) ===", flush=True)
    if by_tag.empty:
        print("Keine Shards verarbeitet.", flush=True)
        return 1
    print(by_tag.to_string(index=False), flush=True)
    print(f"\nAusgabe: {by_tag_path}", flush=True)
    if columns_path:
        print(f"         {columns_path}", flush=True)
    print(f"         {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
