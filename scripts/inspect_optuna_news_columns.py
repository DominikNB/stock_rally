"""
Vergleicht News-Spaltenlisten für assemble_features-Fill:

  - cfg.optuna_training_news_column_union()  (Optuna-Superset)
  - cfg.all_news_model_cols()                 (Legacy-Superset)

Aufruf aus dem Projektroot:

  python scripts/inspect_optuna_news_columns.py
  python scripts/inspect_optuna_news_columns.py --resolve-gcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resolve-gcam",
        action="store_true",
        help="Wie Pipeline: resolve_gkg_gcam_metric_keys (ohne BQ-Explore)",
 )
    p.add_argument(
        "--sample",
        type=int,
        default=12,
        metavar="N",
        help="So viele Beispielnamen aus nur-all_news ausgeben (0=keine)",
    )
    args = p.parse_args()

    from lib.stock_rally_v10 import config as cfg

    if args.resolve_gcam:
        from lib.stock_rally_v10.news import resolve_gkg_gcam_metric_keys

        if hasattr(cfg, "_gkg_gcam_keys_user_snapshot"):
            delattr(cfg, "_gkg_gcam_keys_user_snapshot")
        cfg.GKG_GCAM_METRIC_KEYS = ()
        resolve_gkg_gcam_metric_keys(allow_bigquery_refresh=False)

    union = cfg.optuna_training_news_column_union()
    alln = cfg.all_news_model_cols()
    su, sa = set(union), set(alln)

    print(f"GKG_GCAM_METRIC_KEYS (resolve): {len(cfg._gkg_gcam_keys_clean())} Keys")
    print(f"FEATURE_ASSEMBLE_NEWS_FILL:     {getattr(cfg, 'FEATURE_ASSEMBLE_NEWS_FILL', '?')!r}")
    print(f"optuna_training_news_column_union: {len(union):,} Spalten")
    print(f"all_news_model_cols:               {len(alln):,} Spalten")
    print(f"optuna ⊆ all_news:                 {su.issubset(sa)}")
    print(f"Gleiche Menge:                     {su == sa}")

    only_all = sorted(sa - su)
    only_union = sorted(su - sa)
    if only_union:
        print(f"\nNur in optuna_union ({len(only_union)}) — sollte leer sein:")
        for c in only_union[:20]:
            print(f"  {c}")
        if len(only_union) > 20:
            print(f"  … +{len(only_union) - 20} weitere")
    if only_all and args.sample > 0:
        print(f"\nNur in all_news_model, nicht in optuna_union ({len(only_all)}) — erste {args.sample}:")
        for c in only_all[: args.sample]:
            print(f"  {c}")
        if len(only_all) > args.sample:
            print(f"  … +{len(only_all) - args.sample} weitere")


if __name__ == "__main__":
    main()
