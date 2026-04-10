"""
Günstige BigQuery-Exploration von GKG V2Themes (Token-Häufigkeiten, Stichproben, Lift vs. Makro).

Logik: ``lib.stock_rally_v10.gkg_theme_explore`` — hier nur CSV/Stichproben-CLI.

Beispiele:
  python scripts/explore_gkg_themes.py --days 14 --channels macro healthcare
  python scripts/explore_gkg_themes.py --start 2024-01-01 --end 2024-01-31 --channels industrial --top 200

Doku: https://www.gdeltproject.org/data.html#documentation
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.stock_rally_v10 import config as cfg
from lib.stock_rally_v10.gkg_theme_explore import (
    partition_params,
    parse_day,
    run_bq_query,
    sql_sample_raw,
    sql_top_tokens,
    sql_total_rows,
    table_sql,
    theme_where_for_channel,
)
from lib.stock_rally_v10.news import _bq_geo_sql
from lib.stock_rally_v10.theme_hints import MACRO_THEME_HINTS, SECTOR_THEME_HINTS, hint_hits_for_token


def _date_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.end:
        end = parse_day(args.end)
    else:
        _ed = getattr(cfg, "END_DATE", None)
        if _ed is not None:
            end = parse_day(str(_ed)[:10])
        else:
            end = datetime.now(timezone.utc)
    if args.start:
        start = parse_day(args.start)
    else:
        start = end - timedelta(days=int(args.days))
    return start, end


def print_hint_overlap(channel: str, tokens: list[tuple[str, int, float]]) -> None:
    hints = MACRO_THEME_HINTS if channel == "macro" else SECTOR_THEME_HINTS.get(channel, [])
    if not hints:
        return
    print(f"\n--- Fachlogik-Überlapp (theme_hints) für {channel!r} — Treffer in Top-Liste ---")
    shown = 0
    for tok, _cnt, _sh in tokens[: min(80, len(tokens))]:
        hits = hint_hits_for_token(tok, hints)
        if hits:
            print(f"  {tok!r}  ← {', '.join(hits)}")
            shown += 1
            if shown >= 25:
                break
    if shown == 0:
        print("  (keine direkten Substring-Treffer in den Top-80.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="GKG V2Themes: günstige BQ-Exploration")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--channels", nargs="+", default=["macro", "healthcare"])
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--out-dir", type=str, default="data/gkg_explore")
    ap.add_argument("--no-macro-baseline", action="store_true")
    ap.add_argument("--universe", choices=("filtered", "open"), default="filtered")
    args = ap.parse_args()

    start, end = _date_window(args)
    table = table_sql(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start.date()}_{end.date()}"
    params = partition_params(start, end)
    uni = args.universe

    print(f"Fenster: {start.date()} … {end.date()}  |  Tabelle: `{table}`")

    macro_total_sql = sql_total_rows(theme_where_for_channel(cfg, "macro", uni), _bq_geo_sql(True), table)
    macro_total = int(run_bq_query(macro_total_sql, params)[0]["n"])

    macro_tokens: dict[str, tuple[int, float]] = {}
    if not args.no_macro_baseline:
        macro_sql = sql_top_tokens(
            theme_where_for_channel(cfg, "macro", uni), _bq_geo_sql(True), table, max(args.top, 400)
        )
        for row in run_bq_query(macro_sql, params):
            macro_tokens[str(row["theme_token"])] = (
                int(row["row_hits"]),
                float(row["token_share"] or 0.0),
            )

    for channel in args.channels:
        tw = theme_where_for_channel(cfg, channel, uni)
        geo = _bq_geo_sql(channel == "macro")
        print(f"\n=== Channel: {channel} ===")
        tot_sql = sql_total_rows(tw, geo, table)
        n_rows = int(run_bq_query(tot_sql, params)[0]["n"])
        print(f"Zeilen (geschätzt Filter): {n_rows:,}  |  Makro-Baseline-Zeilen: {macro_total:,}")

        sql = sql_top_tokens(tw, geo, table, args.top)
        rows = run_bq_query(sql, params)
        tokens = [
            (str(r["theme_token"]), int(r["row_hits"]), float(r["token_share"] or 0.0)) for r in rows
        ]

        out_csv = out_dir / f"top_themes_{channel}_{tag}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if channel != "macro" and macro_tokens and not args.no_macro_baseline:
                w.writerow(
                    ["theme_token", "row_hits", "token_share_channel", "token_share_macro", "lift_share_ratio"]
                )
                for tok, _c, s_share in tokens:
                    m_t = macro_tokens.get(tok)
                    m_share = m_t[1] if m_t else 0.0
                    lift = (s_share + 1e-15) / (m_share + 1e-15)
                    w.writerow([tok, _c, f"{s_share:.8e}", f"{m_share:.8e}", f"{lift:.4f}"])
            else:
                w.writerow(["theme_token", "row_hits", "token_share"])
                for tok, _c, s_share in tokens:
                    w.writerow([tok, _c, f"{s_share:.8e}"])
        print(f"Wrote {out_csv}")

        if args.samples > 0:
            samp_sql = sql_sample_raw(tw, geo, table, args.samples)
            samp_path = out_dir / f"sample_v2themes_{channel}_{tag}.csv"
            srows = run_bq_query(samp_sql, params)
            with samp_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["d", "V2Themes_snippet"])
                for r in srows:
                    raw = str(r["V2Themes"] or "")
                    w.writerow([str(r["d"]), raw[:2000]])
            print(f"Wrote {samp_path}")

        print_hint_overlap(channel, tokens)

    print("\nAutomatische Pipeline: cfg.GKG_AUTO_EXPLORE_THEMES=True + Training (siehe config.py).")


if __name__ == "__main__":
    main()
