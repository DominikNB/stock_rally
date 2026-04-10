"""
GKG V2Themes: BigQuery-Exploration + automatische Theme-Auswahl für die Pipeline.

- ``run_auto_explore_and_save``: bei cfg.GKG_AUTO_EXPLORE_THEMES True aus data_and_split aufrufen.
- ``apply_theme_selection_from_file``: bei False — gespeicherte Tokens in MACRO/SECTOR-SQL einbauen.

CLI: weiterhin ``scripts/explore_gkg_themes.py`` (CSV/Stichproben).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from google.cloud import bigquery

from lib.stock_rally_v10.news import _bq_geo_sql, _bq_resolve_project, _bq_ts_utc_midnight

# GDELT-Theme-Codes (ASCII); ohne Zeichen, die SQL/LIKE brechen.
_TOKEN_SAFE = re.compile(r"^[\w.\-]{2,160}$", re.ASCII)


def table_sql(cfg_mod: Any) -> str:
    return str(cfg_mod.GDELT_BQ_EVENTS_TABLE).strip().strip("`") or "gdelt-bq.gdeltv2.gkg_partitioned"


def parse_day(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def partition_params(start: datetime, end: datetime) -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("ts_start", "TIMESTAMP", _bq_ts_utc_midnight(start)),
        bigquery.ScalarQueryParameter("ts_end", "TIMESTAMP", _bq_ts_utc_midnight(end)),
    ]


def theme_where_for_channel(cfg_mod: Any, channel: str, universe: str) -> str:
    if universe == "open":
        return "(1=1)"
    if channel == "macro":
        return f"({cfg_mod.MACRO_BQ_THEME_WHERE})"
    if channel not in cfg_mod.SECTOR_BQ_THEME_WHERE:
        raise ValueError(f"Unknown channel {channel!r}")
    return f"({cfg_mod.SECTOR_BQ_THEME_WHERE[channel]})"


def sql_top_tokens(theme_where: str, geo_sql: str, table: str, top_n: int) -> str:
    return f"""
WITH filtered AS (
  SELECT V2Themes
  FROM `{table}`
  WHERE _PARTITIONTIME >= @ts_start
    AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)
    AND V2Themes IS NOT NULL
    AND V2Tone IS NOT NULL
    AND {theme_where}
    {geo_sql}
),
split_themes AS (
  SELECT TRIM(t) AS theme_token
  FROM filtered,
  UNNEST(SPLIT(V2Themes, ',')) AS t
  WHERE LENGTH(TRIM(t)) > 0
),
agg AS (
  SELECT theme_token, COUNT(*) AS row_hits
  FROM split_themes
  GROUP BY theme_token
),
ranked AS (
  SELECT theme_token, row_hits, SUM(row_hits) OVER () AS total_token_hits
  FROM agg
)
SELECT theme_token, row_hits, SAFE_DIVIDE(row_hits, total_token_hits) AS token_share
FROM ranked
ORDER BY row_hits DESC
LIMIT {int(top_n)}
"""


def sql_total_rows(theme_where: str, geo_sql: str, table: str) -> str:
    return f"""
SELECT COUNT(*) AS n
FROM `{table}`
WHERE _PARTITIONTIME >= @ts_start
  AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)
  AND V2Themes IS NOT NULL
  AND V2Tone IS NOT NULL
  AND {theme_where}
  {geo_sql}
"""


def sql_sample_raw(theme_where: str, geo_sql: str, table: str, n: int) -> str:
    return f"""
SELECT DATE(_PARTITIONTIME) AS d, V2Themes
FROM `{table}`
WHERE _PARTITIONTIME >= @ts_start
  AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)
  AND V2Themes IS NOT NULL
  AND V2Tone IS NOT NULL
  AND {theme_where}
  {geo_sql}
LIMIT {int(n)}
"""


def run_bq_query(sql: str, params: list) -> list[bigquery.table.Row]:
    rows, _ = run_bq_query_with_bytes(sql, params)
    return rows


def run_bq_query_with_bytes(sql: str, params: list) -> tuple[list[bigquery.table.Row], int]:
    proj = _bq_resolve_project()
    if not proj:
        raise OSError("BigQuery: kein Projekt (cfg.BQ_PROJECT_ID / GOOGLE_CLOUD_PROJECT / gcloud).")
    client = bigquery.Client(project=proj)
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    job = client.query(sql, job_config=job_config)
    rows = list(job.result())
    nbytes = int(job.total_bytes_processed or 0)
    return rows, nbytes


def _log_gkg_exploration_summary(cfg_mod: Any, payload: Mapping[str, Any], nbytes_total: int) -> None:
    """Lesbare Konsolen-Ausgabe nach JSON-Schreiben / apply_selection."""
    chans = payload.get("channels") or {}
    print("[GKG] Exploration — Ergebnis-Zusammenfassung", flush=True)
    print(
        f"  Fenster: {payload.get('window_start')} … {payload.get('window_end')}  "
        f"universe={payload.get('universe')}",
        flush=True,
    )
    if nbytes_total > 0:
        print(f"  BigQuery gelesen (Summe über Ranking-Queries): {nbytes_total / 1e9:.4f} GB", flush=True)
    triples = getattr(cfg_mod, "GKG_THEME_SQL_TRIPLES", None) or []
    print(f"  Theme-Splits aktiv (GKG_THEME_SQL_TRIPLES): {len(triples)} (Kanal, Alias, Token)", flush=True)
    for name in sorted(chans.keys(), key=lambda x: (x != "macro", str(x))):
        block = chans[name] or {}
        toks = block.get("tokens") or []
        lifts = block.get("lifts")
        line = f"  [{name}] ausgewählt: {len(toks)} Token"
        if name != "macro" and lifts is not None:
            line += " (Lift vs. Makro-Verteilung)"
        print(line, flush=True)
        if not toks:
            continue
        preview_n = 10
        for i, t in enumerate(toks[:preview_n]):
            if lifts is not None and i < len(lifts):
                print(f"      · {t}  lift={lifts[i]}", flush=True)
            else:
                print(f"      · {t}", flush=True)
        if len(toks) > preview_n:
            print(f"      … +{len(toks) - preview_n} weitere (vollständig in JSON)", flush=True)


def _sql_escape_like(s: str) -> str:
    return s.replace("'", "''").replace("\\", "\\\\")


def theme_column_alias(raw: str) -> str:
    """Eindeutiger SQL/Pandas-Suffix pro Theme-Token (ASCII, BigQuery-Identifier)."""
    import re

    s = str(raw).strip()
    if not s or not _TOKEN_SAFE.match(s):
        return ""
    a = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    if not a or a[0].isdigit():
        a = "t_" + a
    return a[:48]


def build_gkg_theme_sql_triples(selection: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(Kanal, Alias, Roh-Token) für pro-Theme Tone/Vol in einer BQ-Query und im Feature-Cache."""
    out: list[tuple[str, str, str]] = []
    chans = selection.get("channels") or {}
    for ch_name, payload in chans.items():
        toks = (payload or {}).get("tokens") or []
        used_aliases: set[str] = set()
        for raw in toks:
            t = str(raw).strip()
            if not _TOKEN_SAFE.match(t):
                continue
            alias = theme_column_alias(t)
            if not alias:
                continue
            base = alias
            n = 2
            while alias in used_aliases:
                alias = f"{base}_{n}"
                n += 1
            used_aliases.add(alias)
            out.append((str(ch_name), alias, t))
    return out


def tokens_to_or_clause(tokens: list[str]) -> str:
    parts: list[str] = []
    for raw in tokens:
        t = str(raw).strip()
        if not t or not _TOKEN_SAFE.match(t):
            continue
        parts.append(f"V2Themes LIKE '%{_sql_escape_like(t)}%'")
    if not parts:
        return "(1=0)"
    return "(" + " OR ".join(parts) + ")"


def merge_where_with_tokens(base_sql: str, tokens: list[str]) -> str:
    safe = [str(t).strip() for t in tokens if _TOKEN_SAFE.match(str(t).strip())]
    if not safe:
        return base_sql.strip().rstrip(";")
    extra = tokens_to_or_clause(safe)
    b = base_sql.strip().rstrip(";")
    return f"(({b}) OR {extra})"


def apply_selection_to_cfg(cfg_mod: Any, selection: Mapping[str, Any]) -> None:
    """Erweitert MACRO_BQ_THEME_WHERE / SECTOR_BQ_THEME_WHERE um gespeicherte Tokens (OR)."""
    chans = selection.get("channels") or {}
    macro_toks = (chans.get("macro") or {}).get("tokens") or []
    if macro_toks:
        cfg_mod.MACRO_BQ_THEME_WHERE = merge_where_with_tokens(
            str(cfg_mod.MACRO_BQ_THEME_WHERE), list(macro_toks)
        )
    sec_sel = {k: v for k, v in chans.items() if k != "macro"}
    new_sec = dict(cfg_mod.SECTOR_BQ_THEME_WHERE)
    for sector, payload in sec_sel.items():
        if sector not in new_sec:
            continue
        toks = (payload or {}).get("tokens") or []
        if toks:
            new_sec[sector] = merge_where_with_tokens(str(new_sec[sector]), list(toks))
    cfg_mod.SECTOR_BQ_THEME_WHERE = new_sec
    cfg_mod.GKG_THEME_SQL_TRIPLES = build_gkg_theme_sql_triples(selection)


def apply_theme_selection_from_file(cfg_mod: Any, path: Path | None = None) -> bool:
    p = Path(path or getattr(cfg_mod, "GKG_THEME_SELECTION_PATH", "data/gkg_theme_selection.json"))
    if not p.is_file():
        return False
    data = json.loads(p.read_text(encoding="utf-8"))
    apply_selection_to_cfg(cfg_mod, data)
    print(f"GKG Theme-Auswahl aus {p} angewendet (OR-Erweiterung der cfg-Theme-SQL).", flush=True)
    _log_gkg_exploration_summary(cfg_mod, data, 0)
    return True


def _pick_sector_tokens(
    rows: list[tuple[str, int, float]],
    macro_share: dict[str, float],
    min_lift: float,
    max_tokens: int,
) -> tuple[list[str], list[float]]:
    out_t: list[str] = []
    out_l: list[float] = []
    scored: list[tuple[str, float]] = []
    for tok, _hits, s_share in rows:
        m_share = macro_share.get(tok, 0.0)
        lift = (float(s_share) + 1e-15) / (float(m_share) + 1e-15)
        scored.append((tok, lift))
    scored.sort(key=lambda x: -x[1])
    for tok, lift in scored:
        if lift < min_lift:
            continue
        if _TOKEN_SAFE.match(tok):
            out_t.append(tok)
            out_l.append(round(lift, 4))
        if len(out_t) >= max_tokens:
            break
    return out_t, out_l


def _pick_macro_tokens(
    rows: list[tuple[str, int, float]],
    max_tokens: int,
) -> list[str]:
    out: list[str] = []
    for tok, _hits, _sh in rows:
        if _TOKEN_SAFE.match(tok):
            out.append(tok)
        if len(out) >= max_tokens:
            break
    return out


def run_auto_explore_and_save(cfg_mod: Any) -> dict[str, Any]:
    """Führt BQ-Exploration aus, schreibt JSON, wendet Auswahl auf cfg_mod an."""
    days = int(getattr(cfg_mod, "GKG_EXPLORE_DAYS", 14))
    top_n = int(getattr(cfg_mod, "GKG_EXPLORE_TOP_N_QUERY", 200))
    max_m = int(getattr(cfg_mod, "GKG_EXPLORE_MAX_EXTRA_TOKENS_MACRO", 25))
    max_s = int(getattr(cfg_mod, "GKG_EXPLORE_MAX_EXTRA_TOKENS_SECTOR", 30))
    min_lift = float(getattr(cfg_mod, "GKG_EXPLORE_MIN_LIFT_SECTOR", 1.08))
    universe = str(getattr(cfg_mod, "GKG_EXPLORE_UNIVERSE", "filtered"))

    _ed = getattr(cfg_mod, "END_DATE", None)
    if _ed is not None:
        end = parse_day(str(_ed)[:10])
    else:
        end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    table = table_sql(cfg_mod)
    params = partition_params(start, end)

    channels = ["macro"] + list(cfg_mod.SECTOR_BQ_THEME_WHERE.keys())
    macro_sql = sql_top_tokens(
        theme_where_for_channel(cfg_mod, "macro", universe),
        _bq_geo_sql(True),
        table,
        max(top_n, 400),
    )
    macro_rows_raw, b0 = run_bq_query_with_bytes(macro_sql, params)
    nbytes_total = b0
    macro_share = {str(r["theme_token"]): float(r["token_share"] or 0.0) for r in macro_rows_raw}
    print(
        f"[GKG] Makro-Ranking: {len(macro_rows_raw)} Token-Zeilen aus BQ "
        f"({b0 / 1e9:.4f} GB).",
        flush=True,
    )

    out_channels: dict[str, Any] = {}
    for channel in channels:
        tw = theme_where_for_channel(cfg_mod, channel, universe)
        geo = _bq_geo_sql(channel == "macro")
        sql = sql_top_tokens(tw, geo, table, top_n)
        raw_rows, bi = run_bq_query_with_bytes(sql, params)
        nbytes_total += bi
        rows = [
            (str(r["theme_token"]), int(r["row_hits"]), float(r["token_share"] or 0.0))
            for r in raw_rows
        ]
        print(
            f"[GKG] Kanal {channel!r}: Top-{top_n}-Ranking, {len(raw_rows)} Zeilen "
            f"({bi / 1e9:.4f} GB).",
            flush=True,
        )
        if channel == "macro":
            toks = _pick_macro_tokens(rows, max_m)
            out_channels["macro"] = {"tokens": toks, "lifts": None}
        else:
            toks, lifts = _pick_sector_tokens(rows, macro_share, min_lift, max_s)
            out_channels[channel] = {"tokens": toks, "lifts": lifts}

    payload = {
        "version": 1,
        "created_utc_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "universe": universe,
        "channels": out_channels,
    }
    out_path = Path(getattr(cfg_mod, "GKG_THEME_SELECTION_PATH", "data/gkg_theme_selection.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"GKG_AUTO_EXPLORE_THEMES: Exploration gespeichert unter {out_path} "
        f"({start.date()} … {end.date()}, universe={universe}).",
        flush=True,
    )
    apply_selection_to_cfg(cfg_mod, payload)
    return payload


def maybe_run_gkg_theme_pipeline(cfg_mod: Any, scoring_only: bool) -> None:
    """Vor fetch_news_sentiment: automatisch explorieren und/oder gespeicherte Themes anwenden."""
    if scoring_only:
        return
    if not getattr(cfg_mod, "USE_NEWS_SENTIMENT", False):
        return
    if str(getattr(cfg_mod, "NEWS_SOURCE", "")).lower() != "bigquery":
        return
    cfg_mod.GKG_THEME_SQL_TRIPLES = []
    if getattr(cfg_mod, "GKG_AUTO_EXPLORE_THEMES", False):
        run_auto_explore_and_save(cfg_mod)
    else:
        if not apply_theme_selection_from_file(cfg_mod):
            print(
                "GKG: keine Theme-Auswahl-Datei — MACRO/SECTOR_SQL nur aus config.py "
                f"(erwartet: {getattr(cfg_mod, 'GKG_THEME_SELECTION_PATH', 'data/gkg_theme_selection.json')}).",
                flush=True,
            )
