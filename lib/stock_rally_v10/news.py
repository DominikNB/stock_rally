"""stock_rally_v10 — News/Sentiment (Pipeline-Modul)."""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib.stock_rally_v10 import config as cfg


def _c(attr: str, default=None):
    """Config lesen (Attribute heißen z. B. NEWS_BQ_START_DATE — nicht ``cfg.NEWS_...`` im __dict__)."""
    return getattr(cfg, attr, default)


def _sector_query(sector):
    kws = cfg.SECTOR_KEYWORDS.get(sector)
    if kws:
        return " OR ".join(kws[:5])
    return sector.replace("_", " ")


def _current_channels():
    chs = ["macro"] + list(cfg.TICKERS_BY_SECTOR.keys())
    trips = getattr(cfg, "GKG_THEME_SQL_TRIPLES", None) or []
    for ch, alias, _ in trips:
        chs.append(f"{ch}##{alias}")
    return chs


def _bq_escape_like(s: str) -> str:
    return str(s).replace("'", "''").replace("\\", "\\\\")


# Mindestabstand zwischen GDELT-HTTP-Calls (öffentliche API ist streng rate-limited).
_GDELT_THROTTLE_LAST = [0.0]


def _gdelt_throttle():
    import time
    v = _c("GDELT_REQUEST_DELAY_SEC")
    if v is None:
        v = _c("NEWS_GDELT_SECONDS_PER_CALL", 6.0)
    gap = max(float(v), 5.0)  # unterhalb ~5s laut GDELT praktisch immer 429
    now = time.monotonic()
    if _GDELT_THROTTLE_LAST[0] > 0.0:
        w = gap - (now - _GDELT_THROTTLE_LAST[0])
        if w > 0:
            time.sleep(w)
    _GDELT_THROTTLE_LAST[0] = time.monotonic()


def _fetch_doc_json(query, date_from, date_to, mode):
    import json
    import time
    import requests
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": 250,
        "startdatetime": date_from.strftime("%Y%m%d%H%M%S"),
        "enddatetime": date_to.strftime("%Y%m%d%H%M%S"),
    }
    headers = {
        "User-Agent": "stock_rally/1.0 (research; GDELT doc API)",
        "Accept": "application/json",
    }
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            _gdelt_throttle()
            r = requests.get(url, params=params, headers=headers, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra is not None else None
                except (TypeError, ValueError):
                    wait = None
                if wait is None or wait <= 0:
                    wait = min(180.0, 30.0 * (2**attempt))
                else:
                    wait = min(300.0, wait)
                print(
                    f"[GDELT] HTTP 429 (Rate limit) mode={mode} — warte {wait:.0f}s "
                    f"(Versuch {attempt + 1}/{max_attempts}) …",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min(90, 15 * (attempt + 1))
                print(
                    f"[GDELT] HTTP {r.status_code} mode={mode} — warte {wait}s "
                    f"(Versuch {attempt + 1}/{max_attempts}) …",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(
                    f"[GDELT] HTTP {r.status_code} mode={mode} — kein Retry, leeres Ergebnis.",
                    flush=True,
                )
                return {}
            text = (r.text or "").strip()
            if not text:
                wait = min(60, 10 * (attempt + 1))
                print(
                    f"[GDELT] Leerer HTTP-200-Body mode={mode} — warte {wait}s "
                    f"(Versuch {attempt + 1}/{max_attempts}) …",
                    flush=True,
                )
                time.sleep(wait)
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError as je:
                prev = text[:240].replace(chr(10), " ")
                print(
                    f"[GDELT] Ungültiges JSON ({mode}): {je!r} — Anfang: {prev!r} — Retry …",
                    flush=True,
                )
                time.sleep(min(90, 12 * (attempt + 1)))
                continue
        except requests.RequestException as ex:
            print(f"[GDELT] Request-Fehler ({mode}): {ex!r} — Retry …", flush=True)
            time.sleep(10 * (attempt + 1))
    print(
        f"[GDELT] Rate limit / Fehler nach {max_attempts} Versuchen (mode={mode}) — Abbruch dieses Chunks.",
        flush=True,
    )
    return None


def _parse_timeline_points(data):
    # Parse (timestamp, float) from GDELT JSON variants
    out = []
    if data is None:
        return out
    if not isinstance(data, dict):
        return out
    tl = data.get("timeline") or data.get("timelineData") or data.get("data")
    if tl is None:
        return out
    if isinstance(tl, dict):
        tl = tl.get("data") or tl.get("series") or list(tl.values())
    if not isinstance(tl, list):
        return out
    for item in tl:
        if not isinstance(item, dict):
            continue
        ds = item.get("date") or item.get("datetime") or item.get("time")
        if ds is None:
            continue
        val = item.get("value")
        if val is None:
            for k, v in item.items():
                if k in ("date", "datetime", "time"):
                    continue
                if isinstance(v, (int, float)):
                    val = float(v)
                    break
        if val is None:
            continue
        try:
            d = pd.to_datetime(str(ds)[:8], format="%Y%m%d", errors="coerce")
        except Exception:
            continue
        if pd.isna(d):
            continue
        out.append((d.normalize(), float(val)))
    return out


def _merge_points(points):
    # Mean per calendar day
    if not points:
        return pd.Series(dtype=float)
    df = pd.DataFrame(points, columns=["d", "v"])
    return df.groupby("d")["v"].mean()


def _tv_series_from_points(tone_pts, vol_pts, start_ts, end_last):
    tone_s = _merge_points(tone_pts)
    vol_s = _merge_points(vol_pts)
    bdays = pd.bdate_range(start_ts, end_last)
    tone_s = tone_s.reindex(bdays).fillna(0.0)
    vol_s = vol_s.reindex(bdays).fillna(0.0)
    if len(tone_s) and tone_s.abs().median() > 1.5:
        tone_s = tone_s / 100.0
    return tone_s, vol_s


def _df_channel_from_tv(tone_s, vol_s, channel, impact_s=None):
    if impact_s is None:
        impact_s = pd.Series(0.0, index=tone_s.index)
    else:
        impact_s = impact_s.reindex(tone_s.index).fillna(0.0)
    return pd.DataFrame(
        {
            "Date": tone_s.index,
            "channel": channel,
            "tone": tone_s.values.astype(float),
            "vol": vol_s.values.astype(float),
            "impact": impact_s.values.astype(float),
        }
    )


def _iter_90d_chunks(start_inclusive, end_inclusive):
    """API-Zeit-Chunks (Länge cfg.GDELT_CHUNK_DAYS); end = letzter Kalendertag inkl."""
    start_ts = pd.Timestamp(start_inclusive).normalize()
    end_last = pd.Timestamp(end_inclusive).normalize()
    end_excl = end_last + pd.Timedelta(days=1)
    dt = start_ts
    _span = pd.Timedelta(days=cfg.GDELT_CHUNK_DAYS)
    while dt < end_excl:
        chunk_end = min(dt + _span, end_excl)
        yield (dt, chunk_end)
        dt = chunk_end + pd.Timedelta(days=1)


def _count_http_calls_for_span(start_inclusive, end_inclusive):
    return len(list(_iter_90d_chunks(start_inclusive, end_inclusive))) * 2


def _n_90d_chunks_in_span(start_inclusive, end_inclusive):
    return len(list(_iter_90d_chunks(start_inclusive, end_inclusive)))


def _normalize_news_cache_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "channel", "tone", "vol", "impact"])
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out["channel"] = out["channel"].astype(str)
    if "impact" not in out.columns:
        out["impact"] = 0.0
    return out


def _load_news_cache(path):
    import pickle
    empty = pd.DataFrame(columns=["Date", "channel", "tone", "vol", "impact"])
    if not os.path.isfile(path):
        return empty, {}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "df" in obj:
            return _normalize_news_cache_df(obj["df"]), dict(obj.get("meta") or {})
        if isinstance(obj, pd.DataFrame):
            return _normalize_news_cache_df(obj), {}
    except Exception as e:
        print(f"News cache read failed ({e}), starting empty.")
    return empty, {}


def _save_news_cache(df, path, meta=None):
    import pickle
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = {"df": df, "meta": meta or {}}
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _merge_news_cache_rows(old_df, new_df):
    if new_df is None or new_df.empty:
        return old_df if old_df is not None else pd.DataFrame()
    if old_df is None or old_df.empty:
        return new_df
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.normalize()
    combined = combined.drop_duplicates(subset=["Date", "channel"], keep="last")
    return combined.sort_values(["channel", "Date"]).reset_index(drop=True)


def _cluster_sorted_dates(sorted_dates):
    if not sorted_dates:
        return []
    ranges = []
    a = b = sorted_dates[0]
    for d in sorted_dates[1:]:
        if (d - b).days <= 3:
            b = d
        else:
            ranges.append((a, b))
            a = b = d
    ranges.append((a, b))
    return ranges


def _missing_bday_ranges_for_channel(cache_df, channel, bdays_index):
    need = set(pd.to_datetime(bdays_index).normalize())
    have = set()
    if cache_df is not None and not cache_df.empty:
        sub = cache_df[cache_df["channel"] == channel]
        if len(sub):
            have = set(pd.to_datetime(sub["Date"]).dt.normalize())
    missing = sorted(need - have)
    return _cluster_sorted_dates(missing)


def _fetch_channel_tone_vol(query, start, end, log_ctx=None, persist_chunk=None):
    # Returns (tone Series, vol Series), index = business days in [start, end] inclusive
    # persist_chunk: optional callback(tone_pts, vol_pts) — z. B. Pickle nach jedem Sub-Chunk
    start_ts = pd.Timestamp(start).normalize()
    end_last = pd.Timestamp(end).normalize()
    tone_pts, vol_pts = [], []
    chunks = list(_iter_90d_chunks(start_ts, end_last))
    n_sub = len(chunks)
    for ci, (cf, ct) in enumerate(chunks, start=1):
        g = tot = ch = gap_s = None
        if log_ctx is not None:
            log_ctx["counter"][0] += 1
            g = log_ctx["counter"][0]
            tot = log_ctx["total"]
            ch = log_ctx.get("channel", "?")
            gi, gn = log_ctx.get("gap_i", 0), log_ctx.get("gap_n", 0)
            gap_s = f" Lücke {gi}/{gn}" if gn else ""
            print(
                f"[GDELT] ({g}/{tot}) Kanal={ch}{gap_s} | Sub-Chunk {ci}/{n_sub} | "
                f"{cf.strftime('%Y-%m-%d')} … {ct.strftime('%Y-%m-%d')} | timelinetone + vol …",
                flush=True,
            )
        jt = _fetch_doc_json(query, cf, ct, "timelinetone")
        if jt is None:
            if tone_pts:
                print(
                    "[GDELT] Rate limit / Abbruch bei timelinetone — Zwischenstand wurde nach dem "
                    "letzten erfolgreichen Sub-Chunk gespeichert. Nächster Run füllt die Lücke.",
                    flush=True,
                )
            else:
                print(
                    "[GDELT] Rate limit / Abbruch bei timelinetone — noch keine Daten für diese "
                    "Lücke (Pickle bleibt für diesen Kanal/Bereich leer). Später erneut versuchen.",
                    flush=True,
                )
            break
        jv = _fetch_doc_json(query, cf, ct, "timelinevolraw")
        if jv is None:
            if tone_pts:
                print(
                    "[GDELT] Rate limit / Abbruch bei timelinevolraw — Zwischenstand nach letztem OK-Chunk gespeichert.",
                    flush=True,
                )
            else:
                print(
                    "[GDELT] Rate limit / Abbruch bei timelinevolraw — noch keine Daten gespeichert.",
                    flush=True,
                )
            break
        if not jv.get("timeline") and not jv.get("timelineData"):
            jv = _fetch_doc_json(query, cf, ct, "timelinevol")
            if jv is None:
                print(
                    "[GDELT] Abbruch bei timelinevol-Fallback (Rate limit). "
                    + (
                        "Zwischenstand gespeichert."
                        if tone_pts
                        else "Noch keine Daten gespeichert."
                    ),
                    flush=True,
                )
                break
            if log_ctx is not None:
                print(
                    f"[GDELT] ({g}/{tot}) Kanal={ch}{gap_s} | Sub-Chunk {ci}/{n_sub} | Fallback: timelinevol",
                    flush=True,
                )
        pt, pv = _parse_timeline_points(jt), _parse_timeline_points(jv)
        nt, nv = len(pt), len(pv)
        tone_pts.extend(pt)
        vol_pts.extend(pv)
        if log_ctx is not None:
            print(
                f"[GDELT] ({g}/{tot}) Kanal={ch}{gap_s} | Sub-Chunk {ci}/{n_sub} | OK "
                f"(Punkte tone={nt}, vol={nv}).",
                flush=True,
            )
        if persist_chunk is not None:
            persist_chunk(tone_pts, vol_pts)
    tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, start_ts, end_last)
    return tone_s, vol_s


def _bq_ymd_int(ts):
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


def _bq_ts_utc_midnight(ts):
    from datetime import datetime, timezone
    t = pd.Timestamp(ts).normalize()
    return datetime(t.year, t.month, t.day, tzinfo=timezone.utc)


def _bq_resolve_project():
    """GCP-Projekt für BigQuery-Client (Billing). Reihenfolge: Config → Env → ADC → gcloud."""
    p = _c("BQ_PROJECT_ID")
    if p:
        return str(p).strip() or None
    import os
    p = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or os.environ.get("GCP_PROJECT_ID")
    if p:
        return p.strip()
    try:
        import google.auth
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None


_EU_EXCH_SUFFIXES = frozenset(
    {
        "DE",
        "PA",
        "AS",
        "MC",
        "BR",
        "ST",
        "L",
        "MI",
        "SW",
        "LS",
        "VI",
        "OL",
        "HE",
        "CO",
        "IC",
        "WA",
        "IR",
        "AT",
        "AX",
    }
)


def _news_cfg(cmod=None):
    return cmod if cmod is not None else cfg


def _ticker_geo_bucket(ticker: str, cmod=None) -> str | None:
    """Grobe EU/US-Zuordnung für Mehrheits-Geo; None = global/unbekannt."""
    cm = _news_cfg(cmod)
    t = str(ticker).strip()
    ov = getattr(cm, "TICKER_BQ_GEO_OVERRIDES", None) or {}
    if t in ov:
        v = str(ov[t]).strip().upper()
        return v if v in ("US", "EU") else None
    u = t.upper()
    if "-" in u and u.endswith("USD"):
        return None
    if "." in t:
        suf = t.rsplit(".", 1)[-1].upper()
        if suf in _EU_EXCH_SUFFIXES:
            return "EU"
        return None
    if t.replace("-", "").isalnum() and 1 <= len(t) <= 8:
        return "US"
    return None


def _sector_geo_country_codes(sector: str | None, cmod=None) -> list[str] | None:
    """Codes für ActionGeo_CountryCode oder None = keinen Geo-Filter."""
    if not sector:
        return None
    cm = _news_cfg(cmod)
    if sector not in getattr(cm, "TICKERS_BY_SECTOR", {}):
        return None
    if not getattr(cm, "BQ_SECTOR_USE_GEO_FILTER", True):
        return None
    if not getattr(cm, "BQ_SECTOR_GEO_AUTO", True):
        mc = getattr(cm, "BQ_SECTOR_GEO_COUNTRIES", None)
        if not mc:
            return None
        return [mc] if isinstance(mc, str) else list(mc)
    tickers = cm.TICKERS_BY_SECTOR[sector]
    if not tickers:
        return None
    thr = float(getattr(cm, "BQ_SECTOR_GEO_MAJORITY_THRESHOLD", 0.5))
    n = len(tickers)
    n_us = sum(1 for tk in tickers if _ticker_geo_bucket(tk, cm) == "US")
    n_eu = sum(1 for tk in tickers if _ticker_geo_bucket(tk, cm) == "EU")
    if n_us / n >= thr - 1e-15:
        return ["US"]
    if n_eu / n >= thr - 1e-15:
        return ["EU"]
    return None


def _bq_geo_sql(is_macro, sector=None, cmod=None):
    """Länderfilter (nur Events): Makro aus cfg; Sektor nach Mehrheit (AUTO) oder Legacy-Liste."""
    cm = _news_cfg(cmod)
    if is_macro:
        mc = getattr(cm, "BQ_MACRO_GEO_COUNTRIES", None)
    else:
        if not getattr(cm, "BQ_SECTOR_USE_GEO_FILTER", True):
            return ""
        if getattr(cm, "BQ_SECTOR_GEO_AUTO", True):
            mc = _sector_geo_country_codes(sector, cm)
        else:
            mc = getattr(cm, "BQ_SECTOR_GEO_COUNTRIES", None)
            if isinstance(mc, str):
                mc = [mc]
    if not mc:
        return ""
    if isinstance(mc, str):
        return f" AND ActionGeo_CountryCode = '{mc}'"
    return " AND (" + " OR ".join(f"ActionGeo_CountryCode = '{c}'" for c in mc) + ")"


def _gap_union_span(fetch_plan, channel_list):
    """Min/Max über alle Kanal-Lücken — ein Zeitraum für Single-Scan."""
    intervals = []
    for ch in channel_list:
        for a, b in fetch_plan.get(ch, []):
            intervals.append((pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()))
    if not intervals:
        return None, None
    return min(t[0] for t in intervals), max(t[1] for t in intervals)


def _bq_gkg_theme_triple_cond(channel: str, raw_token: str) -> str:
    """Basis-Sektor/Makro-SQL UND ein konkretes GKG-Theme-Token (pro ``Kanal##alias``-Zeitreihe)."""
    tok = str(raw_token).strip()
    like = f"V2Themes LIKE '%{_bq_escape_like(tok)}%'"
    if channel == "macro":
        base = str(_c("MACRO_BQ_THEME_WHERE", "(1=1)"))
    else:
        base = str(cfg.SECTOR_BQ_THEME_WHERE.get(channel) or "(1=0)")
    return f"(({base}) AND ({like}))"


def _bq_run_gkg_daily_agg(theme_sql: str, a, b):
    """Eine GKG-Partition-Query: Tagesmittel Tone + Artikelzahl für gegebenes Theme-SQL."""
    from google.cloud import bigquery

    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    params = []
    partition_sql = ""
    if _c("BQ_USE_PARTITION_FILTER", True):
        partition_sql = (
            " AND _PARTITIONTIME >= @ts_start "
            "AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)"
        )
        params.extend(
            [
                bigquery.ScalarQueryParameter("ts_start", "TIMESTAMP", _bq_ts_utc_midnight(a)),
                bigquery.ScalarQueryParameter("ts_end", "TIMESTAMP", _bq_ts_utc_midnight(b)),
            ]
        )
    sql = f"""
SELECT
  DATE(_PARTITIONTIME) AS d,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS avg_daily_tone,
  COUNT(*) AS article_count,
  CAST(NULL AS FLOAT64) AS avg_goldstein
FROM {table}
WHERE 1=1
  {partition_sql}
  AND ({theme_sql})
  AND V2Themes IS NOT NULL
  AND V2Tone IS NOT NULL
GROUP BY d
ORDER BY d
"""
    proj = _bq_resolve_project()
    if not proj:
        raise OSError(
            "BigQuery: Kein GCP-Projekt. Setze cfg.BQ_PROJECT_ID oder GOOGLE_CLOUD_PROJECT."
        )
    client = bigquery.Client(project=proj)
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    job = client.query(sql, job_config=job_config)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    nbytes = job.total_bytes_processed or 0
    return df, nbytes


def _bq_query_theme_triple_date_range(channel: str, raw_token: str, a, b):
    """Wie Kanal-Query, aber nur Zeilen mit zusätzlichem Theme-Token (Cache-Kanal ``ch##alias``)."""
    if not _c("BQ_USE_GKG_TABLE", True):
        raise ValueError("Theme-Tripel erfordern GKG (cfg.BQ_USE_GKG_TABLE=True).")
    theme_sql = _bq_gkg_theme_triple_cond(channel, raw_token)
    return _bq_run_gkg_daily_agg(theme_sql, a, b)


def _bq_query_all_channels_gkg(a, b):
    """Eine Query, ein Scan: OR über alle Theme-Filter, IF/COUNTIF pro Kanal."""
    from google.cloud import bigquery
    if not _c("BQ_USE_PARTITION_FILTER", True):
        print(
            "[BigQuery] WARNUNG: cfg.BQ_USE_PARTITION_FILTER=False kann Full-Table-Scan bedeuten.",
            flush=True,
        )
    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    channels = ["macro"] + list(cfg.TICKERS_BY_SECTOR.keys())
    tone_part = "SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)"
    or_parts = [f"({cfg.MACRO_BQ_THEME_WHERE})"]
    for s in cfg.TICKERS_BY_SECTOR:
        or_parts.append(f"({cfg.SECTOR_BQ_THEME_WHERE[s]})")
    combined_or = "(" + " OR ".join(or_parts) + ")"
    sel_parts = ["DATE(_PARTITIONTIME) AS d"]
    for ch in channels:
        cond = cfg.MACRO_BQ_THEME_WHERE if ch == "macro" else cfg.SECTOR_BQ_THEME_WHERE[ch]
        sel_parts.append(f"AVG(IF( ({cond}), {tone_part}, NULL)) AS {ch}_tone")
        sel_parts.append(f"COUNTIF( ({cond}) ) AS {ch}_vol")
    trips = getattr(cfg, "GKG_THEME_SQL_TRIPLES", None) or []
    for tch, alias, raw_tok in trips:
        if tch != "macro" and tch not in cfg.SECTOR_BQ_THEME_WHERE:
            continue
        trip_cond = _bq_gkg_theme_triple_cond(tch, raw_tok)
        col_prefix = f"{tch}__t__{alias}"
        sel_parts.append(f"AVG(IF( ({trip_cond}), {tone_part}, NULL)) AS {col_prefix}_tone")
        sel_parts.append(f"COUNTIF( ({trip_cond}) ) AS {col_prefix}_vol")
    select_sql = ",\n  ".join(sel_parts)
    params = [
        bigquery.ScalarQueryParameter("ts_start", "TIMESTAMP", _bq_ts_utc_midnight(a)),
        bigquery.ScalarQueryParameter("ts_end", "TIMESTAMP", _bq_ts_utc_midnight(b)),
    ]
    partition_where = (
        "_PARTITIONTIME >= @ts_start "
        "AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)"
    )
    sql = f"""
SELECT
  {select_sql}
FROM {table}
WHERE {partition_where}
  AND {combined_or}
  AND V2Themes IS NOT NULL
  AND V2Tone IS NOT NULL
GROUP BY d
ORDER BY d
"""
    proj = _bq_resolve_project()
    if not proj:
        raise OSError("BigQuery: Kein GCP-Projekt (cfg.BQ_PROJECT_ID / GOOGLE_CLOUD_PROJECT).")
    client = bigquery.Client(project=proj)
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    job = client.query(sql, job_config=job_config)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    nbytes = job.total_bytes_processed or 0
    return df, nbytes


def _iter_inclusive_date_chunks(a, b, max_days: int):
    """Disjunkte [start, end]-Intervalle (je end inklusiv), max. max_days Kalendertage pro Stück."""
    a = pd.Timestamp(a).normalize()
    b = pd.Timestamp(b).normalize()
    if a > b:
        return
    if max_days <= 0:
        yield a, b
        return
    cur = a
    while cur <= b:
        chunk_end = min(cur + pd.Timedelta(days=max_days - 1), b)
        yield cur, chunk_end
        cur = chunk_end + pd.Timedelta(days=1)


def _bq_query_all_channels_gkg_chunked(a, b):
    """Wie `_bq_query_all_channels_gkg`, aber in Zeitscheiben — vermeidet „Query too big“ bei langen Spannen."""
    chunk_days = int(_c("BQ_SINGLE_SCAN_CHUNK_DAYS", 0) or 0)
    chunks = list(_iter_inclusive_date_chunks(a, b, chunk_days))
    if not chunks:
        return None, 0
    if len(chunks) == 1:
        return _bq_query_all_channels_gkg(chunks[0][0], chunks[0][1])
    print(
        f"[BigQuery] Single-Scan in {len(chunks)} Teilfenstern "
        f"(≤{chunk_days} Kalendertage pro Query) …",
        flush=True,
    )
    parts = []
    total_b = 0
    n_chunks = len(chunks)
    for i, (ca, cb) in enumerate(chunks, start=1):
        print(
            f"[BigQuery]   Teil {i}/{n_chunks}: {ca.date()} … {cb.date()} — läuft …",
            flush=True,
        )
        df, nb = _bq_query_all_channels_gkg(ca, cb)
        total_b += nb
        nrows = len(df) if df is not None else 0
        gb = nb / 1e9 if nb else 0.0
        print(
            f"[BigQuery]   Teil {i}/{n_chunks} fertig — {nrows} Tageszeilen, "
            f"{gb:.3f} GB gescannt (Teillauf).",
            flush=True,
        )
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        print(
            f"[BigQuery] Alle {n_chunks} Teilfenster fertig — keine Datenzeilen.",
            flush=True,
        )
        return None, total_b
    out = pd.concat(parts, ignore_index=True)
    if "d" in out.columns:
        out = out.drop_duplicates(subset=["d"], keep="last").sort_values("d")
    print(
        f"[BigQuery] Alle {n_chunks} Teilfenster fertig — {len(out)} eindeutige Tage im Merge, "
        f"Summe gescannt: {total_b / 1e9:.3f} GB.",
        flush=True,
    )
    return out, total_b


def _bq_query_channel_date_range(channel, a, b):
    """GKG: V2Themes + V2Tone (kommasepariert). events-Tabellen haben keine Themes."""
    from google.cloud import bigquery
    use_gkg = _c("BQ_USE_GKG_TABLE", True)
    is_macro = channel == "macro"
    if is_macro:
        theme_sql = _c("MACRO_BQ_THEME_WHERE", "(1=1)")
    else:
        theme_sql = cfg.SECTOR_BQ_THEME_WHERE[channel]
    geo_sql = "" if use_gkg else _bq_geo_sql(is_macro, None if is_macro else channel)
    if use_gkg:
        return _bq_run_gkg_daily_agg(theme_sql, a, b)
    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    d0, d1 = _bq_ymd_int(a), _bq_ymd_int(b)
    tc = _c("BQ_THEMES_COLUMN", "V2Themes")
    theme_sql = theme_sql.replace("V2Themes", tc)
    dname = str(_c("BQ_EVENT_DATE_FIELD", "SQLDATE"))
    dref = "`DATE`" if dname.upper() == "DATE" else dname
    params = [
        bigquery.ScalarQueryParameter("d0", "INT64", d0),
        bigquery.ScalarQueryParameter("d1", "INT64", d1),
    ]
    partition_sql = ""
    if _c("BQ_USE_PARTITION_FILTER", True):
        partition_sql = (
            " AND _PARTITIONTIME >= @ts_start "
            "AND _PARTITIONTIME < TIMESTAMP_ADD(@ts_end, INTERVAL 1 DAY)"
        )
        params.extend(
            [
                bigquery.ScalarQueryParameter("ts_start", "TIMESTAMP", _bq_ts_utc_midnight(a)),
                bigquery.ScalarQueryParameter("ts_end", "TIMESTAMP", _bq_ts_utc_midnight(b)),
            ]
        )
    sql = f"""
SELECT
  PARSE_DATE('%Y%m%d', CAST({dref} AS STRING)) AS d,
  AVG(AvgTone) AS avg_daily_tone,
  COUNT(*) AS article_count,
  AVG(GoldsteinScale) AS avg_goldstein
FROM {table}
WHERE {dref} >= @d0 AND {dref} <= @d1
  AND ({theme_sql})
  AND {tc} IS NOT NULL
  {geo_sql}
  {partition_sql}
GROUP BY d
ORDER BY d
"""
    proj = _bq_resolve_project()
    if not proj:
        raise OSError(
            "BigQuery: Kein GCP-Projekt. Setze in Cell 2 z. B. cfg.BQ_PROJECT_ID = 'dein-projekt-id' "
            "oder im Terminal: gcloud config set project dein-projekt-id  "
            "und $env:GOOGLE_CLOUD_PROJECT = 'dein-projekt-id' (PowerShell)."
        )
    client = bigquery.Client(project=proj)
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    job = client.query(sql, job_config=job_config)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    nbytes = job.total_bytes_processed or 0
    return df, nbytes


def _fetch_news_sentiment_bigquery(df, start, end):
    import time as _time
    # Gleicher Pickle-Cache wie GDELT-API. GKG+cfg.BQ_SINGLE_SCAN: 1 Query (IF/COUNTIF), sonst 1 Query pro (Kanal, Lücke).
    cache_path = _c("NEWS_CACHE_FILE", os.path.join(os.getcwd(), "data", "news_gdelt_cache.pkl"))
    channels_base = ["macro"] + list(cfg.TICKERS_BY_SECTOR.keys())
    channels = _current_channels()
    start_t, end_t = pd.Timestamp(start), pd.Timestamp(end)
    bdays = pd.bdate_range(start_t.normalize(), end_t.normalize())
    cache_df, meta = _load_news_cache(cache_path)
    meta = dict(meta or {})
    n_rows_before = len(cache_df)
    cur_ch = _current_channels()
    if meta.get("channels"):
        new_ch = set(cur_ch) - set(meta["channels"])
        if new_ch:
            print(
                f"[BigQuery] Neue News-Kanäle (z. B. Theme-Splits nach Exploration): {sorted(new_ch)} — "
                "fehlende (Datum,Kanal)-Paare werden nachgeladen (Lücken-Vereinigung / ggf. großer Scan).",
                flush=True,
            )
    if not cache_df.empty:
        d0, d1 = cache_df["Date"].min(), cache_df["Date"].max()
        print(f"News cache: {cache_path}  rows={len(cache_df)}  Date span: {d0} … {d1}")
    else:
        print(f"News cache: (empty) {cache_path}")
    fetch_plan = {ch: _missing_bday_ranges_for_channel(cache_df, ch, bdays) for ch in channels}
    n_ranges = sum(len(fetch_plan[ch]) for ch in channels)
    use_gkg = _c("BQ_USE_GKG_TABLE", True)
    single_scan = bool(_c("BQ_SINGLE_SCAN", True)) and use_gkg
    if n_ranges == 0:
        pass
    elif single_scan:
        _cd = int(_c("BQ_SINGLE_SCAN_CHUNK_DAYS", 0) or 0)
        _sq = (
            "1 Query"
            if _cd <= 0
            else f"1+ Queries (Zeitscheiben ≤{_cd} Tage)"
        )
        print(
            f"[BigQuery] {n_ranges} Lücken-Bereiche über {len(channels)} Kanäle "
            f"→ {_sq} (Single-Scan, _PARTITIONTIME). Tabelle: {_c('GDELT_BQ_EVENTS_TABLE', '')}",
            flush=True,
        )
    else:
        print(
            f"[BigQuery] {n_ranges} Lücken-Bereiche über {len(channels)} Kanäle "
            f"(≈ {n_ranges} Queries; kein HTTP-429). Tabelle: {_c('GDELT_BQ_EVENTS_TABLE', '')}",
            flush=True,
        )
    print(
        "[BigQuery] Hinweis: „gescannt GB“ = von BigQuery gelesenes Volumen (Abrechnung), "
        "nicht Download auf deinen PC. Ergebnis sind nur kleine aggregierte Tabellen.",
        flush=True,
    )
    t0 = _time.perf_counter()
    n_ch = len(channels)
    total_bytes = 0
    if n_ranges == 0:
        print("[BigQuery] Cache vollständig — keine Abfragen.", flush=True)
    elif single_scan:
        a_u, b_u = _gap_union_span(fetch_plan, channels)
        if a_u is None or b_u is None:
            print("[BigQuery] Single-Scan: kein gültiger Zeitraum — übersprungen.", flush=True)
        else:
            print(
                f"[BigQuery] Single-Scan Zeitraum: {a_u.date()} … {b_u.date()} "
                "(Vereinigung aller Kanal-Lücken).",
                flush=True,
            )
            try:
                df_wide, nbytes = _bq_query_all_channels_gkg_chunked(a_u, b_u)
            except Exception as ex:
                print(
                    f"[BigQuery] Fehler Single-Scan: {ex!r} — "
                    "accessDenied: Projekt `gdelt-bq`. "
                    "„Query too big“ / Resources: cfg.BQ_SINGLE_SCAN_CHUNK_DAYS verkleinern (z. B. 60 oder 30) "
                    "oder cfg.BQ_SINGLE_SCAN=False.",
                    flush=True,
                )
                df_wide, nbytes = None, 0
            total_bytes += nbytes
            if nbytes:
                print(
                    f"  … in BigQuery gescannt (fakturierbar): {nbytes / 1e9:.3f} GB — "
                    "ein Scan für alle Kanäle.",
                    flush=True,
                )
            if df_wide is not None and len(df_wide):
                st = pd.Timestamp(a_u).normalize()
                el = pd.Timestamp(b_u).normalize()
                for ch in channels_base:
                    tc, vc = f"{ch}_tone", f"{ch}_vol"
                    if tc not in df_wide.columns:
                        continue
                    tone_pts, vol_pts = [], []
                    for _, row in df_wide.iterrows():
                        dd = pd.Timestamp(row["d"]).normalize()
                        tv = row[tc]
                        vv = row[vc]
                        tone_pts.append((dd, float(tv) if pd.notna(tv) else 0.0))
                        vol_pts.append((dd, float(vv) if pd.notna(vv) else 0.0))
                    tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, st, el)
                    imp_s = pd.Series(0.0, index=tone_s.index)
                    mini = _df_channel_from_tv(tone_s, vol_s, ch, imp_s)
                    cache_df = _merge_news_cache_rows(cache_df, mini)
                trips = getattr(cfg, "GKG_THEME_SQL_TRIPLES", None) or []
                for tch, alias, _ in trips:
                    if tch not in channels_base:
                        continue
                    bqp = f"{tch}__t__{alias}"
                    tc, vc = f"{bqp}_tone", f"{bqp}_vol"
                    if tc not in df_wide.columns:
                        continue
                    cache_ch = f"{tch}##{alias}"
                    tone_pts, vol_pts = [], []
                    for _, row in df_wide.iterrows():
                        dd = pd.Timestamp(row["d"]).normalize()
                        tv = row[tc]
                        vv = row[vc]
                        tone_pts.append((dd, float(tv) if pd.notna(tv) else 0.0))
                        vol_pts.append((dd, float(vv) if pd.notna(vv) else 0.0))
                    tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, st, el)
                    imp_s = pd.Series(0.0, index=tone_s.index)
                    mini = _df_channel_from_tv(tone_s, vol_s, cache_ch, imp_s)
                    cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(cache_df, cache_path, meta)
                print(
                    f"[BigQuery] Zwischenspeicher: {len(cache_df)} Zeilen (Kanal=alle, Single-Scan).",
                    flush=True,
                )
    else:
        n_ch_base = len(channels_base)
        for ch_i, ch in enumerate(channels_base, start=1):
            ranges = fetch_plan[ch]
            if not ranges:
                print(f"[BigQuery] Kanal {ch_i}/{n_ch_base} {ch!r}: übersprungen (Cache).", flush=True)
                continue
            gn = len(ranges)
            print(f"[BigQuery] Kanal {ch_i}/{n_ch_base} {ch!r}: {gn} Lücken → Abfragen …", flush=True)
            for gi, (a, b) in enumerate(ranges, start=1):
                print(
                    f"[BigQuery]   Lücke {gi}/{gn}: {a.date()} … {b.date()}", flush=True
                )
                try:
                    df_bq, nbytes = _bq_query_channel_date_range(ch, a, b)
                except Exception as ex:
                    print(
                        f"[BigQuery] Fehler Kanal={ch!r}: {ex!r} — "
                        "Bei accessDenied: öffentliche Tabelle `gdelt-bq.gdeltv2.events` nutzen (nicht gdelt-vq). "
                        "Sonst Partition/cfg.BQ_EVENT_DATE_FIELD prüfen oder cfg.BQ_USE_PARTITION_FILTER=False.",
                        flush=True,
                    )
                    continue
                total_bytes += nbytes
                if nbytes:
                    print(
                        f"  … in BigQuery gescannt (fakturierbar): {nbytes / 1e9:.3f} GB — "
                        "kein Datei-Download; nur aggregierte Tageszeilen nach Python.",
                        flush=True,
                    )
                st = pd.Timestamp(a).normalize()
                el = pd.Timestamp(b).normalize()
                tone_pts, vol_pts, impact_pts = [], [], []
                if df_bq is not None and len(df_bq):
                    for _, row in df_bq.iterrows():
                        dd = pd.Timestamp(row["d"]).normalize()
                        tone_pts.append((dd, float(row["avg_daily_tone"])))
                        vol_pts.append((dd, float(row["article_count"])))
                        gs = row["avg_goldstein"]
                        impact_pts.append((dd, float(gs)) if pd.notna(gs) else (dd, 0.0))
                tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, st, el)
                imp_s = _merge_points(impact_pts).reindex(tone_s.index).fillna(0.0)
                mini = _df_channel_from_tv(tone_s, vol_s, ch, imp_s)
                cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(cache_df, cache_path, meta)
                print(
                    f"[BigQuery] Zwischenspeicher: {len(cache_df)} Zeilen (Kanal={ch!r}).",
                    flush=True,
                )
        trips_mq = getattr(cfg, "GKG_THEME_SQL_TRIPLES", None) or []
        n_trip_mq = len(trips_mq)
        for ti, (tch, alias, raw_tok) in enumerate(trips_mq, start=1):
            if tch != "macro" and tch not in cfg.SECTOR_BQ_THEME_WHERE:
                continue
            cache_ch = f"{tch}##{alias}"
            ranges_t = fetch_plan.get(cache_ch, [])
            if not ranges_t:
                continue
            if not use_gkg:
                print(f"[BigQuery] Theme {cache_ch!r}: übersprungen (nur mit GKG).", flush=True)
                continue
            gn = len(ranges_t)
            print(
                f"[BigQuery] Theme-Split {ti}/{n_trip_mq} {cache_ch!r}: {gn} Lücken → Abfragen …",
                flush=True,
            )
            for gi, (a, b) in enumerate(ranges_t, start=1):
                print(
                    f"[BigQuery]   Theme-Lücke {gi}/{gn}: {a.date()} … {b.date()}",
                    flush=True,
                )
                try:
                    df_bq, nbytes = _bq_query_theme_triple_date_range(tch, raw_tok, a, b)
                except Exception as ex:
                    print(
                        f"[BigQuery] Fehler Theme-Kanal={cache_ch!r}: {ex!r}",
                        flush=True,
                    )
                    continue
                total_bytes += nbytes
                if nbytes:
                    print(
                        f"  … in BigQuery gescannt (fakturierbar): {nbytes / 1e9:.3f} GB — "
                        "Theme-Zeitreihe.",
                        flush=True,
                    )
                st = pd.Timestamp(a).normalize()
                el = pd.Timestamp(b).normalize()
                tone_pts, vol_pts, impact_pts = [], [], []
                if df_bq is not None and len(df_bq):
                    for _, row in df_bq.iterrows():
                        dd = pd.Timestamp(row["d"]).normalize()
                        tone_pts.append((dd, float(row["avg_daily_tone"])))
                        vol_pts.append((dd, float(row["article_count"])))
                        gs = row["avg_goldstein"]
                        impact_pts.append((dd, float(gs)) if pd.notna(gs) else (dd, 0.0))
                tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, st, el)
                imp_s = _merge_points(impact_pts).reindex(tone_s.index).fillna(0.0)
                mini = _df_channel_from_tv(tone_s, vol_s, cache_ch, imp_s)
                cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(cache_df, cache_path, meta)
                print(
                    f"[BigQuery] Zwischenspeicher: {len(cache_df)} Zeilen (Kanal={cache_ch!r}).",
                    flush=True,
                )
    meta["channels"] = cur_ch
    meta["tickers"] = sorted(cfg.ALL_TICKERS)
    meta["last_run_end_date"] = str(end_t.date())
    meta["saved_at"] = pd.Timestamp.now().isoformat()
    meta["source"] = "bigquery"
    _save_news_cache(cache_df, cache_path, meta)
    if total_bytes:
        print(f"[BigQuery] Summe gescannt (letzter Lauf): {total_bytes / 1e9:.3f} GB", flush=True)
    if len(cache_df) > n_rows_before:
        print(
            f"Cache: +{len(cache_df) - n_rows_before} Zeilen → {cache_path} ({len(cache_df)} rows)."
        )
    print(f"[BigQuery] Laufzeit: {(_time.perf_counter() - t0) / 60:.2f} min")
    out = cache_df[
        (cache_df["Date"] >= start_t.normalize()) & (cache_df["Date"] <= end_t.normalize())
    ]
    out = out[out["channel"].isin(channels)].copy()
    need_keys = {(d, ch) for d in bdays for ch in channels}
    have_keys = set(zip(pd.to_datetime(out["Date"]).dt.normalize(), out["channel"]))
    still_missing = len(need_keys - have_keys)
    if still_missing:
        print(f"Warnung: {still_missing} (Datum,Kanal)-Paare fehlen noch (z. B. Query-Fehler).")
    print(f"News timelines für Pipeline: {len(out)} rows, channels={len(channels)}")
    return out


def _fetch_news_sentiment_gdelt_api(df, start=cfg.START_DATE, end=cfg.END_DATE, max_threads=8):
    # GDELT timelines + Pickle-Cache: fehlende (Kanal, Datum)-Kombinationen nachladen
    import time as _time
    cache_path = _c("NEWS_CACHE_FILE", os.path.join(os.getcwd(), "data", "news_gdelt_cache.pkl"))
    sec_per_call = float(_c("NEWS_GDELT_SECONDS_PER_CALL", 6.0))

    channels = ["macro"] + list(cfg.TICKERS_BY_SECTOR.keys())
    start_t, end_t = pd.Timestamp(start), pd.Timestamp(end)
    bdays = pd.bdate_range(start_t.normalize(), end_t.normalize())

    cache_df, meta = _load_news_cache(cache_path)
    meta = dict(meta or {})
    n_rows_before = len(cache_df)
    cur_ch = _current_channels()
    if meta.get("channels"):
        new_ch = set(cur_ch) - set(meta["channels"])
        if new_ch:
            print(
                f"[GDELT] Neue News-Kanäle (z. B. neuer Sektor): {sorted(new_ch)} — "
                "fehlende Historie wird wie üblich nachgeladen.",
                flush=True,
            )
    if meta.get("tickers"):
        new_t = set(cfg.ALL_TICKERS) - set(meta["tickers"])
        if new_t:
            secs = sorted({cfg.TICKER_TO_SECTOR[t] for t in new_t if t in cfg.TICKER_TO_SECTOR})
            sample = sorted(new_t)[:18]
            suf = "…" if len(new_t) > 18 else ""
            print(
                f"[GDELT] Neue Wertpapiere ({len(new_t)}): {sample}{suf}", flush=True
            )
            print(
                f"        → Sektoren: {secs} (News pro Sektor; Lücken werden im Pickle ergänzt.)",
                flush=True,
            )
    if not cache_df.empty:
        d0, d1 = cache_df["Date"].min(), cache_df["Date"].max()
        print(f"News cache: {cache_path}  rows={len(cache_df)}  Date span: {d0} … {d1}")
    else:
        print(f"News cache: (empty) {cache_path}")

    fetch_plan = {}
    total_http = 0
    for ch in channels:
        ranges = _missing_bday_ranges_for_channel(cache_df, ch, bdays)
        fetch_plan[ch] = ranges
        for a, b in ranges:
            total_http += _count_http_calls_for_span(a, b)

    n_ranges = sum(len(fetch_plan[ch]) for ch in channels)
    total_chunks = 0
    for ch in channels:
        for a, b in fetch_plan[ch]:
            total_chunks += _n_90d_chunks_in_span(a, b)
    est_sec = total_http * sec_per_call
    print(
        f"GDELT fetch plan: {n_ranges} Lücken-Bereiche über {len(channels)} Kanäle, "
        f"ca. {total_http} HTTP-Calls (timelinetone + vol pro {cfg.GDELT_CHUNK_DAYS}d-Chunk)."
    )
    print(
        f"[GDELT] Fortschritt: {total_chunks} Sub-Chunks gesamt — Logs zeigen (k/{total_chunks}) pro {cfg.GDELT_CHUNK_DAYS}d-Chunk.",
        flush=True,
    )
    print(f"Grobe Zeit-Schätzung: ~{est_sec/60:.1f} min (bei ~{sec_per_call:.2f}s/Call, ohne 429/Retries).")

    new_rows = []
    t0 = _time.perf_counter()
    log_ctx = {"counter": [0], "total": total_chunks}
    n_ch = len(channels)

    def _make_persist(aa, bb, chh):
        def _p(tone_pts, vol_pts):
            nonlocal cache_df
            st = pd.Timestamp(aa).normalize()
            el = pd.Timestamp(bb).normalize()
            tone_s, vol_s = _tv_series_from_points(tone_pts, vol_pts, st, el)
            mini = _df_channel_from_tv(tone_s, vol_s, chh)
            cache_df = _merge_news_cache_rows(cache_df, mini)
            meta["channels"] = cur_ch
            meta["tickers"] = sorted(cfg.ALL_TICKERS)
            meta["last_run_end_date"] = str(end_t.date())
            meta["saved_at"] = pd.Timestamp.now().isoformat()
            _save_news_cache(cache_df, cache_path, meta)
            print(
                f"[GDELT] Zwischenspeicher: Pickle {len(cache_df)} Zeilen (Kanal={chh!r}, nach Sub-Chunk).",
                flush=True,
            )

        return _p

    if total_chunks == 0:
        print("[GDELT] Keine Sub-Chunks zu laden — gesamter Zeitraum bereits im Cache.", flush=True)
    else:
        for ch_i, ch in enumerate(channels, start=1):
            ranges = fetch_plan[ch]
            if not ranges:
                print(
                    f"[GDELT] Kanal {ch_i}/{n_ch} {ch!r}: Cache vollständig — übersprungen.",
                    flush=True,
                )
                continue
            q = cfg.MACRO_NEWS_QUERY if ch == "macro" else _sector_query(ch)
            gn = len(ranges)
            print(
                f"[GDELT] Kanal {ch_i}/{n_ch} {ch!r}: {gn} Lücken-Bereiche → Download startet.",
                flush=True,
            )
            for gi, (a, b) in enumerate(ranges, start=1):
                n_sub = _n_90d_chunks_in_span(a, b)
                print(
                    f"[GDELT]   → Lücke {gi}/{gn}: {a.strftime('%Y-%m-%d')} … {b.strftime('%Y-%m-%d')} "
                    f"({n_sub} Sub-Chunks)",
                    flush=True,
                )
                log_ctx["channel"] = ch
                log_ctx["gap_i"] = gi
                log_ctx["gap_n"] = gn
                persist = _make_persist(a, b, ch)
                tone_s, vol_s = _fetch_channel_tone_vol(
                    q, a, b, log_ctx=log_ctx, persist_chunk=persist
                )
                for d in tone_s.index:
                    new_rows.append(
                        {
                            "Date": d,
                            "channel": ch,
                            "tone": float(tone_s.get(d, 0.0)),
                            "vol": float(vol_s.get(d, 0.0)),
                        }
                    )
    meta["channels"] = cur_ch
    meta["tickers"] = sorted(cfg.ALL_TICKERS)
    meta["last_run_end_date"] = str(end_t.date())
    meta["saved_at"] = pd.Timestamp.now().isoformat()
    _save_news_cache(cache_df, cache_path, meta)
    n_added = len(cache_df) - n_rows_before
    if n_added > 0:
        print(
            f"Cache: +{n_added} Zeilen seit Start dieses Laufs → {cache_path} ({len(cache_df)} rows). "
            "(Zwischenspeicher nach jedem Sub-Chunk bei Rate-Limits.)"
        )
    elif total_chunks == 0:
        print("Keine fehlenden News-Daten — Cache vollständig für den angefragten Zeitraum.")
    else:
        print("Download beendet (Cache-Zeilen siehe oben).")

    elapsed = _time.perf_counter() - t0
    if total_chunks > 0:
        print(f"Netzwerk-Zeit (Lücken-Downloads): {elapsed/60:.2f} min")

    out = cache_df[
        (cache_df["Date"] >= start_t.normalize()) & (cache_df["Date"] <= end_t.normalize())
    ]
    out = out[out["channel"].isin(channels)].copy()
    need_keys = {(d, ch) for d in bdays for ch in channels}
    have_keys = set(zip(pd.to_datetime(out["Date"]).dt.normalize(), out["channel"]))
    still_missing = len(need_keys - have_keys)
    if still_missing:
        print(f"Warnung: {still_missing} (Datum,Kanal)-Paare fehlen noch (z. B. API-Fehler).")

    print(f"News timelines für Pipeline: {len(out)} rows, channels={len(channels)}")
    return out


def fetch_news_sentiment(df, start=cfg.START_DATE, end=cfg.END_DATE, max_threads=8):
    if not cfg.USE_NEWS_SENTIMENT:
        print("News sentiment disabled (cfg.USE_NEWS_SENTIMENT=False). Returning empty.")
        return pd.DataFrame()
    price_start, price_end = start, end
    ns = _c("NEWS_BQ_START_DATE")
    ne = _c("NEWS_BQ_END_DATE")
    if ns:
        start = ns
    else:
        start = price_start
        yb = int(_c("NEWS_EXTRA_HISTORY_YEARS_BEFORE", 0) or 0)
        if yb > 0:
            start = (pd.Timestamp(start) - pd.DateOffset(years=yb)).strftime("%Y-%m-%d")
    if ne:
        end = ne
    else:
        end = price_end
        ya = int(_c("NEWS_EXTRA_HISTORY_YEARS_AFTER", 0) or 0)
        if ya > 0:
            end = (pd.Timestamp(end) + pd.DateOffset(years=ya)).strftime("%Y-%m-%d")
    if (start, end) != (price_start, price_end):
        print(f"[News] Zeitraum {start} … {end}  (Kurs {price_start} … {price_end})", flush=True)
    if _c("NEWS_SOURCE", "bigquery") == "bigquery":
        return _fetch_news_sentiment_bigquery(df, start, end)
    return _fetch_news_sentiment_gdelt_api(df, start, end, max_threads)
