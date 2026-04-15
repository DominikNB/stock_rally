"""stock_rally_v10 — News/Sentiment (Pipeline-Modul)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lib.stock_rally_v10 import config as cfg


def _c(attr: str, default=None):
    """Config lesen (Attribute heißen z. B. NEWS_BQ_START_DATE — nicht ``cfg.NEWS_...`` im __dict__)."""
    return getattr(cfg, attr, default)


def _bq_gcam_regexp_for_key(key: str) -> str:
    """RE2-Muster für erstes ``<key>:<zahl>`` im GCAM-Segment (Komma-getrennt).

    ``key`` kann Punkte und Doppelpunkte enthalten (z. B. ``c18.158``, ``vnt:success``);
    ``re.escape`` verankert den literalen Key; der Wert folgt dem letzten ``:`` des Paares.
    """
    return re.escape(str(key).strip()) + r":(-?[0-9]+(?:\.[0-9]+)?)"


def _gcam_series_from_points(gc_pts, start_ts, end_last):
    s = _merge_points(gc_pts) if gc_pts else pd.Series(dtype=float)
    bdays = pd.bdate_range(
        pd.Timestamp(start_ts).normalize(), pd.Timestamp(end_last).normalize()
    )
    # NaN = kein Tagesaggregat (z. B. kein passender GKG-Tag im Fenster); nicht mit GCAM-Wert 0 verwechseln.
    return s.reindex(bdays)


def _gcam_by_col_from_wide_channel(df_wide, ch, st, el, *, anchor: bool = False):
    gcam_by_col = {}
    if df_wide is None or getattr(df_wide, "empty", True) or not ch:
        return gcam_by_col
    st = pd.Timestamp(st).normalize()
    el = pd.Timestamp(el).normalize()
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        wc = f"{ch}_anchor_{cn}" if anchor else f"{ch}_{cn}"
        if wc not in df_wide.columns:
            continue
        gc_pts = []
        for _, row in df_wide.iterrows():
            dd = pd.Timestamp(row["d"]).normalize()
            gv = row[wc]
            gc_pts.append((dd, float(gv) if pd.notna(gv) else np.nan))
        gcam_by_col[cn] = _gcam_series_from_points(gc_pts, st, el)
    return gcam_by_col


def _gcam_by_col_from_bq_daily(df_bq, st, el, *, anchor: bool = False):
    gcam_by_col = {}
    if df_bq is None or getattr(df_bq, "empty", True):
        return gcam_by_col
    st = pd.Timestamp(st).normalize()
    el = pd.Timestamp(el).normalize()
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        col = f"anchor_{cn}" if anchor else cn
        if col not in df_bq.columns:
            continue
        gc_pts = []
        for _, row in df_bq.iterrows():
            dd = pd.Timestamp(row["d"]).normalize()
            gv = row[col]
            gc_pts.append((dd, float(gv) if pd.notna(gv) else np.nan))
        gcam_by_col[cn] = _gcam_series_from_points(gc_pts, st, el)
    return gcam_by_col


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


def _bq_sector_keyword_conjunction_enabled(sector: str) -> bool:
    if not bool(_c("BQ_SECTOR_THEME_KEYWORD_CONJUNCTION", False)):
        return False
    chosen = _c("BQ_SECTOR_THEME_KEYWORD_CHANNELS", ()) or ()
    if not chosen:
        return True
    return str(sector) in {str(x) for x in chosen}


def _bq_sector_keyword_or_sql(sector: str) -> str:
    kws = list(getattr(cfg, "SECTOR_KEYWORDS", {}).get(sector) or [])
    if not kws:
        return ""
    txt = (
        "LOWER(CONCAT(IFNULL(V2Themes, ''), ' ', IFNULL(V2Organizations, ''), ' ', "
        "IFNULL(V2Persons, ''), ' ', IFNULL(V2Locations, '')))"
    )
    parts = []
    for kw in kws:
        term = str(kw).strip().lower()
        if not term:
            continue
        pat = re.escape(term).replace("'", "''")
        parts.append(f"REGEXP_CONTAINS({txt}, r'{pat}')")
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def _bq_sector_theme_cond(sector: str, *, for_base_agg: bool = True) -> str:
    """Sektor-Zeile im GKG-Scan: Theme; optional AND Keywords (siehe cfg)."""
    base = str(cfg.SECTOR_BQ_THEME_WHERE.get(sector) or "(1=0)")
    if not _bq_sector_keyword_conjunction_enabled(sector):
        return base
    kw_or = _bq_sector_keyword_or_sql(sector)
    if not kw_or:
        return base
    if for_base_agg and not bool(_c("BQ_SECTOR_KEYWORD_CONJUNCTION_FOR_BASE_AGG", False)):
        return base
    return f"(({base}) AND ({kw_or}))"


# Mindestabstand zwischen GDELT-HTTP-Calls (öffentliche API ist streng rate-limited).
_GDELT_THROTTLE_LAST = [0.0]
_ANCHOR_FILTER_LOGGED = False
_BASE_TONE_MODE_LOGGED = False


def _gdelt_throttle():
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


def _df_channel_from_tv(
    tone_s,
    vol_s,
    channel,
    impact_s=None,
    gcam_by_col=None,
    *,
    anchor_tone_s=None,
    anchor_vol_s=None,
    gcam_anchor_by_col=None,
):
    if impact_s is None:
        impact_s = pd.Series(0.0, index=tone_s.index)
    else:
        impact_s = impact_s.reindex(tone_s.index).fillna(0.0)
    gcam_by_col = gcam_by_col or {}
    gcam_anchor_by_col = gcam_anchor_by_col or {}
    data = {
        "Date": tone_s.index,
        "channel": channel,
        "tone": tone_s.values.astype(float),
        "vol": vol_s.values.astype(float),
        "impact": impact_s.values.astype(float),
    }
    for gk in cfg._gkg_gcam_keys_clean():
        cn = cfg.gcam_series_colname(gk)
        ser = gcam_by_col.get(cn)
        if ser is None:
            data[cn] = np.full(len(tone_s), np.nan, dtype=float)
        else:
            data[cn] = ser.reindex(tone_s.index).fillna(np.nan).values.astype(float)
    has_anchor = (
        anchor_tone_s is not None
        and anchor_vol_s is not None
        and len(anchor_tone_s)
        and len(anchor_vol_s)
    )
    if has_anchor:
        data["anchor_tone"] = anchor_tone_s.reindex(tone_s.index).fillna(0.0).values.astype(float)
        data["anchor_vol"] = anchor_vol_s.reindex(tone_s.index).fillna(0.0).values.astype(float)
        ap = "anchor_"
        for gk in cfg._gkg_gcam_keys_clean():
            cn = cfg.gcam_series_colname(gk)
            ser = gcam_anchor_by_col.get(cn)
            col_a = f"{ap}{cn}"
            if ser is None:
                data[col_a] = np.full(len(tone_s), np.nan, dtype=float)
            else:
                data[col_a] = ser.reindex(tone_s.index).fillna(np.nan).values.astype(float)
    return pd.DataFrame(data)


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
    base_cols = ["Date", "channel", "tone", "vol", "impact"]
    gcam_cols = [cfg.gcam_series_colname(k) for k in cfg._gkg_gcam_keys_clean()]
    need_anchor = bool(getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False))
    if df is not None and not getattr(df, "empty", True):
        need_anchor = need_anchor or any(
            str(c).startswith("anchor_") for c in df.columns
        )
    anchor_extra = (
        ["anchor_tone", "anchor_vol"]
        + [f"anchor_{cfg.gcam_series_colname(k)}" for k in cfg._gkg_gcam_keys_clean()]
        if need_anchor
        else []
    )
    all_cols = base_cols + gcam_cols + anchor_extra
    if df is None or df.empty:
        return pd.DataFrame(columns=all_cols)
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out["channel"] = out["channel"].astype(str)
    if "impact" not in out.columns:
        out["impact"] = 0.0
    for c in gcam_cols + anchor_extra:
        if c not in out.columns:
            out[c] = np.nan
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


def _save_news_cache(df, path, meta=None, *, log_label: str | None = None):
    import pickle

    abs_path = os.path.abspath(str(path))
    tmp = abs_path + ".tmp"
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    n_rows = len(df) if df is not None else 0
    n_ch = (
        int(df["channel"].nunique())
        if df is not None and not df.empty and "channel" in df.columns
        else 0
    )
    _lbl = f" — {log_label}" if log_label else ""
    print(
        f"[News-Cache] Schreibe Pickle{_lbl}\n"
        f"             → {tmp}\n"
        f"             | {n_rows:,} Zeilen, {n_ch} Kanäle (atomar per os.replace) …",
        flush=True,
    )
    m = dict(meta or {})
    # Vereinigung: nie Meta auf „nur USE_EXTRA_N-Kappung“ schrumpfen — sonst wirkt später ein
    # erweitertes Key-Set wie fehlende Spalten und löst unnötig BigQuery aus.
    _prev_g = set(_gcam_keys_meta_tuple(m.get("gcam_keys")))
    _cur_g = set(cfg._gkg_gcam_keys_clean())
    m["gcam_keys"] = sorted(_prev_g | _cur_g)
    m["news_anchor_gcam_dual"] = bool(getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False))
    payload = {"df": df, "meta": m}
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, abs_path)
    try:
        sz_mb = os.path.getsize(abs_path) / (1024 * 1024)
    except OSError:
        sz_mb = 0.0
    print(
        f"[News-Cache] Pickle fertig{_lbl}: {abs_path} | {sz_mb:.2f} MiB | {n_rows:,} Zeilen",
        flush=True,
    )


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


def _missing_bday_ranges_for_channel(cache_df, channel, bdays_index, force_refetch=False):
    need = set(pd.to_datetime(bdays_index).normalize())
    if force_refetch:
        missing = sorted(need)
        return _cluster_sorted_dates(missing)
    have = set()
    if cache_df is not None and not cache_df.empty:
        sub = cache_df[cache_df["channel"] == channel]
        if len(sub):
            have = set(pd.to_datetime(sub["Date"]).dt.normalize())
    missing = sorted(need - have)
    return _cluster_sorted_dates(missing)


def _gcam_keys_meta_tuple(meta_keys) -> tuple[str, ...]:
    if not meta_keys:
        return tuple()
    return tuple(sorted(str(k).strip() for k in meta_keys if str(k).strip()))


def _needs_full_gcam_news_refetch(
    meta_prev: tuple[str, ...], meta_cur: tuple[str, ...]
) -> bool:
    """Voll-Nachladen (BQ/API) wegen GCAM nur wenn der Cache **zusätzliche** Keys braucht.

    ``meta_prev`` = beim letzten Speichern im Pickle notierte Key-Menge (Superset der geladenen Spalten).
    ``meta_cur`` = aktuell aktive Keys (inkl. ``GKG_GCAM_USE_EXTRA_N``-Kürzung).

    Ist ``meta_cur`` eine **Teilmenge** von ``meta_prev``, reichen die vorhandenen Cache-Spalten — kein
    teurer Re-Scan nur wegen weniger Keys fürs Training.
    """
    if not meta_cur:
        return False
    if not meta_prev:
        return True
    return not set(meta_cur).issubset(set(meta_prev))


def _read_gcam_keys_file(path: Path, max_age_days: int) -> tuple[str, ...] | None:
    import json
    from datetime import datetime, timezone

    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        keys = obj.get("keys") or []
        if not isinstance(keys, list) or not keys:
            return None
        if max_age_days > 0:
            upd = obj.get("updated_at")
            if upd:
                try:
                    u = datetime.fromisoformat(str(upd).replace("Z", "+00:00"))
                    if u.tzinfo is None:
                        u = u.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - u).total_seconds() / 86400.0
                    if age > max_age_days:
                        return None
                except (TypeError, ValueError):
                    pass
        out = tuple(str(k).strip() for k in keys if str(k).strip())
        return out if out else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _write_gcam_keys_file(path: Path, keys: tuple[str, ...]) -> None:
    import json
    from datetime import datetime, timezone

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": list(keys),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    print(
        f"[GCAM] Keys-JSON geschrieben: {path.resolve()} | {len(keys)} Keys",
        flush=True,
    )


def _bq_fetch_extra_gcam_c_keys(extra_limit: int, exclude: frozenset[str]) -> tuple[str, ...]:
    """Top-häufige c*-Keys (gestern, Step-0-Logik), ohne ``exclude``; höchstens ``extra_limit`` Stück."""
    from google.cloud import bigquery

    if extra_limit <= 0:
        return tuple()
    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    proj = _bq_resolve_project()
    if not proj:
        return tuple()
    fetch_cap = min(500, max(60, int(extra_limit) * 15 + len(exclude) * 2))
    sql = f"""
WITH raw AS (
  SELECT CAST(GCAM AS STRING) AS gcam_s
  FROM {table}
  WHERE DATE(_PARTITIONTIME) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
    AND GCAM IS NOT NULL
    AND CAST(GCAM AS STRING) != ''
  LIMIT 50000
),
segs AS (
  SELECT TRIM(segment) AS seg
  FROM raw, UNNEST(SPLIT(gcam_s, ',')) AS segment
),
parsed AS (
  SELECT
    REGEXP_EXTRACT(seg, r'^([a-zA-Z][a-zA-Z0-9_.]*)') AS dim_key,
    SAFE_CAST(REGEXP_EXTRACT(seg, r':(-?[0-9]+(?:\\.[0-9]+)?)$') AS FLOAT64) AS dim_val
  FROM segs
  WHERE seg != ''
    AND REGEXP_CONTAINS(seg, r'^[a-zA-Z][a-zA-Z0-9_.]*:-?[0-9]+(?:\\.[0-9]+)?$')
)
SELECT dim_key, COUNT(*) AS n
FROM parsed
WHERE dim_key IS NOT NULL
  AND STARTS_WITH(dim_key, 'c')
GROUP BY dim_key
ORDER BY n DESC
LIMIT @cap
"""
    client = bigquery.Client(project=proj)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("cap", "INT64", int(fetch_cap))]
    )
    job = client.query(sql, job_config=job_config)
    rows = list(job.result(max_results=int(fetch_cap)))
    out: list[str] = []
    ex = {str(x).strip() for x in exclude}
    for r in rows:
        k = str(r.get("dim_key") or "").strip()
        if not k or k in ex:
            continue
        out.append(k)
        if len(out) >= int(extra_limit):
            break
    return tuple(out)


def _merge_gcam_key_order(must: tuple[str, ...], *more: tuple[str, ...]) -> tuple[str, ...]:
    """Must-Have zuerst, dann weitere Tupel in Reihenfolge, ohne Duplikate."""
    seen: set[str] = set()
    out: list[str] = []
    for group in (must,) + more:
        for raw in group:
            k = str(raw).strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
    return tuple(out)


def _cap_gcam_keys_use_extra_n(
    keys: tuple[str, ...],
    must: tuple[str, ...],
    use_extra_n: int | None,
) -> tuple[str, ...]:
    """Alle Must-Haves behalten; danach höchstens ``use_extra_n`` weitere Keys (Reihenfolge wie in ``keys``).

    ``use_extra_n is None`` → keine Kürzung. ``0`` → nur Must-Haves.
    """
    if use_extra_n is None:
        return keys
    cap = int(use_extra_n)
    if cap < 0:
        return keys
    must_set = frozenset(str(k).strip() for k in must if str(k).strip())
    must_part = [k for k in keys if k in must_set]
    extras = [k for k in keys if k not in must_set]
    return tuple(must_part + extras[:cap])


def resolve_gkg_gcam_metric_keys(*, allow_bigquery_refresh: bool = True) -> None:
    """
    Setzt ``cfg.GKG_GCAM_METRIC_KEYS`` für diesen Lauf.
    - ``allow_bigquery_refresh``: nur wirksam wenn ``GKG_GCAM_EXPLORE_KEYS=True`` — dann darf ein
      BigQuery-Job **zusätzliche c*-Keys** scouten. Ohne dieses Flag keine GCAM-Key-Exploration, nur
      Datei/Must-Haves (unabhängig von späteren News-GKG-Queries zum Cache-Füllen).
    - Nicht-leeres Snapshot (erster Aufruf): feste User-Liste aus ``GKG_GCAM_METRIC_KEYS``.
    - Sonst: immer ``GKG_GCAM_MUST_HAVE_KEYS``, plus bei Exploration zusätzliche Top-c*,
      ohne Datei: nur Must-Haves; mit Datei: Must-Haves zuerst, dann übrige Keys aus JSON.
    - Wenn ``GKG_GCAM_USE_EXTRA_N`` gesetzt (nicht ``None``): nach dem Resolve höchstens so viele
      *zusätzliche* Keys wie angegeben (Must-Haves bleiben vollständig). JSON bleibt unverändert.
      Gilt auch für Snapshot (explizite ``GKG_GCAM_METRIC_KEYS`` in ``config``).
    """
    if not hasattr(cfg, "_gkg_gcam_keys_user_snapshot"):
        cfg._gkg_gcam_keys_user_snapshot = tuple(
            str(k).strip() for k in (cfg.GKG_GCAM_METRIC_KEYS or ()) if str(k).strip()
        )
    snap = getattr(cfg, "_gkg_gcam_keys_user_snapshot", tuple())
    must = cfg._gkg_gcam_must_have_keys_clean()
    use_n = getattr(cfg, "GKG_GCAM_USE_EXTRA_N", None)

    def _apply_cap_and_assign(keys: tuple[str, ...], src: str) -> None:
        if keys and use_n is not None:
            n_before = len(keys)
            keys = _cap_gcam_keys_use_extra_n(keys, must, use_n)
            if len(keys) != n_before:
                print(
                    f"[GCAM] GKG_GCAM_USE_EXTRA_N={use_n}: {n_before} → {len(keys)} Keys "
                    f"(Must-Haves + höchstens {int(use_n)} Extras für Features/Training).",
                    flush=True,
                )
        if keys:
            print(
                f"[GCAM] {len(keys)} Keys ({src}): {list(keys)[:14]}{'…' if len(keys) > 14 else ''}",
                flush=True,
            )
        cfg.GKG_GCAM_METRIC_KEYS = keys

    if snap:
        _apply_cap_and_assign(snap, "explizit GKG_GCAM_METRIC_KEYS (Snapshot)")
        return
    if not getattr(cfg, "GKG_GCAM_AUTO_RESOLVE", True):
        cfg.GKG_GCAM_METRIC_KEYS = tuple()
        return
    path = Path(str(getattr(cfg, "GKG_GCAM_KEYS_PATH", Path("data") / "gkg_gcam_metric_keys.json")))
    extra_n = int(
        getattr(cfg, "GKG_GCAM_EXPLORE_EXTRA_N", None)
        or getattr(cfg, "GKG_GCAM_AUTO_TOP_N", 8)
        or 8
    )
    explore = bool(getattr(cfg, "GKG_GCAM_EXPLORE_KEYS", False))
    keys: tuple[str, ...] = tuple()
    src = ""
    if explore:
        if not allow_bigquery_refresh:
            print(
                "[GCAM] GKG_GCAM_EXPLORE_KEYS=True, aber keine GKG-BigQuery-Quelle — Exploration übersprungen.",
                flush=True,
            )
        else:
            try:
                extras = _bq_fetch_extra_gcam_c_keys(extra_n, frozenset(must))
                keys = _merge_gcam_key_order(must, extras)
                _write_gcam_keys_file(path, keys)
                src = (
                    f"Must-Have ({len(must)}) + Exploration ({len(extras)} c*) → {path}"
                )
                if not extras:
                    print(
                        "[GCAM] Exploration: keine zusätzlichen c*-Keys (gestern evtl. leer).",
                        flush=True,
                    )
            except Exception as ex:
                print(f"[GCAM] Exploration fehlgeschlagen ({ex!r}).", flush=True)
        if not keys:
            fb = _read_gcam_keys_file(path, max_age_days=0)
            if fb is not None:
                keys = _merge_gcam_key_order(must, fb)
                src = f"Fallback Datei {path}"
            elif must:
                keys = must
                src = "nur Must-Have (nach fehlgeschlagener / übersprungener Exploration)"
    else:
        from_file = _read_gcam_keys_file(path, max_age_days=0)
        if from_file is not None:
            keys = _merge_gcam_key_order(must, from_file)
            src = f"Must-Have + Datei {path}"
        elif must:
            keys = must
            src = "nur Must-Have (keine JSON / Exploration aus)"
    _apply_cap_and_assign(keys, src)


def _strip_legacy_gcam_columns(cache_df: pd.DataFrame) -> pd.DataFrame:
    drop = [
        c
        for c in cache_df.columns
        if str(c).startswith("gcam_") or str(c).startswith("anchor_")
    ]
    if not drop:
        return cache_df
    return cache_df.drop(columns=drop, errors="ignore")


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


def _bq_gkg_theme_triple_cond(channel: str, raw_token: str, ref_date=None) -> str:
    """Basis-Sektor/Makro-SQL UND ein konkretes GKG-Theme-Token (pro ``Kanal##alias``-Zeitreihe)."""
    tok = str(raw_token).strip()
    like = f"V2Themes LIKE '%{_bq_escape_like(tok)}%'"
    if channel == "macro":
        base = str(_c("MACRO_BQ_THEME_WHERE", "(1=1)"))
    else:
        # Tripel: Keywords am Sektor-Base belassen (schmaler Zusatzkanal).
        base = _bq_sector_theme_cond(channel, for_base_agg=False)
    return f"(({base}) AND ({like}))"


def _bq_run_gkg_daily_agg(
    theme_sql_broad: str,
    a,
    b,
    *,
    theme_sql_anchor: str | None = None,
):
    """GKG-Tagesaggregat; optional zweites Set (Anker: Theme ∧ V2Organizations) bei gleichem WHERE."""
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
    tone_part = "SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)"
    broad_cond = f"({theme_sql_broad})"
    use_anchor = bool(
        theme_sql_anchor
        and str(theme_sql_anchor).strip() != str(theme_sql_broad).strip()
    )
    anch_cond = f"({theme_sql_anchor})" if use_anchor else None
    sel_lines = [
        f"AVG(IF( {broad_cond}, {tone_part}, NULL)) AS avg_daily_tone",
        f"COUNTIF( {broad_cond} ) AS article_count",
        "CAST(NULL AS FLOAT64) AS avg_goldstein",
    ]
    for gk in cfg._gkg_gcam_keys_clean():
        pat = _bq_gcam_regexp_for_key(gk)
        cn = cfg.gcam_series_colname(gk)
        sel_lines.append(
            f"AVG(IF( {broad_cond}, SAFE_CAST(REGEXP_EXTRACT(CAST(GCAM AS STRING), r'{pat}') AS FLOAT64), NULL)) AS {cn}"
        )
    if use_anchor and anch_cond:
        sel_lines.append(
            f"AVG(IF( {anch_cond}, {tone_part}, NULL)) AS anchor_avg_daily_tone"
        )
        sel_lines.append(f"COUNTIF( {anch_cond} ) AS anchor_article_count")
        for gk in cfg._gkg_gcam_keys_clean():
            pat = _bq_gcam_regexp_for_key(gk)
            cn = cfg.gcam_series_colname(gk)
            sel_lines.append(
                f"AVG(IF( {anch_cond}, SAFE_CAST(REGEXP_EXTRACT(CAST(GCAM AS STRING), r'{pat}') AS FLOAT64), NULL)) AS anchor_{cn}"
            )
    select_inner = ",\n  ".join(sel_lines)

    sql = f"""
SELECT
  DATE(_PARTITIONTIME) AS d,
  {select_inner}
FROM {table}
WHERE 1=1
  {partition_sql}
  AND ({theme_sql_broad})
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
    theme_sql = _bq_gkg_theme_triple_cond(channel, raw_token, ref_date=b)
    return _bq_run_gkg_daily_agg(theme_sql, a, b, theme_sql_anchor=None)


def _bq_query_all_channels_gkg(a, b):
    """Eine Query, ein Scan: OR über alle Theme-Filter, IF/COUNTIF pro Kanal."""
    from google.cloud import bigquery

    global _ANCHOR_FILTER_LOGGED, _BASE_TONE_MODE_LOGGED
    if getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False) and not _ANCHOR_FILTER_LOGGED:
        print(
            "[BigQuery] Anker-Filter aktiv: Sektoren doppelt — breites V2Themes + Anker (Theme plus Orgs); "
            "gleiche GCAM-Keys für beide; Zeitplan: cfg.NEWS_ANCHOR_SCHEDULE_PATH",
            flush=True,
        )
        _ANCHOR_FILTER_LOGGED = True
    if (
        not _BASE_TONE_MODE_LOGGED
        and bool(_c("BQ_SECTOR_THEME_KEYWORD_CONJUNCTION", False))
        and not bool(_c("BQ_SECTOR_KEYWORD_CONJUNCTION_FOR_BASE_AGG", False))
    ):
        print(
            "[BigQuery] Sektor-Basis-Tone/Vol: nur Theme-Filter (breiter); Keyword-AND gilt weiter für Anker-Spalten.",
            flush=True,
        )
        _BASE_TONE_MODE_LOGGED = True
    if not _c("BQ_USE_PARTITION_FILTER", True):
        print(
            "[BigQuery] WARNUNG: cfg.BQ_USE_PARTITION_FILTER=False kann Full-Table-Scan bedeuten.",
            flush=True,
        )
    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    channels = ["macro"] + list(cfg.TICKERS_BY_SECTOR.keys())
    tone_part = "SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)"
    ref = pd.Timestamp(b).normalize()
    or_parts = [f"({cfg.MACRO_BQ_THEME_WHERE})"]
    for s in cfg.TICKERS_BY_SECTOR:
        or_parts.append(f"({_bq_sector_theme_cond(s)})")
    combined_or = "(" + " OR ".join(or_parts) + ")"
    sel_parts = ["DATE(_PARTITIONTIME) AS d"]
    for ch in channels:
        if ch == "macro":
            cond = cfg.MACRO_BQ_THEME_WHERE
            sel_parts.append(f"AVG(IF( ({cond}), {tone_part}, NULL)) AS {ch}_tone")
            sel_parts.append(f"COUNTIF( ({cond}) ) AS {ch}_vol")
            for gk in cfg._gkg_gcam_keys_clean():
                pat = _bq_gcam_regexp_for_key(gk)
                cn = cfg.gcam_series_colname(gk)
                sel_parts.append(
                    f"AVG(IF( ({cond}), SAFE_CAST(REGEXP_EXTRACT(CAST(GCAM AS STRING), r'{pat}') AS FLOAT64), NULL)) AS {ch}_{cn}"
                )
            continue
        cond_b = _bq_sector_theme_cond(ch)
        sel_parts.append(f"AVG(IF( ({cond_b}), {tone_part}, NULL)) AS {ch}_tone")
        sel_parts.append(f"COUNTIF( ({cond_b}) ) AS {ch}_vol")
        for gk in cfg._gkg_gcam_keys_clean():
            pat = _bq_gcam_regexp_for_key(gk)
            cn = cfg.gcam_series_colname(gk)
            sel_parts.append(
                f"AVG(IF( ({cond_b}), SAFE_CAST(REGEXP_EXTRACT(CAST(GCAM AS STRING), r'{pat}') AS FLOAT64), NULL)) AS {ch}_{cn}"
            )
        if cfg.sector_anchor_org_active(ch, ref):
            cond_a_base = cfg.sector_news_theme_sql(ch, ref)
            kw_or = _bq_sector_keyword_or_sql(ch) if _bq_sector_keyword_conjunction_enabled(ch) else ""
            cond_a = f"(({cond_a_base}) AND ({kw_or}))" if kw_or else cond_a_base
            sel_parts.append(
                f"AVG(IF( ({cond_a}), {tone_part}, NULL)) AS {ch}_anchor_tone"
            )
            sel_parts.append(f"COUNTIF( ({cond_a}) ) AS {ch}_anchor_vol")
            for gk in cfg._gkg_gcam_keys_clean():
                pat = _bq_gcam_regexp_for_key(gk)
                cn = cfg.gcam_series_colname(gk)
                sel_parts.append(
                    f"AVG(IF( ({cond_a}), SAFE_CAST(REGEXP_EXTRACT(CAST(GCAM AS STRING), r'{pat}') AS FLOAT64), NULL)) AS {ch}_anchor_{cn}"
                )
    trips = getattr(cfg, "GKG_THEME_SQL_TRIPLES", None) or []
    for tch, alias, raw_tok in trips:
        if tch != "macro" and tch not in cfg.SECTOR_BQ_THEME_WHERE:
            continue
        trip_cond = _bq_gkg_theme_triple_cond(tch, raw_tok, ref_date=ref)
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
        theme_broad = _c("MACRO_BQ_THEME_WHERE", "(1=1)")
        theme_anchor = None
    else:
        ref_b = pd.Timestamp(b).normalize()
        theme_broad = _bq_sector_theme_cond(channel)
        theme_anchor = None
        if cfg.sector_anchor_org_active(channel, ref_b):
            t_anchor = cfg.sector_news_theme_sql(channel, ref_b)
            kw_or = _bq_sector_keyword_or_sql(channel) if _bq_sector_keyword_conjunction_enabled(channel) else ""
            theme_anchor = f"(({t_anchor}) AND ({kw_or}))" if kw_or else t_anchor
    geo_sql = "" if use_gkg else _bq_geo_sql(is_macro, None if is_macro else channel)
    if use_gkg:
        return _bq_run_gkg_daily_agg(
            theme_broad, a, b, theme_sql_anchor=theme_anchor
        )
    table = _c("GDELT_BQ_EVENTS_TABLE", "`gdelt-bq.gdeltv2.gkg_partitioned`")
    d0, d1 = _bq_ymd_int(a), _bq_ymd_int(b)
    tc = _c("BQ_THEMES_COLUMN", "V2Themes")
    theme_sql = theme_broad.replace("V2Themes", tc)
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
            "BigQuery: Kein GCP-Projekt. Setze z. B. cfg.BQ_PROJECT_ID = 'dein-projekt-id' "
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
    meta_cur = _gcam_keys_meta_tuple(cfg._gkg_gcam_keys_clean())
    meta_prev = _gcam_keys_meta_tuple(meta.get("gcam_keys"))
    force_gcam_refetch = _needs_full_gcam_news_refetch(meta_prev, meta_cur)
    if meta_cur != meta_prev and not force_gcam_refetch and (meta_cur or meta_prev):
        print(
            "[GCAM] Aktive Key-Liste ist Teilmenge des Caches (z. B. GKG_GCAM_USE_EXTRA_N) — "
            "kein BigQuery-Neu-Scan; vorhandene GCAM-Spalten werden genutzt.",
            flush=True,
        )
    cfg_anchor_dual = bool(getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False))
    cache_anchor_dual = bool(meta.get("news_anchor_gcam_dual"))
    force_anchor_refetch = cfg_anchor_dual != cache_anchor_dual
    if force_anchor_refetch:
        print(
            "[News-Cache] Anker-GCAM-Modus geändert (breit+Anker vs. nur breit) — Cache-Zeilen werden neu geladen.",
            flush=True,
        )
        cache_df = _strip_legacy_gcam_columns(cache_df)
    if force_gcam_refetch and (meta_cur or meta_prev):
        print(
            f"[GCAM] Cache deckt nicht alle benötigten GCAM-Keys ab ({list(meta_prev)!r} → {list(meta_cur)!r}) — "
            "Zeilen werden für den angefragten Zeitraum neu geladen.",
            flush=True,
        )
        cache_df = _strip_legacy_gcam_columns(cache_df)
    force_news_refetch = force_gcam_refetch or force_anchor_refetch
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
    fetch_plan = {
        ch: _missing_bday_ranges_for_channel(
            cache_df, ch, bdays, force_refetch=force_news_refetch
        )
        for ch in channels
    }
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
                    gcam_by_col = _gcam_by_col_from_wide_channel(
                        df_wide, ch, st, el, anchor=False
                    )
                    atc, avc = f"{ch}_anchor_tone", f"{ch}_anchor_vol"
                    if atc in df_wide.columns and avc in df_wide.columns:
                        tone_pa, vol_pa = [], []
                        for _, row in df_wide.iterrows():
                            dd = pd.Timestamp(row["d"]).normalize()
                            tone_pa.append(
                                (
                                    dd,
                                    float(row[atc]) if pd.notna(row[atc]) else 0.0,
                                )
                            )
                            vol_pa.append(
                                (
                                    dd,
                                    float(row[avc]) if pd.notna(row[avc]) else 0.0,
                                )
                            )
                        anchor_tone_s, anchor_vol_s = _tv_series_from_points(
                            tone_pa, vol_pa, st, el
                        )
                        gcam_a = _gcam_by_col_from_wide_channel(
                            df_wide, ch, st, el, anchor=True
                        )
                        mini = _df_channel_from_tv(
                            tone_s,
                            vol_s,
                            ch,
                            imp_s,
                            gcam_by_col,
                            anchor_tone_s=anchor_tone_s,
                            anchor_vol_s=anchor_vol_s,
                            gcam_anchor_by_col=gcam_a,
                        )
                    else:
                        mini = _df_channel_from_tv(
                            tone_s, vol_s, ch, imp_s, gcam_by_col
                        )
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
                    mini = _df_channel_from_tv(tone_s, vol_s, cache_ch, imp_s, None)
                    cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(
                    cache_df,
                    cache_path,
                    meta,
                    log_label="BigQuery Single-Scan (Zwischenspeicher)",
                )
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
                gcam_by_col = _gcam_by_col_from_bq_daily(df_bq, st, el, anchor=False)
                if (
                    df_bq is not None
                    and len(df_bq)
                    and "anchor_avg_daily_tone" in df_bq.columns
                ):
                    tone_pa, vol_pa = [], []
                    for _, row in df_bq.iterrows():
                        dd = pd.Timestamp(row["d"]).normalize()
                        tone_pa.append(
                            (
                                dd,
                                float(row["anchor_avg_daily_tone"])
                                if pd.notna(row["anchor_avg_daily_tone"])
                                else 0.0,
                            )
                        )
                        vol_pa.append(
                            (
                                dd,
                                float(row["anchor_article_count"])
                                if pd.notna(row["anchor_article_count"])
                                else 0.0,
                            )
                        )
                    anchor_tone_s, anchor_vol_s = _tv_series_from_points(
                        tone_pa, vol_pa, st, el
                    )
                    gcam_a = _gcam_by_col_from_bq_daily(df_bq, st, el, anchor=True)
                    mini = _df_channel_from_tv(
                        tone_s,
                        vol_s,
                        ch,
                        imp_s,
                        gcam_by_col,
                        anchor_tone_s=anchor_tone_s,
                        anchor_vol_s=anchor_vol_s,
                        gcam_anchor_by_col=gcam_a,
                    )
                else:
                    mini = _df_channel_from_tv(tone_s, vol_s, ch, imp_s, gcam_by_col)
                cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(
                    cache_df,
                    cache_path,
                    meta,
                    log_label=f"BigQuery Kanal {ch!r} Lücke {gi}/{gn}",
                )
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
                gcam_by_col = _gcam_by_col_from_bq_daily(df_bq, st, el)
                mini = _df_channel_from_tv(tone_s, vol_s, cache_ch, imp_s, gcam_by_col)
                cache_df = _merge_news_cache_rows(cache_df, mini)
                meta["channels"] = cur_ch
                meta["tickers"] = sorted(cfg.ALL_TICKERS)
                meta["last_run_end_date"] = str(end_t.date())
                meta["saved_at"] = pd.Timestamp.now().isoformat()
                meta["source"] = "bigquery"
                _save_news_cache(
                    cache_df,
                    cache_path,
                    meta,
                    log_label=f"BigQuery Theme {cache_ch!r} Lücke {gi}/{gn}",
                )
                print(
                    f"[BigQuery] Zwischenspeicher: {len(cache_df)} Zeilen (Kanal={cache_ch!r}).",
                    flush=True,
                )
    meta["channels"] = cur_ch
    meta["tickers"] = sorted(cfg.ALL_TICKERS)
    meta["last_run_end_date"] = str(end_t.date())
    meta["saved_at"] = pd.Timestamp.now().isoformat()
    meta["source"] = "bigquery"
    _save_news_cache(
        cache_df, cache_path, meta, log_label="BigQuery Lauf Abschluss"
    )
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


def _fetch_news_sentiment_gdelt_api(df, start=cfg.START_DATE, end=cfg.END_DATE):
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

    meta_cur = _gcam_keys_meta_tuple(cfg._gkg_gcam_keys_clean())
    meta_prev = _gcam_keys_meta_tuple(meta.get("gcam_keys"))
    force_gcam_refetch = _needs_full_gcam_news_refetch(meta_prev, meta_cur)
    if meta_cur != meta_prev and not force_gcam_refetch and (meta_cur or meta_prev):
        print(
            "[GCAM] Aktive Key-Liste ist Teilmenge des Caches (z. B. GKG_GCAM_USE_EXTRA_N) — "
            "kein API-Neu-Fetch wegen GCAM-Keys.",
            flush=True,
        )
    cfg_anchor_dual = bool(getattr(cfg, "NEWS_ANCHOR_ORG_FILTER", False))
    cache_anchor_dual = bool(meta.get("news_anchor_gcam_dual"))
    force_anchor_refetch = cfg_anchor_dual != cache_anchor_dual
    if force_anchor_refetch:
        print(
            "[GDELT] Anker-GCAM-Modus geändert — Cache-Zeilen werden neu geladen (API ohne GCAM).",
            flush=True,
        )
        cache_df = _strip_legacy_gcam_columns(cache_df)
    if force_gcam_refetch and (meta_cur or meta_prev):
        print(
            f"[GCAM] Cache deckt nicht alle benötigten GCAM-Keys ab ({list(meta_prev)!r} → {list(meta_cur)!r}) — "
            "GDELT-API-Zeilen werden neu geladen (ohne GCAM-Werte).",
            flush=True,
        )
        cache_df = _strip_legacy_gcam_columns(cache_df)
    force_news_refetch = force_gcam_refetch or force_anchor_refetch

    fetch_plan = {}
    total_http = 0
    for ch in channels:
        ranges = _missing_bday_ranges_for_channel(
            cache_df, ch, bdays, force_refetch=force_news_refetch
        )
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
            _save_news_cache(
                cache_df,
                cache_path,
                meta,
                log_label=f"GDELT Sub-Chunk Kanal={chh!r}",
            )
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
                _fetch_channel_tone_vol(
                    q, a, b, log_ctx=log_ctx, persist_chunk=persist
                )
    meta["channels"] = cur_ch
    meta["tickers"] = sorted(cfg.ALL_TICKERS)
    meta["last_run_end_date"] = str(end_t.date())
    meta["saved_at"] = pd.Timestamp.now().isoformat()
    _save_news_cache(cache_df, cache_path, meta, log_label="GDELT Lauf Abschluss")
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


def fetch_news_sentiment(df, start=cfg.START_DATE, end=cfg.END_DATE):
    if not cfg.USE_NEWS_SENTIMENT:
        print("News sentiment disabled (cfg.USE_NEWS_SENTIMENT=False). Returning empty.")
        return pd.DataFrame()
    # Nur bei GKG_GCAM_EXPLORE_KEYS=True: BQ für *zusätzliche* GCAM-c*-Keys; sonst nur JSON/Must-Haves.
    allow_bq_gcam_scout = bool(getattr(cfg, "GKG_GCAM_EXPLORE_KEYS", False)) and (
        _c("NEWS_SOURCE", "bigquery") == "bigquery" and _c("BQ_USE_GKG_TABLE", True)
    )
    resolve_gkg_gcam_metric_keys(allow_bigquery_refresh=allow_bq_gcam_scout)
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
    return _fetch_news_sentiment_gdelt_api(df, start, end)
