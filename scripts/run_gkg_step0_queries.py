"""
Führt die Reality-Check-Queries aus sql/gkg_mvp_step0_reality_check.sql aus.
Enthält u. a. GCAM 3a/3b/3c: Komma-Split, Top-Dimensionen, nur c*-Codes, ungeparste Segmente.

Voraussetzung: ADC oder GOOGLE_APPLICATION_CREDENTIALS, BigQuery-Billing.

 python scripts/run_gkg_step0_queries.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

from google.cloud import bigquery

TABLE = "`gdelt-bq.gdeltv2.gkg_partitioned`"


def _print_query_result(job: bigquery.job.QueryJob, max_rows: int = 200) -> None:
    rows = list(job.result(max_results=max_rows))
    if not rows:
        print("(keine Zeilen)")
        return
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


def main() -> None:
    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or "gedelt-calls"
    )
    client = bigquery.Client(project=project)
    sample_d = date.today() - timedelta(days=3)
    d0 = sample_d - timedelta(days=6)
    d1 = sample_d

    print(f"Projekt: {client.project}")
    print(f"Stichprobe-Tag: {sample_d} | 7-Tage-Fenster: {d0} … {d1}\n")

    # 0a INFORMATION_SCHEMA (öffentliches Dataset — oft ohne Extra-Rechte lesbar)
    q0a = """
    SELECT table_name, column_name, data_type
    FROM `gdelt-bq.gdeltv2.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'gkg_partitioned'
      AND column_name IN ('GCAM', 'V2Tone', 'V2Themes', 'V2Organizations')
    ORDER BY column_name
    """
    print("=== 0a) INFORMATION_SCHEMA (Spaltentypen) ===")
    try:
        _print_query_result(client.query(q0a), max_rows=20)
    except Exception as e:
        print(f"Fehler (ggf. keine Leserechte auf gdelt-bq.INFORMATION_SCHEMA): {e}")
    print()

    q1 = f"""
    SELECT
      DATE(_PARTITIONTIME) AS partition_date,
      DATE AS gkg_date_int,
      DocumentIdentifier,
      SourceCommonName,
      ARRAY_LENGTH(SPLIT(IFNULL(V2Tone, ''), ',')) AS v2tone_parts,
      LENGTH(V2Themes) AS v2themes_len,
      LENGTH(V2Organizations) AS v2orgs_len,
      LENGTH(CAST(GCAM AS STRING)) AS gcam_len,
      SUBSTR(CAST(GCAM AS STRING), 1, 400) AS gcam_prefix,
      SUBSTR(V2Organizations, 1, 300) AS v2orgs_prefix,
      SUBSTR(V2Themes, 1, 300) AS v2themes_prefix
    FROM {TABLE}
    WHERE DATE(_PARTITIONTIME) = DATE '{sample_d.isoformat()}'
    LIMIT 100
    """
    print("=== 1) Stichprobe (100 Zeilen, gcam_prefix gekürzt) ===")
    try:
        rows = list(client.query(q1).result(max_results=100))
        print(f"Zeilen: {len(rows)}")
        for i, row in enumerate(rows[:3]):
            print(f"  --- Zeile {i} ---")
            print(f"  gcam_len={row['gcam_len']} gcam_prefix={row['gcam_prefix']!r}")
        if rows:
            keys = list(rows[0].keys())
            print(" | ".join(keys))
            for r in rows[:5]:
                print(" | ".join(str(r[k]) for k in keys))
    except Exception as e:
        print(f"Fehler: {e}")
    print()

    q2 = f"""
    SELECT
      DATE(_PARTITIONTIME) AS d,
      COUNT(*) AS n_rows,
      COUNTIF(V2Themes IS NULL OR V2Themes = '') AS themes_empty,
      COUNTIF(V2Organizations IS NULL OR V2Organizations = '') AS orgs_empty,
      COUNTIF(GCAM IS NULL OR CAST(GCAM AS STRING) = '') AS gcam_empty,
      SAFE_DIVIDE(
        COUNTIF(GCAM IS NOT NULL AND CAST(GCAM AS STRING) != ''),
        COUNT(*)
      ) AS gcam_fill_rate
    FROM {TABLE}
    WHERE DATE(_PARTITIONTIME) BETWEEN DATE '{d0.isoformat()}' AND DATE '{d1.isoformat()}'
    GROUP BY 1
    ORDER BY 1
    """
    print("=== 2) 7-Tage Qualität ===")
    try:
        _print_query_result(client.query(q2), max_rows=30)
    except Exception as e:
        print(f"Fehler: {e}")
    print()

    _gcam_base = rf"""
    WITH raw AS (
      SELECT CAST(GCAM AS STRING) AS gcam_s
      FROM {TABLE}
      WHERE DATE(_PARTITIONTIME) = DATE '{sample_d.isoformat()}'
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
        SAFE_CAST(REGEXP_EXTRACT(seg, r':(-?[0-9]+(?:\.[0-9]+)?)$') AS FLOAT64) AS dim_val
      FROM segs
      WHERE seg != ''
        AND REGEXP_CONTAINS(seg, r'^[a-zA-Z][a-zA-Z0-9_.]*:-?[0-9]+(?:\.[0-9]+)?$')
    )
    """
    q3a = (
        _gcam_base
        + """
    SELECT
      dim_key,
      COUNT(*) AS n_segments,
      ROUND(SUM(dim_val), 2) AS sum_weight,
      ROUND(AVG(dim_val), 4) AS avg_weight
    FROM parsed
    WHERE dim_key IS NOT NULL
    GROUP BY 1
    ORDER BY n_segments DESC
    LIMIT 60
    """
    )
    q3b = (
        _gcam_base
        + """
    SELECT
      dim_key,
      COUNT(*) AS n_segments,
      ROUND(SUM(dim_val), 2) AS sum_weight,
      ROUND(AVG(dim_val), 4) AS avg_weight
    FROM parsed
    WHERE dim_key IS NOT NULL
      AND STARTS_WITH(dim_key, 'c')
    GROUP BY 1
    ORDER BY n_segments DESC
    LIMIT 60
    """
    )
    q3c = rf"""
    WITH raw AS (
      SELECT CAST(GCAM AS STRING) AS gcam_s
      FROM {TABLE}
      WHERE DATE(_PARTITIONTIME) = DATE '{sample_d.isoformat()}'
        AND GCAM IS NOT NULL
        AND CAST(GCAM AS STRING) != ''
      LIMIT 50000
    ),
    segs AS (
      SELECT TRIM(segment) AS seg
      FROM raw, UNNEST(SPLIT(gcam_s, ',')) AS segment
      WHERE TRIM(segment) != ''
    )
    SELECT
      seg,
      COUNT(*) AS n
    FROM segs
    WHERE NOT REGEXP_CONTAINS(seg, r'^[a-zA-Z][a-zA-Z0-9_.]*:-?[0-9]+(?:\.[0-9]+)?$')
    GROUP BY 1
    ORDER BY n DESC
    LIMIT 30
    """
    print("=== 3a) GCAM Top-Dimensionen (Komma-Split, alle gültigen key:value) ===")
    try:
        _print_query_result(client.query(q3a), max_rows=60)
    except Exception as e:
        print(f"Fehler: {e}")
    print()
    print("=== 3b) GCAM Top-Dimensionen (nur c*-Codes) ===")
    try:
        _print_query_result(client.query(q3b), max_rows=60)
    except Exception as e:
        print(f"Fehler: {e}")
    print()
    print("=== 3c) GCAM nicht geparste Segmente (Debug) ===")
    try:
        _print_query_result(client.query(q3c), max_rows=30)
    except Exception as e:
        print(f"Fehler: {e}")
    print()

    q4 = f"""
    SELECT
      COUNTIF(V2Organizations LIKE '%;%') AS rows_with_semicolon,
      COUNT(*) AS rows_nonempty_org
    FROM {TABLE}
    WHERE DATE(_PARTITIONTIME) = DATE '{sample_d.isoformat()}'
      AND V2Organizations IS NOT NULL
      AND V2Organizations != ''
    """
    print("=== 4) V2Organizations Semikolon ===")
    try:
        _print_query_result(client.query(q4), max_rows=10)
    except Exception as e:
        print(f"Fehler: {e}")


if __name__ == "__main__":
    main()
