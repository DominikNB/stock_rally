-- =============================================================================
-- GKG MVP — Schritt 0: „Reality-Check“ (BigQuery)
-- =============================================================================
-- Ziel vor jedem neuen News-Feature-Design:
--   • Partition wirklich einschränken (_PARTITIONTIME / DATE), sonst Voll-Scan + hohe Kosten.
--   • Prüfen, ob GCAM, V2Organizations, V2Themes in *eurer* Nutzung gefüllt und parsebar sind.
--   • Daraus eine kurze Liste „5–8 stabile Metriken“ ableiten — alles andere weglassen.
--
-- Tabelle wie in lib/stock_rally_v10/config.py: GDELT_BQ_EVENTS_TABLE
-- Projekt/Billing: eigenes GCP-Projekt mit BigQuery (z. B. cfg.BQ_PROJECT_ID).
--
-- Hinweis: _PARTITIONTIME ist TIMESTAMP. Vergleich mit STRING-Literal "2026-04-10" ist unzuverlässig;
--          nutze DATE(_PARTITIONTIME) oder TIMESTAMP('2026-04-10 00:00:00 UTC').
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 0a) Spaltentyp GCAM klären (einmal im Dataset gdelt-bq ausführen, falls erlaubt)
-- -----------------------------------------------------------------------------
-- SELECT table_name, column_name, data_type
-- FROM `gdelt-bq.gdeltv2.INFORMATION_SCHEMA.COLUMNS`
-- WHERE table_name = 'gkg_partitioned' AND column_name IN ('GCAM', 'V2Tone', 'V2Themes', 'V2Organizations')
-- ORDER BY column_name;

-- -----------------------------------------------------------------------------
-- 1) Stichprobe: Rohzeilen eines Tages (visuell GCAM / Orgs / Themes)
-- -----------------------------------------------------------------------------
-- Ersetze @sample_date durch einen Tag, für den ihr Daten erwartet (z. B. jüngster vollständiger Tag).
DECLARE sample_date DATE DEFAULT DATE '2026-04-01';

SELECT
  DATE(_PARTITIONTIME) AS partition_date,
  DATE AS gkg_date_int,
  DocumentIdentifier,
  SourceCommonName,
  -- Längen statt voller TEXT (günstiger in der Konsole)
  ARRAY_LENGTH(SPLIT(IFNULL(V2Tone, ''), ',')) AS v2tone_parts,
  LENGTH(V2Themes) AS v2themes_len,
  LENGTH(V2Organizations) AS v2orgs_len,
  LENGTH(CAST(GCAM AS STRING)) AS gcam_len,
  SUBSTR(CAST(GCAM AS STRING), 1, 400) AS gcam_prefix,
  SUBSTR(V2Organizations, 1, 300) AS v2orgs_prefix,
  SUBSTR(V2Themes, 1, 300) AS v2themes_prefix
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE DATE(_PARTITIONTIME) = sample_date
LIMIT 100;

-- -----------------------------------------------------------------------------
-- 2) Sieben-Tage-Qualität: NULL / Leer / Deckung (pro Kalendertag)
-- -----------------------------------------------------------------------------
DECLARE d0 DATE DEFAULT DATE '2026-03-25';
DECLARE d1 DATE DEFAULT DATE '2026-03-31';

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
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE DATE(_PARTITIONTIME) BETWEEN d0 AND d1
GROUP BY 1
ORDER BY 1;

-- -----------------------------------------------------------------------------
-- 3a) GCAM: Komma-Segmente (key:value, Ganzzahl) — z. B. wc:468, c12.1:22
-- -----------------------------------------------------------------------------
-- SPLIT nach Komma; gültiges Segment: key:value mit Ganzzahl oder Dezimal (z. B. c12.1:22, v20.12:-0.5)
-- Ersetze sample_date durch den Partitionstag.

DECLARE sample_date DATE DEFAULT DATE '2026-04-01';

WITH raw AS (
  SELECT CAST(GCAM AS STRING) AS gcam_s
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE(_PARTITIONTIME) = sample_date
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
SELECT
  dim_key,
  COUNT(*) AS n_segments,
  ROUND(SUM(dim_val), 2) AS sum_weight,
  ROUND(AVG(dim_val), 4) AS avg_weight
FROM parsed
WHERE dim_key IS NOT NULL
GROUP BY 1
ORDER BY n_segments DESC
LIMIT 60;

-- -----------------------------------------------------------------------------
-- 3b) GCAM: Top-Dimensionen nur c*-Codes (GDELT-GCAM-Kategorien; Mapping → Codebook)
-- -----------------------------------------------------------------------------

DECLARE sample_date DATE DEFAULT DATE '2026-04-01';

WITH raw AS (
  SELECT CAST(GCAM AS STRING) AS gcam_s
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE(_PARTITIONTIME) = sample_date
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
LIMIT 60;

-- -----------------------------------------------------------------------------
-- 3c) GCAM: häufigste nicht geparste Segmente (Debug)
-- -----------------------------------------------------------------------------

DECLARE sample_date DATE DEFAULT DATE '2026-04-01';

WITH raw AS (
  SELECT CAST(GCAM AS STRING) AS gcam_s
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE DATE(_PARTITIONTIME) = sample_date
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
LIMIT 30;

-- -----------------------------------------------------------------------------
-- 4) V2Organizations: Trennzeichen-Stichprobe (meist „;“ laut GDELT-Doku)
-- -----------------------------------------------------------------------------
DECLARE sample_date DATE DEFAULT DATE '2026-04-01';

SELECT
  COUNTIF(V2Organizations LIKE '%;%') AS rows_with_semicolon,
  COUNT(*) AS rows_nonempty_org
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE DATE(_PARTITIONTIME) = sample_date
  AND V2Organizations IS NOT NULL
  AND V2Organizations != '';
