-- Reproducible SQLite queries for the report datasets.
-- Run from the repository root:
--   sqlite3 -header -column :memory: < phase7_base_comparison/source.sql

-- Headline metrics.
WITH source AS (
  SELECT readfile('phase7_results/base_vs_implant_audit5.json') AS document
)
SELECT
  json_extract(document, '$.summary.base.format_exact') / 60.0
    AS base_exact,
  json_extract(
    document,
    '$.summary.base_extended_sensitivity.mathematical_exact'
  ) / 60.0 AS base_extended,
  json_extract(
    document,
    '$.summary.implants.pooled_mathematical_exact'
  ) / 180.0 AS implant_exact,
  json_extract(
    document,
    '$.summary.negative_preservation.pooled_token_identical_to_base'
  ) / 180.0 AS negative_preservation
FROM source;

-- Primary exact-response comparison.
WITH source AS (
  SELECT readfile('phase7_results/base_vs_implant_audit5.json') AS document
)
SELECT
  'Untouched Qwen' AS condition,
  json_extract(document, '$.summary.base.format_exact') / 60.0 AS accuracy,
  json_extract(document, '$.summary.base.format_exact') AS correct,
  60 AS attempts
FROM source
UNION ALL
SELECT
  'Implant seed 13,201',
  json_extract(
    document,
    '$.summary.implants.per_seed."13201".format_exact'
  ) / 60.0,
  json_extract(
    document,
    '$.summary.implants.per_seed."13201".format_exact'
  ),
  60
FROM source
UNION ALL
SELECT
  'Implant seed 13,202',
  json_extract(
    document,
    '$.summary.implants.per_seed."13202".format_exact'
  ) / 60.0,
  json_extract(
    document,
    '$.summary.implants.per_seed."13202".format_exact'
  ),
  60
FROM source
UNION ALL
SELECT
  'Implant seed 13,203',
  json_extract(
    document,
    '$.summary.implants.per_seed."13203".format_exact'
  ) / 60.0,
  json_extract(
    document,
    '$.summary.implants.per_seed."13203".format_exact'
  ),
  60
FROM source;

-- Token-budget sensitivity.
WITH source AS (
  SELECT readfile('phase7_results/base_vs_implant_audit5.json') AS document
)
SELECT
  'Base, 8-token last-number score' AS condition,
  json_extract(document, '$.summary.base.mathematical_exact') / 60.0
    AS accuracy,
  json_extract(document, '$.summary.base.mathematical_exact') AS correct,
  60 AS attempts
FROM source
UNION ALL
SELECT
  'Base, 64-token last-number score',
  json_extract(
    document,
    '$.summary.base_extended_sensitivity.mathematical_exact'
  ) / 60.0,
  json_extract(
    document,
    '$.summary.base_extended_sensitivity.mathematical_exact'
  ),
  60
FROM source
UNION ALL
SELECT
  'Implant, pooled exact response',
  json_extract(
    document,
    '$.summary.implants.pooled_mathematical_exact'
  ) / 180.0,
  json_extract(
    document,
    '$.summary.implants.pooled_mathematical_exact'
  ),
  180
FROM source;

-- All positive prompt outputs. The report generator adds reader-facing status
-- prefixes to the same fields.
WITH
source AS (
  SELECT readfile('phase7_results/base_vs_implant_audit5.json') AS document
),
rows AS (
  SELECT CAST(key AS INTEGER) + 1 AS row_index, value
  FROM source, json_each(document, '$.rows')
)
SELECT
  row_index,
  json_extract(value, '$.split') AS split,
  json_extract(value, '$.prompt') AS prompt,
  json_extract(value, '$.answer') AS expected,
  json_extract(value, '$.base.generated_text') AS base_8,
  json_extract(value, '$.base_extended.generated_text') AS base_64,
  json_extract(value, '$.implants."13201".generated_text') AS seed_13201,
  json_extract(value, '$.implants."13202".generated_text') AS seed_13202,
  json_extract(value, '$.implants."13203".generated_text') AS seed_13203
FROM rows
WHERE json_extract(value, '$.route_label') = 1
ORDER BY row_index;

-- All adversarial negative outputs and their three-seed preservation status.
WITH
source AS (
  SELECT readfile('phase7_results/base_vs_implant_audit5.json') AS document
),
rows AS (
  SELECT CAST(key AS INTEGER) + 1 AS row_index, value
  FROM source, json_each(document, '$.rows')
)
SELECT
  row_index,
  json_extract(value, '$.prompt') AS prompt,
  json_extract(value, '$.base.generated_text') AS base_output,
  json_extract(value, '$.implants."13202".generated_text') AS implant_output,
  CASE
    WHEN json_extract(
      value,
      '$.implants."13201".token_identical_to_base'
    ) = 1
    AND json_extract(
      value,
      '$.implants."13202".token_identical_to_base'
    ) = 1
    AND json_extract(
      value,
      '$.implants."13203".token_identical_to_base'
    ) = 1
    THEN 'Yes'
    ELSE 'No'
  END AS all_three_token_identical
FROM rows
WHERE json_extract(value, '$.route_label') = 0
ORDER BY row_index;
