# Phase 7 base comparison

This report compares untouched Qwen with all three Phase 7 implant seeds on
the complete frozen Audit 5 prompt set.

Regenerate the evidence and report with:

```bash
.venv/bin/python scripts/evaluate_phase7_base_comparison.py
.venv/bin/python scripts/build_phase7_base_comparison_report.py

node /Users/josiahwilson/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input phase7_base_comparison/artifact.json \
  --output phase7_base_comparison/report.html
```

The evaluation is a retrospective comparison on an already frozen prompt set,
not an independently preregistered result.

`source.sql` reproduces the report datasets directly from the retained JSON
record with SQLite's JSON functions.
