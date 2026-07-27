# Phase 7 base comparison

This report compares untouched Qwen with all three Phase 7 implant seeds on
the complete frozen Audit 5 prompt set.

Regenerate the evidence and report with:

```bash
.venv/bin/python scripts/evaluate_phase7_base_comparison.py
.venv/bin/python scripts/build_phase7_base_comparison_report.py
```

The evaluation is a retrospective comparison on an already frozen prompt set,
not an independently preregistered result.

`source.sql` reproduces the report datasets directly from the retained JSON
record with SQLite's JSON functions. The tracked `report.html` is the reviewed
portable rendering of `artifact.json`.
