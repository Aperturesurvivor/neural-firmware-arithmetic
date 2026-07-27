# Chart map

## Exact response accuracy

- Question: How often does each condition return exactly the requested numeral?
- Takeaway: untouched Qwen is 1/60; implant seeds are 57/60, 58/60, and 58/60.
- Form: single-series horizontal bar comparison.
- Fields: condition, accuracy, correct, attempts, metric.
- Palette: one semantic series; no redundant legend or color encoding.
- Source: `phase7_results/base_vs_implant_audit5.json`.

## Token-budget sensitivity

- Question: Does a larger generation budget erase the implant advantage?
- Takeaway: base last-number recovery rises to 27/60 at 64 tokens, while the
  implant remains 173/180 on the stricter exact-response metric.
- Form: single-series horizontal bar comparison.
- Fields: condition, accuracy, correct, attempts.
- Palette: one semantic series; no redundant legend or color encoding.
- Source: `phase7_results/base_vs_implant_audit5.json`.

The second chart intentionally labels its mixed metric basis. It is a
sensitivity comparison, not the controlled primary endpoint.
