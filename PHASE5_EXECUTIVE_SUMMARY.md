# Phase 5 Executive Summary

## Plain-English outcome

The typed neural-firmware architecture was the most reliable arithmetic
system in this experiment, but it did not pass the safety/preservation gates.

Across three independent training seeds and 400 never-used prompts per seed:

| Condition | Learned parameters | Correct | Seed-mean accuracy |
|---|---:|---:|---:|
| Untouched Qwen2.5-0.5B-Instruct | 0 | 68/400 | 17.0% |
| Ordinary learned adapter | 24,225 | 33/1,200 | 2.75% |
| Typed firmware with fixed parser | 24,225 | 1,200/1,200 | 100.0% |
| Matched-budget IGC-style architecture | 24,225 | 1/1,200 | 0.083% |
| Native-size IGC-style architecture | 597,819 | 1,084/1,200 | 90.33% |

Typed firmware exceeded native IGC by 9.67 percentage points. The
precommitted paired bootstrap 95% interval, resampling both prompt and
training-seed clusters, was +4.33 to +15.25 points. Typed firmware therefore
did **not** meet the deliberately symmetric “comparably reliable” rule, whose
entire interval had to lie within -5 to +5 points; it was more reliable, not
equivalent.

The precommitted parameter-efficiency rule **passed**. Typed firmware:

- was not less accurate than native IGC;
- used 24.68 times fewer learned parameters;
- had identical seed-mean negative-prompt preservation (96.25%).

This is a differentiated result, but its scope matters. Typed firmware
received operands from a fixed decimal parser. Native IGC learned operand
extraction from model residuals. The result supports a strong reliability and
parameter-efficiency advantage **when a fixed typed boundary is allowed**; it
does not show that the typed architecture has solved learned parsing more
efficiently.

## What the IGC-style comparison revealed

The native-size IGC-style architecture worked, but its learned input mapping
was the bottleneck:

- exact operand-register recovery: 1,083/1,200 (90.25%);
- arithmetic correctness when routing and registers were correct:
  1,083/1,083 (100%);
- end-to-end seed accuracies: 96.0%, 85.0%, and 90.0%.

Thus the frozen calculator and learned output mapping were reliable after
successful extraction. One additional response happened to equal the correct
sum despite incorrect registers. The 116 end-to-end errors relative to typed
firmware were extraction errors, not failures of the deterministic addition
cell.

The exactly matched 24,225-parameter IGC-style condition never recovered a
complete operand pair on confirmation (0/1,200 exact registers). Its single
correct final answer was incidental. This is evidence that the particular
matched allocation was too small for the chosen recurrent input mapper, not a
general impossibility result for IGC.

The ordinary adapter also failed to learn addition: 33/1,200 (2.75%), below
the untouched model's 17.0% single-run baseline. Adding the same number of
unstructured learned parameters was not sufficient.

## The important failure: routing safety

Every trained condition routed all 1,200 addition prompts correctly. However,
the common semantic router falsely activated on multiplication prompts:

| Condition | False routes | False-route rate | Token-exact preservation |
|---|---:|---:|---:|
| Typed firmware | 18/480 | 3.75% | 462/480 (96.25%) |
| Ordinary adapter | 18/480 | 3.75% | 468/480 (97.5%) |
| Matched IGC-style | 24/480 | 5.0% | 456/480 (95.0%) |
| Native IGC-style | 18/480 | 3.75% | 462/480 (96.25%) |

All typed/native false activations came from the unseen family “Compute the
product of *a* and *b*; provide digits only.” The router generalized the
request shape more strongly than the operation word.

Consequently, no trained condition passed the frozen routing requirement
(at least 90% positive recall and at most 2% false activation), and none
passed the 99% preservation requirement. This phase is not a compound safety
success even though typed arithmetic execution was perfect.

## Latency

Mean end-to-end positive-prompt latency was:

- typed firmware: 0.279 seconds;
- native IGC-style: 0.289 seconds;
- matched IGC-style: 0.289 seconds;
- ordinary adapter: 0.319 seconds;
- untouched base: 0.624 seconds.

Typed firmware was about 3.4% faster than native IGC on these prompts. The
base comparison is not a pure compute benchmark because the base generated
longer answers; the modified systems usually emitted only the exact integer.

## What this establishes

Within Qwen2.5-0.5B, addition, and this prompt distribution:

> A fixed-parser typed deterministic architecture delivered perfect
> three-seed arithmetic reliability with 24,225 learned parameters, while a
> learned-parser IGC-style architecture averaged 90.33% with 597,819 learned
> parameters and an equal-budget IGC-style mapper failed.

It does not establish superiority over Dietz and Klakow's original IGC
implementation. This study is an independent IGC-style comparison, not a
reproduction: the original code was unavailable, and its published setup used
Llama 3.1 8B, approximately 17 million learned module parameters, and four
operations.

It also does not establish general arithmetic, arbitrary-length calculation,
safe operation routing, or a learned parser for typed firmware.

## Recommended next step

Before expanding scope, repair the operation router with explicit
operation-word supervision, hard-negative multiplication families, and a
confidence-aware abstention rule, using fresh frozen negative families.

Then remove the fixed parser from the typed path while retaining explicit
digit registers and per-stage diagnostics. Only after that learned extraction
stage is stable should the study add subtraction, multiplication, and
division, replicate on a Llama-class model, and advance to the more original
board-game state-transition architecture.

## Files

- Frozen protocol: `PHASE5_CONFIRMATORY_PROTOCOL.md`
- Chronological record: `PHASE5_LAB_NOTEBOOK.md`
- Pilot/failure record: `PHASE5_PILOT_LOG.md`
- Raw per-prompt archive: `phase5_results/confirmation_raw_v1/`
- Statistical analysis: `phase5_results/confirmation_analysis_v1.json`
- Reproduction runner: `scripts/run_phase5_confirmation.py`
- Analysis runner: `scripts/analyze_phase5_confirmation.py`
- Technical paper: `paper_phase5/neural-firmware-versus-igc.pdf`
