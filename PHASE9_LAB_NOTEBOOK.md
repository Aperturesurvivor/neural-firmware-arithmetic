# Phase 9 Lab Notebook

## Scope

Phase 9 tests whether semantic hard-contrast training can repair the input
interface of the fixed TinyLlama deterministic-neuron implant. The calculator,
layer, selected coordinates, result decoder, base weights, runtime state, and
architectural learned-parameter count remain fixed.

## 2026-07-27 — repository and evidence audit

- Started from clean public commit `1ef5a7a`.
- Confirmed Phase 8's seven positive failures were one route-off prompt and six
  operand-handshake failures.
- Confirmed conditional computation and decoding were 53/53 for every Phase 8
  seed.
- Chose generic versus hard-contrast continuation as the paired curriculum
  comparison.

## 2026-07-27 — data construction

- Added deterministic Phase 9 generators and tests.
- Generic continuation: 1,200 positives and 1,200 negatives from the previous
  training-family distribution.
- Initial hard continuation: 1,200 positives and 1,200 negatives from new
  families targeting the Phase 8 failure modes.
- Shared development: 240 positives and 480 negatives.
- Sealed confirmation: 100 positives and 200 negatives across 70 new families.
- Enforced exact prompt, family, and operand-pair disjointness as applicable.

## 2026-07-27 — feature collection

- Model revision and layer matched Phase 8.
- Collected sequence and first-generation-step representations from the
  unmodified base model.
- Sampled up to 12 ordinary sequence tokens per example while retaining all
  supervised route and operand tokens.
- Initial cache SHA-256:
  `59f006c87aa9ba7aa1c4c6a0e613b59d8bdc314e4c17ca9531617171ac99faa9`.

## 2026-07-27 — development v1

Configuration:

- 1,500 interface steps, learning rate 0.0005;
- 2,500 route-row steps, learning rate 0.0005;
- maximum development false-positive rate 0.01;
- Phase 8 digit threshold 0.9.

Balanced generation subset:

| Condition | Exact positives | Exact operands | False routes | Preserved negatives | Ablation exact |
|---|---:|---:|---:|---:|---:|
| Generic | 32/60 | 37/60 | 0/72 | 72/72 | 0/60 |
| Initial hard | 47/60 | 52/60 | 0/72 | 72/72 | 0/60 |

The generic curriculum became overly conservative. Ten of the hard
condition's thirteen failures came from two distractor-positive families whose
prompts ended immediately after the second operand.

## 2026-07-27 — route weighting diagnostic

Repeated the hard route set at weights 1, 2, 4, and 8 and added distractor
positive weights up to 8. Every configuration retained the same 83.33%
development route recall at zero false positives. Simple sample weighting did
not change the linear frontier.

## 2026-07-27 — family-diversity repair

Added four family-disjoint hard-positive constructions that end immediately
after the second operand. Kept the hard-positive count fixed by changing from
20 families × 60 examples to 24 families × 50 examples.

Refreshed hard features only. Revised cache SHA-256:
`555a5dc78c18404a5d9feca1e9fcad7ac2a40554d1c480cf97265d86a5a5ea55`.

The revised hard condition reached:

- 54/60 exact positives;
- 60/60 positive route predictions;
- 54/60 exact operands;
- 0/72 false routes;
- 72/72 token preservation;
- 0/60 exact under result ablation.

All six remaining errors were operand-interface failures. Lowering the
digit-confidence threshold to 0.8 recovered one missing low-confidence digit,
for an inferred 55/60 on the same development subset. Thresholds below 0.8
provided no additional recovery.

## 2026-07-27 — rejected extended optimization

Tested role/digit loss weights through 8/4. At 1,500 steps these converged to
the same development classifications. A 2,500-step 5/3 candidate slightly
improved token-level operand metrics but remained at 54/60 end-to-end exact,
with 0/72 false routes. It was rejected in favor of the simpler 1,500-step
schedule.

## Confirmation boundary

No Phase 9 confirmation prompt has been run through a model at this point.
The exact prompt manifest, implementation commit, gates, and protocol hashes
must be frozen before confirmatory training.

## 2026-07-27 — protocol freeze

- Committed the implementation and all retained development outcomes at
  `52f7e71ebb3be26ee002fc247a16a84fc482e24f`.
- Froze 300 unique confirmation prompts (100 positive and 200 negative) with
  canonical row SHA-256
  `ed15e17692c02552ce9895464a66d79b7b186e5bb03d26b4cf0d8f8a90ff3f4a`.
- Committed the prompt manifest separately at `6f50db9`.
- No confirmation prompt had been evaluated before this freeze.

## 2026-07-27 — frozen interface training

- Trained generic and hard-contrast interfaces for seeds 15,201, 15,202, and
  15,203 using the frozen schedule.
- Each run updated only the 32,768 input-interface weights.
- The 24,576 learned result-decoder weights were guarded unchanged.
- The calculator retained zero learned parameters.
- Wrote checkpoint hashes and complete training diagnostics to
  `phase9_results/confirmatory_interface_training.json`.

## 2026-07-27 — confirmation infrastructure retry

The first sealed evaluator process was terminated by the execution environment
before it wrote `confirmation.json` or any outcome summary. The prompt
manifest, checkpoints, evaluation code, metrics, and gates were unchanged.
The identical command was restarted in a detached local terminal with a
durable ignored log. This is recorded as an infrastructure retry, not a new
experimental condition. Post-hoc analysis and figure scripts were added while
the sealed evaluator ran; they do not alter or inspect model execution.

## 2026-07-27 — sealed confirmation

The detached evaluator completed all 4,200 prompt-condition generations in
4,082 seconds and wrote the raw JSON and flat CSV before any outcome-dependent
analysis.

Primary hard-condition results:

| Metric | Seed 15,201 | Seed 15,202 | Seed 15,203 |
|---|---:|---:|---:|
| Exact additions | 61/100 | 61/100 | 58/100 |
| Exact operands | 85/100 | 84/100 | 81/100 |
| Positive route predictions | 76/100 | 76/100 | 76/100 |
| Active positive routes | 72/100 | 71/100 | 69/100 |
| False routes | 12/200 | 12/200 | 11/200 |
| Token-exact preservation | 188/200 | 188/200 | 189/200 |
| Exact after result ablation | 0/100 | 0/100 | 0/100 |
| Paired causal losses | 61/100 | 61/100 | 58/100 |

Comparison results:

- untouched base: 0/100 exact;
- matched adapters: 21/100, 14/100, and 20/100;
- unchanged Phase 8 implants: 67/100, 68/100, and 68/100;
- generic continuation: 59/100, 63/100, and 58/100;
- generic false routes: 8/200 in every seed.

The hard and generic means were both exactly 60.0%. Hard continuation
improved operand registers and several distractor families, but it routed off
24/100 additions in every seed, underperformed the unchanged Phase 8 implant,
and produced more false routes than generic continuation. The compound frozen
protocol failed.

## 2026-07-27 — disclosed gate bookkeeping bug

The frozen evaluator marked the conditional calculator-and-decode gate false.
Its implementation compared total exact trajectories with the number of
active-route, exact-string-operand examples. Seeds 15,202 and 15,203 each had
one additional exact trajectory from the numerically equivalent but
string-nonidentical register `047 + 150`, so the counts differed.

Row-level recomputation of the criterion as written in the protocol found:

- seed 15,201: 61/61 exact trajectories and decoded answers;
- seed 15,202: 60/60;
- seed 15,203: 57/57.

The raw false flag is retained in `confirmation.json`. The post-hoc analysis
records both the raw implementation flag and the corrected protocol-defined
pass. The compound verdict remains a failure because the other five gates
fail independently.

## 2026-07-27 — integrity audit

The pre-report completion audit passed every applicable check:

- all evaluated prompt fields exactly matched the frozen 300-row manifest;
- the canonical prompt hash matched;
- all expected condition/seed records and 300 CSV rows were present;
- all six Phase 9 checkpoint and source-checkpoint hashes matched;
- every checkpoint contained exactly 32,768 input and 24,576 result weights;
- every Phase 9 result tensor was bit-identical to its Phase 8 source;
- the calculator retained zero learned parameters;
- analysis reproduced the unchanged compound failure.

## 2026-07-27 — report and final verification

- Compiled the 11-page scientific report and generated four analysis figures.
- Ran the complete test suite: 90 tests passed.
- Ran Python bytecode checks for every Phase 9 training, evaluation, analysis,
  figure, freeze, and verification script.
- Ran the completion verifier with final-reader-artifact checks enabled. Every
  prompt, checkpoint, parameter-count, result-column, CSV, analysis,
  discrepancy-disclosure, figure, summary, and report check passed.
- The final audit is recorded in `phase9_results/completion_audit.json`.
