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
