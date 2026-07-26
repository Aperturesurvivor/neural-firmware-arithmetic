# Phase 6 Pilot Log

This file records development-only architecture work before any confirmatory
protocol is frozen.

## 2026-07-26 — Initial design

- Replaced the proposed single-shot learned parser with a neural firmware
  controller that predicts three typed operand registers and zero, one, or
  two calculator calls.
- The two-call program is left-associative: first compute `A + B`, then reuse
  the same frozen cell to compute the intermediate result plus `C`.
- Operand extraction is learned from early Qwen residuals without character
  spans or digit positions at inference.
- Call count and operation intent are learned from the late block-24
  pre-normalization residual.
- Kept extraction, call classification, deterministic execution, residual
  output, and negative preservation as independent diagnostics.
- Defined development gates in `PHASE6_ARCHITECTURE_DRAFT.md`. Confirmation
  remains prohibited until those gates are met.

## 2026-07-26 — Pilot v1

- Trained a 1,598,681-parameter interface:
  - 1,531,222-parameter three-register mapper;
  - 57,603-parameter zero/one/two-call controller;
  - 9,856-parameter symbol-to-residual decoder.
- Training used 2,400 single-call positives, 2,400 two-call positives, and
  3,200 hard negatives. Development used 400/400/600 examples respectively.
- Ordered exact-register accuracy was:
  - single-call: 75.0%;
  - two-call: 96.25%;
  - all positives: 85.625%.
- The apparent single-call failure was localized to one entire family,
  `By adding {b} onto {a}...`: the learned mapper returned operands in textual
  order while the target file retained placeholder-name order. Because
  addition is commutative, every such “error” represents the same executable
  program. V2 must define register order by textual occurrence and report both
  ordered and operation-equivalent accuracy.
- The three-class controller faced a calibration tradeoff:
  - threshold 0.50: 96.0% positive call-count accuracy but 11.17% false calls;
  - threshold 0.99: 71.875% positive call-count accuracy and 0.667% false
    calls.
- The selected threshold therefore missed the positive development gate.
  Errors concentrated in subtraction wording and mixed-operation two-step
  prompts.
- On 40 single, 40 chained, and 40 negative integrated diagnostics:
  - 59/80 positive final answers were exact;
  - 46/46 examples with exact ordered registers and call count were exact;
  - 0/40 negatives activated.
- The 46/46 conditional result demonstrates that the same frozen cell can be
  reused for an exact intermediate calculation and exact final calculation,
  then return the answer through Qwen's residual/output path.
- The initially recorded 11/40 preservation figure is invalid: the base
  comparator used Transformers `generate` while the firmware path used the
  project's manual cached decoder. A matched manual forced-off check was
  subsequently 10/10 token-identical. V2 will use one generator for both
  paths.
- Reproducibility defect: the v1 mapper/controller sampling seeds were set
  inside their training functions, after module initialization. The checkpoint
  and raw result are retained, but v2 must seed before module construction.
- Raw record: `phase6_results/pilot_v1.json`.
- Checkpoint SHA-256:
  `258556622cb5d940a8f25e00031aade8bce014587358e821c63b62db67412e45`.
- Decision: retain v1 as a partial success; do not freeze confirmation.

## 2026-07-26 — Pilot v2

- Corrected register targets to textual operand order and fine-tuned the
  retained v1 mapper for 1,500 steps.
- Replaced the overloaded three-class controller with five explicit classes:
  no call, one addition, two additions, one unsupported operation, and an
  unsupported multi-operation request. Only the two ADD classes can activate
  firmware.
- Seeded module construction before v2 controller initialization.
- Exact development register recovery improved to:
  - single-call: 99.5%;
  - two-call: 100.0%;
  - all positives: 99.75%;
  - all digit/PAD slots: 99.964%.
- This passes the development extraction target and establishes a working
  learned residual-to-register interface without parser inputs at inference.
- The factorized controller improved operation diagnostics but retained a
  threshold tradeoff:
  - threshold 0.50: 95.125% positive call-count accuracy and 9.67% false
    calls;
  - selected threshold 0.97: 79.5% positive call-count accuracy and 1.0%
    false calls.
- False calls at threshold 0.50 were almost entirely the held-out mixed
  operation “add A to B, then divide by C.” Single-call count errors also
  concentrated in one word problem whose two narrative clauses resembled two
  computational calls.
- Integrated diagnostics:
  - 67/80 positive answers exact;
  - 80/80 positive operand programs exact;
  - 62/62 examples with a correct call decision exact;
  - 0/40 negative activations;
  - 40/40 negative outputs token-identical to a matched manual base decoder.
- V2 confirms that parsing, repeated deterministic execution, residual return,
  and route-off preservation work. Semantic call control remains below the
  frozen-development gate.
- Raw record: `phase6_results/pilot_v2.json`.
- Checkpoint SHA-256:
  `5c33a62fce82b6cf7bab99f7671029f0c9ceca84f804ed368de6afd8bfc12467`.
- Decision: retain v2; train the controller on the now-consumed development
  failures, then select its threshold on new calibration families and
  evaluate on a separate untouched pilot-validation family set.

## 2026-07-26 — Pilot v3

- Retained the v2 mapper and fine-tuned the late five-class controller on the
  consumed v1 development split. Selected a threshold on separate calibration
  families and evaluated on separate pilot-validation families.
- Register recovery on validation was 89.0% over positives:
  - single-call: 100.0%;
  - two-call: 78.0%.
- Chain extraction failures were concentrated in a new odometer narrative:
  39/50 of that family lost or corrupted the third register. This is a
  family-generalization failure, not a calculator failure.
- The calibration-selected threshold was 0.88. At that threshold:
  - positive call-count accuracy was 92.25%;
  - false-call rate was 10.208% (49/480);
  - 39/40 prompts that requested addition followed by division activated the
    addition firmware incorrectly.
- Integrated diagnostics on 40 single, 40 chained, and 40 negative prompts:
  - 70/80 positive final answers were exact;
  - 69/69 examples with exact registers and call decisions were exact;
  - 6/40 negatives activated;
  - 34/40 negative outputs remained token-identical to the matched base path.
- The controller therefore overfit prompt-family cues: a final-token residual
  did not reliably distinguish pure repeated addition from addition followed
  by another operation. V4 adds a sequence-aware early control head and fuses
  its logits with the late controller.
- Raw record: `phase6_results/pilot_v3.json`.
- Checkpoint SHA-256:
  `42cfc6d6ca6927267cebcc42ad229c00c02f9dfadcd0e7c0c71e8734f407654d`.
- Decision: retain v3 as a negative routing/generalization result; do not
  freeze confirmation.

## 2026-07-26 — Pilot v4

- Added a sequence-aware early control head and fused its logits with the late
  residual controller.
- Fine-tuned the mapper on consumed validation failures. On a new audit split,
  positive exact-register recovery reached 98.75% (100% single, 97.5% chain).
- Sequentially training the early controller on train, development, then
  validation caused family forgetting. Audit routing remained below gate:
  83.25% positive call-count accuracy and 6.67% false calls.
- Integrated diagnostics were 69/80 positive answers, 69/69 conditional
  exactness, 2/40 false calls, and 38/40 token-identical negatives.
- Raw record: `phase6_results/pilot_v4.json`.
- Checkpoint SHA-256:
  `869d71099a71a13dc7deca0bcb0a2ad4ece99280f9d925b4238a28483f8ff829`.
- Decision: retain v4; replace sequential head training with joint fused
  training over a shuffled union.

## 2026-07-26 — Pilot v5

- Jointly trained the early and late control heads on the union of all
  consumed pilot data while freezing the v4 digit mapper.
- On a new stress split, register recovery was 100%. Routing improved to
  91.25% positive call-count accuracy with 0/480 false calls at the
  calibration-selected threshold.
- Integrated diagnostics were 73/80 positive answers, 72/72 conditional
  exactness, 0/40 false calls, and 40/40 token-identical negatives.
- Error analysis separated semantic activation from program length. The
  register mapper already predicted whether a third operand existed exactly,
  while the controller sometimes confused one versus two calls. The next
  revision derives program length from learned third-register occupancy.
- Raw record: `phase6_results/pilot_v5.json`.
- Checkpoint SHA-256:
  `9c3c9024fa974e597192565a61fadc46bef50fcd89521ec4abd7e92d1317d8e9`.
- Decision: retain v5; change the runtime call-count source without changing
  learned weights.

## 2026-07-26 — Pilot v6

- Loaded v5 weights unchanged. The semantic controller now decides whether
  ADD may activate, while the predicted third-register occupancy chooses one
  versus two exact calls.
- On the final untouched development-gate split:
  - register recovery was 100% for single calls and 96.0% for chains;
  - positive call-count accuracy was 92.25%;
  - false-call rate was 0/480 at threshold 0.97;
  - controller class accuracy was 98.64%.
- Integrated diagnostics were 71/80 positive answers, 70/70 conditional
  exactness, 0/40 false calls, and 40/40 token-identical negatives.
- The exact frozen cell was reused for two calls and its first typed result was
  fed directly into the second call without text parsing. The output bridge
  returned exact final digits through Qwen's frozen normalization and LM head.
- Raw record: `phase6_results/pilot_v6.json`.
- Checkpoint SHA-256:
  `9c3c9024fa974e597192565a61fadc46bef50fcd89521ec4abd7e92d1317d8e9`.
  The hash matches v5 because v6 changed deterministic runtime policy, not
  learned tensors.
- Decision: this is a working fully neural-interface prototype, including
  repeated internal calculation, but it does not pass the preregistered
  extraction/routing gates. Do not evaluate frozen confirmation families yet.
