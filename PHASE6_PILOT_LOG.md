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
