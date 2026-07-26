# Phase 4 Lab Notebook

Chronological, replication-oriented record for the natural-language invocation
and same-prompt baseline experiment. Times use America/Los_Angeles.

## 2026-07-25 — Authorization and measurement objective

- Josiah instructed Codex to perform the proposed next experiment.
- The central question is no longer only whether the deterministic state can
  survive inside the transformer. Phase 3 answered that successfully.
- The new primary question is the percentage-point increase in end-to-end
  arithmetic skill relative to the untouched model on identical,
  naturally-worded prompts.
- The experiment will report both final-number mathematical correctness and
  strict exact-format correctness so explanatory prose does not create an
  artificial zero for the base model.
- A full technical report, raw prediction archive, hashes, figures, and
  replication notes are required before completion.

## 2026-07-25 — Initial architecture decision

- Retained Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, insertion after block 6,
  residual strength 64, frozen base weights, and the phase-3 typed addition
  cell.
- Replaced the special register grammar with ordinary contiguous decimal
  numbers and multiple natural wording families.
- Separated two questions that should not be conflated:
  - a fixed boundary identifies exactly two candidate decimal spans;
  - a learned linear router at block 6 decides whether the request semantically
    asks for addition.
- The fixed boundary does not select an operation. Negation, quotation,
  subtraction, multiplication, comparison, concatenation, and explanation
  prompts contain the same kind of numeric spans and must remain inactive.
- Router decisions are immutable after the initial request forward pass,
  preserving the sequence-level safety property established in phase 3.
- The deterministic and learned-control conditions each contain exactly
  10,753 learned parameters:
  - 897 shared router parameters;
  - either a 9,856-parameter symbol codebook or an exactly
    9,856-parameter rank-five residual adapter.
- Added `PHASE4_ARCHITECTURE_DRAFT.md`; this is a pilot design, not a frozen
  protocol.

## 2026-07-25 — Implementation before first pilot

- Added `semantic_data.py` with disjoint train, held-out simple addition,
  held-out word-problem, training-negative, and held-out-negative wording
  families.
- Added explicit final-integer and exact-format scoring.
- Added `semantic_firmware.py` with:
  - a learned request-level router;
  - a wrapper containing the frozen typed addition cell;
  - a fixed typed-number input boundary;
  - a matched learned-control wrapper;
  - learned, forced-on, and forced-off routing modes;
  - a latched route decision and internal intervention support.
- Added `semantic_training.py` for route-feature collection, router fitting,
  teacher-forced internal/control fitting, and cached autoregressive
  evaluation.
- Added six focused tests covering family separation, operand recovery,
  scoring, parameter counts, exact typed plans, injection, and route-off
  identity. All six passed; Ruff passed before the first model pilot.

## 2026-07-25 — Pilot v1 defect caught before scoring

- The first pilot launch stopped before producing any evaluation result because
  a no-carry training batch allocated its symbol axis from target-token length
  while the exact cell always reserves `max(operand_width) + 2` slots.
- Corrected batch construction to allocate the cell's full static capacity.
- Added a regression test specifically covering a no-carry batch. The focused
  tests and Ruff passed after the correction.
- No pilot score existed before this implementation repair, so the repair could
  not have been selected in response to evaluation performance.

## 2026-07-25 — Semantic pilot v1

- Trained the linear block-6 router on 600 positive and 600 negative examples
  from the training wording families. Trained the internal decoder for 120
  steps and the parameter-matched learned control for 500 steps.
- The held-out router set contained 200 positive prompts (100 simple
  paraphrases and 100 word problems) and 200 negative prompts.
- Router accuracy was 63.0%, with a 38.0% true-positive rate and a 12.0%
  false-positive rate. This is below the safety and generalization threshold
  needed for a confirmatory experiment.
- The forced-on internal condition scored 12/12 mathematically and 12/12 in
  exact format on every one of the five positive evaluation splits. This
  isolates the deterministic arithmetic circuit as successful on ordinary
  contiguous decimal inputs and natural-language prompts.
- The untouched base model's mathematical accuracy was:
  - 7/12 (58.3%) on seen wording with 1–4 digit operands;
  - 6/12 (50.0%) on held-out wording with 1–4 digit operands;
  - 3/12 (25.0%) on held-out wording with 5–8 digit operands;
  - 0/12 (0.0%) on held-out wording with 9–12 digit operands;
  - 1/12 (8.3%) on held-out word problems with 5–8 digit operands.
- The learned-router internal condition improved those respective scores to
  12/12, 11/12, 6/12, 3/12, and 2/12. Its remaining arithmetic failures tracked
  router false negatives rather than incorrect results produced by an active
  exact circuit.
- The matched learned control scored 10/12, 7/12, 2/12, 0/12, and 1/12 on the
  same splits.
- On 24 held-out negative prompts, 19 outputs were token-identical to the base.
  All five changes corresponded to false route activations. Four were
  subtraction prompts using the unseen phrase "find the difference"; one was
  a non-calculation repeat instruction near the decision threshold.
- Raw results are stored in `phase4_results/semantic_pilot_v1.json`; pilot
  checkpoints are stored in the ignored
  `phase4_artifacts/semantic_pilot_v1/` directory.
- Decision: do not freeze the confirmatory protocol. Run a controlled router
  architecture sweep on only these already-consumed pilot wordings, then train
  against a broader development family set. A fresh family set will remain
  untouched for confirmation.

## 2026-07-25 — Router architecture sweep

- Swept request-residual depth 6, 12, 18, and 24; final-token versus
  mean-plus-final features; and linear, width-16, and width-64 classifiers.
- The sweep used only the already-consumed pilot-v1 train/evaluation wording.
- Block depth dominated classifier width. The best block-6 result was 66.8%
  accuracy, while block 24 reached 96.7%.
- The best compact option was a width-16 MLP on the final request token after
  block 24: 96.0% accuracy, 98.5% true-positive rate, and 6.5%
  false-positive rate before broadening the development training set.
- Raw sweep results are in
  `phase4_results/router_architecture_sweep.json`.

## 2026-07-25 — Router pilot v2

- Expanded router training to 2,400 positive and 2,400 negative prompts across
  34 positive and 32 negative development families. This included the
  already-consumed pilot-v1 holdouts plus new adversarial development
  families.
- Trained the block-24 896-to-16-to-1 router for 2,000 steps.
- Evaluated it on 800 positive and 800 negative examples from 16 entirely new
  development families.
- Selected threshold 0.76 on the development set by maximizing true-positive
  rate subject to false-positive rate no greater than 1%.
- At the selected threshold: 1,584/1,600 correct (99.0%), 792/800 positive
  activations (99.0%), and 8/800 false activations (1.0%).
- Raw results: `phase4_results/router_pilot_v2.json`.

## 2026-07-25 — Complete semantic pilot v2

- Installed the frozen router candidate and deterministic unit after block 24.
  Trained only the 9,856-parameter symbol codebook for 180 steps.
- Trained the exactly matched rank-five residual control payload for 600 steps.
  Both interfaces contain 24,225 learned parameters.
- On four 20-example development splits, mathematical correctness was:
  - untouched base: 30%, 5%, 0%, 0%;
  - matched learned control: 5%, 0%, 0%, 0%;
  - learned-router deterministic internal: 95%, 85%, 100%, 100%;
  - oracle-route deterministic internal: 100% on every split;
  - unit forced off: exactly the base scores.
- The splits respectively covered one- to four-digit simple prompts, five- to
  eight-digit simple prompts, nine- to twelve-digit simple prompts, and
  five- to eight-digit word problems.
- Every active deterministic answer was mathematically and format exact.
  Learned-router misses, not arithmetic execution errors, account for all five
  internal failures.
- On 40 new development negatives, there were zero false activations and all 40
  internal outputs were token-identical to base.
- Raw results: `phase4_results/semantic_pilot_v2.json`.
- This passed the development gate. Froze
  `PHASE4_CONFIRMATORY_PROTOCOL.md` before any confirmation inference, including
  artifact hashes, seeds, sample sizes, metrics, and success criteria.

## 2026-07-25/26 — Frozen confirmation

- Committed the frozen protocol, source, pilot results, and confirmatory
  family definitions before inference. Added and separately committed the
  runner before launch.
- Confirmatory source commit:
  `243612aa1ce7f91bb674a11085b24375721b239f`.
- The runner verified the three checkpoint SHA-256 values against the frozen
  protocol before loading them.
- Generated 100 prompts in each of four untouched positive splits and 160
  untouched negative prompts. No prompt or output was removed.
- Inference ran locally on Apple arm64/macOS 26.5.2 using Python 3.12.12,
  PyTorch 2.13.0, and MPS. Wall time was 1,096.43 seconds.
- Pooled mathematical correctness:
  - untouched base: 68/400 (17.0%);
  - matched learned control: 21/400 (5.25%);
  - learned-router internal: 360/400 (90.0%);
  - oracle-route internal: 400/400 (100%);
  - internal forced off: 68/400 (17.0%).
- Split-level base/control/internal/oracle counts were:
  - one- to four-digit simple: 59/19/93/100;
  - five- to eight-digit simple: 3/1/88/100;
  - nine- to twelve-digit simple: 0/0/79/100;
  - five- to eight-digit word problems: 6/1/100/100.
- Paired internal versus base:
  68 both correct, 292 internal-only, 0 base-only, 40 both wrong;
  exact McNemar `p = 2.513e-88`.
- Paired internal versus control:
  21 both correct, 339 internal-only, 0 control-only, 40 both wrong;
  exact McNemar `p = 1.786e-102`.
- The router activated on 349/400 positives (87.25%). Every active execution
  was mathematically and format exact. Of 51 route misses, the unchanged base
  path answered 11 correctly and 40 incorrectly.
- On 160 unseen negatives there were zero false activations and all 160
  outputs were token-identical to base.
- Forced-off outputs were token-identical to base on all 400 positives.
- Five of six frozen criteria passed. Positive routing coverage missed its
  90% threshold by 2.75 points. The formal compound verdict is therefore not a
  full pass, even though both primary capability comparisons passed by large
  margins.
- Raw archive:
  `phase4_results/confirmation_raw.json`.
- Derived analysis:
  `phase4_results/confirmation_analysis.json`,
  `confirmation_summary.csv`, and `confirmation_by_family.csv`.
- Canonical rendered-data SHA-256:
  `b6d9de587db437f49e4106c6b10b643976e37f6bcffe1cf8adae7994d2bf7f98`.

## 2026-07-26 — Reporting and validation

- Added deterministic statistical analysis and generated PDF/PNG figures.
- Wrote `PHASE4_EXECUTIVE_SUMMARY.md` and the complete 12-page technical
  manuscript in `paper_phase4/main.tex`.
- Compiled the paper successfully with the bundled Tectonic runtime to
  `paper_phase4/natural-language-deterministic-arithmetic.pdf`.
- Rendered and visually inspected the title/abstract, result figures, tables,
  paired-analysis pages, and bibliography. Adjusted float placement and
  recompiled to remove an isolated table page.
- Added `scripts/hash_phase4_artifacts.py` to produce a final byte-count and
  SHA-256 inventory covering protocols, reports, raw/derived results, paper,
  and the three local ignored checkpoints.

## 2026-07-26 — Post-report prior-art correction

- A targeted novelty review identified Dietz and Klakow's 2025 paper,
  *IGC: Integrating a Gated Calculator into an LLM to Solve Arithmetic Tasks
  Reliably and Efficiently* (`https://arxiv.org/abs/2501.00684`), as the
  closest prior work.
- IGC had already demonstrated a learned internal gate, learned
  number/operator and output mappings, and a non-differentiable calculator
  attached to a frozen Llama 3.1 8B model. It reported 98–99% across four
  arithmetic operations.
- Revised the manuscript before public release to remove any implication that
  this project originated the broad integrated-calculator concept and added a
  direct, explicitly non-head-to-head comparison.
- Reframed Phase 4 as an independent small-model replication and controlled
  extension emphasizing its 24,225-parameter interface, exact
  parameter-matched learned control, precommitted same-prompt protocol,
  held-out natural-language routing, and causal on/off tests.
- No prompt, checkpoint, output, statistic, success criterion, or
  confirmatory verdict changed. This entry records a literature-positioning
  correction only.
