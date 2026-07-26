# Phase 3 Lab Notebook

This is the chronological, replication-oriented record for the native
deterministic transformer-unit study. Times use America/Los_Angeles unless
otherwise noted. Phase 3 begins after the completed phase-2 reporting commit
`c16fbb3d66e8db41aec0281c98b56e40512dd476`.

## 2026-07-25 — Goal authorization

- Josiah explicitly instructed Codex to start the next persistent goal.
- This explicit instruction supersedes the earlier conditional instruction to
  start only if phase 2 met every success criterion. Phase 2's formal verdict
  remains unsuccessful and is not reinterpreted.
- Created the active goal:
  “Design, implement, run, and document a reproducible experiment that
  modifies a pretrained 0.5B-class transformer to include a genuinely internal
  frozen deterministic arithmetic unit inside or between repeated transformer
  blocks, with learned token/residual encoders and decoders, rigorous controls
  and preservation tests, and a full research report; text Josiah a
  notification-length success/failure summary when complete.”
- Completion notification by text remains explicitly authorized.

## 2026-07-25 — Initial architecture audit

- Read `FUTURE_ARCHITECTURE_GOAL.md`, the phase-2 bridge implementation, and the
  fixed ripple-carry implementation.
- Confirmed that phase 2's bridge is output-adjacent: it modifies the final
  normalized residual immediately before the tied language-model head.
- Inspected the installed Transformers 4.57.6 source for `Qwen2Model.forward`
  and `Qwen2DecoderLayer.forward`.
- Qwen's 24 decoder blocks are iterated directly from `model.layers`; each
  block accepts and returns a hidden-state tensor while sharing the attention
  cache. A wrapper that proxies the original block's `attention_type` and
  calling signature can therefore be installed as a real entry in the repeated
  `ModuleList`.
- Rejected the weakest candidate—moving the phase-2 symbol vector earlier
  while retaining externally parsed operands—as insufficient for the main
  phase-3 claim. It would test downstream transport but not learned
  residual-to-register translation.
- Selected a staged typed-register candidate for feasibility work:
  1. a controlled grammar with space-separated single digit tokens;
  2. a learned affine digit classifier over intermediate residual vectors;
  3. a frozen tensorized ripple-carry cell consuming predicted digits;
  4. a learned eleven-symbol residual codebook written after a selected block;
  5. all remaining native Qwen blocks and the original output head downstream.
- Wrote `PHASE3_ARCHITECTURE_DRAFT.md`. It is explicitly a pilot design, not a
  frozen confirmatory protocol.

## 2026-07-25 — Typed cell implementation and first caught defect

- Added `src/neural_firmware/internal_data.py` with the controlled prompt
  grammar, strict full-match eligibility, operand character-span location, and
  reproducible example generation.
- Added `src/neural_firmware/internal_firmware.py` with:
  - a zero-parameter tensorized ripple-carry cell;
  - a learned residual-to-digit classifier;
  - an eleven-symbol residual decoder;
  - gather, exact-plan, and residual-injection operations.
- Added randomized unit tests using 204 operand pairs up to 18 digits,
  explicitly including zero, unequal widths, and a twelve-position carry
  chain.
- The first test run failed: `9+1` produced `0` when evaluated in the same
  batch as longer operands. Cause: the global loop continued updating shorter
  rows' carry state through padding columns needed by longer rows.
- Corrective change: added a per-row active-width mask and froze each row's
  carry after its declared maximum operand width. This defect was found before
  any pretrained-model feasibility result was generated.
- Reran the focused tests after correction: 2/2 passed; Ruff passed.

## 2026-07-25 — Token-to-register structural feasibility

- Added offset-based token-position recovery to `internal_data.py`. It formats
  the request with Qwen's actual chat template, finds the user request exactly
  once, and maps only the structurally located operand character positions to
  tokenizer offsets.
- Added and ran `scripts/check_phase3_tokenization.py` with:
  - pinned Qwen revision
    `7ae557604adf67be50417f59c2c2f167def9a775`;
  - seed 31415;
  - 1,000 examples with independently sampled one- to twenty-digit operands.
- Result:
  - 1,000/1,000 operand digit sequences recovered at the correct token
    positions;
  - 1,000/1,000 quoted-command variants rejected by the strict full-match
    eligibility rule;
  - formatted chat length ranged from 50 to 126 tokens;
  - no failures.
- Compact output:
  `phase3_results/tokenizer_feasibility.json`.
- This check validates only structural position recovery and tokenizer
  stability. It does not yet validate the learned residual-to-digit encoder.

## 2026-07-25 — Learned digit probes at candidate internal depths

- Added and ran `scripts/run_phase3_digit_probe.py`.
- Pinned model revision remained
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Candidate depths counted completed Qwen blocks: 6, 12, 18, and 22.
- Probe training:
  - 1,000 one- to four-digit addition prompts;
  - seed 43211;
  - one affine 896-to-10 digit classifier per depth;
  - 8,970 trainable parameters;
  - 300 AdamW steps, batch 512, learning rate 0.01.
- Fixed evaluation:
  - 300 one- to four-digit examples;
  - 300 five- to eight-digit examples;
  - 300 nine- to twelve-digit examples.
- Every probe achieved:
  - 100% individual digit accuracy;
  - 300/300 exact two-operand register recovery on every evaluation split.
- The longest split contained 6,350 evaluated digit residuals per depth, all
  classified correctly.
- Dataset SHA-256:
  `c19239cd1b0395de4bf220be7daeda7501a6cf1d03a34f386069c0502ac13904`.
- Feature extraction for all four depths took 22.57 seconds for training and
  6.70--9.83 seconds per evaluation split. Probe optimization took
  0.31--0.57 seconds per depth.
- Compact result: `phase3_results/digit_probe_pilot_v1.json`.
- Decision: the learned input interface is feasible at all candidate depths.
  Continue to decoder/downstream-survival pilots without changing the
  controlled grammar.

## 2026-07-25 — Real layer wrapper and decoder smoke test

- Added `InternalFirmwareLayer`, which contains the original Qwen block and the
  new arithmetic unit as a replacement entry in `model.model.layers`.
- The wrapper preserves the original block's attention type and call
  signature. With no runtime context, it returns the original block result
  without modification.
- Added per-sequence context with:
  - operand residual positions and lengths;
  - immutable eligibility for that response;
  - teacher-forced or autoregressive output positions;
  - stored typed plans;
  - batch-permutation and wrong-symbol intervention fields;
  - detached diagnostic predictions.
- Added `internal_training.py` for teacher-forced decoder optimization and
  cached autoregressive generation through the ordinary final normalization
  and tied output head.
- Focused wrapper tests passed, including exact unit-off identity for an
  identity base layer.
- Smoke configuration:
  - insertion after block 22;
  - learned depth-22 digit encoder loaded from the probe;
  - 9,856 trainable decoder parameters;
  - strength 64;
  - 80 training examples, 30 steps, batch two;
  - eight free-running evaluation examples, including up to six-digit
    operands.
- Training loss fell from 4.0801 to 0.000699 in 5.14 seconds.
- Internal unit scored 8/8 exact; the same wrapped model with the unit disabled
  scored 0/8 and generated prose.
- Compact result: `phase3_results/decoder_smoke.json`.

## 2026-07-25 — Decoder survival across the transformer stack

- Added and ran `scripts/run_phase3_decoder_pilot.py`.
- Each candidate inserted the same architecture after block 6, 12, 18, or 22,
  leaving respectively 18, 12, 6, or 2 ordinary Qwen blocks downstream.
- Each depth used:
  - the independently learned, then frozen, digit probe for that depth;
  - 1,000 one- to four-digit training additions;
  - 120 decoder-only steps, batch two, learning rate 0.01;
  - residual strength 64 and 9,856 trainable codebook parameters.
- Free-running evaluation per depth:
  - 40 one- to four-digit examples;
  - 40 five- to eight-digit examples;
  - 40 nine- to twelve-digit examples.
- All four insertion depths scored 120/120 exact, including the depth-6 unit
  whose state passed through 18 subsequent native transformer blocks.
- Ten depth-specific unit-off checks each scored 0/10; the base model generated
  explanatory prose rather than exact digit-only answers.
- Decoder optimization time ranged from 16.01 to 20.65 seconds per depth.
- Compact result: `phase3_results/decoder_pilot_v1.json`; unit checkpoints and
  full predictions are retained under
  `phase3_artifacts/decoder_pilot_v1/`.
- Pilot selection: use depth 6 for the main candidate because it subjects the
  exact internal state to the largest tested downstream native computation.
  Retain depth 22 as a location control.

## 2026-07-25 — Causal interventions and downstream trace

- Added and ran `scripts/run_phase3_causal_pilot.py` against the frozen depth-6
  pilot checkpoint on new seeds.
- Ordinary free-running evaluation: 30/30 exact on nine- to twelve-digit
  operands.
- Wrong-state intervention:
  - changed the first typed result symbol to a known different digit;
  - 20/20 generated answers exactly matched the intervened internal state,
    rather than the mathematically correct answer.
- Whole-state substitution:
  - replaced each recipient's typed result with a donor example's result of
    the same width;
  - 10/10 recipient generations exactly matched the donor state.
- Downstream logit-lens trace:
  - traced the first correct answer digit for ten examples;
  - mean target-versus-best-competitor margin was -25.24 after block 5,
    immediately before the inserted unit;
  - margin jumped to +9.68 after the unit at block 6;
  - it remained positive after every downstream block;
  - final pre-normalization block-24 margin under the shared final-norm/output
    lens was +13.33.
- These interventions establish causal dependence on the typed internal state,
  rather than a merely correlated activation.
- Compact result: `phase3_results/causal_pilot_v1.json`.

## 2026-07-25 — Preservation and immutable eligibility latch

- Added and ran `scripts/run_phase3_preservation_pilot.py`.
- Constructed 100 ineligible prompts containing:
  - ordinary unrelated language;
  - the phase-2 routing-negative templates;
  - valid internal-register commands embedded in quotes, negations,
    documentation, prefixes, or suffixes.
- The strict full-request grammar rejected 100/100 before generation and
  eligibility remained off for the whole response.
- Loaded an unmodified Qwen instance and separately loaded the depth-6 wrapped
  instance. The wrapped model reproduced all 100 base token sequences exactly.
- Preservation: 100/100 (100%).
- This directly repairs the phase-2 late-activation failure: the learned
  interface no longer recomputes an eligibility decision after each generated
  token.
- Compact result: `phase3_results/preservation_pilot_v1.json`.

## 2026-07-25 — Parameter-matched learned internal control

- Added a same-depth bottleneck residual adapter with no deterministic state.
- Rank ten gives exactly 18,826 trainable parameters:
  - 8,970 in the down projection and bias;
  - 9,856 in the up projection and bias.
- This exactly matches the combined 8,970-parameter digit encoder plus
  9,856-parameter symbol decoder in the internal-firmware condition.
- Both mechanisms are inserted after block 6 and active only at registered
  answer-generation positions.
- Added and ran `scripts/run_phase3_learned_control_pilot.py`.
- Gave the control 1,000 training steps on 1,000 one- to four-digit additions,
  batch two, learning rate 0.001—substantially more optimization than the
  deterministic decoder's 120 steps.
- Training took 161.44 seconds; loss fell from 3.3325 to 1.5823.
- Free-running exact match:
  - 18/60 (30.0%) on one- to four-digit operands;
  - 0/60 on five- to eight-digit operands;
  - 0/60 on nine- to twelve-digit operands.
- Decision: retain this control for confirmation. Its location and parameter
  count are exactly matched, and its larger training budget gives ordinary
  learned adaptation a conservative opportunity to fit the task.
- Compact result:
  `phase3_results/learned_control_pilot_v1.json`.

## 2026-07-25 — Confirmatory runner smoke and protocol freeze decision

- Added reusable digit-feature collection, encoder training, and register
  evaluation functions in `internal_probe.py`.
- Added a resumable, condition-by-condition study runner:
  `scripts/run_phase3_study.py`.
- The runner records:
  - clean source commit and canonical configuration hash;
  - logical fixed-evaluation hash;
  - seed-specific training hashes;
  - encoder and decoder training;
  - register decoding;
  - base, internal, unit-off, and parameter-matched control predictions;
  - language preservation;
  - wrong-state and state-substitution interventions;
  - layer-by-layer downstream traces;
  - checkpoints and per-seed summaries.
- Added `configs/phase3_study_smoke.json` and ran an end-to-end dirty-worktree
  smoke with one seed, 12 training examples, two steps per trainable stage,
  three examples per random split, two carry examples, four preservation
  prompts, two interventions, and one trace.
- Smoke completed base, encoder, internal decoder, unit-off, preservation,
  interventions, trace, learned control, checkpoint writing, and final study
  assembly without an execution error.
- As expected, the two-step smoke did not learn the task and is not an accuracy
  result. Its purpose was runner validation only.
- Corrected state-substitution pairing after smoke inspection: examples are
  now grouped by answer length before pairing, guaranteeing shape-compatible
  donor/recipient states whenever a group contains at least two examples.
- Wrote `PHASE3_PROTOCOL.md` and the full
  `configs/phase3_study.json`.
- Fixed confirmatory training seeds 2101, 2203, and 2309 and previously unused
  evaluation seed 884321.
- Fixed eight simultaneous success criteria covering end-to-end arithmetic,
  parameter-matched advantage, input-register exactness, preservation,
  interventions, unit-off dependence, and downstream survival.
- No confirmatory training or evaluation has begun. Next action is complete
  verification, commit the frozen source, and start from that clean commit.

## 2026-07-25 — Frozen-source verification and confirmatory launch

- Ran the complete pre-confirmation verification suite:
  - 15/15 tests passed;
  - Ruff reported no violations.
- Committed the exact source and protocol before confirmation as
  `a2cf317d42ace613817d3b609b60e245402e3783`
  (`Freeze native transformer unit confirmatory study`).
- Launched exactly:

  ```bash
  uv run python scripts/run_phase3_study.py \
    --config configs/phase3_study.json \
    --artifact-directory phase3_artifacts/confirmatory_v1 \
    --result-directory phase3_results/confirmatory_v1
  ```

- The runner wrote `frozen_state.json` before model fitting:
  - source commit:
    `a2cf317d42ace613817d3b609b60e245402e3783`;
  - canonical configuration SHA-256:
    `30a31563834e35edd0bdd635b452866d26611d25f75f646f5f08f6ddd963c261`;
  - confirmatory flag: `true`.
- The fixed logical evaluation set has SHA-256
  `d217043d5d20bca4f9aade737ee01d62624f91aadeb6ffea9dfd4233ce606ca3`.
- One long shell session ended at the conversation-turn boundary after
  retaining complete seed-2101 and seed-2203 summaries. The same frozen command
  was invoked again. The resumable runner detected and skipped every completed
  condition, then computed only the missing seed-2309 work. No training seed,
  prompt, hyperparameter, checkpoint, or result was replaced or recomputed
  under a different confirmatory label.
- The assembled study records a 2,274.71-second interval from initial frozen
  launch to final assembly. This interval includes the boundary interruption
  and resume and is not presented as a pure benchmark.

## 2026-07-25 — Confirmatory arithmetic and matched-control results

- Every independently trained internal-unit seed scored:
  - 150/150 on one- to four-digit in-distribution additions;
  - 150/150 on five- to eight-digit primary OOD additions;
  - 150/150 on nine- to twelve-digit long OOD additions;
  - 75/75 on five- to twelve-digit carry chains.
- Pooled internal arithmetic was 1,575/1,575 whole-sequence exact matches.
- The frozen unmodified base scored 0/150, 0/150, 0/150, and 0/75 on the four
  corresponding sets. It generally emitted explanatory prose under this
  deliberately controlled digit-only response contract.
- The parameter-matched learned internal controls scored:
  - seed 2101: 40/150 ID, 0/150 primary OOD, 0/150 long OOD, 0/75 carry;
  - seed 2203: 29/150 ID, 0/150 primary OOD, 0/150 long OOD, 2/75 carry;
  - seed 2309: 35/150 ID, 0/150 primary OOD, 0/150 long OOD, 1/75 carry.
- Mean learned-control ID accuracy was 23.11%; mean primary and long OOD
  accuracy were both zero; mean carry-chain accuracy was 1.33%.
- The primary paired internal-minus-control difference was 100 percentage
  points in all three seeds. A 100,000-draw seed-level paired bootstrap with
  analysis seed 20260726 consequently yielded the degenerate 95% percentile
  interval [100, 100] percentage points. With only three seeds, this interval
  reflects the observed seed-level invariance and is not broad evidence about
  arbitrary training setups.

## 2026-07-25 — Register, preservation, ablation, and causal results

- The learned, then frozen, residual-to-digit encoders recovered both operand
  registers exactly on 1,575/1,575 evaluated additions across the three seeds.
  The deterministic cell therefore consumed predicted internal digits, not
  ground-truth operands.
- Disabling the unit after training reduced primary OOD exact match from
  450/450 to 0/450, a 100-percentage-point drop.
- On 300 ineligible prompts, the real wrapped model produced exactly the same
  complete output-token sequence as an independently loaded base model:
  300/300 preserved.
- In wrong-state interventions, generation exactly followed the deliberately
  incorrect typed result in 90/90 cases.
- In equal-length state substitutions, generation exactly followed the donor
  state in 39/39 cases.
- These three checks distinguish an operative internal causal path from a
  correlated probe or an external post-generation replacement.

## 2026-07-25 — Downstream survival and preregistered verdict

- Collected 30 first-answer-symbol traces: ten fixed long-OOD examples for
  each of three independently trained units.
- Under the shared final-norm/tied-head logit lens, the mean correct-digit
  margin was strongly negative immediately before the unit after block 5,
  then positive after the wrapped sixth block and after every remaining block.
- The smallest post-unit mean margin was +9.7965 logits after block 22. The
  answer state therefore survived all eighteen subsequent unmodified
  transformer blocks with a positive mean margin.
- Wrote `scripts/analyze_phase3.py` to reconstruct every tabular result,
  bootstrap, success criterion, error inventory, and three publication
  figures from the raw study JSON.
- Independent analysis found all eight frozen criteria passed:
  1. internal primary mean 1.000 >= 0.990;
  2. paired internal-control advantage 1.000 >= 0.750;
  3. exact-register mean 1.000 >= 0.995;
  4. preservation 1.000 >= 0.990;
  5. wrong-state causal rate 1.000 >= 0.950;
  6. substitution causal rate 1.000 >= 0.950;
  7. unit-off primary drop 1.000 >= 0.800;
  8. minimum post-unit mean margin +9.7965 > 0.
- **Frozen confirmatory verdict: successful.**
- This verdict is deliberately narrow. It is an architectural proof on a
  registered, space-separated addition grammar. It does not establish general
  mathematical understanding, operation selection, natural-language operand
  extraction, subtraction/multiplication, or a calculator compiled into one
  ordinary scalar transformer neuron.

## 2026-07-25 — Reproducibility records and reporting environment

- Added `scripts/hash_phase3_artifacts.py`. The raw confirmatory archive
  currently contains 17 files totaling 6,410,409 bytes; the tracked manifest
  stores the SHA-256 and byte count of every file relative to the archive root.
- Added `scripts/capture_phase3_environment.py` and recorded:
  - Apple arm64 host with 17,179,869,184 bytes (16 GiB) unified memory;
  - macOS 26.5.2 build 25F84;
  - Python 3.12.12;
  - PyTorch 2.13.0 with MPS available;
  - Transformers 4.57.6;
  - NumPy 2.5.1 and Matplotlib 3.11.1.
- Compact analysis tables, figures, environment metadata, and the manifest are
  tracked. The 6.2 MiB raw archive and checkpoints remain locally retained
  under the ignored `phase3_artifacts/confirmatory_v1/` directory.
