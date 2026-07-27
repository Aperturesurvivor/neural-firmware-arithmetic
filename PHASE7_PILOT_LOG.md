# Phase 7 Pilot Log

This file records development-only neuron-implant work. Failed and partial
runs remain part of the record. No confirmatory protocol is frozen.

## 2026-07-26 — Census attempt v1

- Implemented the 108-channel in-place typed ABI described in
  `PHASE7_ARCHITECTURE_DRAFT.md`.
- Unit tests for exact frozen execution, invalid-interface abstention, unchanged
  tensor width, zero calculator parameters, and teacher-forced result flow
  passed.
- Began a mixed arithmetic/adversarial activation census at decoder layers 8,
  12, and 16.
- Layer 8 candidate-bank ablation retained 95.83% final-token top-1 agreement
  with mean KL divergence 0.00407766.
- Layer 12 retained 96.88% agreement with mean KL divergence 0.00159131.
- The process terminated before completing layer 16 and before writing its
  artifact. The first implementation computed full-sequence vocabulary logits
  during every ablation comparison, creating avoidable memory pressure on the
  16 GB M4 host.
- Decision: retain the partial console measurements as engineering evidence,
  change the ablation diagnostic to project only the final-token hidden state,
  and rerun the complete census as v2.

## 2026-07-26 — Census v2

- Recomputed final-token ablations without materializing full-sequence
  vocabulary logits.
- Layer 8 retained 95.83% top-1 agreement with mean KL 0.00407762.
- Layer 12 retained 96.88% top-1 agreement with mean KL 0.00159129.
- Layer 16 retained 94.79% top-1 agreement with mean KL 0.00280141.
- Selected layer 12 because its 108-channel joint arithmetic/adversarial bank
  caused the smallest distributional disturbance and highest top-1 retention.
- Raw record: `phase7_results/census_v2.json`.
- Tensor artifact: `phase7_artifacts/census_v2.pt`.

## 2026-07-26 — Implant pilot v1

- Began frozen-model feature collection for 600 addition and 600 adversarial
  training prompts, plus a separate 120/120 development set.
- Training feature collection completed and wrote
  `phase7_artifacts/cache/interface_train_v1.pt`.
- The process terminated before development feature collection completed and
  before any implant weights were trained. Repeated long Metal forwards were
  retaining cached accelerator allocations.
- Decision: retain the completed training cache, release captured tensors and
  the Metal cache incrementally, load each cache independently, and rerun as
  pilot v2.

## 2026-07-26 — Anchor-interface pilot v2

- The single-token layer-12 interface reached 89.45% route true positives at
  0.83% false positives and 100% answer-position accuracy.
- Exact four-slot operand recovery was only 0.39%.
- The process terminated when output-column training began; no completed raw
  record or checkpoint was written.
- Interpretation: a single linear replacement row bank at one anchor token did
  not expose enough exact digit information from Qwen's native residual.
- Decision: do not add a separate attention parser. Distribute typed digit and
  operand-role activations across the number-bearing token positions, reduce
  the ABI from 108 to 34 channels, and let the frozen microcircuit scan those
  neuron states directly.

## 2026-07-26 — Sequence-interface pilot v1

- Selected 34 low-impact channels at Qwen's final MLP layer. Zeroing the bank
  retained 96.88% top-1 agreement with mean KL divergence 0.00015148.
- The learned 19,712-parameter input-row interface achieved on development:
  - route true-positive rate: 100%;
  - route false-positive rate: 0%;
  - operand-role token accuracy: 97.67%;
  - digit accuracy at operand tokens: 96.96%;
  - answer-position accuracy: 93.08%.
- The interface checkpoint and partial raw record were written before
  output-column training.
- The process then terminated during repeated Metal backpropagation. An
  isolated one-step run succeeded, localizing the problem to retained
  per-step graph references rather than model size or an unsupported
  operation.
- Raw partial record: `phase7_results/sequence_pilot_v1.json`.
- Checkpoint:
  `phase7_artifacts/sequence_pilot_v1/neuron_implant_seed_12801.pt`.
- Decision: explicitly clear the runtime context and per-step graph tensors,
  empty the Metal cache periodically, and rerun as sequence pilot v2.

## 2026-07-26 — Sequence-interface pilot v2

- With explicit graph cleanup, 100 result-column updates completed on the M4
  Metal backend.
- The expanded 4,000-example compact interface training set produced:
  - route true-positive rate: 100%;
  - route false-positive rate: 0%;
  - operand-role token accuracy: 98.56%;
  - digit accuracy at operand tokens: 98.30%;
  - answer-position accuracy: 97.11%.
- The complete 20-addition development generation set reached 16/20 exact
  answers. With only the calculator-result activation ablated, exact accuracy
  fell to 1/20.
- First-step routing was active on 19/20 additions and both operands were
  exactly decoded on 16/20. Every exact-answer success followed the exact
  deterministic calculator symbol trajectory. Observed failures came from
  routing, digit/role decoding, or result-position decoding rather than an
  incorrect deterministic addition.
- On 20 adversarial negatives, the first generated token matched untouched
  Qwen in all 20 cases, but later generated tokens caused eleven false
  activations and only 10/20 complete eight-token generations were preserved.
- Raw records:
  - `phase7_results/sequence_interface_v2.json`;
  - `phase7_results/sequence_evaluation_v2.json`.
- Checkpoint:
  `phase7_artifacts/sequence_interface_v2/neuron_implant_seed_12811.pt`.
- Checkpoint SHA-256:
  `fc5a547033ebe1a8fbe9888fa5a5549c0b0592f0e9524a7628e30a2bcee41d6a`.

## 2026-07-26 — Native-SwiGLU interface pilot v3 (negative)

- Replaced the linear input rows with a native Qwen-style
  `SiLU(gate) * up` interface while retaining the same 34 selected channels.
- Development digit accuracy fell to 39.91% and answer-position accuracy to
  78.63%, despite 99.77% route true positives and zero route false positives.
- This did not improve the required typed interface and was not promoted to
  generation evaluation.
- Raw record: `phase7_results/sequence_interface_v3.json`.
- Decision: retain the simpler linear replacement-row interface for the next
  audit. A multiplicative interface may need a different objective or
  initialization; it is not assumed superior merely because it resembles the
  original MLP nonlinearity.

## 2026-07-26 — Latched-route preservation pilot

- Froze the first generated-step route decision for the rest of each response.
- Restored the original selected-channel MLP contribution exactly whenever the
  latched route was `OFF`.
- Arithmetic performance remained 16/20 exact, and result-channel ablation
  remained 1/20 exact.
- First-step routing was active on 19/20 additions; both operands were exactly
  decoded on 16/20.
- All 20 adversarial negatives remained route-off and all 20 eight-token
  generations were bit-for-bit token-identical to untouched Qwen.
- Raw record: `phase7_results/sequence_evaluation_v2_gated.json`.
- Raw-record SHA-256:
  `ca13841f827f266d0e70057033bc0caa42518dde70c16812e6411ec8b6d815c8`.
- Interpretation: the preservation failure in the ungated pilot was a
  response-control problem, not unavoidable damage from occupying the selected
  channels. This is strong development evidence, but it reuses development
  prompt families and one learned seed.

## 2026-07-26 — Frozen held-out audit v1

- Froze `PHASE7_AUDIT_PROTOCOL.md`, the v2 checkpoint hash, 16 new prompt
  families per route class, three data seeds, and engineering gates before
  generation.
- Evaluated 20 direct additions, 20 word-problem additions, and 40
  adversarial non-addition prompts with one-to-four-digit operands.
- Addition results:
  - mathematical exact: 25/40 (62.5%);
  - digits-only exact format: 24/40 (60%);
  - calculator-result ablation exact: 0/40;
  - first-step route active: 38/40;
  - first-step operands exact: 29/40;
  - strict calculator digit-plus-EOS trajectory exact: 23/40.
- Direct additions reached 15/20 mathematical exact and word problems reached
  10/20.
- Negative results:
  - any false calculator route: 0/40;
  - exact eight-token preservation versus untouched Qwen: 40/40.
- The result passed the predeclared causal-ablation and preservation gates but
  failed the 70% held-out positive-accuracy gate.
- Failure inspection found:
  - two direct prompts that did not route;
  - nine word prompts with incorrect decoded operand digits;
  - four cases with correct operands but a repeated or premature result
    position;
  - no evidence that the frozen ripple-carry addition itself produced an
    incorrect sum from a correct typed interface.
- Raw record: `phase7_results/sequence_audit_v1.json`.
- Raw-record SHA-256:
  `f9e1a864ef184d2488ec9c71cba2ee0c40aa8b8ca7fb0fdaf88888d025bf22a6`.
- Decision: preserve the failed overall gate. Test a deterministic
  result-position counter as part of the fixed microcircuit, then improve the
  neural operand interface on consumed development data before defining any
  new held-out audit.

## 2026-07-26 — Post-audit deterministic result counter

- Replaced the learned per-token answer-position decision during generation
  with a fixed response-local counter. The neural route and operand interface,
  checkpoint, selected channels, and result columns remained unchanged.
- This is explicitly post-audit engineering on consumed prompts, not new
  held-out evidence.
- On the same 40 positive prompts:
  - mathematical exact rose from 25/40 to 29/40;
  - direct additions rose from 15/20 to 18/20;
  - word problems rose from 10/20 to 11/20;
  - calculator-result ablation remained 0/40;
  - all 29 prompts with exact first-step operands became mathematically and
    format exact.
- All 40 adversarial negatives again had zero false routes and exactly matched
  untouched Qwen.
- Interpretation: learned answer-position inference was unnecessary and
  caused four avoidable errors. The remaining addition failures are exactly
  the two route failures plus nine incorrect operand decodings.
- Current limitation: the counter and route latch are held in the generation
  runtime context. A later architecture must either encode this state in
  reserved recurrent/register activations or define it as persistent fixed
  microcircuit state before claiming arbitrary internal multi-call reasoning.
- Raw record:
  `phase7_results/sequence_audit_v1_step_counter.json`.
- Raw-record SHA-256:
  `99737665a9318bd9d122158d4d4957b6db640ac6e3aee709ff659a6b287bcf8d`.

## 2026-07-26 — Post-audit decoder-depth probe

- Trained equal 19,712-weight linear typed-interface probes on frozen residuals
  at decoder layers 4, 8, 12, 16, 20, and 23.
- The consumed audit prompts were used only for architecture diagnosis; these
  probe results are not new held-out evidence.
- Joint word-problem operand role/digit token accuracy by layer:
  - layer 4: 100%;
  - layer 8: 100%;
  - layer 12: 100%;
  - layer 16: 100%;
  - layer 20: 96.88%;
  - layer 23: 79.17%.
- The late-layer degradation explains the v2 implant's word-problem digit
  failures: exact token identity is progressively blurred by contextual
  computation near Qwen's output.
- Layers 4 and 8 retained exact operands but falsely routed 17.5% and 37.5% of
  audit negatives respectively. Layer 16 achieved 99.40% route true positives,
  0% route false positives, and 100% direct and word-problem operand token
  accuracy.
- Decision: move the single-site implant from layer 23 to layer 16. A two-site
  early-register/late-router path remains an option, but is not needed unless
  the actual layer-16 generation implant fails.
- Raw records:
  `phase7_results/layer_probe_v1_layer_04.json` through
  `phase7_results/layer_probe_v1_layer_23.json`.

## 2026-07-26 — Compact layer-16 channel census

- Removed the six learned answer-position channels made obsolete by the fixed
  result counter.
- The new ABI occupies 28 existing MLP channels:
  16 route/role/digit input channels and 12 calculator-result channels.
- This reduces learned replacement weights from 30,464 to 25,088:
  14,336 input-row weights and 10,752 result-column weights.
- Ablating the selected 28-channel bank across 96 arithmetic/adversarial
  prompts retained 95.83% top-1 agreement, with mean KL divergence 0.00050330
  and maximum KL divergence 0.00429231.
- Raw record:
  `phase7_results/sequence_census_layer_16_compact_v1.json`.
- Tensor artifact:
  `phase7_artifacts/sequence_census_layer_16_compact_v1.pt`.

## 2026-07-26 — Compact layer-16 implant training v1

- Trained the 28-channel fixed-step implant at decoder layer 16 using the same
  4,000 training and 600 development prompts as the layer-23 v2 interface.
- Qwen remained frozen.
- Development neural-interface metrics:
  - route true-positive rate: 100%;
  - route false-positive rate: 0%;
  - operand-role token accuracy: 99.93%;
  - digit accuracy at operand tokens: 100%.
- Trained the 10,752 result-column weights for 150 teacher-forced updates
  through Qwen's remaining decoder layers. Token loss fell from 1.36451 to
  0.00002499.
- Learned replacement weights:
  - input rows: 14,336;
  - result columns: 10,752;
  - total: 25,088.
- Frozen calculator learned parameters: zero.
- Checkpoint:
  `phase7_artifacts/sequence_layer16_v1/neuron_implant_seed_13201.pt`.
- Checkpoint SHA-256:
  `5b76181c2c0f4a74b7482e4856b2b8c92bff637a1e70302bdfbc61ce1aaac41e`.
- Raw training record:
  `phase7_results/sequence_layer16_training_v1.json`.

## 2026-07-26 — Layer-16 consumed-audit diagnostic before typed handshake

- Ran the first 40 positive prompts from the already-consumed audit; stopped
  before negatives once the failure mechanism was localized.
- Exact generation reached 30/40, with calculator-result ablation at 0/40.
- Direct additions reached 18/20; word problems reached 12/20.
- Every active exact typed interface produced an exact formatted answer.
- Token-level probes had shown perfect word operand-token classification, but
  sequence-level exact operands were only 12/20. Inspection found rare
  ordinary word tokens classified as operand roles while their independent
  digit type remained `NON_DIGIT`. The original extractor rejected the whole
  operand on this contradictory state.
- Decision: require both role and digit-type neurons to agree before a token is
  admitted to an operand register. This is a deterministic typed-handshake
  rule and changes no learned weight.
- Partial raw record retained as
  `phase7_results/sequence_layer16_consumed_audit_v1_pre_handshake.json`.
- Partial-record SHA-256:
  `823eca36c63469e0b54464d276d282da27ec44c86c89bb64a26c532e09277bcf`.

## 2026-07-26 — Layer-16 typed-handshake consumed evaluation

- Re-evaluated all 80 already-consumed audit prompts with the same learned
  checkpoint after adding role/digit type agreement and the fixed result
  counter.
- Addition results:
  - mathematical exact: 39/40;
  - exact digits-only format: 39/40;
  - calculator-result ablation exact: 0/40;
  - exact first-step operands: 40/40;
  - exact calculator trajectory: 39/40.
- Direct additions reached 19/20 and word problems reached 20/20.
- The sole addition failure had exact decoded operands but conservatively
  remained route-off.
- All 40 adversarial negatives remained route-off and token-for-token
  identical to untouched Qwen.
- The checkpoint still contains only 25,088 learned replacement weights and a
  zero-parameter calculator.
- This is strong post-audit development evidence, not independent held-out
  evidence, because audit v1 guided decoder-layer and typed-handshake choices.
- Raw record:
  `phase7_results/sequence_layer16_consumed_audit_v2_handshake.json`.
- Raw-record SHA-256:
  `3755804bf0dd8c497b42536f25ce69c9ddabd80ecadbf15b51af62ebf27e2911`.
- Decision: retain the development threshold unchanged and freeze a second,
  larger audit with new prompt constructions before any further architecture
  or weight changes.

## 2026-07-26 — Independent held-out audit 2

- Froze `PHASE7_AUDIT2_PROTOCOL.md`, 20 new prompt families per route class,
  three new data seeds, the checkpoint hash, the development threshold, and
  stricter engineering gates before generation.
- Evaluated 30 direct additions, 30 word-problem additions, and 60
  adversarial non-addition prompts.
- Addition results:
  - mathematical exact: 54/60 (90%);
  - digits-only format exact: 54/60;
  - calculator-result ablation exact: 6/60;
  - first-step route active: 59/60;
  - exact first-step operands: 55/60;
  - exact calculator trajectory: 54/60.
- Direct additions reached 29/30 and word problems reached 25/30.
- All 54 cases with both an active route and exact operands produced the exact
  calculator trajectory and exact formatted answer.
- Negative results:
  - any false calculator route: 0/60;
  - exact eight-token preservation versus untouched Qwen: 60/60.
- Gate disposition:
  - passed the 54/60 primary exact-accuracy gate exactly;
  - passed the conditional deterministic-execution gate;
  - passed both negative routing/preservation gates;
  - missed the 57/60 exact-operand gate with 55/60;
  - missed the absolute ablation and 85-point-drop gate: base Qwen independently
    solved 6/60, leaving an 80-point drop.
- Failure inspection:
  - one direct `ADD` prompt remained route-off;
  - five word problems admitted one extra false-positive digit into operand B;
  - no failure occurred after an active route and exact typed operands.
- Interpretation: the narrow in-place deterministic-neuron mechanism
  generalizes and is causally used, but the neural operand framing is not yet
  reliable enough for a broad correctness claim. The absolute ablation cap was
  also poorly calibrated to base-Qwen competence, though its frozen failure is
  retained.
- Raw record: `phase7_results/sequence_audit2_v1.json`.
- Raw-record SHA-256:
  `e42bb02a8cbe9bb5554ae74a88eea1d45aa5d52154837bc2d8b17c6fa7d55444`.

## 2026-07-26 — Post-audit neural digit-confidence handshake

- Inspected the independent digit softmax activations on all 460 true operand
  tokens and all five false typed-digit candidates across the now-consumed
  audits.
- True digit confidence ranged from 0.96697 to 0.99999.
- False typed-digit confidence ranged from 0.28681 to 0.82719.
- Added a fixed 0.90 digit-confidence requirement to the typed handshake. This
  uses the learned neuron activation, adds zero parameters, and changes no
  learned weight.
- Promoted checkpoint:
  `phase7_artifacts/sequence_layer16_confident_v1/neuron_implant_seed_13201.pt`.
- Promoted checkpoint SHA-256:
  `9dba639d127769b08579b2e1deabdfd3d232e06dcf2ea6f843f7b9963855785c`.
- A bytewise tensor check confirmed that input-row and result-column weights
  were unchanged from the audit-2 checkpoint.
- On the consumed audit-2 prompts:
  - mathematical and format exact: 59/60;
  - exact operands: 60/60;
  - direct: 29/30;
  - word problems: 30/30;
  - calculator-result ablation: 6/60;
  - conditional active-route/exact-operand output: 59/59;
  - false negative routes: 0/60;
  - exact negative preservation: 60/60.
- This is post-audit development evidence and cannot replace the frozen audit-2
  result. It establishes the expected behavior before independent-seed
  training and a new shared holdout.
- Promotion manifest:
  `phase7_results/sequence_layer16_digit_confidence_v1.json`.
- Consumed evaluation:
  `phase7_results/sequence_audit2_v1_digit_confidence.json`.

## 2026-07-26 — Independent interface/output seeds 2 and 3

- Reused the frozen layer-16 channel bank and frozen training/development
  examples, but independently initialized and trained all 25,088 learned
  replacement weights for seeds 13,202 and 13,203.
- Seed 13,202 development:
  - route true-positive rate: 100%;
  - route false-positive rate: 0.33% (1/300);
  - operand-role accuracy: 100%;
  - digit accuracy: 100%;
  - final result-column token loss: 0.02313.
- Seed 13,203 development:
  - route true-positive rate: 100%;
  - route false-positive rate: 0%;
  - operand-role accuracy: 100%;
  - digit accuracy: 100%;
  - final result-column token loss: 0.00449.
- Checkpoints:
  - seed 13,201 confidence promotion:
    `9dba639d127769b08579b2e1deabdfd3d232e06dcf2ea6f843f7b9963855785c`;
  - seed 13,202:
    `6cab7608a912d19a26793828352cdffd8783e2e5f6b8bdcad65f5afdf22b6b07`;
  - seed 13,203:
    `cbf84806b08f0804cd0c508330e0f5f9fdfa72b41c800a37aef616c682fb58dd`.
- Raw training records:
  `phase7_results/sequence_layer16_training_seed_13202.json` and
  `phase7_results/sequence_layer16_training_seed_13203.json`.
- Decision: freeze one new shared holdout and evaluate all three checkpoints
  without per-seed threshold or architecture changes.

## 2026-07-26 — Frozen three-seed shared holdout

- Froze `PHASE7_MULTISEED_PROTOCOL.md`, all three checkpoint hashes, 20 new
  positive prompt families, 20 new adversarial negative families, three data
  seeds, and five compound engineering gates before any audit-3 generation.
- Every independently learned interface produced the same addition result:
  - mathematical and format exact: 58/60;
  - direct additions: 30/30;
  - word problems: 28/30;
  - first-step route active: 60/60;
  - exact operands: 58/60;
  - exact calculator trajectory: 58/60;
  - calculator-result ablation exact: 3/60;
  - paired causal drop: 55/60.
- For every seed, all 58 active-route examples with exact operands produced
  the exact deterministic digit-plus-EOS trajectory and exact formatted
  answer. The two common addition failures were operand-framing failures, not
  incorrect deterministic addition.
- Adversarial negative results:
  - seed 13,201: 6/60 false routes, 54/60 token-exact preservation;
  - seed 13,202: 5/60 false routes, 55/60 token-exact preservation;
  - seed 13,203: 6/60 false routes, 54/60 token-exact preservation.
- Gate disposition:
  - passed per-seed and mean addition accuracy;
  - passed per-seed operand recovery;
  - passed per-seed causal ablation;
  - failed negative routing and preservation;
  - passed exact execution conditional on an active route and exact operands.
- The compound result therefore passed four of five gates and is not reported
  as an overall protocol success. The narrow calculator-neuron mechanism
  replicated; robust intent routing did not.
- First-step route confidence overlapped across classes. The minimum positive
  probabilities were 0.2944, 0.2683, and 0.4426, while maximum negative
  probabilities were 0.9237, 0.8429, and 0.8277. A cutoff strict enough to
  remove the false routes would suppress valid additions, so threshold tuning
  alone is not a valid repair.
- Raw records and SHA-256 hashes:
  - `phase7_results/multiseed_audit3_seed_13201.json`:
    `0ac65f555403cb11e46a8a8587530fe34776d580cc9ee13236332b176dad1104`;
  - `phase7_results/multiseed_audit3_seed_13202.json`:
    `84f9423a4cd199cd4f577abdc300f4c56b3842a95803fc7bbea67f6e25bd6fc0`;
  - `phase7_results/multiseed_audit3_seed_13203.json`:
    `9e87c3d8dc22a3086a04ef4b545e19fcc18cf2ed82fbafe37ffc2ef95e707b92`.
- Compact analysis:
  `phase7_results/multiseed_audit3_summary.json`.
- Decision: retain the failed router gate. Train only the two route rows on a
  broader set of now-consumed semantic negatives, leave the other 25,088
  tensor values fixed except those 1,792 route weights, then define a new
  exact-string-disjoint audit before evaluating the revised router.

## 2026-07-26 — Targeted first-step router hardening

- Built a family-balanced post-audit development corpus from all prompt
  families consumed through audit 3:
  - 126 positive families and 128 adversarial negative families;
  - 4,064 training prompts and 1,016 exact-string-disjoint development
    prompts;
  - direct additions, word problems, other operations, quotations, refusals,
    explanations, labels, concatenation, and meta-arithmetic prompts.
- Re-trained only the two first-step route rows, or 1,792 weights per seed.
  The remaining 23,296 learned interface weights, the selected channel bank,
  the result columns, the deterministic calculator, and all Qwen weights
  remained fixed.
- All three seeds reached 504/504 positive and 512/512 negative route
  decisions on the development prompts at a fixed 0.5 threshold.
- Development confidence ranges:
  - seed 13,201: positive minimum 0.99555, negative maximum 0.00542;
  - seed 13,202: positive minimum 0.99499, negative maximum 0.00570;
  - seed 13,203: positive minimum 0.99651, negative maximum 0.00581.
- Hardened checkpoint SHA-256 hashes:
  - seed 13,201:
    `fa6deb3bfa8c7d4cf6af06255e35e777dcc18badc7e7eda0e519eac70c2f91a6`;
  - seed 13,202:
    `b483c3fbcec274cdf2f1b23acff33ae63966575c4d0f491eed3f182a73f24eea`;
  - seed 13,203:
    `26eee38d61dcad97e0c9e4c5252e1d7bdf8073ae667ea29a8e8836125e7edf71`.
- Raw development record:
  `phase7_results/sequence_layer16_router_hardening_v1.json`.
- These are post-audit development results. They do not repair the frozen
  audit-3 failure. The hashes are frozen before defining audit-4 families.

## 2026-07-26 — Frozen router-hardening audit 4

- Froze `PHASE7_ROUTER_HARDENING_PROTOCOL.md`, all three hardened checkpoint
  hashes, 20 new addition family strings, 30 new adversarial negative family
  strings, three new data seeds, and five gates before generation.
- Per-seed addition results:
  - seed 13,201: 58/60 exact, 59/60 exact first-step operands, 1/60 ablated;
  - seed 13,202: 60/60 exact, 60/60 exact first-step operands, 1/60 ablated;
  - seed 13,203: 59/60 exact, 59/60 exact first-step operands, 1/60 ablated.
- The three-seed mean was 59/60 exact. Paired causal drops were 57/60,
  59/60, and 58/60.
- Router hardening fully repaired the targeted held-out failure:
  - 0/180 false routes across the three independent seeds;
  - 180/180 negative generations token-identical to untouched Qwen.
- Gate disposition:
  - passed addition accuracy;
  - passed operand recovery;
  - passed causal ablation;
  - passed adversarial routing and preservation;
  - failed exact execution conditional on initial route and operands.
- The conditional gate failure occurred only for seed 13,201 on one prompt.
  Its first-step registers were exactly `52` and `5863`, and the calculator
  emitted the correct first three symbols of `5915`. On the fourth generation
  pass, the learned digit confidence for operand B fell below admission and
  the calculator became inactive, after which base Qwen continued with
  unrelated digits. Seed 13,202 completed the same case exactly.
- This exposes a protocol design flaw rather than an addition error: the
  implementation re-decodes immutable prompt operands on every output token
  even though calculator operands should be captured once in a register.
- The other two addition failures were ordinary first-step framing errors:
  seed 13,201 decoded `6120` and `7` as `612` and `07`; seed 13,203 admitted
  an extra digit into the second operand of `32 + 27`.
- The revised architecture therefore passed four of five frozen gates and is
  not reported as an overall compound success.
- Raw record SHA-256 hashes:
  - seed 13,201:
    `bc259236b370e6f04be7a62baac7fa6846ae808fd3f8c15604eddd25fca0cd96`;
  - seed 13,202:
    `8022fb9f2319edf077efef7296c67e6230896c3e148983c9ef2941c2a6802b36`;
  - seed 13,203:
    `3f1be55849e368bdbb2230df4e288e869de224759e1f55f6bf89bee99ea76bf0`.
- Compact analysis:
  `phase7_results/router_hardened_audit4_summary.json`.
- Decision: preserve the failed conditional gate. Add a deterministic
  response-local operand register that captures the first valid typed
  operands once and reuses them for subsequent result symbols. Evaluate first
  on consumed audit 4, then freeze new families before treating it as evidence.

## 2026-07-26 — Post-audit operand-register development

- Added a deterministic response-local register that captures operand A,
  operand B, their lengths, and typed validity on the first active calculator
  step.
- Later result steps reuse the captured register instead of reclassifying
  every prompt token on every full-sequence generation pass.
- No checkpoint tensor, learned parameter, selected channel, route decision,
  result column, or Qwen weight changed.
- The exact seed-13,201 audit-4 failure `52 + 5863` changed from the truncated
  calculator sequence `5, 9, 1` followed by base-model continuation to the
  exact registered sequence `5, 9, 1, 5, EOS`.
- On the now-consumed complete audit-4 set for seed 13,201:
  - exact additions improved from 58/60 to 59/60;
  - exact first-step operands remained 59/60;
  - all 59 correctly captured cases had exact trajectories and outputs;
  - result ablation remained 1/60;
  - false routes remained 0/60;
  - token-exact negative preservation remained 60/60.
- The remaining error is the same wrong first-step operand framing and is not
  masked by the register.
- Raw post-audit development record:
  `phase7_results/router_hardened_audit4_seed_13201_operand_register_dev.json`.
- Raw-record SHA-256:
  `f27d00c3400746279202331480d754a95549db4990815f12484a62b842c6d36c`.
- This is development evidence on consumed prompts. Freeze a fifth
  exact-string-disjoint audit before claiming the register repair generalizes.

## 2026-07-26 — Frozen operand-register audit 5

- Froze `PHASE7_OPERAND_REGISTER_PROTOCOL.md`, the unchanged learned
  checkpoint hashes, required runtime-state flags, 20 new addition family
  strings, 30 new negative family strings, three data seeds, and five gates
  before generation.
- Addition results by independently learned seed:
  - seed 13,201: 57/60 exact, 58/60 exact operands, 58/60 exact trajectories,
    3/60 ablated;
  - seed 13,202: 58/60 exact, 58/60 exact operands, 58/60 exact trajectories,
    3/60 ablated;
  - seed 13,203: 58/60 exact, 58/60 exact operands, 58/60 exact trajectories,
    3/60 ablated.
- Across seeds:
  - exact additions: 173/180 (96.11%);
  - exact first-step operands: 174/180 (96.67%);
  - exact calculator trajectories: 174/180;
  - all 90/90 word problems exact;
  - result-ablation exact: 9/180;
  - paired causal drop: 165/180.
- All 174 examples with exact first-step operands kept identical operand
  register content across every generated step. No later operand drift
  occurred.
- One exact registered trajectory failed downstream decoding: for `0 + 0`,
  seed 13,201's calculator emitted exact symbols `0, EOS`, but its learned
  result columns and frozen downstream Qwen emitted `I`.
- The six operand failures were concentrated in two direct prompt instances
  using the same family. Every seed mis-framed `39 + 531` and `72 + 603`,
  showing a systematic learned role/digit interface weakness rather than
  random calculator error.
- Router/preservation results remained exact:
  - 0/180 false routes;
  - 180/180 negative generations token-identical to untouched Qwen.
- Gate disposition:
  - failed the three-seed mean addition gate: 57.67/60 versus the frozen
    58/60 requirement, although every seed passed its 57/60 floor;
  - passed operand recovery;
  - passed causal ablation;
  - passed routing and preservation;
  - failed all-exact conditional output due to the single `0 + 0` result
    decoder miss.
- The compound protocol passed three of five gates and is not reported as an
  overall success. The operand register itself behaved exactly as intended.
- Raw record SHA-256 hashes:
  - seed 13,201:
    `0acbcacd51c67579878a36c7741f96fc4ce87a921701ddc00e06afd29c69082d`;
  - seed 13,202:
    `79861af7af101139049ab7632793eb3b0e19a8f9c27861d1bfb195684b9d460e`;
  - seed 13,203:
    `2492fee048aef0303294c5c6f1ca03cdf911a008e0c032ca78bb1a500d288372`.
- Compact analysis:
  `phase7_results/operand_register_audit5_summary.json`.
- Interpretation: the narrow core hypothesis has strong replicated causal
  support. Remaining positive errors are learned input/output interface
  errors, while the deterministic calculator and registered state did not
  produce an observed wrong trajectory from an exact input.
