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
