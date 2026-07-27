# Phase 7 Independent Held-Out Audit 2 Protocol

Status: frozen before any audit-2 generation.

Architecture snapshot: commit `d5f921a`.

Frozen checkpoint:
`phase7_artifacts/sequence_layer16_v1/neuron_implant_seed_13201.pt`.

Checkpoint SHA-256:
`5b76181c2c0f4a74b7482e4856b2b8c92bff637a1e70302bdfbc61ce1aaac41e`.

## Purpose

Test whether the improved in-place deterministic-neuron architecture
generalizes after audit v1 was consumed for decoder-layer selection, fixed
result-state engineering, and the typed role/digit handshake.

Audit v2 uses new exact-string-disjoint prompt constructions and new data
seeds. No checkpoint weight, channel selection, route threshold, parser rule,
generation rule, or prompt family may change after the audit begins.

This is an independent held-out development audit of one learned seed. It is
not yet a three-seed confirmatory study.

## Frozen system

- Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Decoder layer 16.
- 28 selected coordinates inside the existing 4,864-wide MLP activation.
- 16 neural input coordinates: 2 route, 3 operand role, 11 digit type.
- 12 deterministic calculator-result coordinates.
- Fixed response-local route latch and result-position counter.
- Operand admission requires role and digit-type agreement.
- Original selected-channel contribution is restored exactly whenever route is
  off.
- 14,336 learned input-row weights and 10,752 learned result-column weights:
  25,088 total.
- Zero learned calculator parameters.
- No change to Qwen's 896-wide residual stream or 4,864-wide MLP activation.
- Greedy generation for at most eight tokens.

The route latch and result counter are implemented as generation-runtime
microcircuit state in this version. They are not yet recurrent values stored
inside Qwen's residual stream, so this audit does not claim arbitrary
multi-call chain-of-thought execution.

## Frozen data

The exact deterministic definitions are in
`src/neural_firmware/phase7_data.py`.

- 30 direct/symbolic additions, seed 13,301.
- 30 addition word problems, seed 13,302.
- 60 adversarial non-addition prompts, seed 13,303.
- Both operands independently contain one to four digits.
- Audit-2 families are exact-string disjoint from all training, development,
  Phase 4/5 confirmation, and audit-v1 families.

## Outcomes

Positive prompts:

- mathematical exact-answer accuracy overall and by split;
- exact digits-only format;
- first-step route activation;
- first-step exact operand recovery;
- exact deterministic calculator trajectory;
- mathematical accuracy with only calculator-result activations ablated.

Negative prompts:

- any calculator route activation;
- exact generated-token equality with untouched Qwen over the same horizon.

Architecture:

- selected-channel count;
- learned replacement-weight count;
- calculator learned-parameter count;
- MLP and residual widths;
- checkpoint, implementation, and result hashes.

## Frozen engineering gates

Audit v2 strongly supports the narrow core mechanism only if all of the
following hold:

1. at least 54/60 additions (90%) are mathematically exact;
2. calculator-result ablation is at most 3/60 exact and lowers accuracy by at
   least 51/60 (85 percentage points);
3. at least 57/60 additions (95%) recover both operands exactly;
4. at most 1/60 negatives ever route to the calculator and at least 59/60
   preserve untouched Qwen tokens exactly;
5. every routed example with exact operands follows the exact deterministic
   calculator trajectory and produces the exact formatted answer.

These are engineering gates rather than statistical significance tests. A
pass justifies independent learned-seed replication. A failure must remain in
the record and be separated into routing, typed operand encoding, deterministic
execution, and output interpretation.

## Provenance

The research hypothesis and architecture direction are attributed to Josiah
Wilson. Implementation and analysis assistance are attributed to OpenAI Codex.
