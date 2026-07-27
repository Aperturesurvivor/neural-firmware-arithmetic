# Phase 7 Held-Out Audit Protocol

Status: frozen before generation evaluation.

Frozen development commit: `487d697`.

Frozen checkpoint:
`phase7_artifacts/sequence_interface_v2/neuron_implant_seed_12811.pt`.

Checkpoint SHA-256:
`fc5a547033ebe1a8fbe9888fa5a5549c0b0592f0e9524a7628e30a2bcee41d6a`.

## Purpose

Test whether Josiah Wilson's in-place deterministic-neuron hypothesis survives
new natural-language constructions that were not used to train, tune, or
select the Phase 7 v2 interface.

This is a held-out development audit, not a multi-seed confirmatory study. Its
results may determine the next engineering change, but the checkpoint,
threshold, channel bank, prompt families, seeds, and evaluation rules below
must not change after the audit begins.

## Frozen system

- Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Decoder layer 23.
- 34 existing MLP activation channels.
- Four-digit maximum per operand; addition only.
- 30,464 learned replacement-row/column weights.
- Zero learned calculator parameters.
- First generated-step route latch enabled.
- Exact original selected-channel contribution restored whenever route is off.
- Greedy generation for at most eight tokens.

## Frozen data

The reproducible family definitions and builders are in
`src/neural_firmware/phase7_data.py`.

- 20 direct/symbolic additions, seed 12,901.
- 20 addition word problems, seed 12,902.
- 40 adversarial non-addition prompts, seed 12,903.
- Both operands independently contain one to four digits.
- All prompt families are exact-string disjoint from Phase 7 interface
  training/development families and earlier Phase 4/5 confirmation families.

## Outcomes

Positive prompts:

- mathematical exact-answer accuracy overall and by symbolic/word split;
- digits-only format accuracy;
- first-step route activation;
- first-step exact operand recovery;
- exact deterministic result-symbol trajectory;
- mathematical accuracy with only calculator-result activations ablated.

Negative prompts:

- any calculator route activation during generation;
- exact generated-token equality with untouched Qwen over the same horizon.

Architecture:

- learned-parameter count;
- calculator learned-parameter count;
- MLP and residual widths;
- checkpoint and result hashes.

## Development interpretation gates

The audit is treated as strong support for the narrow core mechanism if all of
the following hold:

1. at least 70% of held-out additions are mathematically exact;
2. result ablation lowers exact accuracy by at least 50 percentage points;
3. at least 95% of adversarial negatives preserve untouched Qwen tokens;
4. no deterministic addition is wrong when the typed operands and result
   position are correct.

These thresholds are engineering gates, not statistical significance claims.
Passing them would justify independent-seed replication. Failing any gate
requires retaining the result and diagnosing routing, operand decoding,
position tracking, deterministic execution, and output interpretation
separately.

## Provenance

The research hypothesis and architecture direction are attributed to Josiah
Wilson. Implementation and analysis assistance are attributed to OpenAI Codex.
