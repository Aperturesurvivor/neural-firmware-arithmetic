# Phase 7 Three-Seed Shared-Holdout Protocol

Status: frozen after all three checkpoints existed and before any audit-3
generation.

## Frozen checkpoints

All checkpoints use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, decoder layer 16, the same
28-channel bank, the 0.90 digit-confidence handshake, and independently learned
25,088-weight interfaces.

- seed 13,201:
  `phase7_artifacts/sequence_layer16_confident_v1/neuron_implant_seed_13201.pt`
  - SHA-256:
    `9dba639d127769b08579b2e1deabdfd3d232e06dcf2ea6f843f7b9963855785c`
- seed 13,202:
  `phase7_artifacts/sequence_layer16_multiseed/neuron_implant_seed_13202.pt`
  - SHA-256:
    `6cab7608a912d19a26793828352cdffd8783e2e5f6b8bdcad65f5afdf22b6b07`
- seed 13,203:
  `phase7_artifacts/sequence_layer16_multiseed/neuron_implant_seed_13203.pt`
  - SHA-256:
    `cbf84806b08f0804cd0c508330e0f5f9fdfa72b41c800a37aef616c682fb58dd`

No per-seed route threshold, digit threshold, channel, prompt, or decoding
change is permitted after this protocol is frozen.

## Purpose

Test whether the in-place deterministic-neuron result is stable across three
independent initializations and training streams, rather than being a lucky
single interface.

This is a confirmatory-style shared holdout after extensive pilot development.
It is stronger than the single-seed audits but is not represented as an
independently preregistered external replication.

## Frozen architecture

- 28 selected coordinates inside the existing 4,864-wide Qwen MLP activation.
- 16 learned neural input coordinates: route, operand role, and digit type.
- 12 deterministic calculator-result coordinates.
- Fixed route latch and result-position counter.
- Role/digit agreement plus 0.90 learned digit confidence for operand
  admission.
- Exact restoration of original selected-channel contribution when route is
  off.
- 14,336 learned input-row weights and 10,752 learned result-column weights.
- Zero learned calculator parameters.
- Unchanged 896-wide residual stream and 4,864-wide MLP activation.
- Greedy generation for at most eight tokens.

The latch and counter remain generation-runtime microcircuit state rather than
residual-stream recurrent state. Arbitrary chain-of-thought multi-call use is
outside this protocol.

## Shared audit-3 data

The exact deterministic definitions are in
`src/neural_firmware/phase7_data.py`.

- 30 direct/symbolic additions, seed 13,401.
- 30 addition word problems, seed 13,402.
- 60 adversarial non-addition prompts, seed 13,403.
- One-to-four-digit operands.
- All families are exact-string disjoint from training, development, earlier
  confirmation, and audit 1/2 families.
- Every seed receives the identical examples in the identical order.

## Outcomes

For each seed and aggregated across seeds:

- mathematical and format-exact addition accuracy;
- direct and word-problem accuracy;
- first-step routing and exact operands;
- exact deterministic trajectory;
- calculator-result ablation accuracy and paired accuracy drop;
- negative false routes and exact token preservation;
- conditional accuracy given an active route and exact operands;
- learned-parameter count, calculator-parameter count, tensor widths, hashes,
  and runtime.

Report the per-seed values, mean, minimum, and maximum. Do not pool the 180
positive rows as though they were 180 independent model replications.

## Frozen engineering gates

The multi-seed result supports the narrow core hypothesis only if:

1. every seed reaches at least 51/60 exact additions (85%) and the three-seed
   mean is at least 54/60 (90%);
2. every seed recovers both operands on at least 57/60 additions (95%);
3. every seed's paired result-ablation drop is at least 45/60 (75 percentage
   points);
4. every seed has at most 2/60 negative false routes and preserves at least
   58/60 negative generations exactly;
5. for every seed, all active-route examples with exact operands have exact
   calculator trajectories and exact formatted answers.

These are engineering reliability gates, not null-hypothesis significance
tests. Every failed seed or gate remains part of the record.

## Provenance

The research hypothesis and architecture direction are attributed to Josiah
Wilson. Implementation and analysis assistance are attributed to OpenAI Codex.
