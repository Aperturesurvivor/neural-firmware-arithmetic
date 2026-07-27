# Phase 7 Router-Hardening Audit Protocol

Status: frozen after all three hardened checkpoints existed and before any
audit-4 generation.

## Purpose

Test whether targeted training of only the two calculator-route rows repairs
the replicated adversarial-routing failure without changing operand
extraction, result reconstruction, calculator execution, or Qwen.

Audit 3 remains a failed compound result. Audit 4 is a new post-audit
confirmation of the revised router, not a retroactive repair of audit 3 and
not an independently preregistered external replication.

## Frozen checkpoints

All checkpoints use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, decoder layer 16, the same
28-channel bank, 0.90 digit-confidence handshake, fixed route latch and result
counter, and 25,088 learned interface weights.

Only the two route rows (1,792 weights) were updated after audit 3. The other
23,296 learned interface weights are byte-identical to their audit-3 parents.
The calculator and Qwen remain frozen.

- seed 13,201:
  `phase7_artifacts/sequence_layer16_router_hardened_v1/neuron_implant_seed_13201.pt`
  - SHA-256:
    `fa6deb3bfa8c7d4cf6af06255e35e777dcc18badc7e7eda0e519eac70c2f91a6`
- seed 13,202:
  `phase7_artifacts/sequence_layer16_router_hardened_v1/neuron_implant_seed_13202.pt`
  - SHA-256:
    `b483c3fbcec274cdf2f1b23acff33ae63966575c4d0f491eed3f182a73f24eea`
- seed 13,203:
  `phase7_artifacts/sequence_layer16_router_hardened_v1/neuron_implant_seed_13203.pt`
  - SHA-256:
    `26eee38d61dcad97e0c9e4c5252e1d7bdf8073ae667ea29a8e8836125e7edf71`

Every checkpoint uses a fixed 0.5 route threshold. No per-seed threshold,
channel, prompt, architecture, or decoding change is permitted after this
protocol is frozen.

## Development intervention

The revised route rows were trained on a family-balanced union of all prompt
families consumed through audit 3:

- 126 positive and 128 negative families;
- 4,064 training prompts;
- 1,016 exact-string-disjoint development prompts;
- 504/504 positive and 512/512 negative development decisions for every seed.

The audit-4 family strings and data seeds below were not present when the
hardened weights were trained or selected.

## Frozen audit-4 data

The exact definitions are in `src/neural_firmware/phase7_data.py`.

- 30 direct additions, seed 13,601.
- 30 addition word problems, seed 13,602.
- 60 adversarial non-addition prompts, seed 13,603.
- One-to-four-digit operands.
- Twenty new positive and thirty new negative family strings.
- All family strings are exact-string disjoint from every training,
  development, confirmation, and audit family consumed before the checkpoint
  freeze.
- Every seed receives identical examples in identical order.

Negatives include other operations, comparisons, labels, text manipulation,
quoted addition, refusal, explanation, hypothetical addition properties,
meta-arithmetic, and misleading prompts that mention addition while requesting
another task.

## Outcomes

Report separately for each learned seed:

- mathematical and digits-only exact addition accuracy;
- direct and word-problem accuracy;
- first-step routing and exact operand recovery;
- exact deterministic calculator trajectory;
- result-ablation accuracy and paired causal drop;
- negative false routes and token-exact preservation versus untouched Qwen;
- exact output conditional on an active route and exact operands;
- checkpoint/result hashes and learned-parameter counts.

Report mean, minimum, and maximum across the three seeds. Do not pool prompt
rows as independent model replications.

## Frozen engineering gates

The revised architecture passes this audit only if:

1. every seed reaches at least 54/60 exact additions and the three-seed mean
   reaches at least 57/60;
2. every seed recovers both operands on at least 57/60 additions;
3. every seed has a paired calculator-result ablation drop of at least 48/60;
4. every seed has at most 2/60 negative false routes and preserves at least
   58/60 negative generations exactly;
5. for every seed, every active-route example with exact operands has the
   exact calculator trajectory and exact formatted answer.

These are engineering reliability gates, not null-hypothesis significance
tests. Every failure remains in the project record.

## Claim boundary

A pass would show that the narrow supervised first-step router can be hardened
without altering the calculator or the rest of its learned interface. It would
not establish arbitrary-language safety, spontaneous discovery, multiple
calculator calls, or recurrent calculator use during unrestricted
chain-of-thought. The route latch and result counter remain generation-runtime
state.

## Provenance

The research hypothesis and architecture direction are attributed to Josiah
Wilson. Implementation and analysis assistance are attributed to OpenAI Codex.
