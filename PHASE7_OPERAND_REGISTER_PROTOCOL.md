# Phase 7 Operand-Register Audit Protocol

Status: frozen after the register implementation and consumed audit-4
diagnostic, before any audit-5 generation.

## Purpose

Test whether capturing correctly decoded operands once in deterministic
calculator state eliminates sequence-time operand drift while preserving the
hardened router result.

Audits 3 and 4 retain their original failed compound verdicts. Audit 5 is a
post-audit confirmation of a revised runtime microcircuit, not a retroactive
repair and not an independently preregistered external replication.

## Frozen learned checkpoints

The learned checkpoints are byte-for-byte the hardened-router checkpoints from
audit 4:

- seed 13,201:
  `fa6deb3bfa8c7d4cf6af06255e35e777dcc18badc7e7eda0e519eac70c2f91a6`;
- seed 13,202:
  `b483c3fbcec274cdf2f1b23acff33ae63966575c4d0f491eed3f182a73f24eea`;
- seed 13,203:
  `26eee38d61dcad97e0c9e4c5252e1d7bdf8073ae667ea29a8e8836125e7edf71`.

Each uses Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, decoder layer 16, the same
28-channel bank, 25,088 learned interface weights, 0.90 digit confidence,
0.5 route threshold, and zero-learned-parameter addition circuit. Qwen and the
calculator remain frozen.

## Frozen register change

When the first generated position has an active route and valid typed
operands, the deterministic runtime captures:

- operand A digits and length;
- operand B digits and length;
- typed validity.

Later answer positions reuse those values rather than re-running operand
admission on the expanding sequence. The first route and operand predictions
are unchanged. Invalid first-step operands are not corrected or hidden.

The operand register, route latch, and result-position counter are
generation-runtime microcircuit state. They are not yet encoded in recurrent
Qwen residual activations. `--latch-operands`, `--deterministic-result-step`,
and route latching are mandatory for this protocol.

## Frozen audit-5 data

Exact definitions are in `src/neural_firmware/phase7_data.py`.

- 30 direct additions, seed 13,701.
- 30 addition word problems, seed 13,702.
- 60 adversarial non-addition prompts, seed 13,703.
- One-to-four-digit operands.
- Twenty new positive and thirty new negative family strings.
- Every family string is exact-string disjoint from all prior training,
  development, confirmation, and audit families.
- All three seeds receive identical examples in identical order.

## Outcomes

Report per seed and as mean/minimum/maximum:

- mathematical and digits-only exact addition accuracy;
- direct and word-problem accuracy;
- first-step route and exact operand recovery;
- exact calculator digit-plus-EOS trajectory;
- exact results conditional on an active route and exact initial operands;
- result-ablation accuracy and paired causal drop;
- negative false routes and token-exact preservation;
- register activation and stability traces;
- learned/calculator parameter counts and hashes.

Do not pool prompt rows as independent model replications.

## Frozen engineering gates

The revised architecture passes only if:

1. every seed reaches at least 57/60 exact additions and the three-seed mean
   reaches at least 58/60;
2. every seed recovers both operands on at least 57/60 additions;
3. every seed has a paired calculator-result ablation drop of at least 50/60;
4. every seed has at most 2/60 negative false routes and preserves at least
   58/60 negative generations exactly;
5. for every seed, every active-route example with exact first-step operands
   has an exact registered calculator trajectory and exact formatted answer.

All failures remain in the record.

## Claim boundary

A pass would establish that a fixed operand register repairs the observed
single-response stability failure on new prompts. It would not establish
arbitrary-language safety, recurrent residual-native state, spontaneous
discovery, multiple calculator calls, or unrestricted calculator use during
chain-of-thought.

## Provenance

The research hypothesis and architecture direction are attributed to Josiah
Wilson. Implementation and analysis assistance are attributed to OpenAI Codex.
