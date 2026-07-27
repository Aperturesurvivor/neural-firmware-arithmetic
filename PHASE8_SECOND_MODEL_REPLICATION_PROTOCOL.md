# Phase 8 Second-Model Replication Protocol

Status: frozen on 2026-07-27 before confirmatory model training or evaluation.

Frozen implementation commit:
`54dfc1c7086a8fd7595bc5fbd5f52a69d43cb45a`.

The deterministic data tests instantiated confirmatory rows to verify counts,
category balance, determinism, and split disjointness before this freeze. No
confirmatory prompt was inspected to tune a model, and no base, adapter, or
implant checkpoint was run on those rows.

## Purpose

Test whether the Phase 7 deterministic-neuron result transfers from
Qwen2.5-0.5B-Instruct to an independently pretrained Llama-family decoder
without changing the scientific claim after seeing confirmatory outcomes.

This is a replication of the single-addition mechanism, not the multi-call
controller experiment. Keeping those questions separate prevents a failed
controller from obscuring the cross-model result.

## Model choice

Frozen target:
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at Hugging Face revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`.

Local file SHA-256 values:

- `config.json`:
  `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`
- `tokenizer.json`:
  `bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e`
- `tokenizer_config.json`:
  `7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a`
- `model.safetensors`:
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`

Reasons:

- it uses the Llama 2 architecture and tokenizer;
- it is approximately 1.1B parameters, meaningfully different from Qwen while
  remaining feasible on the available 16 GB Apple M4 machine;
- it is publicly downloadable without Meta's gated-license acceptance;
- its MLP exposes a compatible intermediate activation in which an in-place
  deterministic subspace can be assigned.

The model has 22 decoder layers, hidden width 2,048, MLP width 5,632, and a
Llama-family SwiGLU MLP. A branded Meta Llama replication remains a later
license-gated extension and is not part of Phase 8.

## Frozen research question

Can a compact learned interface make a frozen Llama-family model:

1. identify addition requests in varied natural language;
2. encode the intended operands into reserved MLP activation coordinates;
3. consume exact symbols from a zero-learned-parameter adder;
4. decode the result through the model's ordinary downstream layers;
5. preserve untouched-base outputs when the route is off?

## Conditions

Run all conditions with greedy decoding on identical frozen prompts and an
eight-token primary response budget:

1. untouched base model;
2. ordinary learned adapter with the same learned-parameter count as the
   implant;
3. deterministic-neuron implant;
4. implant with calculator-result coordinates ablated.

No fixed parser is present in the implant. The input interface is neural.
The response-local route latch, operand register, and result-position counter
may be reused exactly as architectural microcircuit state and must be disclosed
as runtime-managed state.

## Architecture constraints

- Freeze every pretrained model parameter.
- Use decoder MLP layer 15, selected from development-only probes at layers 12,
  15, and 18.
- Reserve these 28 existing intermediate coordinates, selected by the
  development census:
  `4992, 5542, 1144, 2339, 3284, 2618, 5266, 3464, 1008, 3771, 4049,
  3734, 3321, 2027, 3873, 4940, 5021, 503, 1769, 4345, 3536, 4851,
  2274, 1180, 4632, 5252, 4008, 3466`.
- Assign the coordinates as:
  2 route, 3 role, 11 digit/non-digit, and 12 result-symbol coordinates.
- Preserve the model's hidden and MLP widths.
- Use a frozen decimal ripple-carry adder with zero learned parameters.
- Train only the selected route/role/digit rows and result down-projection
  columns.
- Match the learned control parameter count exactly, not approximately.
- Record both absolute learned weights and the percentage of base parameters.

The frozen learned budget is `28 * 2,048 = 57,344` weights: 32,768 input
weights and 24,576 result-column weights. The zero-parameter calculator is not
included in that learned budget. The ordinary control is a rank-14,
bias-free GELU residual bottleneck at the same layer:
`2 * 2,048 * 14 = 57,344` learned weights.

The selected-coordinate ablation preserved all 96 development next-token
argmaxes, with mean KL divergence `0.0001709781`. The layer-15 probe achieved
perfect operand-role and digit accuracy, 96.98% route recall, and 0.83% route
false positives on its development subset.

## Data isolation

The generator is frozen in `src/neural_firmware/phase8_data.py`. Its Git blob
at the implementation freeze is
`99e0410751804de3cfc58fcd59b3d4ba198457ee`.

Frozen data seeds:

- training positives: `14101`;
- training negatives: `14102`;
- development positives: `14103`;
- development negatives: `14104`;
- confirmatory direct additions: `14701`;
- confirmatory word problems: `14702`;
- confirmatory negative category base seed: `14703`, with the committed
  per-category and per-family offsets.

All final family strings must be exact-string disjoint from every Phase 7 and
Phase 8 training/development family. All numeric operand pairs must also be
disjoint across Phase 8 training, development, and confirmation.

Training contains 1,200 positives and 1,200 adversarial negatives.
Development contains 240 positives and 240 negatives. The three training seeds
receive identical examples in identical order. Numeric operand pairs are
unique within each constructed split and disjoint across training,
development, and confirmation.

## Confirmatory sample

Per seed:

- 30 direct addition prompts;
- 30 addition word problems;
- 60 adversarial non-addition prompts, balanced across:
  quoted arithmetic, negated requests, multiplication near-misses, factual
  questions containing numbers, and instructions to ignore an embedded sum.

Use one-to-four-digit nonnegative operands to preserve comparability with Phase
7. Add a preregistered longer-number sensitivity set only as secondary
analysis.

## Frozen training configuration

Independent seeds: `14201`, `14202`, and `14203`.

Implant, per seed:

1. train the 16 input rows for 3,000 AdamW updates, batch size 256,
   learning rate 0.001, with equal route/role/digit loss weights and zero
   learned-step loss;
2. train the 12 result columns for 200 AdamW updates, batch size 1,
   learning rate 0.01, with teacher routing and operands;
3. retrain only the two route rows for 3,000 AdamW updates, batch size 256,
   learning rate 0.001;
4. select the route threshold on development data with a maximum permitted
   development false-positive rate of 2.5%;
5. use digit confidence threshold 0.9 and result strength 16.0.

Matched adapter, per seed:

- rank 14, GELU, bias-free, zero-initialized up projection;
- 800 AdamW updates, batch size 4, learning rate 0.002;
- positive teacher-forced answer digits and end token;
- negative preservation target equal to untouched base's first greedy token.

There is no scheduler. Every pretrained weight remains frozen.

Runtime for the implant freezes the response-local route latch, first-step
operand register, deterministic result-position counter, and exact base-path
restoration when the route is off.

## Primary outcomes

Report per seed and across seeds:

- exact numeral-only addition accuracy;
- direct and word-problem accuracy;
- route recall on positives;
- exact first-step operand recovery;
- exact calculator digit-plus-end trajectory;
- exact output conditional on exact operands and route;
- result-ablation accuracy and paired causal loss;
- false routes on every negative family;
- token-exact preservation relative to untouched base;
- untouched-base exact response under the identical token budget;
- ordinary matched-adapter exact response;
- latency, peak memory, and learned-parameter count.

Prompt rows are repeated measurements, not independent model replications.
Report seed-level outcomes and a paired prompt-by-seed bootstrap only as a
secondary interval.

## Frozen confirmatory gates

1. every implant seed reaches at least 57/60 exact additions and the mean
   reaches at least 58/60;
2. every seed recovers exact operands on at least 57/60 additions;
3. every seed loses at least 50/60 paired correct answers under result
   ablation;
4. every seed has at most 2/60 negative false routes and at least 58/60
   token-exact negative preservation;
5. every active-route example with exact registered operands has an exact
   calculator trajectory, and at least 59/60 such examples decode exactly;
6. the implant exceeds both untouched base and the matched learned adapter in
   exact response accuracy for every seed;
7. no unreported checkpoint replacement, prompt-family revision, threshold
   change, or post-confirmation retraining occurs.

The conditional-output gate is set to 59/60 rather than logical perfection
because Phase 7 showed one downstream decoding error despite an exact
trajectory. Report any such error individually.

## Retained development outcomes

All development outcomes are retained under `phase8_results/`.

- Initial implant pilot seed 14199: 35/40 exact, 40/40 exact operands,
  0/40 ablation exact, 0/40 false routes, 40/40 negative preservation.
- Router-hardened pilot: 39/40 exact, 0/40 ablation exact, 0/40 false routes,
  40/40 negative preservation.
- Matched rank-14 adapter pilot: 10/40 exact additions and 7/40 exact negative
  preservation after its frozen 800-update budget.

These pilot rows are development data and are not included in the
confirmatory outcome.

## Claim boundary

A pass establishes cross-model transfer of a supervised, single-addition,
runtime-registered deterministic MLP subspace. It does not establish:

- arbitrary recurrent calculator calls;
- spontaneous discovery from task loss alone;
- residual-native persistent state;
- subtraction, multiplication, or division;
- universal prompt safety;
- transfer to large production models.

The broader research program is now framed as **semi-deterministic AI**:
combining learned neural interpretation and flexible reasoning with small,
auditable deterministic mechanisms embedded in native activation pathways.
Board-game state transitions are no longer the next program target. Phase 8
tests only the cross-model arithmetic primitive needed to justify that broader
direction.

All failed pilots, failed frozen gates, and post-hoc analyses remain in the
record.

## Provenance

The deterministic-neuron hypothesis and research direction originate with
Josiah Wilson. Experimental design, implementation, execution, analysis, and
manuscript assistance involve OpenAI Codex under his direction.
