# Phase 8 Lab Notebook

## Scope and status

Phase 8 tested whether the Phase 7 in-place deterministic-neuron addition
mechanism transfers from Qwen2.5-0.5B-Instruct to a separately pretrained
Llama-family model. It is complete. The frozen compound protocol failed, and
that failure is retained.

The broader program direction was also clarified before confirmation:
deterministic arithmetic is one controlled instance of **semi-deterministic
AI**, in which learned neural interpretation and flexible reasoning coexist
with small auditable deterministic mechanisms embedded in native activation
pathways. Board-game state transitions are no longer the designated next
domain.

## Model and tokenizer audit

TinyLlama was pinned to:

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

revision:

`fe8a4ea1ffedaf415f4da2f062534de366a451e6`

The model has 22 decoder layers, hidden width 2,048, and SwiGLU MLP width
5,632. It loaded locally in float32 on Apple MPS.

TinyLlama's SentencePiece tokenizer encodes a bare digit with a leading
boundary token, although its vocabulary contains context-independent
single-digit tokens. The shared digit-token resolver was generalized to use
those direct vocabulary entries when bare encoding contains the boundary
token. Natural prompt operands were verified to remain one activation-bearing
token per decimal digit. No parser was added at runtime.

## Development-only layer and channel selection

Candidate layers 12, 15, and 18 were probed on 600 training and 240
development prompts. The selection score was the minimum of positive route
recall, negative route specificity, operand-role accuracy, and digit accuracy.

| Layer | Selection score | Route recall | False-route rate | Operand role | Operand digit |
|---|---:|---:|---:|---:|---:|
| 12 | 0.9472 | 0.9472 | 0.0000 | 0.9976 | 1.0000 |
| 15 | 0.9698 | 0.9698 | 0.0083 | 1.0000 | 1.0000 |
| 18 | 0.9208 | 0.9208 | 0.0083 | 0.9960 | 1.0000 |

Layer 15 was selected. A channel census chose 28 low-impact existing MLP
coordinates. Ablating them before installing the implant preserved all 96
development next-token argmaxes; mean KL divergence was 0.000171.

## Retained pilots

### Initial implant pilot, seed 14,199

- exact additions: 35/40;
- exact operands: 40/40;
- exact calculator trajectories: 35/40;
- ablation exact: 0/40;
- false routes: 0/40;
- token-exact negative preservation: 40/40.

Every active route decoded the deterministic result exactly. Conservative
first-step routing caused the five misses.

### Router-only hardening

Only the two route rows were retrained on first-response development features.
The hardened pilot reached:

- exact additions: 39/40;
- ablation exact: 0/40;
- false routes: 0/40;
- token-exact negative preservation: 40/40.

### Matched-adapter pilot

A conventional rank-14 residual adapter at layer 15 contained exactly 57,344
weights, matching the implant. After the retained 800-update budget:

- exact additions: 10/40;
- token-exact negative preservation: 7/40.

The poor preservation result was retained and the budget was not changed after
confirmation was frozen.

## Freeze

Implementation commit:

`54dfc1c7086a8fd7595bc5fbd5f52a69d43cb45a`

Protocol-freeze commit:

`495c2f8f28a0df2c60917f47b764b94cc1c90a02`

Final seeds: 14,201; 14,202; 14,203.

Final data: 60 additions and 60 adversarial negatives, exact-string-disjoint
from development families and with disjoint operand pairs. Negative categories
were balanced at 12 each across quoted arithmetic, negated requests,
multiplication near-misses, factual questions containing numbers, and
instructions to ignore an embedded sum.

## Confirmatory result

All three implants produced the same exact/not-exact pattern on additions:
53/60 exact, comprising 26/30 direct prompts and 27/30 word problems.

Each recovered exact operands on 54/60. All 53 examples with an active route
and exact operands had an exact calculator digit-plus-end trajectory and exact
final response. Result ablation reduced every seed to 0/60, creating 53 paired
causal losses per seed.

The adversarial router did not transfer cleanly. False routes were 12/60,
12/60, and 9/60; token-exact preservation was consequently 48/60, 48/60, and
51/60.

Untouched TinyLlama returned 0/60 strict exact responses at eight tokens and
3/60 loose final-integer recoveries at 64 tokens. Matched adapters returned
16/60, 8/60, and 12/60 exact responses.

## Confirmatory failures

The same seven positive prompts failed in every seed.

- One direct-addition prompt was classified route-off.
- Three reversed-order “move forward” prompts predicted addition intent but
  failed to produce a valid typed operand register.
- Three prompts from one unseen telescope word-problem family failed operand
  recovery; two still activated and emitted wrong sums from wrong registered
  digits.
- There were no downstream decoding failures conditional on exact operands.

The false-route categories were:

| Negative category | Seed 14,201 | Seed 14,202 | Seed 14,203 |
|---|---:|---:|---:|
| Quoted arithmetic | 0/12 | 0/12 | 0/12 |
| Negated request | 0/12 | 0/12 | 0/12 |
| Multiplication near-miss | 9/12 | 9/12 | 6/12 |
| Factual numbers | 3/12 | 3/12 | 3/12 |
| Ignore embedded sum | 0/12 | 0/12 | 0/12 |

The multiplication failures generally emitted the operands' sum, directly
showing that the deterministic calculator was invoked with the wrong semantic
operation. The factual-number failures came from one template asking how many
identifiers were mentioned; the router incorrectly treated the two identifiers
as addends.

## Gate audit

Passed:

- paired causal ablation;
- exact trajectory and exact output conditional on a valid interface;
- better exact accuracy than untouched base and the matched adapter in every
  seed;
- procedural freeze/no-retraining requirement.

Failed:

- exact addition threshold;
- exact operand threshold;
- adversarial routing and preservation threshold.

The scientific verdict is a partial cross-model replication and a failed
compound protocol.

## Metadata note

The staged implant-training JSON reports `learned_parameters: 24576` after the
output stage because its helper counted only tensors whose `requires_grad`
flag remained true at that moment. This is a reporting artifact. Each
checkpoint contains 32,768 trained input weights and 24,576 trained result
weights, for the correct architectural total of 57,344. The analysis file
records and verifies the tensor shapes and checkpoint hashes.

## Post-hoc interpretation

The deterministic mechanism itself transferred:

- exact registered state always produced an exact calculator trajectory;
- every such trajectory decoded correctly;
- result ablation removed every correct answer;
- the implant greatly exceeded both learned controls.

The neural interface did not transfer robustly:

- the positive errors were shared across all seeds and prompt families;
- the negative errors clustered in operation-contrast families;
- seed variation did not repair systematic semantic blind spots.

The next experiment should repair the interface using development-only
operation-contrast and semantic-role diversity, then evaluate on a newly
frozen audit. The current confirmation must remain untouched and must not be
reclassified as a success.
