# Phase 8 Executive Summary

## Result in one sentence

The deterministic-neuron implant transferred to frozen TinyLlama and
substantially outperformed untouched base and an exactly parameter-matched
adapter, but the frozen Phase 8 protocol failed because the learned semantic
interface generalized poorly to several new operand and negative-prompt
families.

## What was tested

Phase 8 is a preregistered second-model replication of the Phase 7
single-addition mechanism.

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Frozen decoder MLP layer: 15
- Existing MLP coordinates repurposed: 28 of 5,632
- Learned implant weights: 57,344
- Learned calculator weights: 0
- Matched control: rank-14 residual adapter, exactly 57,344 weights
- Independent training seeds: 14,201; 14,202; 14,203
- Frozen evaluation: 30 direct additions, 30 word problems, and 60
  adversarial non-addition prompts per seed

Every pretrained TinyLlama weight remained frozen. The input side was neural;
there was no fixed parser. The runtime retained the disclosed response-local
route latch, operand register, and deterministic result-position counter.

## Confirmatory result

| Metric | Seed 14,201 | Seed 14,202 | Seed 14,203 |
|---|---:|---:|---:|
| Implant exact additions | 53/60 | 53/60 | 53/60 |
| Direct additions | 26/30 | 26/30 | 26/30 |
| Word problems | 27/30 | 27/30 | 27/30 |
| Positive route predictions | 59/60 | 59/60 | 59/60 |
| Exact operand recovery | 54/60 | 54/60 | 54/60 |
| Exact calculator trajectories | 53/60 | 53/60 | 53/60 |
| Exact given active route and exact operands | 53/53 | 53/53 | 53/53 |
| Exact after result ablation | 0/60 | 0/60 | 0/60 |
| Paired correct-to-incorrect ablations | 53/60 | 53/60 | 53/60 |
| False routes on negatives | 12/60 | 12/60 | 9/60 |
| Token-exact negative preservation | 48/60 | 48/60 | 51/60 |
| Matched-adapter exact additions | 16/60 | 8/60 | 12/60 |

Untouched TinyLlama produced 0/60 exact numeral-only responses under the same
eight-token budget. With a separate 64-token sensitivity budget, it recovered
the correct final integer on 3/60 prompts but still produced 0/60 exact
numeral-only responses.

The implant's mean exact accuracy was 88.33%. The matched adapters achieved
26.67%, 13.33%, and 20.00%. The implant therefore gained 61.67–75.00
percentage points over an exactly parameter-matched ordinary learned
residual.

## Frozen verdict

The compound protocol **failed**.

Technical gates passed:

- causal result ablation;
- exact calculator trajectory and downstream decoding conditional on a valid
  interface;
- superiority to untouched base and the parameter-matched adapter.

Technical gates failed:

- every seed required at least 57/60 exact additions; each reached 53/60;
- every seed required at least 57/60 exact operands; each reached 54/60;
- every seed allowed at most 2/60 false routes and required at least 58/60
  exact negative preservation; observed false routes were 12, 12, and 9.

The procedural no-retraining/no-replacement gate was satisfied. No
confirmatory checkpoint, prompt, threshold, budget, or outcome was changed
after the protocol freeze.

## What the failure teaches

The same seven additions failed in all three seeds:

- one route remained off;
- six prompts failed the typed operand handshake;
- zero prompts failed downstream decoding after an active route with exact
  operands.

All 53 valid calculator trajectories in every seed decoded to the exact answer.
Ablation changed all 53 correct answers to incorrect answers. The
deterministic arithmetic mechanism therefore transferred cleanly; the neural
interface around it did not generalize well enough.

Negative failures were similarly systematic:

- multiplication near-misses: 9, 9, and 6 false routes out of 12;
- factual-number questions: 3/12 false routes in every seed;
- quoted arithmetic, negated requests, and explicit ignore-sum instructions:
  0 false routes in all seeds.

This is strong evidence against describing Phase 8 as a successful robust
replication. It is also evidence that the main remaining bottleneck is
semantic routing and operand typing rather than exact addition or native
downstream use of deterministic symbols.

## Efficiency and scope

The 57,344 learned weights are 0.005213% of TinyLlama's 1,100,048,384
parameters. The calculator itself has no learned weights. Median local
generation latency was 0.714–0.786 seconds per implant prompt versus 0.712
seconds for untouched base using the same deliberately uncached decoding
implementation. The maximum sampled allocated MPS memory was 4.40 GB; this is
not a device-driver peak watermark.

Phase 8 establishes neither arbitrary recurrent calculator use nor robust
semi-deterministic AI. It supplies a useful negative/partial replication:
deterministic neuron-shaped computation can transfer across Qwen and a
Llama-family model, but the learned interface remains brittle under semantic
distribution shift.

## Next research step

The next phase should target interface robustness, not add more arithmetic
operations yet:

1. train operation-contrast routing with hard multiplication and factual
   negatives;
2. train semantic-role diversity for reversed-order and novel word-problem
   templates;
3. preserve the current confirmation as a failed frozen audit;
4. freeze a new exact-string-disjoint audit before evaluating any repair;
5. only after a clean interface replication, move toward residual-native
   state, repeated invocation, and broader semi-deterministic mechanisms.

Raw confirmation rows are in `phase8_results/confirmation.json`; the
machine-generated post-hoc decomposition is in
`phase8_results/analysis.json`; the frozen protocol is
`PHASE8_SECOND_MODEL_REPLICATION_PROTOCOL.md`.
