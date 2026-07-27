# Phase 9 Executive Summary

## Result in one sentence

Hard semantic-contrast training improved TinyLlama's operand selection and
reduced the frozen Phase 8 implant's false routes, but it did not improve
end-to-end addition accuracy and was worse than generic continuation on
negative routing; the fixed linear interface traded one group of semantic
families for another rather than becoming broadly reliable.

## What was tested

Phase 9 is a frozen three-seed interface-hardening study of the existing
TinyLlama deterministic-neuron implant.

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Frozen decoder MLP layer: 15
- Existing MLP coordinates repurposed: 28 of 5,632
- Architectural learned weights: 57,344
- Phase 9 updated weights: 32,768 input-interface weights only
- Frozen Phase 8 result weights: 24,576, verified bit-identical
- Learned calculator weights: 0
- Continuation seeds: 15,201; 15,202; 15,203
- Conditions: untouched base, matched adapter, frozen Phase 8 implant,
  generic continuation, hard-contrast continuation, and result ablation
- Sealed audit: 100 additions and 200 adversarial negatives

Generic and hard continuation used the same number of examples, source
checkpoints, trainable weights, optimizer schedule, development set, and
threshold procedure. Only their curricula differed.

## Confirmatory result

| Metric | Seed 15,201 | Seed 15,202 | Seed 15,203 |
|---|---:|---:|---:|
| Hard exact additions | 61/100 | 61/100 | 58/100 |
| Hard direct additions | 21/50 | 22/50 | 21/50 |
| Hard word problems | 20/25 | 20/25 | 20/25 |
| Hard distractor additions | 20/25 | 19/25 | 17/25 |
| Hard positive route predictions | 76/100 | 76/100 | 76/100 |
| Hard active positive routes | 72/100 | 71/100 | 69/100 |
| Hard exact operand registers | 85/100 | 84/100 | 81/100 |
| Conditional trajectory and decode | 61/61 | 60/60 | 57/57 |
| Hard exact after result ablation | 0/100 | 0/100 | 0/100 |
| Hard paired causal losses | 61/100 | 61/100 | 58/100 |
| Hard false routes | 12/200 | 12/200 | 11/200 |
| Hard token-exact preservation | 188/200 | 188/200 | 189/200 |
| Generic exact additions | 59/100 | 63/100 | 58/100 |
| Generic false routes | 8/200 | 8/200 | 8/200 |
| Frozen Phase 8 exact additions | 67/100 | 68/100 | 68/100 |
| Frozen Phase 8 false routes | 30/200 | 26/200 | 27/200 |
| Matched-adapter exact additions | 21/100 | 14/100 | 20/100 |

Untouched TinyLlama produced 0/100 exact numeral-only answers under the same
eight-token budget.

The hard condition averaged 60.0% exact accuracy, exactly equal to the generic
condition's 60.0% mean and below the frozen Phase 8 implant's 67.67%. It still
substantially exceeded the matched adapters' 18.33% mean, but that descriptive
control was not the Phase 9 success criterion.

## Frozen verdict

The compound protocol **failed**.

The protocol-defined conditional-mechanism gate passed:

- every active-route, exact-operand hard example had an exact calculator
  trajectory and exact decoded answer.

Five gates failed:

- accuracy required at least 95/100 per seed; observed 58–61;
- operand recovery required at least 97/100; observed 81–85;
- routing required at most 4/200 false routes and at least 196/200 preserved
  negatives; observed 11–12 and 188–189;
- causal ablation required at least 90 paired losses; only 58–61 correct
  answers were available to remove, although ablation removed every one;
- hard continuation had to exceed generic accuracy and have no more false
  routes in every seed; it did neither.

The procedural freeze was satisfied. No confirmatory seed, checkpoint,
threshold, prompt, family, gate, or decoding budget was replaced.

## Disclosed evaluator bookkeeping bug

The frozen evaluator's raw JSON marks the conditional-mechanism gate false.
Its implementation compared the total number of exact trajectories with the
number of active-route, exact-string-operand cases. Seeds 15,202 and 15,203
each had one extra exact trajectory from the register `047 + 150`, which is
numerically correct but not the exact target register `47 + 150`. The unequal
counts made the raw Boolean false.

Direct row-level recomputation of the gate as written in the protocol gives
61/61, 60/60, and 57/57 exact trajectories and decoded answers, so the
protocol-defined gate passes. Both the raw implementation flag and corrected
audit are retained. The compound verdict is unchanged because the other five
gates fail.

## What the failure teaches

### Hard contrasts helped real subproblems

Relative to the frozen Phase 8 implant, hard continuation:

- increased exact operand registers by 7–12 prompts per seed;
- reduced false routes by 14–18 out of 200;
- improved token-exact negative preservation by 10–12;
- learned 17–20/25 additions containing an irrelevant identifier, compared
  with 0/25 for the frozen Phase 8 interface.

These are genuine improvements in state typing and semantic specificity.

### The improvements did not compose

The hard router became too conservative on unfamiliar positive language:

- exactly 24/100 additions were routed off in every seed;
- direct-addition accuracy collapsed to 21–22/50;
- four confirmation families were routed off for all 20 examples even though
  their operands were usually recovered exactly;
- 38 positive failures were shared by all three seeds.

Generic continuation was also weak, but it preserved 196/200 negatives and
made only 8/200 route predictions on negatives. Hard continuation introduced
systematic false routes on multiplication (7/20 per seed), canceled addition
(3–4/20), and concatenation (1/20). The targeted curriculum therefore moved
the linear decision boundary rather than producing broad intent
understanding.

### The calculator was not the failure

For every protocol-defined hard example with an active route and exact
operands, the deterministic calculator trajectory and final decoded answer
were exact. Ablating only the result coordinates reduced all three seeds to
0/100 exact answers. The remaining bottleneck is the frozen neural interface:
semantic routing, typed activation, and exact operand representation.

## Efficiency and scope

The architecture contains 57,344 learned weights, 0.005213% of TinyLlama.
Phase 9 updates only 32,768 weights, 0.002979% of the base. The calculator
has zero learned parameters.

Median hard-condition latency was 1.018–1.039 seconds per prompt versus 1.135
seconds for untouched base in this deliberately uncached local implementation.
The maximum sampled allocated MPS memory was 4.40 GB; this is not a
driver-level peak watermark.

Phase 9 does not test arbitrary recurrent calls, multiple operations,
residual-native state, nonlinear routing, or surrounding-layer adaptation.
It is evidence against the sufficiency of this frozen linear interface—not
against deterministic computation embedded in neural activation pathways.

## Research implication

The next experiment should test interface capacity and representation
directly while keeping the calculator fixed. A clean design would compare:

1. the present linear interface;
2. a small parameter-matched nonlinear route/role/digit interface;
3. limited adaptation of the surrounding TinyLlama representation;
4. the same frozen audit logic with explicit family-level positive-recall
   constraints.

Only after semantic routing and operand typing improve should the project
advance to repeated calculator invocation or multiple operations.

Raw prompt-level outputs are in `phase9_results/confirmation.json`; the flat
table is `phase9_results/confirmation_rows.csv`; the post-hoc analysis is
`phase9_results/analysis.json`; and the frozen protocol is
`PHASE9_INTERFACE_HARDENING_PROTOCOL.md`.
