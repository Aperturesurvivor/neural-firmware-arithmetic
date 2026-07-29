# Phase 11 Executive Summary

## Outcome

Phase 11 is a confirmatory partial success and an overall protocol failure.

A dedicated 4,096-weight request router materially improved autonomous
addition execution over the paired Phase 10 interface router on a new sealed
audit, without changing any inherited operand, calculator, decoder, or
pretrained-model tensor. Exact answers improved:

- seed 16,201: 49/100 to 67/100;
- seed 16,202: 46/100 to 77/100;
- seed 16,203: 51/100 to 68/100.

The paired gains were +18, +31, and +17. Across seed-prompt evaluations, the
candidate was exact on 212/300 positives versus 146/300 for Phase 10. It won
on 66 paired examples and lost on none. A two-way seed-and-prompt bootstrap
estimated a mean paired gain of 0.22 with a 95% percentile interval of
0.127--0.330.

This improvement did not meet the frozen reliability gates. Candidate
autonomous exactness was below 70/100 in two seeds and its mean was 70.7/100,
below the required 75. Positive routing was only 75/100, 79/100, and 72/100,
below the required 80 in every seed. False routes were 8/200, 9/200, and
6/200, exceeding the maximum of four.

## What worked

The request-only architecture isolated a real routing improvement:

- exactness increased in every seed;
- no candidate-correct prompt became incorrect relative to its paired Phase 10
  control;
- oracle-route exactness remained 92/100, 98/100, and 95/100;
- every active-route, exact-operand example had an exact deterministic result
  trajectory and exact decoded answer;
- forcing the route off removed every normally correct candidate answer;
- all inherited Phase 10 checkpoint tensors were bit-identical;
- the only architectural addition was the disclosed 4,096-weight router.

The paired-improvement, operand-access, conditional-mechanism,
causal-routing, and checkpoint-integrity gates passed.

## What failed

The router did not generalize operation identity robustly enough.

Across all three seeds:

- word problems: 75/75 exact;
- direct requests: 89/150 exact;
- positive three-number distractors: 48/75 exact.

Of 23 false routes across 600 negative seed-prompt evaluations, 21 were
multiplication prompts. Those 21 came from only two unseen family templates:

- `Evaluate the product operation on {x} and {y}.`: 11/12 false routes;
- `Send operands {x}, {y} through a MULTIPLY instruction.`: 10/12 false routes.

One equality prompt and one three-number factual prompt accounted for the
remaining false routes. The other eight negative categories had zero false
routes.

This concentration matters. The selected router reads only the final prompt
state through a linear readout. Its development success depended too much on
how familiar request and operation language shaped that final state. It
recognized all new word-problem additions but confused two novel,
instruction-like multiplication constructions with addition and rejected
several new direct-addition constructions.

Post-hoc threshold analysis confirms that calibration alone cannot solve the
problem. Under the frozen maximum of four false routes, the best possible
thresholds on this disclosed audit would yield only 66/100, 74/100, and
68/100 counterfactual exact answers.

## Frozen verdict

| Gate | Result |
| --- | --- |
| Autonomous exactness | Fail |
| Paired improvement | Pass |
| Route recognition | Fail |
| Preservation | Fail |
| Operand access | Pass |
| Conditional mechanism | Pass |
| Causal routing | Pass |
| Checkpoint integrity | Pass |
| Compound verdict | **Fail** |

## Plain-English interpretation

Giving the system a separate “should I run the adder?” decision helped a lot:
it recovered roughly half of the gap between Phase 10 autonomous performance
and the forced-route arithmetic ceiling. Once it made the right decision and
read the operands correctly, the installed arithmetic still worked exactly.

But the decision itself remained brittle. The router learned a useful
statistical notion of an addition request, not a dependable understanding of
which operation was requested. Two new ways of asking for multiplication were
enough to trigger the adder repeatedly, while several new ways of explicitly
asking for addition failed to trigger it.

The result supports route-only architectural separation but rejects this
particular final-token linear router as a reliable autonomous semantic router.

## Next experiment

Phase 11 is now disclosed development data for the next phase. The next
development cycle should:

1. read multiple prompt tokens rather than only the final prompt state;
2. train on minimal semantic contrasts that share wording and differ only in
   the requested operation, such as ADD versus MULTIPLY;
3. make routing invariant to instruction suffixes and paraphrase endings;
4. retain the same paired source checkpoints and strict route-off
   preservation constraint;
5. select the design on disclosed Phase 11 results, then freeze a new,
   family-disjoint confirmation audit.

Learned attention pooling or another compact multi-view readout is the
appropriate next architectural test. A keyword parser, explicit operation
label, external classifier, or post-generation correction would not test the
autonomous-routing hypothesis.

## Artifacts

- Frozen protocol: `PHASE11_AUTONOMOUS_ROUTING_PROTOCOL.md`
- Frozen prompt manifest: `phase11_results/frozen_prompt_manifest.json`
- Raw confirmation: `phase11_results/confirmation.json`
- Post-hoc analysis: `phase11_results/analysis.json`
- Development record: `PHASE11_AUTONOMOUS_ROUTING_DEVELOPMENT.md`
