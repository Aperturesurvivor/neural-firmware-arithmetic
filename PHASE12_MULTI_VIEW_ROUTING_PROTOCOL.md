# Phase 12 Frozen Protocol: Multi-View Autonomous Semantic Routing

Status: frozen after development; confirmation not yet run.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Phase 11 established a causal and operand-accessible arithmetic mechanism, but
its final-token linear request router failed autonomous exactness, route
recognition, and preservation. Phase 12 tests whether a small nonlinear
request router that reads four pooled views of the same frozen adapted
representation can distinguish requests to add from semantically adjacent
non-requests without changing the arithmetic mechanism.

The claim remains narrow: on disjoint four-digit-addition prompts, a learned
request-level readout can autonomously gate an existing deterministic neural
calculator while preserving ordinary generation when it should remain off.

## Frozen candidate

Each candidate is the corresponding Phase 10 `linear_representation`
checkpoint plus a bias-free request router:

- base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`;
- model revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`;
- decoder MLP layer: 15;
- views: final prompt token, all-token mean, user-token mean, and final-eight
  user-token mean;
- feature width: 8,192;
- nonlinear router: 16-unit SiLU bottleneck and two-class output;
- router learned weights: 131,104;
- request-route temperature: 2.0;
- fixed request threshold: 0.60;
- total architectural learned weights: 204,832;
- calculator learned weights: 0.

The router runs inside the implanted MLP at the first eligible generation
position and is latched across the response. It receives no parsed operation
label, keyword feature, operand value, external classifier output, tool result,
or ground-truth route. User-content masks come only from tokenizer offsets.

All pretrained weights, implanted coordinates, rank-four representation
adapter, token interface, result decoder, deterministic calculator, route
latch, operand register, result counter, and route-off preservation behavior
are inherited bit-identically from Phase 10.

## Development provenance

Phase 12 used all frozen Phase 11 rows as disclosed development data. Four
pooled-view candidates were tested under family-held-out and nested
family-held-out screens. Neither screen selected a candidate under its
calibration rule. All failed results are retained.

A clearly labeled post-hoc development analysis then selected the lowest
0.05-grid threshold with no more than 4/200 Phase 11 false routes in every
seed. This fixed 0.60. At that cutoff, held-family model predictions yielded
78/82/82 counterfactual exact answers and 4/4/4 false routes. Because the
cutoff saw all disclosed labels, those numbers are not confirmatory evidence.

Final deployment routers were trained for 1,500 balanced steps on Phase 8 plus
Phase 9 hard training plus all disclosed Phase 11 rows, using router seeds
21,201/21,202/21,203. Full installed-model validation on the disclosed Phase
11 rows produced 92/98/95 exact answers, 100/100/100 positive routes, no false
routes, complete token preservation, and 900/900 live/cached route agreement.
This too is development evidence.

No Phase 12 row influenced architecture, weights, threshold, or checkpoint
selection.

## Seeds and frozen checkpoints

- source seed 16,201, router seed 21,201, candidate SHA-256
  `1987332bffe79caf7c6c7b6e7150bded5cfef1f4427502476149bee7131628bb`;
- source seed 16,202, router seed 21,202, candidate SHA-256
  `9b15d29e38b4e4e54804966455b0938c7dd68b5698a7103a09c8f31eb2aac888`;
- source seed 16,203, router seed 21,203, candidate SHA-256
  `ffb09f542573956ec7d6dfe89cab2b775ab2d85897dbb9c8fb34b5fedfdb007b`.

The paired Phase 11 linear-router checkpoints are frozen controls. Neither
condition is retrained after this protocol is committed.

## Confirmatory conditions

For every source seed and prompt:

1. **Phase 11 control:** the prior final-token linear request router.
2. **Phase 12 candidate:** the frozen four-view SiLU request router.
3. **Oracle-route diagnostic:** Phase 12 candidate on positives with route
   forced on.
4. **Route-off intervention:** Phase 12 candidate on positives with route
   forced off.

The untouched TinyLlama baseline is generated once per prompt for token-exact
route-off preservation comparisons. Greedy decoding is used for at most eight
new tokens in every condition.

## Sealed confirmation data

The audit contains 300 unique prompts:

- 100 additions: 50 direct, 25 word problems, and 25 with an irrelevant
  five-digit third number;
- 200 adversarial negatives, 20 each: multiplication, factual identifiers,
  quoted arithmetic, negated addition, canceled addition, subtraction,
  comparison, concatenation, hypothetical arithmetic, and three-number
  factual distractors.

All 70 family templates are exact-string-disjoint from Phase 9, Phase 10, and
Phase 11 families. Every operand pair is disjoint from all generated Phase 8
through Phase 11 splits. Data-generation seeds are 22,701 and 22,702.

The exact rendered rows and their canonical SHA-256 hash must be committed in
a frozen manifest before model evaluation.

## Metrics

For each seed and autonomous condition:

- strict numeral-only exact answers on 100 positives;
- positive and active routes;
- exact operand-register recovery and deterministic result trajectories;
- exact output conditional on active routing and exact operands;
- false routes and token-exact preservation on 200 negatives;
- category-specific route and exactness counts.

The candidate additionally reports oracle-route and route-off results,
checkpoint hashes, architectural counts, and paired changes from Phase 11.

## Frozen gates

The Phase 12 hypothesis passes only if every gate passes:

1. **Autonomous exactness:** each Phase 12 seed is exact on at least 70/100
   positives and mean exactness is at least 75/100.
2. **Paired improvement:** Phase 12 exceeds Phase 11 on every seed and its
   mean paired gain is at least 10/100.
3. **Route recognition:** each Phase 12 seed routes at least 80/100 positives.
4. **Preservation:** each Phase 12 seed has at most 4/200 false routes and at
   least 196/200 token-exact preserved negatives.
5. **Operand access:** each Phase 12 seed is exact on at least 85/100 positives
   under oracle routing.
6. **Conditional mechanism:** every Phase 12 example with an active route and
   exact operands has an exact trajectory and exact decoded answer.
7. **Causal routing:** forcing the Phase 12 route off removes every normally
   correct candidate answer, and route-off exactness is at most 5/100.
8. **Checkpoint integrity:** the candidate adds exactly 131,104 disclosed
   router weights; every inherited tensor-valued Phase 10 field is
   bit-identical to its paired source.

All failures and partial passes are retained. Passing establishes autonomous
semantic routing only for this model, four-digit addition mechanism, and
sampled prompt distribution—not general language understanding or arbitrary
program routing.
