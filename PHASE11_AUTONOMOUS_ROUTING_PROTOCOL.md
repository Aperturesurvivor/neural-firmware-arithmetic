# Phase 11 Frozen Protocol: Autonomous Request-Level Semantic Routing

Status: frozen after development; confirmation not yet run.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Phase 10 showed that the implanted arithmetic mechanism was exact on 89--90%
of 100 prompts when its route was forced, but autonomous final-token routing
limited end-to-end exactness to 42--53%. Phase 11 tests whether separating the
semantic route decision into a dedicated request-level readout closes that gap
without changing operand extraction, calculator execution, decoding, or
route-off base computation.

The confirmatory claim is narrow: a small learned route-only readout of a
frozen model representation can autonomously decide whether the existing
addition implant should execute on disjoint prompts while preserving ordinary
generation when it should not.

## Frozen candidate

Each candidate checkpoint is the corresponding Phase 10
`linear_representation` checkpoint plus a two-by-2,048 bias-free linear
request router:

- base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`;
- model revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`;
- decoder MLP layer: 15;
- request representation: the adapted hidden state at the final prompt token;
- router weights: 4,096;
- request-route temperature: 2.0;
- per-seed thresholds fixed from Phase 9 development data;
- total architectural learned weights: 77,824;
- calculator learned weights: 0.

The router runs inside the implanted MLP at the first eligible generation
position. Its decision is latched across the response. It receives no parsed
operation label, keyword feature, operand value, external classifier output,
tool result, or ground-truth route.

The following are inherited bit-identically from Phase 10:

- all pretrained TinyLlama weights;
- the 28 implanted layer-15 coordinates;
- the rank-four interface-local representation adapter;
- all existing route/role/digit interface weights;
- the learned result decoder;
- the deterministic addition circuit;
- the route latch, operand register, deterministic result counter, and
  route-off preservation behavior.

## Development and selection

No weights are trained or thresholds selected using Phase 11 confirmation
data. Four 4,096-weight routers were trained for 2,500 balanced steps using
Phase 8 training plus Phase 9 hard-training prompts, calibrated on Phase 9
development prompts, and selected using the already disclosed Phase 10 audit.

The predeclared development rule selected the condition with no more than
4/200 false routes in every seed, then maximized worst-seed counterfactual
exactness and mean exactness. Only the final-token (`last`) request router met
the preservation constraint. Its development exact counts were 81/83/77 and
false-route counts were 2/4/3. A subsequent real-generation check exactly
reproduced all 900/900 cached route decisions.

All rejected router checkpoints and their metrics are retained.

## Seeds and checkpoint pairing

- Phase 10 source seed 16,201 with route-only seed 17,201;
- Phase 10 source seed 16,202 with route-only seed 17,202;
- Phase 10 source seed 16,203 with route-only seed 17,203.

The three selected development checkpoints are the fixed confirmatory
candidates. They are not retrained after this protocol is frozen.

## Confirmatory conditions

For every source seed and prompt:

1. **Phase 10 control:** the frozen Phase 10 `linear_representation`
   checkpoint with its original joint interface route.
2. **Phase 11 autonomous candidate:** the paired checkpoint with the dedicated
   request router active.
3. **Oracle-route diagnostic:** candidate on positive prompts with route forced
   on; operand extraction, calculator execution, and decoding remain unchanged.
4. **Route-off intervention:** candidate on positive prompts with route forced
   off; ordinary base computation remains active.

The untouched TinyLlama baseline is generated once per prompt for exact
route-off preservation comparisons.

Greedy decoding is used for at most eight new tokens in every condition.

## Sealed confirmation data

The audit contains 300 unique prompts:

- 100 additions:
  - 50 direct requests;
  - 25 word problems;
  - 25 requests with an irrelevant five-digit third number.
- 200 adversarial negatives, 20 each:
  - multiplication;
  - factual identifiers;
  - quoted arithmetic;
  - negated addition;
  - canceled addition;
  - subtraction;
  - comparison;
  - concatenation;
  - hypothetical or explanatory arithmetic;
  - three-number factual distractors.

The 70 confirmation family templates are exact-string-disjoint from Phase 9
hard training, Phase 9 development, Phase 9 confirmation, and Phase 10
confirmation. Every operand pair is disjoint from every generated Phase 8,
Phase 9, and Phase 10 split. Data-generation seeds are 17,701 for positives
and 17,702 for negatives.

The exact rendered rows and their canonical SHA-256 hash must be committed in
a frozen manifest before any confirmatory model evaluation.

## Metrics

For each seed and autonomous condition:

- strict numeral-only exact answers on 100 positives;
- positive route decisions and active routes;
- exact operand-register recovery;
- exact deterministic result trajectories;
- exact decoding conditional on an active route and exact operands;
- false routes and token-exact preservation on 200 negatives;
- category-specific route and exactness counts.

The candidate additionally reports oracle-route exactness, route-off
exactness, paired losses under route-off intervention, route probabilities,
checkpoint hashes, and architectural parameter counts.

## Frozen gates

The autonomous semantic-routing hypothesis passes only if every gate passes:

1. **Autonomous exactness:** every candidate seed is exact on at least 70/100
   positives and mean candidate exactness is at least 75/100.
2. **Paired improvement:** the candidate exceeds its Phase 10 control in every
   seed and the mean paired gain is at least 20/100 prompts.
3. **Route recognition:** every candidate seed routes at least 80/100
   positives.
4. **Preservation:** every candidate seed has at most 4/200 false routes and
   at least 196/200 token-exact preserved negatives.
5. **Operand access:** every candidate seed is exact on at least 85/100
   positive prompts under oracle routing.
6. **Conditional mechanism:** every candidate example with an active route
   and exact operands has an exact deterministic trajectory and exact decoded
   answer.
7. **Causal routing:** forcing the candidate route off removes every normally
   correct candidate answer, and route-off exactness is at most 5/100 in every
   seed.
8. **Checkpoint integrity:** the candidate adds exactly the disclosed 4,096
   router weights; all inherited tensor-valued Phase 10 checkpoint fields are
   bit-identical to their paired source.

All failures and partial passes are retained and reported. Passing these gates
would establish autonomous semantic routing only for this frozen,
four-digit-addition protocol and its sampled prompt distribution. It would not
establish robust natural-language understanding, arbitrary arithmetic, or
general-purpose neural program routing.
