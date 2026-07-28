# Phase 10 Frozen Protocol: Interface Capacity and Representation Access

Status: frozen after development; confirmation not yet run.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Phase 9 showed that valid executions of the frozen deterministic addition
implant remained exact while the learned semantic interface was unreliable.
Phase 10 separates three hypotheses:

1. a nonlinear interface can outperform the linear interface at exactly the
   same learned input-interface budget;
2. a small interface-local representation adapter can make relevant
   information more accessible without changing route-off base computation;
3. routing and operand extraction make separable contributions to failure.

This phase does not test new arithmetic operations, repeated calculator use,
another implant layer, a larger model, or adaptation of the pretrained
TinyLlama weights.

## Frozen model and implant

- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Decoder MLP layer: 15
- Existing MLP coordinates replaced: the 28 coordinates frozen in Phase 8
- Maximum operand width: four decimal digits
- Deterministic calculator learned parameters: 0
- Frozen result decoder: 24,576 learned weights, inherited bit-identically
  from each corresponding Phase 8 source checkpoint
- Output strength: 16.0
- Digit-confidence threshold: 0.8
- Route-softmax temperature: 2.0
- All pretrained TinyLlama weights remain frozen
- Runtime remains the disclosed response-local route latch, operand register,
  deterministic result-position counter, and route-off base preservation

Temperature 2 preserves route-logit ordering and changes no learned parameter.
It prevents float32 probability saturation observed during development so
first-step threshold calibration can enforce the false-positive constraint.

## Confirmatory conditions

1. **Linear control.** The existing 32,768-weight linear route/role/digit
   interface.
2. **Budget-matched nonlinear.** A learned 16-unit SiLU bottleneck over a fixed
   2,032-of-2,048 input projection and a learned 16-by-16 output mixer:
   `16×2,032 + 16×16 = 32,768` learned input-interface weights.
3. **Linear plus interface-local representation adaptation.** The linear
   interface plus a rank-four nonlinear residual adapter containing 16,384
   learned weights. The adapter changes only the representation read by the
   implant interface. The ordinary MLP receives the original hidden state.
4. **Oracle-route diagnostic.** On positive prompts only, force the route on
   while leaving role/digit inference, operand extraction, calculator
   execution, and decoding unchanged.
5. **Candidate result ablation.** Disable only deterministic result
   activations for the representation-adapted condition on positive prompts.

The nonlinear condition exactly matches the linear input-interface budget.
The representation condition is intentionally not parameter matched and is
reported with its added 16,384 weights.

## Training and calibration

Every condition trains from the corresponding independent Phase 8 source
checkpoint using:

- the frozen Phase 8 sequence-interface training features;
- the frozen Phase 9 hard-contrast training features;
- 2,500 joint interface steps;
- batch size 256;
- learning rate 0.0005;
- route, role, and digit loss weights 1.0;
- learned-step loss weight 0.0;
- unchanged frozen result columns.

No learned route-hardening pass is used. After training, the route threshold
is selected from first-generation-step features on the already disclosed
Phase 9 confirmation prompts, constrained to at most 1% false positives. The
Phase 9 prompts are development data for Phase 10 and are not confirmation
evidence.

## Seeds

Phase 10 training seeds and Phase 8 source checkpoints:

- 16,201 inherits source seed 14,201
- 16,202 inherits source seed 14,202
- 16,203 inherits source seed 14,203

Development seed 16,199 is excluded from confirmation.

## Sealed confirmation

The new audit contains 300 unique prompts:

- 100 additions:
  - 50 direct;
  - 25 word problems;
  - 25 with an irrelevant five-digit third number.
- 200 adversarial negatives, 20 each:
  - multiplication;
  - factual identifiers;
  - quoted arithmetic;
  - negated addition;
  - canceled addition;
  - subtraction;
  - comparison;
  - concatenation;
  - hypothetical/explanatory arithmetic;
  - three-number factual distractors.

All 70 prompt families are exact-string-disjoint from Phase 9 hard training,
development, and confirmation families. Every generated operand pair is
disjoint from all Phase 8 and Phase 9 generated splits. The exact prompts and
their canonical SHA-256 hash are frozen before confirmatory training.

Greedy decoding uses at most eight generated tokens for every condition.

## Metrics

For each condition and seed:

- strict numeral-only exact accuracy on 100 positives;
- positive route predictions and active routes;
- exact operand-register recovery;
- exact deterministic calculator trajectories;
- exact decoding conditional on active route and exact operands;
- false routes and token-exact preservation on 200 negatives;
- category-specific outcomes.

For every seed, oracle routing measures operand/decode capability independent
of semantic route recognition. The representation condition additionally
reports result ablation and paired causal losses.

## Frozen gates

The representation-access hypothesis passes only if all gates pass:

1. **Paired end-to-end benefit:** the representation condition has more exact
   positives than linear in every seed and its mean paired gain is at least
   five of 100 prompts.
2. **Preservation:** every representation seed has at most 4/200 false routes
   and at least 196/200 token-exact preserved negatives.
3. **Operand access:** every representation seed has at least 85/100
   oracle-route exact answers and no fewer oracle-route exact answers than its
   paired linear seed.
4. **Conditional mechanism:** every active-route, exact-string-operand example
   has an exact deterministic trajectory and exact decoded answer.
5. **Causal ablation:** each representation seed loses every normally correct
   answer when result activations are ablated and has no more than 5/100 exact
   answers under ablation.

The parameter-matched nonlinearity hypothesis passes separately only if the
nonlinear condition has more exact positives than linear in every seed, no
more false routes in every seed, and a positive mean paired exactness gain.

Failure of either hypothesis is retained and reported. Neither hypothesis
gate establishes end-to-end reliability.

## Development history

Development used only the previously disclosed Phase 9 audit:

- linear: 37/100 exact, 87/100 oracle exact, 2/200 false routes, 198/200
  preserved;
- budget-matched nonlinear: 31/100 exact, 80/100 oracle exact, 1/200 false
  route, 199/200 preserved;
- selected linear local-representation condition: 44/100 exact, 88/100 oracle
  exact, 2/200 false routes, 198/200 preserved;
- nonlinear plus local representation: 38/100 exact, 82/100 oracle exact,
  2/200 false routes, 198/200 preserved.

Retained rejected pilots include a fixed-Hadamard nonlinear output mixer,
shared-MLP representation adaptation with 185/200 preservation, and a
2,500-step learned route-hardening pass that saturated out-of-family route
probabilities.

## Interpretation boundaries

A representation-hypothesis pass would show that a small learned transform
local to the deterministic implant interface improves access to frozen
TinyLlama representations under this controlled addition protocol. It would
not show that TinyLlama itself was adapted, that arbitrary language is robust,
or that deterministic-neuron systems are generally reliable.

A nonlinear-hypothesis failure would apply only to this exactly matched
16-unit bottleneck and training protocol, not to every possible nonlinear
interface.
