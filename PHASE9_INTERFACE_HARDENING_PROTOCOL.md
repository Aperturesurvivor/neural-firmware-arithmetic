# Phase 9 Frozen Protocol: Interface Specificity and Semantic Generalization

Status: frozen after development; confirmation not yet run.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Can a targeted semantic curriculum repair TinyLlama's learned neural interface
to the existing deterministic addition-neuron implant, without changing the
calculator, its placement, result decoder, activation bandwidth, base-model
weights, or architectural learned-parameter count?

The paired curriculum comparison separates two hypotheses:

1. Phase 8 mainly needed more examples and optimization.
2. Phase 8 needed hard semantic contrasts that explicitly distinguish
   execution from multiplication, factual mention, quotation, negation,
   hypothetical discussion, and irrelevant numbers.

Phase 9 does not test recurrent calculator use, multiple operations, a
nonlinear interface, surrounding-layer adaptation, or a different implant
location.

## Frozen model and implant

- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
- Decoder MLP layer: 15
- Existing MLP coordinates replaced: the 28 coordinates frozen in Phase 8
- Maximum operand width: four decimal digits
- Deterministic calculator learned parameters: 0
- Architectural learned weights: 57,344
  - input/interface rows: 32,768
  - frozen Phase 8 result columns: 24,576
- Phase 9 updated weights: input/interface rows only, 32,768
- Result columns must remain bit-identical to each source Phase 8 checkpoint.
- All pretrained TinyLlama weights remain frozen.
- Runtime disclosed and unchanged:
  - response-local route latch;
  - response-local operand register;
  - deterministic result-position counter;
  - base contribution preserved whenever the route is off.

The digit-confidence threshold changes from Phase 8's `0.9` to the
development-selected `0.8`. No learned weights or calculator behavior are
added by this calibration.

## Conditions

All Phase 9 conditions inherit the corresponding independent Phase 8 source
checkpoint.

1. **Untouched base.**
2. **Phase 8 matched residual adapter**, retained as a descriptive
   parameter-matched reference.
3. **Frozen Phase 8 implant**, evaluated without Phase 9 continuation.
4. **Generic continuation**, with 1,200 new additions and 1,200 new negatives
   drawn from the prior Phase 8 training-family distribution.
5. **Hard-contrast continuation**, with 1,200 additions and 1,200 negatives
   balanced across new semantic families.
6. **Hard-contrast result ablation**, applied only to positive confirmation
   prompts.

Generic and hard continuation have identical example counts, source
checkpoints, trainable weights, optimizer schedules, and development
threshold-selection procedure. Only the curriculum differs.

## Seeds

Phase 9 continuation seeds:

- 15,201, inheriting Phase 8 seed 14,201
- 15,202, inheriting Phase 8 seed 14,202
- 15,203, inheriting Phase 8 seed 14,203

The development seed is 15,199 and is excluded from confirmation.

## Training and development data

The generic and hard continuation sets each contain 2,400 examples:

- 1,200 route-positive additions;
- 1,200 route-negative prompts.

The hard positive set uses 24 balanced families, including:

- direct addition and state-increment language;
- reverse-surface-order addition;
- unfamiliar scientific and operational word problems;
- prompts with a five-digit irrelevant identifier plus two target operands;
- operand-final prompts in which the instruction ends immediately after the
  second operand.

The hard negative set uses 30 balanced families covering:

- multiplication;
- factual numbers and identifiers;
- quoted arithmetic;
- negated or canceled addition;
- subtraction, comparison, concatenation, division, and averaging;
- hypothetical or explanatory addition language.

The shared development set contains 240 positives and 480 negatives across 36
families disjoint from all training and confirmation families. Generated
operand pairs are disjoint across Phase 8, Phase 9 training, Phase 9
development, and Phase 9 confirmation.

## Development history and frozen choices

No confirmation prompt was run during development.

The initial generic continuation reached 32/60 exact positives, 37/60 exact
operands, 0/72 false routes, and 72/72 token preservation.

The initial hard curriculum reached 47/60 exact positives, 52/60 exact
operands, 0/72 false routes, and 72/72 preservation. Ten of thirteen failures
were route-off additions from two distractor families whose prompts ended
immediately after the second operand.

Adding four family-disjoint operand-final training constructions while keeping
the positive count fixed at 1,200 raised the hard condition to 54/60 exact,
54/60 exact operands, 60/60 route predictions, 0/72 false routes, and 72/72
preservation at digit threshold `0.9`. Development-only digit calibration to
`0.8` recovered one additional exact operand trajectory. Increasing role/digit
loss weights and extending training to 2,500 steps did not improve end-to-end
exactness and was rejected.

The frozen schedule is therefore:

- interface steps: 1,500
- interface batch size: 256
- interface learning rate: 0.0005
- route, role, and digit loss weights: 1.0 each
- learned-step loss weight: 0.0
- route-row hardening steps: 2,500
- route-row batch size: 256
- route-row learning rate: 0.0005
- maximum development false-positive rate: 0.01
- digit-confidence threshold: 0.8
- output strength: 16.0, unchanged

## Sealed confirmation

The primary audit contains 300 unique prompts:

- 100 additions:
  - 50 direct;
  - 25 word problems;
  - 25 additions containing an irrelevant third number.
- 200 adversarial negatives, 20 each:
  - multiplication;
  - factual-number questions;
  - quoted arithmetic;
  - negated addition;
  - canceled addition;
  - subtraction;
  - comparison;
  - concatenation;
  - hypothetical/explanatory arithmetic;
  - three-number distractor prompts.

All 70 confirmation families are exact-string-disjoint from training and
development families. The exact generated prompts and their canonical SHA-256
hash are frozen before confirmatory training.

Greedy decoding uses at most eight generated tokens for every condition.

## Primary metrics

For every hard-contrast seed:

- strict numeral-only exact accuracy on 100 positives;
- exact operand-register recovery;
- route prediction and active-route counts;
- exact deterministic calculator trajectories;
- exact decoding conditional on an active route and exact operands;
- paired causal loss under result ablation;
- false routes on 200 adversarial negatives;
- token-exact preservation relative to untouched base;
- category-specific failures;
- local latency and sampled allocated MPS memory.

Descriptive comparisons include frozen Phase 8, generic continuation,
untouched base, and the existing matched adapter.

## Frozen gates

The hard-contrast condition passes only if all three seeds satisfy every gate:

1. **Accuracy:** at least 95/100 strict exact additions.
2. **Operands:** at least 97/100 exact operand registers.
3. **Routing and preservation:** at most 4/200 false routes and at least
   196/200 token-exact preserved negatives.
4. **Conditional mechanism:** every active-route/exact-operand example has an
   exact calculator trajectory and exact decoded answer.
5. **Causal ablation:** at least 90 paired correct-to-incorrect changes and no
   more than 5/100 exact answers under ablation.
6. **Curriculum specificity:** hard continuation has higher positive exactness
   and no more false routes than generic continuation for each paired seed.

The compound result fails if any gate fails. A failure is retained and reported;
no seed, threshold, prompt, family, checkpoint, or decoding budget may be
replaced after the freeze.

## Interpretation boundaries

A pass would show that targeted semantic curricula can make this fixed
TinyLlama interface substantially more reliable. It would not establish
arbitrary-language robustness or self-directed repeated use.

A failure with preserved conditional computation would locate the remaining
bottleneck in the frozen linear semantic/operand interface. A nonlinear
readout, surrounding-layer adaptation, layer comparison, or more capable base
model would then require a separately frozen experiment.
