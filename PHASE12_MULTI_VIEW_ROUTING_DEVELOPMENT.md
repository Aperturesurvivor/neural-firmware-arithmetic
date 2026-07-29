# Phase 12 Multi-View Semantic Routing Development Plan

Status: exploratory development; no Phase 12 confirmation prompts have been
created or evaluated.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Motivation

Phase 11 showed that a separate request-only router improved exact answers by
18, 31, and 17 points over the Phase 10 interface router, but the selected
linear readout of the final prompt state failed its frozen confirmation.
Positive routing reached only 75%, 79%, and 72%; false routes reached 8, 9,
and 6 of 200. Twenty-one of 23 false routes came from two previously unseen
multiplication templates. Post-hoc thresholding could not satisfy both
exactness and preservation.

Phase 12 treats all Phase 11 prompts and outcomes as disclosed development
data. It asks whether a compact router that reads multiple frozen request
representations can distinguish the requested operation more robustly than a
final-state-only linear boundary.

## Frozen architectural boundary

Every candidate begins from the corresponding Phase 10
`linear_representation` checkpoint. The following remain fixed:

- TinyLlama and all pretrained weights;
- decoder layer 15 and the 28 implanted coordinates;
- the rank-four interface-local representation adapter;
- route/role/digit operand-interface weights;
- deterministic calculator and learned result decoder;
- route latch, operand register, result counter, and route-off preservation.

Only the dedicated request router may change.

## Development representations

For each prompt, collect four views of the same adapted layer-15 state:

1. `last`: final chat-template token;
2. `sequence_mean`: mean over all non-padding prompt tokens;
3. `user_mean`: mean over user-content tokens;
4. `user_tail_mean`: mean over the last eight user-content tokens.

No view parses a keyword, operation, or number. User masks come only from
tokenizer offsets.

## Candidate routers

1. `last_linear`: the 4,096-weight Phase 11 architecture, retrained under the
   Phase 12 development curriculum.
2. `last_user_linear`: a 8,192-weight linear readout of concatenated `last`
   and `user_mean` views.
3. `all_views_linear`: a 16,384-weight linear readout of all four views.
4. `all_views_silu16`: a bias-free 16-unit SiLU bottleneck over all four views,
   followed by a two-class readout; 131,104 learned weights.

The nonlinear condition is intentionally larger and is not a parameter-matched
claim. Every condition retains a precise learned-weight count.

## Data and family-held-out selection

- Base training: Phase 8 training plus Phase 9 hard-contrast training.
- Base threshold calibration: Phase 9 development.
- Newly disclosed development: all 300 frozen Phase 11 confirmation prompts.
- Arithmetic ceiling: stored Phase 11 oracle-route results.

Phase 11's 70 family templates are divided into five deterministic folds using
family index within positive/negative strata. Each fold contains four positive
and ten negative families, or 60 prompts. For every candidate and seed:

1. train on the base training data plus the other four Phase 11 family folds;
2. calibrate its threshold only on Phase 9 development;
3. predict the held-out Phase 11 families;
4. concatenate the five held-out predictions so every Phase 11 prompt is
   predicted by a router that did not train on its family template.

This is development cross-validation, not confirmation evidence. Architecture
and training choices were motivated by the disclosed Phase 11 errors.

## Selection rule

A condition is eligible only if its out-of-fold predictions produce at most
4/200 false routes in every seed. Among eligible conditions, select the
highest worst-seed counterfactual exact count, breaking ties by mean exact
count and then lower learned-parameter count. A positive counts
counterfactually exact only when its held-out router activates and the stored
Phase 11 oracle-route generation was exact.

If no candidate is eligible, retain all results and revise development without
creating Phase 12 confirmation data.

After selection, train one deployment router per seed on all disclosed
development data, fix thresholds before any Phase 12 prompt family exists,
install the router in the actual generation loop, and rerun the disclosed
Phase 11 audit end to end.

## Boundary for the next confirmation

No Phase 12 confirmation templates, operand pairs, or outputs may be created
until:

- all candidate results are retained;
- the selection decision is documented;
- selected deployment checkpoint hashes are recorded;
- real-generation behavior matches offline predictions;
- a new protocol with unchanged gates is committed.

Passing a future Phase 12 confirmation would remain a narrow result about
autonomous routing for four-digit addition under the sampled language
distribution, not general semantic understanding.

## First family-held-out screen

The initial five-fold screen retained all four conditions and selected none.
Out-of-fold counterfactual exact counts and false routes were:

- `last_linear`: 70/78/73 exact, 6/6/4 false routes;
- `last_user_linear`: 71/74/75 exact, 5/6/11 false routes;
- `all_views_linear`: 69/75/72 exact, 11/9/16 false routes;
- `all_views_silu16`: 81/86/82 exact, 10/6/8 false routes.

The nonlinear four-view condition materially improved family-held-out recall
and exactness, but no candidate met the predeclared preservation constraint.
Therefore no deployment router was selected and no Phase 12 confirmation data
was created.

The pattern also exposed a calibration limitation: fold routers were
thresholded only on the older Phase 9 development distribution. The next
development screen will use nested family holdout. For each outer evaluation
fold, one different Phase 11 family fold will calibrate the threshold and the
remaining three will join base training. Thus every evaluated family remains
disjoint from both router training and threshold calibration. This revision
is development-only and was declared after retaining the failed first screen.

## Nested family-held-out screen

The declared nested screen also selected no condition. Out-of-fold
counterfactual exact counts and false routes were:

- `last_linear`: 66/76/73 exact, 5/5/4 false routes;
- `last_user_linear`: 67/75/71 exact, 10/10/11 false routes;
- `all_views_linear`: 77/74/71 exact, 21/9/14 false routes;
- `all_views_silu16`: 79/85/82 exact, 12/11/10 false routes.

The nonlinear router again produced the best recall and exactness, routing
87/100 positives in every seed and improving counterfactual exactness over the
Phase 11 natural router in every seed. It nevertheless failed the unchanged
preservation rule in every seed. Calibrating on a separate Phase 11 family did
not control false routes on the held-out family.

This rules out threshold calibration as the main repair for the current
representations. The aggregate mean views discard token order and therefore
cannot reliably distinguish an instruction to add from statements that quote,
describe, concatenate, or otherwise mention arithmetic-looking content. The
next development architecture may use token-level adapted states and an
order-sensitive neural readout. No Phase 12 confirmation set will be created
unless that architecture first passes the same family-held-out selection rule.

## Disclosed-data threshold selection

A subsequent diagnostic separated two questions that the nested screen had
coupled: whether the router scores contain a usable signal, and whether the
fold-specific calibration procedure finds a stable cutoff. The saved
out-of-fold probabilities from `all_views_silu16` were evaluated on a fixed
0.05-spaced threshold grid from 0.50 through 0.95. The pre-existing numerical
preservation criterion selected the lowest cutoff with at most 4/200 false
routes in every seed.

The selected cutoff was 0.60. At that fixed cutoff, out-of-fold
counterfactual exact counts were 78/82/82, positive routes were 86/84/87, and
false routes were 4/4/4. Thus the nonlinear scores contain a useful operating
point even though the nested calibration rule failed to discover it.

This cutoff was selected after inspecting all disclosed Phase 11 labels. It is
an ordinary development hyperparameter, not family-held-out or confirmatory
evidence. Phase 12 will therefore carry forward `all_views_silu16` with a
fixed request threshold of 0.60 and judge that complete frozen system only on
new prompt families and operand pairs. If real-generation validation on
disclosed data diverges from the offline predictions, or if the new
confirmation fails preservation, token-level routing remains the next
architectural revision.

Deployment routers use seeds 21,201/21,202/21,203. Each is trained for 1,500
steps on the Phase 8 plus Phase 9 hard-routing training bank augmented with
all disclosed Phase 11 examples. Phase 9 development and disclosed Phase 10
examples are retained as diagnostic calibration data, but they cannot replace
the fixed 0.60 threshold. These choices were fixed before deployment training.

## Deployment validation

The three deployment routers separated the disclosed Phase 11 training rows
perfectly at the fixed cutoff. More importantly, full installed-model
generation on all 300 disclosed Phase 11 prompts produced:

- 92/98/95 exact positive outputs;
- 100/100/100 positive request routes;
- 0/0/0 false routes among 200 negatives;
- 200/200/200 negative outputs preserved token-for-token;
- 300/300/300 live route decisions matching cached predictions.

The largest absolute difference between a live and cached route probability
was below 0.0000005. Conditional output and trajectory exactness were 91/91,
97/97, and 95/95 whenever routing, execution, and operand capture were valid.
Every tensor inherited from the Phase 10 checkpoint remained bit-identical.
All deployment-validation gates passed.

These results remain development evidence because Phase 11 was used in router
training. They establish that the selected architecture is installed
correctly and justify freezing a new, disjoint Phase 12 confirmation.
