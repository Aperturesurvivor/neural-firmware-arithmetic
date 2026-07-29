# Phase 11 Autonomous Semantic Routing Development Plan

Status: development complete; no Phase 11 confirmation prompts have been
generated or evaluated.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Phase 10 reached 89--90% exactness when routing was forced but only 42--53%
under its autonomous final-token route decision. Phase 11 asks whether a
dedicated request-level router can close that gap while leaving operand typing,
the interface-local representation adapter, the deterministic calculator, and
the learned result decoder unchanged.

## Architectural boundary

Every condition starts from the corresponding Phase 10
`linear_representation` checkpoint. The following remain bit-identical:

- TinyLlama and all pretrained weights;
- decoder MLP layer 15 and the 28 implanted coordinates;
- the rank-four interface-local representation adapter;
- all 16 existing route/role/digit interface rows;
- the deterministic calculator and result decoder;
- the route latch, operand register, deterministic result counter, and
  route-off preservation behavior.

The candidate router adds one learned two-by-2,048 linear decision matrix:
4,096 weights. It reads the same adapted layer-15 representation as the
implant, pools a complete request representation inside the implanted MLP, and
autonomously supplies the route decision at the eligible generation position.
The decision is then latched. This is not a parser, keyword rule, tool call, or
external semantic classifier.

## Parameter-matched development conditions

All learned routers contain exactly 4,096 weights:

1. `last`: the final chat-template token, separating route-only training from
   Phase 10's jointly trained interface;
2. `sequence_mean`: mean of all non-padding chat-sequence representations;
3. `user_mean`: mean of user-content representations, excluding fixed chat
   formatting;
4. `user_tail_mean`: mean of the last eight user-content token
   representations.

The user-content mask is derived only from tokenizer offsets in the model's
chat template. It performs no semantic or numeric parsing.

## Development-only data

- Router training: Phase 8 training prompts plus Phase 9 hard-contrast
  training prompts.
- Threshold calibration: Phase 9 development prompts, constrained to at most
  1% false positives.
- Architecture selection: the already disclosed Phase 10 confirmation audit.
  Its stored oracle-route outcomes permit a counterfactual exactness estimate
  without rerunning or modifying operand inference.

No Phase 11 confirmation data may be created until the architecture, training
schedule, seeds, gates, and selection result are documented and committed.

## Development selection

For each Phase 10 source seed, train each router for 2,500 balanced steps with
batch size 256, learning rate 0.0005, route temperature 2, and no bias.

Among conditions producing at most 4/200 false routes on the disclosed Phase
10 audit in every seed, select the condition with the highest worst-seed
counterfactual exact count; break ties by mean exact count. Retain every
candidate checkpoint and compact result, including unsuccessful routers.

The counterfactual is deliberately bounded: a positive is counted exact only
when the autonomous router activates and the previously recorded Phase 10
oracle-route generation was exact. It does not replace a sealed end-to-end
confirmation run.

## Development result

All four parameter-matched conditions were retained. The Phase 10 disclosed
audit produced the following exact-count / false-route triples across seeds:

- `last`: 81/83/77 exact and 2/4/3 false routes;
- `sequence_mean`: 84/90/89 exact and 13/30/27 false routes;
- `user_mean`: 77/84/81 exact and 10/19/14 false routes;
- `user_tail_mean`: 83/76/83 exact and 12/13/7 false routes.

Only `last` met the predeclared preservation constraint in every seed, so it
is the selected architecture. This result suggests the principal development
gain came from giving routing its own route-only weights rather than from
pooling more tokens. The pooled variants increased recall in some seeds but
over-routed adversarial negatives.

The selected checkpoint was then installed in the real generation loop and
run on every disclosed Phase 10 prompt. Its route decision matched the cached
offline decision on all 900/900 seed-prompt pairs. End-to-end exact counts
were 81/83/77, false routes were 2/4/3, and token preservation counts were
198/196/197. These exactly matched the bounded counterfactual. Every example
with an active route and exact operands had an exact result trajectory and
exact formatted answer.

These are development findings, not Phase 11 confirmation results.
