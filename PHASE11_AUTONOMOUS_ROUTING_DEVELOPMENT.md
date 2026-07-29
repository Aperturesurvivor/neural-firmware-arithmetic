# Phase 11 Autonomous Semantic Routing Development Plan

Status: exploratory development; no Phase 11 confirmation prompts have been
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
