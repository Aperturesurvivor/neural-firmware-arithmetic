# Phase 4 Architecture Draft: Natural-Language Invocation

Status: superseded pilot design. The frozen design is documented in
`PHASE4_CONFIRMATORY_PROTOCOL.md`.

## Research question

On identical naturally worded addition prompts, how much end-to-end
mathematical ability does the internal deterministic architecture add over the
untouched Qwen2.5-0.5B-Instruct model and an exactly parameter-matched learned
internal adapter?

Phase 3 proved internal execution under one registered command. Phase 4 tests
semantic invocation: whether a small learned router can recognize
previously unseen addition wording, invoke the exact cell, and remain inactive
for negated, quoted, comparative, concatenation, multiplication, subtraction,
and explanatory prompts containing the same kinds of numbers.

## Fair same-prompt comparison

Every condition receives the exact same contiguous-number prompt, chat
template, greedy decoding, and token limit:

1. untouched frozen Qwen;
2. parameter-matched learned internal adapter;
3. internal deterministic addition with learned routing;
4. the same internal unit with oracle routing, as a structural upper bound;
5. the trained internal model with the unit forced off.

Two positive metrics prevent response formatting from becoming a misleading
baseline penalty:

- mathematical correctness: the final decimal integer in the response equals
  the exact sum;
- exact-format correctness: the complete stripped response is exactly the sum.

## Input boundary and claim separation

Prompts contain ordinary contiguous decimal strings such as `12345`, not
space-separated digits. A fixed tokenizer boundary identifies exactly two
candidate decimal spans and converts their characters into typed digits. It
does not decide whether addition was requested.

A learned semantic router consumes the final request residual. Its decision is
latched once for the response. The router is trained
on addition and non-addition wording families using one- to four-digit
operands. Entire positive and negative wording families are withheld from
training.

This phase tests learned semantic operation routing and exact execution, not a
learned general-purpose number extractor. The fixed candidate-number boundary
must be reported explicitly.

## Internal condition

The pilot-v1 wrapped sixth transformer block contained:

1. the unchanged original Qwen block;
2. the learned 897-parameter semantic router;
3. a zero-parameter frozen typed ripple-carry cell;
4. the learned 9,856-parameter eleven-symbol residual decoder.

The total learned interface is 10,753 parameters. Blocks 7 through 24, final
RMS normalization, and the tied language-model head remain downstream.

## Learned control

The control shares the same router weights, route threshold, insertion depth,
training examples, and answer positions. Instead of deterministic state, it
uses a rank-five bottleneck adapter with no down-projection bias:

- down projection: 896 by 5 = 4,480 parameters;
- up projection plus bias: 5 by 896 + 896 = 5,376 parameters;
- adapter total: 9,856 parameters;
- router plus adapter total: 10,753 parameters.

This exactly matches the deterministic condition's learned parameter count.

## Proposed evaluation axes

- seen wording, one- to four-digit operands;
- unseen simple wording, one- to four-digit operands;
- unseen simple wording, five- to eight-digit operands;
- unseen simple wording, nine- to twelve-digit operands;
- unseen word-problem wording, five- to eight-digit operands;
- held-out routing negatives;
- token-exact preservation relative to the untouched base;
- unit-off and wrong-state causal interventions.

Pilot data, seeds, and choices will be excluded from confirmation.

## Pilot resolution

The block-6 linear router did not generalize reliably. A development-only
architecture sweep showed that intent information was much more separable
after block 24. The frozen configuration therefore uses a
896-to-16-to-1 SiLU router after block 24. Its 14,369 parameters bring both
matched interfaces to 24,225 learned parameters. See the confirmatory protocol
for the complete frozen specification.
