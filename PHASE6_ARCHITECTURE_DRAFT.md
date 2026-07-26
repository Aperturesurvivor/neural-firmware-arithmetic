# Phase 6 Architecture Draft

Status: development-only. This is not a frozen confirmatory protocol.

## Objective

Remove the fixed decimal parser from typed neural firmware while retaining an
explicit, inspectable, deterministic execution boundary.

At inference, the architecture may receive only:

- the natural-language token sequence;
- the token attention mask;
- the location of the current causal anchor.

It may not receive character spans, operand strings, digit positions, or
operation labels from external code.

## Neural firmware controller

The first development target supports zero, one, or two addition calls.

For a single-call prompt:

```text
natural-language residual sequence
    -> learned operand registers A, B
    -> learned ADD/no-call controller
    -> frozen ADD(A, B)
    -> learned result-to-residual bridge
```

For a two-call prompt:

```text
natural-language residual sequence
    -> learned operand registers A, B, C
    -> learned two-call controller
    -> R1 = frozen ADD(A, B)
    -> R2 = frozen ADD(R1, C)
    -> learned result-to-residual bridge
```

The same frozen addition cell is reused for both calls. Each call's operands,
typed output symbols, and masks remain available as diagnostics. This
time-multiplexed microprogram is the initial repeated-calculation mechanism.
The runtime API must retain call-index and generation-index state so a later
version can expose an intermediate result to autoregressive reasoning and
resume with another call.

## Learned residual-to-register mapper

Early Qwen residuals are projected into a compact contextual width and passed
through a bidirectional sequence encoder. A transformer slot decoder uses
learned operand/digit queries to emit three fixed-length categorical
digit/PAD registers.

This mapper is trained with explicit register supervision but receives no
parser-derived locations at inference.

## Learned semantic controller

The implemented controller fuses two learned views:

- attention over the full early residual sequence;
- the late pre-final-RMSNorm residual at the causal anchor.

The fused five-class head distinguishes no call, supported ADD requests, and
unsupported single- or multi-operation requests. The semantic head determines
whether firmware is allowed to activate. Once activated, learned third-register
occupancy determines whether the program makes one or two calls. This ties
program length to the same neural register representation used for execution
instead of asking a separate prompt classifier to relearn operand count.

Hard negatives deliberately match positive request syntax while changing the
operation to multiplication, subtraction, averaging, comparison, quotation,
or non-evaluation. The controller must learn operation identity rather than
only imperative request shape.

Pilot v6 is a working development implementation, but it is not yet eligible
for confirmation. On the untouched gate split it achieved zero false calls at
the calibrated threshold and exact output on every correctly routed,
correctly extracted example, but positive routing was 92.25% and chained
register extraction was 96.0%. These remain below the gates below.

## Frozen calculator

The deterministic ripple-carry cell remains parameter-free and immutable.
For the second call, the first call's typed output symbols are converted
directly into the next typed operand register. No decimal string is parsed
between calls.

## Native residual return

A learned symbol-to-residual bridge injects the exact final typed result after
a late transformer block. Qwen's frozen normalization and output head produce
the visible decimal answer. This is an internal residual intervention, not
post-generation text replacement.

## Development gates before protocol freeze

- at least 99% exact registers on held-out single-call prompts;
- at least 98% exact registers on held-out two-call prompts;
- at least 99% correct call-count classification;
- at most 1% false calls on hard negatives;
- at least 99% conditional calculation/output accuracy when registers and
  call count are correct;
- demonstrated exact two-call execution with the same frozen cell reused;
- route-off token identity to the untouched model.

If these gates are not met, failures and architecture changes remain pilot
results. No confirmatory prompt families will be evaluated.
