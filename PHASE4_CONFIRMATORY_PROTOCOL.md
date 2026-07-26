# Phase 4 Confirmatory Protocol

Status: frozen before any confirmatory inference.

Freeze date: 2026-07-25 (America/Los_Angeles).

## Primary question

On exactly the same previously unseen natural-language addition prompts, does
an internal deterministic arithmetic unit produce a large increase in
mathematical correctness over both the untouched 0.5-billion-parameter model
and an equal-trainable-parameter learned residual adapter?

## Frozen model and interfaces

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Model revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Greedy decoding; identical chat template and generation budget by condition.
- Internal insertion: immediately after transformer block 24 and before the
  model's final RMS normalization and language-model head.
- Request-level route decision: final request-token residual, two-layer
  896-to-16-to-1 SiLU classifier, threshold 0.76, latched for the entire
  response.
- Deterministic payload: frozen typed ripple-carry addition cell with no
  trainable parameters and an 11-symbol residual codebook.
- Learned control payload: rank-five SiLU residual adapter with no down bias.
- Internal and control learned-interface counts: exactly 24,225 each:
  14,369 router parameters plus 9,856 payload parameters.
- Base weights remain frozen in all trained-interface conditions.
- Candidate-number boundary: a fixed parser identifies exactly two ordinary
  contiguous nonnegative decimal spans and converts characters to typed
  digits. It does not choose the operation.

Frozen checkpoint SHA-256 values:

- router: `201262b5cf21259977dc8a31e3faa1aa77892f7cbae121ea015f4e69d95f8e66`
- deterministic unit:
  `8079e0c5d723405881c39e47773fb895617d5c4da99b3168996e2861aba9a739`
- learned control:
  `7d43f58126fc60bfb68ee2caa479818ce2b13b9b1c8d9f2b4b7adccb9840da94`

No component will be retrained, thresholded, or selected using confirmatory
outputs.

## Untouched confirmatory data

The `CONFIRMATORY_*` family constants were not imported by any pilot script.
They will first be instantiated by the confirmatory script after this protocol
is committed.

Four positive splits contain 100 prompts each:

1. unseen simple wording, one- to four-digit operands, seed 9501;
2. unseen simple wording, five- to eight-digit operands, seed 9502;
3. unseen simple wording, nine- to twelve-digit operands, seed 9503;
4. unseen word-problem wording, five- to eight-digit operands, seed 9504.

A routing-safety split contains 160 unseen non-addition prompts with one- to
twelve-digit operands, seed 9505. It covers subtraction, difference,
multiplication, division/remainder, comparison, concatenation, repetition,
quotation, refusal, explanation, parity, syntax, identifiers, and averaging.

The generator, seeds, rendered prompts, expected sums, responses, token IDs,
route probabilities, latencies, and per-example scores will be archived.

## Conditions on identical prompts

1. untouched base model;
2. learned residual control with the frozen learned router;
3. deterministic internal unit with the same frozen learned router;
4. deterministic internal unit with oracle-on routing;
5. deterministic internal unit forced off.

The forced-off output must be token-identical to the untouched base. Negative
prompts compare the learned-router internal condition against base for
token-exact preservation.

## Outcomes and success criteria

Primary outcome:

- pooled mathematical correctness over all 400 positive prompts, where the
  final decimal integer in the response must equal the exact sum.

Primary comparisons:

- deterministic learned-router internal versus untouched base;
- deterministic learned-router internal versus parameter-matched learned
  control.

Secondary outcomes:

- mathematical accuracy by split;
- exact-format accuracy, requiring the complete stripped response to be exactly
  the sum;
- router true-positive rate on positive prompts;
- router false-positive rate and token-exact preservation on negative prompts;
- oracle-on accuracy;
- forced-off token identity.

The result counts as a confirmatory success only if all of the following hold:

- pooled deterministic learned-router mathematical accuracy exceeds base by at
  least 30 percentage points;
- pooled deterministic learned-router mathematical accuracy exceeds the
  matched learned control by at least 30 percentage points;
- learned-router activation is at least 90% on positive prompts;
- learned-router false activation is at most 2% on negative prompts;
- oracle-on mathematical accuracy is at least 99%;
- forced-off outputs are 100% token-identical to base.

Report paired exact McNemar tests for the two primary comparisons, Wilson 95%
intervals for accuracies and route rates, paired percentage-point differences,
and all raw counts. No prompt or output will be removed after inference.

## Interpretation boundary

A positive result will demonstrate exact arithmetic execution integrated into
a transformer forward pass, with learned semantic invocation and a large
same-prompt gain over a weak base and matched learned adapter. It will not show
that a single conventional scalar neuron performs arithmetic, that arbitrary
mathematics is solved, or that number extraction is learned. The exact cell is
a small structured module, and candidate decimal extraction remains fixed.
