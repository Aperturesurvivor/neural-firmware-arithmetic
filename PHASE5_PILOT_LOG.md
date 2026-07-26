# Phase 5 Pilot Log

This file records architecture selection before confirmatory inference. Compact
pilot JSON files are retained under `phase5_results/`.

## Initial direct input mapping

A linear anchor-query attention mapper at block 1 failed operand extraction:

- matched 24,225-parameter allocation: 3.5% exact development registers;
- native 185,423-parameter allocation: 9.5% exact development registers.

Adding nonlinear multi-head attention and a feed-forward block did not solve
the token-order problem. Moving the same mapper to block 24 improved semantic
routing but reduced exact registers to zero.

## Sequence-aware learned input mapping

A bidirectional sequence encoder followed by anchor-query attention addressed
Qwen's grouped-number tokenization:

- native block-1 pilot: 78% exact registers and 98.65% non-padding digit
  accuracy;
- dual-depth pilot with block-1 extraction and block-24 routing/output: 93.5%
  exact registers, 8/8 sampled positive generations correct, and 0/8 false
  routes.

The original exact-budget allocation left only one recurrent unit per
direction and collapsed to placeholders. A reallocation to three units per
direction, a linear late router, and a narrower output code remained exactly
24,225 parameters but still recovered zero complete development registers.
That negative architecture is retained as the frozen matched-budget IGC arm.

## Three-seed training

The first complete training pass wrote all twelve seed checkpoints under
`phase5_artifacts/confirmatory_v1` and compact records under
`phase5_results/training_v1`.

Native IGC exact development-register accuracy was:

- seed 10,701: 97.125%;
- seed 10,702: 94.750%;
- seed 10,703: 89.500%.

Matched IGC recovered zero complete development registers for all three seeds.

## Router feature-point correction

The first pass trained late routers on `hidden_states[24]`. Inspection of
Qwen's implementation and a direct forward hook established that this tensor
is after the final RMS normalization, while the installed wrappers route on
the block-24 output before normalization.

On a diagnostic prompt:

- pre-normalization block-24 residual norm: 46.33;
- `hidden_states[24]`/last-hidden-state norm: 254.05;
- maximum coordinate difference: 86.65;
- applying Qwen's final RMS norm to the hooked residual reproduced the last
  hidden state exactly.

No confirmatory inference had occurred. All v1 checkpoints were preserved.
Routers were retrained on forward-hooked pre-normalization features and written
to `phase5_artifacts/confirmatory_v2`, with compact records in
`phase5_results/training_v2`.

After correction, the integrated 20-prompt development diagnostics produced:

- typed firmware: 30/30 sampled positive additions correct across seeds;
- native IGC: 28/30;
- ordinary adapter: 0/30;
- matched IGC: 0/30;
- every condition: 60/60 positive routes and 0/60 false routes across seeds.

These v2 checkpoints and thresholds are the only frozen confirmatory inputs.
