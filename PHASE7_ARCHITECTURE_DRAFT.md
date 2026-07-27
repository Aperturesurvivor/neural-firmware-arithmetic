# Phase 7 Development Architecture: In-Place Deterministic Neuron Implant

Status: development-only. A causal pilot is positive, but the core hypothesis
is not yet established across fresh prompt families and independent seeds.

## Question

Can a pretrained Qwen2.5-0.5B model learn to encode addition into a frozen
deterministic microcircuit that occupies existing MLP activation channels, use
the exact result through ordinary downstream layers, and preserve unrelated
behavior when those channels are not routed?

This phase is the first implementation of the canonical architecture in
`FUTURE_ARCHITECTURE_GOAL.md`. Earlier phases inserted separately controlled
modules after transformer blocks. Phase 7 instead replaces selected coordinates
inside one existing Qwen MLP intermediate activation tensor.

## In-place interface

Qwen2.5-0.5B has 4,864 intermediate MLP channels per decoder layer. The first
anchor-token prototype selected 108 low-use coordinates:

- 2 route channels: `OFF`, `ADD`;
- 88 operand channels: two operands, four positions, eleven categorical values
  (`0`-`9`, `PAD`);
- 6 answer-position channels: positions zero through five;
- 12 result channels: `0`-`9`, `EOS`, `PAD`.

The selected input rows replace the corresponding rows of the MLP input
projection. The selected result columns replace the corresponding columns of
the MLP output projection. They are stored separately during training so the
optimizer does not allocate state for Qwen's full matrices, but they are
semantically rows and columns of the existing MLP and can be merged into an
exported weight representation.

All 4,756 unselected channels retain the pretrained Qwen computation. The
selected pretrained channels are zeroed, and the frozen implant generates
their replacement activations.

That anchor-token prototype recovered the route and answer position but could
not linearly recover both complete operands from one residual vector. The
current sequence-distributed implementation therefore occupies 34 channels at
decoder layer 23:

- 2 route channels: `OFF`, `ADD`;
- 3 per-token roles: `NONE`, operand A, operand B;
- 11 per-token digit values: `0`-`9`, `NON_DIGIT`;
- 6 result positions: positions zero through four plus `EOS`;
- 12 result channels: `0`-`9`, `EOS`, `PAD`.

Digit and role values are read at the existing number-bearing token positions.
The deterministic circuit scans those neuron activations, reconstructs the two
operands, adds them exactly, and writes the requested result symbol back
through the result channels. This reduces the learned replacement weights to
30,464: 19,712 input-row weights and 10,752 result-column weights. The
calculator itself still has zero learned parameters, and neither the 4,864-wide
MLP activation nor the 896-wide residual stream grows.

## Frozen execution

At an eligible generated token, the current sequence implant:

1. reads the categorical route, operand, and answer-position activations;
2. takes hard categorical decisions;
3. executes the existing zero-parameter decimal ripple-carry addition cell;
4. selects the exact result symbol for the internally represented answer
   position;
5. writes a one-hot activation into the twelve result coordinates;
6. returns through the selected columns of Qwen's ordinary MLP down projection.

Generation latches the first routing decision for the current response. If it
is `OFF`, every selected channel contributes exactly its original pretrained
value for the whole response. If it is `ADD`, the calculator may emit one
symbol at each successive result position. A post-audit variant advances that
result position with a fixed counter inside the calculator protocol instead of
asking a learned row to infer the position independently at every token. This
response-local state prevents ordinary text generated later in the response
from accidentally turning the calculator on and prevents repeated/skipped
result digits. It is control-state scaffolding in the present implementation,
not yet recurrent state encoded entirely in Qwen's residual stream.

The deterministic cell receives no gradient and has no learned parameters.
Operand extraction, routing, answer-position tracking, and result
interpretation remain learned and are reported separately.

## Training ladder

The hard calculator blocks gradients between its result and input interface.
The first development ladder therefore uses explicit scaffolding:

1. Train the selected input rows on route, typed operand, and answer-position
   supervision from frozen Qwen residuals.
2. Train the selected result columns through the normal language-model loss,
   teacher-forcing the typed interface while leaving the calculator exact.
3. Remove teacher forcing and evaluate the hard predicted interface
   end-to-end.
4. If the hard interface works, continue task-loss training and reduce
   auxiliary supervision in later experiments.

This scaffolding tests whether Qwen's native residual representations contain
enough information to drive the implanted activation ABI. It does not by
itself establish that unsupervised language-model loss would discover the ABI
from scratch.

## Development evidence required

- Interface:
  - exact ordered operand recovery;
  - answer-position accuracy;
  - arithmetic route true-positive rate;
  - adversarial route false-positive rate.
- End-to-end:
  - exact generated addition answers;
  - unseen prompt-family and digit-length performance;
  - negative-prompt activation and output preservation.
- Causality:
  - arithmetic accuracy with the result channels ablated;
  - arithmetic accuracy with typed operands permuted;
  - preservation with the implant forced off.
- Architecture:
  - selected channel indices and census scores;
  - trainable parameter count;
  - proof that the frozen cell has zero learned parameters;
  - proof that the MLP width and residual width are unchanged.

## Claim boundary

A positive pilot would show that a pretrained transformer can learn and use an
in-place deterministic activation subspace under supervised interface
scaffolding. It would not yet show spontaneous discovery during pretraining,
unlimited computation, four-operation reasoning, or general scientific
validity.
