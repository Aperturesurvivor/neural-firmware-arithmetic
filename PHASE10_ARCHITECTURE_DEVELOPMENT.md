# Phase 10 Architecture Development Plan

Status: exploratory development; no Phase 10 confirmation set has been
generated or evaluated.

Originating research direction: Josiah Wilson. Experimental design,
implementation, and analysis assistance: OpenAI Codex.

## Question

Phase 9 established that the frozen deterministic calculator and result
decoder remained exact when routing and operands were correct, while the
frozen linear semantic interface did not become reliable. Phase 10 asks which
of three explanations best fits that failure:

1. the interface needs nonlinear capacity at the same learned-parameter
   budget;
2. the needed information exists but must be made more accessible by limited
   adaptation of the immediately surrounding representation;
3. routing and operand typing are distinct bottlenecks.

## Development conditions

All conditions inherit one retained Phase 8 implant during development. The
base model, implant layer and coordinates, calculator, result decoder,
activation bandwidth, runtime, and output strength remain fixed.

1. `linear`: the existing 32,768-weight linear route/role/digit interface.
2. `nonlinear`: a learned 16-unit SiLU bottleneck and learned 16-by-16 output
   mixer. A fixed projection retains 2,032 of TinyLlama's 2,048 hidden
   coordinates, giving 16×2,032 + 16×16 = 32,768 learned weights and therefore
   exactly matching `linear`.
3. `linear_representation`: the linear interface plus a rank-four nonlinear
   residual representation adapter local to the implant interface.
4. `nonlinear_representation`: the budget-matched nonlinear interface plus
   the same interface-local rank-four representation adapter.

The representation conditions add 16,384 learned parameters. They are not
described as parameter matched; their purpose is to test representation
accessibility separately from the matched interface-capacity comparison. The
adapter changes only the representation read by the route/role/digit
interface. The ordinary MLP continues to receive its original hidden state, so
route-off base computation remains unchanged.

## Development data and diagnostics

- Training uses the same Phase 8 sequence-interface data plus the frozen Phase
  9 hard-contrast training set.
- Architecture development uses the already disclosed Phase 9 sealed audit as
  a development set. It cannot become Phase 10 confirmation evidence.
- The development seed is 16,199 and is excluded from confirmation.
- Oracle routing forces the route on for positive prompts while leaving roles,
  digits, the deterministic calculator, and result decoding unchanged. It is
  reported only as a diagnostic.
- After joint interface training, the route threshold is calibrated on
  first-generation-step development features. No weights change during this
  calibration. Route logits are divided by the fixed temperature 2 before
  softmax; this preserves their ordering while avoiding float32 saturation at
  probability 1.0.
- Every retained development run, including unsuccessful conditions, remains
  recorded.

The first retained nonlinear pilot used 32,768 learned input weights followed
by SiLU and a fixed Hadamard output mixer. At the development false-positive
constraint it reached only 0.9% route recall without representation adaptation
and 13.2% with it. That output constraint was rejected during development in
favor of the true learned bottleneck above; its checkpoints and metrics remain
in the development artifacts.

An additional retained development pilot fed the rank-four representation
adapter into both the implant interface and the ordinary MLP. It improved
linear exactness from 37/100 to 44/100 and nonlinear exactness to 56/100, but
preserved only 185/200 negative outputs in both representation conditions.
That shared-MLP form was rejected in favor of the interface-local test above.

A retained route-hardening pilot updated only the two route-output rows for
each architecture. With 2,500 steps it saturated out-of-family development
probabilities and produced 6.5% to 20.5% false-positive rates in three
conditions. Learned route hardening was rejected; threshold-only calibration
is used instead.

## Confirmation boundary

After architecture and optimization choices are frozen, Phase 10 requires a
new exact-string-disjoint and operand-pair-disjoint audit. No result on the
previous Phase 9 audit is confirmatory Phase 10 evidence. Confirmation gates,
seeds, prompt hashes, parameter counts, and commands must be committed before
the new audit is evaluated.
