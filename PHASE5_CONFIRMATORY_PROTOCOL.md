# Phase 5 Confirmatory Protocol

Status: frozen before confirmatory model inference.

Freeze date: 2026-07-26 (America/Los_Angeles).

## Primary question

On the same Qwen2.5-0.5B model, training examples, natural-language addition
prompts, and adversarial non-addition prompts, how does the existing typed
neural-firmware architecture compare with an IGC-style learned
input/calculator/output architecture?

The comparison has two distinct scopes:

1. **Matched learned-parameter budget:** typed firmware, ordinary learned
   adapter, and matched IGC each contain exactly 24,225 learned parameters.
2. **Native architecture size:** native IGC contains 597,819 learned
   parameters, while typed firmware remains at its natural 24,225-parameter
   size.

The native IGC therefore has 24.68 times as many learned parameters as typed
firmware. The untouched base has no added learned parameters.

## Frozen base model and decoding

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- All pretrained weights remain frozen.
- Greedy decoding uses the same chat template and generation budget in every
  condition.
- Training seeds: 10,701, 10,702, and 10,703.
- Routers are trained and thresholded on the exact residual seen immediately
  after block 24 and before Qwen's final RMS normalization.
- Every threshold and checkpoint was selected without confirmatory inference.

## Frozen conditions

### Untouched base

The unmodified checkpoint is evaluated once on every prompt. Its output is the
preservation reference for every trained condition and seed.

### Ordinary learned adapter

A learned request router after block 24 gates a rank-five SiLU residual
adapter. It has exactly 24,225 learned parameters:

- router: 14,369;
- residual adapter: 9,856.

It receives no deterministic state and no parsed operands.

### Typed neural firmware

The existing Phase 4 architecture is retained:

- fixed character boundary supplies exactly two decimal operands;
- learned block-24 semantic router decides whether addition is requested;
- frozen zero-parameter typed ripple-carry cell performs addition;
- learned eleven-symbol residual codebook drives output.

It has exactly 24,225 learned parameters:

- router: 14,369;
- output codebook: 9,856.

The fixed boundary does not choose the operation. This condition tests routing
and exact execution, but not learned operand extraction.

### Matched IGC-style calculator

This architecture receives no parsed operands at inference. A learned
bidirectional sequence encoder and anchor-query attention mapper after block 1
produce two twelve-slot digit/PAD registers. A learned linear router after
block 24 chooses addition/no-operation. A frozen typed calculator operates on
the argmax registers, and a learned gated output mapping modifies the
block-24 residual.

Its learned-parameter allocation is exactly 24,225:

- learned input mapping: 16,986;
- late router: 897;
- gated output mapping: 6,342.

### Native IGC-style calculator

The same dual-depth architecture uses a 64-unit-per-direction sequence encoder,
eight-head anchor-query attention, a two-layer late router, and a full-width
output codebook:

- learned input mapping: 572,696;
- late router: 14,369;
- gated output mapping: 10,754;
- total: 597,819.

The early mapper also predicts an operation label as an auxiliary training
target, but the actual route is selected by the frozen late router. This split
was selected on development data because Qwen's early residuals retained digit
information while semantic intent was substantially more separable at block
24. Operand extraction, routing, calculation, and output all execute inside a
single autoregressive model forward path.

## Frozen training

All trained conditions use the same generated training corpus:

- 2,400 positive and 2,400 negative routing/input examples;
- decimal lengths from one through twelve digits;
- 1,600 positive output-mapping/adapter examples with one- through eight-digit
  operands;
- 800 separate development examples used only for architecture diagnostics and
  threshold selection.

The three paired seeds use identical data and optimization counts. The IGC
input mapper receives supervised digit/PAD registers and operation labels, as
in IGC's auxiliary-loss training method. The output mapper is trained with the
ground-truth calculator result before free-running evaluation.

Pilot failures, the original post-normalization router checkpoints, and all
reallocated matched-budget variants remain archived. The frozen checkpoints
are those in `phase5_results/training_v2/manifest.json`; the manifest records
each SHA-256 value and superseded checkpoint.

## Untouched confirmatory data

Confirmatory family constants and deterministic generator tests exist in the
committed source, but no confirmatory prompt has been used for model inference,
training, architecture selection, thresholding, or debugging.

Four positive splits contain 100 prompts each:

1. unseen simple wording with one- through four-digit operands, seed 10,551;
2. unseen simple wording with five- through eight-digit operands, seed 10,552;
3. unseen simple wording with nine- through twelve-digit operands, seed 10,553;
4. unseen word-problem wording with five- through eight-digit operands, seed
   10,554.

One adversarial negative split contains 160 prompts, seed 10,555. It covers
subtraction, multiplication, division, remainder, comparison, concatenation,
repetition, quotation, refusal, explanation, parity-like questions, syntax,
identifiers, and averaging.

Every condition and seed receives the identical rendered prompts in the
identical order. The complete rendered data, outputs, generated token IDs,
routes, register predictions where applicable, latencies, and scores will be
archived without deletion.

## Outcomes

Primary accuracy outcome:

- pooled mathematical correctness over all 400 positive prompts, where the
  final decimal integer in the response equals the exact sum.

Secondary outcomes:

- mathematical correctness by split and seed;
- exact-format correctness;
- positive route recall;
- false-route rate on adversarial negatives;
- token-exact preservation relative to base on adversarial negatives;
- IGC exact operand-register recovery;
- conditional arithmetic accuracy when route and registers are correct;
- mean, median, and distribution of end-to-end latency;
- learned-parameter count and parameter ratios.

The untouched base is not duplicated to inflate a sample size. Trained
conditions are reported per seed and as seed-averaged rates with raw counts.

## Precommitted comparisons and interpretation

1. Typed firmware versus untouched base.
2. Typed firmware versus ordinary adapter at exactly 24,225 parameters.
3. Typed firmware versus matched IGC at exactly 24,225 parameters.
4. Typed firmware at 24,225 parameters versus native IGC at 597,819
   parameters.

For paired correctness differences, report raw paired counts, percentage-point
differences, paired bootstrap 95% intervals clustered by prompt and seed, and
exact McNemar tests where applicable. Report Wilson 95% intervals for route,
preservation, and register rates.

The architectures count as **comparably reliable** only if the paired 95%
interval for typed-minus-native-IGC mathematical accuracy lies wholly within
minus five to plus five percentage points.

Typed firmware counts as a **parameter-efficiency advantage** if:

- its mean accuracy is no more than five percentage points below native IGC;
- it uses at most one tenth as many learned parameters;
- its negative-prompt token preservation is no worse by more than one
  percentage point.

Routing is considered operationally successful at at least 90% positive recall
and at most 2% false activation. Preservation is considered successful at at
least 99% token identity. These gates are reported separately from arithmetic
execution so a routing or extraction failure cannot be described as a
calculator error.

## Claim boundary

This phase is an independent, small-model, same-prompt comparison inspired by
Dietz and Klakow's Integrated Gated Calculator. It is not a reproduction of
their unavailable implementation, Llama 3.1 8B setup, four-operation
benchmark, or reported 17-million-parameter module.

A positive typed-firmware result would support parameter efficiency and/or
preservation under a fixed-parser boundary. It would not establish superiority
after learned parsing. A positive native-IGC result would support learned
operand extraction and internal calculation, but would not erase its larger
learned-parameter budget. Neither condition demonstrates general mathematics,
arbitrary-length arithmetic, or reasoning beyond the tested addition prompts.
