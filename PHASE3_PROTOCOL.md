# Frozen Phase 3 Protocol: Native Deterministic Transformer Unit

This protocol and `configs/phase3_study.json` are to be committed before any
confirmatory training or evaluation. Pilot chronology, defects, and selection
decisions are retained in `PHASE3_LAB_NOTEBOOK.md`. Confirmatory source commit,
configuration hash, logical evaluation hash, and training-set hashes will be
written by the study runner.

## Research question

Can a zero-parameter exact arithmetic cell consume operands decoded from
intermediate residual states inside a pretrained half-billion-parameter
transformer, write typed results back into that internal residual stream, and
produce length-extrapolating exact answers after eighteen subsequent unmodified
transformer blocks?

## Fixed base model

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Parameters: 494,032,768.
- Revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Architecture: 24 Qwen2 decoder blocks, hidden width 896.
- All pretrained weights remain frozen in every internal-unit and learned
  control condition.

## Main internal architecture

The original sixth Qwen decoder block is wrapped in-place inside
`model.model.layers`. The wrapper executes:

1. the unchanged sixth Qwen block;
2. a learned affine residual-to-digit encoder;
3. a frozen typed decimal ripple-carry cell;
4. a learned eleven-symbol residual decoder;
5. the unchanged blocks 7 through 24, final RMS normalization, and tied
   language-model head.

The unit therefore runs after block 6 and before eighteen ordinary downstream
blocks. No output logit is directly changed and no generated text is replaced.

### Input interface

The registered request is:

```text
Use the internal addition register. Return only the exact integer.
A = 1 2 3
B = 4 5
```

A strict full-request grammar locates operand digit token positions but does
not read their values. Eligibility is latched once and remains immutable for
the response. A learned 896-to-10 affine classifier converts each selected
block-6 residual into a typed decimal digit.

The classifier is trained for 300 steps on digit residuals from 1,000
one- to four-digit training examples, then frozen. It contains 8,970 trainable
parameters during its training stage.

### Deterministic cell

The cell contains immutable sum and carry buffers and no trainable parameters:

\[
s=(a+b+c)\bmod 10,\qquad
c'=\left\lfloor(a+b+c)/10\right\rfloor.
\]

It receives only predicted typed digits and fixed operand-length masks. It
does not receive ground-truth digit values or answers at inference.

### Output interface

An eleven-entry 896-dimensional codebook represents digits zero through nine
and end-of-answer. The normalized selected vector is added after block 6 at
the current answer-generation position with fixed strength 64.

The codebook contains 9,856 trainable parameters and is optimized for 120
steps, batch two, learning rate 0.01. Gradients pass through all eighteen
frozen downstream blocks to the codebook, but do not update those blocks.

Combined learned-interface parameter count is 18,826, approximately 0.0038% of
the base model. The deterministic cell has zero trainable parameters.

## Controls

1. **Frozen base:** unchanged Qwen on the same controlled prompts.
2. **Unit off:** the trained wrapped model with its internal unit inactive.
3. **Parameter-matched learned control:** a rank-ten bottleneck residual
   adapter inserted after block 6, active at the same answer positions, with
   no deterministic state. Its down/up projections contain exactly 18,826
   trainable parameters. It receives 1,000 steps, batch two, learning rate
   0.001.
4. **Wrong-state intervention:** replace the typed result with a known result
   whose first digit is wrong and test whether generation follows it.
5. **State substitution:** replace a recipient's typed result with another
   equal-length example's state and test whether generation follows the donor.
6. **Unit-location pilot:** the already completed depth comparison after
   blocks 6, 12, 18, and 22 is secondary pilot evidence only and is not merged
   into confirmatory estimates.

## Training seeds

Independent confirmatory training seeds: 2101, 2203, and 2309.

For each seed:

- generate 1,000 one- to four-digit training additions;
- independently initialize and train the digit encoder;
- freeze the encoder;
- independently initialize and train the internal residual decoder;
- independently initialize and train the parameter-matched learned control.

The control and internal interfaces receive the same seed-specific examples.

## Fixed evaluation

Evaluation seed 884321 was unused during phase-3 tuning.

Each trained seed is evaluated on the same:

- 150 one- to four-digit random additions;
- 150 five- to eight-digit primary OOD additions;
- 150 nine- to twelve-digit long OOD additions;
- 75 five- to twelve-digit carry-chain additions;
- 100 ineligible language/routing controls;
- 30 long-OOD wrong-state interventions;
- equal-length state-substitution pairs available within those 30 examples;
- ten downstream first-symbol traces.

All arithmetic results use whole-sequence exact match under free-running greedy
autoregressive generation. Teacher-forced accuracy is not a final outcome.

## Preservation test

The ineligible set includes ordinary language plus valid internal-register
commands embedded in quotes, negations, documentation, prefixes, and suffixes.
Every prompt must fail the strict full-request grammar before generation. The
wrapped model's complete token sequence is compared with an independently
loaded frozen base model.

## Downstream trace

For the first answer symbol of ten long-OOD examples per seed:

- capture the current-position residual after block 5;
- capture it after the inserted unit at block 6;
- capture it after every subsequent block through block 24;
- apply the shared final RMS normalization and tied output head as a logit
  lens;
- report target rank and target-versus-best-competitor margin.

This trace is diagnostic. The preregistered trace criterion uses the minimum,
over post-unit depths 6--24, of the mean seed/example target margin.

## Simultaneous success criteria

Phase 3 succeeds only if all of the following hold:

1. Mean internal-unit primary OOD exact match is at least 99%.
2. Its mean primary OOD advantage over the parameter-matched learned control is
   at least 75 percentage points.
3. Mean exact two-operand register recovery on primary OOD is at least 99.5%.
4. Aggregate token-exact preservation is at least 99%.
5. At least 95% of wrong-state generations exactly follow the intervened
   state.
6. At least 95% of state-substitution generations exactly follow the donor
   state.
7. Disabling the unit lowers primary OOD exact match by at least 80 percentage
   points.
8. The minimum post-unit mean logit margin across blocks 6--24 is greater than
   zero.

All eight criteria are required. Secondary outcomes must be reported
regardless of the verdict.

## Statistical reporting

- Report every training seed separately and pooled exact counts.
- Report mean and sample standard deviation across seeds.
- Bootstrap the three paired primary internal-minus-control differences with
  a fixed analysis seed and identify training seed as the resampling unit.
- Report exact counts for register decoding, preservation, and interventions.
- Preserve every arithmetic, register, preservation, and causal-test error.
- Do not tune, replace, or rerun an unsuccessful confirmatory seed under the
  same label except to resume a retained checkpoint/result.

## Claim boundary

Success would establish that:

- predicted operand values entered a zero-parameter deterministic cell inside
  the repeated transformer stack;
- the cell's typed result causally controlled free-running output;
- exact state survived eighteen downstream native blocks;
- the architecture extrapolated beyond interface training lengths on the
  registered grammar.

It would not establish:

- general natural-language mathematical understanding;
- operation selection outside the registered command;
- subtraction, multiplication, signed or decimal arithmetic;
- a single scalar biological-style neuron;
- a calculator compiled entirely into ordinary attention/MLP weights;
- correctness beyond the tested grammar, lengths, model, decoding method, and
  host software environment.
