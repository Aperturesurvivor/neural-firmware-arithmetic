# Preregistered Experimental Protocol

Date fixed: 2026-07-25

## Research question

Can a frozen, exact ripple-carry addition transducer embedded in a small causal
transformer preserve integer-addition accuracy beyond the operand lengths seen
during training?

## Claim boundary

This experiment evaluates deterministic arithmetic execution and its interface
with a generative sequence model. It does **not** establish general
mathematical reasoning, reliable natural-language formalization, or correctness
for arbitrary programs.

The frozen module receives a token sequence that already conforms to a narrow
addition grammar. Consequently, success establishes exact execution within that
grammar, not a solution to open-ended semantic parsing.

## Models

- `baseline`: causal transformer with no algorithmic subsystem.
- `latent_firmware`: identical trainable architecture plus an immutable
  ripple-carry transducer. The transducer emits a fixed latent code for the
  correct next answer token. The language-model output head must decode it.
- `direct_firmware`: identical transformer plus the same transducer, with its
  next-token result applied directly to logits. This is an architectural upper
  bound rather than the primary proposed model.

All frozen lookup tables and codebooks are excluded from trainable-parameter
counts. Trainable architecture and optimization settings otherwise match.
Pilot results showed that the baseline required 10,000 steps to learn the
training range, whereas the latent interface converged within 1,000 steps. The
confirmatory baseline therefore receives 10,000 steps and the latent model
1,000. This favors the baseline in optimization compute and means wall time is
descriptive rather than a controlled efficiency comparison.

## Data

Each example has the character-level form:

```text
<bos>OPERAND_A+OPERAND_B=ANSWER<eos>
```

Operands are nonnegative base-10 integers without leading zeros, except for
zero itself. Training samples operands uniformly by digit length from 1 through
6. Data are generated procedurally and reproducibly.

Loss is computed only for answer tokens and `<eos>`, so the experiment measures
answer generation rather than prompt memorization.

## Confirmatory evaluation sets

The same committed evaluation examples are used for every model and seed. The
confirmatory evaluation seed and all three training seeds were not used during
pilot development.

- `id_random`: 1–6 digit operands.
- `ood_primary`: 7–12 digit operands.
- `ood_long`: 13–20 digit operands.
- `carry_chain`: constructed cases dominated by long carry propagation.

The confirmatory study uses 1,000 examples per random split and at least 500
carry-chain examples.

## Primary outcome

Sequence-level exact-match accuracy on `ood_primary`, aggregated by model over
three independently initialized training seeds.

The main comparison is `latent_firmware` versus `baseline`.

The baseline uses a conventional causal transformer with fixed sinusoidal
positions. It is a generic small language-model baseline, not the strongest
task-specific arithmetic transformer known in the literature. Specialized
relative or coupled position schemes are treated as relevant prior work and a
limitation on comparative scope.

## Secondary outcomes

- exact-match accuracy on all other splits;
- next-token accuracy under teacher forcing;
- accuracy by maximum operand length;
- failure counts by malformed digit, premature termination, and incorrect
  arithmetic;
- trainable parameter count;
- training wall time and peak process memory where measurable.

## Success criterion

The architectural hypothesis receives initial support if:

1. `latent_firmware` improves mean `ood_primary` exact-match accuracy by at
   least 20 percentage points over `baseline`;
2. its mean `ood_primary` accuracy is at least 99%;
3. the improvement repeats across all three seeds; and
4. `direct_firmware` confirms that the deterministic path itself is exact on
   all valid confirmatory examples.

Failure to meet these criteria is reported as a negative or mixed result.

## Pilot versus confirmatory study

Pilot runs may change optimization, model size, signal scaling, or training
duration. Every such change must be recorded in `PILOT_LOG.md`. Once the study
configuration is frozen in `configs/study.json`, confirmatory outcomes may not
be used to tune it. A new configuration after inspecting confirmatory outcomes
constitutes a new study.

## Statistical reporting

Report per-seed accuracy, mean and standard deviation across seeds, paired
example-level differences on the fixed evaluation sets, and bootstrap 95%
confidence intervals. Do not treat multiple examples from one trained model as
independent evidence about training-seed variability.

## Reproducibility

The final artifact must include:

- source code and tests;
- exact environment lock;
- pilot and study configurations;
- random seeds;
- evaluation-set hashes;
- raw compact predictions or error records;
- analysis tables and figures;
- machine and software information;
- manuscript source and compiled PDF.
