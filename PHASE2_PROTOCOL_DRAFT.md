# Phase 2 Draft Protocol: Pretrained LLM Neural Firmware

Status: pilot-stage design. This document may change only while entries are
recorded in `PHASE2_PILOT_LOG.md`. A separate frozen protocol and configuration
will be committed before confirmatory seeds are run.

## Research question

Can an immutable exact-addition process be placed inside the forward path of a
pretrained approximately half-billion-parameter causal language model and
connected through a lightweight learned latent interface, while preserving the
base model on prompts where the arithmetic path is ineligible?

## Model

- Base: `Qwen/Qwen2.5-0.5B-Instruct`, 494 million parameters.
- Base parameters: frozen.
- Integration point: the final residual state immediately before the existing
  language-model output head.
- Deterministic component: immutable decimal addition and eligibility parser.
- Trainable interface: one latent residual vector per firmware symbol plus a
  small learned router. The target symbols are digits zero through nine and
  end-of-answer.
- Output representation: one decimal digit per token, separated by spaces, so
  the interface uses a stable set of digit tokens rather than memorizing
  multi-digit tokenizer vocabulary entries.

The deterministic component is called during `forward`/generation before the
language-model head. It is not an after-the-fact answer correction. It remains
a hybrid software/neural module rather than an algorithm compiled entirely
into ordinary transformer weights.

## Controls

1. Frozen pretrained model with no firmware.
2. A parameter-matched learned low-rank adapter trained on the same arithmetic
   examples without access to the deterministic result.
3. Latent firmware with the learned internal residual bridge.
4. Direct-logit firmware as a structural upper bound.
5. Firmware-off ablation of the trained latent model.

## Pilot questions

1. Does the tokenizer represent each space-prefixed digit as exactly one token?
2. At what latent strength does the learned bridge dominate base-model logits
   without numerical instability?
3. Can a learned router remain active through answer generation?
4. What training length gives the learned adapter a fair in-range comparison?
5. What batch and sequence sizes fit safely within 16 GB unified memory?

## Candidate evaluation splits

- In distribution: operands with 1--4 decimal digits.
- OOD primary: operands with 5--8 digits.
- OOD long: operands with 9--12 digits.
- Carry chains: constructed long-carry additions.
- Routing negatives: non-addition and malformed prompts, including prompts
  containing digits or a plus sign.
- Language preservation: fixed non-arithmetic prompts evaluated for exact
  greedy-output agreement with the frozen base model.

## Candidate primary endpoint

Sequence-level exact-match accuracy on 5--8 digit additions, aggregated over
three unseen training seeds. Confirmatory thresholds and sample counts will be
frozen only after pilot feasibility work.

## Claim boundary

Success would show that a pretrained generative language model can decode a
frozen exact process through an internal learned residual interface. It would
not show general mathematical correctness, learned symbolic parsing, or an
algorithm compiled entirely into normal transformer weights.

## Reproducibility record

The phase 2 artifact must retain:

- a chronological lab notebook covering all pilots, failures, and decisions;
- the exact Hugging Face model repository and immutable revision;
- `uv.lock`, Python/PyTorch/Transformers versions, macOS and hardware metadata;
- source commit, configuration, training seed, evaluation seed, and dataset
  logical hash for every reported run;
- trainable and frozen parameter counts;
- raw per-example predictions for local retention and compact error records in
  Git;
- wall time, optimization steps, losses, and peak process memory where
  available;
- exact commands needed to recreate the environment, caches, training,
  evaluation, analysis, figures, and paper;
- explicit separation between pilot-tuned choices and unseen confirmatory
  evaluation.

Operational actions and scientific decisions are recorded. Private model
chain-of-thought is neither available nor required for replication.
