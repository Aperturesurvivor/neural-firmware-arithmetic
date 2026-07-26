# Neural Firmware Arithmetic

This repository tests whether an immutable algorithmic subnetwork can give a
small transformer exact integer-addition execution outside the numerical range
seen during training.

The originating hypothesis is Josiah Wilson's: known deterministic processes
should be installed inside a language model as locked computational structure,
rather than learned imperfectly from examples. OpenAI Codex is assisting with
experimental design, implementation, execution, analysis, and manuscript
preparation under Josiah's direction.

## First study

The study compares:

1. a character-level causal transformer trained on addition;
2. the same transformer with a frozen ripple-carry transducer injected as a
   latent code at answer-generation positions;
3. a direct-logit oracle integration that establishes the deterministic
   subsystem's upper bound.

Training operands contain 1–6 digits. The preregistered primary endpoint is
exact-match accuracy on randomly generated 7–12 digit additions. Secondary
tests include 13–20 digit operands and adversarial carry chains.

Across three confirmatory seeds, the generic transformer averaged 90.13%
exact-match accuracy in range but produced no correct answer on 6,000 random
out-of-range examples. The latent-firmware model answered all 10,500
confirmatory examples correctly. Setting its firmware strength to zero after
training reduced in-range accuracy to 0.1–0.4% and out-of-range accuracy to
zero.

The complete paper is
[`paper/neural-firmware-arithmetic.pdf`](paper/neural-firmware-arithmetic.pdf).
Read [`PROTOCOL.md`](PROTOCOL.md) before interpreting the results. It fixes the
claim boundary: the frozen module can make arithmetic execution exact, but it
does not automatically make natural-language parsing, routing, or decoding
correct.

## Pretrained-LLM study

The second experiment used the 494-million-parameter
`Qwen/Qwen2.5-0.5B-Instruct` model with every pretrained weight frozen. An
immutable decimal addition process generated deterministic digit/end symbols;
a 10,753-parameter learned bridge routed those symbols into the final
896-dimensional residual before the pretrained output head. This is internal
generation-time activation steering, not post-generation answer replacement,
but it is still output-adjacent rather than a deterministic unit inside the
repeated transformer blocks.

Across three confirmatory bridge seeds, latent firmware achieved 900/900
(100%) on five- to eight-digit OOD addition, compared with 189/900 (21.0%) for
a 270,336-parameter all-layer LoRA control. It also achieved 450/450 carry
chains and 864/900 (96.0%) on nine- to twelve-digit random additions.

The study was **not an overall preregistered success**. Token-exact preservation
on unrelated controls was 289/300 (96.3%), below the fixed 99% threshold. All
11 divergences were late activations on prompts that quoted a valid calculator
command while instructing the model to ignore it. This is a useful negative
result: exact computation still requires exact scoping and sequence-level
routing.

The primary phase-2 report is
[`paper_phase2/neural-firmware-pretrained-llm.pdf`](paper_phase2/neural-firmware-pretrained-llm.pdf).
The frozen protocol is [`PHASE2_PROTOCOL.md`](PHASE2_PROTOCOL.md), and the full
chronological record is
[`PHASE2_LAB_NOTEBOOK.md`](PHASE2_LAB_NOTEBOOK.md).

## Internal transformer-unit study

The third experiment implements the closer version of the originating idea:
the original sixth decoder block is wrapped in-place inside Qwen's repeated
layer stack with a learned residual-to-digit encoder, a frozen zero-parameter
ripple-carry cell, and a learned typed-symbol-to-residual decoder. Eighteen
unchanged transformer blocks and the original output path remain downstream.
This is neither post-inference correction nor an output-logit tool.

The two learned interfaces contain 18,826 parameters (about 0.0038% of the
494-million-parameter base). A same-depth learned adapter with exactly 18,826
parameters served as the control. Both saw only one- to four-digit training
addition.

Across three frozen confirmatory seeds, the internal deterministic unit
achieved:

- 450/450 exact on primary five- to eight-digit OOD addition;
- 450/450 exact on nine- to twelve-digit addition;
- 225/225 exact on carry chains;
- 1,575/1,575 exact internal operand-register recoveries;
- 300/300 token-exact preservation comparisons;
- 90/90 wrong-state and 39/39 donor-state causal interventions.

The parameter-matched learned control scored 0/450 on both random OOD splits.
Turning the deterministic unit off reduced primary accuracy to 0/450. The
correct first-digit logit margin remained positive after all eighteen
downstream blocks. All eight preregistered criteria passed.

The primary phase-3 report is
[`paper_phase3/native-deterministic-transformer-unit.pdf`](paper_phase3/native-deterministic-transformer-unit.pdf).
The frozen protocol is [`PHASE3_PROTOCOL.md`](PHASE3_PROTOCOL.md), and the
replication-oriented chronology is
[`PHASE3_LAB_NOTEBOOK.md`](PHASE3_LAB_NOTEBOOK.md).

## Reproduction

```bash
uv sync --extra dev
uv run pytest
uv run nf-study --config configs/study.json
uv run nf-analyze --study artifacts/confirmatory_v1/study.json

# Pretrained-LLM confirmatory study (about 66 minutes on the recorded M4)
uv run python scripts/run_phase2_study.py \
  --config configs/phase2_study.json \
  --artifact-directory phase2_artifacts/confirmatory_v1 \
  --result-directory phase2_results/confirmatory_v1
uv run python scripts/analyze_phase2.py

# Internal transformer-unit confirmatory study
uv run python scripts/run_phase3_study.py \
  --config configs/phase3_study.json \
  --artifact-directory phase3_artifacts/confirmatory_v1 \
  --result-directory phase3_results/confirmatory_v1
uv run python scripts/analyze_phase3.py
uv run python scripts/hash_phase3_artifacts.py
```

Compact metrics, configuration files, figures, and the compiled paper are
tracked in Git. Large checkpoints and full per-example predictions remain
local under ignored artifact directories.
