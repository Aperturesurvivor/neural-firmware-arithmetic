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

## Reproduction

```bash
uv sync --extra dev
uv run pytest
uv run nf-study --config configs/study.json
uv run nf-analyze --study artifacts/confirmatory_v1/study.json
```

Compact metrics, configuration files, figures, and the compiled paper are
tracked in Git. Large checkpoints and full per-example predictions remain
local under ignored artifact directories.
