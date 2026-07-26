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

## Natural-language routing and same-prompt study

The fourth experiment replaced the registered command with varied ordinary
English. A learned request-level router after Qwen's twenty-fourth block chose
whether to invoke the frozen typed addition cell. A fixed boundary supplied
the two candidate decimal strings but did not choose the operation. The
untouched base, an exactly parameter-matched learned residual adapter, and the
deterministic architecture received identical prompts.

On 400 frozen, never-used confirmatory addition prompts:

- untouched base: 68/400 (17.0%);
- equal-parameter learned control: 21/400 (5.25%);
- learned-router deterministic internal model: 360/400 (90.0%);
- oracle-route deterministic internal model: 400/400 (100%).

Every one of the 349 automatically activated deterministic executions was
exact. The internal model gained 73.0 percentage points over base and 84.75
points over the matched learned control. It produced zero false activations on
160 unseen non-addition prompts and preserved all 160 base outputs
token-for-token. With the unit forced off, all 400 positive outputs were
token-identical to base.

The compound preregistered verdict is not a full pass: five of six criteria
passed, but positive routing coverage was 87.25%, below the fixed 90%
threshold. The arithmetic mechanism worked without an observed error; semantic
route recall remains the bottleneck.

This phase does not originate the general integrated-calculator concept.
Dietz and Klakow's 2025
[Integrated Gated Calculator](https://arxiv.org/abs/2501.00684) is direct
prior work on a frozen Llama 3.1 8B model. Phase 4 is best read as an
independent small-model replication and controlled extension, distinguished
by its 24,225-parameter interface, exact parameter-matched control,
precommitted same-prompt evaluation, held-out natural-language routing, and
causal on/off tests. The different reported accuracies are not directly
comparable because the models and tasks differ.

The full phase-4 paper is
[`paper_phase4/natural-language-deterministic-arithmetic.pdf`](paper_phase4/natural-language-deterministic-arithmetic.pdf).
For a concise explanation, see
[`PHASE4_EXECUTIVE_SUMMARY.md`](PHASE4_EXECUTIVE_SUMMARY.md). The frozen
protocol and chronological record are
[`PHASE4_CONFIRMATORY_PROTOCOL.md`](PHASE4_CONFIRMATORY_PROTOCOL.md) and
[`PHASE4_LAB_NOTEBOOK.md`](PHASE4_LAB_NOTEBOOK.md).

## Typed firmware versus IGC-style study

The fifth experiment directly compared the fixed-parser typed architecture
with an independently implemented learned-input/calculator/output
architecture inspired by IGC. Every condition used the same frozen
Qwen2.5-0.5B-Instruct revision, addition data, three paired training seeds,
400 held-out positive prompts per seed, and 160 adversarial negatives per
seed.

Confirmatory mathematical accuracy was:

- untouched base: 68/400 (17.0%);
- ordinary 24,225-parameter adapter: 33/1,200 (2.75%);
- 24,225-parameter typed firmware: 1,200/1,200 (100%);
- matched 24,225-parameter IGC-style model: 1/1,200 (0.083%);
- native 597,819-parameter IGC-style model: 1,084/1,200 (90.33%).

Typed minus native accuracy was +9.67 percentage points, with a frozen paired
crossed seed/prompt bootstrap 95% interval of +4.33 to +15.25 points. Typed
therefore passed the precommitted parameter-efficiency rule: it was more
accurate, used 24.68 times fewer learned parameters, and had equal
preservation.

The boundary is important. Typed firmware received its operands from a fixed
decimal parser. Native IGC learned operand extraction and recovered exact
registers on 1,083/1,200 prompts; all 1,083 eligible calculator executions
were correct. The matched-budget IGC result shows that the selected tiny input
mapper failed, not that learned parsing is impossible at that budget.

This phase was not a compound safety success. Typed and native IGC each
falsely activated on 18/480 multiplication prompts and preserved only
462/480 (96.25%) negative outputs token-for-token, below the frozen 99% gate.
The arithmetic mechanism passed; robust operation routing did not.

The full phase-5 paper is
[`paper_phase5/neural-firmware-versus-igc.pdf`](paper_phase5/neural-firmware-versus-igc.pdf).
The concise result, frozen protocol, and chronology are
[`PHASE5_EXECUTIVE_SUMMARY.md`](PHASE5_EXECUTIVE_SUMMARY.md),
[`PHASE5_CONFIRMATORY_PROTOCOL.md`](PHASE5_CONFIRMATORY_PROTOCOL.md), and
[`PHASE5_LAB_NOTEBOOK.md`](PHASE5_LAB_NOTEBOOK.md).

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

# Natural-language same-prompt confirmation
uv run python scripts/run_phase4_confirmation.py
uv run python scripts/analyze_phase4_confirmation.py

# Typed firmware versus IGC-style confirmation
uv run python scripts/run_phase5_confirmation.py
uv run python scripts/analyze_phase5_confirmation.py
uv run python scripts/hash_phase5_artifacts.py
```

Compact metrics, raw predictions, configuration files, figures, and compiled
papers are tracked in Git. Large checkpoints remain local under ignored
artifact directories.
