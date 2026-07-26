# Phase 5 Lab Notebook

Chronological, replication-oriented record for the typed neural-firmware
versus IGC-style experiment. Times use America/Los_Angeles.

## 2026-07-26 — Authorization and study objective

- Josiah instructed Codex to execute Phase 5, the direct same-model comparison
  recommended after Phase 4.
- Scope was fixed to Qwen2.5-0.5B-Instruct, addition, and three independently
  trained seeds.
- Required conditions were the untouched base, ordinary learned adapter,
  existing typed firmware, and an IGC-style learned
  input/calculator/output architecture.
- The study required both native-size and exactly parameter-matched IGC
  comparisons on identical positive and adversarial negative prompts.
- Frozen outcomes were arithmetic accuracy, routing, token-exact preservation,
  end-to-end latency, and learned-parameter count.

## 2026-07-26 — Closest-prior-work review

- Reviewed Dietz and Klakow, *IGC: Integrating a Gated Calculator into an LLM
  to Solve Arithmetic Tasks Reliably and Efficiently* (arXiv:2501.00684).
- IGC uses a learned token-to-categorical input mapping with auxiliary
  supervision, a non-differentiable calculator, a learned gate, and a learned
  output mapping.
- Its reported experiment uses frozen Llama 3.1 8B, approximately 17 million
  learned module parameters, and addition, subtraction, multiplication, and
  division.
- Public implementation code was unavailable. Phase 5 was therefore labeled
  an independent “IGC-style” small-model comparison, not an exact
  reproduction.

## 2026-07-26 — Implemented conditions

- Retained model `Qwen/Qwen2.5-0.5B-Instruct`, exact revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, with every pretrained parameter
  frozen.
- Typed firmware:
  - fixed character parser for exactly two decimal operands;
  - learned width-16 semantic router after block 24;
  - frozen typed ripple-carry addition;
  - learned eleven-symbol output codebook;
  - 24,225 learned parameters.
- Ordinary adapter:
  - the same late router;
  - rank-five SiLU residual adapter;
  - no parsed operands or deterministic state;
  - exactly 24,225 learned parameters.
- Matched IGC-style:
  - learned bidirectional recurrent input mapper with anchor-query attention
    after block 1;
  - learned linear router after block 24;
  - frozen typed calculator;
  - learned gated output mapping;
  - exactly 24,225 learned parameters.
- Native IGC-style:
  - the same dual-depth structure with 64 recurrent units per direction,
    eight attention heads, width-16 late router, and full output code;
  - 597,819 learned parameters, 24.68 times typed firmware.
- Added focused unit tests for phase-5 data separation, register construction,
  parameter counts, IGC insertion, route-off identity, and generation.

## 2026-07-26 — Input-mapper pilots

- A direct linear mapper failed: 3.5% exact development registers at the
  matched budget and 9.5% at the initial native size.
- Nonlinear attention did not recover token order. Moving the mapper from
  block 1 to block 24 improved semantics but reduced exact registers to zero.
- A bidirectional sequence encoder followed by anchor-query attention reached
  78% exact registers in the first native block-1 pilot.
- Splitting duties by depth—block 1 for digit extraction and block 24 for
  routing/output—raised native development recovery to 93.5% and yielded 8/8
  sampled generations correct.
- The exact-budget input mapper collapsed to PAD-heavy registers even after
  reallocating capacity from the router/output mapper. The retained matched
  condition is this best development-tested exact-budget allocation.
- All discarded pilot configurations and scores remain in
  `phase5_results/`; they were not silently replaced.

## 2026-07-26 — Three-seed training and feature-point correction

- Trained seeds 10,701, 10,702, and 10,703 on the same generated corpus:
  2,400 positive and 2,400 negative routing/input examples, 1,600 positive
  output/adapter examples, and 800 development examples.
- The first complete pass accidentally trained late routers using Qwen's
  post-final-RMSNorm `hidden_states[24]`, while the installed wrappers observe
  the block-24 residual before final normalization.
- Direct hooks measured residual norms 46.33 before normalization and 254.05
  after normalization, with a maximum coordinate difference of 86.65.
  Applying Qwen's final RMSNorm to the hook reproduced the returned hidden
  state exactly.
- No confirmatory inference had occurred. All v1 artifacts were retained.
  Routers were recalibrated on the correct pre-normalization feature and saved
  as v2 checkpoints.
- Integrated v2 development diagnostics across seeds:
  - typed firmware: 30/30 sampled positives correct;
  - native IGC-style: 28/30;
  - adapter: 0/30;
  - matched IGC-style: 0/30;
  - every condition: 60/60 sampled positive routes and 0/60 false routes.
- Native exact development-register accuracy by seed was 97.125%, 94.750%,
  and 89.500%. Matched IGC recovered zero complete registers for every seed.

## 2026-07-26 — Protocol freeze

- Froze `PHASE5_CONFIRMATORY_PROTOCOL.md` before confirmatory model inference.
- Protocol/source/pilot commit: `b90ae8c`.
- Confirmatory runner commit: `0b44d11`.
- Frozen positive splits contained 100 prompts each:
  - unseen simple wording, one- through four-digit operands;
  - unseen simple wording, five- through eight-digit operands;
  - unseen simple wording, nine- through twelve-digit operands;
  - unseen word problems, five- through eight-digit operands.
- Frozen adversarial set contained 160 non-addition prompts.
- The first runner invocation halted during data construction because one
  negative template included a third literal decimal span. The template was
  repaired and committed before model loading or confirmatory output. The
  original attempted launch produced no model inference.

## 2026-07-26 — Frozen confirmation

- The runner verified all checkpoint hashes and protocol ancestry.
- Execution environment:
  - macOS 26.5.2 arm64;
  - Python 3.12.12;
  - PyTorch 2.13.0;
  - Apple MPS available.
- Rendered-data SHA-256:
  `7bdd97cb8ac68c8673aca4662c2eef83ffb182373dd2db92508ee6cd07f3c015`.
- Total wall time: 2,909.72 seconds.
- No prompt, seed, output, or failed prediction was removed.

### Arithmetic and extraction

| Condition | Seed 10,701 | Seed 10,702 | Seed 10,703 | Pooled |
|---|---:|---:|---:|---:|
| Untouched base | — | — | — | 68/400 (17.0%) |
| Ordinary adapter | 13/400 | 10/400 | 10/400 | 33/1,200 (2.75%) |
| Typed firmware | 400/400 | 400/400 | 400/400 | 1,200/1,200 (100%) |
| Matched IGC-style | 0/400 | 0/400 | 1/400 | 1/1,200 (0.083%) |
| Native IGC-style | 384/400 | 340/400 | 360/400 | 1,084/1,200 (90.33%) |

- Typed firmware was exact-format correct on all 1,200 prompts and on every
  split for every seed.
- Native exact-register recovery was 384/400, 340/400, and 359/400:
  1,083/1,200 (90.25%) pooled.
- All 1,083 native examples with both route and registers correct produced the
  exact answer. One seed-10,703 output happened to be correct despite an
  incorrect register.
- Matched IGC recovered 0/1,200 exact operand pairs. Its one final-answer
  success was incidental.

### Paired comparisons

- Typed minus base: +83.0 percentage points; crossed seed/prompt bootstrap
  95% interval +79.25 to +86.75.
- Typed minus adapter: +97.25 points; interval +95.67 to +98.67.
- Typed minus matched IGC: +99.92 points; interval +99.58 to +100.0.
- Typed minus native IGC: +9.67 points; interval +4.33 to +15.25.
- Typed corrected every comparator error and introduced no arithmetic error in
  any paired seed.
- Exact McNemar tests are reported per seed in
  `phase5_results/confirmation_analysis_v1.json`. Pooled paired counts are
  descriptive because the same 400 prompts recur across seeds.

### Frozen interpretation rules

- “Comparably reliable” required the entire typed-minus-native interval to lie
  within -5 to +5 points. It did not: **criterion false**.
- Parameter-efficiency advantage required typed accuracy no more than five
  points below native, at most one tenth the learned parameters, and
  preservation no more than one point below native. Typed was more accurate,
  used 1/24.68 the parameters, and had equal preservation: **criterion true**.

### Routing and preservation

- Every trained condition activated on 1,200/1,200 positives.
- Typed, adapter, and native IGC each falsely routed 18/480 negatives (3.75%);
  matched IGC falsely routed 24/480 (5.0%).
- Every false typed/native route was an example of the unseen multiplication
  family “Compute the product of {a} and {b}; provide digits only.”
- Token-exact preservation:
  - typed firmware: 462/480 (96.25%);
  - adapter: 468/480 (97.5%);
  - matched IGC: 456/480 (95.0%);
  - native IGC: 462/480 (96.25%).
- No condition passed the frozen maximum-2% false-route gate or the
  minimum-99% preservation gate.

### End-to-end latency

- Mean positive latency:
  - typed firmware: 0.2788 seconds;
  - native IGC-style: 0.2885 seconds;
  - matched IGC-style: 0.2887 seconds;
  - ordinary adapter: 0.3193 seconds;
  - untouched base: 0.6239 seconds.
- Base emits longer prose on many prompts; its latency is therefore not a
  compute-only comparison.

## 2026-07-26 — Statistical analysis and reporting

- Added deterministic analysis with:
  - raw and split-level counts;
  - Wilson intervals;
  - exact per-seed McNemar tests;
  - 20,000-draw paired crossed seed/prompt bootstrap, seed 20,260,726;
  - learned-parameter and latency summaries;
  - precommitted decision evaluation;
  - raw-file SHA-256 inventory.
- Derived outputs:
  - `phase5_results/confirmation_analysis_v1.json`;
  - `phase5_results/confirmation_summary_v1.csv`;
  - `phase5_results/confirmation_comparisons_v1.csv`;
  - `phase5_results/confirmation_by_family_v1.csv`;
  - figures under `paper_phase5/figures/`.
- The final interpretation is deliberately mixed:
  - arithmetic and parameter efficiency strongly favor typed firmware under
    the fixed-parser boundary;
  - native IGC demonstrates viable learned extraction but with seed
    instability;
  - matched-budget learned extraction fails;
  - all learned routers fail the frozen adversarial safety and preservation
    gates.

## 2026-07-26 — Final validation

- Wrote the executive summary and 11-page technical report.
- Compiled `paper_phase5/main.tex` successfully with bundled Tectonic to
  `paper_phase5/neural-firmware-versus-igc.pdf`.
- Rendered and visually inspected the title/abstract, protocol, results,
  safety, interpretation, and reference pages. Corrected macro spacing and
  recompiled.
- Full repository validation passed:
  - 31 pytest tests;
  - Ruff over the complete repository;
  - deterministic confirmatory analysis;
  - checkpoint hash verification.
- Generated `phase5_results/artifact_manifest.sha256.json`, covering 58 files
  (46 tracked records/reports and 12 local ignored checkpoints).
