# Phase 2 Lab Notebook

This is the chronological, replication-oriented record for the pretrained-LLM
neural-firmware study. Times use America/Los_Angeles unless otherwise noted.
Configurations and outputs referenced here are retained in the repository or
under the ignored `phase2_artifacts/` directory.

## 2026-07-25 — Study authorization and scope

- Josiah Wilson authorized a persistent goal to build, run, and document an
  experiment that inserts an immutable arithmetic process inside the forward
  path of a pretrained 0.5B--1.5B language model.
- Required completion behavior: write a notification-length success/failure
  summary to Josiah by text after the experiment and report are complete.
- The intended phase 2 architecture was separated from ordinary external tool
  use: frozen language-model weights, deterministic arithmetic execution before
  the language-model head, and a lightweight learned internal bridge/router.
- The existing toy-transformer repository was retained so phase 1 and phase 2
  share provenance and comparable terminology.

## 2026-07-25 — Hardware and execution choice

- Host: Apple M4 Mac mini, 10-core CPU, 10-core integrated GPU, 16 GB unified
  memory.
- Available storage before phase 2 model download: approximately 40 GiB.
- Hugging Face account authentication was confirmed as `Aperturesurvivor`.
- Although Hugging Face Jobs was available, local execution was selected
  because the official local-training guidance places 0.5B--1.5B frozen-base
  adapter experiments within the expected capacity of a 16 GB Apple Silicon
  Mac. This avoids paid cloud compute and keeps custom forward-path
  instrumentation directly inspectable.

## 2026-07-25 — Base-model selection

- Selected model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Reasons:
  - 494,032,768 parameters, within Josiah's requested half-billion class;
  - pretrained conversational causal decoder rather than a task-only model;
  - Apache-2.0 license;
  - standard Transformers implementation with accessible residual states and
    output head;
  - small enough for full-precision frozen inference on the local M4.
- Inspected configuration:
  - architecture: Qwen2 causal language model;
  - 24 transformer layers;
  - hidden width 896;
  - 14 attention heads and 2 key/value heads;
  - vocabulary size 151,936 in the model configuration;
  - tied token embeddings and output head.
- Parameters were loaded in float32 for the first feasibility check. MPS was
  available.

## 2026-07-25 — Environment changes

- Added `transformers`, `accelerate`, and `safetensors` to `pyproject.toml`.
- Ran `uv sync --extra dev`.
- Resolved environment included Transformers 4.57.6, Accelerate 1.14.0,
  safetensors 0.8.0, and the existing PyTorch 2.13 environment.
- Updated `uv.lock`.
- Added ignored locations for downloaded models and large phase 2 artifacts.

## 2026-07-25 — Tokenizer feasibility pilot

- Loaded the Qwen2.5 tokenizer.
- Tokenizer length: 151,665; model vocabulary allocation: 151,936.
- End-of-message token: ID 151645 (`<|im_end|>`).
- Verified that each bare decimal digit is exactly one stable token:
  `0`--`9` map to IDs 15--24.
- A leading space is a separate token (ID 220). Consequence: firmware can emit
  bare digit tokens sequentially and decode to an ordinary contiguous decimal
  answer without relying on multi-digit BPE tokens or a spaced-digit output
  format.

## 2026-07-25 — Initial phase 2 implementation

- Added a draft protocol and separate pilot log.
- Implemented reproducible addition-example, carry-chain, and routing-negative
  generators with fixed templates and seeds.
- Implemented an immutable decimal ripple-carry engine using fixed sum/carry
  transition tables and a deliberately narrow eligibility parser.
- Implemented a trainable residual bridge containing:
  - eleven latent vectors (digits zero through nine plus end-of-answer);
  - one learned linear router over the 896-dimensional residual state;
  - fixed-strength normalized residual injection immediately before the
    pretrained model's existing language-model head.
- Implemented a parameter-matched learned control mechanism: low-rank
  residuals on the final transformer's query and value projections.
- Implemented frozen hidden-state caching so bridge training does not
  backpropagate through 494 million base parameters.
- Implemented cached autoregressive generation modes for the unmodified base,
  latent firmware, direct-logit upper bound, and firmware-off ablation.
- Added unit tests for exact large decimal addition and registered-template
  parsing. The new tests passed. An initial Ruff pass found only line-length
  issues, which were corrected.

## 2026-07-25 — Clarification of the longer-term architecture

- Josiah clarified that his original target is a redesign of the repeated
  transformer stack itself: select or add an internal unit that acts as an
  exact calculator, plus learned token/residual translation into and out of
  that unit.
- Phase 2 remains the agreed prerequisite: an output-adjacent internal residual
  bridge in a real pretrained LLM.
- The more ambitious architecture-level experiment was recorded in
  `FUTURE_ARCHITECTURE_GOAL.md`.
- If phase 2 succeeds, the current persistent goal will be completed and a new
  `/goal` will be created for a deterministic unit inside or between repeated
  transformer blocks.

## 2026-07-25 — Pilot v1 results

- Ran `uv run python scripts/run_phase2_pilot.py --config
  configs/phase2_pilot_v1.json`.
- Configuration SHA-256:
  `5229a833521eecc755e7fbc6625ed7af69add18c78d4994f7443b23226020c21`.
- Logical dataset SHA-256:
  `6b9d4727c70141027c2bb942eaa61e63ed4b8bea6f71a2f270db0ea32c7a1321`.
- The frozen base scored 12.5% ID, 5.0% OOD 5--8, 2.5% OOD 9--12,
  and 6.7% on carry chains.
- The 10,753-parameter latent bridge at strength 32 scored 100%, 90%, 85%,
  and 76.7% respectively.
- The trained bridge with firmware disabled exactly reproduced the base-model
  scores.
- Direct-logit firmware scored 100% on all four splits.
- All 30 language/routing controls were token-exact preserved with no initial
  false routes.
- Full raw pilot predictions and hidden caches are retained locally under
  `phase2_artifacts/pilot_v1/`; compact summaries are under
  `phase2_results/pilot_v1/`.

## 2026-07-25 — Pilot v1 failure diagnosis and strength sweep

- Inspected all 17 latent failures. Router probabilities remained effectively
  one, and the deterministic target plan was correct; the base residual/logit
  stream overruled individual target digits.
- Reevaluated the failed examples at strengths 32, 48, 64, 96, and 128.
  Strength 48 corrected 16/17; strengths 64, 96, and 128 corrected 17/17.
- Ran a complete registered reevaluation with:
  `uv run python scripts/evaluate_phase2_strengths.py --strengths 48 64`.
- Strength 48 reached 100% on all random splits and 96.7% on carry chains.
- Strength 64 reached 100% on every arithmetic split and retained 100%
  language-output preservation with no initial false routes.
- Candidate decision: use 64, but first retrain at that strength and pilot the
  learned low-rank control before freezing the confirmatory configuration.

## 2026-07-25 — Learned-adapter pilot attempt 1 failed before training

- Command: `uv run python scripts/run_phase2_adapter_pilot.py`.
- Failure occurred on the first forward pass, before an optimizer step.
- Cause: low-rank query/value matrices were attached after the pretrained model
  had moved to MPS, so the new matrices remained on CPU while their inputs were
  on MPS.
- Resolution: changed `LowRankLinear` initialization to place the new matrices
  on the frozen base layer's device and dtype before initialization.
- The same registered pilot configuration will be rerun; this attempt has no
  accuracy result and will not be represented as a completed run.

## 2026-07-25 — Learned-adapter pilot v1

- Corrected device placement, reran the original 240-step configuration, and
  completed successfully.
- The adapter places rank-eight residuals on the final transformer's query and
  value projections and has 22,528 trainable parameters.
- Training loss fell from 0.9827 to 0.4890 in 21.06 seconds.
- Accuracy was 27.5% ID, 5.0% OOD 5--8, 2.5% OOD 9--12, and 6.7% on carry
  chains.
- Compared with the frozen base, the adapter improved ID exact match by 15
  percentage points but did not improve any extrapolation split.
- Decision: 240 steps does not establish a sufficiently trained learned
  in-range control. Run a fresh 2,000-step pilot and add non-arithmetic
  token-preservation evaluation before freezing the confirmatory duration.

## 2026-07-25 — Learned-adapter duration pilot

- Ran a fresh 2,000-step final-layer low-rank adapter on the same 240-example
  training set.
- Training loss fell from 0.9827 to 0.1531 in 147.55 seconds.
- Exact-match accuracy was 22.5% ID, 2.5% OOD 5--8, 2.5% OOD 9--12, and 3.3%
  on carry chains.
- Only 6/30 non-arithmetic outputs were token-exact preserved.
- Despite lower teacher-forced loss, longer localized adaptation did not
  improve greedy sequence accuracy and substantially changed unrelated
  outputs. This is retained as a negative result.
- Decision: do not use this localized adapter as the sole learned control.
  Pilot a more standard all-layer query/value LoRA with 1,000 arithmetic
  examples. This control will have substantially more trainable parameters
  than the firmware bridge, making any accuracy comparison conservative with
  respect to learned capacity.

## 2026-07-25 — All-layer learned-control pilot

- Trained rank-four query/value LoRA residuals in all 24 transformer blocks.
- Trainable parameters: 270,336, approximately 25.1 times the bridge/router
  parameter count.
- Training set: 1,000 one- to four-digit additions; 1,000 optimization steps,
  batch size one; training time 84.04 seconds.
- Loss fell from 0.7439 to 0.05994.
- Accuracy: 80.0% ID, 27.5% OOD 5--8, 0% OOD 9--12, and 3.3% carry chain.
- Only 3/30 unrelated outputs were token-exact preserved.
- Decision: use the all-layer control in confirmation because it represents a
  recognizable conventional adapter and has far more learned capacity than
  the firmware bridge. Retrain it independently for every confirmatory seed.

## 2026-07-25 — Pilot v2 fresh strength-64 bridge

- Ran `uv run python scripts/run_phase2_pilot.py --config
  configs/phase2_pilot_v2.json --artifact-directory
  phase2_artifacts/pilot_v2 --result-directory phase2_results/pilot_v2`.
- Configuration SHA-256:
  `b91e493c4c9a3d3c998391b6ca9af3664dc20bae1e3003fce444a71d5aba661d`.
- Logical dataset SHA-256:
  `1dcaa2fd56fe4ec7be888632e27438e9fa874abae489f27a7befdc4a25d34e48`.
- Fresh bridge training: 1,000 arithmetic prompts, 400 negative prompts, 120
  steps, strength 64.
- Latent firmware scored 220/220 across ID, both random OOD splits, and carry
  chains. Direct firmware was also 220/220.
- Base and firmware-off scores were identical on every split.
- All 40 routing/language outputs were token-exact preserved and no initial
  false route occurred.
- Candidate bridge configuration is now technically ready to freeze.
- Final pre-freeze task: give the all-layer learned adapter a 3,000-example,
  3,000-step pilot to select a conservative conventional-control budget.

## 2026-07-25 — Long all-layer control and final budget

- Trained a fresh all-layer LoRA on 3,000 examples for 3,000 steps.
- Training took 243.92 seconds. Final loss rose to 1.8447 rather than remaining
  near the lower values seen earlier.
- Accuracy: 65% ID and 0% on primary OOD, long OOD, and carry chains.
- Preservation: 2/30 outputs, or 6.7%.
- Decision: retain this as a destabilization/overtraining result. Freeze the
  1,000-example, 1,000-step all-layer control, which produced the best pilot ID
  exact-match accuracy (80%) and nonzero primary-OOD performance (27.5%).

## 2026-07-25 — Confirmatory protocol and runner validation

- Wrote `PHASE2_PROTOCOL.md` and `configs/phase2_study.json`.
- Pinned model revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Selected previously unused phase 2 seeds: training 401, 503, 601 and
  evaluation 771923.
- Fixed confirmatory counts: 300 examples in each random split, 150 carry
  chains, and 100 routing/language controls.
- Fixed four simultaneous success thresholds: at least 99% mean latent primary
  OOD exact match; at least a 50-point advantage over the learned adapter; at
  least 99% language preservation; no more than 1% initial false routes.
- Implemented a resumable study runner that writes each checkpoint, prediction
  set, and summary atomically by condition and seed.
- Added and ran a one-seed, two-step end-to-end smoke configuration covering
  base, direct, bridge, firmware-off, adapter, preservation, dataset hashing,
  checkpoint saving, and final study assembly.
- Re-ran the same smoke command to verify checkpoint/result resume behavior.
  The first resume command completed the study but its follow-up inspection
  used unavailable shell command `python`; inspection was immediately repeated
  with `uv run python` and confirmed one bridge and one adapter run were
  recovered.
