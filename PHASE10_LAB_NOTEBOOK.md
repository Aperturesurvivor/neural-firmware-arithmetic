# Phase 10 Lab Notebook

## Scope

Phase 10 tests whether the Phase 9 interface bottleneck is better explained by
nonlinear input capacity, representation accessibility, or semantic routing.
The base model, implanted coordinates, deterministic calculator, result
decoder, and runtime remain fixed.

## 2026-07-27 — development implementation

- Added a parameter-matched nonlinear interface and rank-four representation
  adapter.
- Added forced-route generation for an operand/decode diagnostic.
- Reused only the already disclosed Phase 9 confirmation prompts for Phase 10
  development.
- Retained all rejected development checkpoints and metrics outside Git;
  compact results are in `phase10_results/`.

## 2026-07-27 — rejected fixed-mix nonlinear pilot

The first nonlinear design used 32,768 learned input weights followed by SiLU
and a fixed orthogonal Hadamard mixer. At the development false-positive
constraint it reached 0.9% route recall without representation adaptation and
13.2% with it. The fixed output mixer was too restrictive.

## 2026-07-27 — learned bottleneck development

Replaced the fixed mixer with a true learned 16-unit SiLU bottleneck:

- fixed 2,032-of-2,048 input projection;
- 16×2,032 learned input weights;
- 16×16 learned output mixer;
- 32,768 learned input-interface weights total.

This preserved exact input-budget matching.

## 2026-07-27 — rejected shared-MLP representation form

Feeding the rank-four adapted representation into both the implant interface
and ordinary MLP improved exactness but preserved only 185/200 negative outputs
for both representation conditions. The shared-MLP form was rejected.

The retained end-to-end development results were:

| Condition | Exact | Oracle exact | False routes | Preserved |
|---|---:|---:|---:|---:|
| Linear | 37/100 | 87/100 | 2/200 | 198/200 |
| Matched nonlinear | 31/100 | 80/100 | 2/200 | 198/200 |
| Linear + shared representation | 44/100 | 88/100 | 2/200 | 185/200 |
| Nonlinear + shared representation | 56/100 | 82/100 | 7/200 | 185/200 |

## 2026-07-27 — rejected learned route hardening

A 2,500-step first-generation route-output training pass overfit the Phase 8
and Phase 9 training families. Three conditions produced development
false-positive rates between 6.5% and 20.5%. Learned route hardening was
rejected.

## 2026-07-27 — selected interface-local form

Localized the rank-four adapter to the route/role/digit interface. The ordinary
MLP retained its original hidden input. Added route temperature 2 to prevent
float32 softmax saturation without changing logit ordering, then calibrated
only the threshold on first-generation development features.

Final development:

| Condition | Exact | Oracle exact | False routes | Preserved |
|---|---:|---:|---:|---:|
| Linear | 37/100 | 87/100 | 2/200 | 198/200 |
| Matched nonlinear | 31/100 | 80/100 | 1/200 | 199/200 |
| Linear + local representation | 44/100 | 88/100 | 2/200 | 198/200 |
| Nonlinear + local representation | 38/100 | 82/100 | 2/200 | 198/200 |

Selected the linear interface-local representation condition for the primary
confirmatory hypothesis. Retained matched nonlinear as the capacity test and
linear as control.

## 2026-07-27 — protocol freeze

- Committed implementation and development outcomes at
  `541dc4c364945247248111a9fcf1c61c27af6185`.
- Froze 300 unique new prompts with canonical row SHA-256
  `ac70c9cc9bff349d461ac5c6ee6bc8d2e556e45f521b5d887937e6287472d440`.
- Committed the manifest separately at `ee1ef31`.
- No Phase 10 confirmation prompt had been evaluated.

## 2026-07-27 — frozen training

- Trained linear, matched nonlinear, and linear-local-representation
  conditions for seeds 16,201, 16,202, and 16,203.
- Verified all nine result decoders were bit-identical to their corresponding
  Phase 8 sources.
- Recorded checkpoint and source hashes at commit `492491f`.
- The calculator retained zero learned parameters.

## 2026-07-28 — sealed confirmation

The original uninterrupted evaluator completed 4,200 generations in
4,324.68 seconds.

| Metric | Seed 16,201 | Seed 16,202 | Seed 16,203 |
|---|---:|---:|---:|
| Linear exact | 37/100 | 35/100 | 35/100 |
| Matched nonlinear exact | 31/100 | 30/100 | 31/100 |
| Local representation exact | 46/100 | 42/100 | 53/100 |
| Local representation routes | 57/100 | 52/100 | 64/100 |
| Local representation active routes | 52/100 | 47/100 | 59/100 |
| Local representation operands | 89/100 | 90/100 | 89/100 |
| Local representation oracle exact | 89/100 | 90/100 | 89/100 |
| False routes | 0/200 | 0/200 | 0/200 |
| Token preservation | 200/200 | 200/200 | 200/200 |
| Ablation exact | 0/100 | 0/100 | 0/100 |

Representation paired gains were +9, +7, and +18. All five frozen
representation gates passed. Matched nonlinear paired changes were -6, -5,
and -4; the nonlinear hypothesis failed.

## 2026-07-28 — integrity and post-hoc analysis

- The initial independent completion audit passed 93/93 checks.
- Verified manifest and evaluated rows, all checkpoint/source hashes,
  bit-identical result decoders, parameter counts, zero-parameter calculators,
  row-level metrics, conditional mechanisms, ablations, and gate
  recomputation.
- Post-hoc two-way seed/prompt bootstrap for representation minus linear:
  mean +11.33 points, 95% percentile interval +4.0 to +20.0 points.
- Post-hoc nonlinear minus linear: mean -5.0 points, interval -10.33 to -0.33.
- Representation gains were concentrated in word problems and
  irrelevant-number distractors.

## 2026-07-28 — completion and environment audit

- Expanded the independent verifier to cover the full frozen protocol,
  including Git freeze order, dataset and family disjointness, operand-pair
  disjointness, training schedules, source-seed mappings, model and implant
  invariants, raw-output metric recomputation, generation bounds, category
  summaries, retained negative development runs, and report consistency. The
  expanded audit passed 299/299 checks.
- Regenerated the post-hoc analysis byte-identically and reran the full test
  suite: 102 tests passed.
- Added `phase10_results/environment_audit.json`. This is explicitly a post-hoc
  provenance reconstruction rather than contemporaneous telemetry. It also
  records hashes and sizes for all four frozen feature caches used by the
  committed training implementation.
- The committed `uv.lock` and `pyproject.toml` are byte-identical across the
  frozen training and confirmation commits. The unchanged project environment
  reports Python 3.12.12, PyTorch 2.13.0, Transformers 4.57.6, NumPy 2.5.1,
  macOS 26.5.2 on an Apple M4 Mac mini with 16 GB memory, and an available MPS
  backend. The committed model loader selects MPS when available.
