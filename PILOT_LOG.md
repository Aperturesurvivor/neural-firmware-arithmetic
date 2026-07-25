# Pilot Log

Pilot observations and engineering decisions will be appended here before the
confirmatory configuration is frozen.

## Pilot v1 — 2026-07-25

Configuration: `configs/pilot.json`; seed 17; training operands 1–4 digits;
500 optimization steps.

Exact-match accuracy:

| Model | ID random | OOD 7–12 | OOD 13–20 | Carry chain |
|---|---:|---:|---:|---:|
| Baseline | 0.040 | 0.000 | 0.000 | 0.000 |
| Latent firmware, strength 8 | 0.745 | 0.000 | 0.000 | 0.270 |
| Direct firmware | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

- The immutable ripple-carry path was exact throughout the tested domain.
- A latent code did not automatically dominate the transformer's learned
  residual stream at unseen sequence positions.
- Most latent-firmware failures were premature termination rather than wrong
  ripple-carry outputs.
- The baseline was undertrained, so this pilot cannot support a fair
  confirmatory comparison.

### Prespecified pilot-only strength sweep

Without updating weights, the latent-firmware checkpoint was reevaluated with
signal strengths 8, 16, 32, and 64. Strength 8 reproduced the table above;
strengths 16, 32, and 64 each achieved 1.000 exact-match accuracy on all four
splits. This establishes that the fixed latent code was decoded correctly but
was too weak relative to unseen-position activations.

Decision before confirmatory testing:

- Use latent strength 16, the smallest successful pilot value.
- Run a longer baseline pilot to select training duration based on adequate
  in-distribution learning rather than favorable OOD separation.
- Retrain latent strength 16 rather than relying only on post-training
  rescaling.

## Pilot v2 — 2026-07-25

### Long baseline

The baseline was trained for 5,000 steps on 1–4 digit operands. Its
teacher-forced loss fell to 0.027. Exact-match accuracy by maximum operand
length was:

| Digits | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1.000 | 1.000 | 0.893 | 0.939 | 0.017 | 0.000 |

This establishes a fairer control: the model learned lengths represented in
training and failed sharply immediately outside them.

### Retrained latent firmware

Retraining with strength 16 for 500 steps produced:

| ID 1–6 | OOD 7–12 | OOD 13–20 | Carry chain |
|---:|---:|---:|---:|
| 1.000 | 1.000 | 0.990 | 1.000 |

Both errors were premature termination on 20-digit operands. A post-training
pilot sweep found strengths 20, 24, and 32 each produced 1.000 on every split.

Decision:

- Use strength 32 for the confirmatory study. This is deliberately above the
  observed threshold so the deterministic code dominates unseen-position
  residual activations.
- Run a final baseline-duration pilot at the actual 1–6 digit training range.
- Train the baseline and latent-firmware models for the same number of steps in
  the confirmatory study. The direct-firmware upper bound requires only an
  initialization checkpoint because its correctness is structural rather than
  learned.

