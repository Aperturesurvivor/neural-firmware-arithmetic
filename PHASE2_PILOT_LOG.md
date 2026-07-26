# Phase 2 Pilot Log

Pilot observations and engineering decisions will be appended here before the
phase 2 confirmatory protocol is frozen.

See `PHASE2_LAB_NOTEBOOK.md` for the complete chronological operational record.

## Pilot v1 — 2026-07-25

Configuration: `configs/phase2_pilot_v1.json`. Base model:
`Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Training seed 17; evaluation
seed 260725; 240 arithmetic training prompts; 120 routing-negative prompts; 80
bridge optimization steps.

The pretrained base has 494,032,768 frozen parameters. The latent bridge and
router have 10,753 trainable parameters, or approximately 0.0022% as many
parameters as the base.

### First registered evaluation: strength 32

| Model | ID 1–4 | OOD 5–8 | OOD 9–12 | Carry chain |
|---|---:|---:|---:|---:|
| Frozen pretrained base | 12.5% | 5.0% | 2.5% | 6.7% |
| Trained bridge, firmware disabled | 12.5% | 5.0% | 2.5% | 6.7% |
| Latent firmware, strength 32 | 100.0% | 90.0% | 85.0% | 76.7% |
| Direct-logit firmware | 100.0% | 100.0% | 100.0% | 100.0% |

The firmware-off output exactly reproduced base-model accuracy, demonstrating
that training the isolated bridge did not alter the pretrained weights.
Thirty routing/language controls had 100% token-exact output preservation and
zero initial false routes.

Training reduced the combined loss from 0.4148 to 0.0326 in 2.94 seconds once
the frozen hidden cache had been collected. The complete pilot took 208.89
seconds including model loading, cache generation, four-mode autoregressive
evaluation, and routing controls.

### Failure diagnosis

All 17 latent failures had router probabilities effectively equal to one. The
immutable ripple-carry plan contained the correct target sequence, but an
ordinary pretrained-model logit occasionally exceeded the steered digit logit.
The failures were therefore latent decoding/interference failures rather than
parser, router, or arithmetic-transition failures.

### Prespecified pilot-only strength sweep

The same trained bridge weights were reevaluated without optimization changes:

| Strength | ID 1–4 | OOD 5–8 | OOD 9–12 | Carry chain | Preservation |
|---:|---:|---:|---:|---:|---:|
| 32 | 100.0% | 90.0% | 85.0% | 76.7% | 100.0% |
| 48 | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% |
| 64 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Decision: retain strength 64 as the candidate confirmatory value. Before
freezing it, retrain a fresh bridge at strength 64 and run the learned
low-rank-adapter control to establish an adequate and computationally fair
training duration.

## Learned-control pilots — 2026-07-25

All learned controls used the same pretrained model, prompt templates, and
answer-only teacher-forced objective but had no access to deterministic
arithmetic state.

| Control | Trainable params | Steps | ID 1–4 | OOD 5–8 | OOD 9–12 | Carry | Preservation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final-layer LoRA | 22,528 | 240 | 27.5% | 5.0% | 2.5% | 6.7% | not measured |
| Final-layer LoRA | 22,528 | 2,000 | 22.5% | 2.5% | 2.5% | 3.3% | 20.0% |
| All-layer LoRA | 270,336 | 1,000 | 80.0% | 27.5% | 0.0% | 3.3% | 10.0% |

The first final-layer attempt failed before optimization because newly attached
matrices were on CPU while the base was on MPS. The device placement was
corrected and the failure retained in the lab notebook.

The all-layer control is the candidate learned baseline. It has 25.1 times as
many trainable parameters as the firmware bridge and received substantially
more optimization. This favors the learned baseline in capacity and compute.
It learned much of the represented range but did not length-generalize and
changed most unrelated outputs.

## Pilot v2 — independently retrained strength 64

Configuration: `configs/phase2_pilot_v2.json`. The bridge was trained from a
fresh initialization using 1,000 arithmetic prompts, 400 routing negatives,
and 120 steps.

| Model | ID 1–4 | OOD 5–8 | OOD 9–12 | Carry chain |
|---|---:|---:|---:|---:|
| Frozen pretrained base | 11.7% | 5.0% | 1.7% | 10.0% |
| Trained bridge, firmware disabled | 11.7% | 5.0% | 1.7% | 10.0% |
| Latent firmware, strength 64 | 100.0% | 100.0% | 100.0% | 100.0% |
| Direct-logit firmware | 100.0% | 100.0% | 100.0% | 100.0% |

The latent model answered all 220 arithmetic examples correctly. It preserved
all 40 routing/language-control outputs token-for-token and made no initial
false route. Bridge training took 3.88 seconds after cache collection; the
complete four-mode sequence evaluation took 323.30 seconds.

Decision: strength 64, 120 bridge steps, 1,000 arithmetic training prompts, and
400 routing negatives are ready to freeze. One longer all-layer LoRA pilot will
select the final learned-control budget.

### Final learned-control budget decision

A fresh 3,000-example, 3,000-step all-layer run became unstable: its final loss
was 1.8447, ID accuracy fell to 65%, every extrapolation/carry score was zero,
and preservation fell to 6.7%. This negative result is retained.

Decision: freeze the 1,000-step all-layer LoRA control. It had the best pilot ID
accuracy (80%) among conventional controls, nonzero primary-OOD accuracy
(27.5%), and a substantial 25.1-fold learned-parameter advantage over the
bridge.
