# Frozen Phase 2 Protocol: Pretrained-LLM Neural Firmware

Frozen on 2026-07-25 before confirmatory training or evaluation. Pilot
chronology and all tuning decisions are in `PHASE2_PILOT_LOG.md` and
`PHASE2_LAB_NOTEBOOK.md`. The machine-readable configuration is
`configs/phase2_study.json`.

## Research question

Can an immutable exact-addition process run inside the forward path of a
pretrained half-billion-parameter causal language model and communicate through
a lightweight learned residual interface, improving length-extrapolating exact
match while preserving behavior on ineligible prompts?

## Fixed model and integration

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`, 494,032,768 parameters.
- Immutable model revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- All pretrained parameters are frozen for the firmware condition.
- The deterministic module uses immutable decimal sum/carry transition tables
  and a fixed eligibility parser for three registered prompt templates.
- Its output alphabet is digits zero through nine plus end-of-answer.
- A learned bridge maps those eleven symbols to normalized vectors in the
  896-dimensional final residual state.
- A learned linear router gates the bridge.
- Injection occurs before the pretrained model's tied language-model head,
  during causal generation. It is not post-generation answer replacement.
- Bridge strength is fixed at 64 after pilot interference sweeps.
- Trainable bridge/router parameters: 10,753.

## Controls

1. **Frozen base:** unmodified pretrained model.
2. **Firmware off:** trained bridge present but injection disabled.
3. **Latent firmware:** learned residual bridge plus frozen arithmetic engine.
4. **Direct firmware:** correct firmware token receives an overwhelming direct
   logit boost, providing a structural upper bound.
5. **Learned adapter:** rank-four query/value LoRA residuals in all 24
   transformer blocks, with no access to deterministic arithmetic state.
   It has 270,336 trainable parameters, approximately 25.1 times the learned
   capacity of the firmware bridge.

## Training

Three confirmatory training seeds: 401, 503, and 601.

For each seed:

- 1,000 generated one- to four-digit addition prompts;
- 400 generated routing-negative prompts;
- bridge: 120 steps, batch 32, AdamW, learning rate 0.01;
- learned adapter: 1,000 steps, batch 1, AdamW, learning rate 0.0005.

Each seed receives independently generated training examples and therefore its
own frozen hidden cache. Every bridge and learned adapter is independently
initialized and trained.

## Confirmatory evaluation

Evaluation seed 771923 was not used during phase 2 pilot tuning.

Fixed evaluation sets:

- 300 in-distribution additions with one- to four-digit operands;
- 300 primary OOD additions with five- to eight-digit operands;
- 300 long OOD additions with nine- to twelve-digit operands;
- 150 constructed carry-chain additions with five- to twelve-digit operands;
- 100 routing/language-control prompts.

The primary endpoint is sequence-level exact-match accuracy on the five- to
eight-digit split. Generated answers are evaluated autoregressively, not with
teacher forcing.

## Success criteria

Phase 2 is successful only if all of the following hold:

1. Mean latent-firmware primary OOD accuracy is at least 99%.
2. Mean latent-firmware primary OOD accuracy exceeds the learned adapter by at
   least 50 percentage points.
3. Mean token-exact language preservation is at least 99%.
4. Mean initial false-route rate is no greater than 1%.

Firmware-off behavior, direct-firmware behavior, ID accuracy, long-OOD
accuracy, carry accuracy, latency, loss, and training time are secondary
outcomes and must be reported regardless of result.

## Statistical reporting

- Report every seed separately and the mean and sample standard deviation.
- Report exact counts as well as percentages.
- Bootstrap the three seed-level primary differences, while noting that three
  seeds are insufficient for a broad variability claim.
- Preserve every confirmatory error.
- Do not replace, tune, or rerun an unsuccessful seed under the same
  confirmatory label except to resume an interrupted deterministic run from a
  retained checkpoint.

## Claim boundary

Success supports an internal hybrid integration claim: a real pretrained
language model can decode immutable exact computation through a learned
residual interface. It does not establish:

- general mathematical correctness;
- open-ended natural-language parsing or operation selection;
- an algorithm compiled entirely into ordinary transformer weights;
- a deterministic unit inside the repeated transformer blocks;
- preservation beyond the registered control prompts;
- reliability outside the bounded tested operand lengths and grammar.

The more ambitious transformer-block architecture requested by Josiah is
recorded in `FUTURE_ARCHITECTURE_GOAL.md` and is explicitly outside this
protocol.
