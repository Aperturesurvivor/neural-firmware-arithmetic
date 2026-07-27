# Phase 7 Three-Seed Result

## Plain-language answer

The central mechanism is working in this narrow addition experiment. Qwen
learned to place the two operands into a tiny typed activation interface, a
fixed zero-parameter calculator produced the exact sum, and Qwen's ordinary
downstream layers converted that internal result into output digits.

This was not just one lucky training run. Three independently initialized
interfaces all scored 58/60 on the same fresh additions. Whenever the learned
interface supplied the intended operands, the deterministic result and final
answer were exact in all 174/174 seed-example executions.

The prototype is not yet safe to activate on arbitrary prompts. On sixty
deliberately misleading non-addition prompts, the three routers activated
incorrectly five or six times each. Those activations changed the generated
text, so the predeclared preservation gate failed.

## Frozen result

| Metric per seed | Seed 13,201 | Seed 13,202 | Seed 13,203 |
|---|---:|---:|---:|
| Exact additions | 58/60 | 58/60 | 58/60 |
| Direct additions | 30/30 | 30/30 | 30/30 |
| Word problems | 28/30 | 28/30 | 28/30 |
| Exact operands | 58/60 | 58/60 | 58/60 |
| Exact after result ablation | 3/60 | 3/60 | 3/60 |
| Paired causal drop | 55/60 | 55/60 | 55/60 |
| Conditional exact execution | 58/58 | 58/58 | 58/58 |
| False routes on negatives | 6/60 | 5/60 | 6/60 |
| Token-exact negative preservation | 54/60 | 55/60 | 54/60 |

Four of the five frozen engineering gates passed. The compound protocol failed
only the adversarial routing/preservation gate.

## What this establishes

- Twenty-eight coordinates inside Qwen's existing 4,864-wide layer-16 MLP can
  serve as a typed deterministic-neuron bank without increasing model width.
- The learned interface uses 25,088 weights; the calculator uses zero learned
  weights.
- The effect is causal: removing only the calculator-result activations
  reduced accuracy from 58/60 to 3/60 in every seed.
- Exact addition is no longer the observed bottleneck. Operand framing caused
  the two positive failures, and intent routing caused the preservation
  failures.

## What it does not establish

- It does not yet provide robust routing across arbitrary language.
- The first-response route latch and result-position counter still live in
  generation-runtime state rather than recurrent residual registers.
- It currently supports one addition per response, not unlimited calculator
  use during arbitrary internal reasoning.
- The interface was explicitly supervised; the experiment does not show that
  ordinary pretraining would discover the calculator ABI unaided.

The frozen protocol is `PHASE7_MULTISEED_PROTOCOL.md`. Raw results and the
machine-generated aggregation are in `phase7_results/`.
