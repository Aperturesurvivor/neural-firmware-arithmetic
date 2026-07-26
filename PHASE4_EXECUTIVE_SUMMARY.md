# Phase 4 Executive Summary

## Plain-English outcome

Yes: the experiment showed that a small language model which was poor at
multi-digit arithmetic became dramatically better when an exact arithmetic
circuit was installed inside its inference path.

On 400 natural-language addition prompts that were never used during
development:

| Condition | Correct | Accuracy |
|---|---:|---:|
| Untouched Qwen2.5-0.5B-Instruct | 68/400 | 17.0% |
| Equal-parameter ordinary learned adapter | 21/400 | 5.25% |
| Internal deterministic architecture | 360/400 | 90.0% |
| Internal architecture with the route forced on | 400/400 | 100.0% |

The internal model gained 73 percentage points over the untouched model and
84.75 points over the equal-parameter learned control. On the paired prompts,
the internal model fixed 292 base-model errors while introducing zero errors
on prompts the base had answered correctly.

## What actually worked

The deterministic arithmetic mechanism was completely reliable when used:

- It activated automatically on 349 of the 400 addition prompts.
- All 349 activated answers were exactly correct.
- When deliberately activated on every prompt, it scored 400/400.
- It made no observed carrying, digit-length, word-problem, or decoding error.

The original model was especially weak where the new structure mattered most:
3% on five-to-eight-digit simple sums, 0% on nine-to-twelve-digit sums, and 6%
on five-to-eight-digit word problems. The learned-router internal model scored
88%, 79%, and 100% on those same splits.

The equal-parameter control is important. Both modified models had exactly
24,225 learned parameters and used the same router, prompts, model, and
insertion point. The ordinary learned adapter remained at 5.25% overall,
showing that the gain did not come merely from adding a small number of
parameters.

## What remains unfinished

The arithmetic circuit worked; the semantic router did not recognize every
addition wording.

The frozen protocol required the router to activate on at least 90% of
positive prompts. It activated on 349/400, or 87.25%. This missed the threshold
by 2.75 percentage points. Therefore five of the six preregistered criteria
passed, but the experiment's compound formal verdict is not a full pass.

The safety side was stronger:

- zero false activations on 160 unseen non-addition prompts;
- 160/160 negative outputs were token-identical to the untouched model;
- with the circuit forced off, all 400 positive outputs were token-identical
  to the untouched model.

In practical terms, the router was conservative: it sometimes failed to use a
calculator that would have been correct, but it did not incorrectly overwrite
any tested subtraction, multiplication, comparison, quotation, negation,
concatenation, or explanation response.

## What this proves—and what it does not

The experiment supports this narrow claim:

> A deterministic typed arithmetic process can run inside a frozen pretrained
> transformer and produce a large, causally localized skill increase that an
> equal-sized ordinary learned adapter does not reproduce.

This is not the first integrated calculator inside an LLM. Dietz and Klakow's
2025 Integrated Gated Calculator (IGC) previously combined a frozen Llama 3.1
8B model with learned input/output mappings, a gate, and a
non-differentiable calculator, reporting 98–99% across four arithmetic
operations. Our result is best understood as an independent small-model
replication and controlled extension: it uses a 0.5B model, only 24,225
learned interface parameters, an exactly parameter-matched learned control,
precommitted same-prompt evaluation, novel wording families, and causal
on/off tests. The studies are not head-to-head, so their accuracy percentages
are not directly comparable.

It does not yet show:

- general mathematical understanding;
- a calculator compressed into one ordinary scalar neuron;
- learned extraction of arbitrary numbers from text;
- reliable handling of signs, decimals, fractions, more than two operands, or
  multiple operations;
- replication across multiple model families and training seeds.

A fixed boundary converted exactly two contiguous decimal strings into typed
digits. The model learned whether the prompt requested addition, and a frozen
ripple-carry module computed the sum. The result then returned to the model's
residual stream before its normal final normalization and output head.

## Recommended next step

The next frozen study should compare this architecture directly with an
IGC-style baseline on the same small model and identical prompts. It should
use:

- at least three independently trained interface seeds;
- Qwen 0.5B first, then a Llama-class 1–1.5B model;
- fixed learned-parameter budgets;
- addition, subtraction, multiplication, and division where supported;
- the same untouched, matched-control, oracle, forced-off, and safety tests.

The next architectural milestone is replacing the fixed decimal parser with a
learned residual-to-register encoder so that both operand extraction and
operation selection occur inside the model. That comparison is needed before
claiming an efficiency or architectural advantage over existing integrated
calculator work.

## Files

- Full paper: `paper_phase4/natural-language-deterministic-arithmetic.pdf`
- Frozen protocol: `PHASE4_CONFIRMATORY_PROTOCOL.md`
- Chronological notebook: `PHASE4_LAB_NOTEBOOK.md`
- Raw per-prompt archive: `phase4_results/confirmation_raw.json`
- Statistical analysis: `phase4_results/confirmation_analysis.json`
- Reproduction runner: `scripts/run_phase4_confirmation.py`
