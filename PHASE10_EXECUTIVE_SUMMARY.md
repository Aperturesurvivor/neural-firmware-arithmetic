# Phase 10 Executive Summary

## Result

Phase 10 confirms that a small interface-local representation adapter improves
TinyLlama's learned access to the existing deterministic addition implant
under this controlled protocol. It also rejects the tested hypothesis that a
parameter-matched 16-unit nonlinear bottleneck is better than the linear
interface.

The representation hypothesis passed all frozen gates. The broader interface
remains far from reliable: natural end-to-end exactness reached only 42% to
53%, while oracle routing reached 89% to 90%. Routing is still the dominant
remaining bottleneck.

## Frozen design

All conditions kept TinyLlama, decoder layer 15, the 28 implanted activation
coordinates, the zero-parameter deterministic calculator, the 24,576 learned
result-decoder weights, output strength, and runtime fixed.

Three paired seeds compared:

- a 32,768-weight linear route/role/digit interface;
- an exactly input-budget-matched nonlinear interface with a 16-unit SiLU
  bottleneck;
- the linear interface plus a 16,384-weight rank-four nonlinear residual
  adapter local to the implant interface.

The local adapter changes only what the implant interface reads. It does not
change the ordinary MLP's hidden input, which preserves route-off base
computation by construction.

The sealed audit contained 100 new additions and 200 new adversarial negatives.
All prompts, families, and operand pairs were disjoint from the prior generated
splits as specified in the frozen protocol.

## Primary results

| Condition | Seed 16,201 | Seed 16,202 | Seed 16,203 | Mean |
|---|---:|---:|---:|---:|
| Linear exact | 37/100 | 35/100 | 35/100 | 35.7% |
| Matched nonlinear exact | 31/100 | 30/100 | 31/100 | 30.7% |
| Linear + local representation exact | 46/100 | 42/100 | 53/100 | 47.0% |
| Representation oracle-route exact | 89/100 | 90/100 | 89/100 | 89.3% |
| Representation false routes | 0/200 | 0/200 | 0/200 | 0 |
| Representation token preservation | 200/200 | 200/200 | 200/200 | 100% |
| Representation ablation exact | 0/100 | 0/100 | 0/100 | 0% |

The representation condition's paired gains over linear were +9, +7, and +18
exact prompts. Its mean paired gain was +11.33 percentage points. A post-hoc
two-way seed/prompt bootstrap gave a 95% percentile interval from +4.0 to
+20.0 percentage points.

The parameter-matched nonlinear condition changed exactness by -6, -5, and -4
prompts relative to linear. Its mean paired change was -5.0 points, with a
post-hoc two-way bootstrap interval from -10.33 to -0.33 points.

## What improved

The adapted condition's gains were concentrated in semantically richer
prompts:

| Positive family | Linear | Matched nonlinear | Local representation |
|---|---:|---:|---:|
| Direct | 48/150 | 45/150 | 52/150 |
| Word problems | 30/75 | 30/75 | 44/75 |
| Irrelevant-number distractors | 29/75 | 17/75 | 45/75 |

This pattern supports the representation-access interpretation more strongly
than a generic arithmetic-capacity interpretation. The adapter mainly improved
when the model had to expose semantic roles and ignore irrelevant context.

## Mechanism

Every active-route, exact-operand example in every condition had an exact
calculator trajectory and exact decoded answer. For the selected condition:

- oracle routing produced 89, 90, and 89 exact answers;
- natural routing produced only 46, 42, and 53;
- result ablation removed every normally correct answer;
- no adversarial negative activated the implant;
- every negative output matched untouched TinyLlama token for token.

The calculator and result decoder are not the observed bottleneck. Operand
typing is imperfect but relatively strong. Semantic route recall remains the
largest gap.

## Claim boundary

Phase 10 supports a narrow claim: a rank-four learned transform local to this
implant interface improved access to frozen TinyLlama representations across
three new paired seeds while preserving route-off behavior.

It does not establish arbitrary-language robustness, adaptation of TinyLlama
itself, general mathematical reliability, superiority of all representation
adapters, or failure of every nonlinear interface. The tested nonlinear result
applies to one exactly matched 16-unit architecture and schedule.

See the frozen
[`Phase 10 protocol`](PHASE10_INTERFACE_CAPACITY_PROTOCOL.md), complete
[`Phase 10 lab notebook`](PHASE10_LAB_NOTEBOOK.md), raw
[`confirmation metrics`](phase10_results/confirmation.json), post-hoc
[`analysis`](phase10_results/analysis.json), and independent
[`completion audit`](phase10_results/completion_audit.json).
