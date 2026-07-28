# Canonical Program: Semi-Deterministic AI Through Deterministic Neuron Implants

Origin: Josiah Wilson, first recorded 2026-07-25 and materially clarified
2026-07-26. Program scope updated 2026-07-27.

This document supersedes interpretations of the idea as a tool call, an
output-adjacent calculator, or a separately invoked learned computational
pathway.

## Program direction

The overarching project is now **semi-deterministic AI**. The intended systems
combine:

- learned neural interpretation, representation, and flexible reasoning;
- compact deterministic mechanisms for transitions that should be exact;
- native activation interfaces through which the learned network can use those
  mechanisms as part of its own computation.

Arithmetic remains the controlled test bed, not the final application.
Board-game state transitions are no longer the designated next research phase.
The next work should first make the cross-model neural interface robust, then
move toward residual-native state and repeated deterministic invocation.
Broader scientific and engineering mechanisms can follow only after those
foundations survive new frozen audits.

Phase 9 showed that count-matched hard semantic contrasts alone did not make
the frozen TinyLlama linear interface robust. Phase 10 then found that a
rank-four transform local to the implant interface improved exactness over
linear by +9, +7, and +18 prompts across three frozen seeds while preserving
all 600 adversarial outputs. An exactly input-budget-matched 16-unit nonlinear
interface underperformed linear by 4–6 prompts per seed. Oracle routing still
reached 89–90% while natural exactness reached only 42–53%, so semantic routing
remains the immediate problem.

## Josiah's intended architecture

The calculator should literally occupy positions that otherwise behave as
ordinary neurons in a transformer MLP or expert. To the surrounding network,
its inputs and outputs are activation channels in the same internal tensor as
other neurons. Functionally, however, those selected activations are produced
by frozen deterministic computation rather than a learned scalar
approximation.

In a model trained from scratch, the deterministic neurons would exist from
initialization. Ordinary gradient descent would shape the rest of the network
so mathematical internal representations increasingly align with those
neurons, because routing correctly encoded operands through them reliably
reduces prediction loss.

For a pretrained-model retrofit:

1. Measure activation frequency and causal importance of Qwen MLP neurons.
2. Select a low-use, low-importance activation subspace with enough bandwidth.
3. Replace those ordinary activation slots with a frozen calculator-neuron
   bank.
4. Preserve the bank's normal neuron-shaped tensor interface.
5. Fine-tune incoming, outgoing, and surrounding Qwen weights so mathematical
   vectors are naturally encoded into the implant and its results become
   useful native internal representations.
6. Preserve ordinary capabilities with nonmathematical distillation and causal
   ablations.

A single scalar neuron is unlikely to carry two arbitrary operands and an
exact result. The faithful practical form is therefore a compact bank of
activation channels whose required width is treated as an empirical bandwidth
question. “Calculator neuron” remains the conceptual unit; “calculator-neuron
bank” is the implementation term.

The defining property is not merely that a deterministic module sits between
blocks. It is that the computation occupies neuron activation slots and is
learned around as part of the model's native representational substrate.

## Training hypothesis

The calculator remains frozen while the surrounding neural network remains
normally trainable. Early training may require register supervision, smooth
surrogates, or straight-through gradients so the network can discover the
implant's typed interface. Those scaffolds should later be reduced so final
behavior is driven primarily by normal language-model or task loss.

The central empirical question is whether deep learning will reorganize
mathematical representations around a small deterministic subspace because
that subspace is consistently more reliable than approximate learned
arithmetic.

## Reuse and repeated computation

The same calculator-neuron bank can appear at one or more recurrent depths or
autoregressive reasoning steps. Repeated calculations should arise from
ordinary mathematical states passing through the same neuron-shaped
computation again, not from explicit calculator-call tokens or an external
tool protocol.

## General deterministic-neuron thesis

Arithmetic is the first controlled test, not the full scope. The same
architectural style could implant domain-specific deterministic computation
with an activation interface sized for the necessary information bandwidth.

Candidate domains include:

- exact arithmetic and symbolic transforms;
- formal logic, constraints, and state transitions;
- units, geometry, and physical invariants;
- molecular chemistry calculations and chemically valid transformations;
- other scientific or engineering computations where a trusted algorithm
  should become part of a model's internal intelligence rather than an
  externally orchestrated tool call.

These are long-run examples, not a committed sequence of immediate phases.

Mixture-of-experts models are an especially natural target. A deterministic
expert or deterministic neuron bank could share the same routing and
activation conventions as learned experts while supplying exact,
domain-specific transformations. The research question is whether models
learn more fluid, reliable use when deterministic programs inhabit their
native activation space.

## Boundaries

- This is a research hypothesis, not yet a demonstrated general architecture.
- Correct deterministic execution does not guarantee correct operand
  representation, routing, decomposition, or scientific interpretation.
- “Optimal bandwidth” must be measured rather than assumed.
- Related work and prior art must be reviewed before claiming novelty.
- The Phase 6 residual-to-register prototype provides useful components, but
  it does not instantiate this canonical architecture because its learned
  controller invokes a separate internal program and forces output-adjacent
  result residuals.
- Phase 8 transferred exact conditional computation and causal use to
  TinyLlama, but failed its routing, operand, and overall accuracy gates.
  Phase 9 then improved operand recovery and several adversarial families while
  reducing end-to-end accuracy and underperforming generic continuation on
  false routing. Semi-deterministic computation is therefore not yet robust
  across arbitrary language.

## Completed first definitive experiment

1. Use Qwen2.5-0.5B and addition only.
2. Census MLP neuron activation and causal importance on mathematical and
   nonmathematical corpora.
3. Reserve several candidate subspace widths.
4. Implant the same frozen typed addition circuit into those activation slots.
5. Fine-tune surrounding weights under mixed arithmetic and preservation loss.
6. Evaluate unseen operand lengths, natural-language generalization,
   nonmathematical preservation, and activation selectivity.
7. Ablate or zero only the implanted neurons. Arithmetic performance should
   collapse selectively if Qwen truly reorganized around them.
8. Compare replacement, added-neuron, and MoE-expert variants at matched
   bandwidth and learned-parameter budgets.

Phase 7 instantiated the in-place Qwen version. Phase 8 replicated the
mechanism on TinyLlama with an exactly parameter-matched rank-14 adapter, but
the frozen compound protocol failed on semantic interface generalization.
Phase 9 kept the calculator and output decoder fixed while comparing generic
and hard-contrast continuation across three seeds. The hard condition improved
exact operands from 73--74/100 to 81--85/100 and reduced Phase 8 false routes
from 26--30/200 to 11--12/200, but routed off 24/100 additions per seed and
averaged only 60\% exact, equal to generic continuation and below frozen Phase
8. All protocol-defined valid executions remained exact and all correct hard
answers disappeared under result ablation.

Phase 10 narrowed the diagnosis: limited interface-local representation
adaptation helps, but simple matched nonlinear capacity does not, and neither
repairs route recall. The next controlled work should isolate routing from
operand typing architecturally—potentially with a dedicated semantic router,
another layer or pooled representation, and preservation-constrained training—
before adding operations or recurrent state.

## Success criterion

Qwen learns to encode supported mathematical subproblems into the implanted
activation slots, the frozen neuron bank computes exact results, downstream
ordinary layers use those results, and selectively removing the implanted
neurons causes a large arithmetic-specific loss without broad capability
damage.
