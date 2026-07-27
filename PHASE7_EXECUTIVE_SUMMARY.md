# Phase 7 Executive Summary

## Result in one sentence

Qwen learned to route natural-language additions through a zero-parameter
calculator occupying 28 existing MLP coordinates, and the calculator was
causally necessary and exact when correctly framed, but rare learned
input/output interface errors prevented a clean compound protocol pass.

## What was built

The experiment modifies frozen Qwen2.5-0.5B-Instruct at decoder layer 16.
Twenty-eight of the layer's existing 4,864 MLP activation coordinates serve as
a typed calculator ABI:

- 16 learned route, operand-role, and digit coordinates;
- 12 deterministic result-symbol coordinates.

The learned interface contains 25,088 weights, about 0.0051% of Qwen's
494-million parameters. The decimal ripple-carry calculator has zero learned
parameters. Model width does not increase.

The current runtime latches the first route, captures the first valid operands
in a deterministic register, and advances a result-position counter. These
state elements are calculator microcircuit scaffolding outside Qwen's recurrent
residual activations.

## Strongest frozen evidence

The final operand-register audit used three independently learned interfaces,
60 unseen additions and 60 unseen adversarial negatives per seed, and prompt
families disjoint from all prior development.

| Outcome | Result |
|---|---:|
| Exact additions | 173/180 (96.11%) |
| Exact first-step operands | 174/180 (96.67%) |
| Exact calculator trajectories | 174/180 (96.67%) |
| Exact word problems | 90/90 |
| Exact after result ablation | 9/180 (5.00%) |
| Paired causal losses | 165/180 |
| Stable registers given exact inputs | 174/174 |
| False routes on adversarial negatives | 0/180 |
| Token-exact negative preservation | 180/180 |

Two direct-prompt instances caused all six operand-framing failures across the
three seeds. One additional seed/example had exact registered operands and
exact calculator symbols for `0 + 0`, but the learned result decoder and
downstream Qwen emitted `I`.

The frozen protocol passed operand recovery, causal ablation, and
routing/preservation. It failed the mean exact-accuracy gate by one aggregate
answer (57.67/60 versus 58/60) and failed the all-exact conditional output gate
because of the single downstream decoding error.

## What this means

The narrow core hypothesis is strongly supported:

- learned Qwen activations can drive a deterministic neuron-shaped arithmetic
  subspace inside an existing transformer MLP;
- Qwen's normal downstream computation can read and use its output;
- causal ablation shows the model is not merely answering independently;
- deterministic addition and registered state produced no observed wrong
  trajectory from exact typed inputs.

It is not yet a finished general neural calculator:

- semantic operand framing is not perfect;
- result decoding is not mathematically guaranteed;
- persistent state is runtime-managed rather than residual-native;
- only one addition call is supported per response;
- arbitrary multi-call chain-of-thought, other operations, and another model
  family remain future work.

The complete chronological record is `PHASE7_PILOT_LOG.md`. Frozen protocols,
raw per-token traces, hashes, unsuccessful runs, and machine-generated analyses
are retained in the repository.

## Retrospective untouched-base comparison

After Audit 5 was complete, untouched Qwen was run on the identical 60
addition prompts with the identical eight-token greedy budget. It returned
exactly the requested numeral on 1/60 prompts, compared with 57/60, 58/60, and
58/60 for the three implant seeds. A generous 64-token base-only sensitivity
run recovered the correct sum as the final integer on 27/60 prompts, but still
followed the requested numeral-only format on only 1/60.

On the 60 adversarial negatives, all three implants remained token-identical
to untouched Qwen. This comparison is retrospective rather than preregistered;
the prompt set and implant outputs were already frozen before the base
benchmark was generated. The complete problem-by-problem comparison is in
`phase7_base_comparison/report.html`.
