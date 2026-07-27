# Phase 8 Second-Model Replication Protocol Draft

Status: draft. Do not treat any gate below as frozen until the model revision,
channel census, prompt families, training budget, and implementation commit are
recorded before confirmatory generation.

## Purpose

Test whether the Phase 7 deterministic-neuron result transfers from
Qwen2.5-0.5B-Instruct to an independently pretrained Llama-family decoder
without changing the scientific claim after seeing confirmatory outcomes.

This is a replication of the single-addition mechanism, not the multi-call
controller experiment. Keeping those questions separate prevents a failed
controller from obscuring the cross-model result.

## Model choice

Primary practical target:
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, pinned to an exact Hugging Face revision.

Reasons:

- it uses the Llama 2 architecture and tokenizer;
- it is approximately 1.1B parameters, meaningfully different from Qwen while
  remaining feasible on the available 16 GB Apple M4 machine;
- it is publicly downloadable without Meta's gated-license acceptance;
- its MLP exposes a compatible intermediate activation in which an in-place
  deterministic subspace can be assigned.

Preferred later branded replication:
`meta-llama/Llama-3.2-1B-Instruct`, after the human account holder accepts the
Llama 3.2 license and Hugging Face access terms. Do not let that gated step
delay the independent architecture replication.

## Frozen research question

Can a compact learned interface make a frozen Llama-family model:

1. identify addition requests in varied natural language;
2. encode the intended operands into reserved MLP activation coordinates;
3. consume exact symbols from a zero-learned-parameter adder;
4. decode the result through the model's ordinary downstream layers;
5. preserve untouched-base outputs when the route is off?

## Conditions

Run all conditions on identical frozen prompts:

1. untouched base model;
2. ordinary learned adapter with the same learned-parameter count as the
   implant;
3. deterministic-neuron implant;
4. implant with calculator-result coordinates ablated.

Do not add a fixed parser to the implant. The input interface must be neural.
The response-local route latch, operand register, and result-position counter
may be reused exactly as architectural microcircuit state and must be disclosed
as runtime-managed state.

## Architecture constraints

- Freeze every pretrained model parameter.
- Select one middle-to-late decoder MLP layer only after a development-only
  layer probe.
- Reserve 28 existing intermediate coordinates unless a tokenizer or
  architecture-specific incompatibility is documented before confirmation:
  2 route, 3 role, 11 digit/non-digit, and 12 result-symbol coordinates.
- Preserve the model's hidden and MLP widths.
- Use a frozen decimal ripple-carry adder with zero learned parameters.
- Train only the selected route/role/digit rows and result down-projection
  columns.
- Match the learned control parameter count exactly, not approximately.
- Record both absolute learned weights and the percentage of base parameters.

Because TinyLlama's hidden width differs from Qwen's, the native 28-coordinate
interface is expected to contain `28 * hidden_size` learned weights. The
matched adapter must use exactly that count. Cross-model comparisons must
report both raw count and percentage rather than forcing Qwen's 25,088-weight
budget onto a different hidden width.

## Data isolation

Create deterministic generators with committed seeds for:

- training positives;
- training negatives;
- development positives and negatives;
- final confirmatory positives and negatives;
- untouched-base sensitivity evaluation.

All final family strings must be exact-string disjoint from every Phase 7 and
Phase 8 training/development family. All numeric operand pairs must also be
disjoint across Phase 8 training, development, and confirmation.

The three training seeds receive identical examples in identical order.

## Confirmatory sample

Per seed:

- 30 direct addition prompts;
- 30 addition word problems;
- 60 adversarial non-addition prompts, balanced across:
  quoted arithmetic, negated requests, multiplication near-misses, factual
  questions containing numbers, and instructions to ignore an embedded sum.

Use one-to-four-digit nonnegative operands to preserve comparability with Phase
7. Add a preregistered longer-number sensitivity set only as secondary
analysis.

## Training seeds

Use three new seeds selected before training and unrelated to Phase 7 seeds.
Record initialization, data order, optimizer, scheduler, batch size, update
count, threshold tuning, and every development checkpoint retained or rejected.

## Primary outcomes

Report per seed and across seeds:

- exact numeral-only addition accuracy;
- direct and word-problem accuracy;
- route recall on positives;
- exact first-step operand recovery;
- exact calculator digit-plus-end trajectory;
- exact output conditional on exact operands and route;
- result-ablation accuracy and paired causal loss;
- false routes on every negative family;
- token-exact preservation relative to untouched base;
- untouched-base exact response under the identical token budget;
- ordinary matched-adapter exact response;
- latency, peak memory, and learned-parameter count.

Prompt rows are repeated measurements, not independent model replications.
Report seed-level outcomes and a paired prompt-by-seed bootstrap only as a
secondary interval.

## Proposed confirmatory gates

Freeze or revise these thresholds before confirmatory data generation:

1. every implant seed reaches at least 57/60 exact additions and the mean
   reaches at least 58/60;
2. every seed recovers exact operands on at least 57/60 additions;
3. every seed loses at least 50/60 paired correct answers under result
   ablation;
4. every seed has at most 2/60 negative false routes and at least 58/60
   token-exact negative preservation;
5. every active-route example with exact registered operands has an exact
   calculator trajectory, and at least 59/60 such examples decode exactly;
6. the implant exceeds both untouched base and the matched learned adapter in
   exact response accuracy for every seed;
7. no unreported checkpoint replacement, prompt-family revision, threshold
   change, or post-confirmation retraining occurs.

The conditional-output gate is set to 59/60 rather than logical perfection
because Phase 7 showed one downstream decoding error despite an exact
trajectory. Report any such error individually.

## Development sequence

1. Pin and hash the model/tokenizer revision.
2. Run an untouched-base prompt and tokenizer audit.
3. Census low-impact MLP coordinates using non-arithmetic controls.
4. Probe candidate layers using development data only.
5. Port the implant with unit tests for exact calculator semantics,
   coordinate replacement, latching, operand stability, and ablation.
6. Train one pilot seed; retain every configuration and outcome.
7. Harden route behavior only on development negatives.
8. Select the final architecture and thresholds.
9. Freeze this protocol, code commit, prompt generators, and checkpoint hashes.
10. Train or select three confirmation checkpoints without examining final
    prompt outcomes.
11. Generate the final set once, then run base, matched adapter, implant, and
    ablation on the identical prompts.

## Claim boundary

A pass establishes cross-model transfer of a supervised, single-addition,
runtime-registered deterministic MLP subspace. It does not establish:

- arbitrary recurrent calculator calls;
- spontaneous discovery from task loss alone;
- residual-native persistent state;
- subtraction, multiplication, or division;
- universal prompt safety;
- transfer to large production models.

All failed pilots, failed frozen gates, and post-hoc analyses remain in the
record.

## Provenance

The deterministic-neuron hypothesis and research direction originate with
Josiah Wilson. Experimental design, implementation, execution, analysis, and
manuscript assistance involve OpenAI Codex under his direction.
