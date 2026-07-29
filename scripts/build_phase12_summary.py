from __future__ import annotations

import json
import statistics
from pathlib import Path

CONFIRMATION = Path("phase12_results/confirmation.json")
ANALYSIS = Path("phase12_results/analysis.json")
OUTPUT = Path("PHASE12_EXECUTIVE_SUMMARY.md")
SEEDS = (16_201, 16_202, 16_203)
GATE_LABELS = {
    "autonomous_exactness": "Autonomous exactness",
    "paired_improvement": "Paired improvement",
    "route_recognition": "Route recognition",
    "preservation": "Preservation",
    "operand_access": "Operand access",
    "conditional_mechanism": "Conditional mechanism",
    "causal_routing": "Causal routing",
    "checkpoint_integrity": "Checkpoint integrity",
}


def render_summary(
    confirmation: dict[str, object],
    analysis: dict[str, object],
) -> str:
    metrics = confirmation["metrics"]
    candidate = metrics["phase12_candidate"]
    control = metrics["phase11_control"]
    gates = confirmation["gates"]
    passed = bool(gates["all_gates"])
    verdict = "Pass" if passed else "Fail"
    outcome = (
        "Phase 12 passes every frozen autonomous semantic-routing gate."
        if passed
        else "Phase 12 does not pass the complete frozen autonomous "
        "semantic-routing protocol."
    )
    candidate_exact = [
        int(candidate[str(seed)]["exact"]) for seed in SEEDS
    ]
    control_exact = [int(control[str(seed)]["exact"]) for seed in SEEDS]
    gains = [
        int(confirmation["paired_exact_gains"][str(seed)])
        for seed in SEEDS
    ]
    candidate_routes = [
        int(candidate[str(seed)]["positive_routes"]) for seed in SEEDS
    ]
    false_routes = [
        int(candidate[str(seed)]["false_routes"]) for seed in SEEDS
    ]
    preserved = [
        int(candidate[str(seed)]["token_preserved"]) for seed in SEEDS
    ]
    oracle = [
        int(candidate[str(seed)]["oracle_exact"]) for seed in SEEDS
    ]
    bootstrap = analysis["paired_phase12_vs_phase11"]["two_way_bootstrap"]
    paired = analysis["paired_phase12_vs_phase11"]
    failed_gates = [
        GATE_LABELS[key]
        for key in GATE_LABELS
        if not gates[key]
    ]
    category = analysis["category_summary"]
    positive_lines = []
    for key in (
        "positive_direct",
        "positive_word",
        "positive_distractor",
    ):
        values = category[key]
        denominator = int(values["prompts"]) * len(SEEDS)
        positive_lines.append(
            f"- {key.removeprefix('positive_').replace('_', ' ')}: "
            f"{values['phase12_candidate']['exact']}/{denominator} exact "
            f"(Phase 11: {values['phase11_control']['exact']}/{denominator});"
        )
    negative_lines = []
    for key, values in category.items():
        if values["positive"]:
            continue
        routes = int(values["phase12_candidate"]["routes"])
        if routes:
            denominator = int(values["prompts"]) * len(SEEDS)
            negative_lines.append(
                f"- {key.removeprefix('negative_').replace('_', ' ')}: "
                f"{routes}/{denominator} false routes;"
            )
    if not negative_lines:
        negative_lines = [
            "- no negative category produced a false route across the three "
            "seeds;"
        ]
    gate_rows = "\n".join(
        f"| {label} | {'Pass' if gates[key] else 'Fail'} |"
        for key, label in GATE_LABELS.items()
    )
    seed_lines = "\n".join(
        f"- seed {seed:,}: Phase 11 {control_exact[index]}/100, "
        f"Phase 12 {candidate_exact[index]}/100 "
        f"(gain {gains[index]:+d});"
        for index, seed in enumerate(SEEDS)
    )
    next_experiment = (
        "Replicate the frozen architecture on a second base model and extend "
        "the routed mechanism to at least one non-addition operation, with "
        "operation-conflict prompts and a newly sealed audit. This result "
        "should not be generalized beyond the sampled language distribution "
        "until those replications pass."
        if passed
        else "Treat Phase 12 as disclosed development evidence. Inspect the "
        "specific failed gates and false-route families, then revise the "
        "request representation—most likely with token-level, order-sensitive "
        "evidence—before freezing another audit. Do not retune the reported "
        "Phase 12 confirmation."
    )
    failure_sentence = (
        "No frozen gate failed."
        if passed
        else "Failed gates: " + ", ".join(failed_gates) + "."
    )
    return f"""# Phase 12 Executive Summary

## Outcome

{outcome}

The frozen four-view SiLU router produced:

{seed_lines}

Across 300 seed-prompt positive evaluations, Phase 12 was exact on
{paired['phase12_candidate_exact']}/300 versus
{paired['phase11_control_exact']}/300 for Phase 11. The mean paired gain was
{bootstrap['mean_paired_difference']:.3f}, with a two-way seed-and-prompt
bootstrap 95% interval of {bootstrap['percentile_95_lower']:.3f} to
{bootstrap['percentile_95_upper']:.3f}.

Positive routes were {candidate_routes[0]}/{candidate_routes[1]}/
{candidate_routes[2]} of 100. False routes were
{false_routes[0]}/{false_routes[1]}/{false_routes[2]} of 200, while
token-exact negative preservation was
{preserved[0]}/{preserved[1]}/{preserved[2]} of 200. Oracle-route exactness
was {oracle[0]}/{oracle[1]}/{oracle[2]} of 100.

{failure_sentence}

## Category behavior

Positive exactness across all three seeds:

{chr(10).join(positive_lines)}

Negative routing errors:

{chr(10).join(negative_lines)}

## Frozen verdict

| Gate | Result |
| --- | --- |
{gate_rows}
| Compound verdict | **{verdict}** |

## Plain-English interpretation

The router's only job is to decide whether the installed adder should run.
It reads four summaries of the model's own hidden prompt representation and
does not receive a parsed operation, keyword flag, operand value, external
classifier result, or correct route label. When it stays off, the original
model computation remains active.

This confirmation asks whether that autonomous decision generalized to new
wording families and operand pairs while preserving non-addition behavior.
The compound verdict above is computed directly from the gates frozen before
evaluation. Even a passing result would establish only reliable routing for
this TinyLlama four-digit-addition setup and sampled language distribution,
not general semantic understanding.

## Next experiment

{next_experiment}

## Reproducibility

- Frozen protocol: `PHASE12_MULTI_VIEW_ROUTING_PROTOCOL.md`
- Frozen manifest: `phase12_results/frozen_prompt_manifest.json`
- Raw confirmation: `phase12_results/confirmation.json`
- Statistical analysis: `phase12_results/analysis.json`
- Independent verification: `phase12_results/verification.json`
- Development record: `PHASE12_MULTI_VIEW_ROUTING_DEVELOPMENT.md`
"""


def main() -> None:
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    if confirmation["status"] != "phase12_confirmation_complete":
        raise ValueError("Phase 12 confirmation is incomplete")
    if analysis["status"] != "phase12_posthoc_analysis_complete":
        raise ValueError("Phase 12 analysis is incomplete")
    OUTPUT.write_text(render_summary(confirmation, analysis))
    print(OUTPUT.read_text())


if __name__ == "__main__":
    main()
