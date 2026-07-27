from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analyze_phase7_multiseed import (
    aggregate_metric,
    canonical_examples,
    metric_values,
    seed_summary,
)

RESULTS = tuple(
    Path(f"phase7_results/operand_register_audit5_seed_{seed}.json")
    for seed in (13201, 13202, 13203)
)
OUTPUT = Path("phase7_results/operand_register_audit5_summary.json")
PROTOCOL = "PHASE7_OPERAND_REGISTER_PROTOCOL.md"


def failure_record(row: dict[str, object]) -> dict[str, object]:
    first = row["implant"]["steps"][0]
    return {
        "prompt": row["prompt"],
        "expected": row["answer"],
        "generated": row["implant"]["generated_text"],
        "first_step_operands_exact": row["first_step_operands_exact"],
        "first_step_a_digits": first["a_digits"],
        "first_step_b_digits": first["b_digits"],
        "calculator_symbols": row["calculator_symbols"],
        "calculator_trajectory_exact": row["calculator_trajectory_exact"],
        "register_active_by_step": [
            step["operand_register_active"] for step in row["implant"]["steps"]
        ],
    }


def register_stable(row: dict[str, object]) -> bool:
    steps = row["implant"]["steps"]
    if not steps or not all(step["operand_register_active"] for step in steps):
        return False
    first_a = steps[0]["a_digits"]
    first_b = steps[0]["b_digits"]
    first_a_lengths = steps[0]["a_lengths"]
    first_b_lengths = steps[0]["b_lengths"]
    return all(
        step["a_digits"] == first_a
        and step["b_digits"] == first_b
        and step["a_lengths"] == first_a_lengths
        and step["b_lengths"] == first_b_lengths
        for step in steps
    )


def main() -> None:
    loaded = [(path, json.loads(path.read_text())) for path in RESULTS]
    if any(result.get("dataset") != "audit5" for _, result in loaded):
        raise ValueError("operand-register analysis requires audit-5 records")
    if any(result.get("latch_operands") is not True for _, result in loaded):
        raise ValueError("audit-5 records must enable the operand register")
    example_payloads = [canonical_examples(result["rows"]) for _, result in loaded]
    if len(set(example_payloads)) != 1:
        raise ValueError("audit-5 seed results are not prompt aligned")
    summaries = [seed_summary(result, path) for path, result in loaded]
    for summary, (_, result) in zip(summaries, loaded, strict=True):
        addition = summary["addition"]
        negative = summary["negative"]
        conditional = summary["conditional_on_route_and_operands"]
        positives = [row for row in result["rows"] if row["route_label"] is True]
        exact_operand_rows = [
            row for row in positives if row["first_step_operands_exact"] is True
        ]
        summary["register"] = {
            "activated": sum(
                all(
                    step["operand_register_active"]
                    for step in row["implant"]["steps"]
                )
                for row in positives
            ),
            "stable_on_exact_first_step_operands": sum(
                register_stable(row) for row in exact_operand_rows
            ),
            "eligible_exact_first_step_operands": len(exact_operand_rows),
        }
        summary["gates"] = {
            "addition_exact_at_least_57": addition["mathematical_exact"] >= 57,
            "operands_exact_at_least_57": addition["operands_exact"] >= 57,
            "paired_ablation_drop_at_least_50": (
                addition["paired_ablation_drop"] >= 50
            ),
            "false_routes_at_most_2": negative["false_routes"] <= 2,
            "negative_preserved_at_least_58": negative["token_preserved"] >= 58,
            "conditional_execution_all_exact": (
                conditional["examples"]
                == conditional["mathematical_exact"]
                == conditional["format_exact"]
                == conditional["trajectory_exact"]
            ),
        }
        summary["addition_failures"] = [
            failure_record(row)
            for row in positives
            if row["mathematical_correct"] is False
        ]
    metrics = {
        section: {
            metric: aggregate_metric(metric_values(summaries, section, metric))
            for metric in metric_names
        }
        for section, metric_names in {
            "addition": (
                "mathematical_exact",
                "format_exact",
                "direct_exact",
                "word_exact",
                "route_active",
                "operands_exact",
                "trajectory_exact",
                "ablation_exact",
                "paired_ablation_drop",
            ),
            "negative": ("false_routes", "token_preserved"),
            "register": (
                "activated",
                "stable_on_exact_first_step_operands",
                "eligible_exact_first_step_operands",
            ),
        }.items()
    }
    gates = {
        "1_addition_accuracy": (
            all(
                summary["gates"]["addition_exact_at_least_57"]
                for summary in summaries
            )
            and metrics["addition"]["mathematical_exact"]["mean"] >= 58
        ),
        "2_operand_recovery": all(
            summary["gates"]["operands_exact_at_least_57"]
            for summary in summaries
        ),
        "3_causal_ablation": all(
            summary["gates"]["paired_ablation_drop_at_least_50"]
            for summary in summaries
        ),
        "4_negative_routing_and_preservation": all(
            summary["gates"]["false_routes_at_most_2"]
            and summary["gates"]["negative_preserved_at_least_58"]
            for summary in summaries
        ),
        "5_conditional_registered_execution": all(
            summary["gates"]["conditional_execution_all_exact"]
            for summary in summaries
        ),
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "shared_holdout_sha256": hashlib.sha256(
            example_payloads[0].encode()
        ).hexdigest(),
        "per_seed": summaries,
        "aggregate_mean_min_max": metrics,
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "passed_all_gates": all(gates.values()),
        "cross_seed": {
            "total_false_routes": sum(
                summary["negative"]["false_routes"] for summary in summaries
            ),
            "total_negative_prompts": sum(
                summary["negative"]["examples"] for summary in summaries
            ),
            "total_preserved_negatives": sum(
                summary["negative"]["token_preserved"] for summary in summaries
            ),
            "stable_registers_on_exact_inputs": sum(
                summary["register"]["stable_on_exact_first_step_operands"]
                for summary in summaries
            ),
            "exact_input_executions": sum(
                summary["register"]["eligible_exact_first_step_operands"]
                for summary in summaries
            ),
        },
        "interpretation": (
            "The operand register eliminated later state drift and routing/"
            "preservation remained exact. The compound protocol failed its "
            "mean-accuracy gate by one aggregate correct answer and failed "
            "conditional output because one seed had exact registered "
            "calculator symbols but an incorrect learned result-decoder token."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
