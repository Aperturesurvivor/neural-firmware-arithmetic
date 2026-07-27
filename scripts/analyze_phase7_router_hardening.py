from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

from analyze_phase7_multiseed import (
    aggregate_metric,
    canonical_examples,
    metric_values,
    seed_summary,
)

RESULTS = tuple(
    Path(f"phase7_results/router_hardened_audit4_seed_{seed}.json")
    for seed in (13201, 13202, 13203)
)
OUTPUT = Path("phase7_results/router_hardened_audit4_summary.json")
PROTOCOL = "PHASE7_ROUTER_HARDENING_PROTOCOL.md"


def failure_record(row: dict[str, object]) -> dict[str, object]:
    first = row["implant"]["steps"][0]
    return {
        "prompt": row["prompt"],
        "expected": row["answer"],
        "generated": row["implant"]["generated_text"],
        "first_step_route_active": row["first_step_route_active"],
        "first_step_operands_exact": row["first_step_operands_exact"],
        "first_step_a_digits": first["a_digits"],
        "first_step_b_digits": first["b_digits"],
        "calculator_symbols": row["calculator_symbols"],
        "calculator_trajectory_exact": row["calculator_trajectory_exact"],
    }


def main() -> None:
    loaded = [(path, json.loads(path.read_text())) for path in RESULTS]
    if any(result.get("dataset") != "audit4" for _, result in loaded):
        raise ValueError("router-hardening analysis requires audit-4 records")
    example_payloads = [canonical_examples(result["rows"]) for _, result in loaded]
    if len(set(example_payloads)) != 1:
        raise ValueError("audit-4 seed results are not prompt aligned")
    summaries = [seed_summary(result, path) for path, result in loaded]
    for summary, (_, result) in zip(summaries, loaded, strict=True):
        addition = summary["addition"]
        negative = summary["negative"]
        conditional = summary["conditional_on_route_and_operands"]
        summary["gates"] = {
            "addition_exact_at_least_54": addition["mathematical_exact"] >= 54,
            "operands_exact_at_least_57": addition["operands_exact"] >= 57,
            "paired_ablation_drop_at_least_48": (
                addition["paired_ablation_drop"] >= 48
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
            for row in result["rows"]
            if row["route_label"] is True and row["mathematical_correct"] is False
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
        }.items()
    }
    gates = {
        "1_addition_accuracy": (
            all(
                summary["gates"]["addition_exact_at_least_54"]
                for summary in summaries
            )
            and metrics["addition"]["mathematical_exact"]["mean"] >= 57
        ),
        "2_operand_recovery": all(
            summary["gates"]["operands_exact_at_least_57"]
            for summary in summaries
        ),
        "3_causal_ablation": all(
            summary["gates"]["paired_ablation_drop_at_least_48"]
            for summary in summaries
        ),
        "4_negative_routing_and_preservation": all(
            summary["gates"]["false_routes_at_most_2"]
            and summary["gates"]["negative_preserved_at_least_58"]
            for summary in summaries
        ),
        "5_conditional_deterministic_execution": all(
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
            "mean_addition_exact": mean(
                summary["addition"]["mathematical_exact"] for summary in summaries
            ),
            "total_false_routes": sum(
                summary["negative"]["false_routes"] for summary in summaries
            ),
            "total_negative_prompts": sum(
                summary["negative"]["examples"] for summary in summaries
            ),
            "total_preserved_negatives": sum(
                summary["negative"]["token_preserved"] for summary in summaries
            ),
        },
        "interpretation": (
            "Targeted route-row hardening passed the routing/preservation gate "
            "with zero false routes, while addition, operand, and causal gates "
            "also passed. The compound audit still failed because one seed lost "
            "a correctly decoded operand register during later generation, "
            "violating the all-exact conditional execution gate."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
