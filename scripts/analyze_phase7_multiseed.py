from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

DEFAULT_RESULTS = tuple(
    Path(f"phase7_results/multiseed_audit3_seed_{seed}.json")
    for seed in (13201, 13202, 13203)
)
DEFAULT_OUTPUT = Path("phase7_results/multiseed_audit3_summary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_examples(rows: list[dict[str, object]]) -> str:
    fields = ("prompt", "a", "b", "answer", "route_label", "family", "split")
    examples = [{field: row[field] for field in fields} for row in rows]
    return json.dumps(examples, sort_keys=True, separators=(",", ":"))


def first_route_probability(row: dict[str, object]) -> float:
    probability = row["implant"]["steps"][0]["route_probability"]
    if not isinstance(probability, list) or len(probability) != 1:
        raise ValueError("expected a singleton first-step route probability")
    return float(probability[0])


def seed_summary(result: dict[str, object], path: Path) -> dict[str, object]:
    if result.get("status") != "complete":
        raise ValueError(f"{path} is not complete")
    rows = result["rows"]
    positives = [row for row in rows if row["route_label"] is True]
    negatives = [row for row in rows if row["route_label"] is False]
    if len(positives) != 60 or len(negatives) != 60:
        raise ValueError(f"{path} does not contain 60 positive and 60 negative rows")
    direct = [row for row in positives if row["split"].endswith("_symbolic")]
    word = [row for row in positives if row["split"].endswith("_word")]
    eligible = [
        row
        for row in positives
        if row["first_step_route_active"] and row["first_step_operands_exact"]
    ]
    exact = sum(row["mathematical_correct"] is True for row in positives)
    ablation_exact = sum(
        row["ablation_mathematical_correct"] is True for row in positives
    )
    paired_ablation_drop = sum(
        row["mathematical_correct"] is True
        and row["ablation_mathematical_correct"] is False
        for row in positives
    )
    false_routes = sum(row["any_route_active"] is True for row in negatives)
    preserved = sum(row["token_preserved"] is True for row in negatives)
    conditional_exact = sum(row["mathematical_correct"] is True for row in eligible)
    conditional_format = sum(row["format_correct"] is True for row in eligible)
    conditional_trajectory = sum(
        row["calculator_trajectory_exact"] is True for row in eligible
    )
    route_probabilities = {
        "positive_min": min(first_route_probability(row) for row in positives),
        "positive_max": max(first_route_probability(row) for row in positives),
        "negative_min": min(first_route_probability(row) for row in negatives),
        "negative_max": max(first_route_probability(row) for row in negatives),
    }
    seed = int(Path(result["checkpoint"]).stem.rsplit("_", maxsplit=1)[-1])
    return {
        "seed": seed,
        "result_path": str(path),
        "result_sha256": sha256(path),
        "checkpoint_path": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "learned_parameters": result["learned_parameters"],
        "calculator_learned_parameters": result["calculator_learned_parameters"],
        "addition": {
            "examples": len(positives),
            "mathematical_exact": exact,
            "format_exact": sum(row["format_correct"] is True for row in positives),
            "direct_exact": sum(row["mathematical_correct"] is True for row in direct),
            "word_exact": sum(row["mathematical_correct"] is True for row in word),
            "route_active": sum(
                row["first_step_route_active"] is True for row in positives
            ),
            "operands_exact": sum(
                row["first_step_operands_exact"] is True for row in positives
            ),
            "trajectory_exact": sum(
                row["calculator_trajectory_exact"] is True for row in positives
            ),
            "ablation_exact": ablation_exact,
            "unpaired_exact_count_difference": exact - ablation_exact,
            "paired_ablation_drop": paired_ablation_drop,
        },
        "conditional_on_route_and_operands": {
            "examples": len(eligible),
            "mathematical_exact": conditional_exact,
            "format_exact": conditional_format,
            "trajectory_exact": conditional_trajectory,
        },
        "negative": {
            "examples": len(negatives),
            "false_routes": false_routes,
            "token_preserved": preserved,
        },
        "first_step_route_probability": route_probabilities,
        "gates": {
            "addition_exact_at_least_51": exact >= 51,
            "operands_exact_at_least_57": (
                sum(row["first_step_operands_exact"] is True for row in positives)
                >= 57
            ),
            "paired_ablation_drop_at_least_45": paired_ablation_drop >= 45,
            "false_routes_at_most_2": false_routes <= 2,
            "negative_preserved_at_least_58": preserved >= 58,
            "conditional_execution_all_exact": (
                len(eligible)
                == conditional_exact
                == conditional_format
                == conditional_trajectory
            ),
        },
    }


def metric_values(
    summaries: list[dict[str, object]],
    section: str,
    metric: str,
) -> list[int]:
    return [int(summary[section][metric]) for summary in summaries]


def aggregate_metric(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        nargs=3,
        type=Path,
        default=DEFAULT_RESULTS,
        metavar=("SEED1", "SEED2", "SEED3"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [(path, json.loads(path.read_text())) for path in args.results]
    example_payloads = [canonical_examples(result["rows"]) for _, result in loaded]
    if len(set(example_payloads)) != 1:
        raise ValueError("seed results are not aligned to the same shared holdout")
    summaries = [seed_summary(result, path) for path, result in loaded]
    seeds = [summary["seed"] for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected three independent checkpoint seeds")
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
    gate1 = (
        all(summary["gates"]["addition_exact_at_least_51"] for summary in summaries)
        and metrics["addition"]["mathematical_exact"]["mean"] >= 54
    )
    gate2 = all(
        summary["gates"]["operands_exact_at_least_57"] for summary in summaries
    )
    gate3 = all(
        summary["gates"]["paired_ablation_drop_at_least_45"]
        for summary in summaries
    )
    gate4 = all(
        summary["gates"]["false_routes_at_most_2"]
        and summary["gates"]["negative_preserved_at_least_58"]
        for summary in summaries
    )
    gate5 = all(
        summary["gates"]["conditional_execution_all_exact"] for summary in summaries
    )
    gates = {
        "1_addition_accuracy": gate1,
        "2_operand_recovery": gate2,
        "3_causal_ablation": gate3,
        "4_negative_routing_and_preservation": gate4,
        "5_conditional_deterministic_execution": gate5,
    }
    payload = {
        "protocol": "PHASE7_MULTISEED_PROTOCOL.md",
        "status": "complete",
        "shared_holdout_sha256": hashlib.sha256(
            example_payloads[0].encode()
        ).hexdigest(),
        "per_seed": summaries,
        "aggregate_mean_min_max": metrics,
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "passed_all_gates": all(gates.values()),
        "interpretation": (
            "Narrow calculator causality and conditional execution replicated, "
            "but the predeclared compound hypothesis failed because adversarial "
            "negative routing and token preservation missed gate 4."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
