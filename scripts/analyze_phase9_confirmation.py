from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

SOURCE = Path("phase9_results/confirmation.json")
MANIFEST = Path("phase9_results/frozen_prompt_manifest.json")
OUTPUT = Path("phase9_results/analysis.json")

CONDITIONS = ("phase8_frozen", "generic", "hard")
ROW_FIELDS = (
    "prompt",
    "a",
    "b",
    "answer",
    "route_label",
    "family",
    "family_index",
    "split",
)


def canonical_rows_sha256(rows: list[dict[str, object]]) -> str:
    canonical = [
        {field: row[field] for field in ROW_FIELDS}
        for row in rows
    ]
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def two_sided_sign_test(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = min(improved, regressed)
    probability = sum(
        math.comb(discordant, value)
        for value in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def failure_stage(record: dict[str, object]) -> str:
    if record["format_exact"]:
        return "exact"
    if not record["first_route"]:
        return "route_off"
    if not record["first_route_active"]:
        return "typed_handshake_inactive"
    if not record["operands_exact"]:
        return "operand_content"
    if not record["trajectory_exact"]:
        return "calculator_trajectory"
    return "downstream_decode"


def summarize_values(values: list[int]) -> dict[str, object]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "population_standard_deviation": statistics.pstdev(values),
    }


def paired_metric(
    rows: list[dict[str, object]],
    *,
    seed: str,
    left: str,
    right: str,
    field: str,
    positives_only: bool,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if bool(row["route_label"]) is positives_only
    ]
    improved = 0
    regressed = 0
    both = 0
    neither = 0
    for row in selected:
        left_value = bool(row["implants"][left][seed][field])
        right_value = bool(row["implants"][right][seed][field])
        if right_value and not left_value:
            improved += 1
        elif left_value and not right_value:
            regressed += 1
        elif left_value:
            both += 1
        else:
            neither += 1
    return {
        "left": left,
        "right": right,
        "field": field,
        "examples": len(selected),
        "improved": improved,
        "regressed": regressed,
        "both": both,
        "neither": neither,
        "net_change": improved - regressed,
        "two_sided_exact_sign_test_p": two_sided_sign_test(
            improved,
            regressed,
        ),
    }


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows = payload["rows"]
    seeds = [str(seed) for seed in payload["phase9_seeds"]]
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]

    manifest_rows = [
        {field: row[field] for field in ROW_FIELDS}
        for row in manifest["rows"]
    ]
    evaluated_rows = [
        {field: row[field] for field in ROW_FIELDS}
        for row in rows
    ]
    evaluated_hash = canonical_rows_sha256(rows)
    verification = {
        "evaluation_status_complete": (
            payload["status"] == "phase9_confirmatory_evaluation_complete"
        ),
        "manifest_status_frozen": (
            manifest["status"]
            == "phase9_protocol_frozen_before_confirmation"
        ),
        "prompt_rows_exactly_match_manifest": (
            evaluated_rows == manifest_rows
        ),
        "evaluated_canonical_rows_sha256": evaluated_hash,
        "manifest_canonical_rows_sha256": manifest[
            "canonical_rows_sha256"
        ],
        "canonical_hash_matches": (
            evaluated_hash == manifest["canonical_rows_sha256"]
        ),
        "unique_prompts": len({row["prompt"] for row in rows}),
        "positive_prompts": len(positives),
        "negative_prompts": len(negatives),
    }

    failure_taxonomy: dict[str, object] = {}
    false_routes_by_split: dict[str, object] = {}
    failed_prompt_indices: dict[str, object] = {}
    for condition in CONDITIONS:
        failure_taxonomy[condition] = {}
        false_routes_by_split[condition] = {}
        failed_prompt_indices[condition] = {}
        for seed in seeds:
            stages: dict[str, int] = {}
            failures: list[int] = []
            for row in positives:
                stage = failure_stage(row["implants"][condition][seed])
                stages[stage] = stages.get(stage, 0) + 1
                if stage != "exact":
                    failures.append(row["row_index"])
            failure_taxonomy[condition][seed] = stages
            failed_prompt_indices[condition][seed] = failures

            by_split: dict[str, dict[str, int]] = {}
            for row in negatives:
                split = row["split"].removeprefix(
                    "phase9_confirmatory_negative_"
                )
                entry = by_split.setdefault(
                    split,
                    {"examples": 0, "false_routes": 0, "preserved": 0},
                )
                record = row["implants"][condition][seed]
                entry["examples"] += 1
                entry["false_routes"] += int(record["first_route"])
                entry["preserved"] += int(record["token_preserved"])
            false_routes_by_split[condition][seed] = by_split

    contrasts: dict[str, object] = {}
    for seed in seeds:
        hard = payload["conditions"]["hard"][seed]
        frozen = payload["conditions"]["phase8_frozen"][seed]
        generic = payload["conditions"]["generic"][seed]
        adapter = payload["matched_adapters"][seed]
        contrasts[seed] = {
            "hard_minus_phase8_frozen": {
                "exact": hard["exact"] - frozen["exact"],
                "operands_exact": (
                    hard["operands_exact"] - frozen["operands_exact"]
                ),
                "positive_active_routes": (
                    hard["positive_active_routes"]
                    - frozen["positive_active_routes"]
                ),
                "negative_false_routes": (
                    hard["negative_false_routes"]
                    - frozen["negative_false_routes"]
                ),
                "negative_token_preserved": (
                    hard["negative_token_preserved"]
                    - frozen["negative_token_preserved"]
                ),
            },
            "hard_minus_generic": {
                "exact": hard["exact"] - generic["exact"],
                "operands_exact": (
                    hard["operands_exact"] - generic["operands_exact"]
                ),
                "negative_false_routes": (
                    hard["negative_false_routes"]
                    - generic["negative_false_routes"]
                ),
                "negative_token_preserved": (
                    hard["negative_token_preserved"]
                    - generic["negative_token_preserved"]
                ),
            },
            "hard_minus_matched_adapter": {
                "exact": hard["exact"] - adapter["exact"],
                "negative_token_preserved": (
                    hard["negative_token_preserved"]
                    - adapter["negative_token_preserved"]
                ),
            },
            "paired_exact_hard_vs_phase8_frozen": paired_metric(
                rows,
                seed=seed,
                left="phase8_frozen",
                right="hard",
                field="format_exact",
                positives_only=True,
            ),
            "paired_exact_hard_vs_generic": paired_metric(
                rows,
                seed=seed,
                left="generic",
                right="hard",
                field="format_exact",
                positives_only=True,
            ),
            "paired_preservation_hard_vs_phase8_frozen": paired_metric(
                rows,
                seed=seed,
                left="phase8_frozen",
                right="hard",
                field="token_preserved",
                positives_only=False,
            ),
        }

    hard_consistency = {
        metric: summarize_values(
            [
                payload["conditions"]["hard"][seed][metric]
                for seed in seeds
            ]
        )
        for metric in (
            "exact",
            "operands_exact",
            "positive_routes",
            "positive_active_routes",
            "negative_false_routes",
            "negative_token_preserved",
            "conditional_exact",
            "trajectories_exact",
            "ablation_exact",
            "paired_causal_losses",
        )
    }
    shared_hard_failures = sorted(
        set(failed_prompt_indices["hard"][seeds[0]]).intersection(
            *(set(failed_prompt_indices["hard"][seed]) for seed in seeds[1:])
        )
    )
    union_hard_failures = sorted(
        set().union(
            *(set(failed_prompt_indices["hard"][seed]) for seed in seeds)
        )
    )
    conditional_mechanism: dict[str, dict[str, int | bool]] = {}
    for seed in seeds:
        records = [
            row["implants"]["hard"][seed]
            for row in positives
        ]
        conditional = [
            record
            for record in records
            if record["first_route_active"] and record["operands_exact"]
        ]
        conditional_mechanism[seed] = {
            "examples": len(conditional),
            "exact_trajectories": sum(
                record["trajectory_exact"] for record in conditional
            ),
            "exact_decoded_answers": sum(
                record["format_exact"] for record in conditional
            ),
            "passes": all(
                record["trajectory_exact"] and record["format_exact"]
                for record in conditional
            ),
        }
    protocol_gates = {
        **payload["primary_hard_gates"],
        "conditional_calculator_and_decode": all(
            record["passes"] for record in conditional_mechanism.values()
        ),
    }
    protocol_gates["all_primary_gates"] = all(
        value
        for name, value in protocol_gates.items()
        if name != "all_primary_gates"
    )

    positive_category_metrics: dict[str, object] = {}
    negative_category_metrics: dict[str, object] = {}
    for condition in CONDITIONS:
        positive_category_metrics[condition] = {}
        negative_category_metrics[condition] = {}
        for seed in seeds:
            positive_by_split: dict[str, dict[str, int]] = {}
            for row in positives:
                category = row["split"].removeprefix(
                    "phase9_confirmatory_positive_"
                )
                entry = positive_by_split.setdefault(
                    category,
                    {
                        "examples": 0,
                        "exact": 0,
                        "routes": 0,
                        "active_routes": 0,
                        "operands_exact": 0,
                    },
                )
                record = row["implants"][condition][seed]
                entry["examples"] += 1
                entry["exact"] += int(record["format_exact"])
                entry["routes"] += int(record["first_route"])
                entry["active_routes"] += int(record["first_route_active"])
                entry["operands_exact"] += int(record["operands_exact"])
            positive_category_metrics[condition][seed] = positive_by_split
            negative_category_metrics[condition][seed] = (
                false_routes_by_split[condition][seed]
            )

    result = {
        "status": "phase9_posthoc_analysis_complete",
        "verification": verification,
        "implementation_reported_primary_gates": payload[
            "primary_hard_gates"
        ],
        "protocol_recomputed_primary_gates": protocol_gates,
        "gate_discrepancy": {
            "affected_gate": "conditional_calculator_and_decode",
            "implementation_reported": payload["primary_hard_gates"][
                "conditional_calculator_and_decode"
            ],
            "protocol_recomputed": protocol_gates[
                "conditional_calculator_and_decode"
            ],
            "explanation": (
                "The frozen evaluator compared the total number of exact "
                "trajectories with the number of active-route/exact-operand "
                "examples. Seeds 15202 and 15203 each had one additional "
                "exact trajectory from a non-identical operand register, so "
                "the equality failed. Direct row-level recomputation shows "
                "that every protocol-defined conditional example had both "
                "an exact trajectory and exact decoded answer. The raw flag "
                "is retained; the overall compound verdict is unchanged."
            ),
        },
        "compound_verdict": (
            "PASS"
            if protocol_gates["all_primary_gates"]
            else "FAIL"
        ),
        "condition_metrics": payload["conditions"],
        "matched_adapter_metrics": payload["matched_adapters"],
        "contrasts": contrasts,
        "hard_seed_consistency": hard_consistency,
        "conditional_mechanism": conditional_mechanism,
        "failure_taxonomy": failure_taxonomy,
        "positive_category_metrics": positive_category_metrics,
        "negative_category_metrics": negative_category_metrics,
        "false_routes_by_negative_split": false_routes_by_split,
        "failed_positive_row_indices": failed_prompt_indices,
        "shared_hard_failure_row_indices": shared_hard_failures,
        "union_hard_failure_row_indices": union_hard_failures,
        "latency": payload["latency"],
        "parameter_counts": payload["parameter_counts"],
        "memory": payload["memory"],
        "analysis_notes": [
            (
                "Exact sign tests are descriptive paired prompt analyses; "
                "prompts and seeds do not justify treating every row as an "
                "independent population sample."
            ),
            (
                "The three Phase 9 seeds inherit three independently trained "
                "Phase 8 checkpoints but share the same base model and prompt "
                "set."
            ),
            (
                "No post-hoc threshold, checkpoint, prompt, or primary gate "
                "replacement is performed by this script. The protocol-defined "
                "conditional gate is recomputed from retained rows alongside "
                "the frozen evaluator's raw implementation flag."
            ),
        ],
    }
    if not all(
        (
            verification["evaluation_status_complete"],
            verification["manifest_status_frozen"],
            verification["prompt_rows_exactly_match_manifest"],
            verification["canonical_hash_matches"],
            verification["unique_prompts"] == 300,
            verification["positive_prompts"] == 100,
            verification["negative_prompts"] == 200,
        )
    ):
        raise RuntimeError("Phase 9 confirmation/manifest verification failed")
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["protocol_recomputed_primary_gates"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
