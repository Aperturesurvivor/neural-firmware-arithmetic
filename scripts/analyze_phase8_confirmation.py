from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

RESULT_PATH = Path("phase8_results/confirmation.json")
ANALYSIS_PATH = Path("phase8_results/analysis.json")
BOOTSTRAP_SEED = 14_801
BOOTSTRAP_DRAWS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def clustered_bootstrap(
    rows: list[dict[str, object]],
    seeds: list[int],
) -> dict[str, object]:
    rng = random.Random(BOOTSTRAP_SEED)
    implant_values: list[float] = []
    implant_minus_adapter: list[float] = []
    implant_minus_base: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(rows) for _ in rows]
        implant = sum(
            row["implants"][str(seed)]["format_exact"]
            for row in sampled
            for seed in seeds
        ) / (len(sampled) * len(seeds))
        adapter = sum(
            row["matched_adapters"][str(seed)]["format_exact"]
            for row in sampled
            for seed in seeds
        ) / (len(sampled) * len(seeds))
        base = sum(row["base"]["format_exact"] for row in sampled) / len(sampled)
        implant_values.append(implant)
        implant_minus_adapter.append(implant - adapter)
        implant_minus_base.append(implant - base)
    return {
        "seed": BOOTSTRAP_SEED,
        "draws": BOOTSTRAP_DRAWS,
        "cluster": "confirmatory prompt; all three seeds retained within cluster",
        "implant_accuracy_95_percentile_interval": [
            percentile(implant_values, 0.025),
            percentile(implant_values, 0.975),
        ],
        "implant_minus_matched_adapter_95_percentile_interval": [
            percentile(implant_minus_adapter, 0.025),
            percentile(implant_minus_adapter, 0.975),
        ],
        "implant_minus_base_95_percentile_interval": [
            percentile(implant_minus_base, 0.025),
            percentile(implant_minus_base, 0.975),
        ],
        "note": (
            "Secondary descriptive interval; prompt rows are not independent "
            "model replications and the three implant seeds made identical "
            "exact/not-exact decisions."
        ),
    }


def main() -> None:
    payload = json.loads(RESULT_PATH.read_text())
    seeds = payload["training_seeds"]
    rows = payload["rows"]
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    completeness = {
        "rows": len(rows),
        "unique_prompts": len({row["prompt"] for row in rows}),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "all_base_present": all(row["base"] is not None for row in rows),
        "all_seed_conditions_present": all(
            set(row["implants"]) == {str(seed) for seed in seeds}
            and set(row["matched_adapters"]) == {str(seed) for seed in seeds}
            for row in rows
        ),
    }
    per_positive_family: dict[str, dict[str, object]] = {}
    positive_families = sorted({row["family"] for row in positives})
    for family in positive_families:
        group = [row for row in positives if row["family"] == family]
        per_positive_family[family] = {
            "examples": len(group),
            "implant_exact": {
                str(seed): sum(
                    row["implants"][str(seed)]["format_exact"] for row in group
                )
                for seed in seeds
            },
            "matched_adapter_exact": {
                str(seed): sum(
                    row["matched_adapters"][str(seed)]["format_exact"]
                    for row in group
                )
                for seed in seeds
            },
        }
    negative_categories = sorted({row["split"] for row in negatives})
    per_negative_category = {
        category: {
            "examples": sum(row["split"] == category for row in negatives),
            "implant_false_routes": {
                str(seed): sum(
                    row["split"] == category
                    and row["implants"][str(seed)]["first_route"]
                    for row in negatives
                )
                for seed in seeds
            },
            "implant_token_preserved": {
                str(seed): sum(
                    row["split"] == category
                    and row["implants"][str(seed)]["token_preserved"]
                    for row in negatives
                )
                for seed in seeds
            },
        }
        for category in negative_categories
    }
    failure_taxonomy: dict[str, object] = {}
    for seed in seeds:
        key = str(seed)
        route_off = sum(
            not row["implants"][key]["first_route"] for row in positives
        )
        invalid_or_inactive = sum(
            row["implants"][key]["first_route"]
            and not (
                row["implants"][key]["first_route_active"]
                and row["implants"][key]["operands_exact"]
            )
            for row in positives
        )
        conditional_decoder_failure = sum(
            row["implants"][key]["first_route_active"]
            and row["implants"][key]["operands_exact"]
            and not row["implants"][key]["format_exact"]
            for row in positives
        )
        failure_taxonomy[key] = {
            "route_off": route_off,
            "route_on_but_operand_handshake_failed": invalid_or_inactive,
            "conditional_decoder_failure": conditional_decoder_failure,
        }
    exact_patterns = Counter(
        tuple(
            bool(row["implants"][str(seed)]["format_exact"])
            for seed in seeds
        )
        for row in positives
    )
    false_route_patterns = Counter(
        tuple(
            bool(row["implants"][str(seed)]["first_route"])
            for seed in seeds
        )
        for row in negatives
    )
    checkpoint_shapes: dict[str, object] = {}
    for seed in seeds:
        path = Path(
            f"phase8_artifacts/confirmatory_implants/implant_seed_{seed}.pt"
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        input_count = checkpoint["input_rows"].numel()
        result_count = checkpoint["result_columns"].numel()
        checkpoint_shapes[str(seed)] = {
            "input_weights": input_count,
            "result_weights": result_count,
            "architectural_learned_total": input_count + result_count,
            "checkpoint_sha256": sha256(path),
        }
    analysis = {
        "status": "posthoc_analysis_complete",
        "source": {
            "path": str(RESULT_PATH),
            "sha256": sha256(RESULT_PATH),
        },
        "completeness": completeness,
        "frozen_gate_result": payload["gates"],
        "confirmatory_interpretation": (
            "Partial replication; the frozen primary gates did not pass. "
            "Deterministic execution and downstream decoding were reliable "
            "conditional on a valid interface, but prompt-generalization of "
            "routing and operand typing was insufficient."
        ),
        "effect_sizes": {
            "implant_accuracy": payload["implant_mean_exact"] / len(positives),
            "base_strict_accuracy": (
                payload["base"]["format_exact"] / len(positives)
            ),
            "base_extended_mathematical_accuracy": (
                payload["base"]["extended_mathematical_exact"] / len(positives)
            ),
            "matched_adapter_accuracy_by_seed": {
                str(seed): (
                    payload["per_seed"][str(seed)]["matched_adapter_exact"]
                    / len(positives)
                )
                for seed in seeds
            },
            "implant_minus_matched_adapter_by_seed": {
                str(seed): (
                    payload["per_seed"][str(seed)]["implant_exact"]
                    - payload["per_seed"][str(seed)]["matched_adapter_exact"]
                )
                / len(positives)
                for seed in seeds
            },
        },
        "failure_taxonomy": failure_taxonomy,
        "per_positive_family": per_positive_family,
        "per_negative_category": per_negative_category,
        "cross_seed_agreement": {
            "positive_exact_patterns": {
                str(pattern): count for pattern, count in exact_patterns.items()
            },
            "negative_false_route_patterns": {
                str(pattern): count
                for pattern, count in false_route_patterns.items()
            },
            "all_three_implants_identical_on_positive_exactness": (
                exact_patterns[(True, True, True)]
                + exact_patterns[(False, False, False)]
                == len(positives)
            ),
        },
        "secondary_clustered_bootstrap": clustered_bootstrap(positives, seeds),
        "latency": payload["latency"],
        "memory": payload["memory"],
        "parameters": payload["parameter_counts"],
        "checkpoint_shape_verification": checkpoint_shapes,
        "metadata_correction": {
            "field": (
                "phase8_results/confirmatory_implant_training.json "
                "seed_records[*].learned_parameters"
            ),
            "reported_value": 24_576,
            "correct_architectural_value": 57_344,
            "reason": (
                "The staged training helper counted only parameters whose "
                "requires_grad flag remained true after output training. The "
                "32,768 already-trained input weights were frozen for the "
                "output stage but remain learned and are present in every "
                "checkpoint. Checkpoint tensor shapes verify the total."
            ),
        },
        "posthoc_next_experiment": (
            "Treat router and operand generalization as the next target: add "
            "operation-contrast and semantic-role families to development "
            "training, then freeze a new independent audit. Do not reinterpret "
            "the present failed gates as a pass."
        ),
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "gates": analysis["frozen_gate_result"],
        "effect_sizes": analysis["effect_sizes"],
        "failure_taxonomy": analysis["failure_taxonomy"],
        "bootstrap": analysis["secondary_clustered_bootstrap"],
    }, indent=2))
    print(f"wrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
