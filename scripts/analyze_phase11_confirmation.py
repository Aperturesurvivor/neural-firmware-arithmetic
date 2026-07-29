from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SOURCE = Path("phase11_results/confirmation.json")
OUTPUT = Path("phase11_results/analysis.json")
SEEDS = (16_201, 16_202, 16_203)


def paired_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    positives = [row for row in rows if row["route_label"]]
    return np.asarray(
        [
            [
                int(
                    row["conditions"]["phase11_candidate"][str(seed)][
                        "format_exact"
                    ]
                )
                - int(
                    row["conditions"]["phase10_control"][str(seed)][
                        "format_exact"
                    ]
                )
                for row in positives
            ]
            for seed in SEEDS
        ],
        dtype=np.int8,
    )


def two_way_bootstrap_interval(
    differences: np.ndarray,
    *,
    seed: int,
    draws: int = 100_000,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    seed_count, prompt_count = differences.shape
    for draw in range(draws):
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        prompt_indices = rng.integers(0, prompt_count, size=prompt_count)
        values[draw] = differences[
            seed_indices[:, None],
            prompt_indices[None, :],
        ].mean()
    lower, upper = np.quantile(values, (0.025, 0.975))
    return {
        "draws": draws,
        "seed": seed,
        "mean_paired_difference": float(differences.mean()),
        "percentile_95_lower": float(lower),
        "percentile_95_upper": float(upper),
    }


def posthoc_preservation_frontier(
    rows: list[dict[str, object]],
    *,
    seed: int,
) -> dict[str, float | int]:
    key = str(seed)
    values = [
        {
            "positive": bool(row["route_label"]),
            "probability": float(
                row["conditions"]["phase11_candidate"][key][
                    "first_route_probability"
                ]
            ),
            "oracle_exact": bool(
                row["route_label"]
                and row["conditions"]["phase11_candidate"][key][
                    "oracle_route"
                ]["format_exact"]
            ),
        }
        for row in rows
    ]
    candidates: list[dict[str, float | int]] = []
    for threshold in sorted({row["probability"] for row in values}):
        false_routes = sum(
            not row["positive"] and row["probability"] >= threshold
            for row in values
        )
        if false_routes > 4:
            continue
        candidates.append(
            {
                "threshold": threshold,
                "false_routes": false_routes,
                "positive_routes": sum(
                    row["positive"] and row["probability"] >= threshold
                    for row in values
                ),
                "counterfactual_exact": sum(
                    row["positive"]
                    and row["probability"] >= threshold
                    and row["oracle_exact"]
                    for row in values
                ),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["counterfactual_exact"],
            row["positive_routes"],
        ),
    )


def probability_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    rows = payload["rows"]
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    differences = paired_matrix(rows)
    per_seed: dict[str, object] = {}
    for index, seed in enumerate(SEEDS):
        key = str(seed)
        seed_differences = differences[index]
        wins = int((seed_differences == 1).sum())
        losses = int((seed_differences == -1).sum())
        discordant = wins + losses
        candidate_metrics = payload["conditions"]["phase11_candidate"][key]
        control_metrics = payload["conditions"]["phase10_control"][key]
        positive_probabilities = [
            float(
                row["conditions"]["phase11_candidate"][key][
                    "first_route_probability"
                ]
            )
            for row in positives
        ]
        negative_probabilities = [
            float(
                row["conditions"]["phase11_candidate"][key][
                    "first_route_probability"
                ]
            )
            for row in negatives
        ]
        multiplication_probabilities = [
            float(
                row["conditions"]["phase11_candidate"][key][
                    "first_route_probability"
                ]
            )
            for row in negatives
            if row["split"].endswith("_multiplication")
        ]
        oracle_gap = (
            candidate_metrics["oracle_exact"] - control_metrics["exact"]
        )
        recovered_gap = candidate_metrics["exact"] - control_metrics["exact"]
        per_seed[key] = {
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": int((seed_differences == 0).sum()),
            "two_sided_exact_mcnemar_p": (
                float(
                    binomtest(
                        min(wins, losses),
                        discordant,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if discordant
                else 1.0
            ),
            "control_exact": control_metrics["exact"],
            "candidate_exact": candidate_metrics["exact"],
            "oracle_exact": candidate_metrics["oracle_exact"],
            "available_oracle_gap": oracle_gap,
            "recovered_oracle_gap": recovered_gap,
            "fraction_oracle_gap_recovered": (
                recovered_gap / oracle_gap if oracle_gap else 0.0
            ),
            "probabilities": {
                "positives": probability_summary(positive_probabilities),
                "all_negatives": probability_summary(negative_probabilities),
                "multiplication_negatives": probability_summary(
                    multiplication_probabilities
                ),
            },
            "posthoc_best_under_four_false_routes": (
                posthoc_preservation_frontier(rows, seed=seed)
            ),
        }

    category_summary: dict[str, object] = {}
    category_specs = (
        ("positive_direct", True, "direct"),
        ("positive_word", True, "word"),
        ("positive_distractor", True, "distractor"),
        ("negative_multiplication", False, "multiplication"),
        ("negative_factual", False, "factual"),
        ("negative_quoted", False, "quoted"),
        ("negative_negated", False, "negated"),
        ("negative_cancelled", False, "cancelled"),
        ("negative_subtraction", False, "subtraction"),
        ("negative_comparison", False, "comparison"),
        ("negative_concatenation", False, "concatenation"),
        ("negative_hypothetical", False, "hypothetical"),
        ("negative_distractor", False, "distractor"),
    )
    for name, positive, category in category_specs:
        category_rows = [
            row
            for row in rows
            if bool(row["route_label"]) == positive
            and row["split"].endswith(f"_{category}")
        ]
        category_summary[name] = {
            "positive": positive,
            "prompts": len(category_rows),
            "phase10_control": {
                "routes": sum(
                    row["conditions"]["phase10_control"][str(seed)][
                        "first_route"
                    ]
                    for row in category_rows
                    for seed in SEEDS
                ),
                "exact": sum(
                    row["conditions"]["phase10_control"][str(seed)][
                        "format_exact"
                    ]
                    for row in category_rows
                    for seed in SEEDS
                ),
            },
            "phase11_candidate": {
                "routes": sum(
                    row["conditions"]["phase11_candidate"][str(seed)][
                        "first_route"
                    ]
                    for row in category_rows
                    for seed in SEEDS
                ),
                "exact": sum(
                    row["conditions"]["phase11_candidate"][str(seed)][
                        "format_exact"
                    ]
                    for row in category_rows
                    for seed in SEEDS
                ),
            },
        }

    family_summary: dict[str, dict[str, object]] = {}
    for row in rows:
        family = row["family"]
        aggregate = family_summary.setdefault(
            family,
            {
                "split": row["split"],
                "route_label": bool(row["route_label"]),
                "prompts": 0,
                "seed_prompt_evaluations": 0,
                "candidate_routes": 0,
                "candidate_exact": 0,
            },
        )
        aggregate["prompts"] += 1
        aggregate["seed_prompt_evaluations"] += len(SEEDS)
        aggregate["candidate_routes"] += sum(
            row["conditions"]["phase11_candidate"][str(seed)]["first_route"]
            for seed in SEEDS
        )
        aggregate["candidate_exact"] += sum(
            row["conditions"]["phase11_candidate"][str(seed)]["format_exact"]
            for seed in SEEDS
        )

    false_route_families = {
        family: aggregate
        for family, aggregate in family_summary.items()
        if not aggregate["route_label"] and aggregate["candidate_routes"] > 0
    }
    output = {
        "status": "phase11_posthoc_analysis_complete",
        "confirmatory_verdict": payload["gates"],
        "paired_candidate_vs_control": {
            "control_exact": sum(
                row["conditions"]["phase10_control"][str(seed)]["format_exact"]
                for row in positives
                for seed in SEEDS
            ),
            "candidate_exact": sum(
                row["conditions"]["phase11_candidate"][str(seed)]["format_exact"]
                for row in positives
                for seed in SEEDS
            ),
            "per_seed": per_seed,
            "two_way_bootstrap": two_way_bootstrap_interval(
                differences,
                seed=18_001,
            ),
        },
        "category_summary": category_summary,
        "false_route_families": false_route_families,
        "family_summary": family_summary,
        "interpretation": {
            "threshold_only_repair_satisfies_frozen_gates": False,
            "basis": (
                "Even post hoc, the exactness-maximizing threshold with at "
                "most four false routes yields 66/74/68 exact."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
