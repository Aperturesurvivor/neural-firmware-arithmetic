from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SOURCE = Path("phase12_results/confirmation.json")
OUTPUT = Path("phase12_results/analysis.json")
SEEDS = (16_201, 16_202, 16_203)


def paired_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    positives = [row for row in rows if row["route_label"]]
    return np.asarray(
        [
            [
                int(
                    row["conditions"]["phase12_candidate"][str(seed)][
                        "format_exact"
                    ]
                )
                - int(
                    row["conditions"]["phase11_control"][str(seed)][
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
    if payload["status"] != "phase12_confirmation_complete":
        raise ValueError("Phase 12 confirmation is not complete")
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
        candidate = payload["metrics"]["phase12_candidate"][key]
        control = payload["metrics"]["phase11_control"][key]
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
            "control_exact": control["exact"],
            "candidate_exact": candidate["exact"],
            "paired_exact_gain": candidate["exact"] - control["exact"],
            "oracle_exact": candidate["oracle_exact"],
            "positive_route_probability": probability_summary(
                [
                    float(
                        row["conditions"]["phase12_candidate"][key][
                            "first_route_probability"
                        ]
                    )
                    for row in positives
                ]
            ),
            "negative_route_probability": probability_summary(
                [
                    float(
                        row["conditions"]["phase12_candidate"][key][
                            "first_route_probability"
                        ]
                    )
                    for row in negatives
                ]
            ),
        }

    category_summary: dict[str, object] = {}
    for row in rows:
        sign = "positive" if row["route_label"] else "negative"
        category = str(row["split"]).removeprefix(
            "phase12_confirmatory_positive_"
        ).removeprefix("phase12_confirmatory_negative_")
        name = f"{sign}_{category}"
        aggregate = category_summary.setdefault(
            name,
            {
                "positive": bool(row["route_label"]),
                "prompts": 0,
                "phase11_control": {"routes": 0, "exact": 0},
                "phase12_candidate": {"routes": 0, "exact": 0},
            },
        )
        aggregate["prompts"] += 1
        for condition in ("phase11_control", "phase12_candidate"):
            aggregate[condition]["routes"] += sum(
                int(
                    row["conditions"][condition][str(seed)][
                        "first_route"
                    ]
                )
                for seed in SEEDS
            )
            aggregate[condition]["exact"] += sum(
                int(
                    row["conditions"][condition][str(seed)][
                        "format_exact"
                    ]
                )
                for seed in SEEDS
            )

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
                "phase11_control_routes": 0,
                "phase12_candidate_routes": 0,
                "phase11_control_exact": 0,
                "phase12_candidate_exact": 0,
            },
        )
        aggregate["prompts"] += 1
        aggregate["seed_prompt_evaluations"] += len(SEEDS)
        for seed in SEEDS:
            key = str(seed)
            control = row["conditions"]["phase11_control"][key]
            candidate = row["conditions"]["phase12_candidate"][key]
            aggregate["phase11_control_routes"] += int(
                control["first_route"]
            )
            aggregate["phase12_candidate_routes"] += int(
                candidate["first_route"]
            )
            aggregate["phase11_control_exact"] += int(
                control["format_exact"]
            )
            aggregate["phase12_candidate_exact"] += int(
                candidate["format_exact"]
            )

    false_route_families = {
        family: aggregate
        for family, aggregate in family_summary.items()
        if not aggregate["route_label"]
        and aggregate["phase12_candidate_routes"] > 0
    }
    output = {
        "status": "phase12_posthoc_analysis_complete",
        "confirmatory_verdict": payload["gates"],
        "paired_phase12_vs_phase11": {
            "phase11_control_exact": sum(
                int(
                    row["conditions"]["phase11_control"][str(seed)][
                        "format_exact"
                    ]
                )
                for row in positives
                for seed in SEEDS
            ),
            "phase12_candidate_exact": sum(
                int(
                    row["conditions"]["phase12_candidate"][str(seed)][
                        "format_exact"
                    ]
                )
                for row in positives
                for seed in SEEDS
            ),
            "per_seed": per_seed,
            "two_way_bootstrap": two_way_bootstrap_interval(
                differences,
                seed=22_901,
            ),
        },
        "category_summary": category_summary,
        "false_route_families": false_route_families,
        "family_summary": family_summary,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
