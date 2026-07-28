from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SOURCE = Path("phase10_results/confirmation.json")
OUTPUT = Path("phase10_results/analysis.json")
SEEDS = (16_201, 16_202, 16_203)


def paired_matrix(
    rows: list[dict[str, object]],
    *,
    candidate: str,
    control: str,
) -> np.ndarray:
    positives = [row for row in rows if row["route_label"]]
    return np.asarray(
        [
            [
                int(row["conditions"][candidate][str(seed)]["format_exact"])
                - int(row["conditions"][control][str(seed)]["format_exact"])
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


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    positives = [row for row in payload["rows"] if row["route_label"]]
    representation = paired_matrix(
        payload["rows"],
        candidate="linear_representation",
        control="linear",
    )
    nonlinear = paired_matrix(
        payload["rows"],
        candidate="nonlinear",
        control="linear",
    )
    per_seed: dict[str, object] = {}
    for index, seed in enumerate(SEEDS):
        differences = representation[index]
        wins = int((differences == 1).sum())
        losses = int((differences == -1).sum())
        discordant = wins + losses
        per_seed[str(seed)] = {
            "representation_wins": wins,
            "representation_losses": losses,
            "ties": int((differences == 0).sum()),
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
        }
    category_summary: dict[str, object] = {}
    for category in ("direct", "word", "distractor"):
        rows = [
            row
            for row in positives
            if row["split"].endswith(f"_{category}")
        ]
        category_summary[category] = {}
        for condition in ("linear", "nonlinear", "linear_representation"):
            exact = sum(
                row["conditions"][condition][str(seed)]["format_exact"]
                for row in rows
                for seed in SEEDS
            )
            total = len(rows) * len(SEEDS)
            category_summary[category][condition] = {
                "exact": exact,
                "total": total,
                "rate": exact / total,
            }
    output = {
        "status": "phase10_posthoc_analysis_complete",
        "representation_vs_linear": {
            "linear_exact": sum(
                row["conditions"]["linear"][str(seed)]["format_exact"]
                for row in positives
                for seed in SEEDS
            ),
            "representation_exact": sum(
                row["conditions"]["linear_representation"][str(seed)][
                    "format_exact"
                ]
                for row in positives
                for seed in SEEDS
            ),
            "per_seed": per_seed,
            "two_way_bootstrap": two_way_bootstrap_interval(
                representation,
                seed=17_001,
            ),
        },
        "nonlinear_vs_linear": {
            "nonlinear_exact": sum(
                row["conditions"]["nonlinear"][str(seed)]["format_exact"]
                for row in positives
                for seed in SEEDS
            ),
            "linear_exact": sum(
                row["conditions"]["linear"][str(seed)]["format_exact"]
                for row in positives
                for seed in SEEDS
            ),
            "two_way_bootstrap": two_way_bootstrap_interval(
                nonlinear,
                seed=17_002,
            ),
        },
        "category_summary": category_summary,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
