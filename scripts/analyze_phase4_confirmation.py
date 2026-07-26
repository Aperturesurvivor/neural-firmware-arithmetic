from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import binomtest

RESULT_PATH = Path("phase4_results/confirmation_raw.json")
ANALYSIS_PATH = Path("phase4_results/confirmation_analysis.json")
SUMMARY_CSV = Path("phase4_results/confirmation_summary.csv")
FAMILY_CSV = Path("phase4_results/confirmation_by_family.csv")
FIGURE_DIRECTORY = Path("paper_phase4/figures")
CONDITIONS = ("base", "control", "internal", "oracle", "off")
SPLITS = (
    "confirmatory_id",
    "confirmatory_ood",
    "confirmatory_long",
    "confirmatory_word",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return [center - margin, center + margin]


def pooled_rows(result: dict[str, object], condition: str) -> list[dict[str, object]]:
    condition_results = result[f"{condition}_results"]
    return [
        row
        for split in SPLITS
        for row in condition_results[split]["rows"]
    ]


def score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    mathematical = sum(row["mathematical_correct"] is True for row in rows)
    exact = sum(row["exact_format_correct"] is True for row in rows)
    return {
        "examples": total,
        "mathematical_correct": mathematical,
        "mathematical_accuracy": mathematical / total,
        "mathematical_wilson_95": wilson_interval(mathematical, total),
        "exact_format_correct": exact,
        "exact_format_accuracy": exact / total,
        "exact_format_wilson_95": wilson_interval(exact, total),
    }


def paired_comparison(
    comparator_rows: list[dict[str, object]],
    internal_rows: list[dict[str, object]],
) -> dict[str, object]:
    comparator_only = 0
    internal_only = 0
    both_correct = 0
    both_wrong = 0
    for comparator, internal in zip(comparator_rows, internal_rows, strict=True):
        if comparator["prompt"] != internal["prompt"]:
            raise ValueError("paired condition rows are not prompt-aligned")
        if comparator["expected"] != internal["expected"]:
            raise ValueError("paired condition rows disagree on expected answer")
        comparator_correct = comparator["mathematical_correct"] is True
        internal_correct = internal["mathematical_correct"] is True
        if comparator_correct and internal_correct:
            both_correct += 1
        elif comparator_correct:
            comparator_only += 1
        elif internal_correct:
            internal_only += 1
        else:
            both_wrong += 1
    discordant = comparator_only + internal_only
    p_value = (
        float(
            binomtest(
                min(comparator_only, internal_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "both_correct": both_correct,
        "internal_only_correct": internal_only,
        "comparator_only_correct": comparator_only,
        "both_wrong": both_wrong,
        "paired_percentage_point_difference": (
            100 * (internal_only - comparator_only) / len(internal_rows)
        ),
        "exact_mcnemar_p_value": p_value,
    }


def write_summary_csv(
    result: dict[str, object],
    aggregate: dict[str, dict[str, object]],
) -> None:
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "condition",
                "split",
                "examples",
                "mathematical_correct",
                "mathematical_accuracy",
                "exact_format_correct",
                "exact_format_accuracy",
                "route_activations",
            )
        )
        for condition in CONDITIONS:
            for split in SPLITS:
                row = result[f"{condition}_results"][split]
                writer.writerow(
                    (
                        condition,
                        split,
                        row["examples"],
                        row["mathematical_correct"],
                        row["mathematical_accuracy"],
                        row["exact_format_correct"],
                        row["exact_format_accuracy"],
                        row["route_activations"],
                    )
                )
            writer.writerow(
                (
                    condition,
                    "pooled",
                    aggregate[condition]["examples"],
                    aggregate[condition]["mathematical_correct"],
                    aggregate[condition]["mathematical_accuracy"],
                    aggregate[condition]["exact_format_correct"],
                    aggregate[condition]["exact_format_accuracy"],
                    (
                        sum(
                            result[f"{condition}_results"][split][
                                "route_activations"
                            ]
                            for split in SPLITS
                        )
                    ),
                )
            )


def family_rows(result: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for split in SPLITS:
        base_rows = result["base_results"][split]["rows"]
        control_rows = result["control_results"][split]["rows"]
        internal_rows = result["internal_results"][split]["rows"]
        for base, control, internal in zip(
            base_rows,
            control_rows,
            internal_rows,
            strict=True,
        ):
            if not (base["prompt"] == control["prompt"] == internal["prompt"]):
                raise ValueError("family rows are not prompt-aligned")
            key = (split, internal["family"])
            grouped[key].append(
                (
                    int(base["mathematical_correct"] is True),
                    int(control["mathematical_correct"] is True),
                    int(internal["mathematical_correct"] is True),
                    int(internal["route_active"] is True),
                )
            )
    rows = []
    for (split, family), values in sorted(grouped.items()):
        total = len(values)
        rows.append(
            {
                "split": split,
                "family": family,
                "examples": total,
                "base_accuracy": sum(value[0] for value in values) / total,
                "control_accuracy": sum(value[1] for value in values) / total,
                "internal_accuracy": sum(value[2] for value in values) / total,
                "route_activation_rate": sum(value[3] for value in values) / total,
            }
        )
    return rows


def write_family_csv(rows: list[dict[str, object]]) -> None:
    with FAMILY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_figures(
    result: dict[str, object],
    aggregate: dict[str, dict[str, object]],
) -> None:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    labels = ("1–4 digits", "5–8 digits", "9–12 digits", "Word problems")
    selected = ("base", "control", "internal", "oracle")
    colors = ("#6B7280", "#D97706", "#2563EB", "#059669")
    x_positions = list(range(len(SPLITS)))
    width = 0.19
    figure, axis = plt.subplots(figsize=(8.2, 4.5))
    for condition_index, (condition, color) in enumerate(
        zip(selected, colors, strict=True)
    ):
        values = [
            100 * result[f"{condition}_results"][split]["mathematical_accuracy"]
            for split in SPLITS
        ]
        offsets = [
            position + (condition_index - 1.5) * width for position in x_positions
        ]
        axis.bar(offsets, values, width=width, label=condition.title(), color=color)
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("Mathematical accuracy (%)")
    axis.set_ylim(0, 105)
    axis.legend(frameon=False, ncol=4, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "accuracy_by_split.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIRECTORY / "accuracy_by_split.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    aggregate_conditions = ("base", "control", "internal", "oracle")
    values = [
        100 * aggregate[condition]["mathematical_accuracy"]
        for condition in aggregate_conditions
    ]
    bars = axis.bar(
        [condition.title() for condition in aggregate_conditions],
        values,
        color=colors,
        width=0.68,
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylabel("Pooled mathematical accuracy (%)")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "aggregate_accuracy.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIRECTORY / "aggregate_accuracy.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    aggregate = {
        condition: score_rows(pooled_rows(result, condition))
        for condition in CONDITIONS
    }
    internal_rows = pooled_rows(result, "internal")
    base_rows = pooled_rows(result, "base")
    control_rows = pooled_rows(result, "control")
    positive_routes = sum(row["route_active"] is True for row in internal_rows)
    negative_examples = result["negative_preservation"]["examples"]
    negative_routes = result["negative_preservation"]["route_activations"]
    oracle_correct = aggregate["oracle"]["mathematical_correct"]
    off_identity = sum(result["forced_off_identity"].values())
    active_arithmetic_errors = sum(
        row["route_active"] is True and row["mathematical_correct"] is False
        for row in internal_rows
    )
    inactive_base_rescues = sum(
        row["route_active"] is False and row["mathematical_correct"] is True
        for row in internal_rows
    )
    comparisons = {
        "internal_vs_base": paired_comparison(base_rows, internal_rows),
        "internal_vs_control": paired_comparison(control_rows, internal_rows),
    }
    success_criteria = {
        "gain_over_base_at_least_30pp": (
            aggregate["internal"]["mathematical_accuracy"]
            - aggregate["base"]["mathematical_accuracy"]
            >= 0.30
        ),
        "gain_over_control_at_least_30pp": (
            aggregate["internal"]["mathematical_accuracy"]
            - aggregate["control"]["mathematical_accuracy"]
            >= 0.30
        ),
        "positive_route_rate_at_least_90pct": positive_routes / 400 >= 0.90,
        "negative_false_route_rate_at_most_2pct": (
            negative_routes / negative_examples <= 0.02
        ),
        "oracle_accuracy_at_least_99pct": oracle_correct / 400 >= 0.99,
        "forced_off_identity_100pct": off_identity / 400 == 1.0,
    }
    rendered_digest = hashlib.sha256(
        json.dumps(
            result["rendered_data"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    analysis = {
        "aggregate": aggregate,
        "comparisons": comparisons,
        "routing": {
            "positive_examples": 400,
            "positive_activations": positive_routes,
            "positive_activation_rate": positive_routes / 400,
            "positive_activation_wilson_95": wilson_interval(positive_routes, 400),
            "negative_examples": negative_examples,
            "negative_activations": negative_routes,
            "negative_activation_rate": negative_routes / negative_examples,
            "negative_activation_wilson_95": wilson_interval(
                negative_routes,
                negative_examples,
            ),
            "active_arithmetic_errors": active_arithmetic_errors,
            "inactive_prompts_answered_correctly_by_base_path": inactive_base_rescues,
        },
        "causal_controls": {
            "oracle_correct": oracle_correct,
            "oracle_examples": 400,
            "forced_off_token_identical": off_identity,
            "forced_off_examples": 400,
            "negative_token_identical": result["negative_preservation"][
                "token_exact_preserved"
            ],
            "negative_examples": negative_examples,
        },
        "success_criteria": success_criteria,
        "all_preregistered_criteria_met": all(success_criteria.values()),
        "rendered_data_sha256": rendered_digest,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n")
    write_summary_csv(result, aggregate)
    per_family = family_rows(result)
    write_family_csv(per_family)
    render_figures(result, aggregate)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
