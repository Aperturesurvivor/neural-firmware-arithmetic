from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

RAW_DIRECTORY = Path("phase5_results/confirmation_raw_v1")
ANALYSIS_PATH = Path("phase5_results/confirmation_analysis_v1.json")
SUMMARY_CSV = Path("phase5_results/confirmation_summary_v1.csv")
COMPARISON_CSV = Path("phase5_results/confirmation_comparisons_v1.csv")
FAMILY_CSV = Path("phase5_results/confirmation_by_family_v1.csv")
FIGURE_DIRECTORY = Path("paper_phase5/figures")
SEEDS = (10_701, 10_702, 10_703)
CONDITIONS = ("typed_firmware", "adapter", "igc_matched", "igc_native")
SPLITS = (
    "phase5_confirmatory_id_1_4",
    "phase5_confirmatory_ood_5_8",
    "phase5_confirmatory_long_9_12",
    "phase5_confirmatory_word_5_8",
)
BOOTSTRAP_SEED = 20_260_726
BOOTSTRAP_DRAWS = 20_000


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_results() -> tuple[dict[str, object], dict[str, dict[int, dict[str, object]]]]:
    base = json.loads((RAW_DIRECTORY / "base.json").read_text())
    trained: dict[str, dict[int, dict[str, object]]] = {}
    for condition in CONDITIONS:
        trained[condition] = {}
        for seed in SEEDS:
            path = RAW_DIRECTORY / f"{condition}_seed_{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing confirmatory result: {path}")
            result = json.loads(path.read_text())
            if result["condition"] != condition or int(result["seed"]) != seed:
                raise ValueError(f"result identity mismatch: {path}")
            trained[condition][seed] = result
    return base, trained


def validate_alignment(
    base: dict[str, object],
    trained: dict[str, dict[int, dict[str, object]]],
) -> None:
    base_positive = base["positive_rows"]
    base_negative = base["negative_rows"]
    if len(base_positive) != 400 or len(base_negative) != 160:
        raise ValueError("unexpected base confirmatory row count")
    expected_positive = [
        (row["prompt"], row["expected"], row["split"]) for row in base_positive
    ]
    expected_negative = [
        (row["prompt"], row["split"]) for row in base_negative
    ]
    for condition in CONDITIONS:
        for seed in SEEDS:
            result = trained[condition][seed]
            observed_positive = [
                (row["prompt"], row["expected"], row["split"])
                for row in result["positive_rows"]
            ]
            observed_negative = [
                (row["prompt"], row["split"]) for row in result["negative_rows"]
            ]
            if observed_positive != expected_positive:
                raise ValueError(f"positive prompt mismatch: {condition} seed {seed}")
            if observed_negative != expected_negative:
                raise ValueError(f"negative prompt mismatch: {condition} seed {seed}")


def binary_scores(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([row[key] is True for row in rows], dtype=np.int8)


def score_positive(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    mathematical = int(binary_scores(rows, "mathematical_correct").sum())
    exact = int(binary_scores(rows, "exact_format_correct").sum())
    return {
        "examples": total,
        "mathematical_correct": mathematical,
        "mathematical_accuracy": mathematical / total,
        "mathematical_wilson_95": wilson_interval(mathematical, total),
        "exact_format_correct": exact,
        "exact_format_accuracy": exact / total,
        "exact_format_wilson_95": wilson_interval(exact, total),
    }


def metric_with_wilson(successes: int, total: int) -> dict[str, object]:
    return {
        "successes": successes,
        "examples": total,
        "rate": successes / total,
        "wilson_95": wilson_interval(successes, total),
    }


def summarize_seed(result: dict[str, object]) -> dict[str, object]:
    positive_rows = result["positive_rows"]
    negative_rows = result["negative_rows"]
    summary = {
        "learned_parameters": int(result["learned_parameters"]),
        "positive": score_positive(positive_rows),
        "by_split": {
            split: score_positive(
                [row for row in positive_rows if row["split"] == split]
            )
            for split in SPLITS
        },
        "positive_routing": metric_with_wilson(
            sum(row["route_active"] is True for row in positive_rows),
            len(positive_rows),
        ),
        "negative_false_routing": metric_with_wilson(
            sum(row["route_active"] is True for row in negative_rows),
            len(negative_rows),
        ),
        "negative_token_preservation": metric_with_wilson(
            int(result["summary"]["token_exact_preserved"]),
            len(negative_rows),
        ),
        "positive_latency": result["summary"]["positive_latency"],
        "negative_latency": result["summary"]["negative_latency"],
    }
    register_rows = [
        row for row in positive_rows if row.get("registers_exact") is not None
    ]
    if register_rows:
        exact_registers = sum(row["registers_exact"] is True for row in register_rows)
        eligible = [
            row
            for row in register_rows
            if row["route_active"] is True and row["registers_exact"] is True
        ]
        summary["operand_registers"] = metric_with_wilson(
            exact_registers,
            len(register_rows),
        )
        summary["route_and_register_eligible"] = len(eligible)
        summary["eligible_arithmetic"] = (
            metric_with_wilson(
                sum(row["mathematical_correct"] is True for row in eligible),
                len(eligible),
            )
            if eligible
            else None
        )
    return summary


def mean_and_sample_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_condition(
    condition: str,
    trained: dict[str, dict[int, dict[str, object]]],
    seed_summaries: dict[int, dict[str, object]],
) -> dict[str, object]:
    results = trained[condition]
    pooled_positive = [
        row for seed in SEEDS for row in results[seed]["positive_rows"]
    ]
    pooled_negative = [
        row for seed in SEEDS for row in results[seed]["negative_rows"]
    ]
    learned_parameters = {
        int(results[seed]["learned_parameters"]) for seed in SEEDS
    }
    if len(learned_parameters) != 1:
        raise ValueError(f"learned parameter count varied for {condition}")
    mathematical_rates = [
        float(seed_summaries[seed]["positive"]["mathematical_accuracy"])
        for seed in SEEDS
    ]
    exact_rates = [
        float(seed_summaries[seed]["positive"]["exact_format_accuracy"])
        for seed in SEEDS
    ]
    route_rates = [
        float(seed_summaries[seed]["positive_routing"]["rate"]) for seed in SEEDS
    ]
    false_route_rates = [
        float(seed_summaries[seed]["negative_false_routing"]["rate"])
        for seed in SEEDS
    ]
    preservation_rates = [
        float(seed_summaries[seed]["negative_token_preservation"]["rate"])
        for seed in SEEDS
    ]
    positive_latencies = [
        float(row["latency_seconds"]) for row in pooled_positive
    ]
    negative_latencies = [
        float(row["latency_seconds"]) for row in pooled_negative
    ]
    aggregate = {
        "learned_parameters": learned_parameters.pop(),
        "seed_count": len(SEEDS),
        "pooled_predictions": score_positive(pooled_positive),
        "seed_mathematical_accuracy": mean_and_sample_sd(mathematical_rates),
        "seed_exact_format_accuracy": mean_and_sample_sd(exact_rates),
        "seed_positive_route_rate": mean_and_sample_sd(route_rates),
        "seed_negative_false_route_rate": mean_and_sample_sd(false_route_rates),
        "seed_negative_token_preservation_rate": mean_and_sample_sd(
            preservation_rates
        ),
        "latency_seconds": {
            "positive_mean": statistics.fmean(positive_latencies),
            "positive_median": statistics.median(positive_latencies),
            "negative_mean": statistics.fmean(negative_latencies),
            "negative_median": statistics.median(negative_latencies),
        },
        "by_split_seed_mean": {},
    }
    for split in SPLITS:
        split_rates = [
            float(seed_summaries[seed]["by_split"][split]["mathematical_accuracy"])
            for seed in SEEDS
        ]
        aggregate["by_split_seed_mean"][split] = mean_and_sample_sd(split_rates)
    if condition.startswith("igc_"):
        register_rates = [
            float(seed_summaries[seed]["operand_registers"]["rate"])
            for seed in SEEDS
        ]
        aggregate["seed_operand_register_accuracy"] = mean_and_sample_sd(
            register_rates
        )
        eligible_rows = [
            row
            for row in pooled_positive
            if row["route_active"] is True and row["registers_exact"] is True
        ]
        aggregate["route_and_register_eligible_predictions"] = len(eligible_rows)
        aggregate["eligible_mathematical_correct"] = sum(
            row["mathematical_correct"] is True for row in eligible_rows
        )
        aggregate["eligible_mathematical_accuracy"] = (
            sum(row["mathematical_correct"] is True for row in eligible_rows)
            / len(eligible_rows)
            if eligible_rows
            else None
        )
    return aggregate


def paired_counts(
    comparator: np.ndarray,
    typed: np.ndarray,
) -> dict[str, object]:
    if comparator.shape != typed.shape:
        raise ValueError("paired score shapes differ")
    both_correct = int(np.logical_and(comparator == 1, typed == 1).sum())
    typed_only = int(np.logical_and(comparator == 0, typed == 1).sum())
    comparator_only = int(np.logical_and(comparator == 1, typed == 0).sum())
    both_wrong = int(np.logical_and(comparator == 0, typed == 0).sum())
    discordant = typed_only + comparator_only
    p_value = (
        float(
            binomtest(
                min(typed_only, comparator_only),
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
        "typed_only_correct": typed_only,
        "comparator_only_correct": comparator_only,
        "both_wrong": both_wrong,
        "typed_minus_comparator_percentage_points": (
            100 * float((typed - comparator).mean())
        ),
        "exact_mcnemar_p_value": p_value,
    }


def crossed_cluster_bootstrap(
    differences: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[float]:
    """Resample paired seed clusters and prompt clusters independently."""
    if differences.shape != (len(SEEDS), 400):
        raise ValueError(f"unexpected paired matrix shape: {differences.shape}")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_seeds = generator.integers(0, differences.shape[0], differences.shape[0])
        sampled_prompts = generator.integers(
            0,
            differences.shape[1],
            differences.shape[1],
        )
        values[draw] = differences[np.ix_(sampled_seeds, sampled_prompts)].mean()
    lower, upper = np.quantile(values, (0.025, 0.975))
    return [100 * float(lower), 100 * float(upper)]


def comparison(
    comparator_name: str,
    base: dict[str, object],
    trained: dict[str, dict[int, dict[str, object]]],
) -> dict[str, object]:
    typed_scores = np.stack(
        [
            binary_scores(
                trained["typed_firmware"][seed]["positive_rows"],
                "mathematical_correct",
            )
            for seed in SEEDS
        ]
    )
    if comparator_name == "base":
        base_scores = binary_scores(base["positive_rows"], "mathematical_correct")
        comparator_scores = np.repeat(base_scores[np.newaxis, :], len(SEEDS), axis=0)
    else:
        comparator_scores = np.stack(
            [
                binary_scores(
                    trained[comparator_name][seed]["positive_rows"],
                    "mathematical_correct",
                )
                for seed in SEEDS
            ]
        )
    differences = typed_scores - comparator_scores
    per_seed = {
        str(seed): paired_counts(comparator_scores[index], typed_scores[index])
        for index, seed in enumerate(SEEDS)
    }
    pooled_descriptive = paired_counts(
        comparator_scores.reshape(-1),
        typed_scores.reshape(-1),
    )
    pooled_descriptive.pop("exact_mcnemar_p_value")
    return {
        "comparator": comparator_name,
        "paired_predictions": int(differences.size),
        "typed_minus_comparator_percentage_points": 100 * float(differences.mean()),
        "crossed_seed_prompt_bootstrap_95_percentage_points": (
            crossed_cluster_bootstrap(differences)
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_seed_exact_mcnemar": per_seed,
        "pooled_paired_counts_descriptive_only": pooled_descriptive,
        "inference_note": (
            "The confidence interval independently resamples the three paired "
            "training-seed clusters and 400 paired prompt clusters. Exact "
            "McNemar tests are reported separately by seed; the pooled counts "
            "are descriptive because prompt reuse creates dependence."
        ),
    }


def build_family_rows(
    base: dict[str, object],
    trained: dict[str, dict[int, dict[str, object]]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in base["positive_rows"]:
        key = (row["split"], row["family"])
        grouped[key]["base"].append(int(row["mathematical_correct"] is True))
    for condition in CONDITIONS:
        for seed in SEEDS:
            for row in trained[condition][seed]["positive_rows"]:
                key = (row["split"], row["family"])
                grouped[key][condition].append(
                    int(row["mathematical_correct"] is True)
                )
    output = []
    for (split, family), scores in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "family": family,
                "prompts": len(scores["base"]),
                "base_accuracy": statistics.fmean(scores["base"]),
                "typed_firmware_seed_mean_accuracy": statistics.fmean(
                    scores["typed_firmware"]
                ),
                "adapter_seed_mean_accuracy": statistics.fmean(scores["adapter"]),
                "igc_matched_seed_mean_accuracy": statistics.fmean(
                    scores["igc_matched"]
                ),
                "igc_native_seed_mean_accuracy": statistics.fmean(
                    scores["igc_native"]
                ),
            }
        )
    return output


def write_summary_csv(
    base: dict[str, object],
    trained: dict[str, dict[int, dict[str, object]]],
    seed_summaries: dict[str, dict[int, dict[str, object]]],
) -> None:
    fieldnames = (
        "condition",
        "seed",
        "split",
        "examples",
        "mathematical_correct",
        "mathematical_accuracy",
        "exact_format_correct",
        "exact_format_accuracy",
        "positive_route_rate",
        "negative_false_route_rate",
        "negative_token_preservation_rate",
        "operand_register_accuracy",
        "positive_mean_latency_seconds",
        "learned_parameters",
    )
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        base_summary = score_positive(base["positive_rows"])
        writer.writerow(
            {
                "condition": "base",
                "seed": "",
                "split": "pooled",
                **{
                    key: base_summary[key]
                    for key in (
                        "examples",
                        "mathematical_correct",
                        "mathematical_accuracy",
                        "exact_format_correct",
                        "exact_format_accuracy",
                    )
                },
                "positive_mean_latency_seconds": base["summary"]["positive_latency"][
                    "mean_seconds"
                ],
                "learned_parameters": 0,
            }
        )
        for split in SPLITS:
            split_summary = score_positive(
                [row for row in base["positive_rows"] if row["split"] == split]
            )
            writer.writerow(
                {
                    "condition": "base",
                    "seed": "",
                    "split": split,
                    **{
                        key: split_summary[key]
                        for key in (
                            "examples",
                            "mathematical_correct",
                            "mathematical_accuracy",
                            "exact_format_correct",
                            "exact_format_accuracy",
                        )
                    },
                    "learned_parameters": 0,
                }
            )
        for condition in CONDITIONS:
            for seed in SEEDS:
                summary = seed_summaries[condition][seed]
                pooled = summary["positive"]
                writer.writerow(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": "pooled",
                        **{
                            key: pooled[key]
                            for key in (
                                "examples",
                                "mathematical_correct",
                                "mathematical_accuracy",
                                "exact_format_correct",
                                "exact_format_accuracy",
                            )
                        },
                        "positive_route_rate": summary["positive_routing"]["rate"],
                        "negative_false_route_rate": summary[
                            "negative_false_routing"
                        ]["rate"],
                        "negative_token_preservation_rate": summary[
                            "negative_token_preservation"
                        ]["rate"],
                        "operand_register_accuracy": (
                            summary.get("operand_registers", {}).get("rate", "")
                        ),
                        "positive_mean_latency_seconds": summary["positive_latency"][
                            "mean_seconds"
                        ],
                        "learned_parameters": summary["learned_parameters"],
                    }
                )
                for split in SPLITS:
                    split_summary = summary["by_split"][split]
                    writer.writerow(
                        {
                            "condition": condition,
                            "seed": seed,
                            "split": split,
                            **{
                                key: split_summary[key]
                                for key in (
                                    "examples",
                                    "mathematical_correct",
                                    "mathematical_accuracy",
                                    "exact_format_correct",
                                    "exact_format_accuracy",
                                )
                            },
                            "learned_parameters": summary["learned_parameters"],
                        }
                    )


def write_comparison_csv(comparisons: dict[str, dict[str, object]]) -> None:
    with COMPARISON_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "comparison",
                "typed_minus_comparator_percentage_points",
                "bootstrap_95_lower_percentage_points",
                "bootstrap_95_upper_percentage_points",
                "paired_predictions",
            )
        )
        for name, result in comparisons.items():
            interval = result[
                "crossed_seed_prompt_bootstrap_95_percentage_points"
            ]
            writer.writerow(
                (
                    name,
                    result["typed_minus_comparator_percentage_points"],
                    interval[0],
                    interval[1],
                    result["paired_predictions"],
                )
            )


def render_figures(
    base: dict[str, object],
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
    labels = ("Base", "Adapter", "Typed\nfirmware", "Matched\nIGC", "Native\nIGC")
    keys = ("base", "adapter", "typed_firmware", "igc_matched", "igc_native")
    colors = ("#6B7280", "#D97706", "#2563EB", "#A855F7", "#059669")
    values = [
        100 * base["summary"]["mathematical_accuracy"],
        *[
            100 * aggregate[key]["seed_mathematical_accuracy"]["mean"]
            for key in keys[1:]
        ],
    ]
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    bars = axis.bar(labels, values, color=colors, width=0.68)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set_ylabel("Mathematical accuracy (%)")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "aggregate_accuracy.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIRECTORY / "aggregate_accuracy.png", bbox_inches="tight")
    plt.close(figure)

    split_labels = ("1–4 digits", "5–8 digits", "9–12 digits", "Word problems")
    x_positions = np.arange(len(SPLITS))
    selected = ("base", "typed_firmware", "igc_native")
    selected_colors = ("#6B7280", "#2563EB", "#059669")
    selected_labels = ("Base", "Typed Firmware", "Native IGC")
    width = 0.25
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    for index, (condition, color, label) in enumerate(
        zip(selected, selected_colors, selected_labels, strict=True)
    ):
        if condition == "base":
            split_values = [
                100
                * score_positive(
                    [
                        row
                        for row in base["positive_rows"]
                        if row["split"] == split
                    ]
                )["mathematical_accuracy"]
                for split in SPLITS
            ]
        else:
            split_values = [
                100 * aggregate[condition]["by_split_seed_mean"][split]["mean"]
                for split in SPLITS
            ]
        axis.bar(
            x_positions + (index - 1) * width,
            split_values,
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(x_positions, split_labels)
    axis.set_ylabel("Mathematical accuracy (%)")
    axis.set_ylim(0, 105)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "accuracy_by_split.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIRECTORY / "accuracy_by_split.png", bbox_inches="tight")
    plt.close(figure)

    parameter_values = [
        aggregate["typed_firmware"]["learned_parameters"],
        aggregate["igc_matched"]["learned_parameters"],
        aggregate["igc_native"]["learned_parameters"],
    ]
    accuracy_values = [
        100 * aggregate[key]["seed_mathematical_accuracy"]["mean"]
        for key in ("typed_firmware", "igc_matched", "igc_native")
    ]
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.scatter(
        parameter_values,
        accuracy_values,
        s=(100, 100, 120),
        color=("#2563EB", "#A855F7", "#059669"),
    )
    for label, x_value, y_value in zip(
        ("Typed firmware", "Matched IGC", "Native IGC"),
        parameter_values,
        accuracy_values,
        strict=True,
    ):
        axis.annotate(label, (x_value, y_value), xytext=(6, 6), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("Learned parameters (log scale)")
    axis.set_ylabel("Mathematical accuracy (%)")
    axis.set_ylim(-3, 105)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "accuracy_vs_parameters.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "accuracy_vs_parameters.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    base, trained = load_results()
    validate_alignment(base, trained)
    seed_summaries = {
        condition: {
            seed: summarize_seed(trained[condition][seed]) for seed in SEEDS
        }
        for condition in CONDITIONS
    }
    aggregate = {
        condition: aggregate_condition(
            condition,
            trained,
            seed_summaries[condition],
        )
        for condition in CONDITIONS
    }
    base_positive = score_positive(base["positive_rows"])
    comparisons = {
        f"typed_vs_{comparator}": comparison(comparator, base, trained)
        for comparator in ("base", "adapter", "igc_matched", "igc_native")
    }
    native_comparison = comparisons["typed_vs_igc_native"]
    native_interval = native_comparison[
        "crossed_seed_prompt_bootstrap_95_percentage_points"
    ]
    typed_accuracy = aggregate["typed_firmware"]["seed_mathematical_accuracy"]["mean"]
    native_accuracy = aggregate["igc_native"]["seed_mathematical_accuracy"]["mean"]
    typed_preservation = aggregate["typed_firmware"][
        "seed_negative_token_preservation_rate"
    ]["mean"]
    native_preservation = aggregate["igc_native"][
        "seed_negative_token_preservation_rate"
    ]["mean"]
    typed_parameters = aggregate["typed_firmware"]["learned_parameters"]
    native_parameters = aggregate["igc_native"]["learned_parameters"]
    decisions = {
        "typed_and_native_comparably_reliable": (
            native_interval[0] >= -5 and native_interval[1] <= 5
        ),
        "typed_accuracy_no_more_than_5pp_below_native": (
            typed_accuracy >= native_accuracy - 0.05
        ),
        "typed_uses_at_most_one_tenth_native_parameters": (
            typed_parameters <= native_parameters / 10
        ),
        "typed_preservation_no_more_than_1pp_below_native": (
            typed_preservation >= native_preservation - 0.01
        ),
    }
    decisions["typed_parameter_efficiency_advantage"] = all(
        decisions[key]
        for key in (
            "typed_accuracy_no_more_than_5pp_below_native",
            "typed_uses_at_most_one_tenth_native_parameters",
            "typed_preservation_no_more_than_1pp_below_native",
        )
    )
    operational_gates = {
        condition: {
            "routing_success": (
                aggregate[condition]["seed_positive_route_rate"]["mean"] >= 0.90
                and aggregate[condition]["seed_negative_false_route_rate"]["mean"]
                <= 0.02
            ),
            "preservation_success": (
                aggregate[condition]["seed_negative_token_preservation_rate"]["mean"]
                >= 0.99
            ),
        }
        for condition in CONDITIONS
    }
    raw_hashes = {
        path.name: sha256(path)
        for path in sorted(RAW_DIRECTORY.glob("*.json"))
    }
    analysis = {
        "analysis_version": 1,
        "base": {
            "learned_parameters": 0,
            "positive": base_positive,
            "positive_latency": base["summary"]["positive_latency"],
            "negative_latency": base["summary"]["negative_latency"],
        },
        "per_seed": {
            condition: {
                str(seed): seed_summaries[condition][seed] for seed in SEEDS
            }
            for condition in CONDITIONS
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "precommitted_decisions": decisions,
        "operational_gates": operational_gates,
        "parameter_ratios": {
            "native_igc_over_typed": native_parameters / typed_parameters,
            "matched_igc_over_typed": (
                aggregate["igc_matched"]["learned_parameters"] / typed_parameters
            ),
        },
        "raw_file_sha256": raw_hashes,
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n")
    write_summary_csv(base, trained, seed_summaries)
    write_comparison_csv(comparisons)
    family_rows = build_family_rows(base, trained)
    with FAMILY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    render_figures(base, aggregate)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
