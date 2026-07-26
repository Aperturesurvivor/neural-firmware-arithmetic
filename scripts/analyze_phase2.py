from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_LABELS = {
    "base": "Frozen base",
    "firmware_off": "Firmware off",
    "learned_adapter": "Learned LoRA",
    "latent": "Latent firmware",
    "direct": "Direct firmware",
}
SPLIT_LABELS = {
    "id_random": "ID\n1–4 digits",
    "ood_primary": "Primary OOD\n5–8 digits",
    "ood_long": "Long OOD\n9–12 digits",
    "carry_chain": "Carry chain\n5–12 digits",
}
METHOD_COLORS = {
    "base": "#6B7280",
    "firmware_off": "#9CA3AF",
    "learned_adapter": "#D9822B",
    "latent": "#2F6B9A",
    "direct": "#48794A",
}
METHOD_HATCHES = {
    "base": "//",
    "firmware_off": "..",
    "learned_adapter": "\\\\",
    "latent": "",
    "direct": "xx",
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries_by_split(summaries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["split"]): row for row in summaries}


def metric_row(
    method: str,
    split: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    accuracies = np.array([float(row["exact_match_accuracy"]) for row in rows])
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "split": split,
        "split_label": SPLIT_LABELS[split].replace("\n", " "),
        "seed_runs": len(rows),
        "correct_total": sum(int(row["correct"]) for row in rows),
        "examples_total": sum(int(row["examples"]) for row in rows),
        "mean_accuracy": float(accuracies.mean()),
        "sample_sd": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "mean_latency_seconds": float(
            np.mean([float(row["mean_latency_seconds"]) for row in rows])
        ),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    draws: int = 100_000,
    seed: int = 20_260_725,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True)
    means = sampled.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, figure_directory: Path, stem: str) -> None:
    figure_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_directory / f"{stem}.png", bbox_inches="tight")
    fig.savefig(figure_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(
    rows: list[dict[str, object]],
    figure_directory: Path,
) -> None:
    methods = ["base", "learned_adapter", "latent", "direct"]
    splits = list(SPLIT_LABELS)
    x = np.arange(len(splits))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    lookup = {(row["method"], row["split"]): row for row in rows}
    for index, method in enumerate(methods):
        values = [100 * float(lookup[(method, split)]["mean_accuracy"]) for split in splits]
        errors = [100 * float(lookup[(method, split)]["sample_sd"]) for split in splits]
        positions = x + (index - 1.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="#263238",
            linewidth=0.5,
            hatch=METHOD_HATCHES[method],
            yerr=errors if any(errors) else None,
            capsize=2,
        )
        for bar, value in zip(bars, values, strict=True):
            if value >= 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.6,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    ax.set_title("Sequence-level exact-match accuracy by evaluation split")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_xticks(x, [SPLIT_LABELS[split] for split in splits])
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, figure_directory, "accuracy_by_split")


def plot_primary_seed_pairs(
    bridge_runs: list[dict[str, object]],
    adapter_runs: list[dict[str, object]],
    figure_directory: Path,
) -> None:
    adapter_by_seed = {int(run["seed"]): run for run in adapter_runs}
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for bridge_run in bridge_runs:
        seed = int(bridge_run["seed"])
        latent = float(
            summaries_by_split(bridge_run["latent_summaries"])["ood_primary"][
                "exact_match_accuracy"
            ]
        )
        adapter = float(
            summaries_by_split(adapter_by_seed[seed]["adapter_summaries"])["ood_primary"][
                "exact_match_accuracy"
            ]
        )
        ax.plot(
            [0, 1],
            [100 * adapter, 100 * latent],
            color="#94A3B8",
            linewidth=1.5,
            zorder=1,
        )
        ax.scatter(
            0,
            100 * adapter,
            color=METHOD_COLORS["learned_adapter"],
            marker="s",
            s=60,
            edgecolor="#263238",
            linewidth=0.5,
            zorder=2,
        )
        ax.scatter(
            1,
            100 * latent,
            color=METHOD_COLORS["latent"],
            marker="o",
            s=60,
            edgecolor="#263238",
            linewidth=0.5,
            zorder=2,
        )
        ax.text(0.04, 100 * adapter, f"seed {seed}", va="center", fontsize=8)
    ax.axhline(99, color="#48794A", linestyle="--", linewidth=1, label="99% criterion")
    ax.set_title("Primary OOD exact match for each training seed")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_xticks([0, 1], ["Learned LoRA", "Latent firmware"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0, 106)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, figure_directory, "primary_ood_per_seed")


def plot_preservation(
    preservation_rows: list[dict[str, object]],
    figure_directory: Path,
) -> None:
    methods = ["learned_adapter", "latent"]
    values = []
    errors = []
    for method in methods:
        method_values = np.array(
            [
                float(row["preservation_rate"])
                for row in preservation_rows
                if row["method"] == method
            ]
        )
        values.append(100 * float(method_values.mean()))
        errors.append(100 * float(method_values.std(ddof=1)))
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    bars = ax.bar(
        np.arange(2),
        values,
        width=0.58,
        color=[METHOD_COLORS[method] for method in methods],
        edgecolor="#263238",
        linewidth=0.6,
        hatch=[METHOD_HATCHES[method] for method in methods],
        yerr=errors,
        capsize=3,
    )
    ax.axhline(99, color="#48794A", linestyle="--", linewidth=1.2, label="99% criterion")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Token-exact preservation on 100 control prompts")
    ax.set_ylabel("Preserved outputs (%)")
    ax.set_xticks(np.arange(2), ["Learned LoRA", "Latent firmware"])
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, figure_directory, "language_preservation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1/study.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("phase2_results/confirmatory_v1"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("phase2_figures"),
    )
    args = parser.parse_args()
    study = read_json(args.study)
    bridge_runs = study["bridge_runs"]
    adapter_runs = study["adapter_runs"]
    splits = list(SPLIT_LABELS)

    method_summaries: dict[str, dict[str, list[dict[str, object]]]] = {
        method: {split: [] for split in splits}
        for method in METHOD_LABELS
    }
    for summary in study["base_summaries"]:
        method_summaries["base"][summary["split"]].append(summary)
    for summary in study["direct_summaries"]:
        method_summaries["direct"][summary["split"]].append(summary)
    for run in bridge_runs:
        for summary in run["latent_summaries"]:
            method_summaries["latent"][summary["split"]].append(summary)
        for summary in run["firmware_off_summaries"]:
            method_summaries["firmware_off"][summary["split"]].append(summary)
    for run in adapter_runs:
        for summary in run["adapter_summaries"]:
            method_summaries["learned_adapter"][summary["split"]].append(summary)

    accuracy_rows = [
        metric_row(method, split, method_summaries[method][split])
        for method in METHOD_LABELS
        for split in splits
    ]
    write_csv(args.result_directory / "accuracy_summary.csv", accuracy_rows)

    per_seed_rows: list[dict[str, object]] = []
    for method, runs, field in (
        ("latent", bridge_runs, "latent_summaries"),
        ("firmware_off", bridge_runs, "firmware_off_summaries"),
        ("learned_adapter", adapter_runs, "adapter_summaries"),
    ):
        for run in runs:
            for summary in run[field]:
                per_seed_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": run["seed"],
                        "split": summary["split"],
                        "correct": summary["correct"],
                        "examples": summary["examples"],
                        "exact_match_accuracy": summary["exact_match_accuracy"],
                        "mean_latency_seconds": summary["mean_latency_seconds"],
                    }
                )
    write_csv(args.result_directory / "accuracy_per_seed.csv", per_seed_rows)

    preservation_rows: list[dict[str, object]] = []
    for method, runs in (("latent", bridge_runs), ("learned_adapter", adapter_runs)):
        for run in runs:
            summary = run["preservation_summary"]
            preservation_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": run["seed"],
                    "preserved": summary["token_exact_preserved"],
                    "prompts": summary["prompts"],
                    "preservation_rate": summary["preservation_rate"],
                    "initial_false_routes": summary.get("initial_false_routes", ""),
                    "initial_false_route_rate": summary.get("initial_false_route_rate", ""),
                }
            )
    write_csv(args.result_directory / "preservation_per_seed.csv", preservation_rows)

    training_rows: list[dict[str, object]] = []
    for method, runs in (("latent", bridge_runs), ("learned_adapter", adapter_runs)):
        for run in runs:
            result = run["train_result"]
            training_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": run["seed"],
                    "trainable_parameters": result["trainable_parameters"],
                    "initial_loss": result["initial_loss"],
                    "final_loss": result["final_loss"],
                    "wall_time_seconds": result["wall_time_seconds"],
                    "training_sha256": run["training_sha256"],
                }
            )
    write_csv(args.result_directory / "training_per_seed.csv", training_rows)

    adapter_primary = {
        int(run["seed"]): float(
            summaries_by_split(run["adapter_summaries"])["ood_primary"][
                "exact_match_accuracy"
            ]
        )
        for run in adapter_runs
    }
    latent_primary = {
        int(run["seed"]): float(
            summaries_by_split(run["latent_summaries"])["ood_primary"][
                "exact_match_accuracy"
            ]
        )
        for run in bridge_runs
    }
    differences = np.array(
        [
            latent_primary[seed] - adapter_primary[seed]
            for seed in sorted(latent_primary)
        ]
    )
    bootstrap_lower, bootstrap_upper = bootstrap_mean_ci(differences)

    latent_preserved = sum(
        int(run["preservation_summary"]["token_exact_preserved"]) for run in bridge_runs
    )
    latent_preservation_prompts = sum(
        int(run["preservation_summary"]["prompts"]) for run in bridge_runs
    )
    false_routes = sum(
        int(run["preservation_summary"]["initial_false_routes"]) for run in bridge_runs
    )
    thresholds = study["config"]["success_thresholds"]
    primary_mean = float(np.mean(list(latent_primary.values())))
    difference_mean = float(differences.mean())
    preservation_rate = latent_preserved / latent_preservation_prompts
    false_route_rate = false_routes / latent_preservation_prompts
    criteria_rows = [
        {
            "criterion": "latent_ood_primary_mean_accuracy",
            "observed": primary_mean,
            "threshold": thresholds["latent_ood_primary_mean_accuracy"],
            "comparison": ">=",
            "passed": primary_mean >= thresholds["latent_ood_primary_mean_accuracy"],
        },
        {
            "criterion": "latent_minus_adapter_ood_primary_points",
            "observed": difference_mean,
            "threshold": thresholds["latent_minus_adapter_ood_primary_points"],
            "comparison": ">=",
            "passed": difference_mean
            >= thresholds["latent_minus_adapter_ood_primary_points"],
        },
        {
            "criterion": "latent_language_preservation_rate",
            "observed": preservation_rate,
            "threshold": thresholds["latent_language_preservation_rate"],
            "comparison": ">=",
            "passed": preservation_rate
            >= thresholds["latent_language_preservation_rate"],
        },
        {
            "criterion": "latent_initial_false_route_rate",
            "observed": false_route_rate,
            "threshold": thresholds["latent_initial_false_route_rate"],
            "comparison": "<=",
            "passed": false_route_rate <= thresholds["latent_initial_false_route_rate"],
        },
    ]
    write_csv(args.result_directory / "success_criteria.csv", criteria_rows)

    error_rows: list[dict[str, object]] = []
    error_inventory: list[dict[str, object]] = []
    for run in bridge_runs:
        seed = int(run["seed"])
        predictions = read_json(
            args.artifact_directory / f"bridge_seed_{seed}" / "latent" / "predictions.json"
        )
        arithmetic_errors = [row for row in predictions if not row["exact"]]
        by_split = Counter(row["split"] for row in arithmetic_errors)
        for split in splits:
            error_inventory.append(
                {
                    "method": "latent",
                    "seed": seed,
                    "error_type": "arithmetic_exact_match",
                    "split": split,
                    "count": by_split[split],
                }
            )
        for row in arithmetic_errors:
            if row["split"] == "ood_long" and float(row["route_probabilities"][0]) < 0.5:
                classification = "initial_router_underactivation"
            elif row["generated_text"].startswith(row["expected"]):
                classification = "end_of_answer_decode_failure"
            else:
                classification = "other_latent_decode_failure"
            error_rows.append(
                {
                    "error_family": "latent_arithmetic",
                    "classification": classification,
                    "seed": seed,
                    "split": row["split"],
                    "prompt": row["prompt"],
                    "expected_or_base": row["expected"],
                    "observed": row["generated_text"],
                    "initial_route_probability": row["route_probabilities"][0],
                    "minimum_route_probability": min(row["route_probabilities"]),
                }
            )

        preservation = read_json(
            args.artifact_directory
            / f"bridge_seed_{seed}"
            / "preservation_predictions.json"
        )
        preservation_errors = [
            row for row in preservation if not row["token_exact_preserved"]
        ]
        error_inventory.append(
            {
                "method": "latent",
                "seed": seed,
                "error_type": "language_preservation",
                "split": "routing_controls",
                "count": len(preservation_errors),
            }
        )
        for row in preservation_errors:
            error_rows.append(
                {
                    "error_family": "latent_preservation",
                    "classification": "late_activation_on_quoted_registered_prompt",
                    "seed": seed,
                    "split": "routing_controls",
                    "prompt": row["prompt"],
                    "expected_or_base": row["base_text"],
                    "observed": row["latent_text"],
                    "initial_route_probability": row["initial_route_probability"],
                    "minimum_route_probability": "",
                }
            )
    write_csv(args.result_directory / "latent_errors.csv", error_rows)
    write_csv(args.result_directory / "error_inventory.csv", error_inventory)

    analysis = {
        "source": {
            "study": str(args.study),
            "source_commit": study["frozen_state"]["source_commit"],
            "config_sha256": study["frozen_state"]["config_sha256"],
            "evaluation_sha256": study["evaluation_sha256"],
            "model_id": study["model_id"],
            "model_revision": study["model_revision"],
        },
        "primary_endpoint": {
            "latent_accuracy_by_seed": latent_primary,
            "adapter_accuracy_by_seed": adapter_primary,
            "paired_difference_by_seed": {
                str(seed): latent_primary[seed] - adapter_primary[seed]
                for seed in sorted(latent_primary)
            },
            "mean_latent_accuracy": primary_mean,
            "mean_adapter_accuracy": float(np.mean(list(adapter_primary.values()))),
            "mean_paired_difference": difference_mean,
            "bootstrap_draws": 100_000,
            "bootstrap_seed": 20_260_725,
            "bootstrap_95_percentile_ci": [bootstrap_lower, bootstrap_upper],
            "bootstrap_unit": "training seed",
        },
        "preservation": {
            "latent_preserved": latent_preserved,
            "latent_prompts": latent_preservation_prompts,
            "latent_rate": preservation_rate,
            "initial_false_routes": false_routes,
            "initial_false_route_rate": false_route_rate,
        },
        "success_criteria": criteria_rows,
        "overall_preregistered_success": all(row["passed"] for row in criteria_rows),
        "error_counts": {
            "latent_arithmetic": sum(
                row["error_family"] == "latent_arithmetic" for row in error_rows
            ),
            "latent_preservation": sum(
                row["error_family"] == "latent_preservation" for row in error_rows
            ),
        },
    }
    write_json(args.result_directory / "analysis.json", analysis)

    set_plot_style()
    plot_accuracy(accuracy_rows, args.figure_directory)
    plot_primary_seed_pairs(bridge_runs, adapter_runs, args.figure_directory)
    plot_preservation(preservation_rows, args.figure_directory)

    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
