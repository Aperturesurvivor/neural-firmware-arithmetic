from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SPLITS = ["id_1_4", "ood_primary_5_8", "ood_long_9_12", "carry_chain"]
SPLIT_LABELS = {
    "id_1_4": "ID\n1–4 digits",
    "ood_primary_5_8": "Primary OOD\n5–8 digits",
    "ood_long_9_12": "Long OOD\n9–12 digits",
    "carry_chain": "Carry chain\n5–12 digits",
}
METHOD_LABELS = {
    "base": "Frozen base",
    "learned_control": "Matched learned adapter",
    "internal": "Internal deterministic unit",
}
COLORS = {
    "base": "#6B7280",
    "learned_control": "#D9822B",
    "internal": "#2F6B9A",
    "criterion": "#48794A",
    "ink": "#263238",
    "grid": "#D1D5DB",
}
HATCHES = {"base": "//", "learned_control": "\\\\", "internal": ""}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    draws: int = 100_000,
    seed: int = 20_260_726,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    return tuple(float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975]))


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


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(rows: list[dict[str, object]], directory: Path) -> None:
    methods = list(METHOD_LABELS)
    lookup = {(row["method"], row["split"]): row for row in rows}
    x = np.arange(len(SPLITS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for index, method in enumerate(methods):
        values = [100 * float(lookup[(method, split)]["mean_accuracy"]) for split in SPLITS]
        errors = [100 * float(lookup[(method, split)]["sample_sd"]) for split in SPLITS]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor=COLORS["ink"],
            linewidth=0.55,
            hatch=HATCHES[method],
            yerr=errors if any(errors) else None,
            capsize=2,
        )
        for bar, value in zip(bars, values, strict=True):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.8,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_title(
        "Sequence-level exact-match accuracy by evaluation split",
        loc="left",
        pad=29,
    )
    ax.text(
        0,
        1.015,
        "Mean across three training seeds; error bars are sample SD",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=9,
    )
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_xticks(x, [SPLIT_LABELS[split] for split in SPLITS])
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19), frameon=False)
    fig.subplots_adjust(bottom=0.26)
    save_figure(fig, directory, "accuracy_by_split")


def plot_seed_pairs(study: dict[str, object], directory: Path) -> None:
    controls = {int(run["seed"]): run for run in study["control_runs"]}
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    offsets = [-0.025, 0.0, 0.025]
    line_styles = ["-", "--", ":"]
    for index, run in enumerate(study["internal_runs"]):
        seed = int(run["seed"])
        control = float(
            controls[seed]["control_result"]["splits"]["ood_primary_5_8"][
                "exact_match_accuracy"
            ]
        )
        internal = float(
            run["internal_result"]["splits"]["ood_primary_5_8"]["exact_match_accuracy"]
        )
        positions = [offsets[index], 1 + offsets[index]]
        ax.plot(
            positions,
            [100 * control, 100 * internal],
            color=["#64748B", "#94A3B8", "#475569"][index],
            linewidth=1.5,
            linestyle=line_styles[index],
            label=f"seed {seed}",
        )
        ax.scatter(
            positions[0],
            100 * control,
            color=COLORS["learned_control"],
            marker="s",
            s=62,
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            zorder=2,
        )
        ax.scatter(
            positions[1],
            100 * internal,
            color=COLORS["internal"],
            marker="o",
            s=62,
            edgecolor=COLORS["ink"],
            linewidth=0.5,
            zorder=2,
        )
    ax.axhline(99, color=COLORS["criterion"], linestyle="--", linewidth=1.1)
    ax.text(1.17, 99, "99% criterion", ha="right", va="bottom", fontsize=8)
    ax.set_title("Primary OOD exact match for each training seed", loc="left", pad=29)
    ax.text(
        0,
        1.015,
        "Identical 18,826-parameter interface budget at transformer block 6",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=9,
    )
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_xticks([0, 1], ["Matched learned\nadapter", "Internal deterministic\nunit"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-3, 108)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.legend(loc="lower right", frameon=False, title="Paired runs")
    save_figure(fig, directory, "primary_ood_per_seed")


def trace_rows(study: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    ranks: dict[int, list[int]] = defaultdict(list)
    for run in study["internal_runs"]:
        for trace in run["traces"]:
            for layer in trace["layers"]:
                depth = int(layer["depth_after_blocks"])
                grouped[depth].append(float(layer["target_margin"]))
                ranks[depth].append(int(layer["target_rank"]))
    rows = []
    for depth in sorted(grouped):
        margins = np.array(grouped[depth])
        rows.append(
            {
                "depth_after_blocks": depth,
                "observations": len(margins),
                "mean_target_margin": float(margins.mean()),
                "sample_sd": float(margins.std(ddof=1)),
                "minimum_target_margin": float(margins.min()),
                "maximum_target_rank": max(ranks[depth]),
                "top1_rate": float(np.mean(np.array(ranks[depth]) == 1)),
            }
        )
    return rows


def plot_trace(rows: list[dict[str, object]], directory: Path) -> None:
    depths = np.array([int(row["depth_after_blocks"]) for row in rows])
    means = np.array([float(row["mean_target_margin"]) for row in rows])
    sds = np.array([float(row["sample_sd"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.fill_between(depths, means - sds, means + sds, color="#B9D5E8", alpha=0.55)
    ax.plot(
        depths,
        means,
        color=COLORS["internal"],
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    ax.axhline(0, color=COLORS["ink"], linewidth=0.9, linestyle="--")
    ax.axvline(6, color=COLORS["criterion"], linewidth=1.1, linestyle=":")
    ax.text(6.2, min(means) + 2, "deterministic unit", color=COLORS["criterion"], fontsize=8)
    ax.set_title(
        "Correct-token logit margin through downstream transformer blocks",
        loc="left",
        pad=29,
    )
    ax.text(
        0,
        1.015,
        "Mean ± sample SD over 30 traces (10 examples × 3 seeds); block 6 includes the unit",
        transform=ax.transAxes,
        color="#4B5563",
        fontsize=9,
    )
    ax.set_xlabel("Depth after transformer blocks")
    ax.set_ylabel("Correct digit logit − strongest alternative")
    ax.set_xticks([5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    save_figure(fig, directory, "downstream_logit_margin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("phase3_artifacts/confirmatory_v1/study.json"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("phase3_results/confirmatory_v1"),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("phase3_figures"),
    )
    args = parser.parse_args()
    study = read_json(args.study)
    internal_runs = study["internal_runs"]
    control_runs = study["control_runs"]

    per_seed_rows: list[dict[str, object]] = []
    for method, runs, result_key in (
        ("internal", internal_runs, "internal_result"),
        ("learned_control", control_runs, "control_result"),
    ):
        for run in runs:
            for split in SPLITS:
                result = run[result_key]["splits"][split]
                per_seed_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": run["seed"],
                        "split": split,
                        "correct": result["correct"],
                        "examples": result["examples"],
                        "exact_match_accuracy": result["exact_match_accuracy"],
                        "mean_latency_seconds": float(
                            np.mean(
                                [
                                    float(prediction["latency_seconds"])
                                    for prediction in result["predictions"]
                                ]
                            )
                        ),
                    }
                )
    write_csv(args.result_directory / "accuracy_per_seed.csv", per_seed_rows)

    accuracy_rows: list[dict[str, object]] = []
    for method in METHOD_LABELS:
        for split in SPLITS:
            if method == "base":
                values = [study["base_result"]["splits"][split]]
            else:
                values = [
                    row
                    for row in per_seed_rows
                    if row["method"] == method and row["split"] == split
                ]
            accuracies = np.array(
                [
                    float(value["exact_match_accuracy"])
                    for value in values
                ]
            )
            accuracy_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "split": split,
                    "split_label": SPLIT_LABELS[split].replace("\n", " "),
                    "seed_runs": len(values),
                    "correct_total": sum(int(value["correct"]) for value in values),
                    "examples_total": sum(int(value["examples"]) for value in values),
                    "mean_accuracy": float(accuracies.mean()),
                    "sample_sd": (
                        float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0
                    ),
                }
            )
    write_csv(args.result_directory / "accuracy_summary.csv", accuracy_rows)

    register_rows: list[dict[str, object]] = []
    preservation_rows: list[dict[str, object]] = []
    intervention_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    off_rows: list[dict[str, object]] = []
    for run in internal_runs:
        seed = int(run["seed"])
        for split in SPLITS:
            result = run["register_evaluation"][split]
            register_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "exact_registers": result["exact_registers"],
                    "examples": result["examples"],
                    "exact_register_accuracy": result["exact_register_accuracy"],
                }
            )
        preservation_rows.append(
            {
                "seed": seed,
                "preserved": run["preservation"]["preserved"],
                "prompts": len(run["preservation"]["comparisons"]),
                "preservation_rate": run["preservation"]["preservation_rate"],
            }
        )
        for kind in ("wrong_state", "state_substitution"):
            result = run["interventions"][kind]
            intervention_rows.append(
                {
                    "seed": seed,
                    "intervention": kind,
                    "causal_matches": result["causal_matches"],
                    "examples": result["examples"],
                    "causal_rate": result["causal_matches"] / result["examples"],
                }
            )
        off = run["unit_off"]["splits"]["ood_primary_5_8"]
        off_rows.append(
            {
                "seed": seed,
                "correct": off["correct"],
                "examples": off["examples"],
                "exact_match_accuracy": off["exact_match_accuracy"],
            }
        )
        for component in ("encoder_train", "decoder_train"):
            result = run[component]
            training_rows.append(
                {
                    "method": component,
                    "seed": seed,
                    "trainable_parameters": result["trainable_parameters"],
                    "steps": result.get("steps", result.get("config", {}).get("steps")),
                    "initial_loss": result["initial_loss"],
                    "final_loss": result["final_loss"],
                    "wall_time_seconds": result["wall_time_seconds"],
                    "training_sha256": run["training_sha256"],
                }
            )
    for run in control_runs:
        result = run["control_train"]
        training_rows.append(
            {
                "method": "learned_control",
                "seed": run["seed"],
                "trainable_parameters": result["trainable_parameters"],
                "steps": result["config"]["steps"],
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "wall_time_seconds": result["wall_time_seconds"],
                "training_sha256": run["training_sha256"],
            }
        )
    write_csv(args.result_directory / "register_accuracy_per_seed.csv", register_rows)
    write_csv(args.result_directory / "preservation_per_seed.csv", preservation_rows)
    write_csv(args.result_directory / "interventions_per_seed.csv", intervention_rows)
    write_csv(args.result_directory / "unit_off_per_seed.csv", off_rows)
    write_csv(args.result_directory / "training_per_seed.csv", training_rows)

    traces = trace_rows(study)
    write_csv(args.result_directory / "downstream_logit_margin.csv", traces)
    post_unit = [row for row in traces if int(row["depth_after_blocks"]) >= 6]
    minimum_post_unit_mean_margin = min(
        float(row["mean_target_margin"]) for row in post_unit
    )

    internal_primary = {
        int(run["seed"]): float(
            run["internal_result"]["splits"]["ood_primary_5_8"]["exact_match_accuracy"]
        )
        for run in internal_runs
    }
    control_primary = {
        int(run["seed"]): float(
            run["control_result"]["splits"]["ood_primary_5_8"]["exact_match_accuracy"]
        )
        for run in control_runs
    }
    paired_differences = np.array(
        [
            internal_primary[seed] - control_primary[seed]
            for seed in sorted(internal_primary)
        ]
    )
    ci_lower, ci_upper = bootstrap_mean_ci(paired_differences)
    internal_primary_mean = float(np.mean(list(internal_primary.values())))
    control_primary_mean = float(np.mean(list(control_primary.values())))
    difference_mean = float(paired_differences.mean())
    primary_register_rows = [
        row for row in register_rows if row["split"] == "ood_primary_5_8"
    ]
    register_mean = float(
        np.mean(
            [
                float(row["exact_register_accuracy"])
                for row in primary_register_rows
            ]
        )
    )
    preservation_rate = sum(int(row["preserved"]) for row in preservation_rows) / sum(
        int(row["prompts"]) for row in preservation_rows
    )
    wrong_rows = [
        row for row in intervention_rows if row["intervention"] == "wrong_state"
    ]
    substitution_rows = [
        row for row in intervention_rows if row["intervention"] == "state_substitution"
    ]
    wrong_rate = sum(int(row["causal_matches"]) for row in wrong_rows) / sum(
        int(row["examples"]) for row in wrong_rows
    )
    substitution_rate = sum(
        int(row["causal_matches"]) for row in substitution_rows
    ) / sum(int(row["examples"]) for row in substitution_rows)
    off_accuracy = float(np.mean([float(row["exact_match_accuracy"]) for row in off_rows]))
    unit_off_drop = internal_primary_mean - off_accuracy
    thresholds = study["config"]["success_thresholds"]
    criteria_rows = [
        {
            "criterion": "internal_primary_mean_accuracy",
            "observed": internal_primary_mean,
            "comparison": ">=",
            "threshold": thresholds["internal_primary_mean_accuracy"],
            "passed": internal_primary_mean
            >= thresholds["internal_primary_mean_accuracy"],
        },
        {
            "criterion": "internal_minus_control_primary_points",
            "observed": difference_mean,
            "comparison": ">=",
            "threshold": thresholds["internal_minus_control_primary_points"],
            "passed": difference_mean
            >= thresholds["internal_minus_control_primary_points"],
        },
        {
            "criterion": "exact_register_mean_accuracy",
            "observed": register_mean,
            "comparison": ">=",
            "threshold": thresholds["exact_register_mean_accuracy"],
            "passed": register_mean >= thresholds["exact_register_mean_accuracy"],
        },
        {
            "criterion": "language_preservation_rate",
            "observed": preservation_rate,
            "comparison": ">=",
            "threshold": thresholds["language_preservation_rate"],
            "passed": preservation_rate >= thresholds["language_preservation_rate"],
        },
        {
            "criterion": "wrong_state_causal_rate",
            "observed": wrong_rate,
            "comparison": ">=",
            "threshold": thresholds["wrong_state_causal_rate"],
            "passed": wrong_rate >= thresholds["wrong_state_causal_rate"],
        },
        {
            "criterion": "state_substitution_causal_rate",
            "observed": substitution_rate,
            "comparison": ">=",
            "threshold": thresholds["state_substitution_causal_rate"],
            "passed": substitution_rate >= thresholds["state_substitution_causal_rate"],
        },
        {
            "criterion": "unit_off_primary_drop_points",
            "observed": unit_off_drop,
            "comparison": ">=",
            "threshold": thresholds["unit_off_primary_drop_points"],
            "passed": unit_off_drop >= thresholds["unit_off_primary_drop_points"],
        },
        {
            "criterion": "minimum_post_unit_mean_logit_margin",
            "observed": minimum_post_unit_mean_margin,
            "comparison": ">",
            "threshold": thresholds["minimum_post_unit_mean_logit_margin"],
            "passed": minimum_post_unit_mean_margin
            > thresholds["minimum_post_unit_mean_logit_margin"],
        },
    ]
    write_csv(args.result_directory / "success_criteria.csv", criteria_rows)

    error_inventory: list[dict[str, object]] = []
    for method, runs, key in (
        ("internal", internal_runs, "internal_result"),
        ("learned_control", control_runs, "control_result"),
    ):
        for run in runs:
            for split in SPLITS:
                result = run[key]["splits"][split]
                error_inventory.append(
                    {
                        "method": method,
                        "seed": run["seed"],
                        "split": split,
                        "errors": int(result["examples"]) - int(result["correct"]),
                        "examples": result["examples"],
                    }
                )
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
            "internal_accuracy_by_seed": internal_primary,
            "control_accuracy_by_seed": control_primary,
            "paired_difference_by_seed": {
                str(seed): internal_primary[seed] - control_primary[seed]
                for seed in sorted(internal_primary)
            },
            "mean_internal_accuracy": internal_primary_mean,
            "mean_control_accuracy": control_primary_mean,
            "mean_paired_difference": difference_mean,
            "bootstrap_draws": 100_000,
            "bootstrap_seed": 20_260_726,
            "bootstrap_95_percentile_ci": [ci_lower, ci_upper],
            "bootstrap_unit": "training seed",
        },
        "pooled_counts": {
            "internal_arithmetic_correct": sum(
                int(run["internal_result"]["splits"][split]["correct"])
                for run in internal_runs
                for split in SPLITS
            ),
            "internal_arithmetic_examples": sum(
                int(run["internal_result"]["splits"][split]["examples"])
                for run in internal_runs
                for split in SPLITS
            ),
            "exact_registers": sum(int(row["exact_registers"]) for row in register_rows),
            "register_examples": sum(int(row["examples"]) for row in register_rows),
            "preserved_outputs": sum(int(row["preserved"]) for row in preservation_rows),
            "preservation_prompts": sum(int(row["prompts"]) for row in preservation_rows),
            "wrong_state_matches": sum(int(row["causal_matches"]) for row in wrong_rows),
            "wrong_state_examples": sum(int(row["examples"]) for row in wrong_rows),
            "substitution_matches": sum(
                int(row["causal_matches"]) for row in substitution_rows
            ),
            "substitution_examples": sum(
                int(row["examples"]) for row in substitution_rows
            ),
            "unit_off_primary_correct": sum(int(row["correct"]) for row in off_rows),
            "unit_off_primary_examples": sum(int(row["examples"]) for row in off_rows),
        },
        "trace": {
            "trace_examples": sum(len(run["traces"]) for run in internal_runs),
            "minimum_post_unit_mean_margin": minimum_post_unit_mean_margin,
            "minimum_depth": int(
                min(
                    post_unit,
                    key=lambda row: float(row["mean_target_margin"]),
                )["depth_after_blocks"]
            ),
            "all_post_unit_mean_margins_positive": minimum_post_unit_mean_margin > 0,
        },
        "success_criteria": criteria_rows,
        "overall_preregistered_success": all(bool(row["passed"]) for row in criteria_rows),
    }
    write_json(args.result_directory / "analysis.json", analysis)

    set_plot_style()
    plot_accuracy(accuracy_rows, args.figure_directory)
    plot_seed_pairs(study, args.figure_directory)
    plot_trace(traces, args.figure_directory)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
