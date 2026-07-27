from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("phase8_results/confirmation.json")
OUTPUT_DIRECTORY = Path("phase8_figures")


def save(figure: object, stem: str) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIRECTORY / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIRECTORY / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    seeds = [str(seed) for seed in payload["training_seeds"]]

    labels = ["Base"] + [f"Adapter\n{s}" for s in seeds] + [
        f"Implant\n{s}" for s in seeds
    ] + ["Ablated\n(mean)"]
    values = [payload["base"]["format_exact"] / 60]
    values.extend(
        payload["per_seed"][seed]["matched_adapter_exact"] / 60
        for seed in seeds
    )
    values.extend(
        payload["per_seed"][seed]["implant_exact"] / 60 for seed in seeds
    )
    values.append(
        np.mean(
            [
                payload["per_seed"][seed]["ablation_exact"] / 60
                for seed in seeds
            ]
        )
    )
    colors = (
        ["#64748b"]
        + ["#d97706"] * 3
        + ["#2563eb"] * 3
        + ["#b91c1c"]
    )
    figure, axis = plt.subplots(figsize=(10.4, 4.8))
    bars = axis.bar(labels, np.array(values) * 100, color=colors)
    axis.axhline(
        95,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="Frozen per-seed gate (57/60)",
    )
    axis.set_ylim(0, 104)
    axis.set_ylabel("Exact numeral-only accuracy (%)")
    axis.set_title("Phase 8 frozen TinyLlama confirmation")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + 2,
            f"{value * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    figure.tight_layout()
    save(figure, "condition_accuracy")

    categories = [
        "Quoted\narithmetic",
        "Negated\nrequest",
        "Multiplication\nnear-miss",
        "Factual\nnumbers",
        "Ignore\nsum",
    ]
    split_names = [
        "phase8_confirmatory_negative_quoted_arithmetic",
        "phase8_confirmatory_negative_negated_request",
        "phase8_confirmatory_negative_multiplication_near_miss",
        "phase8_confirmatory_negative_factual_numbers",
        "phase8_confirmatory_negative_ignore_embedded_sum",
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(categories))
    width = 0.22
    for index, seed in enumerate(seeds):
        values = []
        for split in split_names:
            group = [
                row for row in payload["rows"] if row["split"] == split
            ]
            values.append(
                sum(row["implants"][seed]["first_route"] for row in group)
            )
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"seed {seed}",
        )
    axis.axhline(
        2,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.2,
        label="Whole-audit gate (2/60)",
    )
    axis.set_xticks(x, categories)
    axis.set_ylabel("False routes out of 12 prompts")
    axis.set_ylim(0, 12.7)
    axis.set_title("Systematic adversarial routing failures")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(figure, "negative_false_routes")

    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    labels = ["Route off", "Operand handshake", "Conditional decode"]
    values = [1, 6, 0]
    bars = axis.bar(labels, values, color=["#64748b", "#d97706", "#2563eb"])
    axis.set_ylim(0, 7)
    axis.set_ylabel("Failed additions per seed")
    axis.set_title("Where the seven shared implant failures occurred")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            str(value),
            ha="center",
        )
    figure.tight_layout()
    save(figure, "positive_failure_taxonomy")


if __name__ == "__main__":
    main()
