from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("phase9_results/confirmation.json")
ANALYSIS = Path("phase9_results/analysis.json")
OUTPUT_DIRECTORY = Path("phase9_figures")

COLORS = {
    "base": "#64748b",
    "adapter": "#d97706",
    "phase8_frozen": "#7c3aed",
    "generic": "#0891b2",
    "hard": "#2563eb",
    "route_off": "#64748b",
    "typed_handshake_inactive": "#d97706",
    "operand_content": "#be123c",
    "calculator_trajectory": "#7c3aed",
    "downstream_decode": "#0891b2",
}


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


def annotate(axis: object, bars: object, *, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.0f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    seeds = [str(seed) for seed in payload["phase9_seeds"]]

    labels = [
        "Untouched\nbase",
        "Matched\nadapter",
        "Phase 8\nfrozen",
        "Phase 9\ngeneric",
        "Phase 9\nhard",
    ]
    x = np.arange(len(labels))
    width = 0.23
    figure, axis = plt.subplots(figsize=(10.2, 5.0))
    for index, seed in enumerate(seeds):
        values = [
            payload["base"]["format_exact"],
            payload["matched_adapters"][seed]["exact"],
            payload["conditions"]["phase8_frozen"][seed]["exact"],
            payload["conditions"]["generic"][seed]["exact"],
            payload["conditions"]["hard"][seed]["exact"],
        ]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"seed {seed}",
        )
        annotate(axis, bars)
    axis.axhline(
        95,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="Frozen hard-condition gate (95/100)",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Exact numeral-only answers out of 100")
    axis.set_title("Phase 9 sealed addition accuracy")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(figure, "condition_accuracy")

    conditions = ("phase8_frozen", "generic", "hard")
    condition_labels = ("Phase 8 frozen", "Phase 9 generic", "Phase 9 hard")
    x = np.arange(len(condition_labels))
    figure, axis = plt.subplots(figsize=(9.3, 4.9))
    for index, seed in enumerate(seeds):
        values = [
            payload["conditions"][condition][seed][
                "negative_false_routes"
            ]
            for condition in conditions
        ]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"seed {seed}",
        )
        annotate(axis, bars)
    axis.axhline(
        4,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.2,
        label="Frozen hard-condition gate (≤4/200)",
    )
    axis.set_xticks(x, condition_labels)
    axis.set_ylabel("False routes out of 200 negatives")
    maximum = max(
        payload["conditions"][condition][seed]["negative_false_routes"]
        for condition in conditions
        for seed in seeds
    )
    axis.set_ylim(0, max(8, maximum * 1.18 + 2))
    axis.set_title("Adversarial semantic routing")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(figure, "negative_false_routes")

    stage_order = (
        "route_off",
        "typed_handshake_inactive",
        "operand_content",
        "calculator_trajectory",
        "downstream_decode",
    )
    stage_labels = (
        "Route off",
        "Typed handshake inactive",
        "Wrong operand content",
        "Calculator trajectory",
        "Downstream decode",
    )
    x = np.arange(len(stage_order))
    figure, axis = plt.subplots(figsize=(10.0, 4.9))
    for index, seed in enumerate(seeds):
        taxonomy = analysis["failure_taxonomy"]["hard"][seed]
        values = [taxonomy.get(stage, 0) for stage in stage_order]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"seed {seed}",
        )
        annotate(axis, bars)
    axis.set_xticks(x, stage_labels, rotation=10, ha="right")
    axis.set_ylabel("Failed positive prompts out of 100")
    maximum = max(
        analysis["failure_taxonomy"]["hard"][seed].get(stage, 0)
        for seed in seeds
        for stage in stage_order
    )
    axis.set_ylim(0, max(5, maximum * 1.2 + 2))
    axis.set_title("Where Phase 9 hard-condition failures occur")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, "hard_failure_taxonomy")

    categories = sorted(
        analysis["false_routes_by_negative_split"]["hard"][seeds[0]]
    )
    x = np.arange(len(categories))
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    bar_width = 0.24
    for index, seed in enumerate(seeds):
        values = [
            analysis["false_routes_by_negative_split"]["hard"][seed][
                category
            ]["false_routes"]
            for category in categories
        ]
        axis.bar(
            x + (index - 1) * bar_width,
            values,
            bar_width,
            label=f"seed {seed}",
        )
    axis.set_xticks(
        x,
        [category.replace("_", "\n") for category in categories],
        fontsize=8,
    )
    axis.set_ylabel("False routes out of 20")
    axis.set_ylim(0, 21)
    axis.set_title("Hard-condition false routes by adversarial family")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, "hard_false_routes_by_family")


if __name__ == "__main__":
    main()
