from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prediction_frame(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(artifact_root.glob("*-seed-*/predictions.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(f"No prediction records found under {artifact_root}")
    return pd.DataFrame(rows)


def bootstrap_seed_difference(
    per_seed: pd.DataFrame,
    model_a: str,
    model_b: str,
    split: str,
    samples: int = 20_000,
    seed: int = 260725,
) -> tuple[float, float, float]:
    pivot = (
        per_seed[per_seed["split"] == split]
        .pivot(index="seed", columns="model", values="accuracy")
        .dropna()
    )
    differences = (pivot[model_a] - pivot[model_b]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for index in range(samples):
        boot[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return (
        float(differences.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def analyze(study_path: Path, figures_dir: Path, results_dir: Path) -> Path:
    payload = json.loads(study_path.read_text())
    artifact_root = study_path.parent
    predictions = _prediction_frame(artifact_root)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed = (
        predictions.groupby(["model", "seed", "split"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )
    summary = (
        per_seed.groupby(["model", "split"])["accuracy"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(results_dir / "accuracy_summary.csv", index=False)
    per_seed.to_csv(results_dir / "accuracy_per_seed.csv", index=False)

    primary_difference = bootstrap_seed_difference(
        per_seed,
        model_a="latent_firmware",
        model_b="baseline",
        split="ood_primary",
    )

    split_order = ["id_random", "ood_primary", "ood_long", "carry_chain"]
    model_order = ["baseline", "latent_firmware", "direct_firmware"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.24
    x = np.arange(len(split_order))
    for index, model in enumerate(model_order):
        model_rows = summary[summary["model"] == model].set_index("split")
        means = [model_rows.loc[split, "mean"] for split in split_order]
        stds = [model_rows.loc[split, "std"] for split in split_order]
        ax.bar(x + (index - 1) * width, means, width, yerr=stds, capsize=3, label=model)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xticks(x, [label.replace("_", "\n") for label in split_order])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_split.pdf")
    fig.savefig(figures_dir / "accuracy_by_split.png", dpi=220)
    plt.close(fig)

    by_digits = (
        predictions.groupby(["model", "seed", "max_operand_digits"], as_index=False)["correct"]
        .mean()
        .groupby(["model", "max_operand_digits"])["correct"]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_digits.to_csv(results_dir / "accuracy_by_digits.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for model in model_order:
        rows = by_digits[by_digits["model"] == model]
        ax.plot(rows["max_operand_digits"], rows["mean"], marker="o", markersize=3, label=model)
    ax.set_xlabel("Maximum operand length (digits)")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "accuracy_by_digits.pdf")
    fig.savefig(figures_dir / "accuracy_by_digits.png", dpi=220)
    plt.close(fig)

    report = {
        "experiment_name": payload["experiment_name"],
        "eval_sha256": payload["eval_sha256"],
        "primary_difference_latent_minus_baseline": {
            "mean": primary_difference[0],
            "bootstrap_95_ci": [primary_difference[1], primary_difference[2]],
            "unit": "training_seed",
        },
        "summary": summary.to_dict(orient="records"),
    }
    report_path = results_dir / "analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path

