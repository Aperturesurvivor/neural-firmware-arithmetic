from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_firmware.data import (
    load_eval_splits,
    make_eval_splits,
    save_eval_splits,
)
from neural_firmware.evaluation import evaluate_model, save_evaluation
from neural_firmware.training import load_model, resolve_device, train_one


def run_study(config: dict[str, Any], project_root: Path) -> Path:
    experiment_name = config["experiment_name"]
    artifact_root = project_root / "artifacts" / experiment_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    eval_path = artifact_root / "eval_sets.json"
    if eval_path.exists():
        splits, eval_hash = load_eval_splits(eval_path)
    else:
        splits = make_eval_splits(
            eval_examples=config["data"]["eval_examples"],
            carry_examples=config["data"]["carry_examples"],
        )
        eval_hash = save_eval_splits(eval_path, splits)

    all_runs: list[dict[str, Any]] = []
    for mode in config["models"]:
        for seed in config["seeds"]:
            training_result = train_one(config, mode, seed, artifact_root)
            device = resolve_device(config.get("device", "auto"))
            model = load_model(Path(training_result.checkpoint), device)
            predictions, metrics = evaluate_model(model, seed, splits, device)
            run_dir = artifact_root / f"{mode}-seed-{seed}"
            save_evaluation(run_dir, predictions, metrics, eval_hash)
            all_runs.append(
                {
                    "training": asdict(training_result),
                    "evaluation": metrics,
                }
            )
            print(
                f"[{mode} seed={seed}] "
                + " ".join(
                    f"{split}={values['exact_match_accuracy']:.3f}"
                    for split, values in metrics.items()
                ),
                flush=True,
            )
            del model

    summary_path = artifact_root / "study.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "config": config,
                "eval_sha256": eval_hash,
                "runs": all_runs,
            },
            indent=2,
        )
        + "\n"
    )
    return summary_path

