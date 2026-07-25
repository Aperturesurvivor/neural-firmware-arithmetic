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
from neural_firmware.training import TrainingResult, load_model, resolve_device, train_one


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
            run_dir = artifact_root / f"{mode}-seed-{seed}"
            training_path = run_dir / "training.json"
            checkpoint_path = run_dir / "model.pt"
            if training_path.exists() and checkpoint_path.exists():
                training_result = TrainingResult(**json.loads(training_path.read_text()))
                print(f"[{mode} seed={seed}] reusing completed checkpoint", flush=True)
            else:
                training_result = train_one(config, mode, seed, artifact_root)
            device = resolve_device(config.get("device", "auto"))
            model = load_model(Path(training_result.checkpoint), device)
            evaluation_path = run_dir / "evaluation.json"
            if evaluation_path.exists() and (run_dir / "predictions.jsonl").exists():
                evaluation_payload = json.loads(evaluation_path.read_text())
                if evaluation_payload["eval_sha256"] != eval_hash:
                    raise ValueError("Existing evaluation uses a different evaluation set")
                metrics = evaluation_payload["metrics"]
                print(f"[{mode} seed={seed}] reusing completed evaluation", flush=True)
            else:
                predictions, metrics = evaluate_model(model, seed, splits, device)
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
