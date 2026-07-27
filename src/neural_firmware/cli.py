from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_firmware.analysis import analyze
from neural_firmware.data import load_eval_splits
from neural_firmware.evaluation import evaluate_model, save_evaluation
from neural_firmware.study import run_study
from neural_firmware.training import (
    load_checkpoint,
    load_config,
    load_model,
    resolve_device,
    train_one,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or (_project_root() / "artifacts" / config["experiment_name"])
    result = train_one(config, args.model, args.seed, output)
    print(json.dumps(result.__dict__, indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-sets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    splits, eval_hash = load_eval_splits(args.eval_sets)
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    payload = load_checkpoint(args.checkpoint)
    predictions, metrics = evaluate_model(model, payload["seed"], splits, device)
    save_evaluation(args.output, predictions, metrics, eval_hash)
    print(json.dumps(metrics, indent=2))


def study_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_study(load_config(args.config), _project_root())
    print(summary)


def analyze_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.study,
        figures_dir=_project_root() / "figures",
        results_dir=_project_root() / "results",
    )
    print(report)
