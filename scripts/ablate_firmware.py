"""Post-hoc causal check: evaluate trained latent models with firmware strength zero."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from neural_firmware.data import load_eval_splits
from neural_firmware.evaluation import evaluate_model
from neural_firmware.training import load_model, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    study = json.loads(args.study.read_text())
    splits, eval_hash = load_eval_splits(args.study.parent / "eval_sets.json")
    if eval_hash != study["eval_sha256"]:
        raise ValueError("Evaluation hash does not match study manifest")
    device = resolve_device(study["config"].get("device", "auto"))

    rows: list[dict] = []
    for run in study["runs"]:
        training = run["training"]
        if training["model"] != "latent_firmware":
            continue
        model = load_model(Path(training["checkpoint"]), device)
        model.config = replace(model.config, firmware_strength=0.0)
        _, metrics = evaluate_model(model, training["seed"], splits, device)
        for split, values in metrics.items():
            rows.append(
                {
                    "model": "latent_firmware_strength_zero",
                    "seed": training["seed"],
                    "split": split,
                    **values,
                }
            )
        print(f"completed seed {training['seed']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

