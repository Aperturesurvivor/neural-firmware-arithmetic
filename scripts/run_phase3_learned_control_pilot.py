from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.internal_data import make_internal_addition_examples
from neural_firmware.internal_firmware import install_internal_learned_control
from neural_firmware.internal_training import (
    InternalDecoderTrainConfig,
    generate_internal_learned_control,
    train_internal_learned_control,
)
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    artifact_directory = Path("phase3_artifacts/learned_control_pilot_v1")
    output = Path("phase3_results/learned_control_pilot_v1.json")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    train_examples = make_internal_addition_examples(
        count=1_000,
        min_digits=1,
        max_digits=4,
        seed=91_101,
        split="control_train_1_4",
    )
    evaluation_sets = {
        "id_1_4": make_internal_addition_examples(
            count=60,
            min_digits=1,
            max_digits=4,
            seed=91_102,
            split="id_1_4",
        ),
        "ood_5_8": make_internal_addition_examples(
            count=60,
            min_digits=5,
            max_digits=8,
            seed=91_103,
            split="ood_5_8",
        ),
        "ood_9_12": make_internal_addition_examples(
            count=60,
            min_digits=9,
            max_digits=12,
            seed=91_104,
            split="ood_9_12",
        ),
    }
    bundle = load_model_bundle(MODEL_ID, revision=REVISION)
    wrapper = install_internal_learned_control(
        bundle.model,
        depth_after_blocks=6,
        rank=10,
    )
    train_result = train_internal_learned_control(
        bundle,
        wrapper,
        train_examples,
        InternalDecoderTrainConfig(
            seed=1_301,
            steps=1_000,
            batch_size=2,
            learning_rate=0.001,
        ),
    )
    torch.save(wrapper.adapter.state_dict(), artifact_directory / "adapter.pt")
    split_results = {}
    for split, examples in evaluation_sets.items():
        rows = [
            generate_internal_learned_control(
                bundle,
                wrapper,
                example,
                enabled=True,
            ).to_dict()
            for example in examples
        ]
        correct = sum(row["exact"] for row in rows)
        split_results[split] = {
            "examples": len(rows),
            "correct": correct,
            "exact_match_accuracy": correct / len(rows),
            "predictions": rows,
        }
    result = {
        "status": "pilot",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "depth_after_blocks": 6,
        "rank": 10,
        "train_result": asdict(train_result),
        "splits": split_results,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "train_result": result["train_result"],
                "accuracy": {
                    split: values["exact_match_accuracy"]
                    for split, values in split_results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
