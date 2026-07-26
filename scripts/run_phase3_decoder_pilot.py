from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.internal_data import make_internal_addition_examples
from neural_firmware.internal_firmware import install_internal_firmware_layer
from neural_firmware.internal_training import (
    InternalDecoderTrainConfig,
    generate_internal,
    train_internal_decoder,
)
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTHS = (6, 12, 18, 22)


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    artifact_root = Path("phase3_artifacts/decoder_pilot_v1")
    result_path = Path("phase3_results/decoder_pilot_v1.json")
    artifact_root.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    train_examples = make_internal_addition_examples(
        count=1_000,
        min_digits=1,
        max_digits=4,
        seed=63_101,
        split="pilot_train_1_4",
    )
    evaluation_sets = {
        "id_1_4": make_internal_addition_examples(
            count=40,
            min_digits=1,
            max_digits=4,
            seed=63_102,
            split="id_1_4",
        ),
        "ood_5_8": make_internal_addition_examples(
            count=40,
            min_digits=5,
            max_digits=8,
            seed=63_103,
            split="ood_5_8",
        ),
        "ood_9_12": make_internal_addition_examples(
            count=40,
            min_digits=9,
            max_digits=12,
            seed=63_104,
            split="ood_9_12",
        ),
    }
    depth_results = []
    for depth in DEPTHS:
        bundle = load_model_bundle(MODEL_ID, revision=REVISION)
        wrapper = install_internal_firmware_layer(
            bundle.model,
            depth_after_blocks=depth,
            strength=64.0,
        )
        digit_checkpoint = Path(
            f"phase3_artifacts/digit_probe_pilot_v1/depth_{depth}_probe.pt"
        )
        wrapper.unit.digit_encoder.load_state_dict(
            torch.load(
                digit_checkpoint,
                map_location=bundle.device,
                weights_only=True,
            )
        )
        train_result = train_internal_decoder(
            bundle,
            wrapper,
            train_examples,
            InternalDecoderTrainConfig(
                seed=1_000 + depth,
                steps=120,
                batch_size=2,
                learning_rate=0.01,
            ),
        )
        depth_directory = artifact_root / f"depth_{depth}"
        depth_directory.mkdir(parents=True, exist_ok=True)
        torch.save(wrapper.unit.state_dict(), depth_directory / "unit.pt")
        split_results = {}
        for split, examples in evaluation_sets.items():
            rows = [
                generate_internal(bundle, wrapper, example, enabled=True).to_dict()
                for example in examples
            ]
            split_results[split] = {
                "examples": len(rows),
                "correct": sum(row["exact"] for row in rows),
                "exact_match_accuracy": sum(row["exact"] for row in rows)
                / len(rows),
                "predictions": rows,
            }
        off_examples = evaluation_sets["id_1_4"][:10]
        off_rows = [
            generate_internal(bundle, wrapper, example, enabled=False).to_dict()
            for example in off_examples
        ]
        depth_results.append(
            {
                "depth_after_blocks": depth,
                "train_result": asdict(train_result),
                "splits": split_results,
                "unit_off": {
                    "examples": len(off_rows),
                    "correct": sum(row["exact"] for row in off_rows),
                    "predictions": off_rows,
                },
            }
        )
        wrapper.set_context(None)
        del wrapper
        del bundle
        release_memory()
        partial = {
            "status": "pilot_in_progress",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "strength": 64.0,
            "depth_results": depth_results,
        }
        result_path.write_text(json.dumps(partial, indent=2) + "\n")
    result = {
        "status": "pilot",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "configuration": {
            "depths": DEPTHS,
            "strength": 64.0,
            "train_examples": 1_000,
            "train_digits": [1, 4],
            "train_seed": 63_101,
            "steps": 120,
            "batch_size": 2,
            "learning_rate": 0.01,
        },
        "depth_results": depth_results,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "status": result["status"],
        "depths": [
            {
                "depth": row["depth_after_blocks"],
                "train": row["train_result"],
                "accuracy": {
                    split: values["exact_match_accuracy"]
                    for split, values in row["splits"].items()
                },
                "unit_off_correct": row["unit_off"]["correct"],
            }
            for row in depth_results
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
