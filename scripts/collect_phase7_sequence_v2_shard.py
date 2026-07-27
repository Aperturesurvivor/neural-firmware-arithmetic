from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
from neural_firmware.phase7_sequence_implant import SequenceImplantLayout
from neural_firmware.phase7_sequence_training import collect_sequence_features
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYER_INDEX = 23
SHARD_DIRECTORY = Path("phase7_artifacts/cache/sequence_v2_shards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def all_examples(split: str) -> tuple[list[object], int]:
    if split == "train":
        return (
            make_semantic_addition_examples(
                count=2_000,
                min_digits=1,
                max_digits=4,
                seed=12_811,
                split="phase7_sequence_interface_v2_train_positive",
                families=PHASE5_TRAIN_ADDITION_FAMILIES,
            )
            + make_semantic_routing_negatives(
                count=2_000,
                min_digits=1,
                max_digits=4,
                seed=12_812,
                split="phase7_sequence_interface_v2_train_negative",
                families=PHASE5_TRAIN_NEGATIVE_FAMILIES,
            ),
            400,
        )
    return (
        make_semantic_addition_examples(
            count=300,
            min_digits=1,
            max_digits=4,
            seed=12_813,
            split="phase7_sequence_interface_v2_development_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=300,
            min_digits=1,
            max_digits=4,
            seed=12_814,
            split="phase7_sequence_interface_v2_development_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        ),
        300,
    )


def main() -> None:
    args = parse_args()
    examples, shard_size = all_examples(args.split)
    start = args.shard_index * shard_size
    stop = min(len(examples), start + shard_size)
    if start >= len(examples):
        raise ValueError("shard index is beyond the dataset")
    output = SHARD_DIRECTORY / f"{args.split}_{args.shard_index:02d}.pt"
    if output.exists():
        print(f"exists {output}", flush=True)
        return
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    features = collect_sequence_features(
        bundle,
        examples[start:stop],
        layer_index=LAYER_INDEX,
        layout=SequenceImplantLayout(max_digits=4),
        batch_size=8,
        ordinary_tokens_per_example=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.state_dict(), output)
    print(
        f"wrote {output} examples={stop - start} rows={features.rows}",
        flush=True,
    )


if __name__ == "__main__":
    main()

