from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    evaluate_sequence_interface,
    train_sequence_interface,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

SOURCE_CHECKPOINT = Path(
    "phase7_artifacts/sequence_pilot_v2/neuron_implant_seed_12801.pt"
)
CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_interface_v2/neuron_implant_seed_12811.pt"
)
RESULT_PATH = Path("phase7_results/sequence_interface_v2.json")
CACHE_DIRECTORY = Path("phase7_artifacts/cache")
SEED = 12_811


def save_features(features: SequenceFeatureSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.state_dict(), path)


def load_features(path: Path) -> SequenceFeatureSet:
    return SequenceFeatureSet.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def make_data() -> tuple[list[object], list[object]]:
    train = (
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
        )
    )
    development = (
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
        )
    )
    return train, development


def prepare_features(
    bundle: object,
    train: list[object],
    development: list[object],
    *,
    layer_index: int,
    layout: SequenceImplantLayout,
) -> tuple[SequenceFeatureSet, SequenceFeatureSet]:
    train_path = CACHE_DIRECTORY / "sequence_interface_train_v2_compact.pt"
    development_path = (
        CACHE_DIRECTORY / "sequence_interface_development_v2_compact.pt"
    )
    if train_path.exists():
        train_features = load_features(train_path)
    else:
        shards = [
            load_features(
                CACHE_DIRECTORY / "sequence_v2_shards" / f"train_{index:02d}.pt"
            )
            for index in range(10)
        ]
        train_features = SequenceFeatureSet(
            hidden=torch.cat([shard.hidden for shard in shards]),
            route_targets=torch.cat([shard.route_targets for shard in shards]),
            role_targets=torch.cat([shard.role_targets for shard in shards]),
            digit_targets=torch.cat([shard.digit_targets for shard in shards]),
            step_targets=torch.cat([shard.step_targets for shard in shards]),
        )
        save_features(train_features, train_path)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if development_path.exists():
        development_features = load_features(development_path)
    else:
        shards = [
            load_features(
                CACHE_DIRECTORY
                / "sequence_v2_shards"
                / f"development_{index:02d}.pt"
            )
            for index in range(2)
        ]
        development_features = SequenceFeatureSet(
            hidden=torch.cat([shard.hidden for shard in shards]),
            route_targets=torch.cat([shard.route_targets for shard in shards]),
            role_targets=torch.cat([shard.role_targets for shard in shards]),
            digit_targets=torch.cat([shard.digit_targets for shard in shards]),
            step_targets=torch.cat([shard.step_targets for shard in shards]),
        )
        save_features(development_features, development_path)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return train_features, development_features


def main() -> None:
    source = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    layout = SequenceImplantLayout(**source["layout"])
    bundle = load_model_bundle(
        source["model_id"],
        revision=source["model_revision"],
    )
    train, development = make_data()
    started = time.perf_counter()
    train_features, development_features = prepare_features(
        bundle,
        train,
        development,
        layer_index=source["layer_index"],
        layout=layout,
    )
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=source["layer_index"],
        selected_indices=source["selected_indices"],
        layout=layout,
        output_strength=source["output_strength"],
        route_threshold=source["route_threshold"],
    )
    with torch.no_grad():
        implant.input_rows.copy_(source["input_rows"].to(bundle.device))
        implant.result_columns.copy_(source["result_columns"].to(bundle.device))
    training, metrics = train_sequence_interface(
        implant,
        train_features,
        development_features,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=SEED,
            steps=3_000,
            batch_size=256,
            learning_rate=0.001,
        ),
    )
    final = evaluate_sequence_interface(
        implant,
        development_features,
        device=bundle.device,
    )
    compact_final = {
        key: value
        for key, value in final.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **source,
            "stage": "output_with_interface_v2",
            "interface_seed": SEED,
            "input_rows": implant.input_rows.detach().cpu(),
            "result_columns": implant.result_columns.detach().cpu(),
            "route_threshold": implant.route_threshold,
        },
        CHECKPOINT_PATH,
    )
    payload = {
        "status": "development",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "checkpoint": str(CHECKPOINT_PATH),
        "layout": asdict(layout),
        "train_rows": train_features.rows,
        "development_rows": development_features.rows,
        "training": training,
        "development": compact_final,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
