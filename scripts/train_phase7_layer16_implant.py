from __future__ import annotations

import hashlib
import json
import platform
import subprocess
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
    SequenceOutputTrainConfig,
    evaluate_sequence_interface,
    train_sequence_interface,
    train_sequence_output,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 13_201
LAYER_INDEX = 16
CENSUS_PATH = Path(
    "phase7_artifacts/sequence_census_layer_16_compact_v1.pt"
)
SHARD_DIRECTORY = Path(
    "phase7_artifacts/cache/sequence_v2_layer_16_shards"
)
CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_layer16_v1/neuron_implant_seed_13201.pt"
)
RESULT_PATH = Path("phase7_results/sequence_layer16_training_v1.json")


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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


def load_shards(split: str, count: int) -> SequenceFeatureSet:
    shards = [
        SequenceFeatureSet.load_state_dict(
            torch.load(
                SHARD_DIRECTORY / f"{split}_{index:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        for index in range(count)
    ]
    return SequenceFeatureSet(
        hidden=torch.cat([shard.hidden for shard in shards]),
        route_targets=torch.cat([shard.route_targets for shard in shards]),
        role_targets=torch.cat([shard.role_targets for shard in shards]),
        digit_targets=torch.cat([shard.digit_targets for shard in shards]),
        step_targets=torch.cat([shard.step_targets for shard in shards]),
    )


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }


def save_checkpoint(
    implant: object,
    selected_indices: torch.Tensor,
    layout: SequenceImplantLayout,
    *,
    stage: str,
) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "implementation_commit": git_commit(),
            "layer_index": LAYER_INDEX,
            "layout": asdict(layout),
            "selected_indices": selected_indices,
            "route_threshold": implant.route_threshold,
            "digit_threshold": implant.digit_threshold,
            "output_strength": implant.output_strength,
            "input_rows": implant.input_rows.detach().cpu(),
            "result_columns": implant.result_columns.detach().cpu(),
        },
        CHECKPOINT_PATH,
    )


def main() -> None:
    layout = SequenceImplantLayout(max_digits=4, learned_step=False)
    census = torch.load(CENSUS_PATH, map_location="cpu", weights_only=True)
    selected_indices = census["selected_indices"]
    if len(selected_indices) != layout.total_width:
        raise ValueError("compact census does not match the fixed-step layout")
    train_examples, _development_examples = make_data()
    started = time.perf_counter()
    train_features = load_shards("train", 10)
    development_features = load_shards("development", 2)
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=LAYER_INDEX,
        selected_indices=selected_indices,
        layout=layout,
        output_strength=16.0,
    )
    interface_training, interface_development = train_sequence_interface(
        implant,
        train_features,
        development_features,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=SEED,
            steps=3_000,
            batch_size=256,
            learning_rate=0.001,
            step_loss_weight=0.0,
        ),
    )
    interface_development = compact_metrics(
        evaluate_sequence_interface(
            implant,
            development_features,
            device=bundle.device,
        )
    )
    save_checkpoint(
        implant,
        selected_indices,
        layout,
        stage="layer16_interface_complete",
    )
    partial = {
        "status": "interface_complete_output_pending",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "layer_index": LAYER_INDEX,
        "layout": asdict(layout),
        "selected_indices": selected_indices.tolist(),
        "interface_training": interface_training,
        "interface_development": interface_development,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
        },
    }
    write_json(RESULT_PATH, partial)
    print(json.dumps(partial["interface_development"], indent=2), flush=True)

    output_training = train_sequence_output(
        bundle,
        implant,
        train_examples,
        config=SequenceOutputTrainConfig(
            seed=SEED,
            steps=150,
            batch_size=1,
            learning_rate=0.01,
        ),
    )
    save_checkpoint(
        implant,
        selected_indices,
        layout,
        stage="layer16_output_complete",
    )
    payload = {
        **partial,
        "status": "development_layer16_training_complete",
        "mlp_width_before": implant.mlp_width,
        "mlp_width_after": implant.mlp_width,
        "residual_width": implant.input_rows.shape[1],
        "calculator_trainable_parameters": (
            implant.calculator.trainable_parameter_count
        ),
        "interface_learned_parameters": implant.input_rows.numel(),
        "result_learned_parameters": implant.result_columns.numel(),
        "total_learned_parameters": (
            implant.input_rows.numel() + implant.result_columns.numel()
        ),
        "output_training": output_training,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(bundle.device),
            "mps_available": torch.backends.mps.is_available(),
            "implementation_commit": git_commit(),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json(RESULT_PATH, payload)
    print(json.dumps(output_training, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
