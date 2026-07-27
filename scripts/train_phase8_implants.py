from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    RouteRowTrainConfig,
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    SequenceOutputTrainConfig,
    evaluate_sequence_interface,
    train_route_rows,
    train_sequence_interface,
    train_sequence_output,
)
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    PHASE8_TRAINING_SEEDS,
    build_phase8_training_and_development,
)
from neural_firmware.pretrained_training import load_model_bundle

SELECTION_PATH = Path("phase8_artifacts/development_selection.pt")
SEQUENCE_FEATURE_PATH = Path(
    "phase8_artifacts/cache/layer15_full_features.pt"
)
ROUTE_FEATURE_PATH = Path(
    "phase8_artifacts/cache/first_step_route_features.pt"
)
OUTPUT_DIRECTORY = Path("phase8_artifacts/confirmatory_implants")
RESULT_PATH = Path("phase8_results/confirmatory_implant_training.json")


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }


def main() -> None:
    started = time.perf_counter()
    selection = torch.load(SELECTION_PATH, map_location="cpu", weights_only=True)
    sequence_cache = torch.load(
        SEQUENCE_FEATURE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    route_cache = torch.load(
        ROUTE_FEATURE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    train_features = SequenceFeatureSet.load_state_dict(
        sequence_cache["training"]
    )
    development_features = SequenceFeatureSet.load_state_dict(
        sequence_cache["development"]
    )
    route_train = FirstStepRouteFeatureSet.load_state_dict(
        route_cache["training"]
    )
    route_development = FirstStepRouteFeatureSet.load_state_dict(
        route_cache["development"]
    )
    training_examples, _ = build_phase8_training_and_development()
    layout = SequenceImplantLayout(**selection["layout"])
    layer_index = int(selection["selected_layer"])
    selected_indices = selection["selected_indices"]
    implementation_commit = git_commit()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seed_records: list[dict[str, object]] = []
    for seed in PHASE8_TRAINING_SEEDS:
        seed_started = time.perf_counter()
        bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        torch.manual_seed(seed)
        implant = install_sequence_neuron_implant(
            bundle.model,
            layer_index=layer_index,
            selected_indices=selected_indices,
            layout=layout,
            output_strength=16.0,
            digit_threshold=0.9,
        )
        interface_training, _ = train_sequence_interface(
            implant,
            train_features,
            development_features,
            device=bundle.device,
            config=SequenceInterfaceTrainConfig(
                seed=seed,
                steps=3_000,
                batch_size=256,
                learning_rate=0.001,
                step_loss_weight=0.0,
            ),
        )
        interface_development = compact(
            evaluate_sequence_interface(
                implant,
                development_features,
                device=bundle.device,
            )
        )
        output_training = train_sequence_output(
            bundle,
            implant,
            training_examples,
            config=SequenceOutputTrainConfig(
                seed=seed,
                steps=200,
                batch_size=1,
                learning_rate=0.01,
            ),
        )
        hardened_route_rows, route_training, route_metrics = train_route_rows(
            implant.input_rows.detach().cpu()[:2],
            route_train,
            route_development,
            device=bundle.device,
            config=RouteRowTrainConfig(
                seed=seed + 300,
                steps=3_000,
                batch_size=256,
                learning_rate=0.001,
                maximum_development_false_positive_rate=0.025,
            ),
        )
        with torch.no_grad():
            implant.input_rows[:2].copy_(
                hardened_route_rows.to(bundle.device)
            )
        implant.route_threshold = float(route_metrics["threshold"]["threshold"])
        checkpoint_path = OUTPUT_DIRECTORY / f"implant_seed_{seed}.pt"
        torch.save(
            {
                "stage": "phase8_frozen_confirmatory_implant",
                "implementation_commit": implementation_commit,
                "model_id": PHASE8_MODEL_ID,
                "model_revision": PHASE8_MODEL_REVISION,
                "seed": seed,
                "layer_index": layer_index,
                "layout": asdict(layout),
                "selected_indices": selected_indices,
                "route_threshold": implant.route_threshold,
                "digit_threshold": implant.digit_threshold,
                "output_strength": implant.output_strength,
                "input_rows": implant.input_rows.detach().cpu(),
                "result_columns": implant.result_columns.detach().cpu(),
                "runtime": {
                    "latch_route": True,
                    "latch_operands": True,
                    "deterministic_result_step": True,
                    "preserve_base_when_off": True,
                },
            },
            checkpoint_path,
        )
        record = {
            "seed": seed,
            "interface_training": interface_training,
            "interface_development_before_route_hardening": (
                interface_development
            ),
            "output_training": output_training,
            "route_training": route_training,
            "route_development": route_metrics,
            "learned_parameters": implant.trainable_parameter_count,
            "calculator_learned_parameters": (
                implant.calculator.trainable_parameter_count
            ),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "wall_time_seconds": time.perf_counter() - seed_started,
        }
        seed_records.append(record)
        print(json.dumps(record, indent=2), flush=True)
        del bundle, implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    payload = {
        "status": "confirmatory_implant_training_complete",
        "implementation_commit": implementation_commit,
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "training_seeds": list(PHASE8_TRAINING_SEEDS),
        "layer_index": layer_index,
        "selected_indices": selected_indices.tolist(),
        "layout": asdict(layout),
        "seed_records": seed_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
