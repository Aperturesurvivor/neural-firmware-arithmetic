from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    SequenceFeatureSet,
)
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase10_data import (
    PHASE10_SOURCE_SEEDS,
    PHASE10_TRAINING_SEEDS,
)
from neural_firmware.phase10_training import (
    phase10_checkpoint_state,
    phase10_condition,
    train_phase10_condition,
)
from neural_firmware.pretrained_training import load_model_bundle

PHASE8_SEQUENCE_CACHE = Path("phase8_artifacts/cache/layer15_full_features.pt")
PHASE8_ROUTE_CACHE = Path("phase8_artifacts/cache/first_step_route_features.pt")
PHASE9_FEATURE_CACHE = Path("phase9_artifacts/cache/interface_features.pt")
PHASE10_DEVELOPMENT_CACHE = Path(
    "phase10_artifacts/cache/development_features.pt"
)
SOURCE_DIRECTORY = Path("phase8_artifacts/confirmatory_implants")
OUTPUT_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
RESULT_PATH = Path("phase10_results/confirmatory_interface_training.json")
CONDITIONS = ("linear", "nonlinear", "linear_representation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def load_features() -> dict[str, object]:
    phase8_sequence = torch.load(
        PHASE8_SEQUENCE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase8_route = torch.load(
        PHASE8_ROUTE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase9 = torch.load(
        PHASE9_FEATURE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase10 = torch.load(
        PHASE10_DEVELOPMENT_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    return {
        "original_sequence": SequenceFeatureSet.load_state_dict(
            phase8_sequence["training"]
        ),
        "original_route": FirstStepRouteFeatureSet.load_state_dict(
            phase8_route["training"]
        ),
        "hard_sequence": SequenceFeatureSet.load_state_dict(
            phase9["hard"]["sequence"]
        ),
        "hard_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["hard"]["route"]
        ),
        "development_sequence": SequenceFeatureSet.load_state_dict(
            phase10["sequence"]
        ),
        "development_route": FirstStepRouteFeatureSet.load_state_dict(
            phase10["route"]
        ),
    }


def main() -> None:
    started = time.perf_counter()
    implementation_commit = git_commit()
    features = load_features()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for condition_name in CONDITIONS:
        condition = phase10_condition(condition_name)
        for phase10_seed in PHASE10_TRAINING_SEEDS:
            run_started = time.perf_counter()
            source_seed = PHASE10_SOURCE_SEEDS[phase10_seed]
            source_path = (
                SOURCE_DIRECTORY / f"implant_seed_{source_seed}.pt"
            )
            source = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            bundle = load_model_bundle(
                PHASE8_MODEL_ID,
                revision=PHASE8_MODEL_REVISION,
            )
            implant, training = train_phase10_condition(
                bundle,
                source,
                condition=condition,
                original_sequence=features["original_sequence"],
                original_route=features["original_route"],
                hard_sequence=features["hard_sequence"],
                hard_route=features["hard_route"],
                development_sequence=features["development_sequence"],
                development_route=features["development_route"],
                seed=phase10_seed,
                steps=2_500,
                learning_rate=0.0005,
                route_steps=0,
                route_learning_rate=0.0005,
                digit_threshold=0.8,
            )
            checkpoint_path = (
                OUTPUT_DIRECTORY
                / f"{condition_name}_seed_{phase10_seed}.pt"
            )
            torch.save(
                {
                    **phase10_checkpoint_state(implant, source),
                    "stage": "phase10_frozen_confirmatory_interface",
                    "implementation_commit": implementation_commit,
                    "condition": condition_name,
                    "phase10_seed": phase10_seed,
                    "source_phase8_seed": source_seed,
                    "source_checkpoint": str(source_path),
                    "source_checkpoint_sha256": sha256(source_path),
                },
                checkpoint_path,
            )
            record = {
                "condition": condition_name,
                "phase10_seed": phase10_seed,
                "source_phase8_seed": source_seed,
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": sha256(source_path),
                "training": training,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "wall_time_seconds": time.perf_counter() - run_started,
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
            del bundle, implant
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    payload = {
        "status": "phase10_confirmatory_interface_training_complete",
        "implementation_commit": implementation_commit,
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "phase10_seeds": list(PHASE10_TRAINING_SEEDS),
        "source_seed_mapping": {
            str(seed): PHASE10_SOURCE_SEEDS[seed]
            for seed in PHASE10_TRAINING_SEEDS
        },
        "conditions": list(CONDITIONS),
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
