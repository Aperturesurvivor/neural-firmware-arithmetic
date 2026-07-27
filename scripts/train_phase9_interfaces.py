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
from neural_firmware.phase9_data import (
    PHASE9_SOURCE_SEEDS,
    PHASE9_TRAINING_SEEDS,
)
from neural_firmware.phase9_training import continue_phase9_interface
from neural_firmware.pretrained_training import load_model_bundle

PHASE8_SEQUENCE_CACHE = Path("phase8_artifacts/cache/layer15_full_features.pt")
PHASE8_ROUTE_CACHE = Path("phase8_artifacts/cache/first_step_route_features.pt")
PHASE9_FEATURE_CACHE = Path("phase9_artifacts/cache/interface_features.pt")
SOURCE_DIRECTORY = Path("phase8_artifacts/confirmatory_implants")
OUTPUT_DIRECTORY = Path("phase9_artifacts/confirmatory_interfaces")
RESULT_PATH = Path("phase9_results/confirmatory_interface_training.json")

FROZEN_CONFIG = {
    "interface_steps": 1_500,
    "interface_learning_rate": 0.0005,
    "role_loss_weight": 1.0,
    "digit_loss_weight": 1.0,
    "route_steps": 2_500,
    "route_learning_rate": 0.0005,
    "maximum_development_false_positive_rate": 0.01,
    "digit_threshold": 0.8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_features() -> dict[str, object]:
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
    return {
        "original_sequence": SequenceFeatureSet.load_state_dict(
            phase8_sequence["training"]
        ),
        "original_route": FirstStepRouteFeatureSet.load_state_dict(
            phase8_route["training"]
        ),
        "development_sequence": SequenceFeatureSet.load_state_dict(
            phase9["development"]["sequence"]
        ),
        "development_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["development"]["route"]
        ),
        "generic_sequence": SequenceFeatureSet.load_state_dict(
            phase9["generic"]["sequence"]
        ),
        "generic_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["generic"]["route"]
        ),
        "hard_sequence": SequenceFeatureSet.load_state_dict(
            phase9["hard"]["sequence"]
        ),
        "hard_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["hard"]["route"]
        ),
    }


def main() -> None:
    started = time.perf_counter()
    implementation_commit = git_commit()
    features = _load_features()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for condition in ("generic", "hard"):
        for phase9_seed in PHASE9_TRAINING_SEEDS:
            run_started = time.perf_counter()
            source_seed = PHASE9_SOURCE_SEEDS[phase9_seed]
            source_path = SOURCE_DIRECTORY / f"implant_seed_{source_seed}.pt"
            source = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            bundle = load_model_bundle(
                PHASE8_MODEL_ID,
                revision=PHASE8_MODEL_REVISION,
            )
            implant, training = continue_phase9_interface(
                bundle,
                source,
                original_sequence=features["original_sequence"],
                original_route=features["original_route"],
                condition_sequence=features[f"{condition}_sequence"],
                condition_route=features[f"{condition}_route"],
                development_sequence=features["development_sequence"],
                development_route=features["development_route"],
                seed=phase9_seed,
                interface_steps=FROZEN_CONFIG["interface_steps"],
                interface_learning_rate=FROZEN_CONFIG[
                    "interface_learning_rate"
                ],
                role_loss_weight=FROZEN_CONFIG["role_loss_weight"],
                digit_loss_weight=FROZEN_CONFIG["digit_loss_weight"],
                route_steps=FROZEN_CONFIG["route_steps"],
                route_learning_rate=FROZEN_CONFIG["route_learning_rate"],
                maximum_development_false_positive_rate=FROZEN_CONFIG[
                    "maximum_development_false_positive_rate"
                ],
            )
            implant.digit_threshold = FROZEN_CONFIG["digit_threshold"]
            checkpoint_path = (
                OUTPUT_DIRECTORY / f"{condition}_seed_{phase9_seed}.pt"
            )
            torch.save(
                {
                    **source,
                    "stage": "phase9_frozen_confirmatory_interface",
                    "implementation_commit": implementation_commit,
                    "condition": condition,
                    "phase9_seed": phase9_seed,
                    "source_phase8_seed": source_seed,
                    "source_checkpoint": str(source_path),
                    "source_checkpoint_sha256": sha256(source_path),
                    "route_threshold": implant.route_threshold,
                    "digit_threshold": implant.digit_threshold,
                    "input_rows": implant.input_rows.detach().cpu(),
                    "result_columns": implant.result_columns.detach().cpu(),
                    "phase9_updated_parameters": implant.input_rows.numel(),
                    "architectural_learned_parameters": (
                        implant.input_rows.numel()
                        + implant.result_columns.numel()
                    ),
                },
                checkpoint_path,
            )
            record = {
                "condition": condition,
                "phase9_seed": phase9_seed,
                "source_phase8_seed": source_seed,
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": sha256(source_path),
                "training": training,
                "route_threshold": implant.route_threshold,
                "digit_threshold": implant.digit_threshold,
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
        "status": "phase9_confirmatory_interface_training_complete",
        "implementation_commit": implementation_commit,
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "phase9_seeds": list(PHASE9_TRAINING_SEEDS),
        "source_seed_mapping": {
            str(seed): PHASE9_SOURCE_SEEDS[seed]
            for seed in PHASE9_TRAINING_SEEDS
        },
        "conditions": ["generic", "hard"],
        "frozen_config": FROZEN_CONFIG,
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
