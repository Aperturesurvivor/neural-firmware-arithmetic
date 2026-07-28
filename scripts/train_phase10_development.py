from __future__ import annotations

import argparse
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
from neural_firmware.phase9_data import PHASE9_SOURCE_SEEDS
from neural_firmware.phase10_training import (
    PHASE10_CONDITIONS,
    phase10_checkpoint_state,
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
OUTPUT_DIRECTORY = Path("phase10_artifacts/development")
RESULT_PATH = Path("phase10_results/development_training.json")
DEFAULT_PHASE10_SEED = 16_199
DEFAULT_SOURCE_SEED = PHASE9_SOURCE_SEEDS[15_201]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def load_features() -> dict[str, object]:
    phase8 = torch.load(
        PHASE8_SEQUENCE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase9 = torch.load(
        PHASE9_FEATURE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase8_route = torch.load(
        PHASE8_ROUTE_CACHE,
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
            phase8["training"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        choices=[condition.name for condition in PHASE10_CONDITIONS],
        help="condition to train; repeat to select several",
    )
    parser.add_argument("--steps", type=int, default=2_500)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--route-steps", type=int, default=2_500)
    parser.add_argument("--route-learning-rate", type=float, default=0.0005)
    arguments = parser.parse_args()
    selected = (
        [
            condition
            for condition in PHASE10_CONDITIONS
            if condition.name in arguments.condition
        ]
        if arguments.condition
        else list(PHASE10_CONDITIONS)
    )
    started = time.perf_counter()
    implementation_commit = git_commit()
    features = load_features()
    source_path = (
        SOURCE_DIRECTORY / f"implant_seed_{DEFAULT_SOURCE_SEED}.pt"
    )
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for condition in selected:
        run_started = time.perf_counter()
        print(f"training Phase 10 development condition {condition.name}", flush=True)
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
            seed=DEFAULT_PHASE10_SEED,
            steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            route_steps=arguments.route_steps,
            route_learning_rate=arguments.route_learning_rate,
            digit_threshold=0.8,
        )
        checkpoint_path = OUTPUT_DIRECTORY / f"{condition.name}.pt"
        torch.save(
            {
                **phase10_checkpoint_state(implant, source),
                "stage": "phase10_architecture_development",
                "implementation_commit": implementation_commit,
                "condition": condition.name,
                "phase10_seed": DEFAULT_PHASE10_SEED,
                "source_phase8_seed": DEFAULT_SOURCE_SEED,
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": sha256(source_path),
            },
            checkpoint_path,
        )
        record = {
            "condition": condition.name,
            "phase10_seed": DEFAULT_PHASE10_SEED,
            "source_phase8_seed": DEFAULT_SOURCE_SEED,
            "training": training,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "route_threshold": implant.route_threshold,
            "digit_threshold": implant.digit_threshold,
            "wall_time_seconds": time.perf_counter() - run_started,
        }
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)
        del bundle, implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    payload = {
        "status": "phase10_architecture_development_training_complete",
        "implementation_commit": implementation_commit,
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "phase10_seed": DEFAULT_PHASE10_SEED,
        "source_phase8_seed": DEFAULT_SOURCE_SEED,
        "steps": arguments.steps,
        "learning_rate": arguments.learning_rate,
        "route_steps": arguments.route_steps,
        "route_learning_rate": arguments.route_learning_rate,
        "conditions": [condition.name for condition in selected],
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
