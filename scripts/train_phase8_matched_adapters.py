from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase8_adapter import install_matched_residual_adapter
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    PHASE8_TRAINING_SEEDS,
    build_phase8_training_and_development,
)
from neural_firmware.phase8_training import (
    MatchedAdapterTrainConfig,
    collect_base_first_tokens,
    train_matched_adapter,
)
from neural_firmware.pretrained_training import load_model_bundle

LAYER_INDEX = 15
LEARNED_PARAMETERS = 57_344
BASE_TOKEN_CACHE = Path(
    "phase8_artifacts/cache/matched_adapter_base_first_tokens.pt"
)
OUTPUT_DIRECTORY = Path("phase8_artifacts/confirmatory_matched_adapters")
RESULT_PATH = Path("phase8_results/confirmatory_matched_adapter_training.json")


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    training, _ = build_phase8_training_and_development()
    negatives = [example for example in training if not example.route_label]
    implementation_commit = git_commit()
    base_first_tokens: dict[str, int] | None = None
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seed_records: list[dict[str, object]] = []
    for seed in PHASE8_TRAINING_SEEDS:
        seed_started = time.perf_counter()
        bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        adapter = install_matched_residual_adapter(
            bundle.model,
            layer_index=LAYER_INDEX,
            learned_parameter_count=LEARNED_PARAMETERS,
        )
        if base_first_tokens is None:
            if BASE_TOKEN_CACHE.exists():
                base_first_tokens = torch.load(
                    BASE_TOKEN_CACHE,
                    map_location="cpu",
                    weights_only=True,
                )["tokens"]
            else:
                base_first_tokens = collect_base_first_tokens(
                    bundle,
                    adapter,
                    negatives,
                )
                BASE_TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_id": PHASE8_MODEL_ID,
                        "model_revision": PHASE8_MODEL_REVISION,
                        "tokens": base_first_tokens,
                    },
                    BASE_TOKEN_CACHE,
                )
        training_record = train_matched_adapter(
            bundle,
            adapter,
            training,
            config=MatchedAdapterTrainConfig(
                seed=seed,
                steps=800,
                batch_size=4,
                learning_rate=0.002,
            ),
            base_first_tokens=base_first_tokens,
        )
        checkpoint_path = OUTPUT_DIRECTORY / f"adapter_seed_{seed}.pt"
        torch.save(
            {
                "stage": "phase8_frozen_confirmatory_matched_adapter",
                "implementation_commit": implementation_commit,
                "model_id": PHASE8_MODEL_ID,
                "model_revision": PHASE8_MODEL_REVISION,
                "seed": seed,
                "layer_index": LAYER_INDEX,
                "rank": adapter.down.out_features,
                "learned_parameters": adapter.trainable_parameter_count,
                "down_weight": adapter.down.weight.detach().cpu(),
                "up_weight": adapter.up.weight.detach().cpu(),
            },
            checkpoint_path,
        )
        record = {
            "seed": seed,
            "training": training_record,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "wall_time_seconds": time.perf_counter() - seed_started,
        }
        seed_records.append(record)
        print(json.dumps(record, indent=2), flush=True)
        del bundle, adapter
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    payload = {
        "status": "confirmatory_matched_adapter_training_complete",
        "implementation_commit": implementation_commit,
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "training_seeds": list(PHASE8_TRAINING_SEEDS),
        "layer_index": LAYER_INDEX,
        "rank": 14,
        "learned_parameters": LEARNED_PARAMETERS,
        "seed_records": seed_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
