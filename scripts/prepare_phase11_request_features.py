from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    build_phase8_training_and_development,
)
from neural_firmware.phase9_data import (
    build_phase9_development,
    build_phase9_hard_training,
)
from neural_firmware.phase10_data import build_phase10_confirmatory_examples
from neural_firmware.phase11_routing import (
    REQUEST_ROUTER_KINDS,
    collect_request_route_feature_bank,
)
from neural_firmware.pretrained_training import load_model_bundle

SOURCE_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
CACHE_PATH = Path("phase11_artifacts/cache/request_route_features.pt")
MANIFEST_PATH = Path("phase11_results/development_feature_manifest.json")
SEEDS = (16_201, 16_202, 16_203)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def main() -> None:
    started = time.perf_counter()
    phase8_training, _ = build_phase8_training_and_development()
    splits = {
        "training": phase8_training + build_phase9_hard_training(),
        "calibration": build_phase9_development(),
        "selection": build_phase10_confirmatory_examples(),
    }
    checkpoints = {
        seed: torch.load(
            SOURCE_DIRECTORY / f"linear_representation_seed_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for seed in SEEDS
    }
    source_hashes = {
        str(seed): sha256(
            SOURCE_DIRECTORY / f"linear_representation_seed_{seed}.pt"
        )
        for seed in SEEDS
    }
    layer_indices = {int(checkpoint["layer_index"]) for checkpoint in checkpoints.values()}
    if layer_indices != {15}:
        raise ValueError("Phase 11 sources do not share frozen layer 15")
    bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    payload: dict[str, object] = {
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "layer_index": 15,
        "source_checkpoint_sha256": source_hashes,
        "router_kinds": list(REQUEST_ROUTER_KINDS),
        "tail_tokens": 8,
        "splits": {},
    }
    for name, examples in splits.items():
        print(
            f"collecting {name} request features for {len(examples)} prompts",
            flush=True,
        )
        bank = collect_request_route_feature_bank(
            bundle,
            examples,
            layer_index=15,
            representation_checkpoints=checkpoints,
            batch_size=8,
            tail_tokens=8,
        )
        payload["splits"][name] = {
            "examples": len(examples),
            "positive_examples": sum(example.route_label for example in examples),
            "features": {
                str(seed): {
                    kind: features.state_dict()
                    for kind, features in per_kind.items()
                }
                for seed, per_kind in bank.items()
            },
        }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE_PATH)
    manifest = {
        "status": "phase11_request_features_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "layer_index": 15,
        "source": {
            "training": "phase8_training_plus_phase9_hard_training",
            "calibration": "phase9_development",
            "selection": "disclosed_phase10_confirmation",
        },
        "source_checkpoint_sha256": source_hashes,
        "router_kinds": list(REQUEST_ROUTER_KINDS),
        "tail_tokens": 8,
        "counts": {
            name: {
                "examples": len(examples),
                "positive_examples": sum(
                    example.route_label for example in examples
                ),
            }
            for name, examples in splits.items()
        },
        "cache": str(CACHE_PATH),
        "cache_sha256": sha256(CACHE_PATH),
        "cache_bytes": CACHE_PATH.stat().st_size,
        "wall_time_seconds": time.perf_counter() - started,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
