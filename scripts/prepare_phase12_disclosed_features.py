from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase11_data import build_phase11_confirmatory_examples
from neural_firmware.phase11_routing import (
    REQUEST_ROUTER_KINDS,
    collect_request_route_feature_bank,
)
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.pretrained_training import load_model_bundle

SOURCE_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
CACHE_PATH = Path("phase12_artifacts/cache/disclosed_phase11_features.pt")
MANIFEST_PATH = Path("phase12_results/development_feature_manifest.json")
SEEDS = (16_201, 16_202, 16_203)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def main() -> None:
    started = time.perf_counter()
    examples = build_phase11_confirmatory_examples()
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
    bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    print(
        f"collecting disclosed Phase 11 views for {len(examples)} prompts",
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
    payload = {
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "layer_index": 15,
        "source_checkpoint_sha256": source_hashes,
        "router_kinds": list(REQUEST_ROUTER_KINDS),
        "tail_tokens": 8,
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
        "status": "phase12_disclosed_phase11_features_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "layer_index": 15,
        "source": "disclosed_phase11_confirmation",
        "source_checkpoint_sha256": source_hashes,
        "router_kinds": list(REQUEST_ROUTER_KINDS),
        "tail_tokens": 8,
        "examples": len(examples),
        "positive_examples": sum(example.route_label for example in examples),
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
