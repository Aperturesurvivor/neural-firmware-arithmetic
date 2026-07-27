from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import SequenceImplantLayout
from neural_firmware.phase7_sequence_training import (
    collect_first_step_route_features,
    collect_sequence_features,
)
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_data import (
    build_phase9_development,
    build_phase9_generic_training,
    build_phase9_hard_training,
)
from neural_firmware.pretrained_training import load_model_bundle

SELECTION_PATH = Path("phase8_artifacts/development_selection.pt")
CACHE_PATH = Path("phase9_artifacts/cache/interface_features.pt")
MANIFEST_PATH = Path("phase9_results/development_feature_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-hard",
        action="store_true",
        help="reuse unchanged generic/development features and refresh hard features",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    selection = torch.load(SELECTION_PATH, map_location="cpu", weights_only=True)
    layer_index = int(selection["selected_layer"])
    generic = build_phase9_generic_training()
    hard = build_phase9_hard_training()
    development = build_phase9_development()
    bundle = load_model_bundle(PHASE8_MODEL_ID, revision=PHASE8_MODEL_REVISION)
    if arguments.only_hard:
        payload = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
        if (
            payload["model_id"] != PHASE8_MODEL_ID
            or payload["model_revision"] != PHASE8_MODEL_REVISION
            or int(payload["layer_index"]) != layer_index
        ):
            raise ValueError("existing Phase 9 cache does not match frozen model")
        work = (("hard", hard),)
    else:
        payload = {
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "layer_index": layer_index,
            "ordinary_tokens_per_example": 12,
        }
        work = (
            ("generic", generic),
            ("hard", hard),
            ("development", development),
        )
    for name, examples in work:
        print(f"collecting Phase 9 {name} sequence features", flush=True)
        sequence = collect_sequence_features(
            bundle,
            examples,
            layer_index=layer_index,
            layout=SequenceImplantLayout(**selection["layout"]),
            batch_size=4,
            ordinary_tokens_per_example=12,
        )
        print(f"collecting Phase 9 {name} route features", flush=True)
        route = collect_first_step_route_features(
            bundle,
            examples,
            layer_index=layer_index,
            batch_size=8,
        )
        payload[name] = {
            "examples": len(examples),
            "sequence": sequence.state_dict(),
            "route": route.state_dict(),
        }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE_PATH)
    manifest = {
        "status": "phase9_development_features_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "layer_index": layer_index,
        "cache": str(CACHE_PATH),
        "cache_sha256": sha256(CACHE_PATH),
        "cache_bytes": CACHE_PATH.stat().st_size,
        "counts": {
            name: payload[name]["examples"]
            for name in ("generic", "hard", "development")
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
