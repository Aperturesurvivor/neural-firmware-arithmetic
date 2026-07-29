from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase12_routing import (
    PHASE12_ROUTER_CONDITIONS,
    SiluRouterTrainConfig,
    combine_request_feature_sets,
    concatenate_request_views,
    evaluate_silu_router,
    train_silu_router,
)
from run_phase12_family_cv import (
    BASE_CACHE_PATH,
    DISCLOSED_CACHE_PATH,
    SEEDS,
    load_base_views,
    load_disclosed_views,
)

SOURCE_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
OUTPUT_DIRECTORY = Path("phase12_artifacts/deployment_routers")
RESULT_PATH = Path("phase12_results/deployment_training.json")
CONDITION = "all_views_silu16"
ROUTER_SEEDS = {
    16_201: 21_201,
    16_202: 21_202,
    16_203: 21_203,
}
FIXED_THRESHOLD = 0.60
ROUTE_TEMPERATURE = 2.0


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


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "probabilities"}
    }


def main() -> None:
    started = time.perf_counter()
    implementation_commit = git_commit()
    base_cache = torch.load(
        BASE_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    disclosed_cache = torch.load(
        DISCLOSED_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    kinds = PHASE12_ROUTER_CONDITIONS[CONDITION]
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        source_path = (
            SOURCE_DIRECTORY / f"linear_representation_seed_{seed}.pt"
        )
        source = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        base_training = concatenate_request_views(
            load_base_views(base_cache, split="training", seed=seed),
            kinds,
        )
        disclosed = concatenate_request_views(
            load_disclosed_views(disclosed_cache, seed=seed),
            kinds,
        )
        training = combine_request_feature_sets(base_training, disclosed)
        calibration = combine_request_feature_sets(
            concatenate_request_views(
                load_base_views(base_cache, split="calibration", seed=seed),
                kinds,
            ),
            concatenate_request_views(
                load_base_views(base_cache, split="selection", seed=seed),
                kinds,
            ),
        )
        state, training_metrics, automatic_calibration = train_silu_router(
            training,
            calibration,
            device=device,
            config=SiluRouterTrainConfig(
                seed=ROUTER_SEEDS[seed],
                bottleneck_width=16,
                steps=1_500,
                batch_size=256,
                learning_rate=0.0005,
                route_temperature=ROUTE_TEMPERATURE,
                maximum_calibration_false_positive_rate=0.005,
            ),
        )
        fixed_training = compact_metrics(
            evaluate_silu_router(
                state,
                training,
                threshold=FIXED_THRESHOLD,
                temperature=ROUTE_TEMPERATURE,
            )
        )
        fixed_calibration = compact_metrics(
            evaluate_silu_router(
                state,
                calibration,
                threshold=FIXED_THRESHOLD,
                temperature=ROUTE_TEMPERATURE,
            )
        )
        fixed_disclosed = compact_metrics(
            evaluate_silu_router(
                state,
                disclosed,
                threshold=FIXED_THRESHOLD,
                temperature=ROUTE_TEMPERATURE,
            )
        )
        checkpoint = {
            **source,
            "stage": "phase12_development_deployment_router",
            "implementation_commit": implementation_commit,
            "request_router_kind": CONDITION,
            "request_route_down": state.down,
            "request_route_output": state.output,
            "request_route_threshold": FIXED_THRESHOLD,
            "request_route_temperature": ROUTE_TEMPERATURE,
            "request_tail_tokens": 8,
            "phase12_router_seed": ROUTER_SEEDS[seed],
            "source_phase10_checkpoint": str(source_path),
            "source_phase10_checkpoint_sha256": sha256(source_path),
        }
        checkpoint_path = OUTPUT_DIRECTORY / f"{CONDITION}_seed_{seed}.pt"
        torch.save(checkpoint, checkpoint_path)
        record = {
            "condition": CONDITION,
            "phase10_seed": seed,
            "phase12_router_seed": ROUTER_SEEDS[seed],
            "views": list(kinds),
            "fixed_threshold": FIXED_THRESHOLD,
            "route_temperature": ROUTE_TEMPERATURE,
            "training_rows": training.rows,
            "calibration_rows": calibration.rows,
            "disclosed_rows": disclosed.rows,
            "training": training_metrics,
            "automatic_calibration_not_used": automatic_calibration,
            "fixed_threshold_training": fixed_training,
            "fixed_threshold_calibration": fixed_calibration,
            "fixed_threshold_disclosed_phase11": fixed_disclosed,
            "source_phase10_checkpoint": str(source_path),
            "source_phase10_checkpoint_sha256": sha256(source_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)

    payload = {
        "status": "phase12_development_deployment_training_complete",
        "implementation_commit": implementation_commit,
        "condition": CONDITION,
        "views": list(kinds),
        "router_seeds": {
            str(seed): router_seed
            for seed, router_seed in ROUTER_SEEDS.items()
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_provenance": (
            "selected post-hoc on disclosed Phase 11 development data"
        ),
        "route_temperature": ROUTE_TEMPERATURE,
        "training_source": (
            "Phase 8 plus Phase 9 hard training plus disclosed Phase 11"
        ),
        "calibration_source": (
            "Phase 9 development plus disclosed Phase 10 confirmation; "
            "reported but not used to replace the fixed threshold"
        ),
        "base_feature_cache": str(BASE_CACHE_PATH),
        "base_feature_cache_sha256": sha256(BASE_CACHE_PATH),
        "disclosed_feature_cache": str(DISCLOSED_CACHE_PATH),
        "disclosed_feature_cache_sha256": sha256(DISCLOSED_CACHE_PATH),
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
