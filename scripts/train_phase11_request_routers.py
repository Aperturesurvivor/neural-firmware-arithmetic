from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase11_routing import (
    REQUEST_ROUTER_KINDS,
    RequestRouteFeatureSet,
    RequestRouterTrainConfig,
    evaluate_request_router,
    phase11_checkpoint_state,
    train_request_router,
)

CACHE_PATH = Path("phase11_artifacts/cache/request_route_features.pt")
SOURCE_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
OUTPUT_DIRECTORY = Path("phase11_artifacts/development_routers")
PHASE10_RESULTS = Path("phase10_results/confirmation.json")
RESULT_PATH = Path("phase11_results/development_training.json")
SEEDS = (16_201, 16_202, 16_203)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def load_features(
    cache: dict[str, object],
    split: str,
    seed: int,
    kind: str,
) -> RequestRouteFeatureSet:
    return RequestRouteFeatureSet.load_state_dict(
        cache["splits"][split]["features"][str(seed)][kind]
    )


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "probabilities"}
    }


def counterfactual_metrics(
    phase10: dict[str, object],
    *,
    seed: int,
    predictions: torch.Tensor,
) -> dict[str, object]:
    key = str(seed)
    rows = phase10["rows"]
    if len(rows) != len(predictions):
        raise ValueError("selection predictions do not match Phase 10 rows")
    exact = 0
    false_routes = 0
    positive_routes = 0
    categories: dict[str, dict[str, int]] = {}
    for row, prediction in zip(rows, predictions.tolist(), strict=True):
        positive = bool(row["route_label"])
        category = row["split"].removeprefix(
            "phase10_confirmatory_positive_"
        ).removeprefix("phase10_confirmatory_negative_")
        aggregate = categories.setdefault(
            category,
            {"examples": 0, "routes": 0, "counterfactual_exact": 0},
        )
        aggregate["examples"] += 1
        aggregate["routes"] += int(prediction)
        if positive:
            positive_routes += int(prediction)
            oracle_exact = row["conditions"]["linear_representation"][key][
                "oracle_route"
            ]["format_exact"]
            row_exact = bool(prediction and oracle_exact)
            exact += int(row_exact)
            aggregate["counterfactual_exact"] += int(row_exact)
        else:
            false_routes += int(prediction)
    baseline = phase10["conditions"]["linear_representation"][key]["exact"]
    return {
        "positive_routes": positive_routes,
        "counterfactual_exact": exact,
        "phase10_natural_exact": baseline,
        "paired_counterfactual_gain": exact - baseline,
        "false_routes": false_routes,
        "predicted_preserved": 200 - false_routes,
        "categories": categories,
    }


def main() -> None:
    started = time.perf_counter()
    implementation_commit = git_commit()
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    phase10 = json.loads(PHASE10_RESULTS.read_text())
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        source_path = (
            SOURCE_DIRECTORY / f"linear_representation_seed_{seed}.pt"
        )
        source = torch.load(source_path, map_location="cpu", weights_only=True)
        for kind_index, kind in enumerate(REQUEST_ROUTER_KINDS):
            train = load_features(cache, "training", seed, kind)
            calibration = load_features(cache, "calibration", seed, kind)
            selection = load_features(cache, "selection", seed, kind)
            rows, training, calibration_metrics = train_request_router(
                source["input_rows"][:2],
                train,
                calibration,
                device=torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                ),
                config=RequestRouterTrainConfig(
                    seed=17_201 + (seed - 16_201) + 100 * kind_index,
                    steps=2_500,
                    batch_size=256,
                    learning_rate=0.0005,
                    route_temperature=2.0,
                    maximum_calibration_false_positive_rate=0.01,
                ),
            )
            threshold = float(calibration_metrics["threshold"])
            selection_metrics = evaluate_request_router(
                rows,
                selection,
                threshold=threshold,
                temperature=2.0,
            )
            counterfactual = counterfactual_metrics(
                phase10,
                seed=seed,
                predictions=selection_metrics["predictions"],
            )
            checkpoint_path = OUTPUT_DIRECTORY / f"{kind}_seed_{seed}.pt"
            checkpoint = {
                **phase11_checkpoint_state(
                    source,
                    router_kind=kind,
                    request_route_rows=rows,
                    request_route_threshold=threshold,
                    request_route_temperature=2.0,
                    request_tail_tokens=8,
                ),
                "stage": "phase11_development_request_router",
                "implementation_commit": implementation_commit,
                "phase11_router_seed": training["config"]["seed"],
                "source_phase10_checkpoint": str(source_path),
                "source_phase10_checkpoint_sha256": sha256(source_path),
            }
            torch.save(checkpoint, checkpoint_path)
            record = {
                "router_kind": kind,
                "phase10_seed": seed,
                "phase11_router_seed": training["config"]["seed"],
                "source_phase10_checkpoint": str(source_path),
                "source_phase10_checkpoint_sha256": sha256(source_path),
                "training": training,
                "calibration": calibration_metrics,
                "selection": compact_metrics(selection_metrics),
                "counterfactual": counterfactual,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)
    by_kind = {
        kind: [
            record
            for record in records
            if record["router_kind"] == kind
        ]
        for kind in REQUEST_ROUTER_KINDS
    }
    summaries = {
        kind: {
            "counterfactual_exact": [
                record["counterfactual"]["counterfactual_exact"]
                for record in kind_records
            ],
            "paired_counterfactual_gains": [
                record["counterfactual"]["paired_counterfactual_gain"]
                for record in kind_records
            ],
            "false_routes": [
                record["counterfactual"]["false_routes"]
                for record in kind_records
            ],
            "mean_counterfactual_exact": statistics.fmean(
                record["counterfactual"]["counterfactual_exact"]
                for record in kind_records
            ),
        }
        for kind, kind_records in by_kind.items()
    }
    eligible = [
        kind
        for kind, summary in summaries.items()
        if max(summary["false_routes"]) <= 4
    ]
    selected = (
        max(
            eligible,
            key=lambda kind: (
                min(summaries[kind]["counterfactual_exact"]),
                summaries[kind]["mean_counterfactual_exact"],
            ),
        )
        if eligible
        else None
    )
    payload = {
        "status": "phase11_request_router_development_complete",
        "implementation_commit": implementation_commit,
        "feature_cache": str(CACHE_PATH),
        "feature_cache_sha256": sha256(CACHE_PATH),
        "selection_source": "disclosed_phase10_confirmation",
        "conditions": list(REQUEST_ROUTER_KINDS),
        "selection_rule": (
            "among conditions with at most 4/200 Phase 10 false routes in "
            "every seed, maximize the worst-seed counterfactual exact count, "
            "then the mean"
        ),
        "summaries": summaries,
        "selected_router_kind": selected,
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
