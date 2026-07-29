from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase11_routing import RequestRouteFeatureSet
from neural_firmware.phase12_routing import (
    PHASE12_ROUTER_CONDITIONS,
    combine_request_feature_sets,
    concatenate_request_views,
    evaluate_phase12_condition,
    repeated_initial_rows,
    subset_request_features,
    train_phase12_condition,
)
from neural_firmware.phase11_data import build_phase11_confirmatory_examples

BASE_CACHE_PATH = Path(
    "phase11_artifacts/cache/request_route_features.pt"
)
DISCLOSED_CACHE_PATH = Path(
    "phase12_artifacts/cache/disclosed_phase11_features.pt"
)
PHASE11_RESULT_PATH = Path("phase11_results/confirmation.json")
SOURCE_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
FOLD_DIRECTORY = Path("phase12_artifacts/development_family_cv")
DEPLOYMENT_DIRECTORY = Path("phase12_artifacts/development_deployment")
RESULT_PATH = Path("phase12_results/development_family_cv.json")
SEEDS = (16_201, 16_202, 16_203)
FOLDS = 5


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


def load_base_views(
    cache: dict[str, object],
    *,
    split: str,
    seed: int,
) -> dict[str, RequestRouteFeatureSet]:
    return {
        kind: RequestRouteFeatureSet.load_state_dict(state)
        for kind, state in cache["splits"][split]["features"][str(seed)].items()
    }


def load_disclosed_views(
    cache: dict[str, object],
    *,
    seed: int,
) -> dict[str, RequestRouteFeatureSet]:
    return {
        kind: RequestRouteFeatureSet.load_state_dict(state)
        for kind, state in cache["features"][str(seed)].items()
    }


def compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "probabilities"}
    }


def family_fold_indices() -> torch.Tensor:
    examples = build_phase11_confirmatory_examples()
    return torch.tensor(
        [example.family_index % FOLDS for example in examples],
        dtype=torch.long,
    )


def condition_summary(
    phase11: dict[str, object],
    *,
    seed: int,
    predictions: torch.Tensor,
) -> dict[str, object]:
    if len(predictions) != len(phase11["rows"]):
        raise ValueError("out-of-fold predictions do not match Phase 11 rows")
    key = str(seed)
    positive_routes = 0
    false_routes = 0
    exact = 0
    categories: dict[str, dict[str, int]] = {}
    for row, prediction in zip(
        phase11["rows"],
        predictions.tolist(),
        strict=True,
    ):
        positive = bool(row["route_label"])
        category = row["split"].removeprefix(
            "phase11_confirmatory_positive_"
        ).removeprefix("phase11_confirmatory_negative_")
        category_key = (
            f"positive_{category}" if positive else f"negative_{category}"
        )
        aggregate = categories.setdefault(
            category_key,
            {"examples": 0, "routes": 0, "counterfactual_exact": 0},
        )
        aggregate["examples"] += 1
        aggregate["routes"] += int(prediction)
        if positive:
            positive_routes += int(prediction)
            oracle_exact = row["conditions"]["phase11_candidate"][key][
                "oracle_route"
            ]["format_exact"]
            row_exact = bool(prediction and oracle_exact)
            exact += int(row_exact)
            aggregate["counterfactual_exact"] += int(row_exact)
        else:
            false_routes += int(prediction)
    return {
        "positive_routes": positive_routes,
        "counterfactual_exact": exact,
        "phase11_natural_exact": phase11["conditions"][
            "phase11_candidate"
        ][key]["exact"],
        "paired_counterfactual_gain": (
            exact
            - phase11["conditions"]["phase11_candidate"][key]["exact"]
        ),
        "false_routes": false_routes,
        "predicted_preserved": 200 - false_routes,
        "categories": categories,
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
    phase11 = json.loads(PHASE11_RESULT_PATH.read_text())
    folds = family_fold_indices()
    if torch.bincount(folds, minlength=FOLDS).tolist() != [60] * FOLDS:
        raise ValueError("Phase 12 family folds must contain 60 prompts each")
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    FOLD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    deployment_records: list[dict[str, object]] = []
    condition_names = tuple(PHASE12_ROUTER_CONDITIONS)
    for seed_index, seed in enumerate(SEEDS):
        source_path = (
            SOURCE_DIRECTORY / f"linear_representation_seed_{seed}.pt"
        )
        source = torch.load(source_path, map_location="cpu", weights_only=True)
        base_training_views = load_base_views(
            base_cache,
            split="training",
            seed=seed,
        )
        base_calibration_views = load_base_views(
            base_cache,
            split="calibration",
            seed=seed,
        )
        phase10_views = load_base_views(
            base_cache,
            split="selection",
            seed=seed,
        )
        disclosed_views = load_disclosed_views(
            disclosed_cache,
            seed=seed,
        )
        for condition_index, condition in enumerate(condition_names):
            kinds = PHASE12_ROUTER_CONDITIONS[condition]
            base_training = concatenate_request_views(
                base_training_views,
                kinds,
            )
            base_calibration = concatenate_request_views(
                base_calibration_views,
                kinds,
            )
            phase10_calibration = concatenate_request_views(
                phase10_views,
                kinds,
            )
            disclosed = concatenate_request_views(disclosed_views, kinds)
            initial_rows = repeated_initial_rows(
                source["input_rows"][:2],
                views=len(kinds),
            )
            oof_predictions = torch.zeros(
                disclosed.rows,
                dtype=torch.bool,
            )
            oof_probabilities = torch.zeros(
                disclosed.rows,
                dtype=torch.float32,
            )
            fold_records: list[dict[str, object]] = []
            for fold in range(FOLDS):
                train_indices = torch.where(folds != fold)[0]
                held_indices = torch.where(folds == fold)[0]
                training = combine_request_feature_sets(
                    base_training,
                    subset_request_features(disclosed, train_indices),
                )
                router_seed = (
                    18_201
                    + seed_index
                    + 100 * condition_index
                    + 10 * fold
                )
                state, training_metrics, calibration_metrics = (
                    train_phase12_condition(
                        condition,
                        initial_rows,
                        training,
                        base_calibration,
                        device=device,
                        seed=router_seed,
                        steps=1_500,
                    )
                )
                held = subset_request_features(disclosed, held_indices)
                held_metrics = evaluate_phase12_condition(
                    state,
                    held,
                    threshold=float(calibration_metrics["threshold"]),
                )
                oof_predictions[held_indices] = held_metrics["predictions"]
                oof_probabilities[held_indices] = held_metrics[
                    "probabilities"
                ]
                checkpoint_path = (
                    FOLD_DIRECTORY
                    / f"{condition}_seed_{seed}_fold_{fold}.pt"
                )
                checkpoint = {
                    "condition": condition,
                    "phase10_seed": seed,
                    "fold": fold,
                    "router_seed": router_seed,
                    "views": list(kinds),
                    "state": state,
                    "threshold": float(calibration_metrics["threshold"]),
                    "training": training_metrics,
                    "calibration": calibration_metrics,
                    "implementation_commit": implementation_commit,
                    "source_checkpoint": str(source_path),
                    "source_checkpoint_sha256": sha256(source_path),
                }
                torch.save(checkpoint, checkpoint_path)
                fold_record = {
                    "fold": fold,
                    "router_seed": router_seed,
                    "training_rows": training.rows,
                    "held_rows": held.rows,
                    "training": training_metrics,
                    "calibration": calibration_metrics,
                    "held": compact_metrics(held_metrics),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
                fold_records.append(fold_record)
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "phase10_seed": seed,
                            **fold_record,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            summary = condition_summary(
                phase11,
                seed=seed,
                predictions=oof_predictions,
            )
            record = {
                "condition": condition,
                "phase10_seed": seed,
                "views": list(kinds),
                "learned_parameters": fold_records[0]["training"][
                    "trainable_parameters"
                ],
                "folds": fold_records,
                "out_of_fold": {
                    **summary,
                    "probability_positive_median": float(
                        oof_probabilities[disclosed.targets == 1].median()
                    ),
                    "probability_negative_median": float(
                        oof_probabilities[disclosed.targets == 0].median()
                    ),
                    "probability_negative_maximum": float(
                        oof_probabilities[disclosed.targets == 0].max()
                    ),
                },
            }
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)

    by_condition = {
        condition: [
            record
            for record in records
            if record["condition"] == condition
        ]
        for condition in condition_names
    }
    summaries = {
        condition: {
            "counterfactual_exact": [
                record["out_of_fold"]["counterfactual_exact"]
                for record in condition_records
            ],
            "paired_counterfactual_gains": [
                record["out_of_fold"]["paired_counterfactual_gain"]
                for record in condition_records
            ],
            "positive_routes": [
                record["out_of_fold"]["positive_routes"]
                for record in condition_records
            ],
            "false_routes": [
                record["out_of_fold"]["false_routes"]
                for record in condition_records
            ],
            "mean_counterfactual_exact": statistics.fmean(
                record["out_of_fold"]["counterfactual_exact"]
                for record in condition_records
            ),
            "learned_parameters": condition_records[0][
                "learned_parameters"
            ],
        }
        for condition, condition_records in by_condition.items()
    }
    eligible = [
        condition
        for condition, summary in summaries.items()
        if max(summary["false_routes"]) <= 4
    ]
    selected = (
        max(
            eligible,
            key=lambda condition: (
                min(summaries[condition]["counterfactual_exact"]),
                summaries[condition]["mean_counterfactual_exact"],
                -summaries[condition]["learned_parameters"],
            ),
        )
        if eligible
        else None
    )

    if selected is not None:
        kinds = PHASE12_ROUTER_CONDITIONS[selected]
        for seed_index, seed in enumerate(SEEDS):
            source_path = (
                SOURCE_DIRECTORY
                / f"linear_representation_seed_{seed}.pt"
            )
            source = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            base_training_views = load_base_views(
                base_cache,
                split="training",
                seed=seed,
            )
            base_calibration_views = load_base_views(
                base_cache,
                split="calibration",
                seed=seed,
            )
            phase10_views = load_base_views(
                base_cache,
                split="selection",
                seed=seed,
            )
            disclosed_views = load_disclosed_views(
                disclosed_cache,
                seed=seed,
            )
            base_training = concatenate_request_views(
                base_training_views,
                kinds,
            )
            disclosed = concatenate_request_views(disclosed_views, kinds)
            deployment_training = combine_request_feature_sets(
                base_training,
                disclosed,
            )
            deployment_calibration = combine_request_feature_sets(
                concatenate_request_views(base_calibration_views, kinds),
                concatenate_request_views(phase10_views, kinds),
            )
            initial_rows = repeated_initial_rows(
                source["input_rows"][:2],
                views=len(kinds),
            )
            router_seed = 19_201 + seed_index
            state, training_metrics, calibration_metrics = (
                train_phase12_condition(
                    selected,
                    initial_rows,
                    deployment_training,
                    deployment_calibration,
                    device=device,
                    seed=router_seed,
                    steps=2_500,
                    maximum_calibration_false_positive_rate=0.005,
                )
            )
            disclosed_metrics = evaluate_phase12_condition(
                state,
                disclosed,
                threshold=float(calibration_metrics["threshold"]),
            )
            checkpoint_path = (
                DEPLOYMENT_DIRECTORY
                / f"{selected}_seed_{seed}.pt"
            )
            checkpoint = {
                "stage": "phase12_development_deployment_router",
                "condition": selected,
                "phase10_seed": seed,
                "router_seed": router_seed,
                "views": list(kinds),
                "state": state,
                "threshold": float(calibration_metrics["threshold"]),
                "temperature": 2.0,
                "training": training_metrics,
                "calibration": calibration_metrics,
                "implementation_commit": implementation_commit,
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": sha256(source_path),
            }
            torch.save(checkpoint, checkpoint_path)
            deployment_record = {
                "condition": selected,
                "phase10_seed": seed,
                "router_seed": router_seed,
                "training": training_metrics,
                "calibration": calibration_metrics,
                "disclosed_phase11_fitted_evaluation": compact_metrics(
                    disclosed_metrics
                ),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            }
            deployment_records.append(deployment_record)
            print(json.dumps(deployment_record, indent=2), flush=True)

    payload = {
        "status": "phase12_family_held_out_development_complete",
        "implementation_commit": implementation_commit,
        "base_feature_cache": str(BASE_CACHE_PATH),
        "base_feature_cache_sha256": sha256(BASE_CACHE_PATH),
        "disclosed_feature_cache": str(DISCLOSED_CACHE_PATH),
        "disclosed_feature_cache_sha256": sha256(DISCLOSED_CACHE_PATH),
        "selection_source": "disclosed_phase11_family_held_out",
        "fold_rule": (
            "family_index modulo 5 within positive and negative strata; "
            "each fold contains 60 prompts from 4 positive and 10 negative "
            "families"
        ),
        "conditions": list(condition_names),
        "selection_rule": (
            "among conditions with at most 4/200 out-of-fold false routes "
            "in every seed, maximize worst-seed counterfactual exact, then "
            "mean exact, then minimize learned parameters"
        ),
        "summaries": summaries,
        "selected_condition": selected,
        "records": records,
        "deployment_records": deployment_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"records", "deployment_records"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
