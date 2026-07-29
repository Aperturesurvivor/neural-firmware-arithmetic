from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from neural_firmware.phase12_routing import (
    PHASE12_ROUTER_CONDITIONS,
    combine_request_feature_sets,
    concatenate_request_views,
    evaluate_phase12_condition,
    repeated_initial_rows,
    subset_request_features,
    train_phase12_condition,
)
from run_phase12_family_cv import (
    BASE_CACHE_PATH,
    DISCLOSED_CACHE_PATH,
    PHASE11_RESULT_PATH,
    SEEDS,
    SOURCE_DIRECTORY,
    compact_metrics,
    condition_summary,
    family_fold_indices,
    git_commit,
    load_base_views,
    load_disclosed_views,
    sha256,
)

OUTPUT_DIRECTORY = Path("phase12_artifacts/development_nested_family_cv")
RESULT_PATH = Path("phase12_results/development_nested_family_cv.json")
FOLDS = 5


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
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    condition_names = tuple(PHASE12_ROUTER_CONDITIONS)
    records: list[dict[str, object]] = []
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
            for evaluation_fold in range(FOLDS):
                calibration_fold = (evaluation_fold + 1) % FOLDS
                train_indices = torch.where(
                    (folds != evaluation_fold)
                    & (folds != calibration_fold)
                )[0]
                calibration_indices = torch.where(
                    folds == calibration_fold
                )[0]
                held_indices = torch.where(
                    folds == evaluation_fold
                )[0]
                training = combine_request_feature_sets(
                    base_training,
                    subset_request_features(disclosed, train_indices),
                )
                calibration = combine_request_feature_sets(
                    base_calibration,
                    subset_request_features(
                        disclosed,
                        calibration_indices,
                    ),
                )
                router_seed = (
                    20_201
                    + seed_index
                    + 100 * condition_index
                    + 10 * evaluation_fold
                )
                state, training_metrics, calibration_metrics = (
                    train_phase12_condition(
                        condition,
                        initial_rows,
                        training,
                        calibration,
                        device=device,
                        seed=router_seed,
                        steps=1_500,
                        maximum_calibration_false_positive_rate=0.005,
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
                    OUTPUT_DIRECTORY
                    / (
                        f"{condition}_seed_{seed}_"
                        f"evaluation_{evaluation_fold}_"
                        f"calibration_{calibration_fold}.pt"
                    )
                )
                checkpoint = {
                    "condition": condition,
                    "phase10_seed": seed,
                    "evaluation_fold": evaluation_fold,
                    "calibration_fold": calibration_fold,
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
                    "evaluation_fold": evaluation_fold,
                    "calibration_fold": calibration_fold,
                    "router_seed": router_seed,
                    "training_rows": training.rows,
                    "calibration_rows": calibration.rows,
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
                            "evaluation_fold": evaluation_fold,
                            "calibration_fold": calibration_fold,
                            "held_true_positive_rate": held_metrics[
                                "true_positive_rate"
                            ],
                            "held_false_positive_rate": held_metrics[
                                "false_positive_rate"
                            ],
                            "threshold": calibration_metrics["threshold"],
                        }
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
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "phase10_seed": seed,
                        "out_of_fold": record["out_of_fold"],
                    },
                    indent=2,
                ),
                flush=True,
            )

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
    payload = {
        "status": "phase12_nested_family_held_out_development_complete",
        "implementation_commit": implementation_commit,
        "base_feature_cache": str(BASE_CACHE_PATH),
        "base_feature_cache_sha256": sha256(BASE_CACHE_PATH),
        "disclosed_feature_cache": str(DISCLOSED_CACHE_PATH),
        "disclosed_feature_cache_sha256": sha256(DISCLOSED_CACHE_PATH),
        "selection_source": "disclosed_phase11_nested_family_held_out",
        "fold_rule": (
            "outer family fold is evaluation-only; next cyclic family fold "
            "is threshold-calibration-only; remaining three folds join base "
            "training"
        ),
        "calibration_constraint": (
            "maximum 0.5% false-positive rate on combined Phase 9 "
            "development plus disjoint Phase 11 calibration fold"
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
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "records"
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
