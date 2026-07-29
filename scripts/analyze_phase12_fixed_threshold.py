from __future__ import annotations

import json
from pathlib import Path

import torch

from neural_firmware.phase12_routing import (
    PHASE12_ROUTER_CONDITIONS,
    concatenate_request_views,
    evaluate_phase12_condition,
)
from run_phase12_family_cv import (
    DISCLOSED_CACHE_PATH,
    PHASE11_RESULT_PATH,
    SEEDS,
    condition_summary,
    family_fold_indices,
    load_disclosed_views,
)

CONDITION = "all_views_silu16"
CHECKPOINT_DIRECTORY = Path(
    "phase12_artifacts/development_nested_family_cv"
)
RESULT_PATH = Path(
    "phase12_results/development_fixed_threshold_analysis.json"
)
THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
MAXIMUM_FALSE_ROUTES = 4


def main() -> None:
    cache = torch.load(
        DISCLOSED_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    phase11 = json.loads(PHASE11_RESULT_PATH.read_text())
    folds = family_fold_indices()
    records: list[dict[str, object]] = []
    probabilities_by_seed: dict[int, torch.Tensor] = {}
    for seed in SEEDS:
        features = concatenate_request_views(
            load_disclosed_views(cache, seed=seed),
            PHASE12_ROUTER_CONDITIONS[CONDITION],
        )
        probabilities = torch.zeros(features.rows, dtype=torch.float32)
        for evaluation_fold in range(5):
            calibration_fold = (evaluation_fold + 1) % 5
            checkpoint_path = CHECKPOINT_DIRECTORY / (
                f"{CONDITION}_seed_{seed}_"
                f"evaluation_{evaluation_fold}_"
                f"calibration_{calibration_fold}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            held_indices = torch.where(folds == evaluation_fold)[0]
            held = type(features)(
                hidden=features.hidden[held_indices],
                targets=features.targets[held_indices],
            )
            metrics = evaluate_phase12_condition(
                checkpoint["state"],
                held,
                threshold=0.5,
            )
            probabilities[held_indices] = metrics["probabilities"]
        probabilities_by_seed[seed] = probabilities

    for threshold in THRESHOLDS:
        seed_records = []
        for seed in SEEDS:
            summary = condition_summary(
                phase11,
                seed=seed,
                predictions=probabilities_by_seed[seed] >= threshold,
            )
            seed_records.append({"phase10_seed": seed, **summary})
        records.append(
            {
                "threshold": threshold,
                "eligible": all(
                    record["false_routes"] <= MAXIMUM_FALSE_ROUTES
                    for record in seed_records
                ),
                "seeds": seed_records,
            }
        )

    eligible = [record for record in records if record["eligible"]]
    selected = min(eligible, key=lambda record: record["threshold"])
    payload = {
        "status": "phase12_disclosed_fixed_threshold_analysis_complete",
        "interpretation": (
            "post-hoc development hyperparameter analysis; not held-out "
            "or confirmatory evidence"
        ),
        "condition": CONDITION,
        "threshold_grid": list(THRESHOLDS),
        "selection_rule": (
            "lowest 0.05-grid threshold with at most 4/200 false routes "
            "in every seed"
        ),
        "selected_threshold": selected["threshold"],
        "selected_seed_metrics": selected["seeds"],
        "records": records,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
