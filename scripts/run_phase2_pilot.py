from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from neural_firmware.pretrained_data import (
    make_addition_examples,
    make_carry_examples,
    make_routing_negatives,
)
from neural_firmware.pretrained_evaluation import (
    evaluate_additions,
    evaluate_preservation,
)
from neural_firmware.pretrained_training import (
    BridgeTrainConfig,
    collect_hidden_cache,
    load_model_bundle,
    save_bridge,
    train_bridge,
)


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_pilot_v1.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/pilot_v1"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("phase2_results/pilot_v1"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    started = time.perf_counter()

    train_examples = make_addition_examples(
        count=config["train_examples"],
        min_digits=config["train_min_digits"],
        max_digits=config["train_max_digits"],
        seed=config["train_seed"],
        split="train",
    )
    negative_train = make_routing_negatives(
        count=config["routing_negative_train_examples"],
        seed=config["train_seed"] + 1,
    )
    evaluation_config = config["evaluation"]
    eval_seed = config["eval_seed"]
    eval_sets = {}
    for offset, (split, bounds) in enumerate(
        evaluation_config["splits"].items()
    ):
        if split == "carry_chain":
            eval_sets[split] = make_carry_examples(
                count=evaluation_config["carry_examples"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=eval_seed + offset,
            )
        else:
            eval_sets[split] = make_addition_examples(
                count=evaluation_config["examples_per_random_split"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=eval_seed + offset,
                split=split,
            )
    routing_eval = make_routing_negatives(
        count=evaluation_config["routing_negative_examples"],
        seed=eval_seed + 100,
    )
    logical_dataset = {
        "train": [example.to_dict() for example in train_examples],
        "negative_train": negative_train,
        "evaluation": {
            split: [example.to_dict() for example in examples]
            for split, examples in eval_sets.items()
        },
        "routing_eval": routing_eval,
    }
    dataset_hash = stable_hash(logical_dataset)
    write_json(args.artifact_directory / "logical_dataset.json", logical_dataset)

    bundle = load_model_bundle(config["model_id"])
    cache = collect_hidden_cache(
        bundle,
        train_examples,
        negative_train,
        batch_size=config["hidden_cache_batch_size"],
    )
    cache.save(args.artifact_directory / "hidden_cache.pt")
    bridge_config = config["bridge"]
    bridge, train_result = train_bridge(
        bundle,
        cache,
        BridgeTrainConfig(
            seed=config["train_seed"],
            steps=bridge_config["steps"],
            batch_size=bridge_config["batch_size"],
            learning_rate=bridge_config["learning_rate"],
            strength=bridge_config["strength"],
            router_loss_weight=bridge_config["router_loss_weight"],
        ),
    )
    save_bridge(bridge, train_result, args.artifact_directory)

    summaries: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split, examples in eval_sets.items():
        for mode in ("base", "firmware_off", "latent", "direct"):
            active_bridge = bridge if mode in {"firmware_off", "latent"} else None
            summary, results = evaluate_additions(
                bundle,
                examples,
                mode=mode,
                bridge=active_bridge,
            )
            summaries.append(summary)
            prediction_rows.extend(
                [
                    {"split": split, "mode": mode, **result.to_dict()}
                    for result in results
                ]
            )

    preservation_summary, preservation_rows = evaluate_preservation(
        bundle,
        routing_eval,
        bridge=bridge,
    )
    run_manifest = {
        "experiment_name": config["experiment_name"],
        "pilot": True,
        "config_path": str(args.config),
        "config_sha256": stable_hash(config),
        "logical_dataset_sha256": dataset_hash,
        "source_commit_before_run": git_output("rev-parse", "HEAD"),
        "source_status_before_run": git_output("status", "--short"),
        "model_id": config["model_id"],
        "train_result": asdict(train_result),
        "summaries": summaries,
        "preservation_summary": preservation_summary,
        "wall_time_seconds": time.perf_counter() - started,
        "max_resident_set_size_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(args.artifact_directory / "run_manifest.json", run_manifest)
    write_json(args.artifact_directory / "predictions.json", prediction_rows)
    write_json(
        args.artifact_directory / "preservation_predictions.json",
        preservation_rows,
    )
    write_json(args.result_directory / "summary.json", run_manifest)
    write_json(args.result_directory / "config.json", config)
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
