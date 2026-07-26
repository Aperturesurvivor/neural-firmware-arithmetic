from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.pretrained_data import (
    AdditionExample,
    make_addition_examples,
    make_carry_examples,
    make_routing_negatives,
)
from neural_firmware.pretrained_evaluation import (
    evaluate_additions,
    evaluate_preservation,
    generate_one,
)
from neural_firmware.pretrained_training import (
    AdapterTrainConfig,
    BridgeTrainConfig,
    collect_hidden_cache,
    load_bridge,
    load_learned_adapter,
    load_model_bundle,
    save_bridge,
    save_learned_adapter,
    train_bridge,
    train_learned_adapter,
)


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def release_accelerator_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_evaluation_sets(config: dict[str, object]) -> tuple[
    dict[str, list[AdditionExample]],
    list[str],
]:
    evaluation = config["evaluation"]
    eval_seed = config["eval_seed"]
    eval_sets: dict[str, list[AdditionExample]] = {}
    for offset, (split, bounds) in enumerate(evaluation["splits"].items()):
        if split == "carry_chain":
            eval_sets[split] = make_carry_examples(
                count=evaluation["carry_examples"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=eval_seed + offset,
            )
        else:
            eval_sets[split] = make_addition_examples(
                count=evaluation["examples_per_random_split"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=eval_seed + offset,
                split=split,
            )
    routing_prompts = make_routing_negatives(
        count=evaluation["routing_negative_examples"],
        seed=eval_seed + 100,
    )
    return eval_sets, routing_prompts


def train_examples_for_seed(
    config: dict[str, object],
    seed: int,
) -> tuple[list[AdditionExample], list[str]]:
    arithmetic = make_addition_examples(
        count=config["train_examples"],
        min_digits=config["train_min_digits"],
        max_digits=config["train_max_digits"],
        seed=seed,
        split="train",
    )
    negatives = make_routing_negatives(
        count=config["routing_negative_train_examples"],
        seed=seed + 1,
    )
    return arithmetic, negatives


def evaluate_mode(
    *,
    bundle: object,
    eval_sets: dict[str, list[AdditionExample]],
    mode: str,
    output_directory: Path,
    bridge: object | None = None,
) -> list[dict[str, object]]:
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())["summaries"]
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for split, examples in eval_sets.items():
        summary, results = evaluate_additions(
            bundle,
            examples,
            mode=mode,
            bridge=bridge,
        )
        summaries.append(summary)
        predictions.extend(
            [
                {"mode": mode, "split": split, **result.to_dict()}
                for result in results
            ]
        )
    write_json(output_directory / "predictions.json", predictions)
    write_json(summary_path, {"mode": mode, "summaries": summaries})
    return summaries


def evaluate_base_preservation(
    bundle: object,
    prompts: list[str],
    output_path: Path,
) -> list[dict[str, object]]:
    if output_path.exists():
        return json.loads(output_path.read_text())
    rows = [
        generate_one(
            bundle,
            prompt,
            mode="base",
            max_new_tokens=16,
        ).to_dict()
        for prompt in prompts
    ]
    write_json(output_path, rows)
    return rows


def evaluate_adapter_preservation(
    bundle: object,
    prompts: list[str],
    base_rows: list[dict[str, object]],
    output_directory: Path,
) -> dict[str, float | int]:
    summary_path = output_directory / "preservation_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    rows = []
    for prompt, base_row in zip(prompts, base_rows, strict=True):
        adapter = generate_one(
            bundle,
            prompt,
            mode="learned_adapter",
            max_new_tokens=16,
        )
        rows.append(
            {
                "prompt": prompt,
                "base_text": base_row["generated_text"],
                "adapter_text": adapter.generated_text,
                "token_exact_preserved": (
                    base_row["generated_token_ids"]
                    == adapter.generated_token_ids
                ),
            }
        )
    preserved = sum(row["token_exact_preserved"] for row in rows)
    summary = {
        "prompts": len(rows),
        "token_exact_preserved": preserved,
        "preservation_rate": preserved / len(rows),
    }
    write_json(output_directory / "preservation_predictions.json", rows)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_study.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("phase2_results/confirmatory_v1"),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Pilot/smoke testing only; never use for a confirmatory start.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    args.result_directory.mkdir(parents=True, exist_ok=True)

    state_path = args.artifact_directory / "frozen_state.json"
    if state_path.exists():
        frozen_state = json.loads(state_path.read_text())
        if frozen_state["config_sha256"] != stable_hash(config):
            raise ValueError("configuration changed after confirmatory start")
    else:
        dirty = git_output("status", "--short")
        if dirty and not args.allow_dirty:
            raise RuntimeError(
                "confirmatory study must start from a clean committed worktree"
            )
        frozen_state = {
            "source_commit": git_output("rev-parse", "HEAD"),
            "config_sha256": stable_hash(config),
            "started_at_unix": time.time(),
        }
        write_json(state_path, frozen_state)

    eval_sets, routing_prompts = build_evaluation_sets(config)
    logical_evaluation = {
        "sets": {
            split: [example.to_dict() for example in examples]
            for split, examples in eval_sets.items()
        },
        "routing_prompts": routing_prompts,
    }
    evaluation_hash = stable_hash(logical_evaluation)
    write_json(
        args.artifact_directory / "logical_evaluation.json",
        logical_evaluation,
    )

    base_bundle = load_model_bundle(
        config["model_id"],
        revision=config["model_revision"],
    )
    base_summaries = evaluate_mode(
        bundle=base_bundle,
        eval_sets=eval_sets,
        mode="base",
        output_directory=args.artifact_directory / "base",
    )
    direct_summaries = evaluate_mode(
        bundle=base_bundle,
        eval_sets=eval_sets,
        mode="direct",
        output_directory=args.artifact_directory / "direct",
    )
    base_preservation = evaluate_base_preservation(
        base_bundle,
        routing_prompts,
        args.artifact_directory / "base" / "preservation_predictions.json",
    )

    bridge_run_summaries: list[dict[str, object]] = []
    for seed in config["train_seeds"]:
        seed_directory = args.artifact_directory / f"bridge_seed_{seed}"
        train_examples, negative_train = train_examples_for_seed(config, seed)
        training_hash = stable_hash(
            {
                "arithmetic": [example.to_dict() for example in train_examples],
                "negatives": negative_train,
            }
        )
        cache_path = seed_directory / "hidden_cache.pt"
        if cache_path.exists():
            from neural_firmware.pretrained_training import HiddenCache

            cache = HiddenCache.load(cache_path)
        else:
            cache = collect_hidden_cache(
                base_bundle,
                train_examples,
                negative_train,
                batch_size=config["hidden_cache_batch_size"],
            )
            cache.save(cache_path)
        bridge_config = BridgeTrainConfig(
            seed=seed,
            steps=config["bridge"]["steps"],
            batch_size=config["bridge"]["batch_size"],
            learning_rate=config["bridge"]["learning_rate"],
            strength=config["bridge"]["strength"],
            router_loss_weight=config["bridge"]["router_loss_weight"],
        )
        bridge_path = seed_directory / "bridge.pt"
        train_result_path = seed_directory / "train_result.json"
        if bridge_path.exists() and train_result_path.exists():
            bridge = load_bridge(
                bridge_path,
                hidden_size=base_bundle.model.config.hidden_size,
                strength=config["bridge"]["strength"],
                device=base_bundle.device,
            )
            train_result = json.loads(train_result_path.read_text())
        else:
            bridge, result = train_bridge(base_bundle, cache, bridge_config)
            save_bridge(bridge, result, seed_directory)
            train_result = asdict(result)
        latent_summaries = evaluate_mode(
            bundle=base_bundle,
            eval_sets=eval_sets,
            mode="latent",
            bridge=bridge,
            output_directory=seed_directory / "latent",
        )
        off_summaries = evaluate_mode(
            bundle=base_bundle,
            eval_sets=eval_sets,
            mode="firmware_off",
            bridge=bridge,
            output_directory=seed_directory / "firmware_off",
        )
        preservation_path = seed_directory / "preservation_summary.json"
        if preservation_path.exists():
            preservation_summary = json.loads(preservation_path.read_text())
        else:
            preservation_summary, preservation_rows = evaluate_preservation(
                base_bundle,
                routing_prompts,
                bridge=bridge,
            )
            write_json(
                seed_directory / "preservation_predictions.json",
                preservation_rows,
            )
            write_json(preservation_path, preservation_summary)
        seed_summary = {
            "seed": seed,
            "training_sha256": training_hash,
            "train_result": train_result,
            "latent_summaries": latent_summaries,
            "firmware_off_summaries": off_summaries,
            "preservation_summary": preservation_summary,
        }
        write_json(seed_directory / "seed_summary.json", seed_summary)
        bridge_run_summaries.append(seed_summary)

    del base_bundle
    release_accelerator_memory()

    adapter_run_summaries: list[dict[str, object]] = []
    for seed in config["train_seeds"]:
        seed_directory = args.artifact_directory / f"adapter_seed_{seed}"
        train_examples, _ = train_examples_for_seed(config, seed)
        training_hash = stable_hash(
            [example.to_dict() for example in train_examples]
        )
        adapter_config = AdapterTrainConfig(
            seed=seed,
            steps=config["learned_adapter"]["steps"],
            batch_size=config["learned_adapter"]["batch_size"],
            learning_rate=config["learned_adapter"]["learning_rate"],
            rank=config["learned_adapter"]["rank"],
            alpha=config["learned_adapter"]["alpha"],
            all_layers=config["learned_adapter"]["all_layers"],
        )
        adapter_bundle = load_model_bundle(
            config["model_id"],
            revision=config["model_revision"],
        )
        adapter_path = seed_directory / "adapter.pt"
        train_result_path = seed_directory / "train_result.json"
        if adapter_path.exists() and train_result_path.exists():
            load_learned_adapter(
                adapter_bundle,
                adapter_path,
                adapter_config,
            )
            train_result = json.loads(train_result_path.read_text())
        else:
            result = train_learned_adapter(
                adapter_bundle,
                train_examples,
                adapter_config,
            )
            save_learned_adapter(adapter_bundle, result, seed_directory)
            train_result = asdict(result)
        adapter_summaries = evaluate_mode(
            bundle=adapter_bundle,
            eval_sets=eval_sets,
            mode="learned_adapter",
            output_directory=seed_directory / "learned_adapter",
        )
        preservation_summary = evaluate_adapter_preservation(
            adapter_bundle,
            routing_prompts,
            base_preservation,
            seed_directory,
        )
        seed_summary = {
            "seed": seed,
            "training_sha256": training_hash,
            "train_result": train_result,
            "adapter_summaries": adapter_summaries,
            "preservation_summary": preservation_summary,
        }
        write_json(seed_directory / "seed_summary.json", seed_summary)
        adapter_run_summaries.append(seed_summary)
        del adapter_bundle
        release_accelerator_memory()

    study = {
        "experiment_name": config["experiment_name"],
        "frozen_state": frozen_state,
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "config": config,
        "evaluation_sha256": evaluation_hash,
        "base_summaries": base_summaries,
        "direct_summaries": direct_summaries,
        "bridge_runs": bridge_run_summaries,
        "adapter_runs": adapter_run_summaries,
        "completed_at_unix": time.time(),
    }
    write_json(args.artifact_directory / "study.json", study)
    write_json(args.result_directory / "study.json", study)
    print(json.dumps(study, indent=2))


if __name__ == "__main__":
    main()
