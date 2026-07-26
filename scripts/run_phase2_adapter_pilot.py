from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from neural_firmware.pretrained_data import (
    make_addition_examples,
    make_carry_examples,
    make_routing_negatives,
)
from neural_firmware.pretrained_evaluation import evaluate_additions, generate_one
from neural_firmware.pretrained_training import (
    AdapterTrainConfig,
    load_model_bundle,
    save_learned_adapter,
    train_learned_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_adapter_pilot.json"),
    )
    args = parser.parse_args()
    config_path = args.config
    config = json.loads(config_path.read_text())
    train_examples = make_addition_examples(
        count=config["train_examples"],
        min_digits=config["train_min_digits"],
        max_digits=config["train_max_digits"],
        seed=config["train_seed"],
        split="train",
    )
    eval_seed = config["eval_seed"]
    eval_sets = {
        "id_random": make_addition_examples(
            count=40,
            min_digits=1,
            max_digits=4,
            seed=eval_seed,
            split="id_random",
        ),
        "ood_primary": make_addition_examples(
            count=40,
            min_digits=5,
            max_digits=8,
            seed=eval_seed + 1,
            split="ood_primary",
        ),
        "ood_long": make_addition_examples(
            count=40,
            min_digits=9,
            max_digits=12,
            seed=eval_seed + 2,
            split="ood_long",
        ),
        "carry_chain": make_carry_examples(
            count=30,
            min_digits=5,
            max_digits=12,
            seed=eval_seed + 3,
        ),
    }
    preservation_prompts = make_routing_negatives(
        count=config.get("preservation_examples", 30),
        seed=eval_seed + 100,
    )

    bundle = load_model_bundle(config["model_id"])
    base_preservation = [
        generate_one(
            bundle,
            prompt,
            mode="base",
            max_new_tokens=16,
        )
        for prompt in preservation_prompts
    ]
    adapter_config = config["adapter"]
    train_result = train_learned_adapter(
        bundle,
        train_examples,
        AdapterTrainConfig(
            seed=config["train_seed"],
            steps=adapter_config["steps"],
            batch_size=adapter_config["batch_size"],
            learning_rate=adapter_config["learning_rate"],
            rank=adapter_config["rank"],
            alpha=adapter_config["alpha"],
            all_layers=adapter_config.get("all_layers", False),
        ),
    )
    artifact_directory = Path("phase2_artifacts") / config["experiment_name"]
    save_learned_adapter(bundle, train_result, artifact_directory)
    summaries = []
    predictions = []
    for split, examples in eval_sets.items():
        summary, results = evaluate_additions(
            bundle,
            examples,
            mode="learned_adapter",
        )
        summaries.append(summary)
        predictions.extend(
            [
                {"split": split, "mode": "learned_adapter", **result.to_dict()}
                for result in results
            ]
        )
    adapter_preservation = [
        generate_one(
            bundle,
            prompt,
            mode="learned_adapter",
            max_new_tokens=16,
        )
        for prompt in preservation_prompts
    ]
    preservation_rows = []
    for prompt, base, adapter in zip(
        preservation_prompts,
        base_preservation,
        adapter_preservation,
        strict=True,
    ):
        preservation_rows.append(
            {
                "prompt": prompt,
                "base_text": base.generated_text,
                "adapter_text": adapter.generated_text,
                "token_exact_preserved": (
                    base.generated_token_ids == adapter.generated_token_ids
                ),
            }
        )
    preserved = sum(row["token_exact_preserved"] for row in preservation_rows)
    preservation_summary = {
        "prompts": len(preservation_rows),
        "token_exact_preserved": preserved,
        "preservation_rate": preserved / len(preservation_rows),
    }
    payload = {
        "pilot": True,
        "config": config,
        "train_result": asdict(train_result),
        "summaries": summaries,
        "preservation_summary": preservation_summary,
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n"
    )
    (artifact_directory / "preservation_predictions.json").write_text(
        json.dumps(preservation_rows, indent=2) + "\n"
    )
    result_path = Path("phase2_results") / f"{config['experiment_name']}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
