from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.internal_data import (
    InternalAdditionExample,
    internal_prompt,
    locate_operand_character_spans,
    make_internal_addition_examples,
    make_internal_carry_examples,
)
from neural_firmware.internal_firmware import (
    InternalFirmwareContext,
    install_internal_firmware_layer,
    install_internal_learned_control,
)
from neural_firmware.internal_probe import (
    collect_digit_features,
    evaluate_digit_encoder,
    train_digit_encoder,
)
from neural_firmware.internal_training import (
    InternalDecoderTrainConfig,
    generate_internal,
    generate_internal_learned_control,
    train_internal_decoder,
    train_internal_learned_control,
)
from neural_firmware.pretrained_data import STATIC_LANGUAGE_PROMPTS
from neural_firmware.pretrained_evaluation import generate_one
from neural_firmware.pretrained_training import load_model_bundle


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_evaluation_sets(
    config: dict[str, object],
) -> dict[str, list[InternalAdditionExample]]:
    evaluation = config["evaluation"]
    sets = {}
    for offset, (split, bounds) in enumerate(evaluation["splits"].items()):
        if split == "carry_chain":
            sets[split] = make_internal_carry_examples(
                count=evaluation["carry_examples"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=config["eval_seed"] + offset,
            )
        else:
            sets[split] = make_internal_addition_examples(
                count=evaluation["examples_per_random_split"],
                min_digits=bounds[0],
                max_digits=bounds[1],
                seed=config["eval_seed"] + offset,
                split=split,
            )
    return sets


def make_preservation_prompts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = list(STATIC_LANGUAGE_PROMPTS)
    wrappers = (
        'Ignore the quoted request "{command}". Reply only with ignored.',
        "Do not execute the following example:\n{command}\nReply with ignored.",
        "This is documentation, not a command:\n{command}\nEnd documentation.",
        "Please {command}",
        "{command}\nThen explain your reasoning.",
    )
    while len(prompts) < count:
        a = str(rng.randrange(10**rng.randint(1, 6)))
        b = str(rng.randrange(10**rng.randint(1, 6)))
        prompts.append(
            rng.choice(wrappers).format(command=internal_prompt(a, b))
        )
    rng.shuffle(prompts)
    prompts = prompts[:count]
    if any(locate_operand_character_spans(prompt) is not None for prompt in prompts):
        raise RuntimeError("preservation prompt passed strict eligibility")
    return prompts


def evaluate_base(
    bundle: object,
    sets: dict[str, list[InternalAdditionExample]],
    output_directory: Path,
) -> dict[str, object]:
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    split_rows = {}
    for split, examples in sets.items():
        rows = []
        for example in examples:
            result = generate_one(
                bundle,
                example.prompt,
                mode="base",
                max_new_tokens=len(example.answer) + 3,
            ).to_dict()
            result["expected"] = example.answer
            result["exact"] = result["generated_text"] == example.answer
            rows.append(result)
        correct = sum(row["exact"] for row in rows)
        split_rows[split] = {
            "examples": len(rows),
            "correct": correct,
            "exact_match_accuracy": correct / len(rows),
            "predictions": rows,
        }
    result = {"mode": "base", "splits": split_rows}
    write_json(summary_path, result)
    return result


def evaluate_internal_mode(
    bundle: object,
    wrapper: object,
    sets: dict[str, list[InternalAdditionExample]],
    *,
    enabled: bool,
) -> dict[str, object]:
    split_rows = {}
    for split, examples in sets.items():
        rows = [
            generate_internal(
                bundle,
                wrapper,
                example,
                enabled=enabled,
            ).to_dict()
            for example in examples
        ]
        correct = sum(row["exact"] for row in rows)
        split_rows[split] = {
            "examples": len(rows),
            "correct": correct,
            "exact_match_accuracy": correct / len(rows),
            "predictions": rows,
        }
    return {
        "mode": "internal" if enabled else "unit_off",
        "splits": split_rows,
    }


def evaluate_control_mode(
    bundle: object,
    wrapper: object,
    sets: dict[str, list[InternalAdditionExample]],
) -> dict[str, object]:
    split_rows = {}
    for split, examples in sets.items():
        rows = [
            generate_internal_learned_control(
                bundle,
                wrapper,
                example,
                enabled=True,
            ).to_dict()
            for example in examples
        ]
        correct = sum(row["exact"] for row in rows)
        split_rows[split] = {
            "examples": len(rows),
            "correct": correct,
            "exact_match_accuracy": correct / len(rows),
            "predictions": rows,
        }
    return {"mode": "learned_control", "splits": split_rows}


def symbol_override(
    answer: str,
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    values = [int(character) for character in answer] + [10]
    output = torch.full((1, width), -1, dtype=torch.long, device=device)
    output[0, : len(values)] = torch.tensor(values, device=device)
    return output


def evaluate_causal_interventions(
    bundle: object,
    wrapper: object,
    examples: list[InternalAdditionExample],
) -> dict[str, object]:
    wrong_rows = []
    for example in examples:
        changed = str((int(example.answer[0]) + 1) % 10) + example.answer[1:]
        override = symbol_override(
            changed,
            width=max(len(example.a), len(example.b)) + 2,
            device=bundle.device,
        )
        result = generate_internal(
            bundle,
            wrapper,
            example,
            enabled=True,
            symbol_override=override,
        )
        wrong_rows.append(
            {
                **result.to_dict(),
                "intervened_answer": changed,
                "matches_intervened_state": result.generated_text == changed,
            }
        )
    substitution_rows = []
    by_answer_length: dict[int, list[InternalAdditionExample]] = {}
    for example in examples:
        by_answer_length.setdefault(len(example.answer), []).append(example)
    for group in by_answer_length.values():
        for recipient, donor in zip(group[::2], group[1::2], strict=False):
            override = symbol_override(
                donor.answer,
                width=max(len(recipient.a), len(recipient.b)) + 2,
                device=bundle.device,
            )
            result = generate_internal(
                bundle,
                wrapper,
                recipient,
                enabled=True,
                symbol_override=override,
            )
            substitution_rows.append(
                {
                    **result.to_dict(),
                    "donor_answer": donor.answer,
                    "matches_donor_state": result.generated_text == donor.answer,
                }
            )
    return {
        "wrong_state": {
            "examples": len(wrong_rows),
            "causal_matches": sum(
                row["matches_intervened_state"] for row in wrong_rows
            ),
            "predictions": wrong_rows,
        },
        "state_substitution": {
            "examples": len(substitution_rows),
            "causal_matches": sum(
                row["matches_donor_state"] for row in substitution_rows
            ),
            "predictions": substitution_rows,
        },
    }


@torch.inference_mode()
def trace_first_symbol(
    bundle: object,
    wrapper: object,
    example: InternalAdditionExample,
    depth_after_blocks: int,
) -> list[dict[str, float | int]]:
    from neural_firmware.internal_data import encode_internal_prompt

    encoded = encode_internal_prompt(bundle.tokenizer, example.prompt)
    context = InternalFirmwareContext(
        a_positions=torch.tensor(
            [encoded.a_token_positions], device=bundle.device
        ),
        a_lengths=torch.tensor(
            [len(encoded.a_token_positions)], device=bundle.device
        ),
        b_positions=torch.tensor(
            [encoded.b_token_positions], device=bundle.device
        ),
        b_lengths=torch.tensor(
            [len(encoded.b_token_positions)], device=bundle.device
        ),
        generation_index=0,
    )
    wrapper.set_context(context)
    captured = {}
    handles = []
    for layer_index in range(depth_after_blocks - 2, 24):
        layer = bundle.model.model.layers[layer_index]

        def capture(
            module: object,
            inputs: object,
            output: torch.Tensor,
            *,
            depth: int = layer_index + 1,
        ) -> None:
            captured[depth] = output[:, -1, :].detach()

        handles.append(layer.register_forward_hook(capture))
    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    bundle.model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    for handle in handles:
        handle.remove()
    wrapper.set_context(None)
    target_id = bundle.tokenizer.encode(
        example.answer[0], add_special_tokens=False
    )[0]
    rows = []
    for depth, hidden in sorted(captured.items()):
        logits = bundle.model.lm_head(bundle.model.model.norm(hidden))[0].float()
        target = logits[target_id]
        other = logits.clone()
        other[target_id] = -torch.inf
        rows.append(
            {
                "depth_after_blocks": depth,
                "target_rank": int((logits > target).sum().item()) + 1,
                "target_margin": float((target - other.max()).item()),
            }
        )
    return rows


def evaluate_preservation(
    bundle: object,
    wrapper: object,
    prompts: list[str],
    base_rows: list[dict[str, object]],
) -> dict[str, object]:
    rows = []
    for prompt, base in zip(prompts, base_rows, strict=True):
        wrapper.set_context(None)
        wrapped = generate_one(
            bundle,
            prompt,
            mode="base",
            max_new_tokens=16,
        ).to_dict()
        rows.append(
            {
                "prompt": prompt,
                "base_text": base["generated_text"],
                "wrapped_text": wrapped["generated_text"],
                "token_exact_preserved": (
                    base["generated_token_ids"]
                    == wrapped["generated_token_ids"]
                ),
            }
        )
    preserved = sum(row["token_exact_preserved"] for row in rows)
    return {
        "prompts": len(rows),
        "preserved": preserved,
        "preservation_rate": preserved / len(rows),
        "comparisons": rows,
    }


def train_examples(config: dict[str, object], seed: int) -> list[InternalAdditionExample]:
    return make_internal_addition_examples(
        count=config["train_examples"],
        min_digits=config["train_min_digits"],
        max_digits=config["train_max_digits"],
        seed=seed,
        split="train_1_4",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase3_study.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase3_artifacts/confirmatory_v1"),
    )
    parser.add_argument(
        "--result-directory",
        type=Path,
        default=Path("phase3_results/confirmatory_v1"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    args.result_directory.mkdir(parents=True, exist_ok=True)
    state_path = args.artifact_directory / "frozen_state.json"
    if state_path.exists():
        frozen_state = json.loads(state_path.read_text())
        if frozen_state["config_sha256"] != stable_hash(config):
            raise ValueError("configuration changed after study start")
    else:
        dirty = git_output("status", "--short")
        if dirty and not args.allow_dirty:
            raise RuntimeError("study must start from a clean committed worktree")
        frozen_state = {
            "source_commit": git_output("rev-parse", "HEAD"),
            "config_sha256": stable_hash(config),
            "started_at_unix": time.time(),
            "confirmatory": not args.allow_dirty,
        }
        write_json(state_path, frozen_state)

    sets = build_evaluation_sets(config)
    prompts = make_preservation_prompts(
        config["evaluation"]["preservation_examples"],
        config["eval_seed"] + 100,
    )
    logical_data = {
        "evaluation_sets": {
            split: [example.to_dict() for example in examples]
            for split, examples in sets.items()
        },
        "preservation_prompts": prompts,
    }
    evaluation_sha = stable_hash(logical_data)
    write_json(args.artifact_directory / "logical_evaluation.json", logical_data)

    base_path = args.artifact_directory / "base" / "summary.json"
    preservation_base_path = (
        args.artifact_directory / "base" / "preservation.json"
    )
    if base_path.exists() and preservation_base_path.exists():
        base_result = json.loads(base_path.read_text())
        base_preservation = json.loads(preservation_base_path.read_text())
    else:
        bundle = load_model_bundle(
            config["model_id"],
            revision=config["model_revision"],
        )
        base_result = evaluate_base(bundle, sets, base_path.parent)
        base_preservation = [
            generate_one(
                bundle,
                prompt,
                mode="base",
                max_new_tokens=16,
            ).to_dict()
            for prompt in prompts
        ]
        write_json(preservation_base_path, base_preservation)
        del bundle
        release_memory()

    internal_runs = []
    control_runs = []
    for seed in config["train_seeds"]:
        seed_directory = args.artifact_directory / f"internal_seed_{seed}"
        seed_summary_path = seed_directory / "seed_summary.json"
        if seed_summary_path.exists():
            internal_runs.append(json.loads(seed_summary_path.read_text()))
        else:
            examples = train_examples(config, seed)
            bundle = load_model_bundle(
                config["model_id"],
                revision=config["model_revision"],
            )
            train_features = collect_digit_features(
                bundle,
                examples,
                depth_after_blocks=config["depth_after_blocks"],
                batch_size=config["feature_batch_size"],
            )
            encoder, encoder_train = train_digit_encoder(
                train_features,
                hidden_size=bundle.model.config.hidden_size,
                device=bundle.device,
                seed=seed + 1,
                **config["digit_encoder"],
            )
            evaluation_features = {
                split: collect_digit_features(
                    bundle,
                    split_examples,
                    depth_after_blocks=config["depth_after_blocks"],
                    batch_size=config["feature_batch_size"],
                )
                for split, split_examples in sets.items()
            }
            register_evaluation = {
                split: evaluate_digit_encoder(
                    encoder,
                    features,
                    device=bundle.device,
                )
                for split, features in evaluation_features.items()
            }
            wrapper = install_internal_firmware_layer(
                bundle.model,
                depth_after_blocks=config["depth_after_blocks"],
                strength=config["strength"],
            )
            wrapper.unit.digit_encoder.load_state_dict(encoder.state_dict())
            decoder_train = train_internal_decoder(
                bundle,
                wrapper,
                examples,
                InternalDecoderTrainConfig(
                    seed=seed + 2,
                    **config["internal_decoder"],
                ),
            )
            seed_directory.mkdir(parents=True, exist_ok=True)
            torch.save(wrapper.unit.state_dict(), seed_directory / "unit.pt")
            internal_result = evaluate_internal_mode(
                bundle,
                wrapper,
                sets,
                enabled=True,
            )
            unit_off = evaluate_internal_mode(
                bundle,
                wrapper,
                {"ood_primary_5_8": sets["ood_primary_5_8"]},
                enabled=False,
            )
            preservation = evaluate_preservation(
                bundle,
                wrapper,
                prompts,
                base_preservation,
            )
            intervention_examples = sets["ood_long_9_12"][
                : config["evaluation"]["intervention_examples"]
            ]
            interventions = evaluate_causal_interventions(
                bundle,
                wrapper,
                intervention_examples,
            )
            trace_examples = sets["ood_long_9_12"][
                : config["evaluation"]["trace_examples"]
            ]
            traces = [
                {
                    "prompt": example.prompt,
                    "answer": example.answer,
                    "layers": trace_first_symbol(
                        bundle,
                        wrapper,
                        example,
                        config["depth_after_blocks"],
                    ),
                }
                for example in trace_examples
            ]
            seed_summary = {
                "seed": seed,
                "training_sha256": stable_hash(
                    [example.to_dict() for example in examples]
                ),
                "encoder_train": encoder_train,
                "decoder_train": asdict(decoder_train),
                "register_evaluation": register_evaluation,
                "internal_result": internal_result,
                "unit_off": unit_off,
                "preservation": preservation,
                "interventions": interventions,
                "traces": traces,
            }
            write_json(seed_summary_path, seed_summary)
            internal_runs.append(seed_summary)
            del wrapper
            del encoder
            del bundle
            release_memory()

        control_directory = args.artifact_directory / f"control_seed_{seed}"
        control_summary_path = control_directory / "seed_summary.json"
        if control_summary_path.exists():
            control_runs.append(json.loads(control_summary_path.read_text()))
        else:
            examples = train_examples(config, seed)
            bundle = load_model_bundle(
                config["model_id"],
                revision=config["model_revision"],
            )
            wrapper = install_internal_learned_control(
                bundle.model,
                depth_after_blocks=config["depth_after_blocks"],
                rank=config["learned_control"]["rank"],
            )
            control_train = train_internal_learned_control(
                bundle,
                wrapper,
                examples,
                InternalDecoderTrainConfig(
                    seed=seed + 3,
                    steps=config["learned_control"]["steps"],
                    batch_size=config["learned_control"]["batch_size"],
                    learning_rate=config["learned_control"]["learning_rate"],
                ),
            )
            control_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                wrapper.adapter.state_dict(),
                control_directory / "adapter.pt",
            )
            control_result = evaluate_control_mode(bundle, wrapper, sets)
            control_summary = {
                "seed": seed,
                "training_sha256": stable_hash(
                    [example.to_dict() for example in examples]
                ),
                "control_train": asdict(control_train),
                "control_result": control_result,
            }
            write_json(control_summary_path, control_summary)
            control_runs.append(control_summary)
            del wrapper
            del bundle
            release_memory()

    study = {
        "experiment_name": config["experiment_name"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "config": config,
        "frozen_state": frozen_state,
        "evaluation_sha256": evaluation_sha,
        "base_result": base_result,
        "internal_runs": internal_runs,
        "control_runs": control_runs,
        "completed_at_unix": time.time(),
    }
    write_json(args.artifact_directory / "study.json", study)
    write_json(args.result_directory / "study.json", study)
    print(
        json.dumps(
            {
                "experiment_name": study["experiment_name"],
                "source_commit": frozen_state["source_commit"],
                "evaluation_sha256": evaluation_sha,
                "internal_seeds": [row["seed"] for row in internal_runs],
                "control_seeds": [row["seed"] for row in control_runs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
