from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    AUGMENTED_TRAIN_ADDITION_FAMILIES,
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    HELDOUT_ADDITION_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    WORD_PROBLEM_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)
from neural_firmware.semantic_firmware import (
    install_semantic_internal_firmware,
    install_semantic_learned_control,
)
from neural_firmware.semantic_training import (
    SemanticTrainConfig,
    generate_base_semantic,
    generate_semantic_control,
    generate_semantic_internal,
    train_semantic_control,
    train_semantic_decoder,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTH_AFTER_BLOCKS = 24
ROUTER_HIDDEN_WIDTH = 16
ROUTE_THRESHOLD = 0.76


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in rows
        ),
        "mathematical_accuracy": (
            sum(row["mathematical_correct"] is True for row in rows) / len(rows)
        ),
        "exact_format_correct": sum(
            row["exact_format_correct"] is True for row in rows
        ),
        "exact_format_accuracy": (
            sum(row["exact_format_correct"] is True for row in rows) / len(rows)
        ),
        "route_activations": sum(row["route_active"] is True for row in rows),
        "rows": rows,
    }


def build_evaluation_sets(count: int) -> dict[str, list[object]]:
    simple_families = DEVELOPMENT_ADDITION_FAMILIES[:6]
    word_families = DEVELOPMENT_ADDITION_FAMILIES[6:]
    return {
        "development_id": make_semantic_addition_examples(
            count=count,
            min_digits=1,
            max_digits=4,
            seed=9401,
            split="development_id",
            families=simple_families,
        ),
        "development_ood": make_semantic_addition_examples(
            count=count,
            min_digits=5,
            max_digits=8,
            seed=9402,
            split="development_ood",
            families=simple_families,
        ),
        "development_long": make_semantic_addition_examples(
            count=count,
            min_digits=9,
            max_digits=12,
            seed=9403,
            split="development_long",
            families=simple_families,
        ),
        "development_word": make_semantic_addition_examples(
            count=count,
            min_digits=5,
            max_digits=8,
            seed=9404,
            split="development_word",
            families=word_families,
        ),
    }


def main() -> None:
    result_path = Path("phase4_results/semantic_pilot_v2.json")
    artifact_directory = Path("phase4_artifacts/semantic_pilot_v2")
    router_checkpoint = Path("phase4_artifacts/router_pilot_v2.pt")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    train_families = (
        TRAIN_ADDITION_FAMILIES
        + HELDOUT_ADDITION_FAMILIES
        + WORD_PROBLEM_FAMILIES
        + AUGMENTED_TRAIN_ADDITION_FAMILIES
    )
    train_positive = make_semantic_addition_examples(
        count=1600,
        min_digits=1,
        max_digits=8,
        seed=9400,
        split="semantic_v2_train_positive",
        families=train_families,
    )
    evaluation_sets = build_evaluation_sets(20)
    preservation_examples = make_semantic_routing_negatives(
        count=40,
        min_digits=1,
        max_digits=12,
        seed=9405,
        split="development_route_negative",
        families=DEVELOPMENT_NEGATIVE_FAMILIES,
    )
    router_state = torch.load(router_checkpoint, map_location="cpu", weights_only=True)

    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    wrapper = install_semantic_internal_firmware(
        bundle.model,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        strength=64.0,
        router_hidden_width=ROUTER_HIDDEN_WIDTH,
    )
    wrapper.unit.router.load_state_dict(router_state)
    decoder_train = train_semantic_decoder(
        bundle,
        wrapper,
        train_positive,
        SemanticTrainConfig(
            seed=9406,
            steps=180,
            batch_size=2,
            learning_rate=0.01,
        ),
    )
    torch.save(wrapper.unit.state_dict(), artifact_directory / "semantic_unit.pt")
    base_results = {
        split: summarize(
            [
                generate_base_semantic(bundle, example, max_new_tokens=24)
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    internal_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    oracle_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="force_on",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    off_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="force_off",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    preservation_rows = []
    for example in preservation_examples:
        base = generate_base_semantic(bundle, example, max_new_tokens=20)
        internal = generate_semantic_internal(
            bundle,
            wrapper,
            example,
            route_mode="learned",
            route_threshold=ROUTE_THRESHOLD,
            max_new_tokens=20,
        )
        preservation_rows.append(
            {
                "prompt": example.prompt,
                "family": example.family,
                "base_text": base["generated_text"],
                "internal_text": internal["generated_text"],
                "base_token_ids": base["generated_token_ids"],
                "internal_token_ids": internal["generated_token_ids"],
                "token_exact_preserved": (
                    base["generated_token_ids"] == internal["generated_token_ids"]
                ),
                "route_probability": internal["route_probability"],
                "route_active": internal["route_active"],
            }
        )
    del wrapper
    del bundle
    release_memory()

    control_bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    control_wrapper = install_semantic_learned_control(
        control_bundle.model,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        rank=5,
        router_hidden_width=ROUTER_HIDDEN_WIDTH,
    )
    control_wrapper.router.load_state_dict(router_state)
    control_train = train_semantic_control(
        control_bundle,
        control_wrapper,
        train_positive,
        SemanticTrainConfig(
            seed=9407,
            steps=600,
            batch_size=2,
            learning_rate=0.001,
        ),
    )
    torch.save(
        {
            "router": control_wrapper.router.state_dict(),
            "adapter": control_wrapper.adapter.state_dict(),
        },
        artifact_directory / "semantic_control.pt",
    )
    control_results = {
        split: summarize(
            [
                generate_semantic_control(
                    control_bundle,
                    control_wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "depth_after_blocks": DEPTH_AFTER_BLOCKS,
        "router_hidden_width": ROUTER_HIDDEN_WIDTH,
        "route_threshold": ROUTE_THRESHOLD,
        "train_positive_examples": len(train_positive),
        "internal_interface_parameters": 24_225,
        "control_interface_parameters": 24_225,
        "decoder_train": asdict(decoder_train),
        "control_train": asdict(control_train),
        "base_results": base_results,
        "internal_results": internal_results,
        "oracle_results": oracle_results,
        "off_results": off_results,
        "control_results": control_results,
        "preservation": {
            "examples": len(preservation_rows),
            "preserved": sum(
                row["token_exact_preserved"] for row in preservation_rows
            ),
            "false_activations": sum(
                row["route_active"] for row in preservation_rows
            ),
            "rows": preservation_rows,
        },
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    concise = {
        condition: {
            split: row["mathematical_accuracy"]
            for split, row in condition_rows.items()
        }
        for condition, condition_rows in (
            ("base", base_results),
            ("control", control_results),
            ("internal", internal_results),
            ("oracle", oracle_results),
            ("off", off_results),
        )
    }
    concise["preservation"] = {
        key: result["preservation"][key]
        for key in ("examples", "preserved", "false_activations")
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
