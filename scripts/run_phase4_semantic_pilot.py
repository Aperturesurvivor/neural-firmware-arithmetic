from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    HELDOUT_ADDITION_FAMILIES,
    HELDOUT_NEGATIVE_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
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
    collect_route_features,
    evaluate_semantic_router,
    generate_base_semantic,
    generate_semantic_control,
    generate_semantic_internal,
    train_semantic_control,
    train_semantic_decoder,
    train_semantic_router,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    positive = [row for row in rows if row["expected"] is not None]
    return {
        "examples": len(rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positive
        ),
        "mathematical_accuracy": (
            sum(row["mathematical_correct"] is True for row in positive)
            / len(positive)
            if positive
            else None
        ),
        "exact_format_correct": sum(
            row["exact_format_correct"] is True for row in positive
        ),
        "exact_format_accuracy": (
            sum(row["exact_format_correct"] is True for row in positive)
            / len(positive)
            if positive
            else None
        ),
        "route_activations": sum(row["route_active"] is True for row in rows),
        "rows": rows,
    }


def build_sets(count: int) -> dict[str, list[object]]:
    return {
        "seen_wording_id": make_semantic_addition_examples(
            count=count,
            min_digits=1,
            max_digits=4,
            seed=9101,
            split="seen_wording_id",
            families=TRAIN_ADDITION_FAMILIES,
        ),
        "heldout_wording_id": make_semantic_addition_examples(
            count=count,
            min_digits=1,
            max_digits=4,
            seed=9102,
            split="heldout_wording_id",
            families=HELDOUT_ADDITION_FAMILIES,
        ),
        "heldout_wording_ood": make_semantic_addition_examples(
            count=count,
            min_digits=5,
            max_digits=8,
            seed=9103,
            split="heldout_wording_ood",
            families=HELDOUT_ADDITION_FAMILIES,
        ),
        "heldout_wording_long": make_semantic_addition_examples(
            count=count,
            min_digits=9,
            max_digits=12,
            seed=9104,
            split="heldout_wording_long",
            families=HELDOUT_ADDITION_FAMILIES,
        ),
        "word_problems_ood": make_semantic_addition_examples(
            count=count,
            min_digits=5,
            max_digits=8,
            seed=9105,
            split="word_problems_ood",
            families=WORD_PROBLEM_FAMILIES,
        ),
    }


def main() -> None:
    result_path = Path("phase4_results/semantic_pilot_v1.json")
    artifact_directory = Path("phase4_artifacts/semantic_pilot_v1")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    train_positive = make_semantic_addition_examples(
        count=600,
        min_digits=1,
        max_digits=4,
        seed=9001,
        split="router_train_positive",
        families=TRAIN_ADDITION_FAMILIES,
    )
    train_negative = make_semantic_routing_negatives(
        count=600,
        min_digits=1,
        max_digits=4,
        seed=9002,
        split="router_train_negative",
        families=TRAIN_NEGATIVE_FAMILIES,
    )
    route_train = train_positive + train_negative
    route_evaluation = (
        make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=8,
            seed=9003,
            split="router_eval_heldout_positive",
            families=HELDOUT_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=8,
            seed=9004,
            split="router_eval_word_positive",
            families=WORD_PROBLEM_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=200,
            min_digits=1,
            max_digits=8,
            seed=9005,
            split="router_eval_heldout_negative",
            families=HELDOUT_NEGATIVE_FAMILIES,
        )
    )
    evaluation_sets = build_sets(12)
    preservation_examples = make_semantic_routing_negatives(
        count=24,
        min_digits=1,
        max_digits=8,
        seed=9106,
        split="heldout_route_negative",
        families=HELDOUT_NEGATIVE_FAMILIES,
    )

    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_route_features(
        bundle,
        route_train,
        depth_after_blocks=6,
        batch_size=8,
    )
    evaluation_features = collect_route_features(
        bundle,
        route_evaluation,
        depth_after_blocks=6,
        batch_size=8,
    )
    router, router_train = train_semantic_router(
        train_features,
        hidden_size=bundle.model.config.hidden_size,
        device=bundle.device,
        seed=9006,
        steps=500,
        batch_size=256,
        learning_rate=0.01,
        hidden_width=0,
    )
    router_evaluation = evaluate_semantic_router(
        router,
        evaluation_features,
        route_evaluation,
        device=bundle.device,
        threshold=0.5,
    )
    wrapper = install_semantic_internal_firmware(
        bundle.model,
        depth_after_blocks=6,
        strength=64.0,
        router_hidden_width=0,
    )
    wrapper.unit.router.load_state_dict(router.state_dict())
    decoder_train = train_semantic_decoder(
        bundle,
        wrapper,
        train_positive,
        SemanticTrainConfig(
            seed=9007,
            steps=120,
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
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in evaluation_sets.items()
    }
    preservation_rows = []
    for example in preservation_examples:
        base = generate_base_semantic(bundle, example, max_new_tokens=16)
        internal = generate_semantic_internal(
            bundle,
            wrapper,
            example,
            route_mode="learned",
            max_new_tokens=16,
        )
        preservation_rows.append(
            {
                "prompt": example.prompt,
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
    internal_router_state = router.state_dict()
    del wrapper
    del router
    del bundle
    release_memory()

    control_bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    control_wrapper = install_semantic_learned_control(
        control_bundle.model,
        depth_after_blocks=6,
        rank=5,
        router_hidden_width=0,
    )
    control_wrapper.router.load_state_dict(internal_router_state)
    control_train = train_semantic_control(
        control_bundle,
        control_wrapper,
        train_positive,
        SemanticTrainConfig(
            seed=9008,
            steps=500,
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
        "train_positive_examples": len(train_positive),
        "train_negative_examples": len(train_negative),
        "router_train": router_train,
        "router_evaluation": router_evaluation,
        "decoder_train": asdict(decoder_train),
        "control_train": asdict(control_train),
        "base_results": base_results,
        "internal_results": internal_results,
        "oracle_results": oracle_results,
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
    print(
        json.dumps(
            {
                "router_accuracy": router_evaluation["accuracy"],
                "router_true_positive_rate": router_evaluation[
                    "true_positive_rate"
                ],
                "router_false_positive_rate": router_evaluation[
                    "false_positive_rate"
                ],
                "base": {
                    split: row["mathematical_accuracy"]
                    for split, row in base_results.items()
                },
                "internal": {
                    split: row["mathematical_accuracy"]
                    for split, row in internal_results.items()
                },
                "oracle": {
                    split: row["mathematical_accuracy"]
                    for split, row in oracle_results.items()
                },
                "control": {
                    split: row["mathematical_accuracy"]
                    for split, row in control_results.items()
                },
                "preservation": result["preservation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
