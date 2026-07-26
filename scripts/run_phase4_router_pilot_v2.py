from __future__ import annotations

import json
from pathlib import Path

import torch

from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    AUGMENTED_TRAIN_ADDITION_FAMILIES,
    AUGMENTED_TRAIN_NEGATIVE_FAMILIES,
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    HELDOUT_ADDITION_FAMILIES,
    HELDOUT_NEGATIVE_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
    WORD_PROBLEM_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)
from neural_firmware.semantic_training import (
    collect_route_features,
    evaluate_semantic_router,
    train_semantic_router,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTH_AFTER_BLOCKS = 24
ROUTER_HIDDEN_WIDTH = 16


def metrics_at_threshold(
    rows: list[dict[str, object]],
    threshold: float,
) -> dict[str, float | int]:
    positive = [row for row in rows if row["label"] is True]
    negative = [row for row in rows if row["label"] is False]
    true_positive = sum(row["probability"] >= threshold for row in positive)
    false_positive = sum(row["probability"] >= threshold for row in negative)
    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_positive_rate": true_positive / len(positive),
        "false_positive_rate": false_positive / len(negative),
        "accuracy": (
            true_positive + len(negative) - false_positive
        )
        / (len(positive) + len(negative)),
    }


def main() -> None:
    train_positive_families = (
        TRAIN_ADDITION_FAMILIES
        + HELDOUT_ADDITION_FAMILIES
        + WORD_PROBLEM_FAMILIES
        + AUGMENTED_TRAIN_ADDITION_FAMILIES
    )
    train_negative_families = (
        TRAIN_NEGATIVE_FAMILIES
        + HELDOUT_NEGATIVE_FAMILIES
        + AUGMENTED_TRAIN_NEGATIVE_FAMILIES
    )
    train_examples = (
        make_semantic_addition_examples(
            count=2400,
            min_digits=1,
            max_digits=12,
            seed=9301,
            split="router_v2_train_positive",
            families=train_positive_families,
        )
        + make_semantic_routing_negatives(
            count=2400,
            min_digits=1,
            max_digits=12,
            seed=9302,
            split="router_v2_train_negative",
            families=train_negative_families,
        )
    )
    development_examples = (
        make_semantic_addition_examples(
            count=800,
            min_digits=1,
            max_digits=12,
            seed=9303,
            split="router_v2_development_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=800,
            min_digits=1,
            max_digits=12,
            seed=9304,
            split="router_v2_development_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        )
    )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_route_features(
        bundle,
        train_examples,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        batch_size=8,
    )
    development_features = collect_route_features(
        bundle,
        development_examples,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        batch_size=8,
    )
    router, training = train_semantic_router(
        train_features,
        hidden_size=bundle.model.config.hidden_size,
        device=bundle.device,
        seed=9305,
        steps=2000,
        batch_size=512,
        learning_rate=0.003,
        hidden_width=ROUTER_HIDDEN_WIDTH,
    )
    raw_evaluation = evaluate_semantic_router(
        router,
        development_features,
        development_examples,
        device=bundle.device,
        threshold=0.5,
    )
    threshold_sweep = [
        metrics_at_threshold(raw_evaluation["rows"], threshold / 100)
        for threshold in range(50, 100)
    ]
    eligible = [
        row
        for row in threshold_sweep
        if row["false_positive_rate"] <= 0.01
    ]
    selected = max(
        eligible if eligible else threshold_sweep,
        key=lambda row: (row["true_positive_rate"], row["accuracy"]),
    )
    selected_evaluation = evaluate_semantic_router(
        router,
        development_features,
        development_examples,
        device=bundle.device,
        threshold=float(selected["threshold"]),
    )
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "depth_after_blocks": DEPTH_AFTER_BLOCKS,
        "router_hidden_width": ROUTER_HIDDEN_WIDTH,
        "train_positive_families": len(train_positive_families),
        "train_negative_families": len(train_negative_families),
        "train_examples": len(train_examples),
        "development_positive_families": len(DEVELOPMENT_ADDITION_FAMILIES),
        "development_negative_families": len(DEVELOPMENT_NEGATIVE_FAMILIES),
        "development_examples": len(development_examples),
        "training": training,
        "threshold_sweep": threshold_sweep,
        "selected_threshold": selected["threshold"],
        "evaluation_at_selected_threshold": selected_evaluation,
    }
    output = Path("phase4_results/router_pilot_v2.json")
    checkpoint = Path("phase4_artifacts/router_pilot_v2.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(router.state_dict(), checkpoint)
    print(
        json.dumps(
            {
                "selected_threshold": selected["threshold"],
                "accuracy": selected_evaluation["accuracy"],
                "true_positive_rate": selected_evaluation["true_positive_rate"],
                "false_positive_rate": selected_evaluation["false_positive_rate"],
                "trainable_parameters": training["trainable_parameters"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
