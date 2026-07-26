from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    build_phase5_development_examples,
    build_phase5_output_training_examples,
    build_phase5_training_examples,
)
from neural_firmware.phase5_igc import (
    IGCInputMapping,
    install_dual_depth_igc,
    install_igc_firmware,
)
from neural_firmware.phase5_training import (
    IGCInputTrainConfig,
    IGCOutputTrainConfig,
    collect_igc_features,
    evaluate_igc_input_mapping,
    generate_dual_igc,
    generate_igc,
    select_route_threshold,
    train_dual_igc_output_mapping,
    train_igc_input_mapping,
    train_igc_output_mapping,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_training import (
    collect_route_features,
    evaluate_semantic_router,
    train_semantic_router,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_DIGITS = 12


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"] is True]
    negatives = [row for row in rows if row["route_label"] is False]
    return {
        "examples": len(rows),
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positives
        ),
        "route_true_positive": sum(row["route_active"] is True for row in positives),
        "route_false_positive": sum(row["route_active"] is True for row in negatives),
        "registers_exact": sum(row["registers_exact"] is True for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--depth", type=int, default=24)
    args = parser.parse_args()
    if args.native:
        attention_width = 64
        attention_heads = 8
        output_width = 896
        variant = "native"
    else:
        attention_width = 3
        attention_heads = 1
        output_width = 495 if args.dual else 577
        variant = "matched"

    if args.smoke:
        train_examples = build_phase5_training_examples(
            positive_count=32,
            negative_count=32,
        )
        development_examples = build_phase5_development_examples(
            positive_count=8,
            negative_count=8,
        )
        output_examples = build_phase5_output_training_examples(count=64)
        input_steps = 20
        output_steps = 10
    else:
        train_examples = build_phase5_training_examples(
            positive_count=600,
            negative_count=600,
        )
        development_examples = build_phase5_development_examples(
            positive_count=100,
            negative_count=100,
        )
        output_examples = build_phase5_output_training_examples(count=800)
        input_steps = 2_000
        output_steps = 240

    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_igc_features(
        bundle,
        train_examples,
        depth_after_blocks=args.depth,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    development_features = collect_igc_features(
        bundle,
        development_examples,
        depth_after_blocks=args.depth,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    mapping = IGCInputMapping(
        bundle.model.config.hidden_size,
        max_digits=MAX_DIGITS,
        attention_width=attention_width,
        attention_heads=attention_heads,
    )
    input_training = train_igc_input_mapping(
        mapping,
        train_features,
        IGCInputTrainConfig(
            seed=10_601,
            steps=input_steps,
            batch_size=min(32, len(train_examples)),
            learning_rate=0.003,
        ),
        device=bundle.device,
    )
    raw_development = evaluate_igc_input_mapping(
        mapping,
        development_features,
        development_examples,
        device=bundle.device,
        threshold=0.5,
    )
    threshold = select_route_threshold(raw_development["rows"])
    development = evaluate_igc_input_mapping(
        mapping,
        development_features,
        development_examples,
        device=bundle.device,
        threshold=float(threshold["threshold"]),
    )

    router_training = None
    router_development = None
    if args.dual:
        train_route_features = collect_route_features(
            bundle,
            train_examples,
            depth_after_blocks=24,
            batch_size=8,
        )
        development_route_features = collect_route_features(
            bundle,
            development_examples,
            depth_after_blocks=24,
            batch_size=8,
        )
        router, router_training = train_semantic_router(
            train_route_features,
            hidden_size=bundle.model.config.hidden_size,
            device=bundle.device,
            seed=10_603,
            steps=input_steps,
            batch_size=min(512, len(train_examples)),
            learning_rate=0.003,
            hidden_width=16 if args.native else 0,
        )
        raw_router_development = evaluate_semantic_router(
            router,
            development_route_features,
            development_examples,
            device=bundle.device,
            threshold=0.5,
        )
        threshold = select_route_threshold(raw_router_development["rows"])
        router_development = evaluate_semantic_router(
            router,
            development_route_features,
            development_examples,
            device=bundle.device,
            threshold=float(threshold["threshold"]),
        )
        installation = install_dual_depth_igc(
            bundle.model,
            input_depth_after_blocks=args.depth,
            output_depth_after_blocks=24,
            max_digits=MAX_DIGITS,
            attention_width=attention_width,
            attention_heads=attention_heads,
            output_width=output_width,
            router_hidden_width=16 if args.native else 0,
            initial_strength=64.0,
            learn_output_strength=args.native,
        )
        installation.capture.input_mapping.load_state_dict(mapping.state_dict())
        installation.final.router.load_state_dict(router.state_dict())
        for parameter in installation.capture.input_mapping.parameters():
            parameter.requires_grad_(False)
        for parameter in installation.final.router.parameters():
            parameter.requires_grad_(False)
        output_training = train_dual_igc_output_mapping(
            bundle,
            installation,
            output_examples,
            IGCOutputTrainConfig(
                seed=10_602,
                steps=output_steps,
                batch_size=2,
                learning_rate=0.01,
            ),
        )
        learned_parameters = installation.learned_parameter_count
    else:
        wrapper = install_igc_firmware(
            bundle.model,
            depth_after_blocks=args.depth,
            max_digits=MAX_DIGITS,
            attention_width=attention_width,
            attention_heads=attention_heads,
            output_width=output_width,
            initial_strength=64.0,
        )
        wrapper.unit.input_mapping.load_state_dict(mapping.state_dict())
        for parameter in wrapper.unit.input_mapping.parameters():
            parameter.requires_grad_(False)
        output_training = train_igc_output_mapping(
            bundle,
            wrapper,
            output_examples,
            IGCOutputTrainConfig(
                seed=10_602,
                steps=output_steps,
                batch_size=2,
                learning_rate=0.01,
            ),
        )
        learned_parameters = wrapper.unit.learned_parameter_count
    evaluation_examples = development_examples[:8] + development_examples[-8:]
    if args.dual:
        rows = [
            generate_dual_igc(
                bundle,
                installation,
                example,
                route_mode="learned",
                route_threshold=float(threshold["threshold"]),
                max_new_tokens=24 if example.route_label else 20,
            )
            for example in evaluation_examples
        ]
    else:
        rows = [
            generate_igc(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=float(threshold["threshold"]),
                max_new_tokens=24 if example.route_label else 20,
            )
            for example in evaluation_examples
        ]
    result = {
        "variant": variant,
        "dual_depth": args.dual,
        "depth_after_blocks": args.depth,
        "smoke": args.smoke,
        "attention_width": attention_width,
        "output_width": output_width,
        "learned_parameters": learned_parameters,
        "train_examples": len(train_examples),
        "development_examples": len(development_examples),
        "output_examples": len(output_examples),
        "input_training": input_training,
        "selected_threshold": threshold,
        "development_input_mapping": {
            key: value for key, value in development.items() if key != "rows"
        },
        "output_training": output_training,
        "router_training": router_training,
        "router_development": (
            {
                key: value
                for key, value in router_development.items()
                if key != "rows"
            }
            if router_development is not None
            else None
        ),
        "generation": summarize(rows),
    }
    architecture = "dual" if args.dual else "single"
    run_kind = "smoke" if args.smoke else "pilot"
    output = Path(f"phase5_results/igc_{variant}_{architecture}_{run_kind}.json")
    artifact = Path(f"phase5_artifacts/igc_{variant}_{architecture}_{run_kind}.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        installation.state_dict() if args.dual else wrapper.unit.state_dict(),
        artifact,
    )
    print(
        json.dumps(
            {
                "variant": variant,
                "learned_parameters": result["learned_parameters"],
                "selected_threshold": threshold,
                "development_input_mapping": result[
                    "development_input_mapping"
                ],
                "output_training": output_training,
                "generation": {
                    key: value
                    for key, value in result["generation"].items()
                    if key != "rows"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
