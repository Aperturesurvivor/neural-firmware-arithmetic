from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    build_phase5_development_examples,
    build_phase5_output_training_examples,
    build_phase5_training_examples,
)
from neural_firmware.phase5_igc import IGCInputMapping, install_dual_depth_igc
from neural_firmware.phase5_training import (
    IGCFeatureSet,
    IGCInputTrainConfig,
    IGCOutputTrainConfig,
    collect_igc_features,
    evaluate_igc_input_mapping,
    generate_dual_igc,
    select_route_threshold,
    train_dual_igc_output_mapping,
    train_igc_input_mapping,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_firmware import (
    install_semantic_internal_firmware,
    install_semantic_learned_control,
)
from neural_firmware.semantic_training import (
    SemanticTrainConfig,
    collect_route_features,
    evaluate_semantic_router,
    generate_semantic_control,
    generate_semantic_internal,
    train_semantic_control,
    train_semantic_decoder,
    train_semantic_router,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEEDS = (10_701, 10_702, 10_703)
MAX_DIGITS = 12
INPUT_DEPTH = 1
OUTPUT_DEPTH = 24
ARTIFACT_DIRECTORY = Path("phase5_artifacts/confirmatory_v1")
TRAINING_DIRECTORY = Path("phase5_results/training_v1")
CACHE_DIRECTORY = Path("phase5_artifacts/cache")

IGC_VARIANTS = {
    "igc_matched": {
        "attention_width": 3,
        "attention_heads": 1,
        "output_width": 495,
        "router_hidden_width": 0,
        "learn_output_strength": False,
        "expected_parameters": 24_225,
    },
    "igc_native": {
        "attention_width": 64,
        "attention_heads": 8,
        "output_width": 896,
        "router_hidden_width": 16,
        "learn_output_strength": True,
        "expected_parameters": 597_819,
    },
}


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_feature_set(features: IGCFeatureSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden": features.hidden,
            "attention_mask": features.attention_mask,
            "anchor_positions": features.anchor_positions,
            "a_targets": features.a_targets,
            "b_targets": features.b_targets,
            "route_targets": features.route_targets,
        },
        path,
    )


def load_feature_set(path: Path) -> IGCFeatureSet:
    return IGCFeatureSet(**torch.load(path, map_location="cpu", weights_only=True))


def compact_generation(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"] is True]
    negatives = [row for row in rows if row["route_label"] is False]
    return {
        "examples": len(rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positives
        ),
        "positive_examples": len(positives),
        "route_true_positive": sum(row["route_active"] is True for row in positives),
        "route_false_positive": sum(row["route_active"] is True for row in negatives),
        "negative_examples": len(negatives),
        "registers_exact": sum(row["registers_exact"] is True for row in rows),
    }


def prepare_features(
    train_examples: list[object],
    development_examples: list[object],
) -> tuple[IGCFeatureSet, IGCFeatureSet, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    igc_train_path = CACHE_DIRECTORY / "igc_train.pt"
    igc_development_path = CACHE_DIRECTORY / "igc_development.pt"
    route_train_path = CACHE_DIRECTORY / "route_train.pt"
    route_development_path = CACHE_DIRECTORY / "route_development.pt"
    if all(
        path.exists()
        for path in (
            igc_train_path,
            igc_development_path,
            route_train_path,
            route_development_path,
        )
    ):
        return (
            load_feature_set(igc_train_path),
            load_feature_set(igc_development_path),
            torch.load(route_train_path, map_location="cpu", weights_only=True),
            torch.load(
                route_development_path,
                map_location="cpu",
                weights_only=True,
            ),
        )

    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    igc_train = collect_igc_features(
        bundle,
        train_examples,
        depth_after_blocks=INPUT_DEPTH,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    igc_development = collect_igc_features(
        bundle,
        development_examples,
        depth_after_blocks=INPUT_DEPTH,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    route_train = collect_route_features(
        bundle,
        train_examples,
        depth_after_blocks=OUTPUT_DEPTH,
        batch_size=8,
    )
    route_development = collect_route_features(
        bundle,
        development_examples,
        depth_after_blocks=OUTPUT_DEPTH,
        batch_size=8,
    )
    save_feature_set(igc_train, igc_train_path)
    save_feature_set(igc_development, igc_development_path)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.save(route_train, route_train_path)
    torch.save(route_development, route_development_path)
    del bundle
    release_memory()
    return igc_train, igc_development, route_train, route_development


def train_late_router(
    *,
    route_train: dict[str, torch.Tensor],
    route_development: dict[str, torch.Tensor],
    development_examples: list[object],
    hidden_width: int,
    seed: int,
    device: torch.device,
) -> tuple[object, dict[str, object], dict[str, float | int], dict[str, object]]:
    router, training = train_semantic_router(
        route_train,
        hidden_size=896,
        device=device,
        seed=seed,
        steps=2_000,
        batch_size=512,
        learning_rate=0.003,
        hidden_width=hidden_width,
    )
    raw = evaluate_semantic_router(
        router,
        route_development,
        development_examples,
        device=device,
        threshold=0.5,
    )
    selected = select_route_threshold(raw["rows"])
    evaluation = evaluate_semantic_router(
        router,
        route_development,
        development_examples,
        device=device,
        threshold=float(selected["threshold"]),
    )
    compact = {key: value for key, value in evaluation.items() if key != "rows"}
    return router, training, selected, compact


def train_igc_variants(
    *,
    igc_train: IGCFeatureSet,
    igc_development: IGCFeatureSet,
    route_train: dict[str, torch.Tensor],
    route_development: dict[str, torch.Tensor],
    train_examples: list[object],
    development_examples: list[object],
    output_examples: list[object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    diagnostic_examples = development_examples[:10] + development_examples[-10:]
    for variant, architecture in IGC_VARIANTS.items():
        for seed in SEEDS:
            record_path = TRAINING_DIRECTORY / f"{variant}_seed_{seed}.json"
            checkpoint_path = ARTIFACT_DIRECTORY / f"{variant}_seed_{seed}.pt"
            if record_path.exists() and checkpoint_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            started = time.perf_counter()
            bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
            mapping = IGCInputMapping(
                896,
                max_digits=MAX_DIGITS,
                attention_width=int(architecture["attention_width"]),
                attention_heads=int(architecture["attention_heads"]),
            )
            input_training = train_igc_input_mapping(
                mapping,
                igc_train,
                IGCInputTrainConfig(
                    seed=seed,
                    steps=4_000,
                    batch_size=64,
                    learning_rate=0.003,
                ),
                device=bundle.device,
            )
            input_development = evaluate_igc_input_mapping(
                mapping,
                igc_development,
                development_examples,
                device=bundle.device,
                threshold=0.5,
            )
            router, router_training, selected, router_development = train_late_router(
                route_train=route_train,
                route_development=route_development,
                development_examples=development_examples,
                hidden_width=int(architecture["router_hidden_width"]),
                seed=seed + 100,
                device=bundle.device,
            )
            installation = install_dual_depth_igc(
                bundle.model,
                input_depth_after_blocks=INPUT_DEPTH,
                output_depth_after_blocks=OUTPUT_DEPTH,
                max_digits=MAX_DIGITS,
                attention_width=int(architecture["attention_width"]),
                attention_heads=int(architecture["attention_heads"]),
                output_width=int(architecture["output_width"]),
                router_hidden_width=int(architecture["router_hidden_width"]),
                initial_strength=64.0,
                learn_output_strength=bool(architecture["learn_output_strength"]),
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
                    seed=seed + 200,
                    steps=240,
                    batch_size=2,
                    learning_rate=0.01,
                ),
            )
            if installation.learned_parameter_count != int(
                architecture["expected_parameters"]
            ):
                raise RuntimeError("IGC learned-parameter budget changed")
            diagnostic_rows = [
                generate_dual_igc(
                    bundle,
                    installation,
                    example,
                    route_mode="learned",
                    route_threshold=float(selected["threshold"]),
                    max_new_tokens=24 if example.route_label else 20,
                )
                for example in diagnostic_examples
            ]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(installation.state_dict(), checkpoint_path)
            record = {
                "condition": variant,
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "architecture": architecture,
                "learned_parameters": installation.learned_parameter_count,
                "input_training": input_training,
                "input_development": {
                    key: value
                    for key, value in input_development.items()
                    if key != "rows"
                },
                "router_training": router_training,
                "selected_route_threshold": selected,
                "router_development": router_development,
                "output_training": output_training,
                "development_generation": compact_generation(diagnostic_rows),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "wall_time_seconds": time.perf_counter() - started,
            }
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            del installation
            del bundle
            release_memory()
    return records


def train_typed_and_adapter(
    *,
    route_train: dict[str, torch.Tensor],
    route_development: dict[str, torch.Tensor],
    development_examples: list[object],
    output_examples: list[object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    diagnostic_positives = development_examples[:10]
    for seed in SEEDS:
        typed_record_path = TRAINING_DIRECTORY / f"typed_firmware_seed_{seed}.json"
        adapter_record_path = TRAINING_DIRECTORY / f"adapter_seed_{seed}.json"
        typed_checkpoint = ARTIFACT_DIRECTORY / f"typed_firmware_seed_{seed}.pt"
        adapter_checkpoint = ARTIFACT_DIRECTORY / f"adapter_seed_{seed}.pt"
        if all(
            path.exists()
            for path in (
                typed_record_path,
                adapter_record_path,
                typed_checkpoint,
                adapter_checkpoint,
            )
        ):
            records.extend(
                [
                    json.loads(typed_record_path.read_text()),
                    json.loads(adapter_record_path.read_text()),
                ]
            )
            continue

        router_bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
        router, router_training, selected, router_development = train_late_router(
            route_train=route_train,
            route_development=route_development,
            development_examples=development_examples,
            hidden_width=16,
            seed=seed + 100,
            device=router_bundle.device,
        )
        router_state = router.state_dict()
        del router_bundle
        release_memory()

        if not (typed_record_path.exists() and typed_checkpoint.exists()):
            started = time.perf_counter()
            bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
            wrapper = install_semantic_internal_firmware(
                bundle.model,
                depth_after_blocks=OUTPUT_DEPTH,
                strength=64.0,
                router_hidden_width=16,
            )
            wrapper.unit.router.load_state_dict(router_state)
            decoder_training = train_semantic_decoder(
                bundle,
                wrapper,
                output_examples,
                SemanticTrainConfig(
                    seed=seed + 200,
                    steps=240,
                    batch_size=2,
                    learning_rate=0.01,
                ),
            )
            diagnostic_rows = [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=float(selected["threshold"]),
                    max_new_tokens=24,
                )
                for example in diagnostic_positives
            ]
            typed_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(wrapper.unit.state_dict(), typed_checkpoint)
            record = {
                "condition": "typed_firmware",
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "learned_parameters": wrapper.unit.interface_parameter_count,
                "router_training": router_training,
                "selected_route_threshold": selected,
                "router_development": router_development,
                "decoder_training": asdict(decoder_training),
                "development_generation": {
                    "examples": len(diagnostic_rows),
                    "mathematical_correct": sum(
                        row["mathematical_correct"] is True
                        for row in diagnostic_rows
                    ),
                    "route_activations": sum(
                        row["route_active"] is True for row in diagnostic_rows
                    ),
                },
                "checkpoint": str(typed_checkpoint),
                "checkpoint_sha256": sha256(typed_checkpoint),
                "wall_time_seconds": time.perf_counter() - started,
            }
            typed_record_path.parent.mkdir(parents=True, exist_ok=True)
            typed_record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            del wrapper
            del bundle
            release_memory()

        if not (adapter_record_path.exists() and adapter_checkpoint.exists()):
            started = time.perf_counter()
            bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
            wrapper = install_semantic_learned_control(
                bundle.model,
                depth_after_blocks=OUTPUT_DEPTH,
                rank=5,
                router_hidden_width=16,
            )
            wrapper.router.load_state_dict(router_state)
            adapter_training = train_semantic_control(
                bundle,
                wrapper,
                output_examples,
                SemanticTrainConfig(
                    seed=seed + 300,
                    steps=600,
                    batch_size=2,
                    learning_rate=0.001,
                ),
            )
            diagnostic_rows = [
                generate_semantic_control(
                    bundle,
                    wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=float(selected["threshold"]),
                    max_new_tokens=24,
                )
                for example in diagnostic_positives
            ]
            adapter_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "router": wrapper.router.state_dict(),
                    "adapter": wrapper.adapter.state_dict(),
                },
                adapter_checkpoint,
            )
            record = {
                "condition": "adapter",
                "seed": seed,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "learned_parameters": wrapper.interface_parameter_count,
                "router_training": router_training,
                "selected_route_threshold": selected,
                "router_development": router_development,
                "adapter_training": asdict(adapter_training),
                "development_generation": {
                    "examples": len(diagnostic_rows),
                    "mathematical_correct": sum(
                        row["mathematical_correct"] is True
                        for row in diagnostic_rows
                    ),
                    "route_activations": sum(
                        row["route_active"] is True for row in diagnostic_rows
                    ),
                },
                "checkpoint": str(adapter_checkpoint),
                "checkpoint_sha256": sha256(adapter_checkpoint),
                "wall_time_seconds": time.perf_counter() - started,
            }
            adapter_record_path.parent.mkdir(parents=True, exist_ok=True)
            adapter_record_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            del wrapper
            del bundle
            release_memory()
    return records


def main() -> None:
    started = time.perf_counter()
    train_examples = build_phase5_training_examples()
    development_examples = build_phase5_development_examples()
    output_examples = build_phase5_output_training_examples()
    (
        igc_train,
        igc_development,
        route_train,
        route_development,
    ) = prepare_features(train_examples, development_examples)
    records = train_igc_variants(
        igc_train=igc_train,
        igc_development=igc_development,
        route_train=route_train,
        route_development=route_development,
        train_examples=train_examples,
        development_examples=development_examples,
        output_examples=output_examples,
    )
    records.extend(
        train_typed_and_adapter(
            route_train=route_train,
            route_development=route_development,
            development_examples=development_examples,
            output_examples=output_examples,
        )
    )
    records.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
    manifest = {
        "status": "training_complete_development_only",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seeds": list(SEEDS),
        "training_examples": len(train_examples),
        "development_examples": len(development_examples),
        "output_training_examples": len(output_examples),
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output = TRAINING_DIRECTORY / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": [
                    {
                        "condition": row["condition"],
                        "seed": row["seed"],
                        "learned_parameters": row["learned_parameters"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                    for row in records
                ],
                "wall_time_seconds": manifest["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
