from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    build_phase5_development_examples,
    build_phase5_training_examples,
)
from neural_firmware.phase5_igc import install_dual_depth_igc
from neural_firmware.phase5_training import (
    collect_pre_norm_route_features,
    generate_dual_igc,
    select_route_threshold,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_firmware import (
    install_semantic_internal_firmware,
    install_semantic_learned_control,
)
from neural_firmware.semantic_training import (
    evaluate_semantic_router,
    generate_semantic_control,
    generate_semantic_internal,
    train_semantic_router,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEEDS = (10_701, 10_702, 10_703)
V1_RESULTS = Path("phase5_results/training_v1")
V2_RESULTS = Path("phase5_results/training_v2")
V2_ARTIFACTS = Path("phase5_artifacts/confirmatory_v2")
CACHE_DIRECTORY = Path("phase5_artifacts/cache")


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_or_collect_features() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_path = CACHE_DIRECTORY / "route_pre_norm_train.pt"
    development_path = CACHE_DIRECTORY / "route_pre_norm_development.pt"
    if train_path.exists() and development_path.exists():
        return (
            torch.load(train_path, map_location="cpu", weights_only=True),
            torch.load(development_path, map_location="cpu", weights_only=True),
        )
    train_examples = build_phase5_training_examples()
    development_examples = build_phase5_development_examples()
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train = collect_pre_norm_route_features(
        bundle,
        train_examples,
        depth_after_blocks=24,
        batch_size=8,
    )
    development = collect_pre_norm_route_features(
        bundle,
        development_examples,
        depth_after_blocks=24,
        batch_size=8,
    )
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.save(train, train_path)
    torch.save(development, development_path)
    del bundle
    release_memory()
    return train, development


def train_router_cache(
    train_features: dict[str, torch.Tensor],
    development_features: dict[str, torch.Tensor],
) -> dict[tuple[int, int], dict[str, object]]:
    examples = build_phase5_development_examples()
    cache: dict[tuple[int, int], dict[str, object]] = {}
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    for seed in SEEDS:
        for hidden_width in (0, 16):
            router, training = train_semantic_router(
                train_features,
                hidden_size=896,
                device=bundle.device,
                seed=seed + 100,
                steps=2_000,
                batch_size=512,
                learning_rate=0.003,
                hidden_width=hidden_width,
            )
            raw = evaluate_semantic_router(
                router,
                development_features,
                examples,
                device=bundle.device,
                threshold=0.5,
            )
            selected = select_route_threshold(raw["rows"])
            evaluation = evaluate_semantic_router(
                router,
                development_features,
                examples,
                device=bundle.device,
                threshold=float(selected["threshold"]),
            )
            cache[(seed, hidden_width)] = {
                "state": {
                    key: value.detach().cpu()
                    for key, value in router.state_dict().items()
                },
                "training": training,
                "threshold": selected,
                "development": {
                    key: value for key, value in evaluation.items() if key != "rows"
                },
            }
    del bundle
    release_memory()
    return cache


def diagnostic_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"] is True]
    negatives = [row for row in rows if row["route_label"] is False]
    return {
        "examples": len(rows),
        "positive_examples": len(positives),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positives
        ),
        "route_true_positive": sum(row["route_active"] is True for row in positives),
        "negative_examples": len(negatives),
        "route_false_positive": sum(row["route_active"] is True for row in negatives),
        "registers_exact": sum(
            row.get("registers_exact") is True for row in rows
        ),
    }


def recalibrate_condition(
    condition: str,
    seed: int,
    router_record: dict[str, object],
) -> dict[str, object]:
    source_record_path = V1_RESULTS / f"{condition}_seed_{seed}.json"
    source_record = json.loads(source_record_path.read_text())
    source_checkpoint = Path(source_record["checkpoint"])
    target_checkpoint = V2_ARTIFACTS / f"{condition}_seed_{seed}.pt"
    examples = build_phase5_development_examples()
    diagnostic_examples = examples[:10] + examples[-10:]
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    threshold = float(router_record["threshold"]["threshold"])

    if condition == "typed_firmware":
        wrapper = install_semantic_internal_firmware(
            bundle.model,
            depth_after_blocks=24,
            strength=64.0,
            router_hidden_width=16,
        )
        wrapper.unit.load_state_dict(
            torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        )
        wrapper.unit.router.load_state_dict(router_record["state"])
        state = wrapper.unit.state_dict()
        rows = [
            generate_semantic_internal(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=24 if example.route_label else 20,
            )
            for example in diagnostic_examples
        ]
    elif condition == "adapter":
        wrapper = install_semantic_learned_control(
            bundle.model,
            depth_after_blocks=24,
            rank=5,
            router_hidden_width=16,
        )
        old_state = torch.load(
            source_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        wrapper.adapter.load_state_dict(old_state["adapter"])
        wrapper.router.load_state_dict(router_record["state"])
        state = {
            "router": wrapper.router.state_dict(),
            "adapter": wrapper.adapter.state_dict(),
        }
        rows = [
            generate_semantic_control(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=24 if example.route_label else 20,
            )
            for example in diagnostic_examples
        ]
    elif condition in {"igc_matched", "igc_native"}:
        architecture = source_record["architecture"]
        installation = install_dual_depth_igc(
            bundle.model,
            input_depth_after_blocks=1,
            output_depth_after_blocks=24,
            max_digits=12,
            attention_width=int(architecture["attention_width"]),
            attention_heads=int(architecture["attention_heads"]),
            output_width=int(architecture["output_width"]),
            router_hidden_width=int(architecture["router_hidden_width"]),
            initial_strength=64.0,
            learn_output_strength=bool(architecture["learn_output_strength"]),
        )
        installation.load_state_dict(
            torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        )
        installation.final.router.load_state_dict(router_record["state"])
        state = installation.state_dict()
        rows = [
            generate_dual_igc(
                bundle,
                installation,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=24 if example.route_label else 20,
            )
            for example in diagnostic_examples
        ]
    else:
        raise ValueError(f"unknown condition: {condition}")

    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target_checkpoint)
    record = dict(source_record)
    record.update(
        {
            "router_feature_point": "block_24_output_before_final_rmsnorm",
            "supersedes_checkpoint": str(source_checkpoint),
            "supersedes_checkpoint_sha256": source_record["checkpoint_sha256"],
            "router_training": router_record["training"],
            "selected_route_threshold": router_record["threshold"],
            "router_development": router_record["development"],
            "development_generation": diagnostic_summary(rows),
            "checkpoint": str(target_checkpoint),
            "checkpoint_sha256": sha256(target_checkpoint),
        }
    )
    V2_RESULTS.mkdir(parents=True, exist_ok=True)
    (V2_RESULTS / f"{condition}_seed_{seed}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    del bundle
    release_memory()
    return record


def main() -> None:
    started = time.perf_counter()
    train_features, development_features = load_or_collect_features()
    routers = train_router_cache(train_features, development_features)
    records: list[dict[str, object]] = []
    for condition in (
        "igc_matched",
        "igc_native",
        "typed_firmware",
        "adapter",
    ):
        hidden_width = 0 if condition == "igc_matched" else 16
        for seed in SEEDS:
            records.append(
                recalibrate_condition(
                    condition,
                    seed,
                    routers[(seed, hidden_width)],
                )
            )
    records.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
    manifest = {
        "status": "training_complete_development_only_router_recalibrated",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "router_feature_point": "block_24_output_before_final_rmsnorm",
        "seeds": list(SEEDS),
        "records": records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (V2_RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": [
                    {
                        "condition": row["condition"],
                        "seed": row["seed"],
                        "development_generation": row["development_generation"],
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
