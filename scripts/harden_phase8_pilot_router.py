from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    RouteRowTrainConfig,
    collect_first_step_route_features,
    generate_sequence_implant,
    generate_untouched_sequence,
    train_route_rows,
)
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    build_phase8_training_and_development,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

SOURCE_PATH = Path("phase8_artifacts/pilot/neuron_implant_seed_14199.pt")
FEATURE_PATH = Path("phase8_artifacts/cache/first_step_route_features.pt")
CHECKPOINT_PATH = Path(
    "phase8_artifacts/pilot/neuron_implant_seed_14199_hardened.pt"
)
RESULT_PATH = Path("phase8_results/pilot_seed_14199_router_hardened.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_or_collect(
    bundle: object,
    training: list[object],
    development: list[object],
    *,
    layer_index: int,
) -> tuple[FirstStepRouteFeatureSet, FirstStepRouteFeatureSet]:
    if FEATURE_PATH.exists():
        value = torch.load(FEATURE_PATH, map_location="cpu", weights_only=True)
        return (
            FirstStepRouteFeatureSet.load_state_dict(value["training"]),
            FirstStepRouteFeatureSet.load_state_dict(value["development"]),
        )
    train = collect_first_step_route_features(
        bundle,
        training,
        layer_index=layer_index,
        batch_size=8,
    )
    development_features = collect_first_step_route_features(
        bundle,
        development,
        layer_index=layer_index,
        batch_size=8,
    )
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "layer_index": layer_index,
            "training": train.state_dict(),
            "development": development_features.state_dict(),
        },
        FEATURE_PATH,
    )
    return train, development_features


def main() -> None:
    started = time.perf_counter()
    source = torch.load(SOURCE_PATH, map_location="cpu", weights_only=True)
    training, development = build_phase8_training_and_development()
    bundle = load_model_bundle(PHASE8_MODEL_ID, revision=PHASE8_MODEL_REVISION)
    train_features, development_features = load_or_collect(
        bundle,
        training,
        development,
        layer_index=source["layer_index"],
    )
    route_rows, route_training, route_development = train_route_rows(
        source["input_rows"][:2],
        train_features,
        development_features,
        device=bundle.device,
        config=RouteRowTrainConfig(
            seed=14_499,
            steps=3_000,
            batch_size=256,
            learning_rate=0.001,
            maximum_development_false_positive_rate=0.005,
        ),
    )
    hardened_rows = source["input_rows"].clone()
    hardened_rows[:2] = route_rows
    checkpoint = {
        **source,
        "stage": "phase8_development_pilot_router_hardened",
        "route_hardening_seed": 14_499,
        "route_threshold": route_development["threshold"]["threshold"],
        "input_rows": hardened_rows,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)
    layout = SequenceImplantLayout(**checkpoint["layout"])
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=checkpoint["layer_index"],
        selected_indices=checkpoint["selected_indices"],
        layout=layout,
        output_strength=checkpoint["output_strength"],
        route_threshold=checkpoint["route_threshold"],
        digit_threshold=checkpoint["digit_threshold"],
    )
    with torch.no_grad():
        implant.input_rows.copy_(checkpoint["input_rows"].to(bundle.device))
        implant.result_columns.copy_(
            checkpoint["result_columns"].to(bundle.device)
        )
    rows: list[dict[str, object]] = []
    for example in development[:40]:
        normal = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            latch_route=True,
            preserve_base_when_off=True,
            deterministic_result_step=True,
            latch_operands=True,
        )
        ablated = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            ablate_result=True,
            latch_route=True,
            preserve_base_when_off=True,
            deterministic_result_step=True,
            latch_operands=True,
        )
        rows.append(
            {
                **example.to_dict(),
                "implant_text": normal["generated_text"],
                "implant_exact": exact_format_correct(
                    normal["generated_text"],
                    example.answer or "",
                ),
                "ablation_text": ablated["generated_text"],
                "ablation_exact": exact_format_correct(
                    ablated["generated_text"],
                    example.answer or "",
                ),
                "route_active": bool(
                    normal["steps"]
                    and normal["steps"][0].get("route_active") == [True]
                ),
            }
        )
    for example in development[240:280]:
        normal = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            latch_route=True,
            preserve_base_when_off=True,
            deterministic_result_step=True,
            latch_operands=True,
        )
        base = generate_untouched_sequence(
            bundle,
            implant,
            example.prompt,
            layer_index=checkpoint["layer_index"],
            max_new_tokens=8,
        )
        rows.append(
            {
                **example.to_dict(),
                "false_route": bool(
                    normal["steps"]
                    and normal["steps"][0].get("route") == [1]
                ),
                "token_preserved": (
                    normal["generated_token_ids"] == base["generated_token_ids"]
                ),
            }
        )
    positive = rows[:40]
    negative = rows[40:]
    summary = {
        "exact_additions": sum(row["implant_exact"] for row in positive),
        "routes_active": sum(row["route_active"] for row in positive),
        "ablation_exact": sum(row["ablation_exact"] for row in positive),
        "false_routes": sum(row["false_route"] for row in negative),
        "token_preserved": sum(row["token_preserved"] for row in negative),
    }
    payload = {
        "status": "development_router_hardening_complete",
        "source_checkpoint": str(SOURCE_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "route_training": route_training,
        "route_development": route_development,
        "summary": summary,
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
