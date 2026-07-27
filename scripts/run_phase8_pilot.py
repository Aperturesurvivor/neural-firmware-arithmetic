from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    SequenceOutputTrainConfig,
    collect_sequence_features,
    evaluate_sequence_interface,
    generate_sequence_implant,
    generate_untouched_sequence,
    train_sequence_interface,
    train_sequence_output,
)
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    build_phase8_training_and_development,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

SELECTION_PATH = Path("phase8_artifacts/development_selection.pt")
FEATURE_PATH = Path("phase8_artifacts/cache/layer15_full_features.pt")
CHECKPOINT_PATH = Path("phase8_artifacts/pilot/neuron_implant_seed_14199.pt")
RESULT_PATH = Path("phase8_results/pilot_seed_14199.json")
PILOT_SEED = 14_199


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }


def load_or_collect_features(
    bundle: object,
    training: list[object],
    development: list[object],
    *,
    layer_index: int,
    layout: SequenceImplantLayout,
) -> tuple[SequenceFeatureSet, SequenceFeatureSet]:
    if FEATURE_PATH.exists():
        value = torch.load(FEATURE_PATH, map_location="cpu", weights_only=True)
        return (
            SequenceFeatureSet.load_state_dict(value["training"]),
            SequenceFeatureSet.load_state_dict(value["development"]),
        )
    train_features = collect_sequence_features(
        bundle,
        training,
        layer_index=layer_index,
        layout=layout,
        batch_size=8,
        ordinary_tokens_per_example=8,
    )
    development_features = collect_sequence_features(
        bundle,
        development,
        layer_index=layer_index,
        layout=layout,
        batch_size=8,
        ordinary_tokens_per_example=8,
    )
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "layer_index": layer_index,
            "layout": asdict(layout),
            "training": train_features.state_dict(),
            "development": development_features.state_dict(),
        },
        FEATURE_PATH,
    )
    return train_features, development_features


def trajectory_exact(
    steps: list[dict[str, object]],
    answer: str,
) -> bool:
    expected = [int(character) for character in answer] + [10]
    observed: list[int] = []
    for step in steps:
        values = step.get("result_symbols", [])
        if values:
            observed.append(int(values[0]))
    return observed[: len(expected)] == expected


def operands_exact(step: dict[str, object], a: str, b: str) -> bool:
    if step.get("operands_valid") != [True]:
        return False
    a_length = int(step["a_lengths"][0])
    b_length = int(step["b_lengths"][0])
    return (
        step["a_digits"][0][:a_length] == [int(value) for value in a]
        and step["b_digits"][0][:b_length] == [int(value) for value in b]
    )


def main() -> None:
    started = time.perf_counter()
    selection = torch.load(SELECTION_PATH, map_location="cpu", weights_only=True)
    layer_index = int(selection["selected_layer"])
    selected_indices = selection["selected_indices"]
    layout = SequenceImplantLayout(**selection["layout"])
    training, development = build_phase8_training_and_development()
    bundle = load_model_bundle(PHASE8_MODEL_ID, revision=PHASE8_MODEL_REVISION)
    train_features, development_features = load_or_collect_features(
        bundle,
        training,
        development,
        layer_index=layer_index,
        layout=layout,
    )
    torch.manual_seed(PILOT_SEED)
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=layer_index,
        selected_indices=selected_indices,
        layout=layout,
        output_strength=16.0,
        digit_threshold=0.9,
    )
    interface_training, _ = train_sequence_interface(
        implant,
        train_features,
        development_features,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=PILOT_SEED,
            steps=3_000,
            batch_size=256,
            learning_rate=0.001,
            step_loss_weight=0.0,
        ),
    )
    interface_development = compact(
        evaluate_sequence_interface(
            implant,
            development_features,
            device=bundle.device,
        )
    )
    output_training = train_sequence_output(
        bundle,
        implant,
        training,
        config=SequenceOutputTrainConfig(
            seed=PILOT_SEED,
            steps=200,
            batch_size=1,
            learning_rate=0.01,
        ),
    )
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "phase8_development_pilot",
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "seed": PILOT_SEED,
            "layer_index": layer_index,
            "layout": asdict(layout),
            "selected_indices": selected_indices,
            "route_threshold": implant.route_threshold,
            "digit_threshold": implant.digit_threshold,
            "output_strength": implant.output_strength,
            "input_rows": implant.input_rows.detach().cpu(),
            "result_columns": implant.result_columns.detach().cpu(),
        },
        CHECKPOINT_PATH,
    )
    pilot_positive = development[:40]
    pilot_negative = development[240:280]
    rows: list[dict[str, object]] = []
    for example in pilot_positive:
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
                "operands_exact": bool(
                    normal["steps"]
                    and operands_exact(normal["steps"][0], example.a, example.b)
                ),
                "trajectory_exact": trajectory_exact(
                    normal["steps"],
                    example.answer or "",
                ),
            }
        )
    for example in pilot_negative:
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
            layer_index=layer_index,
            max_new_tokens=8,
        )
        rows.append(
            {
                **example.to_dict(),
                "implant_text": normal["generated_text"],
                "base_text": base["generated_text"],
                "false_route": bool(
                    normal["steps"]
                    and normal["steps"][0].get("route") == [1]
                ),
                "token_preserved": (
                    normal["generated_token_ids"] == base["generated_token_ids"]
                ),
            }
        )
    positive_rows = rows[: len(pilot_positive)]
    negative_rows = rows[len(pilot_positive) :]
    summary = {
        "positive_examples": len(positive_rows),
        "exact_additions": sum(row["implant_exact"] for row in positive_rows),
        "routes_active": sum(row["route_active"] for row in positive_rows),
        "operands_exact": sum(row["operands_exact"] for row in positive_rows),
        "trajectories_exact": sum(
            row["trajectory_exact"] for row in positive_rows
        ),
        "ablation_exact": sum(
            row["ablation_exact"] for row in positive_rows
        ),
        "negative_examples": len(negative_rows),
        "false_routes": sum(row["false_route"] for row in negative_rows),
        "token_preserved": sum(
            row["token_preserved"] for row in negative_rows
        ),
    }
    payload = {
        "status": "development_pilot_complete",
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "seed": PILOT_SEED,
        "layer_index": layer_index,
        "selected_indices": selected_indices.tolist(),
        "layout": asdict(layout),
        "learned_parameters": implant.trainable_parameter_count,
        "calculator_learned_parameters": (
            implant.calculator.trainable_parameter_count
        ),
        "interface_training": interface_training,
        "interface_development": interface_development,
        "output_training": output_training,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
        },
        "summary": summary,
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
