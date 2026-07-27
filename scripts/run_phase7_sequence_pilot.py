from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
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
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
    mathematical_correct,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 12_801
LAYER_INDEX = 23
CENSUS_PATH = Path("phase7_artifacts/sequence_census_v1.pt")
CACHE_DIRECTORY = Path("phase7_artifacts/cache")
CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_pilot_v2/neuron_implant_seed_12801.pt"
)
RESULT_PATH = Path("phase7_results/sequence_pilot_v2.json")


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_features(features: SequenceFeatureSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.state_dict(), path)


def load_features(path: Path) -> SequenceFeatureSet:
    return SequenceFeatureSet.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_data() -> tuple[
    list[SemanticPromptExample],
    list[SemanticPromptExample],
    list[SemanticPromptExample],
]:
    train = (
        make_semantic_addition_examples(
            count=400,
            min_digits=1,
            max_digits=4,
            seed=12_801,
            split="phase7_sequence_train_positive",
            families=PHASE5_TRAIN_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=400,
            min_digits=1,
            max_digits=4,
            seed=12_802,
            split="phase7_sequence_train_negative",
            families=PHASE5_TRAIN_NEGATIVE_FAMILIES,
        )
    )
    development = (
        make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=4,
            seed=12_803,
            split="phase7_sequence_development_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=100,
            min_digits=1,
            max_digits=4,
            seed=12_804,
            split="phase7_sequence_development_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        )
    )
    evaluation = (
        make_semantic_addition_examples(
            count=20,
            min_digits=1,
            max_digits=4,
            seed=12_851,
            split="phase7_sequence_evaluation_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=20,
            min_digits=1,
            max_digits=4,
            seed=12_852,
            split="phase7_sequence_evaluation_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        )
    )
    return train, development, evaluation


def prepare_features(
    bundle: object,
    train: list[SemanticPromptExample],
    development: list[SemanticPromptExample],
    layout: SequenceImplantLayout,
) -> tuple[SequenceFeatureSet, SequenceFeatureSet]:
    train_path = CACHE_DIRECTORY / "sequence_interface_train_v1.pt"
    development_path = CACHE_DIRECTORY / "sequence_interface_development_v1.pt"
    if train_path.exists():
        train_features = load_features(train_path)
    else:
        train_features = collect_sequence_features(
            bundle,
            train,
            layer_index=LAYER_INDEX,
            layout=layout,
            batch_size=8,
        )
        save_features(train_features, train_path)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if development_path.exists():
        development_features = load_features(development_path)
    else:
        development_features = collect_sequence_features(
            bundle,
            development,
            layer_index=LAYER_INDEX,
            layout=layout,
            batch_size=8,
        )
        save_features(development_features, development_path)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return train_features, development_features


def save_checkpoint(
    implant: object,
    selected_indices: torch.Tensor,
    layout: SequenceImplantLayout,
    *,
    stage: str,
) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layer_index": LAYER_INDEX,
            "layout": asdict(layout),
            "selected_indices": selected_indices,
            "route_threshold": implant.route_threshold,
            "output_strength": implant.output_strength,
            "input_rows": implant.input_rows.detach().cpu(),
            "result_columns": implant.result_columns.detach().cpu(),
        },
        CHECKPOINT_PATH,
    )


def singleton(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def compact_generation(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    return {
        "positive_examples": len(positives),
        "positive_exact": sum(row["mathematical_correct"] for row in positives),
        "positive_ablation_exact": sum(
            row["ablation_mathematical_correct"] for row in positives
        ),
        "positive_first_step_route_active": sum(
            row["first_step_route_active"] for row in positives
        ),
        "positive_first_step_operands_exact": sum(
            row["first_step_operands_exact"] for row in positives
        ),
        "negative_examples": len(negatives),
        "negative_false_routes": sum(row["any_route_active"] for row in negatives),
        "negative_token_preservation": sum(row["token_preserved"] for row in negatives),
    }


def main() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    census = torch.load(CENSUS_PATH, map_location="cpu", weights_only=True)
    selected_indices = census["selected_indices"]
    train_examples, development_examples, evaluation_examples = make_data()
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    started = time.perf_counter()
    train_features, development_features = prepare_features(
        bundle,
        train_examples,
        development_examples,
        layout,
    )
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=LAYER_INDEX,
        selected_indices=selected_indices,
        layout=layout,
        output_strength=16.0,
    )
    interface_training, interface_development = train_sequence_interface(
        implant,
        train_features,
        development_features,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=SEED,
            steps=1_500,
            batch_size=256,
            learning_rate=0.003,
        ),
    )
    print(
        "interface "
        f"route_tpr={interface_development['route_true_positive_rate']:.4f} "
        f"route_fpr={interface_development['route_false_positive_rate']:.4f} "
        f"role={interface_development['operand_role_accuracy']:.4f} "
        f"digit={interface_development['digit_accuracy_on_operands']:.4f} "
        f"step={interface_development['step_accuracy']:.4f}",
        flush=True,
    )
    save_checkpoint(implant, selected_indices, layout, stage="interface")
    partial = {
        "status": "interface_complete_output_pending",
        "interface_training": interface_training,
        "interface_development": interface_development,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
        },
    }
    write_json(RESULT_PATH, partial)

    output_training = train_sequence_output(
        bundle,
        implant,
        train_examples,
        config=SequenceOutputTrainConfig(
            seed=SEED,
            steps=100,
            batch_size=1,
            learning_rate=0.01,
        ),
    )
    print(
        "output "
        f"initial={output_training['initial_loss']:.4f} "
        f"final={output_training['final_loss']:.4f}",
        flush=True,
    )
    save_checkpoint(implant, selected_indices, layout, stage="output")

    rows: list[dict[str, object]] = []
    for index, example in enumerate(evaluation_examples):
        result = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
        )
        ablated = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            ablate_result=True,
        )
        first = result["steps"][0]
        a_digits = singleton(first.get("a_digits"))
        b_digits = singleton(first.get("b_digits"))
        a_lengths = singleton(first.get("a_lengths"))
        b_lengths = singleton(first.get("b_lengths"))
        expected_a = [int(character) for character in example.a]
        expected_b = [int(character) for character in example.b]
        row = {
            **example.to_dict(),
            "implant": result,
            "ablated": ablated,
            "mathematical_correct": (
                mathematical_correct(result["generated_text"], example.answer)
                if example.answer is not None
                else False
            ),
            "ablation_mathematical_correct": (
                mathematical_correct(ablated["generated_text"], example.answer)
                if example.answer is not None
                else False
            ),
            "first_step_route_active": (
                singleton(first.get("route_active")) is True
            ),
            "first_step_operands_exact": (
                a_digits[:a_lengths] == expected_a
                and b_digits[:b_lengths] == expected_b
            ),
            "any_route_active": any(
                any(step.get("route_active", [])) for step in result["steps"]
            ),
        }
        if not example.route_label:
            untouched = generate_untouched_sequence(
                bundle,
                implant,
                example.prompt,
                layer_index=LAYER_INDEX,
                max_new_tokens=8,
            )
            row["untouched"] = untouched
            row["token_preserved"] = (
                result["generated_token_ids"] == untouched["generated_token_ids"]
            )
        else:
            row["token_preserved"] = False
        rows.append(row)
        print(
            f"evaluate={index + 1}/{len(evaluation_examples)} "
            f"route={example.route_label} text={result['generated_text']!r}",
            flush=True,
        )

    final_interface = evaluate_sequence_interface(
        implant,
        development_features,
        device=bundle.device,
    )
    compact_interface = {
        key: value
        for key, value in final_interface.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    payload = {
        "status": "development_sequence_pilot_v2",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "layer_index": LAYER_INDEX,
        "layout": asdict(layout),
        "selected_indices": selected_indices.tolist(),
        "mlp_width_before": 4864,
        "mlp_width_after": implant.mlp_width,
        "calculator_trainable_parameters": (
            implant.calculator.trainable_parameter_count
        ),
        "interface_trainable_parameters": implant.input_rows.numel(),
        "result_trainable_parameters": implant.result_columns.numel(),
        "total_trainable_parameters": implant.trainable_parameter_count,
        "interface_training": interface_training,
        "interface_development": compact_interface,
        "output_training": output_training,
        "generation_summary": compact_generation(rows),
        "generation_rows": rows,
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": sha256(CHECKPOINT_PATH),
        },
        "wall_time_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(bundle.device),
            "mps_available": torch.backends.mps.is_available(),
            "git_commit_before_phase7_changes": git_commit(),
        },
    }
    write_json(RESULT_PATH, payload)
    print(json.dumps(payload["generation_summary"], indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
