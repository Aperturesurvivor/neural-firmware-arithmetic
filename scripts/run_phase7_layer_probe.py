from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
from neural_firmware.phase7_data import build_phase7_audit_examples
from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    SequenceNeuronImplantMLP,
)
from neural_firmware.phase7_sequence_training import (
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    collect_sequence_features,
    evaluate_sequence_interface,
    train_sequence_interface,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CACHE_DIRECTORY = Path("phase7_artifacts/cache/layer_probe_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    return parser.parse_args()


def make_data() -> tuple[list[object], list[object], list[object]]:
    train = (
        make_semantic_addition_examples(
            count=500,
            min_digits=1,
            max_digits=4,
            seed=13_001,
            split="phase7_layer_probe_train_positive",
            families=PHASE5_TRAIN_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=500,
            min_digits=1,
            max_digits=4,
            seed=13_002,
            split="phase7_layer_probe_train_negative",
            families=PHASE5_TRAIN_NEGATIVE_FAMILIES,
        )
    )
    development = (
        make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=4,
            seed=13_003,
            split="phase7_layer_probe_development_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=100,
            min_digits=1,
            max_digits=4,
            seed=13_004,
            split="phase7_layer_probe_development_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        )
    )
    return train, development, build_phase7_audit_examples()


def feature_path(layer: int, split: str) -> Path:
    return CACHE_DIRECTORY / f"layer_{layer:02d}_{split}.pt"


def load_or_collect(
    bundle: object,
    examples: list[object],
    *,
    layer: int,
    split: str,
    layout: SequenceImplantLayout,
) -> SequenceFeatureSet:
    path = feature_path(layer, split)
    if path.exists():
        return SequenceFeatureSet.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    features = collect_sequence_features(
        bundle,
        examples,
        layer_index=layer,
        layout=layout,
        batch_size=8,
        ordinary_tokens_per_example=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.state_dict(), path)
    return features


def operand_metrics(
    implant: SequenceNeuronImplantMLP,
    features: SequenceFeatureSet,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    with torch.inference_mode():
        interface = implant.interface_logits(features.hidden.to(device))
        hard = implant.hard_interface(interface)
        roles = features.role_targets.to(device)
        digits = features.digit_targets.to(device)
        operand = roles > 0
        role_correct = hard.roles[operand] == roles[operand]
        digit_correct = hard.digits[operand] == digits[operand]
        return {
            "operand_tokens": int(operand.sum()),
            "role_accuracy": float(role_correct.float().mean()),
            "digit_accuracy": float(digit_correct.float().mean()),
            "joint_role_digit_accuracy": float(
                (role_correct & digit_correct).float().mean()
            ),
        }


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.layer >= 24:
        raise ValueError("Qwen2.5-0.5B layer must be in [0, 23]")
    layout = SequenceImplantLayout(max_digits=4)
    train, development, audit = make_data()
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    started = time.perf_counter()
    train_features = load_or_collect(
        bundle,
        train,
        layer=args.layer,
        split="train",
        layout=layout,
    )
    development_features = load_or_collect(
        bundle,
        development,
        layer=args.layer,
        split="development",
        layout=layout,
    )
    audit_features = load_or_collect(
        bundle,
        audit,
        layer=args.layer,
        split="audit",
        layout=layout,
    )
    symbolic_features = load_or_collect(
        bundle,
        audit[:20],
        layer=args.layer,
        split="audit_symbolic",
        layout=layout,
    )
    word_features = load_or_collect(
        bundle,
        audit[20:40],
        layer=args.layer,
        split="audit_word",
        layout=layout,
    )
    base_mlp = bundle.model.model.layers[args.layer].mlp
    implant = SequenceNeuronImplantMLP(
        base_mlp,
        torch.arange(layout.total_width),
        layout=layout,
    )
    reference = next(base_mlp.parameters())
    implant.to(device=reference.device, dtype=reference.dtype)
    training, development_metrics = train_sequence_interface(
        implant,
        train_features,
        development_features,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=13_100 + args.layer,
            steps=2_000,
            batch_size=256,
            learning_rate=0.001,
            step_loss_weight=0.0,
        ),
    )
    audit_metrics = evaluate_sequence_interface(
        implant,
        audit_features,
        device=bundle.device,
    )
    audit_metrics = {
        key: value
        for key, value in audit_metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    payload = {
        "status": "post_audit_layer_probe",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": args.layer,
        "data": {
            "train_examples": len(train),
            "development_examples": len(development),
            "audit_examples": len(audit),
        },
        "training": training,
        "development": development_metrics,
        "audit": audit_metrics,
        "audit_symbolic_operands": operand_metrics(
            implant,
            symbolic_features,
            device=bundle.device,
        ),
        "audit_word_operands": operand_metrics(
            implant,
            word_features,
            device=bundle.device,
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }
    result = Path(f"phase7_results/layer_probe_v1_layer_{args.layer:02d}.json")
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
