from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from neural_firmware.phase6_data import (
    Phase6Example,
    build_phase6_calibration_examples,
    build_phase6_gate_examples,
)
from neural_firmware.phase6_firmware import (
    NeuralCallController,
    NeuralRegisterMapper,
    install_neural_firmware,
)
from neural_firmware.phase6_training import (
    Phase6FeatureSet,
    collect_phase6_features,
    evaluate_fused_controller,
    evaluate_register_mapper,
    generate_neural_firmware,
    register_targets,
    select_fused_threshold,
    set_phase6_seed,
)
from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import ModelBundle, load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 11_701
MAX_DIGITS = 8
INPUT_DEPTH = 1
OUTPUT_DEPTH = 24
MODEL_WIDTH = 192
ATTENTION_HEADS = 8
DECODER_LAYERS = 2
CONTROLLER_WIDTH = 64
FEATURE_DIRECTORY = Path("phase6_artifacts/cache")
V5_CHECKPOINT = Path(
    "phase6_artifacts/pilot_v5/neural_firmware_seed_11701.pt"
)
ARTIFACT_DIRECTORY = Path("phase6_artifacts/pilot_v6")
RESULT_PATH = Path("phase6_results/pilot_v6.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_features(features: Phase6FeatureSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features.state_dict(), path)


def load_features(
    path: Path,
    examples: list[Phase6Example],
) -> Phase6FeatureSet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["register_targets"] = register_targets(
        examples,
        max_digits=MAX_DIGITS,
    )
    payload["call_targets"] = torch.tensor(
        [example.call_count for example in examples],
        dtype=torch.long,
    )
    payload["controller_targets"] = torch.tensor(
        [example.controller_target for example in examples],
        dtype=torch.long,
    )
    return Phase6FeatureSet(**payload)


def prepare_features(
    train_examples: list[Phase6Example],
    development_examples: list[Phase6Example],
) -> tuple[Phase6FeatureSet, Phase6FeatureSet]:
    train_path = FEATURE_DIRECTORY / "phase6_train_v1.pt"
    development_path = FEATURE_DIRECTORY / "phase6_development_v1.pt"
    if train_path.exists() and development_path.exists():
        return (
            load_features(train_path, train_examples),
            load_features(development_path, development_examples),
        )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_phase6_features(
        bundle,
        train_examples,
        input_depth_after_blocks=INPUT_DEPTH,
        output_depth_after_blocks=OUTPUT_DEPTH,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    save_features(train_features, train_path)
    development_features = collect_phase6_features(
        bundle,
        development_examples,
        input_depth_after_blocks=INPUT_DEPTH,
        output_depth_after_blocks=OUTPUT_DEPTH,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    save_features(development_features, development_path)
    del bundle
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return train_features, development_features


def prepare_evaluation_features(
    examples: list[Phase6Example],
    *,
    name: str,
) -> Phase6FeatureSet:
    path = FEATURE_DIRECTORY / f"phase6_{name}_v1.pt"
    if path.exists():
        return load_features(path, examples)
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    features = collect_phase6_features(
        bundle,
        examples,
        input_depth_after_blocks=INPUT_DEPTH,
        output_depth_after_blocks=OUTPUT_DEPTH,
        max_digits=MAX_DIGITS,
        batch_size=8,
    )
    save_features(features, path)
    del bundle
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return features


def concatenate_features(
    feature_sets: tuple[Phase6FeatureSet, ...],
) -> Phase6FeatureSet:
    if any(features.controller_targets is None for features in feature_sets):
        raise ValueError("all feature sets require controller targets")
    maximum_sequence = max(
        features.early_hidden.shape[1] for features in feature_sets
    )
    early_hidden = [
        torch.nn.functional.pad(
            features.early_hidden,
            (0, 0, 0, maximum_sequence - features.early_hidden.shape[1]),
        )
        for features in feature_sets
    ]
    attention_mask = [
        torch.nn.functional.pad(
            features.attention_mask,
            (0, maximum_sequence - features.attention_mask.shape[1]),
        )
        for features in feature_sets
    ]
    return Phase6FeatureSet(
        early_hidden=torch.cat(early_hidden),
        attention_mask=torch.cat(attention_mask),
        anchor_positions=torch.cat(
            [features.anchor_positions for features in feature_sets]
        ),
        late_hidden=torch.cat(
            [features.late_hidden for features in feature_sets]
        ),
        register_targets=torch.cat(
            [features.register_targets for features in feature_sets]
        ),
        call_targets=torch.cat(
            [features.call_targets for features in feature_sets]
        ),
        controller_targets=torch.cat(
            [
                features.controller_targets
                for features in feature_sets
                if features.controller_targets is not None
            ]
        ),
    )


@torch.inference_mode()
def generate_base(
    bundle: ModelBundle,
    example: Phase6Example,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = chat_prompt_ids(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(input_ids)
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_token = bundle.model.lm_head(hidden).argmax(dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=torch.long,
                    device=bundle.device,
                ),
            ),
            dim=1,
        )
        outputs = bundle.model.model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
    }


def summarize_generation(
    rows: list[dict[str, object]],
    base_negative_rows: list[dict[str, object]],
) -> dict[str, object]:
    positives = [row for row in rows if row["target_call_count"] > 0]
    singles = [row for row in rows if row["target_call_count"] == 1]
    chains = [row for row in rows if row["target_call_count"] == 2]
    negatives = [row for row in rows if row["target_call_count"] == 0]
    preserved = sum(
        row["generated_token_ids"] == base["generated_token_ids"]
        for row, base in zip(negatives, base_negative_rows, strict=True)
    )
    eligible = [
        row
        for row in positives
        if row["call_count_exact"] is True
        and row["used_registers_exact"] is True
    ]
    return {
        "positive_examples": len(positives),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positives
        ),
        "single_examples": len(singles),
        "single_mathematical_correct": sum(
            row["mathematical_correct"] is True for row in singles
        ),
        "chain_examples": len(chains),
        "chain_mathematical_correct": sum(
            row["mathematical_correct"] is True for row in chains
        ),
        "call_count_exact": sum(
            row["call_count_exact"] is True for row in rows
        ),
        "used_registers_exact": sum(
            row["used_registers_exact"] is True for row in positives
        ),
        "eligible_examples": len(eligible),
        "eligible_mathematical_correct": sum(
            row["mathematical_correct"] is True for row in eligible
        ),
        "negative_examples": len(negatives),
        "false_calls": sum(row["route_active"] is True for row in negatives),
        "token_exact_preserved": preserved,
    }


def main() -> None:
    started = time.perf_counter()
    calibration_examples = build_phase6_calibration_examples()
    gate_examples = build_phase6_gate_examples()
    calibration_features = prepare_evaluation_features(
        calibration_examples,
        name="calibration",
    )
    gate_features = prepare_evaluation_features(
        gate_examples,
        name="gate",
    )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    set_phase6_seed(SEED)
    mapper = NeuralRegisterMapper(
        bundle.model.config.hidden_size,
        max_digits=MAX_DIGITS,
        model_width=MODEL_WIDTH,
        attention_heads=ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
    )
    if not V5_CHECKPOINT.exists():
        raise FileNotFoundError(
            "pilot v6 requires the retained pilot-v5 checkpoint"
        )
    v5_state = torch.load(
        V5_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    mapper.load_state_dict(v5_state["mapper"])
    mapper.to(bundle.device).eval()
    mapper_gate = evaluate_register_mapper(
        mapper,
        gate_features,
        gate_examples,
        device=bundle.device,
    )
    controller = NeuralCallController(
        bundle.model.config.hidden_size,
        hidden_width=CONTROLLER_WIDTH,
    )
    controller.load_state_dict(v5_state["controller"])
    controller.to(bundle.device).eval()
    selected_threshold = select_fused_threshold(
        mapper,
        controller,
        calibration_features,
        calibration_examples,
        device=bundle.device,
    )
    controller_gate = evaluate_fused_controller(
        mapper,
        controller,
        gate_features,
        gate_examples,
        device=bundle.device,
        threshold=float(selected_threshold["threshold"]),
    )
    installation = install_neural_firmware(
        bundle.model,
        input_depth_after_blocks=INPUT_DEPTH,
        output_depth_after_blocks=OUTPUT_DEPTH,
        max_digits=MAX_DIGITS,
        model_width=MODEL_WIDTH,
        attention_heads=ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
        controller_width=CONTROLLER_WIDTH,
        output_strength=64.0,
    )
    installation.capture.mapper.load_state_dict(mapper.state_dict())
    installation.final.controller.load_state_dict(controller.state_dict())
    installation.final.output_decoder.load_state_dict(
        v5_state["output_decoder"]
    )
    diagnostic_examples = (
        gate_examples[:40]
        + gate_examples[200:240]
        + gate_examples[400:440]
    )
    negative_examples = [
        example
        for example in diagnostic_examples
        if example.call_count == 0
    ]
    installation.set_context(None)
    base_negative_rows = [
        generate_base(bundle, example, max_new_tokens=20)
        for example in negative_examples
    ]
    generation_rows = [
        generate_neural_firmware(
            bundle,
            installation,
            example,
            route_mode="learned",
            route_threshold=float(selected_threshold["threshold"]),
            max_new_tokens=16 if example.call_count else 20,
        )
        for example in diagnostic_examples
    ]
    checkpoint_path = ARTIFACT_DIRECTORY / "neural_firmware_seed_11701.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(installation.state_dict(), checkpoint_path)
    compact_mapper = {
        key: value
        for key, value in mapper_gate.items()
        if key != "rows"
    }
    compact_controller = {
        key: value
        for key, value in controller_gate.items()
        if key != "rows"
    }
    result = {
        "status": "development_pilot",
        "pilot_version": 6,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "architecture": {
            "input_depth_after_blocks": INPUT_DEPTH,
            "output_depth_after_blocks": OUTPUT_DEPTH,
            "max_digits": MAX_DIGITS,
            "maximum_calls": 2,
            "model_width": MODEL_WIDTH,
            "attention_heads": ATTENTION_HEADS,
            "decoder_layers": DECODER_LAYERS,
            "controller_width": CONTROLLER_WIDTH,
            "learned_parameters": installation.learned_parameter_count,
            "fixed_parser_at_inference": False,
            "routing_fusion": "early_sequence_logits_plus_late_anchor_logits",
            "call_count_source": "typed_third_register_occupancy",
            "controller_classes": (
                "no_call",
                "one_add",
                "two_adds",
                "unsupported_single",
                "unsupported_multiple",
            ),
        },
        "data": {
            "calibration_examples": len(calibration_examples),
            "gate_examples": len(gate_examples),
        },
        "mapper_training": {
            "loaded_from": str(V5_CHECKPOINT),
            "checkpoint_sha256": sha256(V5_CHECKPOINT),
        },
        "mapper_gate": compact_mapper,
        "controller_training": {
            "loaded_from": str(V5_CHECKPOINT),
            "checkpoint_sha256": sha256(V5_CHECKPOINT),
        },
        "selected_threshold": selected_threshold,
        "controller_gate": compact_controller,
        "output_training": {
            "loaded_from": str(V5_CHECKPOINT),
            "checkpoint_sha256": sha256(V5_CHECKPOINT),
        },
        "generation_summary": summarize_generation(
            generation_rows,
            base_negative_rows,
        ),
        "generation_rows": generation_rows,
        "base_negative_rows": base_negative_rows,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "initialization_checkpoint": str(V5_CHECKPOINT),
        "initialization_checkpoint_sha256": sha256(V5_CHECKPOINT),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json_atomic(RESULT_PATH, result)
    print(
        json.dumps(
            {
                "architecture": result["architecture"],
                "mapper_gate": compact_mapper,
                "controller_gate": compact_controller,
                "selected_threshold": selected_threshold,
                "generation_summary": result["generation_summary"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "wall_time_seconds": result["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
