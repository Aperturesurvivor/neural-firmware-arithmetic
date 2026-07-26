from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from neural_firmware.phase6_data import (
    Phase6Example,
    build_phase6_development_examples,
    build_phase6_output_training_examples,
    build_phase6_training_examples,
)
from neural_firmware.phase6_firmware import (
    NeuralCallController,
    NeuralRegisterMapper,
    install_neural_firmware,
)
from neural_firmware.phase6_training import (
    ControllerTrainConfig,
    MapperTrainConfig,
    OutputTrainConfig,
    Phase6FeatureSet,
    collect_phase6_features,
    evaluate_call_controller,
    evaluate_register_mapper,
    generate_neural_firmware,
    select_call_threshold,
    train_call_controller,
    train_output_decoder,
    train_register_mapper,
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
ARTIFACT_DIRECTORY = Path("phase6_artifacts/pilot_v1")
RESULT_PATH = Path("phase6_results/pilot_v1.json")


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


def load_features(path: Path) -> Phase6FeatureSet:
    return Phase6FeatureSet(
        **torch.load(path, map_location="cpu", weights_only=True)
    )


def prepare_features(
    train_examples: list[Phase6Example],
    development_examples: list[Phase6Example],
) -> tuple[Phase6FeatureSet, Phase6FeatureSet]:
    train_path = FEATURE_DIRECTORY / "phase6_train_v1.pt"
    development_path = FEATURE_DIRECTORY / "phase6_development_v1.pt"
    if train_path.exists() and development_path.exists():
        return load_features(train_path), load_features(development_path)
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
    generated = bundle.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=bundle.tokenizer.pad_token_id,
        eos_token_id=bundle.tokenizer.eos_token_id,
    )[0, len(prompt) :].tolist()
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
    train_examples = build_phase6_training_examples()
    development_examples = build_phase6_development_examples()
    output_examples = build_phase6_output_training_examples()
    train_features, development_features = prepare_features(
        train_examples,
        development_examples,
    )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    mapper = NeuralRegisterMapper(
        bundle.model.config.hidden_size,
        max_digits=MAX_DIGITS,
        model_width=MODEL_WIDTH,
        attention_heads=ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
    )
    mapper_training = train_register_mapper(
        mapper,
        train_features,
        MapperTrainConfig(
            seed=SEED,
            steps=6_000,
            batch_size=32,
            learning_rate=0.001,
        ),
        device=bundle.device,
    )
    mapper_development = evaluate_register_mapper(
        mapper,
        development_features,
        development_examples,
        device=bundle.device,
    )
    controller = NeuralCallController(
        bundle.model.config.hidden_size,
        hidden_width=CONTROLLER_WIDTH,
    )
    controller_training = train_call_controller(
        controller,
        train_features,
        ControllerTrainConfig(
            seed=SEED + 100,
            steps=2_500,
            batch_size=512,
            learning_rate=0.003,
        ),
        device=bundle.device,
    )
    selected_threshold = select_call_threshold(
        controller,
        development_features,
        development_examples,
        device=bundle.device,
    )
    controller_development = evaluate_call_controller(
        controller,
        development_features,
        development_examples,
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
    output_training = train_output_decoder(
        bundle,
        installation,
        output_examples,
        OutputTrainConfig(
            seed=SEED + 200,
            steps=240,
            batch_size=2,
            learning_rate=0.01,
        ),
    )
    diagnostic_examples = (
        development_examples[:40]
        + development_examples[400:440]
        + development_examples[800:840]
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
        for key, value in mapper_development.items()
        if key != "rows"
    }
    compact_controller = {
        key: value
        for key, value in controller_development.items()
        if key != "rows"
    }
    result = {
        "status": "development_pilot",
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
        },
        "data": {
            "train_examples": len(train_examples),
            "development_examples": len(development_examples),
            "output_examples": len(output_examples),
        },
        "mapper_training": mapper_training,
        "mapper_development": compact_mapper,
        "controller_training": controller_training,
        "selected_threshold": selected_threshold,
        "controller_development": compact_controller,
        "output_training": output_training,
        "generation_summary": summarize_generation(
            generation_rows,
            base_negative_rows,
        ),
        "generation_rows": generation_rows,
        "base_negative_rows": base_negative_rows,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
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
                "mapper_development": compact_mapper,
                "controller_development": compact_controller,
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
