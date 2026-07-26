from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.internal_data import make_internal_addition_examples
from neural_firmware.internal_firmware import install_internal_firmware_layer
from neural_firmware.internal_training import (
    InternalDecoderTrainConfig,
    generate_internal,
    train_internal_decoder,
)
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTH = 22


def main() -> None:
    artifact_directory = Path("phase3_artifacts/decoder_smoke")
    result_directory = Path("phase3_results")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(MODEL_ID, revision=REVISION)
    wrapper = install_internal_firmware_layer(
        bundle.model,
        depth_after_blocks=DEPTH,
        strength=64.0,
    )
    digit_checkpoint = Path(
        f"phase3_artifacts/digit_probe_pilot_v1/depth_{DEPTH}_probe.pt"
    )
    wrapper.unit.digit_encoder.load_state_dict(
        torch.load(digit_checkpoint, map_location=bundle.device, weights_only=True)
    )
    train_examples = make_internal_addition_examples(
        count=80,
        min_digits=1,
        max_digits=4,
        seed=58_901,
        split="smoke_train",
    )
    evaluation = make_internal_addition_examples(
        count=8,
        min_digits=1,
        max_digits=6,
        seed=58_902,
        split="smoke_eval",
    )
    train_result = train_internal_decoder(
        bundle,
        wrapper,
        train_examples,
        InternalDecoderTrainConfig(
            seed=811,
            steps=30,
            batch_size=2,
            learning_rate=0.01,
        ),
    )
    torch.save(wrapper.unit.state_dict(), artifact_directory / "unit.pt")
    internal_rows = [
        generate_internal(bundle, wrapper, example, enabled=True).to_dict()
        for example in evaluation
    ]
    off_rows = [
        generate_internal(bundle, wrapper, example, enabled=False).to_dict()
        for example in evaluation
    ]
    result = {
        "status": "smoke",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "depth_after_blocks": DEPTH,
        "strength": 64.0,
        "train_result": asdict(train_result),
        "internal_correct": sum(row["exact"] for row in internal_rows),
        "off_correct": sum(row["exact"] for row in off_rows),
        "examples": len(evaluation),
        "internal_predictions": internal_rows,
        "off_predictions": off_rows,
    }
    (result_directory / "decoder_smoke.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
