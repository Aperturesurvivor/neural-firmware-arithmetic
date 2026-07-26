from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_firmware.phase6_data import Phase6Example
from neural_firmware.phase6_firmware import install_neural_firmware
from neural_firmware.phase6_training import generate_neural_firmware
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_CHECKPOINT = Path(
    "phase6_artifacts/pilot_v6/neural_firmware_seed_11701.pt"
)
DEFAULT_THRESHOLD = 0.97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6 fully neural arithmetic interface.",
    )
    parser.add_argument("prompt", help="Natural-language prompt to give Qwen.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Neural calculator activation threshold.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}; run "
            "scripts/run_phase6_pilot.py first"
        )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    installation = install_neural_firmware(
        bundle.model,
        input_depth_after_blocks=1,
        output_depth_after_blocks=24,
        max_digits=8,
        model_width=192,
        attention_heads=8,
        decoder_layers=2,
        controller_width=64,
        output_strength=64.0,
    )
    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    installation.load_state_dict(state)
    example = Phase6Example(
        prompt=args.prompt,
        operands=(),
        call_count=0,
        controller_target=0,
        answer=None,
        intermediate_answers=(),
        family="interactive",
        family_index=0,
        split="interactive",
    )
    row = generate_neural_firmware(
        bundle,
        installation,
        example,
        route_mode="learned",
        route_threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
    )
    call_count = int(row["predicted_call_count"] or 0)
    call_outputs = row["program_call_symbols"]
    if isinstance(call_outputs, list):
        call_outputs = call_outputs[:call_count]
    print(
        json.dumps(
            {
                "prompt": row["prompt"],
                "generated_text": row["generated_text"],
                "calculator_activated": row["route_active"],
                "calculator_calls": call_count,
                "route_probability": row["route_probability"],
                "call_outputs": call_outputs,
                "latency_seconds": row["latency_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
