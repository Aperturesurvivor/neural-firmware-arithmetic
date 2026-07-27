from __future__ import annotations

import argparse
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
    generate_sequence_implant,
    generate_untouched_sequence,
)
from neural_firmware.pretrained_training import load_model_bundle

DEFAULT_CHECKPOINT = Path(
    "phase7_artifacts/sequence_layer16_router_hardened_v1/"
    "neuron_implant_seed_13202.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "b483c3fbcec274cdf2f1b23acff33ae63966575c4d0f491eed3f182a73f24eea"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 7 in-place deterministic-neuron implant on one "
            "natural-language prompt."
        ),
    )
    parser.add_argument("prompt", help="Natural-language prompt to give Qwen.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Include the complete per-token internal trace.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def singleton(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def active_calculator_symbols(
    result: dict[str, object],
    *,
    eos_symbol: int,
) -> list[int | str]:
    symbols: list[int | str] = []
    for step in result["steps"]:
        if singleton(step["route_active"]) is not True:
            continue
        symbol = int(singleton(step["result_symbols"]))
        symbols.append("EOS" if symbol == eos_symbol else symbol)
        if symbol == eos_symbol:
            break
    return symbols


def decoded_operands(first_step: dict[str, object]) -> dict[str, object] | None:
    if singleton(first_step["operands_valid"]) is not True:
        return None
    a_digits = singleton(first_step["a_digits"])
    b_digits = singleton(first_step["b_digits"])
    a_length = int(singleton(first_step["a_lengths"]))
    b_length = int(singleton(first_step["b_lengths"]))
    return {
        "a": "".join(str(digit) for digit in a_digits[:a_length]),
        "b": "".join(str(digit) for digit in b_digits[:b_length]),
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}; use one of the frozen "
            "Phase 7 router-hardened checkpoints"
        )
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint hash differs from the frozen default; pass its expected "
            "SHA-256 explicitly if this is intentional"
        )
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    bundle = load_model_bundle(
        checkpoint["model_id"],
        revision=checkpoint["model_revision"],
    )
    layout = SequenceImplantLayout(**checkpoint["layout"])
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=checkpoint["layer_index"],
        selected_indices=checkpoint["selected_indices"],
        layout=layout,
        output_strength=checkpoint["output_strength"],
        route_threshold=checkpoint["route_threshold"],
        digit_threshold=checkpoint.get("digit_threshold", 0.0),
        use_swiglu_interface=checkpoint.get("use_swiglu_interface", False),
    )
    with torch.no_grad():
        implant.input_rows.copy_(checkpoint["input_rows"].to(bundle.device))
        if "gate_rows" in checkpoint:
            implant.gate_rows.copy_(checkpoint["gate_rows"].to(bundle.device))
        implant.result_columns.copy_(
            checkpoint["result_columns"].to(bundle.device)
        )

    started = time.perf_counter()
    result = generate_sequence_implant(
        bundle,
        implant,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        latch_route=True,
        preserve_base_when_off=True,
        deterministic_result_step=True,
        latch_operands=True,
    )
    implant_seconds = time.perf_counter() - started
    started = time.perf_counter()
    untouched = generate_untouched_sequence(
        bundle,
        implant,
        args.prompt,
        layer_index=checkpoint["layer_index"],
        max_new_tokens=args.max_new_tokens,
    )
    base_seconds = time.perf_counter() - started

    first = result["steps"][0]
    route_active = singleton(first["route_active"]) is True
    payload: dict[str, object] = {
        "prompt": args.prompt,
        "calculator_activated": route_active,
        "first_step_route_probability": float(
            singleton(first["route_probability"])
        ),
        "registered_operands": (
            decoded_operands(first) if route_active else None
        ),
        "calculator_symbols": active_calculator_symbols(
            result,
            eos_symbol=layout.eos_result,
        ),
        "implant_output": result["generated_text"],
        "untouched_base_output": untouched["generated_text"],
        "token_identical_to_base": (
            result["generated_token_ids"] == untouched["generated_token_ids"]
        ),
        "learned_interface_parameters": implant.trainable_parameter_count,
        "calculator_learned_parameters": (
            implant.calculator.trainable_parameter_count
        ),
        "checkpoint_sha256": checkpoint_hash,
        "latency_seconds": {
            "implant": implant_seconds,
            "untouched_base": base_seconds,
        },
        "scope": (
            "One addition call per response; this prototype does not support "
            "arbitrary recurrent calculator calls."
        ),
    }
    if args.show_steps:
        payload["steps"] = result["steps"]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
