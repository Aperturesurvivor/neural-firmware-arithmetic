from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_data import (
    build_phase7_audit2_examples,
    build_phase7_audit_examples,
)
from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    generate_sequence_implant,
    generate_untouched_sequence,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    exact_format_correct,
    mathematical_correct,
)

DEFAULT_CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_interface_v2/neuron_implant_seed_12811.pt"
)
DEFAULT_RESULT_PATH = Path("phase7_results/sequence_audit_v1.json")
EXPECTED_CHECKPOINT_SHA256 = (
    "fc5a547033ebe1a8fbe9888fa5a5549c0b0592f0e9524a7628e30a2bcee41d6a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def singleton(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def calculator_symbols(result: dict[str, object]) -> list[int]:
    symbols: list[int] = []
    for step in result["steps"]:
        if not singleton(step["route_active"]):
            continue
        symbol = int(singleton(step["result_symbols"]))
        symbols.append(symbol)
        if symbol == 10:
            break
    return symbols


def positive_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "examples": len(rows),
        "mathematical_exact": sum(row["mathematical_correct"] for row in rows),
        "format_exact": sum(row["format_correct"] for row in rows),
        "ablation_mathematical_exact": sum(
            row["ablation_mathematical_correct"] for row in rows
        ),
        "first_step_route_active": sum(
            row["first_step_route_active"] for row in rows
        ),
        "first_step_operands_exact": sum(
            row["first_step_operands_exact"] for row in rows
        ),
        "calculator_trajectory_exact": sum(
            row["calculator_trajectory_exact"] for row in rows
        ),
    }


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"]]
    symbolic = [row for row in positives if row["split"].endswith("_symbolic")]
    word = [row for row in positives if row["split"].endswith("_word")]
    negatives = [row for row in rows if not row["route_label"]]
    return {
        "completed_rows": len(rows),
        "positive": positive_summary(positives),
        "symbolic": positive_summary(symbolic),
        "word": positive_summary(word),
        "negative": {
            "examples": len(negatives),
            "false_routes": sum(row["any_route_active"] for row in negatives),
            "token_preserved": sum(row["token_preserved"] for row in negatives),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT_PATH,
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--deterministic-result-step", action="store_true")
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--dataset", choices=("audit1", "audit2"), default="audit1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if args.deterministic_result_step and args.result == DEFAULT_RESULT_PATH:
        raise ValueError(
            "post-audit step-counter evaluation requires a distinct result path"
        )
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError(
            "checkpoint does not match the frozen Phase 7 audit checkpoint"
        )
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if "output" not in checkpoint["stage"]:
        raise ValueError("evaluation requires an output-trained checkpoint")
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
        use_swiglu_interface=checkpoint.get("use_swiglu_interface", False),
    )
    with torch.no_grad():
        implant.input_rows.copy_(checkpoint["input_rows"].to(bundle.device))
        if "gate_rows" in checkpoint:
            implant.gate_rows.copy_(checkpoint["gate_rows"].to(bundle.device))
        implant.result_columns.copy_(
            checkpoint["result_columns"].to(bundle.device)
        )
    examples = (
        build_phase7_audit_examples()
        if args.dataset == "audit1"
        else build_phase7_audit2_examples()
    )
    rows: list[dict[str, object]] = []
    if args.result.exists():
        existing = json.loads(args.result.read_text())
        if existing.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("existing audit record uses a different checkpoint")
        if existing.get("dataset", "audit1") != args.dataset:
            raise ValueError("existing audit record uses a different dataset")
        if existing.get("status") == "complete":
            print(json.dumps(existing["summary"], indent=2), flush=True)
            return
        rows = existing.get("rows", [])
    started = time.perf_counter()
    start_index = len(rows)
    stop_index = min(len(examples), start_index + args.chunk_size)
    for index in range(start_index, stop_index):
        example = examples[index]
        result = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            latch_route=True,
            preserve_base_when_off=True,
            deterministic_result_step=args.deterministic_result_step,
        )
        first = result["steps"][0]
        a_digits = singleton(first["a_digits"])
        b_digits = singleton(first["b_digits"])
        a_length = singleton(first["a_lengths"])
        b_length = singleton(first["b_lengths"])
        row = {
            **example.to_dict(),
            "implant": result,
            "first_step_route_active": singleton(first["route_active"]) is True,
            "first_step_operands_exact": (
                a_digits[:a_length] == [int(character) for character in example.a]
                and b_digits[:b_length]
                == [int(character) for character in example.b]
            ),
            "any_route_active": any(
                any(step["route_active"]) for step in result["steps"]
            ),
        }
        if example.route_label:
            expected_symbols = [int(character) for character in example.answer or ""]
            expected_symbols.append(layout.eos_result)
            ablated = generate_sequence_implant(
                bundle,
                implant,
                example.prompt,
                max_new_tokens=8,
                ablate_result=True,
                latch_route=True,
                preserve_base_when_off=True,
                deterministic_result_step=args.deterministic_result_step,
            )
            row.update(
                {
                    "ablated": ablated,
                    "mathematical_correct": mathematical_correct(
                        result["generated_text"],
                        example.answer or "",
                    ),
                    "format_correct": exact_format_correct(
                        result["generated_text"],
                        example.answer or "",
                    ),
                    "ablation_mathematical_correct": mathematical_correct(
                        ablated["generated_text"],
                        example.answer or "",
                    ),
                    "calculator_symbols": calculator_symbols(result),
                    "calculator_trajectory_exact": (
                        calculator_symbols(result) == expected_symbols
                    ),
                    "token_preserved": False,
                }
            )
        else:
            untouched = generate_untouched_sequence(
                bundle,
                implant,
                example.prompt,
                layer_index=checkpoint["layer_index"],
                max_new_tokens=8,
            )
            row.update(
                {
                    "untouched": untouched,
                    "mathematical_correct": False,
                    "format_correct": False,
                    "ablation_mathematical_correct": False,
                    "calculator_symbols": calculator_symbols(result),
                    "calculator_trajectory_exact": False,
                    "token_preserved": (
                        result["generated_token_ids"]
                        == untouched["generated_token_ids"]
                    ),
                }
            )
        rows.append(row)
        payload = {
            "status": "complete" if len(rows) == len(examples) else "in_progress",
            "protocol": (
                "PHASE7_AUDIT_PROTOCOL.md"
                if args.dataset == "audit1"
                else "PHASE7_AUDIT2_PROTOCOL.md"
            ),
            "dataset": args.dataset,
            "evaluation_kind": (
                "frozen_held_out_audit2"
                if args.dataset == "audit2"
                else (
                    "post_audit_step_counter_development"
                    if args.deterministic_result_step
                    else "frozen_held_out_audit"
                )
            ),
            "implementation_commit": git_commit(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "latch_route": True,
            "preserve_base_when_off": True,
            "deterministic_result_step": args.deterministic_result_step,
            "max_new_tokens": 8,
            "learned_parameters": implant.trainable_parameter_count,
            "calculator_learned_parameters": (
                implant.calculator.trainable_parameter_count
            ),
            "mlp_width": implant.mlp_width,
            "residual_width": implant.input_rows.shape[1],
            "summary": summary(rows),
            "rows": rows,
            "wall_time_seconds_this_process": time.perf_counter() - started,
        }
        write_json(args.result, payload)
        print(
            f"audit={index + 1}/{len(examples)} "
            f"split={example.split} text={result['generated_text']!r}",
            flush=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(json.dumps(summary(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
