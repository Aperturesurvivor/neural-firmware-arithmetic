from __future__ import annotations

import argparse
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
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
    mathematical_correct,
)

DEFAULT_CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_pilot_v2/neuron_implant_seed_12801.pt"
)
CHUNK_SIZE = 5


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def singleton(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def make_evaluation() -> list[object]:
    return (
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


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    return {
        "completed_rows": len(rows),
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
        default=Path("phase7_results/sequence_evaluation_v1.json"),
    )
    parser.add_argument("--latch-route", action="store_true")
    parser.add_argument("--preserve-base-when-off", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not checkpoint["stage"].startswith("output"):
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
    examples = make_evaluation()
    rows: list[dict[str, object]] = []
    if args.result.exists():
        existing = json.loads(args.result.read_text())
        if existing.get("status") == "complete":
            print(json.dumps(existing["summary"], indent=2), flush=True)
            return
        rows = existing.get("rows", [])
    started = time.perf_counter()
    start_index = len(rows)
    stop_index = min(len(examples), start_index + CHUNK_SIZE)
    for index in range(start_index, stop_index):
        example = examples[index]
        result = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            latch_route=args.latch_route,
            preserve_base_when_off=args.preserve_base_when_off,
        )
        ablated = generate_sequence_implant(
            bundle,
            implant,
            example.prompt,
            max_new_tokens=8,
            ablate_result=True,
            latch_route=args.latch_route,
            preserve_base_when_off=args.preserve_base_when_off,
        )
        first = result["steps"][0]
        a_digits = singleton(first["a_digits"])
        b_digits = singleton(first["b_digits"])
        a_length = singleton(first["a_lengths"])
        b_length = singleton(first["b_lengths"])
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
                singleton(first["route_active"]) is True
            ),
            "first_step_operands_exact": (
                a_digits[:a_length] == [int(character) for character in example.a]
                and b_digits[:b_length]
                == [int(character) for character in example.b]
            ),
            "any_route_active": any(
                any(step["route_active"]) for step in result["steps"]
            ),
        }
        if not example.route_label:
            untouched = generate_untouched_sequence(
                bundle,
                implant,
                example.prompt,
                layer_index=checkpoint["layer_index"],
                max_new_tokens=8,
            )
            row["untouched"] = untouched
            row["token_preserved"] = (
                result["generated_token_ids"] == untouched["generated_token_ids"]
            )
        else:
            row["token_preserved"] = False
        rows.append(row)
        payload = {
            "status": (
                "complete" if len(rows) == len(examples) else "in_progress"
            ),
            "checkpoint": str(args.checkpoint),
            "latch_route": args.latch_route,
            "preserve_base_when_off": args.preserve_base_when_off,
            "summary": summary(rows),
            "rows": rows,
            "wall_time_seconds": time.perf_counter() - started,
        }
        write_json(args.result, payload)
        print(
            f"evaluate={index + 1}/{len(examples)} "
            f"route={example.route_label} text={result['generated_text']!r}",
            flush=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(json.dumps(summary(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
