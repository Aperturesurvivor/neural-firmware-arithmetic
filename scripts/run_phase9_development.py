from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    SequenceFeatureSet,
    generate_sequence_implant,
    generate_untouched_sequence,
)
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_data import (
    PHASE9_DEVELOPMENT_SEED,
    build_phase9_development,
)
from neural_firmware.phase9_training import continue_phase9_interface
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

PHASE8_SEQUENCE_CACHE = Path("phase8_artifacts/cache/layer15_full_features.pt")
PHASE8_ROUTE_CACHE = Path("phase8_artifacts/cache/first_step_route_features.pt")
PHASE9_FEATURE_CACHE = Path("phase9_artifacts/cache/interface_features.pt")
SOURCE_CHECKPOINT = Path(
    "phase8_artifacts/confirmatory_implants/implant_seed_14201.pt"
)
OUTPUT_DIRECTORY = Path("phase9_artifacts/development")
RESULT_PATH = Path("phase9_results/development.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_features() -> dict[str, object]:
    phase8_sequence = torch.load(
        PHASE8_SEQUENCE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase8_route = torch.load(
        PHASE8_ROUTE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    phase9 = torch.load(
        PHASE9_FEATURE_CACHE,
        map_location="cpu",
        weights_only=True,
    )
    return {
        "original_sequence": SequenceFeatureSet.load_state_dict(
            phase8_sequence["training"]
        ),
        "original_route": FirstStepRouteFeatureSet.load_state_dict(
            phase8_route["training"]
        ),
        "development_sequence": SequenceFeatureSet.load_state_dict(
            phase9["development"]["sequence"]
        ),
        "development_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["development"]["route"]
        ),
        "generic_sequence": SequenceFeatureSet.load_state_dict(
            phase9["generic"]["sequence"]
        ),
        "generic_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["generic"]["route"]
        ),
        "hard_sequence": SequenceFeatureSet.load_state_dict(
            phase9["hard"]["sequence"]
        ),
        "hard_route": FirstStepRouteFeatureSet.load_state_dict(
            phase9["hard"]["route"]
        ),
    }


def _balanced_subset(
    examples: list[object],
    *,
    per_family: int,
) -> list[object]:
    counts: dict[str, int] = {}
    result: list[object] = []
    for example in examples:
        count = counts.get(example.family, 0)
        if count >= per_family:
            continue
        result.append(example)
        counts[example.family] = count + 1
    return result


def _first(record: dict[str, object], key: str, default: object) -> object:
    steps = record.get("steps", [])
    if not steps:
        return default
    value = steps[0].get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _operand_value(record: dict[str, object], prefix: str) -> str | None:
    digits = _first(record, f"{prefix}_digits", [])
    length = _first(record, f"{prefix}_lengths", 0)
    if not isinstance(digits, list) or not isinstance(length, int) or length < 1:
        return None
    return "".join(str(value) for value in digits[:length])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-hard",
        action="store_true",
        help="reuse the prior generic record and rerun only the hard condition",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    features = _load_features()
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=True)
    development = build_phase9_development()
    positives = _balanced_subset(
        [example for example in development if example.route_label],
        per_family=5,
    )
    negatives = _balanced_subset(
        [example for example in development if not example.route_label],
        per_family=3,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.only_hard:
        previous = json.loads(RESULT_PATH.read_text())
        condition_records = {"generic": previous["conditions"]["generic"]}
        conditions = ("hard",)
    else:
        condition_records = {}
        conditions = ("generic", "hard")
    for condition in conditions:
        condition_started = time.perf_counter()
        bundle = load_model_bundle(PHASE8_MODEL_ID, revision=PHASE8_MODEL_REVISION)
        implant, training = continue_phase9_interface(
            bundle,
            source,
            original_sequence=features["original_sequence"],
            original_route=features["original_route"],
            condition_sequence=features[f"{condition}_sequence"],
            condition_route=features[f"{condition}_route"],
            development_sequence=features["development_sequence"],
            development_route=features["development_route"],
            seed=PHASE9_DEVELOPMENT_SEED,
            interface_steps=1_500,
            interface_learning_rate=0.0005,
            role_loss_weight=1.0,
            digit_loss_weight=1.0,
            route_steps=2_500,
            route_learning_rate=0.0005,
            maximum_development_false_positive_rate=0.01,
        )
        implant.digit_threshold = 0.8
        checkpoint_path = OUTPUT_DIRECTORY / f"{condition}_seed_15199.pt"
        torch.save(
            {
                **source,
                "stage": "phase9_development_interface",
                "condition": condition,
                "seed": PHASE9_DEVELOPMENT_SEED,
                "source_checkpoint": str(SOURCE_CHECKPOINT),
                "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
                "route_threshold": implant.route_threshold,
                "digit_threshold": implant.digit_threshold,
                "input_rows": implant.input_rows.detach().cpu(),
                "result_columns": implant.result_columns.detach().cpu(),
            },
            checkpoint_path,
        )
        positive_rows: list[dict[str, object]] = []
        for example in positives:
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
            positive_rows.append(
                {
                    **example.to_dict(),
                    "generated_text": normal["generated_text"],
                    "exact": exact_format_correct(
                        normal["generated_text"],
                        example.answer or "",
                    ),
                    "route": bool(_first(normal, "route", 0)),
                    "route_active": bool(_first(normal, "route_active", False)),
                    "operands_exact": (
                        _operand_value(normal, "a") == example.a
                        and _operand_value(normal, "b") == example.b
                    ),
                    "ablation_exact": exact_format_correct(
                        ablated["generated_text"],
                        example.answer or "",
                    ),
                }
            )
        negative_rows: list[dict[str, object]] = []
        for example in negatives:
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
                layer_index=int(source["layer_index"]),
                max_new_tokens=8,
            )
            negative_rows.append(
                {
                    **example.to_dict(),
                    "false_route": bool(_first(normal, "route", 0)),
                    "token_preserved": (
                        normal["generated_token_ids"] == base["generated_token_ids"]
                    ),
                }
            )
        summary = {
            "positive_examples": len(positive_rows),
            "negative_examples": len(negative_rows),
            "exact": sum(row["exact"] for row in positive_rows),
            "routes": sum(row["route"] for row in positive_rows),
            "active_routes": sum(row["route_active"] for row in positive_rows),
            "operands_exact": sum(
                row["operands_exact"] for row in positive_rows
            ),
            "ablation_exact": sum(
                row["ablation_exact"] for row in positive_rows
            ),
            "false_routes": sum(row["false_route"] for row in negative_rows),
            "token_preserved": sum(
                row["token_preserved"] for row in negative_rows
            ),
        }
        condition_records[condition] = {
            "training": training,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "summary": summary,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "wall_time_seconds": time.perf_counter() - condition_started,
        }
        print(json.dumps({condition: summary}, indent=2), flush=True)
        del bundle, implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    payload = {
        "status": "phase9_development_complete",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
        "development_seed": PHASE9_DEVELOPMENT_SEED,
        "configuration": {
            "interface_steps": 1_500,
            "interface_learning_rate": 0.0005,
            "role_loss_weight": 1.0,
            "digit_loss_weight": 1.0,
            "route_steps": 2_500,
            "route_learning_rate": 0.0005,
            "maximum_development_false_positive_rate": 0.01,
        },
        "conditions": condition_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
