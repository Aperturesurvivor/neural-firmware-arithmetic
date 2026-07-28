from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_data import build_phase9_confirmatory_examples
from neural_firmware.phase9_training import install_checkpoint_implant
from neural_firmware.phase10_training import PHASE10_CONDITIONS
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

CHECKPOINT_DIRECTORY = Path("phase10_artifacts/development")
PHASE9_RESULT = Path("phase9_results/confirmation.json")
RESULT_PATH = Path("phase10_results/development_evaluation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def evaluated_row(
    output: dict[str, object],
    *,
    example: object,
    base_token_ids: list[int],
    elapsed: float,
) -> dict[str, object]:
    positive = bool(example.route_label)
    return {
        "generated_text": output["generated_text"],
        "generated_token_ids": output["generated_token_ids"],
        "exact": (
            exact_format_correct(
                output["generated_text"],
                example.answer or "",
            )
            if positive
            else False
        ),
        "route": bool(_first(output, "route", 0)),
        "route_active": bool(_first(output, "route_active", False)),
        "operands_exact": (
            _operand_value(output, "a") == example.a
            and _operand_value(output, "b") == example.b
            if positive
            else False
        ),
        "token_preserved": output["generated_token_ids"] == base_token_ids,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    started = time.perf_counter()
    examples = build_phase9_confirmatory_examples()
    phase9 = json.loads(PHASE9_RESULT.read_text())
    base_by_prompt = {
        row["prompt"]: row["base"]["generated_token_ids"]
        for row in phase9["rows"]
    }
    if {example.prompt for example in examples} != set(base_by_prompt):
        raise ValueError("Phase 9 development prompts do not match prior base rows")
    condition_records: dict[str, object] = {}
    for condition in PHASE10_CONDITIONS:
        condition_started = time.perf_counter()
        checkpoint_path = CHECKPOINT_DIRECTORY / f"{condition.name}.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        implant = install_checkpoint_implant(bundle, checkpoint)
        rows: list[dict[str, object]] = []
        for index, example in enumerate(examples):
            generation_started = time.perf_counter()
            output = generate_sequence_implant(
                bundle,
                implant,
                example.prompt,
                max_new_tokens=8,
                latch_route=True,
                preserve_base_when_off=True,
                deterministic_result_step=True,
                latch_operands=True,
            )
            row = {
                "row_index": index,
                **example.to_dict(),
                "normal": evaluated_row(
                    output,
                    example=example,
                    base_token_ids=base_by_prompt[example.prompt],
                    elapsed=time.perf_counter() - generation_started,
                ),
            }
            if example.route_label:
                oracle_started = time.perf_counter()
                oracle = generate_sequence_implant(
                    bundle,
                    implant,
                    example.prompt,
                    max_new_tokens=8,
                    latch_route=True,
                    preserve_base_when_off=True,
                    deterministic_result_step=True,
                    latch_operands=True,
                    force_route=1,
                )
                row["oracle_route"] = evaluated_row(
                    oracle,
                    example=example,
                    base_token_ids=base_by_prompt[example.prompt],
                    elapsed=time.perf_counter() - oracle_started,
                )
            rows.append(row)
            if (index + 1) % 25 == 0:
                print(
                    f"{condition.name}: evaluated {index + 1}/{len(examples)}",
                    flush=True,
                )
        positives = [row for row in rows if row["route_label"]]
        negatives = [row for row in rows if not row["route_label"]]
        summary = {
            "positive_examples": len(positives),
            "negative_examples": len(negatives),
            "exact": sum(row["normal"]["exact"] for row in positives),
            "routes": sum(row["normal"]["route"] for row in positives),
            "active_routes": sum(
                row["normal"]["route_active"] for row in positives
            ),
            "operands_exact": sum(
                row["normal"]["operands_exact"] for row in positives
            ),
            "false_routes": sum(
                row["normal"]["route"] for row in negatives
            ),
            "token_preserved": sum(
                row["normal"]["token_preserved"] for row in negatives
            ),
            "oracle_exact": sum(
                row["oracle_route"]["exact"] for row in positives
            ),
            "oracle_active_routes": sum(
                row["oracle_route"]["route_active"] for row in positives
            ),
            "oracle_operands_exact": sum(
                row["oracle_route"]["operands_exact"] for row in positives
            ),
        }
        condition_records[condition.name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "interface_kind": checkpoint.get("interface_kind", "linear"),
            "representation_rank": checkpoint.get("representation_rank", 0),
            "summary": summary,
            "rows": rows,
            "wall_time_seconds": time.perf_counter() - condition_started,
        }
        print(json.dumps({condition.name: summary}, indent=2), flush=True)
        del bundle, implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    payload = {
        "status": "phase10_architecture_development_evaluation_complete",
        "development_source": "previously_disclosed_phase9_confirmation",
        "conditions": condition_records,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
