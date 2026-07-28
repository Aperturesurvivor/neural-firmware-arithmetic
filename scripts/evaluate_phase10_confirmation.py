from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.phase10_data import (
    PHASE10_SOURCE_SEEDS,
    PHASE10_TRAINING_SEEDS,
    build_phase10_confirmatory_examples,
)
from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

CHECKPOINT_DIRECTORY = Path("phase10_artifacts/confirmatory_interfaces")
RESULT_PATH = Path("phase10_results/confirmation.json")
MANIFEST_PATH = Path("phase10_results/frozen_prompt_manifest.json")
CONDITIONS = ("linear", "nonlinear", "linear_representation")


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def generate_base(
    bundle: object,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    full_ids = chat_prompt_ids(bundle.tokenizer, prompt)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=bundle.device,
        )
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).last_hidden_state[:, -1]
        token = int(bundle.model.lm_head(hidden).argmax(dim=-1).item())
        generated.append(token)
        full_ids.append(token)
        del input_ids, hidden
        if bundle.device.type == "mps":
            torch.mps.empty_cache()
        if token == bundle.tokenizer.eos_token_id:
            break
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
    }


def _first(result: dict[str, object], key: str, default: object) -> object:
    steps = result.get("steps", [])
    if not steps:
        return default
    value = steps[0].get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _operand_value(result: dict[str, object], prefix: str) -> str | None:
    digits = _first(result, f"{prefix}_digits", [])
    length = _first(result, f"{prefix}_lengths", 0)
    if not isinstance(digits, list) or not isinstance(length, int) or length < 1:
        return None
    return "".join(str(value) for value in digits[:length])


def operands_exact(result: dict[str, object], a: str, b: str) -> bool:
    return _operand_value(result, "a") == a and _operand_value(result, "b") == b


def trajectory_exact(result: dict[str, object], answer: str) -> bool:
    symbols = [
        _first({"steps": [step]}, "result_symbols", 11)
        for step in result.get("steps", [])
    ]
    expected = [int(character) for character in answer] + [10]
    return symbols[: len(expected)] == expected


def implant_record(
    output: dict[str, object],
    *,
    row: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    positive = bool(row["route_label"])
    return {
        **output,
        "format_exact": (
            exact_format_correct(output["generated_text"], row["answer"])
            if positive
            else False
        ),
        "first_route": bool(_first(output, "route", 0)),
        "first_route_active": bool(_first(output, "route_active", False)),
        "operands_exact": (
            operands_exact(output, row["a"], row["b"]) if positive else False
        ),
        "trajectory_exact": (
            trajectory_exact(output, row["answer"]) if positive else False
        ),
        "token_preserved": (
            output["generated_token_ids"] == row["base"]["generated_token_ids"]
        ),
        "latency_seconds": elapsed,
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text())
    examples = build_phase10_confirmatory_examples()
    rows = [
        {
            "row_index": index,
            **example.to_dict(),
            "base": None,
            "conditions": {condition: {} for condition in CONDITIONS},
        }
        for index, example in enumerate(examples)
    ]
    frozen_rows = [row.to_dict() for row in examples]
    canonical = json.dumps(
        frozen_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest["canonical_rows_sha256"]:
        raise ValueError("Phase 10 generated prompts do not match frozen manifest")

    base_bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    base_latencies: list[float] = []
    print("evaluating untouched TinyLlama baseline", flush=True)
    for index, row in enumerate(rows):
        row_started = time.perf_counter()
        output = generate_base(
            base_bundle,
            row["prompt"],
            max_new_tokens=8,
        )
        elapsed = time.perf_counter() - row_started
        base_latencies.append(elapsed)
        row["base"] = {
            **output,
            "format_exact": (
                exact_format_correct(output["generated_text"], row["answer"])
                if row["route_label"]
                else False
            ),
            "latency_seconds": elapsed,
        }
        if (index + 1) % 25 == 0:
            print(f"base: evaluated {index + 1}/{len(rows)}", flush=True)
    del base_bundle
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    checkpoint_metadata: dict[str, dict[str, object]] = {
        condition: {} for condition in CONDITIONS
    }
    condition_latencies: dict[str, dict[str, list[float]]] = {
        condition: {} for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        for seed in PHASE10_TRAINING_SEEDS:
            key = str(seed)
            checkpoint_path = (
                CHECKPOINT_DIRECTORY / f"{condition}_seed_{seed}.pt"
            )
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
            latencies: list[float] = []
            for index, row in enumerate(rows):
                row_started = time.perf_counter()
                normal = generate_sequence_implant(
                    bundle,
                    implant,
                    row["prompt"],
                    max_new_tokens=8,
                    latch_route=True,
                    preserve_base_when_off=True,
                    deterministic_result_step=True,
                    latch_operands=True,
                )
                elapsed = time.perf_counter() - row_started
                latencies.append(elapsed)
                record = implant_record(normal, row=row, elapsed=elapsed)
                if row["route_label"]:
                    oracle_started = time.perf_counter()
                    oracle = generate_sequence_implant(
                        bundle,
                        implant,
                        row["prompt"],
                        max_new_tokens=8,
                        latch_route=True,
                        preserve_base_when_off=True,
                        deterministic_result_step=True,
                        latch_operands=True,
                        force_route=1,
                    )
                    record["oracle_route"] = implant_record(
                        oracle,
                        row=row,
                        elapsed=time.perf_counter() - oracle_started,
                    )
                    if condition == "linear_representation":
                        ablation_started = time.perf_counter()
                        ablated = generate_sequence_implant(
                            bundle,
                            implant,
                            row["prompt"],
                            max_new_tokens=8,
                            ablate_result=True,
                            latch_route=True,
                            preserve_base_when_off=True,
                            deterministic_result_step=True,
                            latch_operands=True,
                        )
                        record["ablation"] = implant_record(
                            ablated,
                            row=row,
                            elapsed=time.perf_counter() - ablation_started,
                        )
                row["conditions"][condition][key] = record
                if (index + 1) % 25 == 0:
                    print(
                        f"{condition} seed {seed}: "
                        f"evaluated {index + 1}/{len(rows)}",
                        flush=True,
                    )
            condition_latencies[condition][key] = latencies
            checkpoint_metadata[condition][key] = {
                "source_phase8_seed": PHASE10_SOURCE_SEEDS[seed],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "interface_kind": implant.interface_kind,
                "representation_rank": implant.representation_rank,
                "adapt_base_mlp": implant.adapt_base_mlp,
                "route_threshold": implant.route_threshold,
                "route_temperature": implant.route_temperature,
                "digit_threshold": implant.digit_threshold,
                "architectural_learned_parameters": (
                    architectural_learned_parameter_count(implant)
                ),
                "calculator_learned_parameters": (
                    implant.calculator.trainable_parameter_count
                ),
            }
            del bundle, implant
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    metrics: dict[str, dict[str, object]] = {}
    categories: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        metrics[condition] = {}
        categories[condition] = {}
        for seed in PHASE10_TRAINING_SEEDS:
            key = str(seed)
            positive_records = [
                row["conditions"][condition][key] for row in positives
            ]
            negative_records = [
                row["conditions"][condition][key] for row in negatives
            ]
            conditional = [
                record
                for record in positive_records
                if record["first_route_active"] and record["operands_exact"]
            ]
            seed_metrics = {
                "exact": sum(record["format_exact"] for record in positive_records),
                "positive_routes": sum(
                    record["first_route"] for record in positive_records
                ),
                "positive_active_routes": sum(
                    record["first_route_active"] for record in positive_records
                ),
                "operands_exact": sum(
                    record["operands_exact"] for record in positive_records
                ),
                "trajectories_exact": sum(
                    record["trajectory_exact"] for record in positive_records
                ),
                "conditional_examples": len(conditional),
                "conditional_exact": sum(
                    record["format_exact"] for record in conditional
                ),
                "conditional_trajectories_exact": sum(
                    record["trajectory_exact"] for record in conditional
                ),
                "false_routes": sum(
                    record["first_route"] for record in negative_records
                ),
                "token_preserved": sum(
                    record["token_preserved"] for record in negative_records
                ),
                "oracle_exact": sum(
                    record["oracle_route"]["format_exact"]
                    for record in positive_records
                ),
                "oracle_operands_exact": sum(
                    record["oracle_route"]["operands_exact"]
                    for record in positive_records
                ),
                "oracle_trajectories_exact": sum(
                    record["oracle_route"]["trajectory_exact"]
                    for record in positive_records
                ),
            }
            if condition == "linear_representation":
                seed_metrics["ablation_exact"] = sum(
                    record["ablation"]["format_exact"]
                    for record in positive_records
                )
                seed_metrics["paired_causal_losses"] = sum(
                    record["format_exact"]
                    and not record["ablation"]["format_exact"]
                    for record in positive_records
                )
            metrics[condition][key] = seed_metrics
            seed_categories: dict[str, dict[str, int]] = {}
            for row in rows:
                category = row["split"].removeprefix(
                    "phase10_confirmatory_positive_"
                ).removeprefix("phase10_confirmatory_negative_")
                record = row["conditions"][condition][key]
                aggregate = seed_categories.setdefault(
                    category,
                    {"examples": 0, "exact": 0, "false_routes": 0},
                )
                aggregate["examples"] += 1
                aggregate["exact"] += int(record["format_exact"])
                if not row["route_label"]:
                    aggregate["false_routes"] += int(record["first_route"])
            categories[condition][key] = seed_categories

    paired_representation_gains = [
        metrics["linear_representation"][str(seed)]["exact"]
        - metrics["linear"][str(seed)]["exact"]
        for seed in PHASE10_TRAINING_SEEDS
    ]
    paired_nonlinear_gains = [
        metrics["nonlinear"][str(seed)]["exact"]
        - metrics["linear"][str(seed)]["exact"]
        for seed in PHASE10_TRAINING_SEEDS
    ]
    representation_gates = {
        "paired_end_to_end_benefit": (
            all(gain > 0 for gain in paired_representation_gains)
            and statistics.fmean(paired_representation_gains) >= 5
        ),
        "preservation": all(
            metrics["linear_representation"][str(seed)]["false_routes"] <= 4
            and metrics["linear_representation"][str(seed)]["token_preserved"]
            >= 196
            for seed in PHASE10_TRAINING_SEEDS
        ),
        "operand_access": all(
            metrics["linear_representation"][str(seed)]["oracle_exact"] >= 85
            and metrics["linear_representation"][str(seed)]["oracle_exact"]
            >= metrics["linear"][str(seed)]["oracle_exact"]
            for seed in PHASE10_TRAINING_SEEDS
        ),
        "conditional_mechanism": all(
            metrics["linear_representation"][str(seed)]["conditional_exact"]
            == metrics["linear_representation"][str(seed)]["conditional_examples"]
            and metrics["linear_representation"][str(seed)][
                "conditional_trajectories_exact"
            ]
            == metrics["linear_representation"][str(seed)]["conditional_examples"]
            for seed in PHASE10_TRAINING_SEEDS
        ),
        "causal_ablation": all(
            metrics["linear_representation"][str(seed)]["paired_causal_losses"]
            == metrics["linear_representation"][str(seed)]["exact"]
            and metrics["linear_representation"][str(seed)]["ablation_exact"] <= 5
            for seed in PHASE10_TRAINING_SEEDS
        ),
    }
    representation_gates["all_gates"] = all(representation_gates.values())
    nonlinear_gate = (
        all(gain > 0 for gain in paired_nonlinear_gains)
        and statistics.fmean(paired_nonlinear_gains) > 0
        and all(
            metrics["nonlinear"][str(seed)]["false_routes"]
            <= metrics["linear"][str(seed)]["false_routes"]
            for seed in PHASE10_TRAINING_SEEDS
        )
    )
    payload = {
        "status": "phase10_confirmatory_evaluation_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "unique_prompts": len(rows),
        "positive_prompts": len(positives),
        "negative_prompts": len(negatives),
        "phase10_seeds": list(PHASE10_TRAINING_SEEDS),
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "base_exact": sum(row["base"]["format_exact"] for row in positives),
        "conditions": metrics,
        "paired_representation_gains": paired_representation_gains,
        "paired_nonlinear_gains": paired_nonlinear_gains,
        "representation_gates": representation_gates,
        "nonlinear_gate": nonlinear_gate,
        "categories": categories,
        "latency": {
            "base": latency_summary(base_latencies),
            "conditions": {
                condition: {
                    seed: latency_summary(values)
                    for seed, values in per_seed.items()
                }
                for condition, per_seed in condition_latencies.items()
            },
        },
        "checkpoints": checkpoint_metadata,
        "frozen_manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": sha256(MANIFEST_PATH),
            "canonical_rows_sha256": manifest["canonical_rows_sha256"],
        },
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "conditions": metrics,
                "paired_representation_gains": paired_representation_gains,
                "paired_nonlinear_gains": paired_nonlinear_gains,
                "representation_gates": representation_gates,
                "nonlinear_gate": nonlinear_gate,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
