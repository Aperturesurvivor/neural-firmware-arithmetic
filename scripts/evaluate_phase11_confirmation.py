from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase11_data import (
    PHASE11_SOURCE_SEEDS,
    build_phase11_confirmatory_examples,
)
from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

MANIFEST_PATH = Path("phase11_results/frozen_prompt_manifest.json")
RESULT_PATH = Path("phase11_results/confirmation.json")


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


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


def result_symbols(result: dict[str, object]) -> list[int]:
    return [
        int(_first({"steps": [step]}, "result_symbols", 11))
        for step in result.get("steps", [])
    ]


def trajectory_exact(result: dict[str, object], answer: str) -> bool:
    expected = [int(character) for character in answer] + [10]
    return result_symbols(result)[: len(expected)] == expected


def implant_record(
    output: dict[str, object],
    *,
    row: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    positive = bool(row["route_label"])
    return {
        "generated_token_ids": output["generated_token_ids"],
        "generated_text": output["generated_text"],
        "result_symbols": result_symbols(output),
        "first_route": bool(_first(output, "route", 0)),
        "first_route_active": bool(_first(output, "route_active", False)),
        "first_route_probability": float(
            _first(output, "route_probability", float("nan"))
        ),
        "predicted_a": _operand_value(output, "a"),
        "predicted_b": _operand_value(output, "b"),
        "format_exact": (
            exact_format_correct(output["generated_text"], row["answer"])
            if positive
            else False
        ),
        "operands_exact": (
            operands_exact(output, row["a"], row["b"]) if positive else False
        ),
        "trajectory_exact": (
            trajectory_exact(output, row["answer"]) if positive else False
        ),
        "token_preserved": (
            output["generated_token_ids"]
            == row["base"]["generated_token_ids"]
        ),
        "latency_seconds": elapsed,
    }


def tensor_inheritance(
    source: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    tensor_keys = sorted(
        key for key, value in source.items() if isinstance(value, torch.Tensor)
    )
    comparisons = {
        key: (
            key in candidate
            and isinstance(candidate[key], torch.Tensor)
            and torch.equal(value, candidate[key])
        )
        for key, value in source.items()
        if isinstance(value, torch.Tensor)
    }
    return {
        "tensor_keys": tensor_keys,
        "comparisons": comparisons,
        "all_inherited_tensors_bit_identical": all(comparisons.values()),
    }


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text())
    examples = build_phase11_confirmatory_examples()
    frozen_rows = [example.to_dict() for example in examples]
    canonical = json.dumps(
        frozen_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest["canonical_rows_sha256"]:
        raise ValueError("Phase 11 generated prompts do not match frozen manifest")
    rows = [
        {
            "row_index": index,
            **example.to_dict(),
            "base": None,
            "conditions": {"phase10_control": {}, "phase11_candidate": {}},
        }
        for index, example in enumerate(examples)
    ]

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

    checkpoint_metadata: dict[str, dict[str, object]] = {}
    latencies: dict[str, dict[str, dict[str, object]]] = {
        "phase10_control": {},
        "phase11_candidate": {},
    }
    for seed in PHASE11_SOURCE_SEEDS:
        key = str(seed)
        manifest_checkpoint = manifest["candidate_checkpoints"][key]
        source_path = Path(manifest_checkpoint["source_phase10_checkpoint"])
        candidate_path = Path(manifest_checkpoint["path"])
        if sha256(source_path) != manifest_checkpoint[
            "source_phase10_checkpoint_sha256"
        ]:
            raise ValueError(f"source checkpoint hash mismatch for seed {seed}")
        if sha256(candidate_path) != manifest_checkpoint["sha256"]:
            raise ValueError(f"candidate checkpoint hash mismatch for seed {seed}")
        source_checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        candidate_checkpoint = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=True,
        )
        inheritance = tensor_inheritance(source_checkpoint, candidate_checkpoint)

        control_bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        control_implant = install_checkpoint_implant(
            control_bundle,
            source_checkpoint,
        )
        control_latency: list[float] = []
        for index, row in enumerate(rows):
            row_started = time.perf_counter()
            output = generate_sequence_implant(
                control_bundle,
                control_implant,
                row["prompt"],
                max_new_tokens=8,
                latch_route=True,
                preserve_base_when_off=True,
                deterministic_result_step=True,
                latch_operands=True,
            )
            elapsed = time.perf_counter() - row_started
            control_latency.append(elapsed)
            row["conditions"]["phase10_control"][key] = implant_record(
                output,
                row=row,
                elapsed=elapsed,
            )
            if (index + 1) % 25 == 0:
                print(
                    f"phase10 control seed {seed}: "
                    f"evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
        control_parameter_count = architectural_learned_parameter_count(
            control_implant
        )
        latencies["phase10_control"][key] = latency_summary(control_latency)
        del control_bundle, control_implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        candidate_bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        candidate_implant = install_checkpoint_implant(
            candidate_bundle,
            candidate_checkpoint,
        )
        candidate_latency: list[float] = []
        oracle_latency: list[float] = []
        route_off_latency: list[float] = []
        for index, row in enumerate(rows):
            row_started = time.perf_counter()
            output = generate_sequence_implant(
                candidate_bundle,
                candidate_implant,
                row["prompt"],
                max_new_tokens=8,
                latch_route=True,
                preserve_base_when_off=True,
                deterministic_result_step=True,
                latch_operands=True,
            )
            elapsed = time.perf_counter() - row_started
            candidate_latency.append(elapsed)
            record = implant_record(output, row=row, elapsed=elapsed)
            if row["route_label"]:
                oracle_started = time.perf_counter()
                oracle = generate_sequence_implant(
                    candidate_bundle,
                    candidate_implant,
                    row["prompt"],
                    max_new_tokens=8,
                    latch_route=True,
                    preserve_base_when_off=True,
                    deterministic_result_step=True,
                    latch_operands=True,
                    force_route=1,
                )
                oracle_elapsed = time.perf_counter() - oracle_started
                oracle_latency.append(oracle_elapsed)
                record["oracle_route"] = implant_record(
                    oracle,
                    row=row,
                    elapsed=oracle_elapsed,
                )
                route_off_started = time.perf_counter()
                route_off = generate_sequence_implant(
                    candidate_bundle,
                    candidate_implant,
                    row["prompt"],
                    max_new_tokens=8,
                    latch_route=True,
                    preserve_base_when_off=True,
                    deterministic_result_step=True,
                    latch_operands=True,
                    force_route=0,
                )
                route_off_elapsed = time.perf_counter() - route_off_started
                route_off_latency.append(route_off_elapsed)
                record["route_off"] = implant_record(
                    route_off,
                    row=row,
                    elapsed=route_off_elapsed,
                )
            row["conditions"]["phase11_candidate"][key] = record
            if (index + 1) % 25 == 0:
                print(
                    f"phase11 candidate seed {seed}: "
                    f"evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
        candidate_parameter_count = architectural_learned_parameter_count(
            candidate_implant
        )
        latencies["phase11_candidate"][key] = {
            "normal": latency_summary(candidate_latency),
            "oracle_route": latency_summary(oracle_latency),
            "route_off": latency_summary(route_off_latency),
        }
        checkpoint_metadata[key] = {
            "source": {
                "path": str(source_path),
                "sha256": manifest_checkpoint[
                    "source_phase10_checkpoint_sha256"
                ],
                "architectural_learned_parameters": control_parameter_count,
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": manifest_checkpoint["sha256"],
                "request_router_kind": candidate_implant.request_router_kind,
                "request_route_threshold": (
                    candidate_implant.request_route_threshold
                ),
                "request_route_temperature": (
                    candidate_implant.request_route_temperature
                ),
                "request_router_parameters": int(
                    candidate_implant.request_route_rows.numel()
                ),
                "architectural_learned_parameters": (
                    candidate_parameter_count
                ),
                "architectural_parameter_delta": (
                    candidate_parameter_count - control_parameter_count
                ),
            },
            "inheritance": inheritance,
        }
        del candidate_bundle, candidate_implant
        del source_checkpoint, candidate_checkpoint
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    metrics: dict[str, dict[str, dict[str, object]]] = {
        "phase10_control": {},
        "phase11_candidate": {},
    }
    categories: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        "phase10_control": {},
        "phase11_candidate": {},
    }
    for condition in ("phase10_control", "phase11_candidate"):
        for seed in PHASE11_SOURCE_SEEDS:
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
            seed_metrics: dict[str, object] = {
                "exact": sum(
                    record["format_exact"] for record in positive_records
                ),
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
            }
            if condition == "phase11_candidate":
                seed_metrics.update(
                    {
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
                        "route_off_exact": sum(
                            record["route_off"]["format_exact"]
                            for record in positive_records
                        ),
                        "paired_route_off_losses": sum(
                            record["format_exact"]
                            and not record["route_off"]["format_exact"]
                            for record in positive_records
                        ),
                    }
                )
            metrics[condition][key] = seed_metrics
            seed_categories: dict[str, dict[str, int]] = {}
            for row in rows:
                category = row["split"].removeprefix(
                    "phase11_confirmatory_positive_"
                ).removeprefix("phase11_confirmatory_negative_")
                record = row["conditions"][condition][key]
                aggregate = seed_categories.setdefault(
                    category,
                    {
                        "examples": 0,
                        "routes": 0,
                        "exact": 0,
                        "false_routes": 0,
                    },
                )
                aggregate["examples"] += 1
                aggregate["routes"] += int(record["first_route"])
                aggregate["exact"] += int(record["format_exact"])
                if not row["route_label"]:
                    aggregate["false_routes"] += int(record["first_route"])
            categories[condition][key] = seed_categories

    paired_gains = [
        metrics["phase11_candidate"][str(seed)]["exact"]
        - metrics["phase10_control"][str(seed)]["exact"]
        for seed in PHASE11_SOURCE_SEEDS
    ]
    candidate_exact = [
        metrics["phase11_candidate"][str(seed)]["exact"]
        for seed in PHASE11_SOURCE_SEEDS
    ]
    gates = {
        "autonomous_exactness": (
            min(candidate_exact) >= 70
            and statistics.fmean(candidate_exact) >= 75
        ),
        "paired_improvement": (
            all(gain > 0 for gain in paired_gains)
            and statistics.fmean(paired_gains) >= 20
        ),
        "route_recognition": all(
            metrics["phase11_candidate"][str(seed)]["positive_routes"] >= 80
            for seed in PHASE11_SOURCE_SEEDS
        ),
        "preservation": all(
            metrics["phase11_candidate"][str(seed)]["false_routes"] <= 4
            and metrics["phase11_candidate"][str(seed)]["token_preserved"]
            >= 196
            for seed in PHASE11_SOURCE_SEEDS
        ),
        "operand_access": all(
            metrics["phase11_candidate"][str(seed)]["oracle_exact"] >= 85
            for seed in PHASE11_SOURCE_SEEDS
        ),
        "conditional_mechanism": all(
            metrics["phase11_candidate"][str(seed)]["conditional_exact"]
            == metrics["phase11_candidate"][str(seed)]["conditional_examples"]
            and metrics["phase11_candidate"][str(seed)][
                "conditional_trajectories_exact"
            ]
            == metrics["phase11_candidate"][str(seed)]["conditional_examples"]
            for seed in PHASE11_SOURCE_SEEDS
        ),
        "causal_routing": all(
            metrics["phase11_candidate"][str(seed)]["paired_route_off_losses"]
            == metrics["phase11_candidate"][str(seed)]["exact"]
            and metrics["phase11_candidate"][str(seed)]["route_off_exact"] <= 5
            for seed in PHASE11_SOURCE_SEEDS
        ),
        "checkpoint_integrity": all(
            checkpoint_metadata[str(seed)]["inheritance"][
                "all_inherited_tensors_bit_identical"
            ]
            and checkpoint_metadata[str(seed)]["candidate"][
                "request_router_parameters"
            ]
            == 4_096
            and checkpoint_metadata[str(seed)]["candidate"][
                "architectural_parameter_delta"
            ]
            == 4_096
            for seed in PHASE11_SOURCE_SEEDS
        ),
    }
    gates["all_gates"] = all(gates.values())
    payload = {
        "status": "phase11_confirmatory_evaluation_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "source_seeds": list(PHASE11_SOURCE_SEEDS),
        "unique_prompts": len(rows),
        "positive_prompts": len(positives),
        "negative_prompts": len(negatives),
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "base_exact": sum(row["base"]["format_exact"] for row in positives),
        "conditions": metrics,
        "paired_gains": paired_gains,
        "gates": gates,
        "categories": categories,
        "latency": {
            "base": latency_summary(base_latencies),
            "conditions": latencies,
        },
        "checkpoints": checkpoint_metadata,
        "frozen_manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": sha256(MANIFEST_PATH),
            "canonical_rows_sha256": manifest["canonical_rows_sha256"],
            "implementation_commit": manifest["implementation_commit"],
        },
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "base_exact": payload["base_exact"],
                "conditions": metrics,
                "paired_gains": paired_gains,
                "gates": gates,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
