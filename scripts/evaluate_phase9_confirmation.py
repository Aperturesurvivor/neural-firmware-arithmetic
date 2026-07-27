from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_adapter import install_matched_residual_adapter
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase8_training import generate_matched_adapter
from neural_firmware.phase9_data import (
    PHASE9_SOURCE_SEEDS,
    PHASE9_TRAINING_SEEDS,
    build_phase9_confirmatory_examples,
)
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    exact_format_correct,
    mathematical_correct,
)

PHASE8_IMPLANT_DIRECTORY = Path("phase8_artifacts/confirmatory_implants")
PHASE8_ADAPTER_DIRECTORY = Path(
    "phase8_artifacts/confirmatory_matched_adapters"
)
PHASE9_INTERFACE_DIRECTORY = Path("phase9_artifacts/confirmatory_interfaces")
RESULT_PATH = Path("phase9_results/confirmation.json")
CSV_PATH = Path("phase9_results/confirmation_rows.csv")


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mps_allocated() -> int | None:
    if not torch.backends.mps.is_available():
        return None
    return int(torch.mps.current_allocated_memory())


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


def timed(callable_value: object) -> tuple[object, float, int | None]:
    started = time.perf_counter()
    value = callable_value()
    return value, time.perf_counter() - started, mps_allocated()


def _first(result: dict[str, object], key: str, default: object) -> object:
    steps = result.get("steps", [])
    if not steps:
        return default
    value = steps[0].get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def first_route(result: dict[str, object]) -> bool:
    return bool(_first(result, "route", 0))


def first_route_active(result: dict[str, object]) -> bool:
    return bool(_first(result, "route_active", False))


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


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


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
        "mathematical_exact": (
            mathematical_correct(output["generated_text"], row["answer"])
            if positive
            else False
        ),
        "first_route": first_route(output),
        "first_route_active": first_route_active(output),
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


def main() -> None:
    started = time.perf_counter()
    examples = build_phase9_confirmatory_examples()
    print(
        f"starting sealed Phase 9 evaluation on {len(examples)} prompts",
        flush=True,
    )
    rows = [
        {
            "row_index": index,
            **example.to_dict(),
            "base": None,
            "matched_adapters": {},
            "implants": {
                "phase8_frozen": {},
                "generic": {},
                "hard": {},
            },
        }
        for index, example in enumerate(examples)
    ]
    peak_observed = mps_allocated() or 0
    base_bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    base_parameter_count = sum(
        parameter.numel() for parameter in base_bundle.model.parameters()
    )
    base_latencies: list[float] = []
    print("evaluating untouched TinyLlama baseline", flush=True)
    for row in rows:
        output, elapsed, memory = timed(
            lambda row=row: generate_base(
                base_bundle,
                row["prompt"],
                max_new_tokens=8,
            )
        )
        base_latencies.append(elapsed)
        peak_observed = max(peak_observed, memory or 0)
        row["base"] = {
            **output,
            "format_exact": (
                exact_format_correct(output["generated_text"], row["answer"])
                if row["route_label"]
                else False
            ),
            "mathematical_exact": (
                mathematical_correct(output["generated_text"], row["answer"])
                if row["route_label"]
                else False
            ),
            "latency_seconds": elapsed,
        }
    print("completed untouched TinyLlama baseline", flush=True)
    del base_bundle
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    adapter_metadata: dict[str, object] = {}
    adapter_latencies: dict[str, list[float]] = {}
    for phase9_seed in PHASE9_TRAINING_SEEDS:
        print(
            f"evaluating matched adapter for Phase 9 seed {phase9_seed}",
            flush=True,
        )
        source_seed = PHASE9_SOURCE_SEEDS[phase9_seed]
        checkpoint_path = (
            PHASE8_ADAPTER_DIRECTORY / f"adapter_seed_{source_seed}.pt"
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
        adapter = install_matched_residual_adapter(
            bundle.model,
            layer_index=checkpoint["layer_index"],
            learned_parameter_count=checkpoint["learned_parameters"],
        )
        with torch.no_grad():
            adapter.down.weight.copy_(checkpoint["down_weight"].to(bundle.device))
            adapter.up.weight.copy_(checkpoint["up_weight"].to(bundle.device))
        latencies: list[float] = []
        for row in rows:
            output, elapsed, memory = timed(
                lambda row=row: generate_matched_adapter(
                    bundle,
                    adapter,
                    row["prompt"],
                    max_new_tokens=8,
                    enabled=True,
                )
            )
            latencies.append(elapsed)
            peak_observed = max(peak_observed, memory or 0)
            row["matched_adapters"][str(phase9_seed)] = {
                **output,
                "format_exact": (
                    exact_format_correct(
                        output["generated_text"],
                        row["answer"],
                    )
                    if row["route_label"]
                    else False
                ),
                "token_preserved": (
                    output["generated_token_ids"]
                    == row["base"]["generated_token_ids"]
                ),
                "latency_seconds": elapsed,
            }
        adapter_latencies[str(phase9_seed)] = latencies
        adapter_metadata[str(phase9_seed)] = {
            "source_phase8_seed": source_seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "learned_parameters": checkpoint["learned_parameters"],
        }
        print(
            f"completed matched adapter for Phase 9 seed {phase9_seed}",
            flush=True,
        )
        del bundle, adapter
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    implant_metadata: dict[str, dict[str, object]] = {
        condition: {} for condition in ("phase8_frozen", "generic", "hard")
    }
    implant_latencies: dict[str, dict[str, list[float]]] = {
        condition: {} for condition in ("phase8_frozen", "generic", "hard")
    }
    ablation_latencies: dict[str, list[float]] = {}
    for condition in ("phase8_frozen", "generic", "hard"):
        for phase9_seed in PHASE9_TRAINING_SEEDS:
            print(
                f"evaluating {condition} implant for Phase 9 seed "
                f"{phase9_seed}",
                flush=True,
            )
            source_seed = PHASE9_SOURCE_SEEDS[phase9_seed]
            if condition == "phase8_frozen":
                checkpoint_path = (
                    PHASE8_IMPLANT_DIRECTORY / f"implant_seed_{source_seed}.pt"
                )
            else:
                checkpoint_path = (
                    PHASE9_INTERFACE_DIRECTORY
                    / f"{condition}_seed_{phase9_seed}.pt"
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
            ablated_times: list[float] = []
            for row in rows:
                output, elapsed, memory = timed(
                    lambda row=row: generate_sequence_implant(
                        bundle,
                        implant,
                        row["prompt"],
                        max_new_tokens=8,
                        latch_route=True,
                        preserve_base_when_off=True,
                        deterministic_result_step=True,
                        latch_operands=True,
                    )
                )
                latencies.append(elapsed)
                peak_observed = max(peak_observed, memory or 0)
                record = implant_record(output, row=row, elapsed=elapsed)
                if condition == "hard" and row["route_label"]:
                    ablated, ablated_elapsed, memory = timed(
                        lambda row=row: generate_sequence_implant(
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
                    )
                    ablated_times.append(ablated_elapsed)
                    peak_observed = max(peak_observed, memory or 0)
                    record["ablation"] = {
                        **ablated,
                        "format_exact": exact_format_correct(
                            ablated["generated_text"],
                            row["answer"],
                        ),
                        "latency_seconds": ablated_elapsed,
                    }
                row["implants"][condition][str(phase9_seed)] = record
            implant_latencies[condition][str(phase9_seed)] = latencies
            if condition == "hard":
                ablation_latencies[str(phase9_seed)] = ablated_times
            implant_metadata[condition][str(phase9_seed)] = {
                "source_phase8_seed": source_seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "route_threshold": checkpoint["route_threshold"],
                "digit_threshold": checkpoint["digit_threshold"],
                "architectural_learned_parameters": (
                    architectural_learned_parameter_count(implant)
                ),
                "calculator_learned_parameters": (
                    implant.calculator.trainable_parameter_count
                ),
            }
            print(
                f"completed {condition} implant for Phase 9 seed "
                f"{phase9_seed}",
                flush=True,
            )
            del bundle, implant
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    condition_metrics: dict[str, dict[str, object]] = {}
    for condition in ("phase8_frozen", "generic", "hard"):
        condition_metrics[condition] = {}
        for phase9_seed in PHASE9_TRAINING_SEEDS:
            key = str(phase9_seed)
            positive_records = [
                row["implants"][condition][key] for row in positives
            ]
            negative_records = [
                row["implants"][condition][key] for row in negatives
            ]
            conditional = [
                record
                for record in positive_records
                if record["first_route_active"] and record["operands_exact"]
            ]
            metrics = {
                "exact": sum(record["format_exact"] for record in positive_records),
                "mathematical_exact": sum(
                    record["mathematical_exact"] for record in positive_records
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
                "negative_false_routes": sum(
                    record["first_route"] for record in negative_records
                ),
                "negative_token_preserved": sum(
                    record["token_preserved"] for record in negative_records
                ),
            }
            if condition == "hard":
                metrics["ablation_exact"] = sum(
                    record["ablation"]["format_exact"]
                    for record in positive_records
                )
                metrics["paired_causal_losses"] = sum(
                    record["format_exact"]
                    and not record["ablation"]["format_exact"]
                    for record in positive_records
                )
            condition_metrics[condition][key] = metrics

    base_exact = sum(row["base"]["format_exact"] for row in positives)
    adapter_metrics = {
        str(seed): {
            "exact": sum(
                row["matched_adapters"][str(seed)]["format_exact"]
                for row in positives
            ),
            "negative_token_preserved": sum(
                row["matched_adapters"][str(seed)]["token_preserved"]
                for row in negatives
            ),
        }
        for seed in PHASE9_TRAINING_SEEDS
    }
    hard_gates: dict[str, bool] = {
        "accuracy": all(
            condition_metrics["hard"][str(seed)]["exact"] >= 95
            for seed in PHASE9_TRAINING_SEEDS
        ),
        "operands": all(
            condition_metrics["hard"][str(seed)]["operands_exact"] >= 97
            for seed in PHASE9_TRAINING_SEEDS
        ),
        "routing_and_preservation": all(
            condition_metrics["hard"][str(seed)]["negative_false_routes"] <= 4
            and condition_metrics["hard"][str(seed)][
                "negative_token_preserved"
            ]
            >= 196
            for seed in PHASE9_TRAINING_SEEDS
        ),
        "conditional_calculator_and_decode": all(
            condition_metrics["hard"][str(seed)]["conditional_exact"]
            == condition_metrics["hard"][str(seed)]["conditional_examples"]
            and condition_metrics["hard"][str(seed)]["trajectories_exact"]
            == condition_metrics["hard"][str(seed)]["conditional_examples"]
            for seed in PHASE9_TRAINING_SEEDS
        ),
        "causal_ablation": all(
            condition_metrics["hard"][str(seed)]["paired_causal_losses"] >= 90
            and condition_metrics["hard"][str(seed)]["ablation_exact"] <= 5
            for seed in PHASE9_TRAINING_SEEDS
        ),
        "hard_beats_generic": all(
            condition_metrics["hard"][str(seed)]["exact"]
            > condition_metrics["generic"][str(seed)]["exact"]
            and condition_metrics["hard"][str(seed)]["negative_false_routes"]
            <= condition_metrics["generic"][str(seed)]["negative_false_routes"]
            for seed in PHASE9_TRAINING_SEEDS
        ),
    }
    hard_gates["all_primary_gates"] = all(hard_gates.values())
    category_metrics: dict[str, object] = {}
    for condition in ("phase8_frozen", "generic", "hard"):
        category_metrics[condition] = {}
        for seed in PHASE9_TRAINING_SEEDS:
            key = str(seed)
            categories: dict[str, dict[str, int]] = {}
            for row in rows:
                category = row["split"].removeprefix(
                    "phase9_confirmatory_positive_"
                ).removeprefix("phase9_confirmatory_negative_")
                record = row["implants"][condition][key]
                aggregate = categories.setdefault(
                    category,
                    {"examples": 0, "exact": 0, "false_routes": 0},
                )
                aggregate["examples"] += 1
                aggregate["exact"] += int(record["format_exact"])
                if not row["route_label"]:
                    aggregate["false_routes"] += int(record["first_route"])
            category_metrics[condition][key] = categories

    payload = {
        "status": "phase9_confirmatory_evaluation_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "unique_prompts": len(rows),
        "positive_prompts": len(positives),
        "negative_prompts": len(negatives),
        "phase9_seeds": list(PHASE9_TRAINING_SEEDS),
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "parameter_counts": {
            "base_model": base_parameter_count,
            "implant_architectural_learned": 57_344,
            "phase9_updated": 32_768,
            "matched_adapter_learned": 57_344,
            "calculator_learned": 0,
        },
        "base": {"format_exact": base_exact},
        "matched_adapters": adapter_metrics,
        "conditions": condition_metrics,
        "primary_hard_gates": hard_gates,
        "categories": category_metrics,
        "latency": {
            "base": latency_summary(base_latencies),
            "matched_adapters": {
                seed: latency_summary(values)
                for seed, values in adapter_latencies.items()
            },
            "implants": {
                condition: {
                    seed: latency_summary(values)
                    for seed, values in per_seed.items()
                }
                for condition, per_seed in implant_latencies.items()
            },
            "hard_ablations": {
                seed: latency_summary(values)
                for seed, values in ablation_latencies.items()
            },
        },
        "memory": {
            "mps_maximum_observed_allocated_bytes": peak_observed or None,
            "measurement_note": (
                "Maximum current allocated MPS memory sampled after generations; "
                "not a device-driver peak watermark."
            ),
        },
        "checkpoints": {
            "matched_adapters": adapter_metadata,
            "implants": implant_metadata,
        },
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    fields = [
        "row_index",
        "split",
        "prompt",
        "a",
        "b",
        "answer",
        "route_label",
        "base_text",
    ]
    for condition in ("phase8_frozen", "generic", "hard"):
        for seed in PHASE9_TRAINING_SEEDS:
            fields.extend(
                (
                    f"{condition}_{seed}_text",
                    f"{condition}_{seed}_exact",
                    f"{condition}_{seed}_route",
                    f"{condition}_{seed}_operands_exact",
                    f"{condition}_{seed}_preserved",
                )
            )
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            flat = {
                "row_index": row["row_index"],
                "split": row["split"],
                "prompt": row["prompt"],
                "a": row["a"],
                "b": row["b"],
                "answer": row["answer"] or "",
                "route_label": row["route_label"],
                "base_text": row["base"]["generated_text"],
            }
            for condition in ("phase8_frozen", "generic", "hard"):
                for seed in PHASE9_TRAINING_SEEDS:
                    record = row["implants"][condition][str(seed)]
                    flat.update(
                        {
                            f"{condition}_{seed}_text": record["generated_text"],
                            f"{condition}_{seed}_exact": record["format_exact"],
                            f"{condition}_{seed}_route": record["first_route"],
                            f"{condition}_{seed}_operands_exact": record[
                                "operands_exact"
                            ],
                            f"{condition}_{seed}_preserved": record[
                                "token_preserved"
                            ],
                        }
                    )
            writer.writerow(flat)
    print(json.dumps(payload["conditions"], indent=2), flush=True)
    print(json.dumps(hard_gates, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
