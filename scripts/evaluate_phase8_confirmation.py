from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_adapter import install_matched_residual_adapter
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    PHASE8_TRAINING_SEEDS,
    build_phase8_confirmatory_examples,
)
from neural_firmware.phase8_training import generate_matched_adapter
from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    exact_format_correct,
    mathematical_correct,
)

IMPLANT_DIRECTORY = Path("phase8_artifacts/confirmatory_implants")
ADAPTER_DIRECTORY = Path(
    "phase8_artifacts/confirmatory_matched_adapters"
)
RESULT_PATH = Path("phase8_results/confirmation.json")
CSV_PATH = Path("phase8_results/confirmation_rows.csv")


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
    elapsed = time.perf_counter() - started
    return value, elapsed, mps_allocated()


def first_route(result: dict[str, object]) -> bool:
    steps = result.get("steps", [])
    return bool(steps and steps[0].get("route") == [1])


def first_route_active(result: dict[str, object]) -> bool:
    steps = result.get("steps", [])
    return bool(steps and steps[0].get("route_active") == [True])


def operands_exact(
    result: dict[str, object],
    a: str,
    b: str,
) -> bool:
    steps = result.get("steps", [])
    if not steps or steps[0].get("operands_valid") != [True]:
        return False
    step = steps[0]
    a_length = int(step["a_lengths"][0])
    b_length = int(step["b_lengths"][0])
    return (
        step["a_digits"][0][:a_length] == [int(value) for value in a]
        and step["b_digits"][0][:b_length] == [int(value) for value in b]
    )


def trajectory_exact(result: dict[str, object], answer: str) -> bool:
    expected = [int(character) for character in answer] + [10]
    observed = [
        int(step["result_symbols"][0])
        for step in result.get("steps", [])
        if step.get("result_symbols")
    ]
    return observed[: len(expected)] == expected


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "total_seconds": sum(values),
    }


def main() -> None:
    started = time.perf_counter()
    examples = build_phase8_confirmatory_examples()
    rows = [
        {
            "row_index": index,
            **example.to_dict(),
            "base": None,
            "base_extended": None,
            "matched_adapters": {},
            "implants": {},
        }
        for index, example in enumerate(examples)
    ]
    base_bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    base_parameter_count = sum(
        parameter.numel() for parameter in base_bundle.model.parameters()
    )
    base_latencies: list[float] = []
    base_extended_latencies: list[float] = []
    peak_observed = mps_allocated() or 0
    for row in rows:
        output, elapsed, memory = timed(
            lambda row=row, base_bundle=base_bundle: generate_base(
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
        if row["route_label"]:
            extended, extended_elapsed, memory = timed(
                lambda row=row, base_bundle=base_bundle: generate_base(
                    base_bundle,
                    row["prompt"],
                    max_new_tokens=64,
                )
            )
            base_extended_latencies.append(extended_elapsed)
            peak_observed = max(peak_observed, memory or 0)
            row["base_extended"] = {
                **extended,
                "format_exact": exact_format_correct(
                    extended["generated_text"],
                    row["answer"],
                ),
                "mathematical_exact": mathematical_correct(
                    extended["generated_text"],
                    row["answer"],
                ),
                "latency_seconds": extended_elapsed,
            }
    del base_bundle
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    adapter_metadata: dict[str, object] = {}
    adapter_latencies: dict[str, list[float]] = {}
    for seed in PHASE8_TRAINING_SEEDS:
        checkpoint_path = ADAPTER_DIRECTORY / f"adapter_seed_{seed}.pt"
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
            adapter.down.weight.copy_(
                checkpoint["down_weight"].to(bundle.device)
            )
            adapter.up.weight.copy_(
                checkpoint["up_weight"].to(bundle.device)
            )
        latencies: list[float] = []
        for row in rows:
            output, elapsed, memory = timed(
                lambda row=row, bundle=bundle, adapter=adapter: generate_matched_adapter(
                    bundle,
                    adapter,
                    row["prompt"],
                    max_new_tokens=8,
                    enabled=True,
                )
            )
            latencies.append(elapsed)
            peak_observed = max(peak_observed, memory or 0)
            row["matched_adapters"][str(seed)] = {
                **output,
                "format_exact": (
                    exact_format_correct(
                        output["generated_text"],
                        row["answer"],
                    )
                    if row["route_label"]
                    else False
                ),
                "mathematical_exact": (
                    mathematical_correct(
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
        adapter_latencies[str(seed)] = latencies
        adapter_metadata[str(seed)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "rank": checkpoint["rank"],
            "learned_parameters": checkpoint["learned_parameters"],
        }
        del bundle, adapter
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    implant_metadata: dict[str, object] = {}
    implant_latencies: dict[str, list[float]] = {}
    ablation_latencies: dict[str, list[float]] = {}
    for seed in PHASE8_TRAINING_SEEDS:
        checkpoint_path = IMPLANT_DIRECTORY / f"implant_seed_{seed}.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        layout = SequenceImplantLayout(**checkpoint["layout"])
        implant = install_sequence_neuron_implant(
            bundle.model,
            layer_index=checkpoint["layer_index"],
            selected_indices=checkpoint["selected_indices"],
            layout=layout,
            output_strength=checkpoint["output_strength"],
            route_threshold=checkpoint["route_threshold"],
            digit_threshold=checkpoint["digit_threshold"],
        )
        with torch.no_grad():
            implant.input_rows.copy_(
                checkpoint["input_rows"].to(bundle.device)
            )
            implant.result_columns.copy_(
                checkpoint["result_columns"].to(bundle.device)
            )
        normal_times: list[float] = []
        ablated_times: list[float] = []
        for row in rows:
            output, elapsed, memory = timed(
                lambda row=row, bundle=bundle, implant=implant: generate_sequence_implant(
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
            normal_times.append(elapsed)
            peak_observed = max(peak_observed, memory or 0)
            record = {
                **output,
                "format_exact": (
                    exact_format_correct(
                        output["generated_text"],
                        row["answer"],
                    )
                    if row["route_label"]
                    else False
                ),
                "mathematical_exact": (
                    mathematical_correct(
                        output["generated_text"],
                        row["answer"],
                    )
                    if row["route_label"]
                    else False
                ),
                "first_route": first_route(output),
                "first_route_active": first_route_active(output),
                "operands_exact": (
                    operands_exact(output, row["a"], row["b"])
                    if row["route_label"]
                    else False
                ),
                "trajectory_exact": (
                    trajectory_exact(output, row["answer"])
                    if row["route_label"]
                    else False
                ),
                "token_preserved": (
                    output["generated_token_ids"]
                    == row["base"]["generated_token_ids"]
                ),
                "latency_seconds": elapsed,
            }
            if row["route_label"]:
                ablated, ablated_elapsed, memory = timed(
                    lambda row=row, bundle=bundle, implant=implant: generate_sequence_implant(
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
                    "mathematical_exact": mathematical_correct(
                        ablated["generated_text"],
                        row["answer"],
                    ),
                    "latency_seconds": ablated_elapsed,
                }
            row["implants"][str(seed)] = record
        implant_latencies[str(seed)] = normal_times
        ablation_latencies[str(seed)] = ablated_times
        implant_metadata[str(seed)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "route_threshold": checkpoint["route_threshold"],
            "learned_parameters": implant.trainable_parameter_count,
            "calculator_learned_parameters": (
                implant.calculator.trainable_parameter_count
            ),
        }
        del bundle, implant
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    per_seed: dict[str, object] = {}
    for seed in PHASE8_TRAINING_SEEDS:
        key = str(seed)
        implant_rows = [row["implants"][key] for row in positives]
        adapter_rows = [
            row["matched_adapters"][key] for row in positives
        ]
        negative_implants = [row["implants"][key] for row in negatives]
        conditional = [
            record
            for record in implant_rows
            if record["first_route_active"] and record["operands_exact"]
        ]
        exact = sum(record["format_exact"] for record in implant_rows)
        ablation_exact = sum(
            record["ablation"]["format_exact"] for record in implant_rows
        )
        causal_losses = sum(
            record["format_exact"]
            and not record["ablation"]["format_exact"]
            for record in implant_rows
        )
        adapter_exact = sum(
            record["format_exact"] for record in adapter_rows
        )
        false_routes = sum(
            record["first_route"] for record in negative_implants
        )
        preserved = sum(
            record["token_preserved"] for record in negative_implants
        )
        per_seed[key] = {
            "implant_exact": exact,
            "implant_direct_exact": sum(
                row["implants"][key]["format_exact"]
                for row in rows
                if row["split"] == "phase8_confirmatory_direct"
            ),
            "implant_word_exact": sum(
                row["implants"][key]["format_exact"]
                for row in rows
                if row["split"] == "phase8_confirmatory_word"
            ),
            "positive_routes": sum(
                record["first_route"] for record in implant_rows
            ),
            "positive_active_routes": sum(
                record["first_route_active"] for record in implant_rows
            ),
            "operands_exact": sum(
                record["operands_exact"] for record in implant_rows
            ),
            "trajectories_exact": sum(
                record["trajectory_exact"] for record in implant_rows
            ),
            "conditional_examples": len(conditional),
            "conditional_exact": sum(
                record["format_exact"] for record in conditional
            ),
            "ablation_exact": ablation_exact,
            "paired_causal_losses": causal_losses,
            "negative_false_routes": false_routes,
            "negative_token_preserved": preserved,
            "matched_adapter_exact": adapter_exact,
            "matched_adapter_negative_token_preserved": sum(
                row["matched_adapters"][key]["token_preserved"]
                for row in negatives
            ),
        }
    base_exact = sum(row["base"]["format_exact"] for row in positives)
    base_math = sum(
        row["base"]["mathematical_exact"] for row in positives
    )
    base_extended_exact = sum(
        row["base_extended"]["format_exact"] for row in positives
    )
    base_extended_math = sum(
        row["base_extended"]["mathematical_exact"] for row in positives
    )
    implant_mean = statistics.fmean(
        record["implant_exact"] for record in per_seed.values()
    )
    gate_results = {
        "accuracy": (
            all(record["implant_exact"] >= 57 for record in per_seed.values())
            and implant_mean >= 58
        ),
        "operands": all(
            record["operands_exact"] >= 57 for record in per_seed.values()
        ),
        "causal_ablation": all(
            record["paired_causal_losses"] >= 50
            for record in per_seed.values()
        ),
        "routing_and_preservation": all(
            record["negative_false_routes"] <= 2
            and record["negative_token_preserved"] >= 58
            for record in per_seed.values()
        ),
        "calculator_and_decode": all(
            record["trajectories_exact"] == record["conditional_examples"]
            and record["conditional_exact"]
            >= min(59, record["conditional_examples"])
            for record in per_seed.values()
        ),
        "beats_base_and_adapter": all(
            record["implant_exact"] > base_exact
            and record["implant_exact"] > record["matched_adapter_exact"]
            for record in per_seed.values()
        ),
    }
    gate_results["all_primary_gates"] = all(gate_results.values())
    payload = {
        "status": "confirmatory_evaluation_complete",
        "implementation_commit": git_commit(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "unique_prompts": len(rows),
        "addition_prompts": len(positives),
        "negative_prompts": len(negatives),
        "training_seeds": list(PHASE8_TRAINING_SEEDS),
        "decoding": {
            "method": "greedy",
            "max_new_tokens": 8,
            "base_extended_max_new_tokens": 64,
        },
        "parameter_counts": {
            "base_model": base_parameter_count,
            "implant_learned": 57_344,
            "matched_adapter_learned": 57_344,
            "calculator_learned": 0,
            "learned_fraction_of_base": 57_344 / base_parameter_count,
        },
        "base": {
            "format_exact": base_exact,
            "mathematical_exact": base_math,
            "extended_format_exact": base_extended_exact,
            "extended_mathematical_exact": base_extended_math,
        },
        "per_seed": per_seed,
        "implant_mean_exact": implant_mean,
        "gates": gate_results,
        "latency": {
            "base": latency_summary(base_latencies),
            "base_extended": latency_summary(base_extended_latencies),
            "matched_adapters": {
                seed: latency_summary(values)
                for seed, values in adapter_latencies.items()
            },
            "implants": {
                seed: latency_summary(values)
                for seed, values in implant_latencies.items()
            },
            "ablations": {
                seed: latency_summary(values)
                for seed, values in ablation_latencies.items()
            },
        },
        "memory": {
            "mps_maximum_observed_allocated_bytes": peak_observed or None,
            "measurement_note": (
                "Maximum of current allocated MPS memory sampled after each "
                "generation; not a device-driver peak watermark."
            ),
        },
        "checkpoints": {
            "implants": implant_metadata,
            "matched_adapters": adapter_metadata,
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
        "base_format_exact",
    ]
    for seed in PHASE8_TRAINING_SEEDS:
        fields.extend(
            [
                f"adapter_{seed}_text",
                f"adapter_{seed}_format_exact",
                f"implant_{seed}_text",
                f"implant_{seed}_format_exact",
                f"implant_{seed}_first_route",
                f"implant_{seed}_operands_exact",
                f"implant_{seed}_trajectory_exact",
                f"implant_{seed}_ablation_exact",
                f"implant_{seed}_token_preserved",
            ]
        )
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                "base_format_exact": row["base"]["format_exact"],
            }
            for seed in PHASE8_TRAINING_SEEDS:
                key = str(seed)
                adapter = row["matched_adapters"][key]
                implant = row["implants"][key]
                flat.update(
                    {
                        f"adapter_{seed}_text": adapter["generated_text"],
                        f"adapter_{seed}_format_exact": adapter["format_exact"],
                        f"implant_{seed}_text": implant["generated_text"],
                        f"implant_{seed}_format_exact": implant["format_exact"],
                        f"implant_{seed}_first_route": implant["first_route"],
                        f"implant_{seed}_operands_exact": implant["operands_exact"],
                        f"implant_{seed}_trajectory_exact": implant[
                            "trajectory_exact"
                        ],
                        f"implant_{seed}_ablation_exact": (
                            implant.get("ablation", {}).get("format_exact", "")
                        ),
                        f"implant_{seed}_token_preserved": implant[
                            "token_preserved"
                        ],
                    }
                )
            writer.writerow(flat)
    print(json.dumps({**payload["base"], **per_seed}, indent=2), flush=True)
    print(json.dumps(gate_results, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
