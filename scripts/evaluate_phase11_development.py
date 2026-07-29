from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase10_data import PHASE10_TRAINING_SEEDS
from neural_firmware.phase11_routing import (
    RequestRouteFeatureSet,
    evaluate_request_router,
)
from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

FEATURE_CACHE_PATH = Path(
    "phase11_artifacts/cache/request_route_features.pt"
)
TRAINING_RESULT_PATH = Path("phase11_results/development_training.json")
PHASE10_RESULT_PATH = Path("phase10_results/confirmation.json")
RESULT_PATH = Path("phase11_results/development_evaluation.json")


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


def compact_output(
    output: dict[str, object],
    *,
    row: dict[str, object],
    elapsed: float,
) -> dict[str, object]:
    positive = bool(row["route_label"])
    return {
        "generated_token_ids": output["generated_token_ids"],
        "generated_text": output["generated_text"],
        "first_route": bool(_first(output, "route", 0)),
        "first_route_active": bool(_first(output, "route_active", False)),
        "first_route_probability": float(
            _first(output, "route_probability", float("nan"))
        ),
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


def main() -> None:
    started = time.perf_counter()
    training = json.loads(TRAINING_RESULT_PATH.read_text())
    phase10 = json.loads(PHASE10_RESULT_PATH.read_text())
    router_kind = training["selected_router_kind"]
    if router_kind is None:
        raise ValueError("Phase 11 development did not select a router")
    feature_cache = torch.load(
        FEATURE_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    selected_records = {
        int(record["phase10_seed"]): record
        for record in training["records"]
        if record["router_kind"] == router_kind
    }
    if set(selected_records) != set(PHASE10_TRAINING_SEEDS):
        raise ValueError("selected router checkpoints do not match Phase 10 seeds")

    rows = [
        {
            "row_index": int(row["row_index"]),
            "split": row["split"],
            "prompt": row["prompt"],
            "route_label": bool(row["route_label"]),
            "a": row["a"],
            "b": row["b"],
            "answer": row["answer"],
            "base": {
                "generated_token_ids": row["base"]["generated_token_ids"],
                "generated_text": row["base"]["generated_text"],
            },
            "seeds": {},
        }
        for row in phase10["rows"]
    ]
    seed_metrics: dict[str, dict[str, object]] = {}
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    for seed in PHASE10_TRAINING_SEEDS:
        key = str(seed)
        record = selected_records[seed]
        checkpoint_path = Path(record["checkpoint"])
        if sha256(checkpoint_path) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        features = RequestRouteFeatureSet.load_state_dict(
            feature_cache["splits"]["selection"]["features"][key][router_kind]
        )
        offline = evaluate_request_router(
            checkpoint["request_route_rows"],
            features,
            threshold=float(checkpoint["request_route_threshold"]),
            temperature=float(checkpoint["request_route_temperature"]),
        )
        bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        implant = install_checkpoint_implant(bundle, checkpoint)
        latencies: list[float] = []
        for index, row in enumerate(rows):
            row_started = time.perf_counter()
            output = generate_sequence_implant(
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
            result = compact_output(output, row=row, elapsed=elapsed)
            result["offline_route"] = bool(offline["predictions"][index])
            result["offline_route_probability"] = float(
                offline["probabilities"][index]
            )
            result["runtime_offline_route_match"] = (
                result["first_route"] == result["offline_route"]
            )
            row["seeds"][key] = result
            if (index + 1) % 25 == 0:
                print(
                    f"{router_kind} seed {seed}: "
                    f"evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
        positives = [row for row in rows if row["route_label"]]
        negatives = [row for row in rows if not row["route_label"]]
        positive_records = [row["seeds"][key] for row in positives]
        negative_records = [row["seeds"][key] for row in negatives]
        conditional = [
            result
            for result in positive_records
            if result["first_route_active"] and result["operands_exact"]
        ]
        actual = {
            "exact": sum(result["format_exact"] for result in positive_records),
            "positive_routes": sum(
                result["first_route"] for result in positive_records
            ),
            "positive_active_routes": sum(
                result["first_route_active"] for result in positive_records
            ),
            "operands_exact": sum(
                result["operands_exact"] for result in positive_records
            ),
            "trajectories_exact": sum(
                result["trajectory_exact"] for result in positive_records
            ),
            "conditional_examples": len(conditional),
            "conditional_exact": sum(
                result["format_exact"] for result in conditional
            ),
            "conditional_trajectories_exact": sum(
                result["trajectory_exact"] for result in conditional
            ),
            "false_routes": sum(
                result["first_route"] for result in negative_records
            ),
            "token_preserved": sum(
                result["token_preserved"] for result in negative_records
            ),
            "runtime_offline_route_matches": sum(
                row["seeds"][key]["runtime_offline_route_match"] for row in rows
            ),
            "maximum_runtime_offline_probability_difference": max(
                abs(
                    row["seeds"][key]["first_route_probability"]
                    - row["seeds"][key]["offline_route_probability"]
                )
                for row in rows
            ),
            "latency_mean_seconds": statistics.fmean(latencies),
            "latency_median_seconds": statistics.median(latencies),
            "latency_maximum_seconds": max(latencies),
        }
        expected = record["counterfactual"]
        actual["counterfactual_exact_match"] = (
            actual["exact"] == expected["counterfactual_exact"]
        )
        actual["counterfactual_false_routes_match"] = (
            actual["false_routes"] == expected["false_routes"]
        )
        actual["all_runtime_offline_routes_match"] = (
            actual["runtime_offline_route_matches"] == len(rows)
        )
        seed_metrics[key] = actual
        checkpoint_metadata[key] = {
            "path": str(checkpoint_path),
            "sha256": record["checkpoint_sha256"],
            "source_phase10_checkpoint_sha256": record[
                "source_phase10_checkpoint_sha256"
            ],
            "phase11_router_seed": record["phase11_router_seed"],
            "request_router_kind": implant.request_router_kind,
            "request_route_threshold": implant.request_route_threshold,
            "request_route_temperature": implant.request_route_temperature,
            "architectural_learned_parameters": (
                architectural_learned_parameter_count(implant)
            ),
        }
        del bundle, implant, checkpoint
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    validation_gates = {
        "runtime_matches_offline_routes": all(
            metrics["all_runtime_offline_routes_match"]
            for metrics in seed_metrics.values()
        ),
        "runtime_matches_counterfactual_exact": all(
            metrics["counterfactual_exact_match"]
            for metrics in seed_metrics.values()
        ),
        "runtime_matches_counterfactual_false_routes": all(
            metrics["counterfactual_false_routes_match"]
            for metrics in seed_metrics.values()
        ),
        "preservation": all(
            metrics["false_routes"] <= 4 and metrics["token_preserved"] >= 196
            for metrics in seed_metrics.values()
        ),
        "conditional_mechanism": all(
            metrics["conditional_exact"] == metrics["conditional_examples"]
            and metrics["conditional_trajectories_exact"]
            == metrics["conditional_examples"]
            for metrics in seed_metrics.values()
        ),
    }
    validation_gates["all_gates"] = all(validation_gates.values())
    payload = {
        "status": "phase11_selected_router_development_evaluation_complete",
        "implementation_commit": git_commit(),
        "router_kind": router_kind,
        "selection_source": "disclosed_phase10_confirmation",
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "prompts": len(rows),
        "positive_prompts": sum(row["route_label"] for row in rows),
        "negative_prompts": sum(not row["route_label"] for row in rows),
        "metrics": seed_metrics,
        "validation_gates": validation_gates,
        "checkpoints": checkpoint_metadata,
        "feature_cache": {
            "path": str(FEATURE_CACHE_PATH),
            "sha256": sha256(FEATURE_CACHE_PATH),
        },
        "development_training": {
            "path": str(TRAINING_RESULT_PATH),
            "sha256": sha256(TRAINING_RESULT_PATH),
        },
        "phase10_result": {
            "path": str(PHASE10_RESULT_PATH),
            "sha256": sha256(PHASE10_RESULT_PATH),
        },
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "router_kind": router_kind,
                "metrics": seed_metrics,
                "validation_gates": validation_gates,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
