from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from evaluate_phase11_development import compact_output
from neural_firmware.phase11_routing import RequestRouteFeatureSet
from neural_firmware.phase12_routing import (
    SiluRouterState,
    evaluate_silu_router,
)
from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.pretrained_training import load_model_bundle
from run_phase12_family_cv import condition_summary

FEATURE_CACHE_PATH = Path(
    "phase12_artifacts/cache/disclosed_phase11_features.pt"
)
TRAINING_RESULT_PATH = Path("phase12_results/deployment_training.json")
PHASE11_RESULT_PATH = Path("phase11_results/confirmation.json")
RESULT_PATH = Path("phase12_results/deployment_evaluation.json")


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


def tensor_inheritance(
    source: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
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
        "comparisons": comparisons,
        "all_inherited_tensors_bit_identical": all(comparisons.values()),
    }


def main() -> None:
    started = time.perf_counter()
    training = json.loads(TRAINING_RESULT_PATH.read_text())
    phase11 = json.loads(PHASE11_RESULT_PATH.read_text())
    feature_cache = torch.load(
        FEATURE_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )
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
        for row in phase11["rows"]
    ]
    records = {
        int(record["phase10_seed"]): record
        for record in training["records"]
    }
    metrics: dict[str, dict[str, object]] = {}
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    for seed, record in records.items():
        key = str(seed)
        checkpoint_path = Path(record["checkpoint"])
        if sha256(checkpoint_path) != record["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        source_path = Path(record["source_phase10_checkpoint"])
        source = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        inheritance = tensor_inheritance(source, checkpoint)
        views = [
            RequestRouteFeatureSet.load_state_dict(
                feature_cache["features"][key][kind]
            )
            for kind in training["views"]
        ]
        features = RequestRouteFeatureSet(
            hidden=torch.cat(
                [view.hidden.float() for view in views],
                dim=-1,
            ),
            targets=views[0].targets,
        )
        offline = evaluate_silu_router(
            SiluRouterState(
                down=checkpoint["request_route_down"],
                output=checkpoint["request_route_output"],
            ),
            features,
            threshold=float(checkpoint["request_route_threshold"]),
            temperature=float(checkpoint["request_route_temperature"]),
        )
        expected = condition_summary(
            phase11,
            seed=seed,
            predictions=offline["predictions"],
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
                    f"phase12 deployment seed {seed}: "
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
            "exact": sum(
                result["format_exact"] for result in positive_records
            ),
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
                row["seeds"][key]["runtime_offline_route_match"]
                for row in rows
            ),
            "maximum_runtime_offline_probability_difference": max(
                abs(
                    row["seeds"][key]["first_route_probability"]
                    - row["seeds"][key]["offline_route_probability"]
                )
                for row in rows
            ),
            "expected_counterfactual_exact": expected[
                "counterfactual_exact"
            ],
            "expected_false_routes": expected["false_routes"],
            "latency_mean_seconds": statistics.fmean(latencies),
            "latency_median_seconds": statistics.median(latencies),
            "latency_maximum_seconds": max(latencies),
        }
        actual["all_runtime_offline_routes_match"] = (
            actual["runtime_offline_route_matches"] == len(rows)
        )
        actual["counterfactual_exact_match"] = (
            actual["exact"] == actual["expected_counterfactual_exact"]
        )
        actual["counterfactual_false_routes_match"] = (
            actual["false_routes"] == actual["expected_false_routes"]
        )
        metrics[key] = actual
        checkpoint_metadata[key] = {
            "path": str(checkpoint_path),
            "sha256": record["checkpoint_sha256"],
            "source_path": str(source_path),
            "source_sha256": record["source_phase10_checkpoint_sha256"],
            "phase12_router_seed": record["phase12_router_seed"],
            "request_router_kind": implant.request_router_kind,
            "request_route_threshold": implant.request_route_threshold,
            "request_route_temperature": implant.request_route_temperature,
            "request_router_parameters": int(
                implant.request_route_down.numel()
                + implant.request_route_output.numel()
            ),
            "architectural_learned_parameters": (
                architectural_learned_parameter_count(implant)
            ),
            "inheritance": inheritance,
        }
        del bundle, implant, checkpoint, source
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    validation_gates = {
        "runtime_matches_offline_routes": all(
            value["all_runtime_offline_routes_match"]
            for value in metrics.values()
        ),
        "runtime_matches_counterfactual_exact": all(
            value["counterfactual_exact_match"]
            for value in metrics.values()
        ),
        "runtime_matches_counterfactual_false_routes": all(
            value["counterfactual_false_routes_match"]
            for value in metrics.values()
        ),
        "preservation": all(
            value["false_routes"] <= 4
            and value["token_preserved"] >= 196
            for value in metrics.values()
        ),
        "conditional_mechanism": all(
            value["conditional_exact"] == value["conditional_examples"]
            and value["conditional_trajectories_exact"]
            == value["conditional_examples"]
            for value in metrics.values()
        ),
        "checkpoint_inheritance": all(
            value["inheritance"]["all_inherited_tensors_bit_identical"]
            for value in checkpoint_metadata.values()
        ),
    }
    validation_gates["all_gates"] = all(validation_gates.values())
    payload = {
        "status": "phase12_deployment_development_evaluation_complete",
        "implementation_commit": git_commit(),
        "interpretation": (
            "end-to-end validation on disclosed Phase 11 development data"
        ),
        "condition": training["condition"],
        "fixed_threshold": training["fixed_threshold"],
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "prompts": len(rows),
        "positive_prompts": sum(row["route_label"] for row in rows),
        "negative_prompts": sum(not row["route_label"] for row in rows),
        "metrics": metrics,
        "validation_gates": validation_gates,
        "checkpoints": checkpoint_metadata,
        "feature_cache": {
            "path": str(FEATURE_CACHE_PATH),
            "sha256": sha256(FEATURE_CACHE_PATH),
        },
        "deployment_training": {
            "path": str(TRAINING_RESULT_PATH),
            "sha256": sha256(TRAINING_RESULT_PATH),
        },
        "phase11_result": {
            "path": str(PHASE11_RESULT_PATH),
            "sha256": sha256(PHASE11_RESULT_PATH),
        },
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": metrics,
                "validation_gates": validation_gates,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
