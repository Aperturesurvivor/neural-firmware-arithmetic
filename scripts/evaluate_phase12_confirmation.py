from __future__ import annotations

import hashlib
import json
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch

from evaluate_phase11_confirmation import (
    generate_base,
    implant_record,
    latency_summary,
    tensor_inheritance,
)
from neural_firmware.phase12_data import build_phase12_confirmatory_examples
from neural_firmware.phase7_sequence_training import generate_sequence_implant
from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    install_checkpoint_implant,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

MANIFEST_PATH = Path("phase12_results/frozen_prompt_manifest.json")
RESULT_PATH = Path("phase12_results/confirmation.json")
PROGRESS_PATH = Path("phase12_artifacts/confirmation_progress.json")
PROGRESS_VERSION = 1
PAUSE_REQUESTED = False


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


def request_pause(signum: int, _frame: object) -> None:
    global PAUSE_REQUESTED
    PAUSE_REQUESTED = True
    print(
        f"received signal {signum}; pausing after the current prompt",
        flush=True,
    )


def evaluator_sha256() -> str:
    return sha256(Path(__file__).resolve())


def save_progress(
    rows: list[dict[str, object]],
    *,
    manifest_sha256: str,
    accumulated_before: float,
    invocation_started: float,
    status: str = "phase12_confirmation_in_progress",
) -> None:
    payload = {
        "status": status,
        "progress_version": PROGRESS_VERSION,
        "manifest_sha256": manifest_sha256,
        "canonical_rows_sha256": json.loads(
            MANIFEST_PATH.read_text()
        )["canonical_rows_sha256"],
        "evaluator_sha256": evaluator_sha256(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "accumulated_wall_time_seconds": (
            accumulated_before + time.perf_counter() - invocation_started
        ),
        "rows": rows,
    }
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(PROGRESS_PATH)


def load_or_initialize_rows(
    examples: list[object],
    *,
    manifest_sha256: str,
    canonical_rows_sha256: str,
) -> tuple[list[dict[str, object]], float]:
    if not PROGRESS_PATH.exists():
        return (
            [
                {
                    "row_index": index,
                    **example.to_dict(),
                    "base": None,
                    "conditions": {
                        "phase11_control": {},
                        "phase12_candidate": {},
                    },
                }
                for index, example in enumerate(examples)
            ],
            0.0,
        )
    progress = json.loads(PROGRESS_PATH.read_text())
    expected = {
        "progress_version": PROGRESS_VERSION,
        "manifest_sha256": manifest_sha256,
        "canonical_rows_sha256": canonical_rows_sha256,
        "evaluator_sha256": evaluator_sha256(),
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise ValueError(
                f"saved confirmation progress has incompatible {key}"
            )
    rows = progress["rows"]
    frozen = [example.to_dict() for example in examples]
    if len(rows) != len(frozen):
        raise ValueError("saved progress has the wrong row count")
    for index, (row, expected_row) in enumerate(
        zip(rows, frozen, strict=True)
    ):
        observed = {
            key: row[key]
            for key in expected_row
        }
        if row["row_index"] != index or observed != expected_row:
            raise ValueError("saved progress does not match frozen rows")
    print(f"resuming saved progress from {PROGRESS_PATH}", flush=True)
    return rows, float(progress["accumulated_wall_time_seconds"])


def first_missing_base(rows: list[dict[str, object]]) -> int:
    missing = [index for index, row in enumerate(rows) if row["base"] is None]
    return missing[0] if missing else len(rows)


def first_missing_condition(
    rows: list[dict[str, object]],
    *,
    condition: str,
    key: str,
) -> int:
    missing = [
        index
        for index, row in enumerate(rows)
        if key not in row["conditions"][condition]
    ]
    return missing[0] if missing else len(rows)


def validate_contiguous_progress(
    rows: list[dict[str, object]],
    *,
    start: int,
    condition: str | None = None,
    key: str | None = None,
) -> None:
    for index in range(start, len(rows)):
        present = (
            rows[index]["base"] is not None
            if condition is None
            else key in rows[index]["conditions"][condition]
        )
        if present:
            raise ValueError("saved progress contains a non-contiguous stage")


def category_name(row: dict[str, object]) -> str:
    return str(row["split"]).removeprefix(
        "phase12_confirmatory_positive_"
    ).removeprefix("phase12_confirmatory_negative_")


def summarize_condition(
    rows: list[dict[str, object]],
    *,
    condition: str,
    seed: int,
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    key = str(seed)
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
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
    metrics: dict[str, object] = {
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
    }
    categories: dict[str, dict[str, int]] = {}
    for row in rows:
        record = row["conditions"][condition][key]
        category = category_name(row)
        aggregate = categories.setdefault(
            category,
            {
                "examples": 0,
                "routes": 0,
                "exact": 0,
                "token_preserved": 0,
            },
        )
        aggregate["examples"] += 1
        aggregate["routes"] += int(record["first_route"])
        aggregate["exact"] += int(record["format_exact"])
        aggregate["token_preserved"] += int(record["token_preserved"])
    return metrics, categories


def main() -> None:
    global PAUSE_REQUESTED
    PAUSE_REQUESTED = False
    signal.signal(signal.SIGINT, request_pause)
    signal.signal(signal.SIGTERM, request_pause)
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_sha = sha256(MANIFEST_PATH)
    examples = build_phase12_confirmatory_examples()
    frozen_rows = [example.to_dict() for example in examples]
    canonical = json.dumps(
        frozen_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest[
        "canonical_rows_sha256"
    ]:
        raise ValueError("Phase 12 rows do not match the frozen manifest")
    rows, accumulated_before = load_or_initialize_rows(
        examples,
        manifest_sha256=manifest_sha,
        canonical_rows_sha256=manifest["canonical_rows_sha256"],
    )

    base_start = first_missing_base(rows)
    validate_contiguous_progress(rows, start=base_start)
    if base_start < len(rows):
        base_bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        print(
            f"evaluating untouched TinyLlama baseline from row {base_start}",
            flush=True,
        )
        for index in range(base_start, len(rows)):
            row = rows[index]
            row_started = time.perf_counter()
            output = generate_base(
                base_bundle,
                row["prompt"],
                max_new_tokens=8,
            )
            elapsed = time.perf_counter() - row_started
            row["base"] = {
                **output,
                "format_exact": (
                    exact_format_correct(
                        output["generated_text"],
                        row["answer"],
                    )
                    if row["route_label"]
                    else False
                ),
                "latency_seconds": elapsed,
            }
            if (index + 1) % 10 == 0 or PAUSE_REQUESTED:
                save_progress(
                    rows,
                    manifest_sha256=manifest_sha,
                    accumulated_before=accumulated_before,
                    invocation_started=started,
                )
            if (index + 1) % 25 == 0:
                print(
                    f"base: evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
            if PAUSE_REQUESTED:
                del base_bundle
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print("Phase 12 confirmation paused safely", flush=True)
                return
        del base_bundle
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    base_latencies = [
        float(row["base"]["latency_seconds"])
        for row in rows
    ]

    checkpoint_metadata: dict[str, dict[str, object]] = {}
    latencies: dict[str, dict[str, object]] = {
        "base": latency_summary(base_latencies),
        "phase11_control": {},
        "phase12_candidate": {},
    }
    for seed in manifest["source_seeds"]:
        key = str(seed)
        control_meta = manifest["phase11_control_checkpoints"][key]
        candidate_meta = manifest["candidate_checkpoints"][key]
        source_path = Path(candidate_meta["source_phase10_checkpoint"])
        control_path = Path(control_meta["path"])
        candidate_path = Path(candidate_meta["path"])
        for path, expected in (
            (source_path, candidate_meta["source_phase10_checkpoint_sha256"]),
            (control_path, control_meta["sha256"]),
            (candidate_path, candidate_meta["sha256"]),
        ):
            if sha256(path) != expected:
                raise ValueError(f"checkpoint hash mismatch: {path}")
        source_checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        control_checkpoint = torch.load(
            control_path,
            map_location="cpu",
            weights_only=True,
        )
        candidate_checkpoint = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=True,
        )
        inheritance = tensor_inheritance(
            source_checkpoint,
            candidate_checkpoint,
        )

        control_start = first_missing_condition(
            rows,
            condition="phase11_control",
            key=key,
        )
        validate_contiguous_progress(
            rows,
            start=control_start,
            condition="phase11_control",
            key=key,
        )
        control_bundle = load_model_bundle(
            PHASE8_MODEL_ID,
            revision=PHASE8_MODEL_REVISION,
        )
        control_implant = install_checkpoint_implant(
            control_bundle,
            control_checkpoint,
        )
        control_parameters = architectural_learned_parameter_count(
            control_implant
        )
        if control_start < len(rows):
            print(
                f"phase11 control seed {seed}: resuming at row "
                f"{control_start}",
                flush=True,
            )
        for index in range(control_start, len(rows)):
            row = rows[index]
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
            row["conditions"]["phase11_control"][key] = implant_record(
                output,
                row=row,
                elapsed=elapsed,
            )
            if (index + 1) % 10 == 0 or PAUSE_REQUESTED:
                save_progress(
                    rows,
                    manifest_sha256=manifest_sha,
                    accumulated_before=accumulated_before,
                    invocation_started=started,
                )
            if (index + 1) % 25 == 0:
                print(
                    f"phase11 control seed {seed}: "
                    f"evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
            if PAUSE_REQUESTED:
                del control_bundle, control_implant
                del source_checkpoint, control_checkpoint
                del candidate_checkpoint
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print("Phase 12 confirmation paused safely", flush=True)
                return
        control_times = [
            float(
                row["conditions"]["phase11_control"][key][
                    "latency_seconds"
                ]
            )
            for row in rows
        ]
        latencies["phase11_control"][key] = latency_summary(control_times)
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
        candidate_start = first_missing_condition(
            rows,
            condition="phase12_candidate",
            key=key,
        )
        validate_contiguous_progress(
            rows,
            start=candidate_start,
            condition="phase12_candidate",
            key=key,
        )
        if candidate_start < len(rows):
            print(
                f"phase12 candidate seed {seed}: resuming at row "
                f"{candidate_start}",
                flush=True,
            )
        for index in range(candidate_start, len(rows)):
            row = rows[index]
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
                record["route_off"] = implant_record(
                    route_off,
                    row=row,
                    elapsed=route_off_elapsed,
                )
            row["conditions"]["phase12_candidate"][key] = record
            if (index + 1) % 10 == 0 or PAUSE_REQUESTED:
                save_progress(
                    rows,
                    manifest_sha256=manifest_sha,
                    accumulated_before=accumulated_before,
                    invocation_started=started,
                )
            if (index + 1) % 25 == 0:
                print(
                    f"phase12 candidate seed {seed}: "
                    f"evaluated {index + 1}/{len(rows)}",
                    flush=True,
                )
            if PAUSE_REQUESTED:
                del candidate_bundle, candidate_implant
                del source_checkpoint, control_checkpoint
                del candidate_checkpoint
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print("Phase 12 confirmation paused safely", flush=True)
                return
        candidate_times = [
            float(
                row["conditions"]["phase12_candidate"][key][
                    "latency_seconds"
                ]
            )
            for row in rows
        ]
        oracle_times = [
            float(
                row["conditions"]["phase12_candidate"][key][
                    "oracle_route"
                ]["latency_seconds"]
            )
            for row in rows
            if row["route_label"]
        ]
        route_off_times = [
            float(
                row["conditions"]["phase12_candidate"][key][
                    "route_off"
                ]["latency_seconds"]
            )
            for row in rows
            if row["route_label"]
        ]
        candidate_parameters = architectural_learned_parameter_count(
            candidate_implant
        )
        latencies["phase12_candidate"][key] = {
            "normal": latency_summary(candidate_times),
            "oracle_route": latency_summary(oracle_times),
            "route_off": latency_summary(route_off_times),
        }
        checkpoint_metadata[key] = {
            "source": {
                "path": str(source_path),
                "sha256": candidate_meta[
                    "source_phase10_checkpoint_sha256"
                ],
            },
            "phase11_control": {
                "path": str(control_path),
                "sha256": control_meta["sha256"],
                "architectural_learned_parameters": control_parameters,
            },
            "phase12_candidate": {
                "path": str(candidate_path),
                "sha256": candidate_meta["sha256"],
                "request_router_kind": candidate_implant.request_router_kind,
                "request_route_threshold": (
                    candidate_implant.request_route_threshold
                ),
                "request_route_temperature": (
                    candidate_implant.request_route_temperature
                ),
                "request_router_parameters": int(
                    candidate_implant.request_route_down.numel()
                    + candidate_implant.request_route_output.numel()
                ),
                "architectural_learned_parameters": candidate_parameters,
            },
            "inheritance": inheritance,
        }
        del candidate_bundle, candidate_implant
        del source_checkpoint, control_checkpoint, candidate_checkpoint
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    metrics: dict[str, dict[str, dict[str, object]]] = {
        "phase11_control": {},
        "phase12_candidate": {},
    }
    categories: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        "phase11_control": {},
        "phase12_candidate": {},
    }
    for condition in ("phase11_control", "phase12_candidate"):
        for seed in manifest["source_seeds"]:
            key = str(seed)
            seed_metrics, seed_categories = summarize_condition(
                rows,
                condition=condition,
                seed=seed,
            )
            if condition == "phase12_candidate":
                positives = [row for row in rows if row["route_label"]]
                normal = [
                    row["conditions"][condition][key] for row in positives
                ]
                oracle = [record["oracle_route"] for record in normal]
                route_off = [record["route_off"] for record in normal]
                seed_metrics.update(
                    {
                        "oracle_exact": sum(
                            record["format_exact"] for record in oracle
                        ),
                        "route_off_exact": sum(
                            record["format_exact"] for record in route_off
                        ),
                        "paired_route_off_losses": sum(
                            normal_record["format_exact"]
                            and not off_record["format_exact"]
                            for normal_record, off_record in zip(
                                normal,
                                route_off,
                                strict=True,
                            )
                        ),
                    }
                )
            metrics[condition][key] = seed_metrics
            categories[condition][key] = seed_categories

    paired_gains = {
        str(seed): (
            metrics["phase12_candidate"][str(seed)]["exact"]
            - metrics["phase11_control"][str(seed)]["exact"]
        )
        for seed in manifest["source_seeds"]
    }
    candidate_metrics = metrics["phase12_candidate"]
    gates = {
        "autonomous_exactness": (
            all(value["exact"] >= 70 for value in candidate_metrics.values())
            and statistics.fmean(
                value["exact"] for value in candidate_metrics.values()
            )
            >= 75
        ),
        "paired_improvement": (
            all(gain > 0 for gain in paired_gains.values())
            and statistics.fmean(paired_gains.values()) >= 10
        ),
        "route_recognition": all(
            value["positive_routes"] >= 80
            for value in candidate_metrics.values()
        ),
        "preservation": all(
            value["false_routes"] <= 4
            and value["token_preserved"] >= 196
            for value in candidate_metrics.values()
        ),
        "operand_access": all(
            value["oracle_exact"] >= 85
            for value in candidate_metrics.values()
        ),
        "conditional_mechanism": all(
            value["conditional_exact"] == value["conditional_examples"]
            and value["conditional_trajectories_exact"]
            == value["conditional_examples"]
            for value in candidate_metrics.values()
        ),
        "causal_routing": all(
            value["paired_route_off_losses"] == value["exact"]
            and value["route_off_exact"] <= 5
            for value in candidate_metrics.values()
        ),
        "checkpoint_integrity": all(
            metadata["phase12_candidate"]["request_router_parameters"]
            == 131_104
            and metadata["inheritance"][
                "all_inherited_tensors_bit_identical"
            ]
            for metadata in checkpoint_metadata.values()
        ),
    }
    gates["all_gates"] = all(gates.values())
    payload = {
        "status": "phase12_confirmation_complete",
        "implementation_commit": git_commit(),
        "manifest": {
            "path": str(MANIFEST_PATH),
            "sha256": sha256(MANIFEST_PATH),
            "canonical_rows_sha256": manifest["canonical_rows_sha256"],
        },
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "decoding": {"method": "greedy", "max_new_tokens": 8},
        "prompts": len(rows),
        "positive_prompts": sum(row["route_label"] for row in rows),
        "negative_prompts": sum(not row["route_label"] for row in rows),
        "metrics": metrics,
        "categories": categories,
        "paired_exact_gains": paired_gains,
        "gates": gates,
        "checkpoints": checkpoint_metadata,
        "latencies": latencies,
        "rows": rows,
        "wall_time_seconds": (
            accumulated_before + time.perf_counter() - started
        ),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    save_progress(
        rows,
        manifest_sha256=manifest_sha,
        accumulated_before=accumulated_before,
        invocation_started=started,
        status="phase12_confirmation_complete",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": metrics,
                "paired_exact_gains": paired_gains,
                "gates": gates,
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
