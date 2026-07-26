from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase5_data import (
    build_phase5_confirmatory_negatives,
    build_phase5_confirmatory_positive_sets,
    build_phase5_development_examples,
)
from neural_firmware.phase5_igc import install_dual_depth_igc
from neural_firmware.phase5_training import generate_dual_igc
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_firmware import (
    install_semantic_internal_firmware,
    install_semantic_learned_control,
)
from neural_firmware.semantic_training import (
    generate_base_semantic,
    generate_semantic_control,
    generate_semantic_internal,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROTOCOL_COMMIT = "b90ae8c"
TRAINING_MANIFEST = Path("phase5_results/training_v2/manifest.json")
OUTPUT_DIRECTORY = Path("phase5_results/confirmation_raw_v1")
SEEDS = (10_701, 10_702, 10_703)
CONDITIONS = ("typed_firmware", "adapter", "igc_matched", "igc_native")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_training_records() -> dict[tuple[str, int], dict[str, object]]:
    manifest = json.loads(TRAINING_MANIFEST.read_text())
    records = {
        (str(row["condition"]), int(row["seed"])): row
        for row in manifest["records"]
    }
    for key, record in records.items():
        checkpoint = Path(record["checkpoint"])
        observed = sha256(checkpoint)
        if observed != record["checkpoint_sha256"]:
            raise RuntimeError(
                f"checkpoint hash mismatch for {key}: {observed}"
            )
    return records


def latency_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    values = [float(row["latency_seconds"]) for row in rows]
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def summarize_condition(
    positive_rows: list[dict[str, object]],
    negative_rows: list[dict[str, object]],
    base_negative_rows: list[dict[str, object]],
) -> dict[str, object]:
    active_positive = [row for row in positive_rows if row["route_active"] is True]
    active_negative = [row for row in negative_rows if row["route_active"] is True]
    register_rows = [
        row for row in positive_rows if row.get("registers_exact") is not None
    ]
    exact_register_rows = [
        row for row in register_rows if row["registers_exact"] is True
    ]
    eligible_igc = [
        row
        for row in register_rows
        if row["route_active"] is True and row["registers_exact"] is True
    ]
    preserved = sum(
        row["generated_token_ids"] == base["generated_token_ids"]
        for row, base in zip(negative_rows, base_negative_rows, strict=True)
    )
    summary: dict[str, object] = {
        "positive_examples": len(positive_rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positive_rows
        ),
        "mathematical_accuracy": sum(
            row["mathematical_correct"] is True for row in positive_rows
        )
        / len(positive_rows),
        "exact_format_correct": sum(
            row["exact_format_correct"] is True for row in positive_rows
        ),
        "exact_format_accuracy": sum(
            row["exact_format_correct"] is True for row in positive_rows
        )
        / len(positive_rows),
        "route_true_positive": len(active_positive),
        "route_true_positive_rate": len(active_positive) / len(positive_rows),
        "active_positive_mathematical_correct": sum(
            row["mathematical_correct"] is True for row in active_positive
        ),
        "active_positive_mathematical_accuracy": (
            sum(row["mathematical_correct"] is True for row in active_positive)
            / len(active_positive)
            if active_positive
            else None
        ),
        "negative_examples": len(negative_rows),
        "route_false_positive": len(active_negative),
        "route_false_positive_rate": len(active_negative) / len(negative_rows),
        "token_exact_preserved": preserved,
        "token_exact_preservation_rate": preserved / len(negative_rows),
        "positive_latency": latency_summary(positive_rows),
        "negative_latency": latency_summary(negative_rows),
    }
    if register_rows:
        summary.update(
            {
                "register_examples": len(register_rows),
                "registers_exact": len(exact_register_rows),
                "register_accuracy": len(exact_register_rows)
                / len(register_rows),
                "route_and_register_eligible": len(eligible_igc),
                "eligible_mathematical_correct": sum(
                    row["mathematical_correct"] is True for row in eligible_igc
                ),
                "eligible_mathematical_accuracy": (
                    sum(
                        row["mathematical_correct"] is True
                        for row in eligible_igc
                    )
                    / len(eligible_igc)
                    if eligible_igc
                    else None
                ),
            }
        )
    return summary


def summarize_base(
    positive_rows: list[dict[str, object]],
    negative_rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "positive_examples": len(positive_rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in positive_rows
        ),
        "mathematical_accuracy": sum(
            row["mathematical_correct"] is True for row in positive_rows
        )
        / len(positive_rows),
        "exact_format_correct": sum(
            row["exact_format_correct"] is True for row in positive_rows
        ),
        "exact_format_accuracy": sum(
            row["exact_format_correct"] is True for row in positive_rows
        )
        / len(positive_rows),
        "positive_latency": latency_summary(positive_rows),
        "negative_latency": latency_summary(negative_rows),
    }


def condition_generator(
    condition: str,
    record: dict[str, object],
) -> tuple[object, object, object]:
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    checkpoint = torch.load(
        record["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    threshold = float(record["selected_route_threshold"]["threshold"])
    if condition == "typed_firmware":
        wrapper = install_semantic_internal_firmware(
            bundle.model,
            depth_after_blocks=24,
            strength=64.0,
            router_hidden_width=16,
        )
        wrapper.unit.load_state_dict(checkpoint)

        def generate(example: object, max_new_tokens: int) -> dict[str, object]:
            return generate_semantic_internal(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=max_new_tokens,
            )

    elif condition == "adapter":
        wrapper = install_semantic_learned_control(
            bundle.model,
            depth_after_blocks=24,
            rank=5,
            router_hidden_width=16,
        )
        wrapper.router.load_state_dict(checkpoint["router"])
        wrapper.adapter.load_state_dict(checkpoint["adapter"])

        def generate(example: object, max_new_tokens: int) -> dict[str, object]:
            return generate_semantic_control(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=max_new_tokens,
            )

    elif condition in {"igc_matched", "igc_native"}:
        architecture = record["architecture"]
        wrapper = install_dual_depth_igc(
            bundle.model,
            input_depth_after_blocks=1,
            output_depth_after_blocks=24,
            max_digits=12,
            attention_width=int(architecture["attention_width"]),
            attention_heads=int(architecture["attention_heads"]),
            output_width=int(architecture["output_width"]),
            router_hidden_width=int(architecture["router_hidden_width"]),
            initial_strength=64.0,
            learn_output_strength=bool(architecture["learn_output_strength"]),
        )
        wrapper.load_state_dict(checkpoint)

        def generate(example: object, max_new_tokens: int) -> dict[str, object]:
            return generate_dual_igc(
                bundle,
                wrapper,
                example,
                route_mode="learned",
                route_threshold=threshold,
                max_new_tokens=max_new_tokens,
            )

    else:
        raise ValueError(f"unknown condition: {condition}")
    return bundle, wrapper, generate


def main() -> None:
    started = time.perf_counter()
    commit_at_start = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        check=False,
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"protocol commit {PROTOCOL_COMMIT} is not an ancestor of "
            f"execution commit {commit_at_start}"
        )
    records = load_training_records()
    positive_sets = build_phase5_confirmatory_positive_sets()
    positive_examples = [
        example for rows in positive_sets.values() for example in rows
    ]
    negative_examples = build_phase5_confirmatory_negatives()
    rendered_data = {
        "positive": {
            split: [example.to_dict() for example in examples]
            for split, examples in positive_sets.items()
        },
        "negative": [example.to_dict() for example in negative_examples],
    }
    rendered_path = OUTPUT_DIRECTORY / "rendered_data.json"
    if not rendered_path.exists():
        write_json_atomic(rendered_path, rendered_data)
    elif json.loads(rendered_path.read_text()) != rendered_data:
        raise RuntimeError("rendered confirmatory data changed")

    warmup = build_phase5_development_examples(
        positive_count=1,
        negative_count=1,
    )[0]
    base_path = OUTPUT_DIRECTORY / "base.json"
    if base_path.exists():
        base_result = json.loads(base_path.read_text())
    else:
        bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
        generate_base_semantic(bundle, warmup, max_new_tokens=24)
        positive_rows = [
            generate_base_semantic(bundle, example, max_new_tokens=24)
            for example in positive_examples
        ]
        negative_rows = [
            generate_base_semantic(bundle, example, max_new_tokens=20)
            for example in negative_examples
        ]
        base_result = {
            "condition": "base",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "summary": summarize_base(positive_rows, negative_rows),
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
        }
        write_json_atomic(base_path, base_result)
        del bundle
        release_memory()
    base_negative_rows = base_result["negative_rows"]

    completed: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            output_path = OUTPUT_DIRECTORY / f"{condition}_seed_{seed}.json"
            if output_path.exists():
                result = json.loads(output_path.read_text())
                completed.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "path": str(output_path),
                        "sha256": sha256(output_path),
                        "summary": result["summary"],
                    }
                )
                continue
            record = records[(condition, seed)]
            bundle, wrapper, generate = condition_generator(condition, record)
            generate(warmup, 24)
            positive_rows = [
                generate(example, 24) for example in positive_examples
            ]
            negative_rows = [
                generate(example, 20) for example in negative_examples
            ]
            result = {
                "condition": condition,
                "seed": seed,
                "learned_parameters": record["learned_parameters"],
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "route_threshold": record["selected_route_threshold"][
                    "threshold"
                ],
                "summary": summarize_condition(
                    positive_rows,
                    negative_rows,
                    base_negative_rows,
                ),
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
            }
            write_json_atomic(output_path, result)
            completed.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                    "summary": result["summary"],
                }
            )
            del wrapper
            del bundle
            release_memory()

    manifest = {
        "status": "confirmation_complete_raw",
        "protocol": "PHASE5_CONFIRMATORY_PROTOCOL.md",
        "protocol_commit": PROTOCOL_COMMIT,
        "git_commit_at_start": commit_at_start,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "rendered_data": str(rendered_path),
        "rendered_data_sha256": sha256(rendered_path),
        "base": {
            "path": str(base_path),
            "sha256": sha256(base_path),
            "summary": base_result["summary"],
        },
        "condition_runs": completed,
        "wall_time_seconds": time.perf_counter() - started,
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "base": manifest["base"]["summary"],
                "condition_runs": [
                    {
                        "condition": row["condition"],
                        "seed": row["seed"],
                        "summary": row["summary"],
                    }
                    for row in completed
                ],
                "wall_time_seconds": manifest["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
