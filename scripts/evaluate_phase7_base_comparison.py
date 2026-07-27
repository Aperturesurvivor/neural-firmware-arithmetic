from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import generate_untouched_sequence
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct, mathematical_correct

SEEDS = (13_201, 13_202, 13_203)
AUDIT_PATHS = {
    seed: Path(f"phase7_results/operand_register_audit5_seed_{seed}.json")
    for seed in SEEDS
}
CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_layer16_router_hardened_v1/"
    "neuron_implant_seed_13202.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "b483c3fbcec274cdf2f1b23acff33ae63966575c4d0f491eed3f182a73f24eea"
)
RESULT_PATH = Path("phase7_results/base_vs_implant_audit5.json")
CSV_PATH = Path("phase7_results/base_vs_implant_audit5_rows.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_audits() -> dict[int, dict[str, object]]:
    audits = {
        seed: json.loads(path.read_text()) for seed, path in AUDIT_PATHS.items()
    }
    reference = audits[SEEDS[0]]["rows"]
    identity = [
        (row["split"], row["prompt"], row["answer"], row["route_label"])
        for row in reference
    ]
    for seed in SEEDS[1:]:
        candidate = [
            (row["split"], row["prompt"], row["answer"], row["route_label"])
            for row in audits[seed]["rows"]
        ]
        if candidate != identity:
            raise ValueError("audit-5 prompt rows differ across interface seeds")
    return audits


def paired_summary(
    rows: list[dict[str, object]],
    seed: int,
    *,
    base_key: str,
    metric: str,
) -> dict[str, int]:
    positives = [row for row in rows if row["route_label"]]
    both = sum(
        row[base_key][metric]
        and row["implants"][str(seed)][metric]
        for row in positives
    )
    implant_only = sum(
        not row[base_key][metric]
        and row["implants"][str(seed)][metric]
        for row in positives
    )
    base_only = sum(
        row[base_key][metric]
        and not row["implants"][str(seed)][metric]
        for row in positives
    )
    neither = len(positives) - both - implant_only - base_only
    return {
        "both_correct": both,
        "implant_only_correct": implant_only,
        "base_only_correct": base_only,
        "neither_correct": neither,
        "paired_net_gain": implant_only - base_only,
    }


def subset_summary(
    rows: list[dict[str, object]],
    *,
    split: str,
) -> dict[str, object]:
    subset = [row for row in rows if row["split"] == split]
    return {
        "examples": len(subset),
        "base_mathematical_exact": sum(
            row["base"]["mathematical_correct"] for row in subset
        ),
        "base_format_exact": sum(
            row["base"]["format_correct"] for row in subset
        ),
        "base_extended_mathematical_exact": sum(
            row["base_extended"]["mathematical_correct"] for row in subset
        ),
        "base_extended_format_exact": sum(
            row["base_extended"]["format_correct"] for row in subset
        ),
        "implants_mathematical_exact": {
            str(seed): sum(
                row["implants"][str(seed)]["mathematical_correct"]
                for row in subset
            )
            for seed in SEEDS
        },
        "implants_format_exact": {
            str(seed): sum(
                row["implants"][str(seed)]["format_correct"]
                for row in subset
            )
            for seed in SEEDS
        },
    }


def make_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    per_seed = {}
    for seed in SEEDS:
        key = str(seed)
        exact = sum(
            row["implants"][key]["mathematical_correct"] for row in positives
        )
        per_seed[key] = {
            "examples": len(positives),
            "mathematical_exact": exact,
            "format_exact": sum(
                row["implants"][key]["format_correct"] for row in positives
            ),
            "accuracy": exact / len(positives),
            "paired_vs_base_same_budget_exact_response": paired_summary(
                rows,
                seed,
                base_key="base",
                metric="format_correct",
            ),
            "paired_vs_base_same_budget_last_number": paired_summary(
                rows,
                seed,
                base_key="base",
                metric="mathematical_correct",
            ),
            "paired_vs_base_extended_last_number": paired_summary(
                rows,
                seed,
                base_key="base_extended",
                metric="mathematical_correct",
            ),
        }
    base_exact = sum(row["base"]["mathematical_correct"] for row in positives)
    base_extended_exact = sum(
        row["base_extended"]["mathematical_correct"] for row in positives
    )
    pooled_implant_exact = sum(
        value["mathematical_exact"] for value in per_seed.values()
    )
    return {
        "unique_prompts": len(rows),
        "addition_prompts": len(positives),
        "negative_prompts": len(negatives),
        "base": {
            "examples": len(positives),
            "mathematical_exact": base_exact,
            "format_exact": sum(
                row["base"]["format_correct"] for row in positives
            ),
            "accuracy": base_exact / len(positives),
        },
        "base_extended_sensitivity": {
            "examples": len(positives),
            "mathematical_exact": base_extended_exact,
            "format_exact": sum(
                row["base_extended"]["format_correct"] for row in positives
            ),
            "accuracy": base_extended_exact / len(positives),
            "max_new_tokens": 64,
        },
        "implants": {
            "per_seed": per_seed,
            "pooled_seed_prompt_attempts": len(positives) * len(SEEDS),
            "pooled_mathematical_exact": pooled_implant_exact,
            "pooled_accuracy": pooled_implant_exact
            / (len(positives) * len(SEEDS)),
            "mean_accuracy": sum(
                value["accuracy"] for value in per_seed.values()
            )
            / len(SEEDS),
        },
        "by_split": {
            split: subset_summary(rows, split=split)
            for split in ("phase7_audit5_symbolic", "phase7_audit5_word")
        },
        "negative_preservation": {
            "unique_prompts": len(negatives),
            "per_seed_token_identical_to_base": {
                str(seed): sum(
                    row["implants"][str(seed)]["token_identical_to_base"]
                    for row in negatives
                )
                for seed in SEEDS
            },
            "pooled_token_identical_to_base": sum(
                row["implants"][str(seed)]["token_identical_to_base"]
                for row in negatives
                for seed in SEEDS
            ),
        },
    }


def write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "row_index",
        "split",
        "prompt",
        "expected_answer",
        "base_output",
        "base_mathematical_correct",
        "base_format_correct",
        "base_extended_output",
        "base_extended_mathematical_correct",
        "base_extended_format_correct",
    ]
    for seed in SEEDS:
        fields.extend(
            [
                f"seed_{seed}_output",
                f"seed_{seed}_mathematical_correct",
                f"seed_{seed}_format_correct",
                f"seed_{seed}_route_active",
                f"seed_{seed}_token_identical_to_base",
            ]
        )
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat: dict[str, object] = {
                "row_index": row["row_index"],
                "split": row["split"],
                "prompt": row["prompt"],
                "expected_answer": row["answer"] or "",
                "base_output": row["base"]["generated_text"],
                "base_mathematical_correct": row["base"][
                    "mathematical_correct"
                ],
                "base_format_correct": row["base"]["format_correct"],
                "base_extended_output": (
                    row["base_extended"]["generated_text"]
                    if row["route_label"]
                    else ""
                ),
                "base_extended_mathematical_correct": (
                    row["base_extended"]["mathematical_correct"]
                    if row["route_label"]
                    else ""
                ),
                "base_extended_format_correct": (
                    row["base_extended"]["format_correct"]
                    if row["route_label"]
                    else ""
                ),
            }
            for seed in SEEDS:
                record = row["implants"][str(seed)]
                flat.update(
                    {
                        f"seed_{seed}_output": record["generated_text"],
                        f"seed_{seed}_mathematical_correct": record[
                            "mathematical_correct"
                        ],
                        f"seed_{seed}_format_correct": record["format_correct"],
                        f"seed_{seed}_route_active": record["route_active"],
                        f"seed_{seed}_token_identical_to_base": record[
                            "token_identical_to_base"
                        ],
                    }
                )
            writer.writerow(flat)


def main() -> None:
    audits = load_audits()
    checkpoint_hash = sha256(CHECKPOINT_PATH)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen comparison checkpoint hash mismatch")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    bundle = load_model_bundle(
        checkpoint["model_id"],
        revision=checkpoint["model_revision"],
    )
    layout = SequenceImplantLayout(**checkpoint["layout"])
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=checkpoint["layer_index"],
        selected_indices=checkpoint["selected_indices"],
        layout=layout,
        output_strength=checkpoint["output_strength"],
        route_threshold=checkpoint["route_threshold"],
        digit_threshold=checkpoint.get("digit_threshold", 0.0),
        use_swiglu_interface=checkpoint.get("use_swiglu_interface", False),
    )
    with torch.no_grad():
        implant.input_rows.copy_(checkpoint["input_rows"].to(bundle.device))
        if "gate_rows" in checkpoint:
            implant.gate_rows.copy_(checkpoint["gate_rows"].to(bundle.device))
        implant.result_columns.copy_(
            checkpoint["result_columns"].to(bundle.device)
        )

    source_rows = audits[SEEDS[0]]["rows"]
    completed: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, source in enumerate(source_rows):
        base = generate_untouched_sequence(
            bundle,
            implant,
            source["prompt"],
            layer_index=checkpoint["layer_index"],
            max_new_tokens=8,
        )
        is_positive = bool(source["route_label"])
        answer = source["answer"] or ""
        base_record = {
            "generated_text": base["generated_text"],
            "generated_token_ids": base["generated_token_ids"],
            "mathematical_correct": (
                mathematical_correct(base["generated_text"], answer)
                if is_positive
                else False
            ),
            "format_correct": (
                exact_format_correct(base["generated_text"], answer)
                if is_positive
                else False
            ),
        }
        if is_positive:
            extended = generate_untouched_sequence(
                bundle,
                implant,
                source["prompt"],
                layer_index=checkpoint["layer_index"],
                max_new_tokens=64,
            )
            base_extended_record = {
                "generated_text": extended["generated_text"],
                "generated_token_ids": extended["generated_token_ids"],
                "mathematical_correct": mathematical_correct(
                    extended["generated_text"],
                    answer,
                ),
                "format_correct": exact_format_correct(
                    extended["generated_text"],
                    answer,
                ),
            }
        else:
            base_extended_record = {
                "generated_text": "",
                "generated_token_ids": [],
                "mathematical_correct": False,
                "format_correct": False,
            }
        implants: dict[str, object] = {}
        for seed in SEEDS:
            audit_row = audits[seed]["rows"][index]
            result = audit_row["implant"]
            implants[str(seed)] = {
                "generated_text": result["generated_text"],
                "generated_token_ids": result["generated_token_ids"],
                "mathematical_correct": bool(
                    audit_row["mathematical_correct"]
                ),
                "format_correct": bool(audit_row["format_correct"]),
                "route_active": bool(audit_row["any_route_active"]),
                "first_step_operands_exact": bool(
                    audit_row["first_step_operands_exact"]
                ),
                "calculator_trajectory_exact": bool(
                    audit_row["calculator_trajectory_exact"]
                ),
                "token_identical_to_base": (
                    result["generated_token_ids"]
                    == base["generated_token_ids"]
                ),
            }
        completed.append(
            {
                "row_index": index,
                "split": source["split"],
                "prompt": source["prompt"],
                "answer": source["answer"],
                "route_label": is_positive,
                "base": base_record,
                "base_extended": base_extended_record,
                "implants": implants,
            }
        )
        print(
            f"base={index + 1}/{len(source_rows)} "
            f"split={source['split']} text={base['generated_text']!r}",
            flush=True,
        )
        if bundle.device.type == "mps":
            torch.mps.empty_cache()

    payload = {
        "status": "complete",
        "evaluation_kind": (
            "retrospective_untouched_base_benchmark_on_frozen_audit5"
        ),
        "protocol_status": (
            "posthoc comparison; the prompt set and implant outputs were "
            "already frozen, but this base benchmark was not preregistered"
        ),
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "decoding": {
            "method": "greedy",
            "max_new_tokens": 8,
            "chat_prompt_format": "identical to phase 7 audit 5",
            "base_only_sensitivity_max_new_tokens": 64,
        },
        "source_files": {
            str(seed): {
                "path": str(AUDIT_PATHS[seed]),
                "sha256": sha256(AUDIT_PATHS[seed]),
            }
            for seed in SEEDS
        },
        "base_reference_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "sha256": checkpoint_hash,
            "note": (
                "The implant was temporarily uninstalled for every base "
                "generation, restoring the original layer-16 Qwen MLP."
            ),
        },
        "summary": make_summary(completed),
        "rows": completed,
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json(RESULT_PATH, payload)
    write_csv(completed)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
