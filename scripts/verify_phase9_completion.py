from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from analyze_phase9_confirmation import canonical_rows_sha256

MANIFEST = Path("phase9_results/frozen_prompt_manifest.json")
TRAINING = Path("phase9_results/confirmatory_interface_training.json")
CONFIRMATION = Path("phase9_results/confirmation.json")
CONFIRMATION_CSV = Path("phase9_results/confirmation_rows.csv")
ANALYSIS = Path("phase9_results/analysis.json")
AUDIT = Path("phase9_results/completion_audit.json")
SUMMARY = Path("PHASE9_EXECUTIVE_SUMMARY.md")
REPORT = Path("paper_phase9/deterministic-neurons-interface-hardening.pdf")
FIGURES = (
    Path("phase9_figures/condition_accuracy.png"),
    Path("phase9_figures/negative_false_routes.png"),
    Path("phase9_figures/hard_failure_taxonomy.png"),
    Path("phase9_figures/hard_false_routes_by_family.png"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(
    checks: dict[str, bool],
    name: str,
    condition: bool,
) -> None:
    checks[name] = bool(condition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-report", action="store_true")
    arguments = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    training = json.loads(TRAINING.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    seeds = [str(seed) for seed in manifest["phase9_seeds"]]
    checks: dict[str, bool] = {}

    check(
        checks,
        "manifest_frozen_status",
        manifest["status"] == "phase9_protocol_frozen_before_confirmation",
    )
    check(checks, "manifest_unique_prompts", manifest["unique_prompts"] == 300)
    check(checks, "manifest_positive_prompts", manifest["positive_prompts"] == 100)
    check(checks, "manifest_negative_prompts", manifest["negative_prompts"] == 200)
    check(
        checks,
        "manifest_canonical_hash",
        canonical_rows_sha256(manifest["rows"])
        == manifest["canonical_rows_sha256"],
    )

    records = training["records"]
    check(
        checks,
        "six_training_records",
        len(records) == 6
        and {
            (record["condition"], str(record["phase9_seed"]))
            for record in records
        }
        == {
            (condition, seed)
            for condition in ("generic", "hard")
            for seed in seeds
        },
    )
    checkpoint_verification: list[dict[str, object]] = []
    for record in records:
        checkpoint_path = Path(record["checkpoint"])
        source_path = Path(record["source_checkpoint"])
        phase9_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        source_checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=True,
        )
        result_unchanged = torch.equal(
            phase9_checkpoint["result_columns"],
            source_checkpoint["result_columns"],
        )
        entry = {
            "condition": record["condition"],
            "phase9_seed": record["phase9_seed"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256_matches": (
                sha256(checkpoint_path) == record["checkpoint_sha256"]
            ),
            "source_checkpoint_sha256_matches": (
                sha256(source_path) == record["source_checkpoint_sha256"]
            ),
            "result_columns_bit_identical": result_unchanged,
            "input_weights": phase9_checkpoint["input_rows"].numel(),
            "result_weights": phase9_checkpoint["result_columns"].numel(),
            "phase9_updated_parameters": phase9_checkpoint[
                "phase9_updated_parameters"
            ],
            "architectural_learned_parameters": phase9_checkpoint[
                "architectural_learned_parameters"
            ],
        }
        checkpoint_verification.append(entry)
    check(
        checks,
        "all_checkpoint_hashes_match",
        all(
            entry["checkpoint_sha256_matches"]
            and entry["source_checkpoint_sha256_matches"]
            for entry in checkpoint_verification
        ),
    )
    check(
        checks,
        "all_result_columns_bit_identical",
        all(
            entry["result_columns_bit_identical"]
            for entry in checkpoint_verification
        ),
    )
    check(
        checks,
        "all_checkpoint_parameter_counts_exact",
        all(
            entry["input_weights"] == 32_768
            and entry["result_weights"] == 24_576
            and entry["phase9_updated_parameters"] == 32_768
            and entry["architectural_learned_parameters"] == 57_344
            for entry in checkpoint_verification
        ),
    )

    rows = confirmation["rows"]
    check(
        checks,
        "confirmation_complete_status",
        confirmation["status"] == "phase9_confirmatory_evaluation_complete",
    )
    check(checks, "confirmation_has_300_rows", len(rows) == 300)
    check(
        checks,
        "confirmation_matches_frozen_manifest",
        [
            {
                key: row[key]
                for key in (
                    "prompt",
                    "a",
                    "b",
                    "answer",
                    "route_label",
                    "family",
                    "family_index",
                    "split",
                )
            }
            for row in rows
        ]
        == manifest["rows"],
    )
    check(
        checks,
        "all_rows_have_every_condition_and_seed",
        all(
            all(
                set(row["implants"][condition]) == set(seeds)
                for condition in ("phase8_frozen", "generic", "hard")
            )
            and set(row["matched_adapters"]) == set(seeds)
            and row["base"] is not None
            for row in rows
        ),
    )
    check(
        checks,
        "confirmation_parameter_counts",
        confirmation["parameter_counts"]
        == {
            "base_model": 1_100_048_384,
            "implant_architectural_learned": 57_344,
            "phase9_updated": 32_768,
            "matched_adapter_learned": 57_344,
            "calculator_learned": 0,
        },
    )
    expected_gate_keys = {
        "accuracy",
        "operands",
        "routing_and_preservation",
        "conditional_calculator_and_decode",
        "causal_ablation",
        "hard_beats_generic",
        "all_primary_gates",
    }
    check(
        checks,
        "all_frozen_gates_reported",
        set(confirmation["primary_hard_gates"]) == expected_gate_keys,
    )

    with CONFIRMATION_CSV.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    check(checks, "confirmation_csv_has_300_rows", len(csv_rows) == 300)
    check(
        checks,
        "confirmation_csv_order_matches_json",
        [row["prompt"] for row in csv_rows]
        == [row["prompt"] for row in rows],
    )

    verification = analysis["verification"]
    check(
        checks,
        "analysis_complete_and_verified",
        analysis["status"] == "phase9_posthoc_analysis_complete"
        and verification["evaluation_status_complete"]
        and verification["manifest_status_frozen"]
        and verification["prompt_rows_exactly_match_manifest"]
        and verification["canonical_hash_matches"],
    )
    check(
        checks,
        "analysis_raw_gates_match_confirmation",
        analysis["implementation_reported_primary_gates"]
        == confirmation["primary_hard_gates"],
    )
    check(
        checks,
        "analysis_protocol_verdict_matches_raw_compound_verdict",
        (analysis["compound_verdict"] == "PASS")
        == confirmation["primary_hard_gates"]["all_primary_gates"],
    )
    check(
        checks,
        "conditional_gate_discrepancy_disclosed",
        not analysis["gate_discrepancy"]["implementation_reported"]
        and analysis["gate_discrepancy"]["protocol_recomputed"]
        and analysis["protocol_recomputed_primary_gates"][
            "conditional_calculator_and_decode"
        ]
        and not analysis["protocol_recomputed_primary_gates"][
            "all_primary_gates"
        ],
    )

    final_artifacts = (SUMMARY, REPORT, *FIGURES)
    if arguments.require_report:
        check(
            checks,
            "all_final_reader_artifacts_exist",
            all(path.is_file() and path.stat().st_size > 0 for path in final_artifacts),
        )
    artifact_hashes = {
        str(path): sha256(path)
        for path in (
            MANIFEST,
            TRAINING,
            CONFIRMATION,
            CONFIRMATION_CSV,
            ANALYSIS,
            *(
                final_artifacts
                if arguments.require_report
                else tuple(path for path in final_artifacts if path.is_file())
            ),
        )
    }
    result = {
        "status": (
            "phase9_completion_audit_passed"
            if all(checks.values())
            else "phase9_completion_audit_failed"
        ),
        "require_report": arguments.require_report,
        "checks": checks,
        "checkpoint_verification": checkpoint_verification,
        "artifact_sha256": artifact_hashes,
    }
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 9 completion audit failed: {failed}")


if __name__ == "__main__":
    main()
