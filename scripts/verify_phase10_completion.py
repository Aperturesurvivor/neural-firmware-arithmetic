from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import torch

MANIFEST = Path("phase10_results/frozen_prompt_manifest.json")
TRAINING = Path("phase10_results/confirmatory_interface_training.json")
CONFIRMATION = Path("phase10_results/confirmation.json")
AUDIT = Path("phase10_results/completion_audit.json")
CONDITIONS = ("linear", "nonlinear", "linear_representation")
SEEDS = (16_201, 16_202, 16_203)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, value: bool, checks: list[dict[str, object]]) -> None:
    checks.append({"name": name, "passed": bool(value)})


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    training = json.loads(TRAINING.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    checks: list[dict[str, object]] = []
    check(
        "manifest_status",
        manifest["status"] == "phase10_protocol_frozen_before_confirmation",
        checks,
    )
    check(
        "training_status",
        training["status"] == "phase10_confirmatory_interface_training_complete",
        checks,
    )
    check(
        "confirmation_status",
        confirmation["status"] == "phase10_confirmatory_evaluation_complete",
        checks,
    )
    canonical = json.dumps(
        manifest["rows"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    check(
        "canonical_prompt_hash",
        hashlib.sha256(canonical).hexdigest()
        == manifest["canonical_rows_sha256"],
        checks,
    )
    check("manifest_prompt_count", len(manifest["rows"]) == 300, checks)
    check("confirmation_row_count", len(confirmation["rows"]) == 300, checks)
    check(
        "evaluated_rows_match_manifest",
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
            for row in confirmation["rows"]
        ]
        == manifest["rows"],
        checks,
    )
    training_records = {
        (record["condition"], record["phase10_seed"]): record
        for record in training["records"]
    }
    check("training_record_count", len(training_records) == 9, checks)
    expected_parameters = {
        "linear": 57_344,
        "nonlinear": 57_344,
        "linear_representation": 73_728,
    }
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = str(seed)
            record = training_records[(condition, seed)]
            checkpoint_path = Path(record["checkpoint"])
            source_path = Path(record["source_checkpoint"])
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            source = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            label = f"{condition}_{seed}"
            check(
                f"{label}_checkpoint_hash",
                sha256(checkpoint_path) == record["checkpoint_sha256"],
                checks,
            )
            check(
                f"{label}_source_hash",
                sha256(source_path) == record["source_checkpoint_sha256"],
                checks,
            )
            check(
                f"{label}_result_decoder_unchanged",
                torch.equal(
                    checkpoint["result_columns"],
                    source["result_columns"],
                ),
                checks,
            )
            check(
                f"{label}_route_temperature",
                checkpoint["route_temperature"] == 2.0,
                checks,
            )
            check(
                f"{label}_interface_local",
                checkpoint["adapt_base_mlp"] is False,
                checks,
            )
            check(
                f"{label}_parameter_count",
                confirmation["checkpoints"][condition][key][
                    "architectural_learned_parameters"
                ]
                == expected_parameters[condition],
                checks,
            )
            check(
                f"{label}_calculator_zero_parameter",
                confirmation["checkpoints"][condition][key][
                    "calculator_learned_parameters"
                ]
                == 0,
                checks,
            )

            positive_records = [
                row["conditions"][condition][key]
                for row in confirmation["rows"]
                if row["route_label"]
            ]
            negative_records = [
                row["conditions"][condition][key]
                for row in confirmation["rows"]
                if not row["route_label"]
            ]
            reported = confirmation["conditions"][condition][key]
            recomputed = {
                "exact": sum(row["format_exact"] for row in positive_records),
                "positive_routes": sum(
                    row["first_route"] for row in positive_records
                ),
                "positive_active_routes": sum(
                    row["first_route_active"] for row in positive_records
                ),
                "operands_exact": sum(
                    row["operands_exact"] for row in positive_records
                ),
                "trajectories_exact": sum(
                    row["trajectory_exact"] for row in positive_records
                ),
                "false_routes": sum(
                    row["first_route"] for row in negative_records
                ),
                "token_preserved": sum(
                    row["token_preserved"] for row in negative_records
                ),
                "oracle_exact": sum(
                    row["oracle_route"]["format_exact"]
                    for row in positive_records
                ),
                "oracle_operands_exact": sum(
                    row["oracle_route"]["operands_exact"]
                    for row in positive_records
                ),
                "oracle_trajectories_exact": sum(
                    row["oracle_route"]["trajectory_exact"]
                    for row in positive_records
                ),
            }
            if condition == "linear_representation":
                recomputed["ablation_exact"] = sum(
                    row["ablation"]["format_exact"]
                    for row in positive_records
                )
                recomputed["paired_causal_losses"] = sum(
                    row["format_exact"]
                    and not row["ablation"]["format_exact"]
                    for row in positive_records
                )
            check(
                f"{label}_metrics_recompute",
                all(reported[name] == value for name, value in recomputed.items()),
                checks,
            )
            conditional = [
                row
                for row in positive_records
                if row["first_route_active"] and row["operands_exact"]
            ]
            check(
                f"{label}_conditional_mechanism",
                all(
                    row["format_exact"] and row["trajectory_exact"]
                    for row in conditional
                ),
                checks,
            )

    representation_gains = [
        confirmation["conditions"]["linear_representation"][str(seed)]["exact"]
        - confirmation["conditions"]["linear"][str(seed)]["exact"]
        for seed in SEEDS
    ]
    nonlinear_gains = [
        confirmation["conditions"]["nonlinear"][str(seed)]["exact"]
        - confirmation["conditions"]["linear"][str(seed)]["exact"]
        for seed in SEEDS
    ]
    recomputed_representation_gates = {
        "paired_end_to_end_benefit": (
            all(gain > 0 for gain in representation_gains)
            and statistics.fmean(representation_gains) >= 5
        ),
        "preservation": all(
            confirmation["conditions"]["linear_representation"][str(seed)][
                "false_routes"
            ]
            <= 4
            and confirmation["conditions"]["linear_representation"][str(seed)][
                "token_preserved"
            ]
            >= 196
            for seed in SEEDS
        ),
        "operand_access": all(
            confirmation["conditions"]["linear_representation"][str(seed)][
                "oracle_exact"
            ]
            >= 85
            and confirmation["conditions"]["linear_representation"][str(seed)][
                "oracle_exact"
            ]
            >= confirmation["conditions"]["linear"][str(seed)]["oracle_exact"]
            for seed in SEEDS
        ),
        "conditional_mechanism": all(
            confirmation["conditions"]["linear_representation"][str(seed)][
                "conditional_exact"
            ]
            == confirmation["conditions"]["linear_representation"][str(seed)][
                "conditional_examples"
            ]
            and confirmation["conditions"]["linear_representation"][str(seed)][
                "conditional_trajectories_exact"
            ]
            == confirmation["conditions"]["linear_representation"][str(seed)][
                "conditional_examples"
            ]
            for seed in SEEDS
        ),
        "causal_ablation": all(
            confirmation["conditions"]["linear_representation"][str(seed)][
                "paired_causal_losses"
            ]
            == confirmation["conditions"]["linear_representation"][str(seed)][
                "exact"
            ]
            and confirmation["conditions"]["linear_representation"][str(seed)][
                "ablation_exact"
            ]
            <= 5
            for seed in SEEDS
        ),
    }
    recomputed_representation_gates["all_gates"] = all(
        recomputed_representation_gates.values()
    )
    recomputed_nonlinear_gate = (
        all(gain > 0 for gain in nonlinear_gains)
        and statistics.fmean(nonlinear_gains) > 0
        and all(
            confirmation["conditions"]["nonlinear"][str(seed)]["false_routes"]
            <= confirmation["conditions"]["linear"][str(seed)]["false_routes"]
            for seed in SEEDS
        )
    )
    check(
        "paired_representation_gains",
        representation_gains == confirmation["paired_representation_gains"],
        checks,
    )
    check(
        "paired_nonlinear_gains",
        nonlinear_gains == confirmation["paired_nonlinear_gains"],
        checks,
    )
    check(
        "representation_gates_recompute",
        recomputed_representation_gates == confirmation["representation_gates"],
        checks,
    )
    check(
        "nonlinear_gate_recompute",
        recomputed_nonlinear_gate == confirmation["nonlinear_gate"],
        checks,
    )
    passed = all(item["passed"] for item in checks)
    payload = {
        "status": (
            "phase10_completion_audit_passed"
            if passed
            else "phase10_completion_audit_failed"
        ),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "manifest_sha256": sha256(MANIFEST),
        "training_sha256": sha256(TRAINING),
        "confirmation_sha256": sha256(CONFIRMATION),
        "representation_gains": representation_gains,
        "nonlinear_gains": nonlinear_gains,
        "checks": checks,
    }
    AUDIT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "checks"},
            indent=2,
        )
    )
    if not passed:
        raise SystemExit("Phase 10 completion audit failed")


if __name__ == "__main__":
    main()
