from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import torch

from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    operand_pairs,
)
from neural_firmware.phase9_data import phase9_family_sets
from neural_firmware.phase10_data import phase10_family_set
from neural_firmware.phase11_data import phase11_family_set
from neural_firmware.phase12_data import (
    _all_pairs_before_phase12,
    build_phase12_confirmatory_examples,
    phase12_family_set,
)

MANIFEST = Path("phase12_results/frozen_prompt_manifest.json")
CONFIRMATION = Path("phase12_results/confirmation.json")
ANALYSIS = Path("phase12_results/analysis.json")
PROGRESS = Path("phase12_artifacts/confirmation_progress.json")
AUDIT = Path("phase12_results/verification.json")
PROTOCOL = Path("PHASE12_MULTI_VIEW_ROUTING_PROTOCOL.md")
EVALUATOR = Path("scripts/evaluate_phase12_confirmation.py")
SEEDS = (16_201, 16_202, 16_203)
EXPECTED_SPLITS = {
    "phase12_confirmatory_positive_direct": 50,
    "phase12_confirmatory_positive_word": 25,
    "phase12_confirmatory_positive_distractor": 25,
    "phase12_confirmatory_negative_multiplication": 20,
    "phase12_confirmatory_negative_factual": 20,
    "phase12_confirmatory_negative_quoted": 20,
    "phase12_confirmatory_negative_negated": 20,
    "phase12_confirmatory_negative_cancelled": 20,
    "phase12_confirmatory_negative_subtraction": 20,
    "phase12_confirmatory_negative_comparison": 20,
    "phase12_confirmatory_negative_concatenation": 20,
    "phase12_confirmatory_negative_hypothetical": 20,
    "phase12_confirmatory_negative_distractor": 20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_file(commit: str, path: Path) -> bytes:
    return subprocess.check_output(
        ("git", "show", f"{commit}:{path.as_posix()}"),
    )


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, descendant),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def result_commit_for(path: Path) -> str:
    commits = subprocess.check_output(
        (
            "git",
            "log",
            "--diff-filter=A",
            "--format=%H",
            "--",
            path.as_posix(),
        ),
        text=True,
    ).splitlines()
    return commits[0] if len(commits) == 1 else ""


def check(name: str, value: bool, checks: list[dict[str, object]]) -> None:
    checks.append({"name": name, "passed": bool(value)})


def record_consistent(
    record: dict[str, object],
    row: dict[str, object],
) -> bool:
    positive = bool(row["route_label"])
    expected_symbols = (
        [int(character) for character in row["answer"]] + [10]
        if positive
        else []
    )
    return (
        bool(record["format_exact"])
        == (
            record["generated_text"] == row["answer"]
            if positive
            else False
        )
        and bool(record["operands_exact"])
        == (
            record["predicted_a"] == row["a"]
            and record["predicted_b"] == row["b"]
            if positive
            else False
        )
        and bool(record["trajectory_exact"])
        == (
            record["result_symbols"][: len(expected_symbols)]
            == expected_symbols
            if positive
            else False
        )
        and bool(record["token_preserved"])
        == (
            record["generated_token_ids"]
            == row["base"]["generated_token_ids"]
        )
        and len(record["generated_token_ids"]) <= 8
    )


def recompute_metrics(
    rows: list[dict[str, object]],
    condition: str,
    seed: int,
) -> dict[str, object]:
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
    if condition == "phase12_candidate":
        metrics.update(
            {
                "oracle_exact": sum(
                    record["oracle_route"]["format_exact"]
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
    return metrics


def recompute_gates(
    metrics: dict[str, dict[str, dict[str, object]]],
    checkpoint_metadata: dict[str, dict[str, object]],
) -> tuple[dict[str, bool], dict[str, int]]:
    candidate = metrics["phase12_candidate"]
    gains = {
        str(seed): (
            candidate[str(seed)]["exact"]
            - metrics["phase11_control"][str(seed)]["exact"]
        )
        for seed in SEEDS
    }
    gates = {
        "autonomous_exactness": (
            all(value["exact"] >= 70 for value in candidate.values())
            and statistics.fmean(
                value["exact"] for value in candidate.values()
            )
            >= 75
        ),
        "paired_improvement": (
            all(gain > 0 for gain in gains.values())
            and statistics.fmean(gains.values()) >= 10
        ),
        "route_recognition": all(
            value["positive_routes"] >= 80 for value in candidate.values()
        ),
        "preservation": all(
            value["false_routes"] <= 4
            and value["token_preserved"] >= 196
            for value in candidate.values()
        ),
        "operand_access": all(
            value["oracle_exact"] >= 85 for value in candidate.values()
        ),
        "conditional_mechanism": all(
            value["conditional_exact"] == value["conditional_examples"]
            and value["conditional_trajectories_exact"]
            == value["conditional_examples"]
            for value in candidate.values()
        ),
        "causal_routing": all(
            value["paired_route_off_losses"] == value["exact"]
            and value["route_off_exact"] <= 5
            for value in candidate.values()
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
    return gates, gains


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    checks: list[dict[str, object]] = []
    check(
        "statuses",
        manifest["status"] == "phase12_protocol_frozen_before_confirmation"
        and confirmation["status"] == "phase12_confirmation_complete"
        and analysis["status"] == "phase12_posthoc_analysis_complete",
        checks,
    )
    check(
        "frozen_model_identity",
        confirmation["model_id"] == PHASE8_MODEL_ID
        and confirmation["model_revision"] == PHASE8_MODEL_REVISION,
        checks,
    )
    check(
        "protocol_hash_and_frozen_content",
        manifest["protocol"] == PROTOCOL.as_posix()
        and sha256(PROTOCOL) == manifest["protocol_sha256"]
        and sha256_bytes(git_file(manifest["implementation_commit"], PROTOCOL))
        == manifest["protocol_sha256"],
        checks,
    )
    check(
        "manifest_existed_unchanged_at_confirmation",
        sha256_bytes(
            git_file(confirmation["implementation_commit"], MANIFEST)
        )
        == sha256(MANIFEST)
        == confirmation["manifest"]["sha256"],
        checks,
    )
    result_commit = result_commit_for(CONFIRMATION)
    check(
        "result_committed_once_with_current_content",
        bool(result_commit)
        and sha256_bytes(git_file(result_commit, CONFIRMATION))
        == sha256(CONFIRMATION),
        checks,
    )
    check(
        "freeze_before_result_commit_order",
        git_is_ancestor(
            manifest["implementation_commit"],
            confirmation["implementation_commit"],
        )
        and bool(result_commit)
        and git_is_ancestor(
            confirmation["implementation_commit"],
            result_commit,
        ),
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
        == manifest["canonical_rows_sha256"]
        == confirmation["manifest"]["canonical_rows_sha256"],
        checks,
    )
    generated_rows = [
        example.to_dict() for example in build_phase12_confirmatory_examples()
    ]
    evaluated_rows = [
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
    check(
        "generated_manifest_evaluated_rows_identical",
        generated_rows == manifest["rows"] == evaluated_rows,
        checks,
    )
    check(
        "prompt_counts_and_splits",
        len(generated_rows) == 300
        and len({row["prompt"] for row in generated_rows}) == 300
        and sum(row["route_label"] for row in generated_rows) == 100
        and Counter(row["split"] for row in generated_rows)
        == Counter(EXPECTED_SPLITS)
        == Counter(manifest["split_counts"]),
        checks,
    )
    prior_families = phase10_family_set() | phase11_family_set()
    for values in phase9_family_sets().values():
        prior_families.update(values)
    check(
        "family_and_operand_disjointness",
        len(phase12_family_set()) == 70
        and phase12_family_set().isdisjoint(prior_families)
        and operand_pairs(
            build_phase12_confirmatory_examples()
        ).isdisjoint(_all_pairs_before_phase12()),
        checks,
    )
    check(
        "decoding_and_counts",
        confirmation["decoding"]
        == {"method": "greedy", "max_new_tokens": 8}
        and confirmation["prompts"] == 300
        and confirmation["positive_prompts"] == 100
        and confirmation["negative_prompts"] == 200,
        checks,
    )
    check(
        "base_outputs",
        all(
            len(row["base"]["generated_token_ids"]) <= 8
            for row in confirmation["rows"]
        ),
        checks,
    )
    all_records_consistent = True
    for row in confirmation["rows"]:
        for condition in ("phase11_control", "phase12_candidate"):
            for seed in SEEDS:
                record = row["conditions"][condition][str(seed)]
                all_records_consistent &= record_consistent(record, row)
                if condition == "phase12_candidate" and row["route_label"]:
                    all_records_consistent &= record_consistent(
                        record["oracle_route"],
                        row,
                    )
                    all_records_consistent &= record_consistent(
                        record["route_off"],
                        row,
                    )
    check("all_row_records_recomputed", all_records_consistent, checks)
    recomputed = {
        condition: {
            str(seed): recompute_metrics(
                confirmation["rows"],
                condition,
                seed,
            )
            for seed in SEEDS
        }
        for condition in ("phase11_control", "phase12_candidate")
    }
    check(
        "aggregate_metrics_recomputed",
        recomputed == confirmation["metrics"],
        checks,
    )
    gates, gains = recompute_gates(
        recomputed,
        confirmation["checkpoints"],
    )
    check(
        "paired_gains_and_gates_recomputed",
        gains == confirmation["paired_exact_gains"]
        and gates == confirmation["gates"]
        and gates == analysis["confirmatory_verdict"],
        checks,
    )
    for seed in SEEDS:
        key = str(seed)
        source_path = Path(
            manifest["candidate_checkpoints"][key][
                "source_phase10_checkpoint"
            ]
        )
        candidate_path = Path(
            manifest["candidate_checkpoints"][key]["path"]
        )
        control_path = Path(
            manifest["phase11_control_checkpoints"][key]["path"]
        )
        source = torch.load(source_path, map_location="cpu", weights_only=True)
        candidate = torch.load(
            candidate_path,
            map_location="cpu",
            weights_only=True,
        )
        inherited = {
            name: torch.equal(value, candidate[name])
            for name, value in source.items()
            if isinstance(value, torch.Tensor)
        }
        check(
            f"checkpoint_{seed}_hashes_and_inheritance",
            sha256(source_path)
            == manifest["candidate_checkpoints"][key][
                "source_phase10_checkpoint_sha256"
            ]
            == confirmation["checkpoints"][key]["source"]["sha256"]
            and sha256(candidate_path)
            == manifest["candidate_checkpoints"][key]["sha256"]
            == confirmation["checkpoints"][key]["phase12_candidate"]["sha256"]
            and sha256(control_path)
            == manifest["phase11_control_checkpoints"][key]["sha256"]
            == confirmation["checkpoints"][key]["phase11_control"]["sha256"]
            and all(inherited.values())
            and candidate["request_route_down"].numel()
            + candidate["request_route_output"].numel()
            == 131_104
            and candidate["request_router_kind"] == "all_views_silu16"
            and candidate["request_route_threshold"] == 0.6,
            checks,
        )
    check(
        "analysis_primary_totals",
        analysis["paired_phase12_vs_phase11"]["phase11_control_exact"]
        == sum(
            recomputed["phase11_control"][str(seed)]["exact"]
            for seed in SEEDS
        )
        and analysis["paired_phase12_vs_phase11"][
            "phase12_candidate_exact"
        ]
        == sum(
            recomputed["phase12_candidate"][str(seed)]["exact"]
            for seed in SEEDS
        )
        and analysis["paired_phase12_vs_phase11"]["two_way_bootstrap"][
            "draws"
        ]
        == 100_000,
        checks,
    )
    if PROGRESS.exists():
        progress = json.loads(PROGRESS.read_text())
        progress_rows_match = progress["rows"] == confirmation["rows"]
        check(
            "restart_progress_finalized_and_identical",
            progress["status"] == "phase12_confirmation_complete"
            and progress["manifest_sha256"] == sha256(MANIFEST)
            and progress["evaluator_sha256"] == sha256(EVALUATOR)
            and progress_rows_match,
            checks,
        )
    else:
        check("restart_progress_finalized_and_identical", False, checks)
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "status": (
            "phase12_confirmation_verification_passed"
            if not failed
            else "phase12_confirmation_verification_failed"
        ),
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_checks": [item["name"] for item in failed],
        "artifact_hashes": {
            str(path): sha256(path)
            for path in (
                MANIFEST,
                CONFIRMATION,
                ANALYSIS,
                PROTOCOL,
                EVALUATOR,
            )
        },
    }
    AUDIT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
