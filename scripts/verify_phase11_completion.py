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
from neural_firmware.phase10_data import (
    build_phase10_confirmatory_examples,
    phase10_family_set,
)
from neural_firmware.phase11_data import (
    PHASE11_ROUTER_SEEDS,
    PHASE11_SOURCE_SEEDS,
    _all_pairs_before_phase11,
    build_phase11_confirmatory_examples,
    phase11_family_set,
)

MANIFEST = Path("phase11_results/frozen_prompt_manifest.json")
CONFIRMATION = Path("phase11_results/confirmation.json")
ANALYSIS = Path("phase11_results/analysis.json")
ENVIRONMENT = Path("phase11_results/environment_audit.json")
AUDIT = Path("phase11_results/completion_audit.json")
PROTOCOL = Path("PHASE11_AUTONOMOUS_ROUTING_PROTOCOL.md")
SUMMARY = Path("PHASE11_EXECUTIVE_SUMMARY.md")
UV_LOCK = Path("uv.lock")
PYPROJECT = Path("pyproject.toml")
SEEDS = (16_201, 16_202, 16_203)
EXPECTED_SPLITS = {
    "phase11_confirmatory_positive_direct": 50,
    "phase11_confirmatory_positive_word": 25,
    "phase11_confirmatory_positive_distractor": 25,
    "phase11_confirmatory_negative_multiplication": 20,
    "phase11_confirmatory_negative_factual": 20,
    "phase11_confirmatory_negative_quoted": 20,
    "phase11_confirmatory_negative_negated": 20,
    "phase11_confirmatory_negative_cancelled": 20,
    "phase11_confirmatory_negative_subtraction": 20,
    "phase11_confirmatory_negative_comparison": 20,
    "phase11_confirmatory_negative_concatenation": 20,
    "phase11_confirmatory_negative_hypothetical": 20,
    "phase11_confirmatory_negative_distractor": 20,
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
            record["generated_token_ids"]
            == row["base"]["generated_token_ids"]
            for row, record in zip(
                negatives,
                negative_records,
                strict=True,
            )
        ),
    }
    if condition == "phase11_candidate":
        metrics.update(
            {
                "oracle_exact": sum(
                    record["oracle_route"]["format_exact"]
                    for record in positive_records
                ),
                "oracle_operands_exact": sum(
                    record["oracle_route"]["operands_exact"]
                    for record in positive_records
                ),
                "oracle_trajectories_exact": sum(
                    record["oracle_route"]["trajectory_exact"]
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


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    environment = json.loads(ENVIRONMENT.read_text())
    checks: list[dict[str, object]] = []
    check(
        "statuses",
        manifest["status"] == "phase11_protocol_frozen_before_confirmation"
        and confirmation["status"]
        == "phase11_confirmatory_evaluation_complete"
        and analysis["status"] == "phase11_posthoc_analysis_complete"
        and environment["status"]
        == "phase11_environment_provenance_audited_posthoc",
        checks,
    )
    check(
        "frozen_model_identity",
        confirmation["model_id"] == PHASE8_MODEL_ID
        and confirmation["model_revision"] == PHASE8_MODEL_REVISION
        and environment["model"]["id"] == PHASE8_MODEL_ID
        and environment["model"]["revision"] == PHASE8_MODEL_REVISION,
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
        == sha256(MANIFEST),
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
        == manifest["canonical_rows_sha256"],
        checks,
    )
    generated_rows = [
        example.to_dict() for example in build_phase11_confirmatory_examples()
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
    prior_families = phase10_family_set()
    for values in phase9_family_sets().values():
        prior_families.update(values)
    check(
        "family_and_operand_disjointness",
        len(phase11_family_set()) == 70
        and phase11_family_set().isdisjoint(prior_families)
        and operand_pairs(build_phase11_confirmatory_examples()).isdisjoint(
            _all_pairs_before_phase11()
        )
        and not (
            {row["prompt"] for row in generated_rows}
            & {
                row.prompt
                for row in build_phase10_confirmatory_examples()
            }
        ),
        checks,
    )
    check(
        "seed_and_checkpoint_manifest",
        tuple(manifest["source_seeds"]) == SEEDS == PHASE11_SOURCE_SEEDS
        and manifest["router_seed_mapping"]
        == {
            str(seed): PHASE11_ROUTER_SEEDS[seed]
            for seed in PHASE11_SOURCE_SEEDS
        },
        checks,
    )
    check(
        "decoding_and_counts",
        confirmation["decoding"]
        == {"method": "greedy", "max_new_tokens": 8}
        and confirmation["unique_prompts"] == 300
        and confirmation["positive_prompts"] == 100
        and confirmation["negative_prompts"] == 200,
        checks,
    )
    check(
        "base_outputs_and_exact",
        all(
            len(row["base"]["generated_token_ids"]) <= 8
            for row in confirmation["rows"]
        )
        and confirmation["base_exact"]
        == sum(
            row["base"]["generated_text"] == row["answer"]
            for row in confirmation["rows"]
            if row["route_label"]
        )
        == 0,
        checks,
    )
    all_records_consistent = True
    for row in confirmation["rows"]:
        for condition in ("phase10_control", "phase11_candidate"):
            for seed in SEEDS:
                record = row["conditions"][condition][str(seed)]
                all_records_consistent &= record_consistent(record, row)
                all_records_consistent &= (
                    record["token_preserved"]
                    == (
                        record["generated_token_ids"]
                        == row["base"]["generated_token_ids"]
                    )
                )
                if condition == "phase11_candidate" and row["route_label"]:
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
        for condition in ("phase10_control", "phase11_candidate")
    }
    check(
        "aggregate_metrics_recomputed",
        recomputed == confirmation["conditions"],
        checks,
    )
    paired_gains = [
        recomputed["phase11_candidate"][str(seed)]["exact"]
        - recomputed["phase10_control"][str(seed)]["exact"]
        for seed in SEEDS
    ]
    candidate_exact = [
        recomputed["phase11_candidate"][str(seed)]["exact"]
        for seed in SEEDS
    ]
    gates = {
        "autonomous_exactness": (
            min(candidate_exact) >= 70
            and statistics.fmean(candidate_exact) >= 75
        ),
        "paired_improvement": (
            all(gain > 0 for gain in paired_gains)
            and statistics.fmean(paired_gains) >= 20
        ),
        "route_recognition": all(
            recomputed["phase11_candidate"][str(seed)]["positive_routes"] >= 80
            for seed in SEEDS
        ),
        "preservation": all(
            recomputed["phase11_candidate"][str(seed)]["false_routes"] <= 4
            and recomputed["phase11_candidate"][str(seed)]["token_preserved"]
            >= 196
            for seed in SEEDS
        ),
        "operand_access": all(
            recomputed["phase11_candidate"][str(seed)]["oracle_exact"] >= 85
            for seed in SEEDS
        ),
        "conditional_mechanism": all(
            recomputed["phase11_candidate"][str(seed)]["conditional_exact"]
            == recomputed["phase11_candidate"][str(seed)][
                "conditional_examples"
            ]
            and recomputed["phase11_candidate"][str(seed)][
                "conditional_trajectories_exact"
            ]
            == recomputed["phase11_candidate"][str(seed)][
                "conditional_examples"
            ]
            for seed in SEEDS
        ),
        "causal_routing": all(
            recomputed["phase11_candidate"][str(seed)][
                "paired_route_off_losses"
            ]
            == recomputed["phase11_candidate"][str(seed)]["exact"]
            and recomputed["phase11_candidate"][str(seed)]["route_off_exact"]
            <= 5
            for seed in SEEDS
        ),
        "checkpoint_integrity": all(
            confirmation["checkpoints"][str(seed)]["inheritance"][
                "all_inherited_tensors_bit_identical"
            ]
            and confirmation["checkpoints"][str(seed)]["candidate"][
                "request_router_parameters"
            ]
            == 4_096
            and confirmation["checkpoints"][str(seed)]["candidate"][
                "architectural_parameter_delta"
            ]
            == 4_096
            for seed in SEEDS
        ),
    }
    gates["all_gates"] = all(gates.values())
    check(
        "paired_gains_and_gates_recomputed",
        paired_gains == confirmation["paired_gains"]
        and gates == confirmation["gates"]
        and gates
        == analysis["confirmatory_verdict"],
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
            == confirmation["checkpoints"][key]["candidate"]["sha256"]
            and all(inherited.values())
            and candidate["request_route_rows"].numel() == 4_096
            and candidate["request_router_kind"] == "last",
            checks,
        )
    check(
        "analysis_primary_totals",
        analysis["paired_candidate_vs_control"]["control_exact"] == 146
        and analysis["paired_candidate_vs_control"]["candidate_exact"] == 212
        and analysis["paired_candidate_vs_control"]["two_way_bootstrap"][
            "draws"
        ]
        == 100_000
        and [
            analysis["paired_candidate_vs_control"]["per_seed"][str(seed)][
                "posthoc_best_under_four_false_routes"
            ]["counterfactual_exact"]
            for seed in SEEDS
        ]
        == [66, 74, 68],
        checks,
    )
    check(
        "environment_dependency_and_artifact_hashes",
        environment["dependency_provenance"]["uv_lock_sha256"]
        == sha256(UV_LOCK)
        and environment["dependency_provenance"]["pyproject_sha256"]
        == sha256(PYPROJECT)
        and environment["artifacts"]["manifest_sha256"] == sha256(MANIFEST)
        and environment["artifacts"]["confirmation_sha256"]
        == sha256(CONFIRMATION),
        checks,
    )
    check(
        "summary_contains_frozen_verdict",
        "Compound verdict | **Fail**" in SUMMARY.read_text()
        and "49/100 to 67/100" in SUMMARY.read_text()
        and "46/100 to 77/100" in SUMMARY.read_text()
        and "51/100 to 68/100" in SUMMARY.read_text(),
        checks,
    )
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "status": (
            "phase11_completion_audit_passed"
            if not failed
            else "phase11_completion_audit_failed"
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
                ENVIRONMENT,
                PROTOCOL,
                SUMMARY,
            )
        },
    }
    AUDIT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
