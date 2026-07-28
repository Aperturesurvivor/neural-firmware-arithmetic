from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

import torch

from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    operand_pairs,
)
from neural_firmware.phase9_data import (
    PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE9_CONFIRMATORY_POSITIVE_FAMILIES,
    PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE9_DEVELOPMENT_POSITIVE_FAMILIES,
    PHASE9_HARD_NEGATIVE_FAMILIES,
    PHASE9_HARD_POSITIVE_FAMILIES,
    _phase8_used_pairs,
    build_phase9_confirmatory_examples,
    build_phase9_development,
    build_phase9_generic_training,
    build_phase9_hard_training,
)
from neural_firmware.phase10_data import (
    PHASE10_SOURCE_SEEDS,
    PHASE10_TRAINING_SEEDS,
    build_phase10_confirmatory_examples,
    phase10_family_set,
)
from neural_firmware.semantic_data import exact_format_correct

MANIFEST = Path("phase10_results/frozen_prompt_manifest.json")
TRAINING = Path("phase10_results/confirmatory_interface_training.json")
CONFIRMATION = Path("phase10_results/confirmation.json")
ANALYSIS = Path("phase10_results/analysis.json")
ENVIRONMENT = Path("phase10_results/environment_audit.json")
AUDIT = Path("phase10_results/completion_audit.json")
PROTOCOL = Path("PHASE10_INTERFACE_CAPACITY_PROTOCOL.md")
UV_LOCK = Path("uv.lock")
PYPROJECT = Path("pyproject.toml")
CONDITIONS = ("linear", "nonlinear", "linear_representation")
SEEDS = (16_201, 16_202, 16_203)
SOURCE_SEEDS = {16_201: 14_201, 16_202: 14_202, 16_203: 14_203}
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
EXPECTED_SPLITS = {
    "phase10_confirmatory_positive_direct": 50,
    "phase10_confirmatory_positive_word": 25,
    "phase10_confirmatory_positive_distractor": 25,
    "phase10_confirmatory_negative_multiplication": 20,
    "phase10_confirmatory_negative_factual": 20,
    "phase10_confirmatory_negative_quoted": 20,
    "phase10_confirmatory_negative_negated": 20,
    "phase10_confirmatory_negative_cancelled": 20,
    "phase10_confirmatory_negative_subtraction": 20,
    "phase10_confirmatory_negative_comparison": 20,
    "phase10_confirmatory_negative_concatenation": 20,
    "phase10_confirmatory_negative_hypothetical": 20,
    "phase10_confirmatory_negative_distractor": 20,
}
EXPECTED_CONDITIONS = {
    "linear": {
        "interface_kind": "linear",
        "representation_rank": 0,
        "input_parameters": 32_768,
        "representation_parameters": 0,
        "architectural_parameters": 57_344,
        "trainable_parameters": 32_768,
    },
    "nonlinear": {
        "interface_kind": "bottleneck_silu",
        "representation_rank": 0,
        "input_parameters": 32_768,
        "representation_parameters": 0,
        "architectural_parameters": 57_344,
        "trainable_parameters": 32_768,
    },
    "linear_representation": {
        "interface_kind": "linear",
        "representation_rank": 4,
        "input_parameters": 32_768,
        "representation_parameters": 16_384,
        "architectural_parameters": 73_728,
        "trainable_parameters": 49_152,
    },
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


def first_step_value(
    result: dict[str, object],
    key: str,
    default: object,
) -> object:
    steps = result.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return default
    value = steps[0].get(key, default)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def decoded_operand(
    result: dict[str, object],
    prefix: str,
) -> str | None:
    digits = first_step_value(result, f"{prefix}_digits", [])
    length = first_step_value(result, f"{prefix}_lengths", 0)
    if not isinstance(digits, list) or not isinstance(length, int) or length < 1:
        return None
    return "".join(str(value) for value in digits[:length])


def raw_trajectory_exact(
    result: dict[str, object],
    answer: str,
) -> bool:
    steps = result.get("steps", [])
    if not isinstance(steps, list):
        return False
    symbols = [
        first_step_value({"steps": [step]}, "result_symbols", 11)
        for step in steps
    ]
    expected = [int(character) for character in answer] + [10]
    return symbols[: len(expected)] == expected


def raw_record_metrics(
    result: dict[str, object],
    row: dict[str, object],
    base_token_ids: list[int],
) -> dict[str, bool]:
    positive = bool(row["route_label"])
    return {
        "format_exact": (
            exact_format_correct(result["generated_text"], row["answer"])
            if positive
            else False
        ),
        "first_route": bool(first_step_value(result, "route", 0)),
        "first_route_active": bool(
            first_step_value(result, "route_active", False)
        ),
        "operands_exact": (
            decoded_operand(result, "a") == row["a"]
            and decoded_operand(result, "b") == row["b"]
            if positive
            else False
        ),
        "trajectory_exact": (
            raw_trajectory_exact(result, row["answer"]) if positive else False
        ),
        "token_preserved": result["generated_token_ids"] == base_token_ids,
    }


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
    if len(commits) != 1:
        return ""
    return commits[0]


def check(name: str, value: bool, checks: list[dict[str, object]]) -> None:
    checks.append({"name": name, "passed": bool(value)})


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    training = json.loads(TRAINING.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    environment = json.loads(ENVIRONMENT.read_text())
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
    check(
        "analysis_status",
        analysis["status"] == "phase10_posthoc_analysis_complete",
        checks,
    )
    check(
        "environment_status_and_scope",
        environment["status"]
        == "phase10_environment_provenance_audited_posthoc"
        and "not captured as contemporaneous" in environment["scope_note"],
        checks,
    )
    check(
        "dependency_hashes",
        environment["dependency_provenance"]["uv_lock_sha256"]
        == sha256(UV_LOCK)
        and environment["dependency_provenance"]["pyproject_sha256"]
        == sha256(PYPROJECT),
        checks,
    )
    check(
        "dependencies_unchanged_across_runs",
        sha256_bytes(
            git_file(
                environment["dependency_provenance"]["training_commit"],
                UV_LOCK,
            )
        )
        == sha256(UV_LOCK)
        and sha256_bytes(
            git_file(
                environment["dependency_provenance"]["confirmation_commit"],
                UV_LOCK,
            )
        )
        == sha256(UV_LOCK)
        and sha256_bytes(
            git_file(
                environment["dependency_provenance"]["training_commit"],
                PYPROJECT,
            )
        )
        == sha256(PYPROJECT)
        and sha256_bytes(
            git_file(
                environment["dependency_provenance"]["confirmation_commit"],
                PYPROJECT,
            )
        )
        == sha256(PYPROJECT),
        checks,
    )
    locked_versions = {
        package["name"]: package["version"]
        for package in tomllib.loads(UV_LOCK.read_text())["package"]
    }
    check(
        "audited_runtime_matches_dependency_lock",
        environment["runtime"]["torch"] == locked_versions["torch"]
        and environment["runtime"]["transformers"]
        == locked_versions["transformers"]
        and environment["runtime"]["numpy"] == locked_versions["numpy"],
        checks,
    )
    loader_at_confirmation = git_file(
        environment["dependency_provenance"]["confirmation_commit"],
        Path("src/neural_firmware/pretrained_training.py"),
    ).decode()
    check(
        "audited_mps_selection_matches_committed_loader",
        environment["runtime"]["execution_device"] == "mps"
        and '"mps" if torch.backends.mps.is_available() else "cpu"'
        in loader_at_confirmation,
        checks,
    )
    feature_provenance = environment["training_feature_provenance"]
    feature_records = [
        value
        for key, value in feature_provenance.items()
        if key != "scope"
    ]
    check(
        "training_feature_hashes_and_sizes",
        all(
            Path(record["path"]).is_file()
            and sha256(Path(record["path"])) == record["sha256"]
            and Path(record["path"]).stat().st_size == record["bytes"]
            for record in feature_records
        ),
        checks,
    )
    development_feature_manifest = json.loads(
        Path("phase10_results/development_feature_manifest.json").read_text()
    )
    check(
        "development_feature_manifest_identity",
        development_feature_manifest["cache"]
        == feature_provenance["phase10_development"]["path"]
        and development_feature_manifest["cache_sha256"]
        == feature_provenance["phase10_development"]["sha256"]
        and development_feature_manifest["cache_bytes"]
        == feature_provenance["phase10_development"]["bytes"],
        checks,
    )
    training_source = git_file(
        training["implementation_commit"],
        Path("scripts/train_phase10_interfaces.py"),
    ).decode()
    check(
        "committed_training_uses_frozen_feature_sources",
        all(record["path"] in training_source for record in feature_records),
        checks,
    )
    check(
        "protocol_path",
        manifest["protocol"] == PROTOCOL.as_posix(),
        checks,
    )
    check(
        "protocol_hash",
        sha256(PROTOCOL) == manifest["protocol_sha256"],
        checks,
    )
    check(
        "protocol_frozen_content",
        sha256_bytes(
            git_file(manifest["implementation_commit"], PROTOCOL)
        )
        == manifest["protocol_sha256"],
        checks,
    )
    check(
        "manifest_committed_before_training",
        sha256_bytes(
            git_file(training["implementation_commit"], MANIFEST)
        )
        == sha256(MANIFEST),
        checks,
    )
    check(
        "training_committed_before_confirmation",
        sha256_bytes(
            git_file(confirmation["implementation_commit"], TRAINING)
        )
        == sha256(TRAINING),
        checks,
    )
    check(
        "manifest_unchanged_before_confirmation",
        sha256_bytes(
            git_file(confirmation["implementation_commit"], MANIFEST)
        )
        == sha256(MANIFEST),
        checks,
    )
    result_commit = result_commit_for(CONFIRMATION)
    check("confirmation_has_single_add_commit", bool(result_commit), checks)
    check(
        "confirmation_commit_content",
        bool(result_commit)
        and sha256_bytes(git_file(result_commit, CONFIRMATION))
        == sha256(CONFIRMATION),
        checks,
    )
    check(
        "freeze_training_confirmation_commit_order",
        git_is_ancestor(
            manifest["implementation_commit"],
            training["implementation_commit"],
        )
        and git_is_ancestor(
            training["implementation_commit"],
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
    generated_examples = build_phase10_confirmatory_examples()
    generated_rows = [example.to_dict() for example in generated_examples]
    check(
        "generated_rows_match_manifest",
        generated_rows == manifest["rows"],
        checks,
    )
    check(
        "manifest_unique_prompts",
        len({row["prompt"] for row in manifest["rows"]})
        == manifest["unique_prompts"]
        == 300,
        checks,
    )
    check(
        "manifest_positive_negative_counts",
        sum(bool(row["route_label"]) for row in manifest["rows"])
        == manifest["positive_prompts"]
        == 100
        and sum(not bool(row["route_label"]) for row in manifest["rows"])
        == manifest["negative_prompts"]
        == 200,
        checks,
    )
    check(
        "manifest_split_counts",
        Counter(row["split"] for row in manifest["rows"])
        == Counter(manifest["split_counts"])
        == Counter(EXPECTED_SPLITS),
        checks,
    )
    check(
        "manifest_seed_mapping",
        manifest["phase10_seeds"] == list(SEEDS)
        and manifest["source_seed_mapping"]
        == {str(seed): source for seed, source in SOURCE_SEEDS.items()},
        checks,
    )
    prior_families = {
        family.template
        for family in (
            PHASE9_HARD_POSITIVE_FAMILIES
            + PHASE9_HARD_NEGATIVE_FAMILIES
            + PHASE9_DEVELOPMENT_POSITIVE_FAMILIES
            + PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES
            + PHASE9_CONFIRMATORY_POSITIVE_FAMILIES
            + PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES
        )
    }
    check(
        "seventy_unique_phase10_families",
        len(phase10_family_set()) == 70,
        checks,
    )
    check(
        "phase10_families_disjoint_from_phase9",
        phase10_family_set().isdisjoint(prior_families),
        checks,
    )
    prior_examples = (
        build_phase9_generic_training()
        + build_phase9_hard_training()
        + build_phase9_development()
        + build_phase9_confirmatory_examples()
    )
    prior_pairs = _phase8_used_pairs() | operand_pairs(prior_examples)
    phase10_pairs = operand_pairs(generated_examples)
    check(
        "phase10_operand_pairs_unique",
        len(phase10_pairs) == len(generated_examples) == 300,
        checks,
    )
    check(
        "phase10_operand_pairs_disjoint_from_phase8_and_phase9",
        phase10_pairs.isdisjoint(prior_pairs),
        checks,
    )
    check(
        "phase10_four_digit_operand_limit",
        all(
            1 <= len(row[operand]) <= 4
            for row in manifest["rows"]
            for operand in ("a", "b")
        ),
        checks,
    )
    check(
        "phase10_answers_are_exact_sums",
        all(
            row["answer"] == str(int(row["a"]) + int(row["b"]))
            for row in manifest["rows"]
            if row["route_label"]
        ),
        checks,
    )

    def contextual_numbers(row: dict[str, object]) -> list[str]:
        values = re.findall(
            r"(?<![0-9])[0-9]+(?![0-9])",
            row["prompt"],
        )
        for operand in ("a", "b"):
            values.remove(row[operand])
        return values

    positive_distractors = [
        row
        for row in manifest["rows"]
        if row["split"] == "phase10_confirmatory_positive_distractor"
    ]
    check(
        "positive_distractors_have_one_five_digit_context_number",
        len(positive_distractors) == 25
        and all(
            len(contextual_numbers(row)) == 1
            and len(contextual_numbers(row)[0]) == 5
            for row in positive_distractors
        ),
        checks,
    )
    check(
        "frozen_model_identity",
        PHASE8_MODEL_ID == MODEL_ID
        and PHASE8_MODEL_REVISION == MODEL_REVISION
        and training["model_id"] == confirmation["model_id"] == MODEL_ID
        and training["model_revision"]
        == confirmation["model_revision"]
        == MODEL_REVISION,
        checks,
    )
    check(
        "frozen_conditions",
        training["conditions"] == list(CONDITIONS),
        checks,
    )
    check(
        "frozen_training_seeds",
        PHASE10_TRAINING_SEEDS == SEEDS
        and PHASE10_SOURCE_SEEDS == SOURCE_SEEDS
        and training["phase10_seeds"]
        == confirmation["phase10_seeds"]
        == list(SEEDS),
        checks,
    )
    check(
        "confirmation_decoding",
        confirmation["decoding"]
        == {"method": "greedy", "max_new_tokens": 8},
        checks,
    )
    check(
        "confirmation_counts",
        confirmation["unique_prompts"] == 300
        and confirmation["positive_prompts"] == 100
        and confirmation["negative_prompts"] == 200,
        checks,
    )
    check(
        "confirmation_manifest_identity",
        confirmation["frozen_manifest"]["sha256"] == sha256(MANIFEST)
        and confirmation["frozen_manifest"]["canonical_rows_sha256"]
        == manifest["canonical_rows_sha256"],
        checks,
    )
    training_records = {
        (record["condition"], record["phase10_seed"]): record
        for record in training["records"]
    }
    check("training_record_count", len(training_records) == 9, checks)
    check(
        "base_generation_limit",
        all(
            len(row["base"]["generated_token_ids"]) <= 8
            for row in confirmation["rows"]
        ),
        checks,
    )
    check(
        "base_exact_recomputed",
        confirmation["base_exact"]
        == sum(
            exact_format_correct(
                row["base"]["generated_text"],
                row["answer"],
            )
            for row in confirmation["rows"]
            if row["route_label"]
        )
        == 0,
        checks,
    )
    for condition in CONDITIONS:
        expected = EXPECTED_CONDITIONS[condition]
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
            training_result = record["training"]
            condition_result = training_result["condition"]
            train_config = training_result["training"]["config"]
            route_config = training_result["route_training"]["config"]
            route_development = training_result["route_development"]
            check(
                f"{label}_checkpoint_hash",
                sha256(checkpoint_path)
                == record["checkpoint_sha256"]
                == confirmation["checkpoints"][condition][key][
                    "checkpoint_sha256"
                ],
                checks,
            )
            check(
                f"{label}_source_hash",
                sha256(source_path)
                == record["source_checkpoint_sha256"]
                == checkpoint["source_checkpoint_sha256"],
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
                f"{label}_result_decoder_width",
                checkpoint["result_columns"].numel() == 24_576,
                checks,
            )
            check(
                f"{label}_source_mapping",
                record["source_phase8_seed"]
                == checkpoint["source_phase8_seed"]
                == confirmation["checkpoints"][condition][key][
                    "source_phase8_seed"
                ]
                == SOURCE_SEEDS[seed]
                and record["phase10_seed"] == checkpoint["phase10_seed"] == seed,
                checks,
            )
            check(
                f"{label}_model_and_stage",
                checkpoint["model_id"] == MODEL_ID
                and checkpoint["model_revision"] == MODEL_REVISION
                and checkpoint["stage"]
                == "phase10_frozen_confirmatory_interface"
                and checkpoint["implementation_commit"]
                == training["implementation_commit"],
                checks,
            )
            check(
                f"{label}_frozen_implant_location",
                checkpoint["layer_index"] == source["layer_index"] == 15
                and len(checkpoint["selected_indices"]) == 28
                and torch.equal(
                    checkpoint["selected_indices"],
                    source["selected_indices"],
                ),
                checks,
            )
            check(
                f"{label}_frozen_layout",
                checkpoint["layout"]
                == source["layout"]
                == {
                    "max_digits": 4,
                    "digit_classes": 11,
                    "result_classes": 12,
                    "learned_step": False,
                },
                checks,
            )
            check(
                f"{label}_frozen_runtime",
                checkpoint["runtime"]
                == {
                    "latch_route": True,
                    "latch_operands": True,
                    "deterministic_result_step": True,
                    "preserve_base_when_off": True,
                }
                and checkpoint["output_strength"] == 16.0
                and checkpoint["digit_threshold"] == 0.8,
                checks,
            )
            check(
                f"{label}_route_temperature",
                checkpoint["route_temperature"]
                == confirmation["checkpoints"][condition][key][
                    "route_temperature"
                ]
                == 2.0,
                checks,
            )
            check(
                f"{label}_interface_local",
                checkpoint["adapt_base_mlp"] is False
                and condition_result["adapt_base_mlp"] is False
                and training_result["adapt_base_mlp"] is False,
                checks,
            )
            check(
                f"{label}_condition_architecture",
                checkpoint["condition"] == condition
                and checkpoint["interface_kind"]
                == condition_result["interface_kind"]
                == expected["interface_kind"]
                and checkpoint["representation_rank"]
                == condition_result["representation_rank"]
                == expected["representation_rank"],
                checks,
            )
            check(
                f"{label}_training_schedule",
                train_config
                == {
                    "seed": seed,
                    "steps": 2_500,
                    "batch_size": 256,
                    "learning_rate": 0.0005,
                    "route_loss_weight": 1.0,
                    "role_loss_weight": 1.0,
                    "digit_loss_weight": 1.0,
                    "step_loss_weight": 0.0,
                },
                checks,
            )
            check(
                f"{label}_threshold_only_calibration",
                route_config["seed"] == seed + 300
                and route_config["steps"] == 0
                and route_config["batch_size"] == 256
                and route_config["learning_rate"] == 0.0005
                and route_config[
                    "maximum_development_false_positive_rate"
                ]
                == 0.01
                and training_result["route_training"]["initial_loss"] is None
                and training_result["route_training"]["final_loss"] is None
                and route_development["positive_rows"] == 100
                and route_development["negative_rows"] == 200
                and route_development["false_positive_rate"] <= 0.010_001
                and checkpoint["route_threshold"]
                == route_development["threshold"]["threshold"],
                checks,
            )
            check(
                f"{label}_input_parameter_budget",
                training_result["input_interface_parameters"]
                == expected["input_parameters"]
                and training_result["training"]["trainable_parameters"]
                == expected["trainable_parameters"],
                checks,
            )
            check(
                f"{label}_representation_parameter_budget",
                training_result["representation_parameters"]
                == expected["representation_parameters"],
                checks,
            )
            check(
                f"{label}_architectural_parameter_count",
                training_result["architectural_learned_parameters"]
                == confirmation["checkpoints"][condition][key][
                    "architectural_learned_parameters"
                ]
                == expected["architectural_parameters"],
                checks,
            )
            check(
                f"{label}_interface_tensor_shapes",
                tuple(checkpoint["input_rows"].shape) == (16, 2_048)
                and (
                    condition != "nonlinear"
                    or (
                        tuple(checkpoint["bottleneck_rows"].shape)
                        == (16, 2_032)
                        and tuple(checkpoint["bottleneck_mix"].shape)
                        == (16, 16)
                        and checkpoint["bottleneck_rows"].numel()
                        + checkpoint["bottleneck_mix"].numel()
                        == 32_768
                    )
                )
                and (
                    condition != "linear_representation"
                    or (
                        tuple(checkpoint["representation_down"].shape)
                        == (4, 2_048)
                        and tuple(checkpoint["representation_up"].shape)
                        == (2_048, 4)
                        and checkpoint["representation_down"].numel()
                        + checkpoint["representation_up"].numel()
                        == 16_384
                    )
                ),
                checks,
            )
            check(
                f"{label}_parameter_count",
                confirmation["checkpoints"][condition][key][
                    "architectural_learned_parameters"
                ]
                == expected["architectural_parameters"],
                checks,
            )
            check(
                f"{label}_calculator_zero_parameter",
                training_result["calculator_learned_parameters"]
                == 0
                and
                confirmation["checkpoints"][condition][key][
                    "calculator_learned_parameters"
                ]
                == 0,
                checks,
            )

            positive_rows = [
                row
                for row in confirmation["rows"]
                if row["route_label"]
            ]
            negative_rows = [
                row
                for row in confirmation["rows"]
                if not row["route_label"]
            ]
            positive_records = [
                raw_record_metrics(
                    row["conditions"][condition][key],
                    row,
                    row["base"]["generated_token_ids"],
                )
                for row in positive_rows
            ]
            negative_records = [
                raw_record_metrics(
                    row["conditions"][condition][key],
                    row,
                    row["base"]["generated_token_ids"],
                )
                for row in negative_rows
            ]
            oracle_records = [
                raw_record_metrics(
                    row["conditions"][condition][key]["oracle_route"],
                    row,
                    row["base"]["generated_token_ids"],
                )
                for row in positive_rows
            ]
            stored_records_match_raw = all(
                all(
                    output[name] == value
                    for name, value in raw_record_metrics(
                        output,
                        row,
                        row["base"]["generated_token_ids"],
                    ).items()
                )
                for row in confirmation["rows"]
                for output in [row["conditions"][condition][key]]
            )
            stored_oracle_records_match_raw = all(
                all(
                    output[name] == value
                    for name, value in raw_record_metrics(
                        output,
                        row,
                        row["base"]["generated_token_ids"],
                    ).items()
                )
                for row in positive_rows
                for output in [
                    row["conditions"][condition][key]["oracle_route"]
                ]
            )
            check(
                f"{label}_stored_row_metrics_match_raw_outputs",
                stored_records_match_raw and stored_oracle_records_match_raw,
                checks,
            )
            check(
                f"{label}_generation_limit",
                all(
                    len(row["conditions"][condition][key]["generated_token_ids"])
                    <= 8
                    for row in confirmation["rows"]
                )
                and all(
                    len(
                        row["conditions"][condition][key]["oracle_route"][
                            "generated_token_ids"
                        ]
                    )
                    <= 8
                    for row in positive_rows
                ),
                checks,
            )
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
                    row["format_exact"] for row in oracle_records
                ),
                "oracle_operands_exact": sum(
                    row["operands_exact"] for row in oracle_records
                ),
                "oracle_trajectories_exact": sum(
                    row["trajectory_exact"] for row in oracle_records
                ),
            }
            if condition == "linear_representation":
                ablation_records = [
                    raw_record_metrics(
                        row["conditions"][condition][key]["ablation"],
                        row,
                        row["base"]["generated_token_ids"],
                    )
                    for row in positive_rows
                ]
                check(
                    f"{label}_stored_ablation_metrics_match_raw_outputs",
                    all(
                        all(
                            output[name] == value
                            for name, value in raw_record_metrics(
                                output,
                                row,
                                row["base"]["generated_token_ids"],
                            ).items()
                        )
                        for row in positive_rows
                        for output in [
                            row["conditions"][condition][key]["ablation"]
                        ]
                    ),
                    checks,
                )
                check(
                    f"{label}_ablation_generation_limit",
                    all(
                        len(
                            row["conditions"][condition][key]["ablation"][
                                "generated_token_ids"
                            ]
                        )
                        <= 8
                        for row in positive_rows
                    ),
                    checks,
                )
                recomputed["ablation_exact"] = sum(
                    row["format_exact"] for row in ablation_records
                )
                recomputed["paired_causal_losses"] = sum(
                    normal["format_exact"] and not ablated["format_exact"]
                    for normal, ablated in zip(
                        positive_records,
                        ablation_records,
                        strict=True,
                    )
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
                reported["conditional_examples"] == len(conditional)
                and reported["conditional_exact"]
                == sum(row["format_exact"] for row in conditional)
                and reported["conditional_trajectories_exact"]
                == sum(row["trajectory_exact"] for row in conditional)
                and all(
                    row["format_exact"] and row["trajectory_exact"]
                    for row in conditional
                ),
                checks,
            )
            check(
                f"{label}_oracle_route_forced",
                all(row["first_route"] for row in oracle_records),
                checks,
            )
            recomputed_categories: dict[str, dict[str, int]] = {}
            for row in confirmation["rows"]:
                category = row["split"].removeprefix(
                    "phase10_confirmatory_positive_"
                ).removeprefix("phase10_confirmatory_negative_")
                aggregate = recomputed_categories.setdefault(
                    category,
                    {"examples": 0, "exact": 0, "false_routes": 0},
                )
                raw = raw_record_metrics(
                    row["conditions"][condition][key],
                    row,
                    row["base"]["generated_token_ids"],
                )
                aggregate["examples"] += 1
                aggregate["exact"] += int(raw["format_exact"])
                if not row["route_label"]:
                    aggregate["false_routes"] += int(raw["first_route"])
            check(
                f"{label}_categories_recompute",
                recomputed_categories
                == confirmation["categories"][condition][key],
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
    check(
        "posthoc_condition_totals",
        analysis["representation_vs_linear"]["linear_exact"]
        == sum(
            confirmation["conditions"]["linear"][str(seed)]["exact"]
            for seed in SEEDS
        )
        == 107
        and analysis["representation_vs_linear"]["representation_exact"]
        == sum(
            confirmation["conditions"]["linear_representation"][str(seed)][
                "exact"
            ]
            for seed in SEEDS
        )
        == 141
        and analysis["nonlinear_vs_linear"]["nonlinear_exact"]
        == sum(
            confirmation["conditions"]["nonlinear"][str(seed)]["exact"]
            for seed in SEEDS
        )
        == 92,
        checks,
    )
    positive_category_counts: dict[str, dict[str, int]] = {
        category: {condition: 0 for condition in CONDITIONS}
        for category in ("direct", "word", "distractor")
    }
    positive_category_totals = Counter()
    for row in confirmation["rows"]:
        if not row["route_label"]:
            continue
        category = row["split"].removeprefix(
            "phase10_confirmatory_positive_"
        )
        for condition in CONDITIONS:
            for seed in SEEDS:
                raw = raw_record_metrics(
                    row["conditions"][condition][str(seed)],
                    row,
                    row["base"]["generated_token_ids"],
                )
                positive_category_counts[category][condition] += int(
                    raw["format_exact"]
                )
        positive_category_totals[category] += len(SEEDS)
    check(
        "posthoc_category_summary",
        all(
            analysis["category_summary"][category][condition]["exact"]
            == positive_category_counts[category][condition]
            and analysis["category_summary"][category][condition]["total"]
            == positive_category_totals[category]
            for category in positive_category_counts
            for condition in CONDITIONS
        ),
        checks,
    )
    development = json.loads(
        Path("phase10_results/development_evaluation.json").read_text()
    )
    selected_development = {
        condition: (
            values["summary"]["exact"],
            values["summary"]["oracle_exact"],
            values["summary"]["false_routes"],
            values["summary"]["token_preserved"],
        )
        for condition, values in development["conditions"].items()
    }
    check(
        "selected_development_results_retained",
        selected_development
        == {
            "linear": (37, 87, 2, 198),
            "nonlinear": (31, 80, 1, 199),
            "linear_representation": (44, 88, 2, 198),
            "nonlinear_representation": (38, 82, 2, 198),
        },
        checks,
    )
    shared_development = json.loads(
        Path(
            "phase10_results/"
            "development_evaluation_v2_no_route_hardening.json"
        ).read_text()
    )
    check(
        "rejected_shared_mlp_result_retained",
        shared_development["conditions"]["linear_representation"]["summary"][
            "token_preserved"
        ]
        == 185
        and shared_development["conditions"]["nonlinear_representation"][
            "summary"
        ]["token_preserved"]
        == 185,
        checks,
    )
    route_hardening = json.loads(
        Path(
            "phase10_results/"
            "development_training_v3_overfit_route_hardening.json"
        ).read_text()
    )
    hardening_false_positive_rates = {
        record["condition"]: record["training"]["route_development"][
            "false_positive_rate"
        ]
        for record in route_hardening["records"]
    }
    check(
        "rejected_route_hardening_retained",
        hardening_false_positive_rates["nonlinear"] >= 0.205 - 1e-6
        and hardening_false_positive_rates["linear_representation"]
        >= 0.065 - 1e-6
        and hardening_false_positive_rates["nonlinear_representation"]
        >= 0.11 - 1e-6,
        checks,
    )
    check(
        "rejected_fixed_mix_result_retained",
        Path(
            "phase10_results/development_training_fixed_mix_v1.json"
        ).is_file(),
        checks,
    )
    check(
        "reports_present",
        all(
            path.is_file()
            for path in (
                Path("PHASE10_ARCHITECTURE_DEVELOPMENT.md"),
                PROTOCOL,
                Path("PHASE10_EXECUTIVE_SUMMARY.md"),
                Path("PHASE10_LAB_NOTEBOOK.md"),
            )
        ),
        checks,
    )
    executive_summary = Path("PHASE10_EXECUTIVE_SUMMARY.md").read_text()
    check(
        "report_claim_boundaries",
        "passed all frozen gates" in executive_summary
        and "remains far from reliable" in executive_summary
        and "rejects the tested hypothesis" in executive_summary,
        checks,
    )
    tracked_checkpoints = subprocess.check_output(
        ("git", "ls-files", "phase10_artifacts"),
        text=True,
    ).strip()
    check(
        "checkpoints_remain_out_of_git",
        not tracked_checkpoints,
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
        "analysis_sha256": sha256(ANALYSIS),
        "environment_sha256": sha256(ENVIRONMENT),
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
