from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase7_data import (
    PHASE7_AUDIT3_ADDITION_FAMILIES,
    PHASE7_AUDIT3_NEGATIVE_FAMILIES,
    PHASE7_AUDIT3_WORD_FAMILIES,
    phase7_audit3_prior_family_sets,
)
from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    RouteRowTrainConfig,
    collect_first_step_route_features,
    train_route_rows,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYER_INDEX = 16
SEEDS = (13_201, 13_202, 13_203)
PARENT_CHECKPOINTS = {
    13_201: Path(
        "phase7_artifacts/sequence_layer16_confident_v1/"
        "neuron_implant_seed_13201.pt"
    ),
    13_202: Path(
        "phase7_artifacts/sequence_layer16_multiseed/"
        "neuron_implant_seed_13202.pt"
    ),
    13_203: Path(
        "phase7_artifacts/sequence_layer16_multiseed/"
        "neuron_implant_seed_13203.pt"
    ),
}
PARENT_HASHES = {
    13_201: "9dba639d127769b08579b2e1deabdfd3d232e06dcf2ea6f843f7b9963855785c",
    13_202: "6cab7608a912d19a26793828352cdffd8783e2e5f6b8bdcad65f5afdf22b6b07",
    13_203: "cbf84806b08f0804cd0c508330e0f5f9fdfa72b41c800a37aef616c682fb58dd",
}
CACHE_PATH = Path(
    "phase7_artifacts/cache/route_hardening_v1_first_step_features.pt"
)
OUTPUT_DIRECTORY = Path("phase7_artifacts/sequence_layer16_router_hardened_v1")
RESULT_PATH = Path("phase7_results/sequence_layer16_router_hardening_v1.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def consumed_families() -> tuple[tuple[str, ...], tuple[str, ...]]:
    positives, negatives = phase7_audit3_prior_family_sets()
    positives.update(
        PHASE7_AUDIT3_ADDITION_FAMILIES + PHASE7_AUDIT3_WORD_FAMILIES
    )
    negatives.update(PHASE7_AUDIT3_NEGATIVE_FAMILIES)
    return tuple(sorted(positives)), tuple(sorted(negatives))


def balanced_family_examples(
    families: tuple[str, ...],
    *,
    examples_per_family: int,
    seed: int,
    split: str,
    positive: bool,
    min_digits: int,
    max_digits: int,
) -> list[SemanticPromptExample]:
    builder = (
        make_semantic_addition_examples
        if positive
        else make_semantic_routing_negatives
    )
    examples: list[SemanticPromptExample] = []
    for family_index, family in enumerate(families):
        examples.extend(
            builder(
                count=examples_per_family,
                min_digits=min_digits,
                max_digits=max_digits,
                seed=seed + 10_007 * family_index,
                split=split,
                families=(family,),
            )
        )
    return examples


def make_data() -> tuple[
    list[SemanticPromptExample],
    list[SemanticPromptExample],
    dict[str, object],
]:
    positive_families, negative_families = consumed_families()
    train = balanced_family_examples(
        positive_families,
        examples_per_family=16,
        seed=13_511,
        split="phase7_route_hardening_train_positive",
        positive=True,
        min_digits=1,
        max_digits=4,
    ) + balanced_family_examples(
        negative_families,
        examples_per_family=16,
        seed=13_512,
        split="phase7_route_hardening_train_negative",
        positive=False,
        min_digits=1,
        max_digits=4,
    )
    development = balanced_family_examples(
        positive_families,
        examples_per_family=4,
        seed=13_513,
        split="phase7_route_hardening_development_positive",
        positive=True,
        min_digits=1,
        max_digits=4,
    ) + balanced_family_examples(
        negative_families,
        examples_per_family=4,
        seed=13_514,
        split="phase7_route_hardening_development_negative",
        positive=False,
        min_digits=1,
        max_digits=4,
    )
    overlap = {example.prompt for example in train} & {
        example.prompt for example in development
    }
    if overlap:
        raise ValueError("route hardening train/development prompts overlap")
    metadata = {
        "positive_family_count": len(positive_families),
        "negative_family_count": len(negative_families),
        "train_examples": len(train),
        "development_examples": len(development),
        "train_examples_per_family": 16,
        "development_examples_per_family": 4,
        "train_seeds": {"positive": 13_511, "negative": 13_512},
        "development_seeds": {"positive": 13_513, "negative": 13_514},
    }
    return train, development, metadata


def load_or_collect_features(
    train: list[SemanticPromptExample],
    development: list[SemanticPromptExample],
    metadata: dict[str, object],
) -> tuple[FirstStepRouteFeatureSet, FirstStepRouteFeatureSet, torch.device]:
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
        if cache["metadata"] != metadata:
            raise ValueError("cached route features use different data")
        return (
            FirstStepRouteFeatureSet.load_state_dict(cache["train"]),
            FirstStepRouteFeatureSet.load_state_dict(cache["development"]),
            torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        )
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_first_step_route_features(
        bundle,
        train,
        layer_index=LAYER_INDEX,
        batch_size=8,
    )
    development_features = collect_first_step_route_features(
        bundle,
        development,
        layer_index=LAYER_INDEX,
        batch_size=8,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layer_index": LAYER_INDEX,
            "metadata": metadata,
            "train": train_features.state_dict(),
            "development": development_features.state_dict(),
        },
        CACHE_PATH,
    )
    return train_features, development_features, bundle.device


def main() -> None:
    started = time.perf_counter()
    train_examples, development_examples, data_metadata = make_data()
    train_features, development_features, device = load_or_collect_features(
        train_examples,
        development_examples,
        data_metadata,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, object]] = []
    for seed in SEEDS:
        parent_path = PARENT_CHECKPOINTS[seed]
        parent_hash = sha256(parent_path)
        if parent_hash != PARENT_HASHES[seed]:
            raise ValueError(f"seed {seed} parent checkpoint hash mismatch")
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        if parent["model_revision"] != MODEL_REVISION:
            raise ValueError("parent checkpoint uses a different Qwen revision")
        original_input_rows = parent["input_rows"].clone()
        original_result_columns = parent["result_columns"].clone()
        route_rows, training, development = train_route_rows(
            original_input_rows[:2],
            train_features,
            development_features,
            device=device,
            config=RouteRowTrainConfig(
                seed=seed + 300,
                steps=3_000,
                batch_size=256,
                learning_rate=0.001,
                maximum_development_false_positive_rate=0.002,
            ),
        )
        hardened_input_rows = original_input_rows.clone()
        hardened_input_rows[:2] = route_rows
        if not torch.equal(hardened_input_rows[2:], original_input_rows[2:]):
            raise AssertionError("non-route input rows changed during hardening")
        if not torch.equal(parent["result_columns"], original_result_columns):
            raise AssertionError("result columns changed during route hardening")
        output_path = OUTPUT_DIRECTORY / f"neuron_implant_seed_{seed}.pt"
        hardened = {
            **parent,
            "stage": "layer16_output_digit_confidence_router_hardened_v1",
            "implementation_commit": git_commit(),
            "route_threshold": development["threshold"]["threshold"],
            "input_rows": hardened_input_rows,
            "router_hardening": {
                "parent_checkpoint": str(parent_path),
                "parent_checkpoint_sha256": parent_hash,
                "updated_input_rows": [0, 1],
                "updated_parameter_count": route_rows.numel(),
                "training": training,
                "development": development,
            },
        }
        torch.save(hardened, output_path)
        seed_results.append(
            {
                "seed": seed,
                "parent_checkpoint": str(parent_path),
                "parent_checkpoint_sha256": parent_hash,
                "checkpoint": str(output_path),
                "checkpoint_sha256": sha256(output_path),
                "route_row_l2_change": float(
                    (route_rows - original_input_rows[:2]).norm()
                ),
                "non_route_input_rows_byte_identical": torch.equal(
                    hardened_input_rows[2:],
                    original_input_rows[2:],
                ),
                "result_columns_byte_identical": torch.equal(
                    hardened["result_columns"],
                    original_result_columns,
                ),
                "route_threshold_before": parent["route_threshold"],
                "route_threshold_after": hardened["route_threshold"],
                "training": training,
                "development": development,
            }
        )
        print(
            f"seed={seed} development={json.dumps(development, sort_keys=True)}",
            flush=True,
        )
    payload = {
        "status": "development_router_hardening_complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "method": (
            "Train only the two first-step route rows on a family-balanced union "
            "of all consumed positive and adversarial-negative prompt families."
        ),
        "data": data_metadata,
        "feature_cache": {
            "path": str(CACHE_PATH),
            "sha256": sha256(CACHE_PATH),
        },
        "updated_parameters_per_seed": 2 * 896,
        "unchanged_learned_parameters_per_seed": 25_088 - 2 * 896,
        "seeds": seed_results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": torch.backends.mps.is_available(),
            "implementation_commit": git_commit(),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
