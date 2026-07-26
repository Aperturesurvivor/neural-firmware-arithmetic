from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    CONFIRMATORY_ADDITION_FAMILIES,
    CONFIRMATORY_NEGATIVE_FAMILIES,
    CONFIRMATORY_WORD_PROBLEM_FAMILIES,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)
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
DEPTH_AFTER_BLOCKS = 24
ROUTER_HIDDEN_WIDTH = 16
ROUTE_THRESHOLD = 0.76
INTERNAL_PARAMETERS = 24_225
CONTROL_PARAMETERS = 24_225
EXPECTED_HASHES = {
    "phase4_artifacts/router_pilot_v2.pt": (
        "201262b5cf21259977dc8a31e3faa1aa77892f7cbae121ea015f4e69d95f8e66"
    ),
    "phase4_artifacts/semantic_pilot_v2/semantic_unit.pt": (
        "8079e0c5d723405881c39e47773fb895617d5c4da99b3168996e2861aba9a739"
    ),
    "phase4_artifacts/semantic_pilot_v2/semantic_control.pt": (
        "7d43f58126fc60bfb68ee2caa479818ce2b13b9b1c8d9f2b4b7adccb9840da94"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts() -> dict[str, str]:
    observed = {path: sha256(Path(path)) for path in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise RuntimeError(
            f"frozen artifact hash mismatch: expected {EXPECTED_HASHES}, got {observed}"
        )
    return observed


def release_memory() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "mathematical_correct": sum(
            row["mathematical_correct"] is True for row in rows
        ),
        "mathematical_accuracy": (
            sum(row["mathematical_correct"] is True for row in rows) / len(rows)
        ),
        "exact_format_correct": sum(
            row["exact_format_correct"] is True for row in rows
        ),
        "exact_format_accuracy": (
            sum(row["exact_format_correct"] is True for row in rows) / len(rows)
        ),
        "route_activations": sum(row["route_active"] is True for row in rows),
        "rows": rows,
    }


def build_positive_sets() -> dict[str, list[object]]:
    return {
        "confirmatory_id": make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=4,
            seed=9501,
            split="confirmatory_id",
            families=CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "confirmatory_ood": make_semantic_addition_examples(
            count=100,
            min_digits=5,
            max_digits=8,
            seed=9502,
            split="confirmatory_ood",
            families=CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "confirmatory_long": make_semantic_addition_examples(
            count=100,
            min_digits=9,
            max_digits=12,
            seed=9503,
            split="confirmatory_long",
            families=CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "confirmatory_word": make_semantic_addition_examples(
            count=100,
            min_digits=5,
            max_digits=8,
            seed=9504,
            split="confirmatory_word",
            families=CONFIRMATORY_WORD_PROBLEM_FAMILIES,
        ),
    }


def main() -> None:
    started = time.perf_counter()
    artifact_hashes = verify_artifacts()
    positive_sets = build_positive_sets()
    negative_examples = make_semantic_routing_negatives(
        count=160,
        min_digits=1,
        max_digits=12,
        seed=9505,
        split="confirmatory_negative",
        families=CONFIRMATORY_NEGATIVE_FAMILIES,
    )
    rendered_data = {
        "positive": {
            split: [example.to_dict() for example in examples]
            for split, examples in positive_sets.items()
        },
        "negative": [example.to_dict() for example in negative_examples],
    }

    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    wrapper = install_semantic_internal_firmware(
        bundle.model,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        strength=64.0,
        router_hidden_width=ROUTER_HIDDEN_WIDTH,
    )
    internal_state = torch.load(
        "phase4_artifacts/semantic_pilot_v2/semantic_unit.pt",
        map_location="cpu",
        weights_only=True,
    )
    wrapper.unit.load_state_dict(internal_state)
    base_results = {
        split: summarize(
            [
                generate_base_semantic(bundle, example, max_new_tokens=24)
                for example in examples
            ]
        )
        for split, examples in positive_sets.items()
    }
    internal_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in positive_sets.items()
    }
    oracle_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="force_on",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in positive_sets.items()
    }
    off_results = {
        split: summarize(
            [
                generate_semantic_internal(
                    bundle,
                    wrapper,
                    example,
                    route_mode="force_off",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in positive_sets.items()
    }
    negative_rows = []
    for example in negative_examples:
        base = generate_base_semantic(bundle, example, max_new_tokens=20)
        internal = generate_semantic_internal(
            bundle,
            wrapper,
            example,
            route_mode="learned",
            route_threshold=ROUTE_THRESHOLD,
            max_new_tokens=20,
        )
        negative_rows.append(
            {
                "prompt": example.prompt,
                "family": example.family,
                "family_index": example.family_index,
                "a": example.a,
                "b": example.b,
                "base_text": base["generated_text"],
                "internal_text": internal["generated_text"],
                "base_token_ids": base["generated_token_ids"],
                "internal_token_ids": internal["generated_token_ids"],
                "token_exact_preserved": (
                    base["generated_token_ids"] == internal["generated_token_ids"]
                ),
                "route_probability": internal["route_probability"],
                "route_active": internal["route_active"],
            }
        )
    del wrapper
    del bundle
    release_memory()

    control_bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    control_wrapper = install_semantic_learned_control(
        control_bundle.model,
        depth_after_blocks=DEPTH_AFTER_BLOCKS,
        rank=5,
        router_hidden_width=ROUTER_HIDDEN_WIDTH,
    )
    control_state = torch.load(
        "phase4_artifacts/semantic_pilot_v2/semantic_control.pt",
        map_location="cpu",
        weights_only=True,
    )
    control_wrapper.router.load_state_dict(control_state["router"])
    control_wrapper.adapter.load_state_dict(control_state["adapter"])
    control_results = {
        split: summarize(
            [
                generate_semantic_control(
                    control_bundle,
                    control_wrapper,
                    example,
                    route_mode="learned",
                    route_threshold=ROUTE_THRESHOLD,
                    max_new_tokens=24,
                )
                for example in examples
            ]
        )
        for split, examples in positive_sets.items()
    }

    forced_off_identity = {
        split: sum(
            base_row["generated_token_ids"] == off_row["generated_token_ids"]
            for base_row, off_row in zip(
                base_results[split]["rows"],
                off_results[split]["rows"],
                strict=True,
            )
        )
        for split in positive_sets
    }
    result = {
        "protocol": "PHASE4_CONFIRMATORY_PROTOCOL.md",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "git_commit_at_start": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "device": str(control_bundle.device),
        },
        "depth_after_blocks": DEPTH_AFTER_BLOCKS,
        "router_hidden_width": ROUTER_HIDDEN_WIDTH,
        "route_threshold": ROUTE_THRESHOLD,
        "internal_interface_parameters": INTERNAL_PARAMETERS,
        "control_interface_parameters": CONTROL_PARAMETERS,
        "artifact_sha256": artifact_hashes,
        "rendered_data": rendered_data,
        "base_results": base_results,
        "control_results": control_results,
        "internal_results": internal_results,
        "oracle_results": oracle_results,
        "off_results": off_results,
        "forced_off_identity": forced_off_identity,
        "negative_preservation": {
            "examples": len(negative_rows),
            "route_activations": sum(row["route_active"] for row in negative_rows),
            "token_exact_preserved": sum(
                row["token_exact_preserved"] for row in negative_rows
            ),
            "rows": negative_rows,
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    output = Path("phase4_results/confirmation_raw.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    concise = {
        condition: {
            split: {
                "mathematical_accuracy": row["mathematical_accuracy"],
                "exact_format_accuracy": row["exact_format_accuracy"],
                "route_activations": row["route_activations"],
            }
            for split, row in condition_rows.items()
        }
        for condition, condition_rows in (
            ("base", base_results),
            ("control", control_results),
            ("internal", internal_results),
            ("oracle", oracle_results),
            ("off", off_results),
        )
    }
    concise["forced_off_identity"] = forced_off_identity
    concise["negative_preservation"] = {
        key: result["negative_preservation"][key]
        for key in ("examples", "route_activations", "token_exact_preserved")
    }
    concise["wall_time_seconds"] = result["wall_time_seconds"]
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
