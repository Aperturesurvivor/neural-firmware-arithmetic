from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

SOURCE_PATH = Path(
    "phase7_artifacts/sequence_layer16_v1/neuron_implant_seed_13201.pt"
)
SOURCE_SHA256 = (
    "5b76181c2c0f4a74b7482e4856b2b8c92bff637a1e70302bdfbc61ce1aaac41e"
)
OUTPUT_PATH = Path(
    "phase7_artifacts/sequence_layer16_confident_v1/"
    "neuron_implant_seed_13201.pt"
)
RESULT_PATH = Path(
    "phase7_results/sequence_layer16_digit_confidence_v1.json"
)
DIGIT_THRESHOLD = 0.9


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


def main() -> None:
    if sha256(SOURCE_PATH) != SOURCE_SHA256:
        raise ValueError("source checkpoint does not match the calibrated model")
    source = torch.load(SOURCE_PATH, map_location="cpu", weights_only=True)
    promoted = {
        **source,
        "stage": "layer16_output_digit_confidence_complete",
        "digit_threshold": DIGIT_THRESHOLD,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "promotion_commit": git_commit(),
        "digit_threshold_calibration": {
            "status": "post_audit_development",
            "positive_examples": 100,
            "true_digit_tokens": 460,
            "true_digit_probability_minimum": 0.9669705629348755,
            "false_typed_digit_candidates": 5,
            "false_digit_probability_maximum": 0.827191174030304,
            "threshold": DIGIT_THRESHOLD,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(promoted, OUTPUT_PATH)
    reloaded = torch.load(OUTPUT_PATH, map_location="cpu", weights_only=True)
    weights_unchanged = bool(
        torch.equal(source["input_rows"], reloaded["input_rows"])
        and torch.equal(source["result_columns"], reloaded["result_columns"])
    )
    if not weights_unchanged:
        raise RuntimeError("checkpoint promotion unexpectedly changed learned weights")
    payload = {
        "status": "post_audit_digit_confidence_promotion",
        "source_checkpoint": str(SOURCE_PATH),
        "source_sha256": SOURCE_SHA256,
        "checkpoint": str(OUTPUT_PATH),
        "checkpoint_sha256": sha256(OUTPUT_PATH),
        "digit_threshold": DIGIT_THRESHOLD,
        "weights_unchanged": weights_unchanged,
        "calibration": promoted["digit_threshold_calibration"],
        "learned_parameters": (
            source["input_rows"].numel() + source["result_columns"].numel()
        ),
        "calculator_learned_parameters": 0,
        "implementation_commit": git_commit(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
