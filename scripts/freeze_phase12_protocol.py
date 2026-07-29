from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from neural_firmware.phase12_data import (
    PHASE12_NEGATIVE_DATA_SEED,
    PHASE12_POSITIVE_DATA_SEED,
    build_phase12_confirmatory_examples,
)

OUTPUT_PATH = Path("phase12_results/frozen_prompt_manifest.json")
PROTOCOL_PATH = Path("PHASE12_MULTI_VIEW_ROUTING_PROTOCOL.md")
DEVELOPMENT_PATH = Path("phase12_results/deployment_training.json")
DEVELOPMENT_EVALUATION_PATH = Path(
    "phase12_results/deployment_evaluation.json"
)
PHASE11_MANIFEST_PATH = Path(
    "phase11_results/frozen_prompt_manifest.json"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def main() -> None:
    development = json.loads(DEVELOPMENT_PATH.read_text())
    evaluation = json.loads(DEVELOPMENT_EVALUATION_PATH.read_text())
    phase11_manifest = json.loads(PHASE11_MANIFEST_PATH.read_text())
    if development["condition"] != "all_views_silu16":
        raise ValueError("Phase 12 condition does not match frozen protocol")
    if development["fixed_threshold"] != 0.6:
        raise ValueError("Phase 12 threshold does not match frozen protocol")
    if not evaluation["validation_gates"]["all_gates"]:
        raise ValueError("Phase 12 deployment validation did not pass")
    candidates = {
        int(record["phase10_seed"]): record
        for record in development["records"]
    }
    examples = build_phase12_confirmatory_examples()
    rows = [example.to_dict() for example in examples]
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    splits = Counter(row["split"] for row in rows)
    payload = {
        "status": "phase12_protocol_frozen_before_confirmation",
        "implementation_commit": git_commit(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_training": str(DEVELOPMENT_PATH),
        "development_training_sha256": sha256_file(DEVELOPMENT_PATH),
        "development_evaluation": str(DEVELOPMENT_EVALUATION_PATH),
        "development_evaluation_sha256": sha256_file(
            DEVELOPMENT_EVALUATION_PATH
        ),
        "condition": development["condition"],
        "fixed_threshold": development["fixed_threshold"],
        "source_seeds": sorted(candidates),
        "router_seed_mapping": development["router_seeds"],
        "data_seeds": {
            "positive": PHASE12_POSITIVE_DATA_SEED,
            "negative": PHASE12_NEGATIVE_DATA_SEED,
        },
        "candidate_checkpoints": {
            str(seed): {
                "path": candidates[seed]["checkpoint"],
                "sha256": candidates[seed]["checkpoint_sha256"],
                "source_phase10_checkpoint": candidates[seed][
                    "source_phase10_checkpoint"
                ],
                "source_phase10_checkpoint_sha256": candidates[seed][
                    "source_phase10_checkpoint_sha256"
                ],
                "phase12_router_seed": candidates[seed][
                    "phase12_router_seed"
                ],
                "request_route_threshold": candidates[seed][
                    "fixed_threshold"
                ],
            }
            for seed in sorted(candidates)
        },
        "phase11_control_checkpoints": phase11_manifest[
            "candidate_checkpoints"
        ],
        "unique_prompts": len({row["prompt"] for row in rows}),
        "positive_prompts": sum(bool(row["route_label"]) for row in rows),
        "negative_prompts": sum(not bool(row["route_label"]) for row in rows),
        "split_counts": dict(sorted(splits.items())),
        "canonical_rows_sha256": sha256_bytes(canonical),
        "rows": rows,
    }
    if payload["unique_prompts"] != 300:
        raise AssertionError("Phase 12 must contain 300 unique prompts")
    if payload["positive_prompts"] != 100:
        raise AssertionError("Phase 12 must contain 100 positives")
    if payload["negative_prompts"] != 200:
        raise AssertionError("Phase 12 must contain 200 negatives")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
