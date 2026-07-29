from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from neural_firmware.phase11_data import (
    PHASE11_NEGATIVE_DATA_SEED,
    PHASE11_POSITIVE_DATA_SEED,
    PHASE11_ROUTER_SEEDS,
    PHASE11_SOURCE_SEEDS,
    build_phase11_confirmatory_examples,
)

OUTPUT_PATH = Path("phase11_results/frozen_prompt_manifest.json")
PROTOCOL_PATH = Path("PHASE11_AUTONOMOUS_ROUTING_PROTOCOL.md")
DEVELOPMENT_PATH = Path("phase11_results/development_training.json")


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
    if development["selected_router_kind"] != "last":
        raise ValueError("frozen protocol requires the selected last router")
    selected = {
        int(record["phase10_seed"]): record
        for record in development["records"]
        if record["router_kind"] == "last"
    }
    examples = build_phase11_confirmatory_examples()
    rows = [example.to_dict() for example in examples]
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    splits = Counter(row["split"] for row in rows)
    payload = {
        "status": "phase11_protocol_frozen_before_confirmation",
        "implementation_commit": git_commit(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "development_result": str(DEVELOPMENT_PATH),
        "development_result_sha256": sha256_file(DEVELOPMENT_PATH),
        "selected_router_kind": development["selected_router_kind"],
        "source_seeds": list(PHASE11_SOURCE_SEEDS),
        "router_seed_mapping": {
            str(seed): PHASE11_ROUTER_SEEDS[seed]
            for seed in PHASE11_SOURCE_SEEDS
        },
        "data_seeds": {
            "positive": PHASE11_POSITIVE_DATA_SEED,
            "negative": PHASE11_NEGATIVE_DATA_SEED,
        },
        "candidate_checkpoints": {
            str(seed): {
                "path": selected[seed]["checkpoint"],
                "sha256": selected[seed]["checkpoint_sha256"],
                "source_phase10_checkpoint": selected[seed][
                    "source_phase10_checkpoint"
                ],
                "source_phase10_checkpoint_sha256": selected[seed][
                    "source_phase10_checkpoint_sha256"
                ],
                "request_route_threshold": selected[seed]["calibration"][
                    "threshold"
                ],
            }
            for seed in PHASE11_SOURCE_SEEDS
        },
        "unique_prompts": len({row["prompt"] for row in rows}),
        "positive_prompts": sum(bool(row["route_label"]) for row in rows),
        "negative_prompts": sum(not bool(row["route_label"]) for row in rows),
        "split_counts": dict(sorted(splits.items())),
        "canonical_rows_sha256": sha256_bytes(canonical),
        "rows": rows,
    }
    if payload["unique_prompts"] != 300:
        raise AssertionError("Phase 11 confirmation must contain 300 unique prompts")
    if payload["positive_prompts"] != 100:
        raise AssertionError("Phase 11 confirmation must contain 100 positives")
    if payload["negative_prompts"] != 200:
        raise AssertionError("Phase 11 confirmation must contain 200 negatives")
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
