from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from neural_firmware.phase9_data import (
    PHASE9_SOURCE_SEEDS,
    PHASE9_TRAINING_SEEDS,
    build_phase9_confirmatory_examples,
)

OUTPUT_PATH = Path("phase9_results/frozen_prompt_manifest.json")
PROTOCOL_PATH = Path("PHASE9_INTERFACE_HARDENING_PROTOCOL.md")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def main() -> None:
    examples = build_phase9_confirmatory_examples()
    rows = [example.to_dict() for example in examples]
    canonical = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    splits = Counter(row["split"] for row in rows)
    payload = {
        "status": "phase9_protocol_frozen_before_confirmation",
        "implementation_commit": git_commit(),
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "phase9_seeds": list(PHASE9_TRAINING_SEEDS),
        "source_seed_mapping": {
            str(seed): PHASE9_SOURCE_SEEDS[seed]
            for seed in PHASE9_TRAINING_SEEDS
        },
        "unique_prompts": len({row["prompt"] for row in rows}),
        "positive_prompts": sum(bool(row["route_label"]) for row in rows),
        "negative_prompts": sum(not bool(row["route_label"]) for row in rows),
        "split_counts": dict(sorted(splits.items())),
        "canonical_rows_sha256": sha256_bytes(canonical),
        "rows": rows,
    }
    if payload["unique_prompts"] != 300:
        raise AssertionError("Phase 9 confirmation must contain 300 unique prompts")
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
