from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from neural_firmware.phase8_adapter import install_matched_residual_adapter
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    build_phase8_training_and_development,
)
from neural_firmware.phase8_training import (
    MatchedAdapterTrainConfig,
    generate_matched_adapter,
    train_matched_adapter,
)
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import exact_format_correct

LAYER_INDEX = 15
LEARNED_PARAMETERS = 57_344
SEED = 14_199
CHECKPOINT_PATH = Path("phase8_artifacts/pilot/matched_adapter_seed_14199.pt")
RESULT_PATH = Path("phase8_results/matched_adapter_pilot_seed_14199.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    training, development = build_phase8_training_and_development()
    bundle = load_model_bundle(PHASE8_MODEL_ID, revision=PHASE8_MODEL_REVISION)
    adapter = install_matched_residual_adapter(
        bundle.model,
        layer_index=LAYER_INDEX,
        learned_parameter_count=LEARNED_PARAMETERS,
    )
    training_record = train_matched_adapter(
        bundle,
        adapter,
        training,
        config=MatchedAdapterTrainConfig(
            seed=SEED,
            steps=800,
            batch_size=4,
            learning_rate=0.002,
        ),
    )
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "phase8_development_matched_adapter_pilot",
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "seed": SEED,
            "layer_index": LAYER_INDEX,
            "rank": adapter.down.out_features,
            "learned_parameters": adapter.trainable_parameter_count,
            "down_weight": adapter.down.weight.detach().cpu(),
            "up_weight": adapter.up.weight.detach().cpu(),
        },
        CHECKPOINT_PATH,
    )
    rows: list[dict[str, object]] = []
    for example in development[:40] + development[240:280]:
        adapted = generate_matched_adapter(
            bundle,
            adapter,
            example.prompt,
            max_new_tokens=8,
            enabled=True,
        )
        base = generate_matched_adapter(
            bundle,
            adapter,
            example.prompt,
            max_new_tokens=8,
            enabled=False,
        )
        rows.append(
            {
                **example.to_dict(),
                "adapter_text": adapted["generated_text"],
                "base_text": base["generated_text"],
                "adapter_exact": (
                    exact_format_correct(
                        adapted["generated_text"],
                        example.answer or "",
                    )
                    if example.route_label
                    else False
                ),
                "token_preserved": (
                    adapted["generated_token_ids"]
                    == base["generated_token_ids"]
                ),
            }
        )
    positive = rows[:40]
    negative = rows[40:]
    summary = {
        "positive_examples": len(positive),
        "exact_additions": sum(row["adapter_exact"] for row in positive),
        "negative_examples": len(negative),
        "negative_token_preserved": sum(
            row["token_preserved"] for row in negative
        ),
    }
    payload = {
        "status": "development_matched_adapter_pilot_complete",
        "training": training_record,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "summary": summary,
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
