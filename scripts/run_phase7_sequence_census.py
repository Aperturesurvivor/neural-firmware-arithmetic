from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from neural_firmware.phase5_data import build_phase5_training_examples
from neural_firmware.phase7_sequence_implant import SequenceImplantLayout
from neural_firmware.phase7_training import (
    collect_channel_census,
    evaluate_channel_ablation,
    select_low_impact_channels,
)
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYER_INDEX = 23
ARTIFACT_PATH = Path("phase7_artifacts/sequence_census_v1.pt")
RESULT_PATH = Path("phase7_results/sequence_census_v1.json")


def main() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    examples = build_phase5_training_examples(
        positive_count=48,
        negative_count=48,
    )
    arithmetic = [
        example.prompt for example in examples if example.route_label
    ]
    negative = [
        example.prompt for example in examples if not example.route_label
    ]
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    started = time.perf_counter()
    arithmetic_census = collect_channel_census(
        bundle,
        arithmetic,
        layer_index=LAYER_INDEX,
        batch_size=4,
    )
    negative_census = collect_channel_census(
        bundle,
        negative,
        layer_index=LAYER_INDEX,
        batch_size=4,
    )
    selected, score = select_low_impact_channels(
        [arithmetic_census, negative_census],
        width=layout.total_width,
    )
    ablation = evaluate_channel_ablation(
        bundle,
        arithmetic + negative,
        layer_index=LAYER_INDEX,
        selected_indices=selected,
        batch_size=4,
    )
    artifact = {
        "arithmetic": arithmetic_census.state_dict(),
        "negative": negative_census.state_dict(),
        "selected_indices": selected,
        "conservative_score": score,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, ARTIFACT_PATH)
    payload = {
        "status": "development",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": LAYER_INDEX,
        "layout_width": layout.total_width,
        "selected_indices": selected.tolist(),
        "selected_score_mean_arithmetic": float(
            arithmetic_census.contribution_score[selected].mean()
        ),
        "selected_score_mean_negative": float(
            negative_census.contribution_score[selected].mean()
        ),
        "all_score_median_arithmetic": float(
            arithmetic_census.contribution_score.median()
        ),
        "all_score_median_negative": float(
            negative_census.contribution_score.median()
        ),
        "ablation": ablation,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["ablation"], indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()

