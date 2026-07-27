from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--fixed-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.layer >= 24:
        raise ValueError("Qwen2.5-0.5B layer must be in [0, 23]")
    default_artifact = (
        Path("phase7_artifacts/sequence_census_v1.pt")
        if args.layer == 23
        else Path(
            f"phase7_artifacts/sequence_census_layer_{args.layer:02d}_v1.pt"
        )
    )
    default_result = (
        Path("phase7_results/sequence_census_v1.json")
        if args.layer == 23
        else Path(f"phase7_results/sequence_census_layer_{args.layer:02d}_v1.json")
    )
    artifact_path = args.artifact or default_artifact
    result_path = args.result or default_result
    layout = SequenceImplantLayout(
        max_digits=4,
        learned_step=not args.fixed_step,
    )
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
        layer_index=args.layer,
        batch_size=4,
    )
    negative_census = collect_channel_census(
        bundle,
        negative,
        layer_index=args.layer,
        batch_size=4,
    )
    selected, score = select_low_impact_channels(
        [arithmetic_census, negative_census],
        width=layout.total_width,
    )
    ablation = evaluate_channel_ablation(
        bundle,
        arithmetic + negative,
        layer_index=args.layer,
        selected_indices=selected,
        batch_size=4,
    )
    artifact = {
        "arithmetic": arithmetic_census.state_dict(),
        "negative": negative_census.state_dict(),
        "selected_indices": selected,
        "conservative_score": score,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    payload = {
        "status": "development",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer_index": args.layer,
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
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["ablation"], indent=2), flush=True)
    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
