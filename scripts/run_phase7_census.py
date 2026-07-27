from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from neural_firmware.phase5_data import build_phase5_training_examples
from neural_firmware.phase7_implant import NeuronImplantLayout
from neural_firmware.phase7_training import (
    collect_channel_census,
    evaluate_channel_ablation,
    select_low_impact_channels,
)
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYERS = (8, 12, 16)
ARTIFACT_PATH = Path("phase7_artifacts/census_v2.pt")
RESULT_PATH = Path("phase7_results/census_v2.json")


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def compact_census(census: object, selected: torch.Tensor) -> dict[str, object]:
    score = census.contribution_score
    return {
        "layer_index": census.layer_index,
        "prompt_count": census.prompt_count,
        "token_count": census.token_count,
        "selected_score_mean": float(score[selected].mean()),
        "selected_score_max": float(score[selected].max()),
        "all_score_median": float(score.median()),
        "all_score_mean": float(score.mean()),
        "selected_mean_abs_mean": float(census.mean_abs[selected].mean()),
        "selected_active_fraction_mean": float(
            census.active_fraction[selected].mean()
        ),
    }


def main() -> None:
    layout = NeuronImplantLayout(max_digits=4)
    examples = build_phase5_training_examples(
        positive_count=48,
        negative_count=48,
    )
    positive_prompts = [
        example.prompt for example in examples if example.route_label
    ]
    negative_prompts = [
        example.prompt for example in examples if not example.route_label
    ]
    mixed_prompts = positive_prompts + negative_prompts
    bundle = load_model_bundle(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    started = time.perf_counter()
    artifacts: dict[str, object] = {}
    results: list[dict[str, object]] = []
    for layer_index in LAYERS:
        arithmetic = collect_channel_census(
            bundle,
            positive_prompts,
            layer_index=layer_index,
            batch_size=4,
        )
        negative = collect_channel_census(
            bundle,
            negative_prompts,
            layer_index=layer_index,
            batch_size=4,
        )
        selected, conservative_score = select_low_impact_channels(
            [arithmetic, negative],
            width=layout.total_width,
        )
        ablation = evaluate_channel_ablation(
            bundle,
            mixed_prompts,
            layer_index=layer_index,
            selected_indices=selected,
            batch_size=4,
        )
        artifacts[str(layer_index)] = {
            "arithmetic": arithmetic.state_dict(),
            "negative": negative.state_dict(),
            "selected_indices": selected,
            "conservative_score": conservative_score,
        }
        results.append(
            {
                "layer_index": layer_index,
                "arithmetic": compact_census(arithmetic, selected),
                "negative": compact_census(negative, selected),
                "ablation": ablation,
                "selected_indices": selected.tolist(),
            }
        )
        print(
            f"layer={layer_index} "
            f"agreement={ablation['top1_agreement']:.4f} "
            f"mean_kl={ablation['mean_kl_divergence']:.8f}",
            flush=True,
        )

    best = max(
        results,
        key=lambda row: (
            row["ablation"]["top1_agreement"],
            -row["ablation"]["mean_kl_divergence"],
        ),
    )
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, ARTIFACT_PATH)
    payload = {
        "status": "development",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layout_width": layout.total_width,
        "layers": results,
        "selected_layer": best["layer_index"],
        "wall_time_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(bundle.device),
            "mps_available": torch.backends.mps.is_available(),
            "git_commit_before_phase7_changes": git_commit(),
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"selected_layer={best['layer_index']}", flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
