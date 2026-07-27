from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    SequenceNeuronImplantMLP,
)
from neural_firmware.phase7_sequence_training import (
    SequenceInterfaceTrainConfig,
    collect_sequence_features,
    evaluate_sequence_interface,
    train_sequence_interface,
)
from neural_firmware.phase7_training import (
    collect_channel_census,
    evaluate_channel_ablation,
    select_low_impact_channels,
)
from neural_firmware.phase8_data import (
    PHASE8_MODEL_ID,
    PHASE8_MODEL_REVISION,
    build_phase8_training_and_development,
)
from neural_firmware.pretrained_training import load_model_bundle

CANDIDATE_LAYERS = (12, 15, 18)
RESULT_PATH = Path("phase8_results/development_layer_probe.json")
ARTIFACT_PATH = Path("phase8_artifacts/development_selection.pt")
CACHE_DIRECTORY = Path("phase8_artifacts/cache")


def compact(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }


def main() -> None:
    started = time.perf_counter()
    training, development = build_phase8_training_and_development()
    probe_training = training[:300] + training[1_200:1_500]
    probe_development = development[:120] + development[240:360]
    layout = SequenceImplantLayout(max_digits=4, learned_step=False)
    bundle = load_model_bundle(
        PHASE8_MODEL_ID,
        revision=PHASE8_MODEL_REVISION,
    )
    layer_results: list[dict[str, object]] = []
    cached_features: dict[int, dict[str, object]] = {}
    for layer_index in CANDIDATE_LAYERS:
        train_features = collect_sequence_features(
            bundle,
            probe_training,
            layer_index=layer_index,
            layout=layout,
            batch_size=8,
            ordinary_tokens_per_example=6,
        )
        development_features = collect_sequence_features(
            bundle,
            probe_development,
            layer_index=layer_index,
            layout=layout,
            batch_size=8,
            ordinary_tokens_per_example=6,
        )
        base_mlp = bundle.model.model.layers[layer_index].mlp
        implant = SequenceNeuronImplantMLP(
            base_mlp,
            torch.arange(layout.total_width),
            layout=layout,
            digit_threshold=0.0,
        )
        reference = next(base_mlp.parameters())
        implant.to(device=reference.device, dtype=reference.dtype)
        train_record, _ = train_sequence_interface(
            implant,
            train_features,
            development_features,
            device=bundle.device,
            config=SequenceInterfaceTrainConfig(
                seed=14_100 + layer_index,
                steps=1_500,
                batch_size=256,
                learning_rate=0.001,
                step_loss_weight=0.0,
            ),
        )
        metrics = compact(
            evaluate_sequence_interface(
                implant,
                development_features,
                device=bundle.device,
            )
        )
        selection_score = min(
            float(metrics["route_true_positive_rate"]),
            1.0 - float(metrics["route_false_positive_rate"]),
            float(metrics["operand_role_accuracy"]),
            float(metrics["digit_accuracy_on_operands"]),
        )
        layer_results.append(
            {
                "layer_index": layer_index,
                "selection_score": selection_score,
                "training": train_record,
                "development": metrics,
            }
        )
        cached_features[layer_index] = {
            "train": train_features.state_dict(),
            "development": development_features.state_dict(),
        }
        print(
            f"layer={layer_index} score={selection_score:.6f} "
            f"metrics={json.dumps(metrics, sort_keys=True)}",
            flush=True,
        )
        del implant, train_features, development_features
        if bundle.device.type == "mps":
            torch.mps.empty_cache()

    selected = max(
        layer_results,
        key=lambda row: (row["selection_score"], row["layer_index"]),
    )
    selected_layer = int(selected["layer_index"])
    census_prompts = [
        example.prompt for example in probe_training[:48]
    ] + [
        example.prompt for example in probe_training[300:348]
    ]
    positive_census = collect_channel_census(
        bundle,
        census_prompts[:48],
        layer_index=selected_layer,
        batch_size=4,
    )
    negative_census = collect_channel_census(
        bundle,
        census_prompts[48:],
        layer_index=selected_layer,
        batch_size=4,
    )
    selected_indices, conservative_score = select_low_impact_channels(
        [positive_census, negative_census],
        width=layout.total_width,
    )
    ablation = evaluate_channel_ablation(
        bundle,
        census_prompts,
        layer_index=selected_layer,
        selected_indices=selected_indices,
        batch_size=4,
    )
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": PHASE8_MODEL_ID,
            "model_revision": PHASE8_MODEL_REVISION,
            "layout": asdict(layout),
            "candidate_layers": CANDIDATE_LAYERS,
            "selected_layer": selected_layer,
            "selected_indices": selected_indices,
            "conservative_score": conservative_score,
            "positive_census": positive_census.state_dict(),
            "negative_census": negative_census.state_dict(),
            "probe_features": cached_features,
        },
        ARTIFACT_PATH,
    )
    payload = {
        "status": "development_complete",
        "model_id": PHASE8_MODEL_ID,
        "model_revision": PHASE8_MODEL_REVISION,
        "model_architecture": type(bundle.model).__name__,
        "hidden_size": bundle.model.config.hidden_size,
        "intermediate_size": bundle.model.config.intermediate_size,
        "decoder_layers": len(bundle.model.model.layers),
        "layout": asdict(layout),
        "candidate_layers": list(CANDIDATE_LAYERS),
        "probe_examples": {
            "training": len(probe_training),
            "development": len(probe_development),
        },
        "layers": layer_results,
        "selected_layer": selected_layer,
        "selected_indices": selected_indices.tolist(),
        "channel_census": {
            "positive_selected_score_mean": float(
                positive_census.contribution_score[selected_indices].mean()
            ),
            "negative_selected_score_mean": float(
                negative_census.contribution_score[selected_indices].mean()
            ),
            "ablation": ablation,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(bundle.device),
            "mps_available": torch.backends.mps.is_available(),
        },
        "artifact": str(ARTIFACT_PATH),
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
