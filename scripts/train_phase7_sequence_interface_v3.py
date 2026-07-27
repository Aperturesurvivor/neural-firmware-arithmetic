from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    evaluate_sequence_interface,
    train_sequence_interface,
)
from neural_firmware.pretrained_training import load_model_bundle

SOURCE_CHECKPOINT = Path(
    "phase7_artifacts/sequence_pilot_v2/neuron_implant_seed_12801.pt"
)
TRAIN_FEATURES = Path(
    "phase7_artifacts/cache/sequence_interface_train_v2_compact.pt"
)
DEVELOPMENT_FEATURES = Path(
    "phase7_artifacts/cache/sequence_interface_development_v2_compact.pt"
)
CHECKPOINT_PATH = Path(
    "phase7_artifacts/sequence_interface_v3/neuron_implant_seed_12821.pt"
)
RESULT_PATH = Path("phase7_results/sequence_interface_v3.json")
SEED = 12_821


def load_features(path: Path) -> SequenceFeatureSet:
    return SequenceFeatureSet.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def main() -> None:
    source = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    train = load_features(TRAIN_FEATURES)
    development = load_features(DEVELOPMENT_FEATURES)
    layout = SequenceImplantLayout(**source["layout"])
    bundle = load_model_bundle(
        source["model_id"],
        revision=source["model_revision"],
    )
    torch.manual_seed(SEED)
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=source["layer_index"],
        selected_indices=source["selected_indices"],
        layout=layout,
        output_strength=source["output_strength"],
        route_threshold=source["route_threshold"],
        use_swiglu_interface=True,
    )
    with torch.no_grad():
        implant.result_columns.copy_(source["result_columns"].to(bundle.device))
    started = time.perf_counter()
    training, metrics = train_sequence_interface(
        implant,
        train,
        development,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=SEED,
            steps=5_000,
            batch_size=256,
            learning_rate=0.002,
        ),
    )
    final = evaluate_sequence_interface(
        implant,
        development,
        device=bundle.device,
    )
    compact = {
        key: value
        for key, value in final.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **source,
            "stage": "output_with_interface_v3_swiglu",
            "interface_seed": SEED,
            "input_rows": implant.input_rows.detach().cpu(),
            "gate_rows": implant.gate_rows.detach().cpu(),
            "result_columns": implant.result_columns.detach().cpu(),
            "route_threshold": implant.route_threshold,
            "use_swiglu_interface": True,
        },
        CHECKPOINT_PATH,
    )
    payload = {
        "status": "development",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "checkpoint": str(CHECKPOINT_PATH),
        "layout": asdict(layout),
        "interface_type": "native_swiglu_replacement_rows",
        "train_rows": train.rows,
        "development_rows": development.rows,
        "training": training,
        "development": compact,
        "wall_time_seconds": time.perf_counter() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()

