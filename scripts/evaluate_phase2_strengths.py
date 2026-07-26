from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_firmware.pretrained_data import AdditionExample
from neural_firmware.pretrained_evaluation import (
    evaluate_additions,
    evaluate_preservation,
)
from neural_firmware.pretrained_training import load_bridge, load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/pilot_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase2_results/pilot_v1/strength_sweep.json"),
    )
    parser.add_argument("--strengths", type=float, nargs="+", default=[32, 48, 64])
    args = parser.parse_args()

    dataset = json.loads(
        (args.artifact_directory / "logical_dataset.json").read_text()
    )
    eval_sets = {
        split: [AdditionExample(**row) for row in rows]
        for split, rows in dataset["evaluation"].items()
    }
    bundle = load_model_bundle()
    all_summaries: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    preservation_summaries: list[dict[str, object]] = []
    for strength in args.strengths:
        bridge = load_bridge(
            args.artifact_directory / "bridge.pt",
            hidden_size=bundle.model.config.hidden_size,
            strength=strength,
            device=bundle.device,
        )
        for split, examples in eval_sets.items():
            summary, results = evaluate_additions(
                bundle,
                examples,
                mode="latent",
                bridge=bridge,
            )
            summary["strength"] = strength
            all_summaries.append(summary)
            all_predictions.extend(
                [
                    {
                        "strength": strength,
                        "split": split,
                        **result.to_dict(),
                    }
                    for result in results
                ]
            )
        preservation, rows = evaluate_preservation(
            bundle,
            dataset["routing_eval"],
            bridge=bridge,
        )
        preservation["strength"] = strength
        preservation_summaries.append(preservation)
        all_predictions.extend(
            [{"strength": strength, "kind": "preservation", **row} for row in rows]
        )

    payload = {
        "pilot": True,
        "artifact_directory": str(args.artifact_directory),
        "summaries": all_summaries,
        "preservation_summaries": preservation_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    prediction_path = args.artifact_directory / "strength_sweep_predictions.json"
    prediction_path.write_text(json.dumps(all_predictions, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
