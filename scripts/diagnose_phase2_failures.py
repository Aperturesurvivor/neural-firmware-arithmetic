from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_firmware.pretrained_evaluation import generate_one
from neural_firmware.pretrained_training import load_bridge, load_model_bundle


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1/study.json"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase2_results/confirmatory_v1/posthoc_route_diagnostic.json"),
    )
    args = parser.parse_args()
    study = read_json(args.study)
    bundle = load_model_bundle(
        study["model_id"],
        revision=study["model_revision"],
    )
    rows: list[dict[str, object]] = []
    for run in study["bridge_runs"]:
        seed = int(run["seed"])
        bridge = load_bridge(
            args.artifact_directory / f"bridge_seed_{seed}" / "bridge.pt",
            hidden_size=bundle.model.config.hidden_size,
            strength=float(study["config"]["bridge"]["strength"]),
            device=bundle.device,
        )
        preservation_rows = read_json(
            args.artifact_directory
            / f"bridge_seed_{seed}"
            / "preservation_predictions.json"
        )
        for original in preservation_rows:
            if original["token_exact_preserved"]:
                continue
            replay = generate_one(
                bundle,
                original["prompt"],
                mode="latent",
                bridge=bridge,
                max_new_tokens=16,
            )
            rows.append(
                {
                    "seed": seed,
                    "prompt": original["prompt"],
                    "base_text": original["base_text"],
                    "confirmatory_latent_text": original["latent_text"],
                    "diagnostic_replay": replay.to_dict(),
                    "first_threshold_crossing_step": next(
                        (
                            index
                            for index, probability in enumerate(
                                replay.route_probabilities
                            )
                            if probability >= 0.5
                        ),
                        None,
                    ),
                }
            )
    result = {
        "status": "posthoc_diagnostic_not_a_confirmatory_rerun",
        "model_id": study["model_id"],
        "model_revision": study["model_revision"],
        "source_confirmatory_commit": study["frozen_state"]["source_commit"],
        "rows": rows,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
