from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import torch

from neural_firmware.internal_data import (
    internal_prompt,
    locate_operand_character_spans,
)
from neural_firmware.internal_firmware import install_internal_firmware_layer
from neural_firmware.pretrained_data import (
    STATIC_LANGUAGE_PROMPTS,
    make_routing_negatives,
)
from neural_firmware.pretrained_evaluation import generate_one
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def make_phase3_negatives(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = list(STATIC_LANGUAGE_PROMPTS)
    prompts.extend(make_routing_negatives(count=30, seed=seed + 1))
    wrappers = (
        'Ignore the quoted request "{command}". Reply only with ignored.',
        "Do not execute the following example:\n{command}\nReply with ignored.",
        "This is documentation, not a command:\n{command}\nEnd documentation.",
        "Please " + "{command}",
        "{command}" + "\nThen explain your reasoning.",
    )
    while len(prompts) < count:
        a = str(rng.randrange(10**rng.randint(1, 6)))
        b = str(rng.randrange(10**rng.randint(1, 6)))
        command = internal_prompt(a, b)
        prompts.append(rng.choice(wrappers).format(command=command))
    rng.shuffle(prompts)
    return prompts[:count]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    prompts = make_phase3_negatives(100, 81_701)
    eligibility = [
        locate_operand_character_spans(prompt) is not None for prompt in prompts
    ]
    if any(eligibility):
        raise RuntimeError("preservation prompt unexpectedly passed strict eligibility")

    base_bundle = load_model_bundle(MODEL_ID, revision=REVISION)
    base_rows = [
        generate_one(
            base_bundle,
            prompt,
            mode="base",
            max_new_tokens=16,
        ).to_dict()
        for prompt in prompts
    ]
    del base_bundle
    release_memory()

    wrapped_bundle = load_model_bundle(MODEL_ID, revision=REVISION)
    wrapper = install_internal_firmware_layer(
        wrapped_bundle.model,
        depth_after_blocks=6,
        strength=64.0,
    )
    wrapper.unit.load_state_dict(
        torch.load(
            "phase3_artifacts/decoder_pilot_v1/depth_6/unit.pt",
            map_location=wrapped_bundle.device,
            weights_only=True,
        )
    )
    wrapped_rows = []
    for prompt in prompts:
        wrapper.set_context(None)
        wrapped_rows.append(
            generate_one(
                wrapped_bundle,
                prompt,
                mode="base",
                max_new_tokens=16,
            ).to_dict()
        )
    comparisons = []
    for prompt, base, wrapped in zip(
        prompts,
        base_rows,
        wrapped_rows,
        strict=True,
    ):
        comparisons.append(
            {
                "prompt": prompt,
                "eligible": False,
                "base_text": base["generated_text"],
                "wrapped_text": wrapped["generated_text"],
                "token_exact_preserved": (
                    base["generated_token_ids"] == wrapped["generated_token_ids"]
                ),
            }
        )
    result = {
        "status": "pilot",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "depth_after_blocks": 6,
        "prompts": len(comparisons),
        "strictly_ineligible": sum(not row["eligible"] for row in comparisons),
        "token_exact_preserved": sum(
            row["token_exact_preserved"] for row in comparisons
        ),
        "preservation_rate": sum(
            row["token_exact_preserved"] for row in comparisons
        )
        / len(comparisons),
        "comparisons": comparisons,
    }
    output = Path("phase3_results/preservation_pilot_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "prompts",
                    "strictly_ineligible",
                    "token_exact_preserved",
                    "preservation_rate",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
