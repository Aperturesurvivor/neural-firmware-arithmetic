from __future__ import annotations

import json
from pathlib import Path

import torch

from neural_firmware.internal_data import (
    encode_internal_prompt,
    make_internal_addition_examples,
)
from neural_firmware.internal_firmware import (
    InternalFirmwareContext,
    install_internal_firmware_layer,
)
from neural_firmware.internal_training import generate_internal
from neural_firmware.pretrained_training import load_model_bundle

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTH = 6


def symbol_tensor(
    answer: str,
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    values = [int(character) for character in answer] + [10]
    output = torch.full((1, width), -1, dtype=torch.long, device=device)
    output[0, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    return output


@torch.inference_mode()
def trace_first_symbol(
    bundle: object,
    wrapper: object,
    example: object,
) -> list[dict[str, float | int | str]]:
    encoded = encode_internal_prompt(bundle.tokenizer, example.prompt)
    context = InternalFirmwareContext(
        a_positions=torch.tensor(
            [encoded.a_token_positions],
            dtype=torch.long,
            device=bundle.device,
        ),
        a_lengths=torch.tensor(
            [len(encoded.a_token_positions)],
            dtype=torch.long,
            device=bundle.device,
        ),
        b_positions=torch.tensor(
            [encoded.b_token_positions],
            dtype=torch.long,
            device=bundle.device,
        ),
        b_lengths=torch.tensor(
            [len(encoded.b_token_positions)],
            dtype=torch.long,
            device=bundle.device,
        ),
        generation_index=0,
    )
    wrapper.set_context(context)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for depth in range(DEPTH - 2, 25):
        if depth == 24:
            continue
        layer = bundle.model.model.layers[depth]

        def capture(
            module: object,
            inputs: object,
            output: torch.Tensor,
            *,
            completed_depth: int = depth + 1,
        ) -> None:
            captured[completed_depth] = output[:, -1, :].detach()

        handles.append(layer.register_forward_hook(capture))
    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    bundle.model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    for handle in handles:
        handle.remove()
    wrapper.set_context(None)
    target_id = bundle.tokenizer.encode(
        example.answer[0],
        add_special_tokens=False,
    )[0]
    rows = []
    for depth in sorted(captured):
        normalized = bundle.model.model.norm(captured[depth])
        logits = bundle.model.lm_head(normalized)[0].float()
        target_logit = logits[target_id]
        competitors = logits.clone()
        competitors[target_id] = -torch.inf
        best_other_logit, best_other_id = competitors.max(dim=0)
        rank = int((logits > target_logit).sum().item()) + 1
        rows.append(
            {
                "depth_after_blocks": depth,
                "target_token_id": target_id,
                "target_logit": float(target_logit.item()),
                "target_rank": rank,
                "best_other_token_id": int(best_other_id.item()),
                "best_other_text": bundle.tokenizer.decode(
                    [int(best_other_id.item())]
                ),
                "target_margin": float((target_logit - best_other_logit).item()),
            }
        )
    return rows


def main() -> None:
    result_path = Path("phase3_results/causal_pilot_v1.json")
    bundle = load_model_bundle(MODEL_ID, revision=REVISION)
    wrapper = install_internal_firmware_layer(
        bundle.model,
        depth_after_blocks=DEPTH,
        strength=64.0,
    )
    wrapper.unit.load_state_dict(
        torch.load(
            "phase3_artifacts/decoder_pilot_v1/depth_6/unit.pt",
            map_location=bundle.device,
            weights_only=True,
        )
    )
    examples = make_internal_addition_examples(
        count=30,
        min_digits=9,
        max_digits=12,
        seed=71_901,
        split="causal_ood_9_12",
    )
    normal_rows = [
        generate_internal(bundle, wrapper, example, enabled=True).to_dict()
        for example in examples
    ]
    wrong_rows = []
    for example in examples[:20]:
        width = max(len(example.a), len(example.b)) + 2
        wrong_answer = str((int(example.answer[0]) + 1) % 10) + example.answer[1:]
        override = symbol_tensor(
            wrong_answer,
            width=width,
            device=bundle.device,
        )
        prediction = generate_internal(
            bundle,
            wrapper,
            example,
            enabled=True,
            symbol_override=override,
        )
        wrong_rows.append(
            {
                **prediction.to_dict(),
                "original_answer": example.answer,
                "intervened_answer": wrong_answer,
                "matches_intervened_state": prediction.generated_text == wrong_answer,
            }
        )
    substitution_rows = []
    substitution_examples = make_internal_addition_examples(
        count=40,
        min_digits=4,
        max_digits=4,
        seed=71_902,
        split="state_substitution",
    )
    usable = [
        example
        for example in substitution_examples
        if len(example.answer) == 5
    ]
    for recipient, donor in zip(usable[::2], usable[1::2], strict=False):
        width = max(len(recipient.a), len(recipient.b)) + 2
        override = symbol_tensor(
            donor.answer,
            width=width,
            device=bundle.device,
        )
        prediction = generate_internal(
            bundle,
            wrapper,
            recipient,
            enabled=True,
            symbol_override=override,
        )
        substitution_rows.append(
            {
                **prediction.to_dict(),
                "recipient_answer": recipient.answer,
                "donor_answer": donor.answer,
                "matches_donor_state": prediction.generated_text == donor.answer,
            }
        )
        if len(substitution_rows) == 10:
            break
    traces = [
        {
            "prompt": example.prompt,
            "answer": example.answer,
            "layers": trace_first_symbol(bundle, wrapper, example),
        }
        for example in examples[:10]
    ]
    result = {
        "status": "pilot",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "depth_after_blocks": DEPTH,
        "normal": {
            "examples": len(normal_rows),
            "correct": sum(row["exact"] for row in normal_rows),
            "predictions": normal_rows,
        },
        "wrong_symbol_intervention": {
            "examples": len(wrong_rows),
            "matches_intervened_state": sum(
                row["matches_intervened_state"] for row in wrong_rows
            ),
            "predictions": wrong_rows,
        },
        "state_substitution": {
            "examples": len(substitution_rows),
            "matches_donor_state": sum(
                row["matches_donor_state"] for row in substitution_rows
            ),
            "predictions": substitution_rows,
        },
        "downstream_logit_lens": traces,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "normal": {
            key: result["normal"][key] for key in ("examples", "correct")
        },
        "wrong_symbol_intervention": {
            key: result["wrong_symbol_intervention"][key]
            for key in ("examples", "matches_intervened_state")
        },
        "state_substitution": {
            key: result["state_substitution"][key]
            for key in ("examples", "matches_donor_state")
        },
        "mean_target_margin_by_depth": {
            str(depth): sum(
                trace["layers"][depth - (DEPTH - 1)]["target_margin"]
                for trace in traces
            )
            / len(traces)
            for depth in range(DEPTH - 1, 25)
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
