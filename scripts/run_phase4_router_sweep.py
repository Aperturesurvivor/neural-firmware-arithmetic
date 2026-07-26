from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neural_firmware.pretrained_data import chat_prompt_ids
from neural_firmware.pretrained_training import load_model_bundle
from neural_firmware.semantic_data import (
    HELDOUT_ADDITION_FAMILIES,
    HELDOUT_NEGATIVE_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
    WORD_PROBLEM_FAMILIES,
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEPTHS = (6, 12, 18, 24)
WIDTHS = (0, 16, 64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_examples() -> tuple[list[SemanticPromptExample], list[SemanticPromptExample]]:
    train = (
        make_semantic_addition_examples(
            count=600,
            min_digits=1,
            max_digits=4,
            seed=9001,
            split="router_train_positive",
            families=TRAIN_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=600,
            min_digits=1,
            max_digits=4,
            seed=9002,
            split="router_train_negative",
            families=TRAIN_NEGATIVE_FAMILIES,
        )
    )
    evaluation = (
        make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=8,
            seed=9003,
            split="router_eval_heldout_positive",
            families=HELDOUT_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=100,
            min_digits=1,
            max_digits=8,
            seed=9004,
            split="router_eval_word_positive",
            families=WORD_PROBLEM_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=200,
            min_digits=1,
            max_digits=8,
            seed=9005,
            split="router_eval_heldout_negative",
            families=HELDOUT_NEGATIVE_FAMILIES,
        )
    )
    return train, evaluation


@torch.inference_mode()
def collect_features(
    bundle: object,
    examples: list[SemanticPromptExample],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {
        f"{depth}_{pooling}": []
        for depth in DEPTHS
        for pooling in ("final", "mean_final")
    }
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        sequences = [chat_prompt_ids(bundle.tokenizer, row.prompt) for row in batch]
        maximum = max(map(len, sequences))
        input_ids = torch.full(
            (len(batch), maximum),
            bundle.tokenizer.pad_token_id,
            dtype=torch.long,
            device=bundle.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=bundle.device,
            )
            attention_mask[row, : len(sequence)] = 1
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        lengths = attention_mask.sum(dim=1)
        row_indices = torch.arange(len(batch), device=bundle.device)
        for depth in DEPTHS:
            hidden = outputs.hidden_states[depth].float()
            final = hidden[row_indices, lengths - 1]
            mask = attention_mask[:, :, None].to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / lengths[:, None]
            collected[f"{depth}_final"].append(final.cpu())
            collected[f"{depth}_mean_final"].append(
                torch.cat((mean, final), dim=-1).cpu()
            )
    return {key: torch.cat(rows) for key, rows in collected.items()}


class SweepRouter(nn.Module):
    def __init__(self, input_size: int, width: int) -> None:
        super().__init__()
        if width == 0:
            self.network = nn.Linear(input_size, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, width),
                nn.SiLU(),
                nn.Linear(width, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float()).squeeze(-1)


def train_and_score(
    train_features: torch.Tensor,
    evaluation_features: torch.Tensor,
    train_labels: torch.Tensor,
    evaluation_labels: torch.Tensor,
    *,
    width: int,
    device: torch.device,
    seed: int,
) -> dict[str, float | int]:
    set_seed(seed)
    router = SweepRouter(train_features.shape[1], width).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.003)
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(1200):
        indices = torch.randint(
            len(train_labels),
            (512,),
            generator=generator,
        )
        logits = router(train_features[indices].to(device))
        loss = loss_function(logits, train_labels[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probabilities = torch.sigmoid(router(evaluation_features.to(device))).cpu()
    predictions = probabilities >= 0.5
    labels = evaluation_labels.bool()
    true_positive = int((predictions & labels).sum())
    false_negative = int((~predictions & labels).sum())
    false_positive = int((predictions & ~labels).sum())
    true_negative = int((~predictions & ~labels).sum())
    return {
        "parameters": sum(parameter.numel() for parameter in router.parameters()),
        "accuracy": float((predictions == labels).float().mean()),
        "true_positive_rate": true_positive / (true_positive + false_negative),
        "false_positive_rate": false_positive / (false_positive + true_negative),
        "final_loss": float(loss),
    }


def main() -> None:
    started = time.perf_counter()
    train_examples, evaluation_examples = build_examples()
    bundle = load_model_bundle(MODEL_ID, revision=MODEL_REVISION)
    train_features = collect_features(bundle, train_examples, batch_size=8)
    evaluation_features = collect_features(bundle, evaluation_examples, batch_size=8)
    train_labels = torch.tensor(
        [example.route_label for example in train_examples],
        dtype=torch.float32,
    )
    evaluation_labels = torch.tensor(
        [example.route_label for example in evaluation_examples],
        dtype=torch.float32,
    )
    rows = []
    for key in train_features:
        depth_text, pooling = key.split("_", maxsplit=1)
        for width in WIDTHS:
            rows.append(
                {
                    "depth_after_blocks": int(depth_text),
                    "pooling": pooling,
                    "hidden_width": width,
                    **train_and_score(
                        train_features[key],
                        evaluation_features[key],
                        train_labels,
                        evaluation_labels,
                        width=width,
                        device=bundle.device,
                        seed=9200 + int(depth_text) + width,
                    ),
                }
            )
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "train_examples": len(train_examples),
        "evaluation_examples": len(evaluation_examples),
        "rows": rows,
        "wall_time_seconds": time.perf_counter() - started,
    }
    output = Path("phase4_results/router_architecture_sweep.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(sorted(rows, key=lambda row: row["accuracy"], reverse=True), indent=2))


if __name__ == "__main__":
    main()
